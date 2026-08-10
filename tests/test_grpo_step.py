"""A full GRPO step with a non-degenerate group (Sec 5.2).

`test_grpo_loss.py` covers the objective; this covers the loop around it --
advantage assignment, the frozen behaviour-policy log-probs, backward, clipping,
`optimizer.step()`, and the checkpoint.

Real CPU runs cannot reach this path. An untrained 0.5B policy fails every
rollout the same way, so the group's rewards are identical, the advantage is 0
and the update is a mathematically correct no-op -- verified against two task
sets and up to 640 new tokens per turn. Sampling itself is fine (three HFClient
draws at temperature 1.0 differ); the policy is simply uniformly wrong. So the
group is stubbed here, which is the only way to see a real gradient without a
GPU-scale cold-start first.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deepprep.agent.agent import SolveResult  # noqa: E402
from deepprep.training.grpo import GRPOConfig, GRPOTrainer, Rollout  # noqa: E402
from deepprep.training.masking import MaskedSequence  # noqa: E402
from deepprep.training.rewards import RewardBreakdown  # noqa: E402
from deepprep.types import ADPTask, TableSchema  # noqa: E402

VOCAB = 128


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )


class _StubTokenizer:
    pad_token_id = 0

    def save_pretrained(self, path) -> None:
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)


def _task() -> ADPTask:
    import pandas as pd

    from deepprep.types import Table, TableSet

    return ADPTask(
        task_id="stub",
        sources=TableSet([Table(name="t", df=pd.DataFrame({"a": [1]}))]),
        target_schema=TableSchema(columns=[], description="stub"),
    )


def _group(rewards: list[float]) -> list[Rollout]:
    out = []
    for i, r in enumerate(rewards):
        seq = MaskedSequence(
            input_ids=[3, 9, 14, 5, 27, 8, 11, 2 + i],
            action_mask=[0, 0, 1, 1, 1, 1, 1, 1],
            spans=[(2, 8)],
        )
        out.append(
            Rollout(
                task_id="stub",
                result=SolveResult(task_id="stub"),
                reward=RewardBreakdown(similarity=None, total=r),
                sequence=seq,
            )
        )
    return out


@pytest.fixture
def trainer(tmp_path):
    cfg = GRPOConfig(
        model_name_or_path="<injected>",
        output_dir=str(tmp_path / "grpo"),
        group_size=4,
        tasks_per_batch=1,
        n_epochs=1,
        kl_coef=0.0,
        learning_rate=1e-2,  # large enough that one step visibly moves the weights
        use_lora=False,
    )
    t = GRPOTrainer(cfg)
    t._torch = torch
    t.device = "cpu"
    t.tokenizer = _StubTokenizer()
    t.model = _tiny_model()
    return t


def test_a_non_degenerate_group_actually_updates_the_policy(trainer):
    """The path every real CPU run misses: nonzero advantage reaching the weights."""
    trainer.collect_group = lambda task: _group([0.0, 0.3, 0.7, 1.0])
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

    history = trainer.train([_task()])

    assert len(history) == 1
    assert history[0]["n_groups_used"] == 1
    assert history[0]["n_groups_skipped"] == 0
    moved = [n for n, p in trainer.model.named_parameters() if not torch.equal(p, before[n])]
    assert moved, "nonzero advantage did not reach the policy weights"


def test_a_degenerate_group_leaves_the_policy_untouched(trainer):
    """Equal rewards carry no signal; the step must be skipped, not applied."""
    trainer.collect_group = lambda task: _group([0.5, 0.5, 0.5, 0.5])
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

    history = trainer.train([_task()])

    assert history[0]["n_groups_used"] == 0
    assert history[0]["n_groups_skipped"] == 1
    unchanged = all(
        torch.equal(p, before[n]) for n, p in trainer.model.named_parameters()
    )
    assert unchanged, "a zero-signal group moved the weights"


def test_the_reported_reward_is_the_group_mean(trainer):
    trainer.collect_group = lambda task: _group([0.0, 0.3, 0.7, 1.0])
    history = trainer.train([_task()])
    assert history[0]["mean_reward"] == pytest.approx(0.5, abs=1e-6)


def test_a_checkpoint_and_history_are_written(trainer, tmp_path):
    trainer.collect_group = lambda task: _group([0.0, 1.0])
    trainer.train([_task()])
    out = tmp_path / "grpo"
    assert (out / "deepprep_grpo_config.json").exists()
    assert (out / "deepprep_grpo_history.json").exists()
