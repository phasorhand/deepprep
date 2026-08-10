"""The cold-start SFT loop actually updates the policy (Eq. 4 / Eq. 5).

Everything else in this suite runs without torch, so `training/sft.py` had zero
test coverage: nothing verified that the masked labels reach the model, that the
backward pass touches the trainable parameters, or that a checkpoint is written.
The first real run of `deepprep train-sft` duly died on its first example.

These tests build a randomly initialized model in process -- no download, no
network -- and are skipped when the optional training extra is not installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from deepprep.training.sft import SFTConfig, SFTTrainer  # noqa: E402
from deepprep.training.sft_data import SFTExample  # noqa: E402

VOCAB = 512


class TinyTokenizer:
    """Bounded-vocabulary whitespace tokenizer with a chat template."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0}

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = 1 + (len(self.vocab) % (VOCAB - 1))
        return self.vocab[tok]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [self._id(t) for t in text.split()]}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kw
    ) -> str:
        if not messages:
            raise ValueError("Cannot apply chat template to an empty conversation.")
        parts = [f"<|{m['role']}|> {m['content']} <|end|>" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def save_pretrained(self, path) -> None:
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_stub.json").write_text("{}")


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def _examples() -> list[SFTExample]:
    return [
        SFTExample(
            messages=[
                {"role": "system", "content": "SYS PROMPT"},
                {"role": "user", "content": f"INPUT TABLE {i}"},
                {"role": "assistant", "content": f"Deduplicate movies id first {i}"},
            ],
            trainable=[False, False, True],
        )
        for i in range(4)
    ]


@pytest.fixture
def trainer(tmp_path):
    cfg = SFTConfig(
        model_name_or_path="<injected>",
        output_dir=str(tmp_path / "ckpt"),
        n_epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
        max_length=128,
        learning_rate=1e-2,  # large enough that one step visibly moves the weights
        use_lora=False,
        gradient_checkpointing=False,
        log_every=1,
    )
    t = SFTTrainer(cfg)
    t._torch = torch
    t.device = "cpu"
    t.tokenizer = TinyTokenizer()
    t.model = _tiny_model()
    return t


def test_a_training_step_changes_the_weights(trainer):
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    stats = trainer.train(_examples())

    assert stats["steps"] > 0
    assert stats["n_examples"] == 4
    moved = [n for n, p in trainer.model.named_parameters() if not torch.equal(p, before[n])]
    assert moved, "no parameter changed: the backward pass never reached the policy"


def test_the_reported_loss_is_finite(trainer):
    stats = trainer.train(_examples())
    losses = [rec["loss"] for rec in stats["history"]]
    assert losses
    for loss in losses:
        assert loss == loss and abs(loss) != float("inf"), f"degenerate loss {loss}"


def test_a_checkpoint_is_written(trainer, tmp_path):
    trainer.train(_examples())
    ckpt = tmp_path / "ckpt"
    assert (ckpt / "deepprep_train_config.json").exists()


def test_only_masked_tokens_carry_a_label(trainer):
    """Eq. 5's mask must reach the labels tensor, not just the MaskedSequence."""
    seqs = trainer.encode(_examples())
    batch = trainer._collate(seqs[:2])

    supervised = (batch.labels != -100).sum().item()
    assert supervised == batch.n_action_tokens
    assert 0 < supervised < batch.labels.numel(), "the mask supervised everything or nothing"


def test_truncation_keeps_the_trailing_action_tokens(trainer):
    """Truncation drops the oldest context, so the tokens being learned survive.

    This is what makes the long-trajectory case trainable at all: a Stage-2
    trajectory is mostly environment feedback, and the answer is at the end.
    """
    trainer.cfg.max_length = 8
    long_example = SFTExample(
        messages=[
            {"role": "system", "content": " ".join(["FILLER"] * 200)},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "ANSWER HERE"},
        ],
        trainable=[False, False, True],
    )
    seqs = trainer.encode([long_example])

    assert len(seqs) == 1
    seq = seqs[0]
    assert len(seq.input_ids) <= 8
    assert seq.n_action_tokens > 0, "truncation evicted the tokens we train on"
    # The supervised tokens sit at the tail, not scattered through the context.
    assert seq.action_mask[-1] == 1
