"""Progressive Agentic Training data path (paper Sec 5.1-5.2).

These tests cover everything that does not require ``torch``: the Stage-1 and
Stage-2 dataset construction, the loss/gradient mask, the input perturbation
``Phi``, and the GRPO advantage computation.  The trainers themselves are
exercised only for their pure logic, since running them needs a GPU.
"""

from __future__ import annotations

import pytest

from deepprep.agent import DeepPrepAgent
from deepprep.agent.llm import ScriptedClient
from deepprep.eval import table_match
from deepprep.operators import parse_pipeline
from deepprep.training import (
    DistillConfig,
    SFTExample,
    build_masked_sequence,
    build_op_syntax_dataset,
    build_op_syntax_examples,
    distill_trajectories,
    materialize_states,
    perturb_task,
    read_jsonl,
    refine_mask_to_action_tags,
    trajectory_to_sft_example,
    write_jsonl,
)


# --------------------------------------------------------------------------- #
# Stage 1: operator syntax learning (Eq. 4)
# --------------------------------------------------------------------------- #
def test_materialize_states_yields_T0_through_TN(demo_task):
    ops = parse_pipeline("\n".join(demo_task.gold_pipeline))
    states = materialize_states(demo_task.sources, ops)
    assert len(states) == len(ops) + 1
    # T_0 = S (Sec 2.1).
    assert states[0].names == demo_task.sources.names
    assert len(states[0]["movies"].df) == len(demo_task.sources["movies"].df)


def test_op_syntax_examples_condition_on_T_in_and_T_out(demo_task):
    examples = build_op_syntax_examples(demo_task, max_span=2, include_all_spans=True)
    assert examples
    ex = examples[0]
    user = ex.messages[1]["content"]
    assert "INPUT tables (T_in)" in user
    assert "OUTPUT tables (T_out)" in user
    # Stage 1 "isolat[es] operator execution logic from higher-level planning
    # decisions", so the target schema must NOT appear.
    assert "ratings by genre for each director" not in user


def test_op_syntax_target_is_exactly_the_operator_subsequence(demo_task):
    ops = parse_pipeline("\n".join(demo_task.gold_pipeline))
    for ex in build_op_syntax_examples(demo_task, max_span=3, include_all_spans=True):
        i, j = ex.meta["span"]
        expected = [op.name for op in ops[i : j + 1]]
        assert ex.meta["operators"] == expected
        target = ex.messages[-1]["content"]
        for name in expected:
            assert f"{name}(" in target


def test_op_syntax_targets_reparse_and_reproduce_T_out(demo_task):
    """The supervision is only valid if executing the target on T_in gives T_out."""
    ops = parse_pipeline("\n".join(demo_task.gold_pipeline))
    states = materialize_states(demo_task.sources, ops)
    for ex in build_op_syntax_examples(demo_task, max_span=3, include_all_spans=True):
        i, j = ex.meta["span"]
        state = states[i].copy()
        for op in parse_pipeline(ex.messages[-1]["content"]):
            state = op.execute(state)
        for t in states[j + 1]:
            assert table_match(state[t.name].df, t.df), f"span {i}-{j} on table {t.name}"


def test_op_syntax_span_length_is_bounded(demo_task):
    for ex in build_op_syntax_examples(demo_task, max_span=2, include_all_spans=True):
        i, j = ex.meta["span"]
        assert 1 <= j - i + 1 <= 2


def test_op_syntax_sampling_is_deterministic_for_a_seed(demo_task):
    a = build_op_syntax_dataset([demo_task], seed=7, verbose=False)
    b = build_op_syntax_dataset([demo_task], seed=7, verbose=False)
    assert [e.meta["span"] for e in a] == [e.meta["span"] for e in b]


def test_tasks_without_a_gold_pipeline_are_skipped(demo_task):
    demo_task.gold_pipeline = []
    assert build_op_syntax_dataset([demo_task], verbose=False) == []


def test_tasks_whose_gold_pipeline_does_not_execute_are_skipped(demo_task):
    demo_task.gold_pipeline = ["SelectColumn(movies, [does_not_exist])"]
    assert build_op_syntax_dataset([demo_task], verbose=False) == []


# --------------------------------------------------------------------------- #
# SFTExample invariants
# --------------------------------------------------------------------------- #
def test_example_requires_a_trainable_message():
    with pytest.raises(ValueError):
        SFTExample(messages=[{"role": "user", "content": "x"}], trainable=[False])


def test_example_rejects_a_mismatched_mask():
    with pytest.raises(ValueError):
        SFTExample(messages=[{"role": "user", "content": "x"}], trainable=[False, True])


def test_examples_round_trip_through_jsonl(demo_task, tmp_path):
    examples = build_op_syntax_dataset([demo_task], verbose=False)
    path = write_jsonl(examples, tmp_path / "d.jsonl")
    loaded = list(read_jsonl(path))
    assert len(loaded) == len(examples)
    assert loaded[0].messages == examples[0].messages
    assert loaded[0].trainable == examples[0].trainable


# --------------------------------------------------------------------------- #
# Stage 2: trajectory -> masked example (Eq. 5)
# --------------------------------------------------------------------------- #
def test_trajectory_example_trains_only_on_agent_turns(demo_task, figure4_trajectory):
    result = DeepPrepAgent(llm=ScriptedClient(figure4_trajectory), max_turns=5).solve(demo_task)
    ex = trajectory_to_sft_example(demo_task, result)

    assert len(ex.messages) == 1 + 2 * len(result.trajectory)
    for msg, train in zip(ex.messages, ex.trainable, strict=False):
        assert train == (msg["role"] == "assistant")

    # M_t must exclude every environment token -- Sec 5.2.
    for msg, train in zip(ex.messages, ex.trainable, strict=False):
        if train:
            assert "<execute>" not in msg["content"]
            assert "Current state:" not in msg["content"]


def test_trajectory_example_uses_the_students_prompt_not_the_teachers(demo_task, figure4_trajectory):
    """The teacher needs the in-context exemplar; training the student on a
    prompt it will never see at inference would be a train/test mismatch."""
    result = DeepPrepAgent(llm=ScriptedClient(figure4_trajectory), max_turns=5).solve(demo_task)
    ex = trajectory_to_sft_example(demo_task, result)
    assert "Example trajectory (abridged)" not in ex.messages[0]["content"]


def test_trajectory_example_conditions_on_phi_S_and_sigma_star(demo_task, figure4_trajectory):
    result = DeepPrepAgent(llm=ScriptedClient(figure4_trajectory), max_turns=5).solve(demo_task)
    ex = trajectory_to_sft_example(demo_task, result)
    first_user = ex.messages[1]["content"]
    assert "Source Tables" in first_user
    assert "Target Schema" in first_user
    assert demo_task.target_schema.columns[0].name in first_user


# --------------------------------------------------------------------------- #
# Input perturbation Phi
# --------------------------------------------------------------------------- #
def test_perturbation_shuffles_rows_and_columns(demo_task):
    import random

    perturbed = perturb_task(demo_task, random.Random(1))
    orig, new = demo_task.sources["ratings"], perturbed.sources["ratings"]
    assert set(orig.columns) == set(new.columns)
    assert (orig.columns != new.columns) or (
        orig.df["values"].tolist() != new.df["values"].tolist()
    )


def test_perturbation_preserves_the_task_semantics(demo_task):
    """Phi is a robustness perturbation, not a corruption: the gold pipeline must
    still produce T*."""
    import random

    for seed in range(5):
        perturbed = perturb_task(demo_task, random.Random(seed))
        state = perturbed.sources.copy()
        for op in parse_pipeline("\n".join(perturbed.gold_pipeline)):
            state = op.execute(state)
        assert table_match(state["joined"].df, demo_task.target_table), f"seed {seed}"


def test_perturbation_keeps_column_descriptions_with_their_columns(demo_task):
    import random

    perturbed = perturb_task(demo_task, random.Random(3))
    for t in perturbed.sources:
        original = demo_task.sources[t.name]
        for c in t.schema.columns:
            want = original.schema.get(c.name)
            if want is not None and want.description:
                assert c.description == want.description


# --------------------------------------------------------------------------- #
# Distillation filtering
# --------------------------------------------------------------------------- #
def test_distillation_keeps_correct_trajectories(demo_task, figure4_trajectory):
    class Teacher:
        model = "teacher"

        def __init__(self):
            self.inner = None
            self.n = 0

        def generate(self, messages, **kw):
            if self.n % len(figure4_trajectory) == 0:
                self.inner = ScriptedClient(figure4_trajectory)
            self.n += 1
            return self.inner.generate(messages, **kw)

    examples, stats = distill_trajectories(
        [demo_task],
        Teacher(),
        DistillConfig(n_candidates=1, top_k=1, perturb=False),
        verbose=False,
    )
    assert stats.n_correct == 1
    assert stats.n_kept == 1
    assert stats.n_with_backtrack == 1
    assert examples[0].meta["stage"] == "reasoning"


def test_distillation_discards_incorrect_trajectories(demo_task):
    bad = ["<plan>p</plan><expand><from></from><ops>Sort(movies, [id])</ops></expand>"] * 5
    examples, stats = distill_trajectories(
        [demo_task],
        ScriptedClient(bad * 4),
        DistillConfig(n_candidates=2, top_k=1, perturb=False),
        verbose=False,
    )
    assert stats.n_correct == 0
    assert stats.n_kept == 0
    assert examples == []


# --------------------------------------------------------------------------- #
# Token masking (Eq. 5 / Sec 5.2)
# --------------------------------------------------------------------------- #
class FakeTokenizer:
    """A whitespace tokenizer with a prefix-consistent chat template.

    Exercises the masking logic without downloading a real tokenizer.
    """

    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab))

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [self._id(t) for t in text.split()]}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kw
    ) -> str:
        parts = [f"<|{m['role']}|> {m['content']} <|end|>" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def decode(self, ids, skip_special_tokens=False) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "") for i in ids)


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


def test_mask_covers_assistant_tokens_only(tok):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER ONE"},
        {"role": "assistant", "content": "AGENT ONE"},
        {"role": "user", "content": "ENV FEEDBACK"},
        {"role": "assistant", "content": "AGENT TWO"},
    ]
    seq = build_masked_sequence(tok, messages)
    trained = tok.decode([t for t, m in zip(seq.input_ids, seq.action_mask, strict=False) if m]).split()

    assert "AGENT" in trained and "ONE" in trained and "TWO" in trained
    for env_token in ("SYS", "USER", "ENV", "FEEDBACK"):
        assert env_token not in trained


def test_mask_excludes_the_generation_prompt(tok):
    """The `<|assistant|>` header is emitted by the template, not sampled;
    training on it teaches boilerplate rather than decisions."""
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    seq = build_masked_sequence(tok, messages)
    trained = tok.decode([t for t, m in zip(seq.input_ids, seq.action_mask, strict=False) if m]).split()
    assert "<|assistant|>" not in trained
    assert trained[0] == "A"


def test_mask_honours_an_explicit_trainable_list(tok):
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "SECOND"},
    ]
    seq = build_masked_sequence(tok, messages, trainable=[False, False, False, True])
    trained = tok.decode([t for t, m in zip(seq.input_ids, seq.action_mask, strict=False) if m]).split()
    assert "SECOND" in trained and "FIRST" not in trained


def test_labels_ignore_masked_positions(tok):
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    seq = build_masked_sequence(tok, messages)
    labels = seq.labels()
    assert len(labels) == len(seq.input_ids)
    for tokid, m, lab in zip(seq.input_ids, seq.action_mask, labels, strict=False):
        assert lab == (tokid if m else -100)


def test_left_truncation_keeps_the_most_recent_turns(tok):
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A B C"}]
    full = build_masked_sequence(tok, messages)
    cut = build_masked_sequence(tok, messages, max_length=len(full) - 2)
    assert len(cut) == len(full) - 2
    assert cut.input_ids == full.input_ids[2:]
    assert cut.n_action_tokens > 0


def test_refine_mask_drops_inlined_execute_blocks(tok):
    """Only needed when a caller inlines environment output into an assistant
    turn; the default transport already excludes it."""
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "<plan> P </plan> <execute> ENVDATA </execute>"},
    ]
    seq = build_masked_sequence(tok, messages)
    refined = refine_mask_to_action_tags(tok, seq)
    trained = tok.decode(
        [t for t, m in zip(refined.input_ids, refined.action_mask, strict=False) if m]
    ).split()
    assert "P" in trained
    assert "ENVDATA" not in trained


def test_full_trajectory_mask_excludes_all_environment_tokens(
    tok, demo_task, figure4_trajectory
):
    """End to end: the tokens the policy is optimized over never include a
    serialized intermediate table."""
    result = DeepPrepAgent(llm=ScriptedClient(figure4_trajectory), max_turns=5).solve(demo_task)
    ex = trajectory_to_sft_example(demo_task, result)
    seq = build_masked_sequence(tok, ex.messages, ex.trainable)

    trained = tok.decode([t for t, m in zip(seq.input_ids, seq.action_mask, strict=False) if m])
    assert "<execute>" not in trained
    assert "Current" not in trained  # from "Current state:" in the feedback
    assert "<plan>" in trained and "<expand>" in trained and "<answer>" in trained
    assert 0 < seq.n_action_tokens < len(seq)


# --------------------------------------------------------------------------- #
# GRPO advantage (Sec 5.2)
# --------------------------------------------------------------------------- #
def _rollouts(rewards):
    from deepprep.training.grpo import Rollout
    from deepprep.training.rewards import RewardBreakdown

    return [
        Rollout(task_id="t", result=None, reward=RewardBreakdown(total=r)) for r in rewards
    ]


def test_advantage_is_the_group_normalized_reward():
    from deepprep.training.grpo import GRPOTrainer

    group = _rollouts([0.0, 1.0])
    assert GRPOTrainer.assign_advantages(group, skip_degenerate=True)
    assert group[0].advantage == pytest.approx(-1.0)
    assert group[1].advantage == pytest.approx(1.0)
    assert sum(r.advantage for r in group) == pytest.approx(0.0, abs=1e-9)


def test_degenerate_group_is_skipped():
    """Equal rewards give zero advantage everywhere: no signal, but still a full
    backward pass if not skipped."""
    from deepprep.training.grpo import GRPOTrainer

    assert not GRPOTrainer.assign_advantages(_rollouts([0.5, 0.5, 0.5]), skip_degenerate=True)
    assert GRPOTrainer.assign_advantages(_rollouts([0.5, 0.5]), skip_degenerate=False)


def test_a_group_of_one_has_no_baseline():
    from deepprep.training.grpo import GRPOTrainer

    assert not GRPOTrainer.assign_advantages(_rollouts([1.0]), skip_degenerate=True)


def test_training_package_imports_without_torch():
    """Inference and evaluation must stay installable without the training stack."""
    import importlib

    mod = importlib.import_module("deepprep.training")
    assert hasattr(mod, "build_op_syntax_dataset")
    assert "SFTTrainer" not in mod.__dict__  # exposed lazily via __getattr__
