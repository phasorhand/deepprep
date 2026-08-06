"""Cold-start curriculum datasets (paper Sec 5.1).

    "To mitigate the 'cold start' issue where an untrained LLM fails to produce
     executable pipelines, we design a Cold-Start Curriculum module ... pre-trained
     LLMs generally lack an explicit understanding of data preparation operator
     syntax and the tree-based agentic reasoning procedure ... To address this, we
     employ a two-stage supervised fine-tuning (SFT) curriculum."

**Stage 1 — Operator Syntax Learning.**

    "Given source tables and a ground-truth operator pipeline P_total ...
     consisting of a sequence of operators (o^(1), ..., o^(N)). We first execute
     these operators sequentially to collect a sequence of generated intermediate
     table sets {T_0, ..., T_N}. Then, we sample a contiguous sub-sequence of
     operators P_sub = (o^(i), ..., o^(j)) with the corresponding input
     intermediate table set T_in = T_{i-1} and output intermediate table set
     T_out = T_j. In this way, we construct the operator syntax learning dataset
     D_op consisting of training triples (P_sub, T_in, T_out)."

        L_op = - sum_{(P_sub,T_in,T_out) in D_op} log Pr_LLM( phi(P_sub) | phi(T_in), phi(T_out) )   (4)

    "Through Eq. (4), the agent learns to recognize operator function signatures
     and argument formats, while *isolating operator execution logic from
     higher-level planning decisions*."

Note the conditioning: the model is shown the input state and the *desired output
state* and must recover the operators between them.  This is deliberately not the
agentic task — no target schema, no tree, no planning.

**Stage 2 — Reasoning Procedure Learning.**

        L_reason = - sum_{tau in U*} sum_t M_t * log Pr_LLM( phi(r_t) | phi(r_<t), phi(S), Sigma* )   (5)

    "we introduce a loss mask M_t to ensure that gradients are only calculated for
     tokens corresponding to the agent's outputs."

Stage 2 examples are built from trajectories collected by :mod:`deepprep.training.distill`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..operators import OperatorCall, OperatorError, parse_pipeline
from ..serialize import serialize_pipeline, serialize_table_set
from ..types import ADPTask, TableSet

__all__ = [
    "OP_SYNTAX_SYSTEM_PROMPT",
    "SFTExample",
    "build_op_syntax_dataset",
    "build_op_syntax_examples",
    "materialize_states",
    "write_jsonl",
]


OP_SYNTAX_SYSTEM_PROMPT = """
You are a data preparation operator synthesizer.

You are given an INPUT table set and the OUTPUT table set that results from
applying a short sequence of data preparation operators to it. Recover that
operator sequence.

Reply with the operator calls only, one per line, and nothing else.
""".strip()


@dataclass
class SFTExample:
    """One supervised example with an explicit loss mask.

    ``messages`` is a chat conversation.  ``trainable`` marks, per message,
    whether its tokens contribute to the loss — this is ``M_t`` of Eq. (5).
    Stage 1 has exactly one trainable message (the operator sequence); Stage 2
    has one per agent turn.
    """

    messages: list[dict[str, str]]
    trainable: list[bool]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.messages) != len(self.trainable):
            raise ValueError(
                f"messages ({len(self.messages)}) and trainable ({len(self.trainable)}) "
                f"must have the same length"
            )
        if not any(self.trainable):
            raise ValueError("an SFT example must have at least one trainable message")

    def to_dict(self) -> dict[str, Any]:
        return {"messages": self.messages, "trainable": self.trainable, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SFTExample:
        return cls(messages=d["messages"], trainable=d["trainable"], meta=d.get("meta", {}))


# --------------------------------------------------------------------------- #
# Stage 1: operator syntax learning (Eq. 4)
# --------------------------------------------------------------------------- #
def materialize_states(
    sources: TableSet, pipeline: Sequence[OperatorCall]
) -> list[TableSet]:
    """Execute ``P_total`` sequentially, returning ``{T_0, ..., T_N}``.

    ``T_0 = S``, so the returned list has ``len(pipeline) + 1`` entries.  Raises
    :class:`OperatorError` if the gold pipeline does not execute — a task whose
    supervision does not run is not usable training data.
    """
    states = [sources.copy()]
    current = states[0]
    for op in pipeline:
        current = op.execute(current.copy())
        states.append(current)
    return states


def build_op_syntax_examples(
    task: ADPTask,
    max_span: int = 3,
    max_per_task: int = 6,
    max_rows: int = 5,
    rng: random.Random | None = None,
    include_all_spans: bool = False,
) -> list[SFTExample]:
    """Sample ``(P_sub, T_in, T_out)`` triples from one task's gold pipeline.

    ``max_span`` bounds ``j - i + 1``.  Short spans are what teach *syntax*: a
    long span would reintroduce the planning problem that Stage 1 exists to
    isolate from operator usage.
    """
    rng = rng or random.Random(0)
    if not task.gold_pipeline:
        return []
    try:
        ops = parse_pipeline("\n".join(task.gold_pipeline))
        states = materialize_states(task.sources, ops)
    except OperatorError:
        return []
    if not ops:
        return []

    n = len(ops)
    spans = [
        (i, j)
        for i in range(n)
        for j in range(i, min(i + max_span, n))
    ]
    if not include_all_spans and len(spans) > max_per_task:
        spans = rng.sample(spans, max_per_task)

    examples: list[SFTExample] = []
    for i, j in sorted(spans):
        t_in, t_out = states[i], states[j + 1]
        sub = ops[i : j + 1]
        user = (
            "## INPUT tables (T_in)\n"
            f"{serialize_table_set(t_in, max_rows=max_rows, with_description=False)}\n\n"
            "## OUTPUT tables (T_out)\n"
            f"{serialize_table_set(t_out, max_rows=max_rows, with_description=False)}\n\n"
            f"Recover the {len(sub)} operator(s) that transform T_in into T_out."
        )
        examples.append(
            SFTExample(
                messages=[
                    {"role": "system", "content": OP_SYNTAX_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": serialize_pipeline(sub)},
                ],
                trainable=[False, False, True],
                meta={
                    "stage": "op_syntax",
                    "task_id": task.task_id,
                    "span": [i, j],
                    "operators": [op.name for op in sub],
                },
            )
        )
    return examples


def build_op_syntax_dataset(
    tasks: Sequence[ADPTask],
    max_span: int = 3,
    max_per_task: int = 6,
    max_rows: int = 5,
    seed: int = 0,
    verbose: bool = True,
) -> list[SFTExample]:
    """Build ``D_op`` over a task set."""
    rng = random.Random(seed)
    out: list[SFTExample] = []
    n_skipped = 0
    for task in tasks:
        ex = build_op_syntax_examples(
            task, max_span=max_span, max_per_task=max_per_task, max_rows=max_rows, rng=rng
        )
        if not ex:
            n_skipped += 1
        out.extend(ex)
    if verbose:
        print(
            f"D_op: {len(out)} examples from {len(tasks) - n_skipped}/{len(tasks)} tasks "
            f"({n_skipped} skipped: no gold pipeline, or it failed to execute)"
        )
    return out


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def write_jsonl(examples: Sequence[SFTExample], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: str | Path) -> Iterator[SFTExample]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield SFTExample.from_dict(json.loads(line))
