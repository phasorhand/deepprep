"""Shared scaffolding for the prompting baselines of paper Sec 6.1.

Every baseline exposes the same surface as
:class:`~deepprep.agent.agent.DeepPrepAgent` — an ``llm`` attribute and a
``solve(task) -> SolveResult`` method — so :func:`deepprep.eval.evaluate` scores
all of them through one code path and Table 2 stays comparable.

What is deliberately *not* shared is the reasoning structure.  Each baseline
removes one ingredient of DeepPrep's tree-based inference (Sec 4.2), and the
whole point of Sec 6.1 is that those removals are the source of the gap:

* CodeGen         — no interaction with the environment at all.
* Plan-and-Solve  — a plan, then one shot at the whole pipeline; no feedback.
* ReAct           — feedback, but a single linear trajectory with no backtracking.
* MCTS-OP         — search, but guided by *scalar* rollout statistics rather than
  the structured execution feedback DeepPrep reasons over.

The helpers here only cover the mechanics those four have in common: token
accounting, answer-table acceptance, and the best-effort fallback table.  None of
them ever touches ``task.target_table``: the ground-truth ``T*`` "is used only
for evaluation" (Sec 2.1), so a baseline that peeked at it would not be a
baseline.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

from ..agent.agent import SolveResult, select_answer_table
from ..agent.llm import LLMClient, LLMResponse
from ..env import EnvironmentLimits
from ..operators import OperatorCall, operator_manual
from ..types import ADPTask, TableSchema, TableSet

__all__ = [
    "BaselineAgent",
    "LLMCallError",
    "Message",
    "operator_space_section",
    "schema_alignment",
]

Message = dict[str, str]


class LLMCallError(RuntimeError):
    """The backbone could not be reached.

    Raised instead of returning a sentinel so a dead endpoint unwinds the whole
    baseline in one place rather than being silently scored as a wrong answer.
    """


# --------------------------------------------------------------------------- #
# Prompt fragments
# --------------------------------------------------------------------------- #
def operator_space_section() -> str:
    """The operator space ``O`` (Sec 2.2), rendered for a baseline prompt.

    Identical to the manual the DeepPrep agent receives, so the comparison in
    Table 2 isolates the *reasoning structure* rather than a difference in how
    much the model knows about the operators.
    """
    return (
        "# Operator space\n"
        "Every operator maps a set of tables to a set of tables. Tables not mentioned by "
        "an operator are carried through unchanged. Reference tables and columns by bare "
        "name (e.g. `Deduplicate(movies, [id], first)`).\n"
        f"{operator_manual()}"
    )


# --------------------------------------------------------------------------- #
# Schema-only scoring
# --------------------------------------------------------------------------- #
def schema_alignment(columns: Iterable[Any], target: TableSchema | None) -> float:
    """Jaccard overlap between produced column names and ``Sigma*``.

    This is the ``S_sch`` term of Eq. (8) evaluated against the *target schema*
    instead of the gold table, which is the strongest signal a baseline may
    legitimately compute: the schema is part of the task input, the gold tuples
    are not.  MCTS-OP uses it as half of its scalar reward.
    """
    have = {str(c).strip().lower() for c in columns}
    want = {c.name.strip().lower() for c in (target.columns if target else [])}
    if not want:
        return 0.0
    union = have | want
    return len(have & want) / len(union) if union else 0.0


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaselineAgent:
    """Common plumbing for the Sec 6.1 prompting baselines.

    Subclasses implement :meth:`solve`; everything here is bookkeeping that must
    stay identical across methods for the Cost and Completion Rate metrics to be
    comparable.
    """

    #: Name used in :class:`~deepprep.eval.evaluate.EvalReport`.
    METHOD: ClassVar[str] = "Baseline"

    def __init__(
        self,
        llm: LLMClient,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        max_rows_in_prompt: int = 5,
        limits: EnvironmentLimits | None = None,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        # "For all prompting methods, the temperature is set to 0.01." (Sec 6.1)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_rows_in_prompt = max_rows_in_prompt
        self.limits = limits or EnvironmentLimits()
        self.verbose = verbose

    # -- to be implemented -------------------------------------------------- #
    def solve(self, task: ADPTask) -> SolveResult:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- LLM ---------------------------------------------------------------- #
    def _generate(
        self,
        messages: Sequence[Message],
        result: SolveResult,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One backbone call, with the usage folded into the Cost metric."""
        try:
            resp = self.llm.generate(
                messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
        except Exception as e:  # noqa: BLE001 - a dead endpoint must not lose the run
            raise LLMCallError(f"LLM call failed: {type(e).__name__}: {e}") from e
        result.usage.add(resp.usage)
        return resp

    @staticmethod
    def _abort_llm_error(
        result: SolveResult, exc: LLMCallError, t_start: float, n_turns: int
    ) -> SolveResult:
        result.error = str(exc)
        result.stop_reason = "llm_error"
        result.n_turns = n_turns
        result.elapsed_s = time.perf_counter() - t_start
        return result

    # -- answering ---------------------------------------------------------- #
    def _accept_answer(
        self,
        task: ADPTask,
        state: TableSet,
        pipeline: Sequence[OperatorCall],
        result: SolveResult,
    ) -> str | None:
        """Try to promote a state's answer table to ``result.table``.

        Returns ``None`` on success, otherwise a diagnostic.  ``result.table`` is
        set *only* here, because Completion Rate (Sec 6.1) keys off it: a case
        that errored or produced an empty result must stay incomplete.
        """
        name = select_answer_table(state, task.target_schema)
        if name is None:
            return "EmptyAnswer: the final state contains no table."
        df = state[name].df
        if len(df.columns) == 0:
            return f"EmptyAnswer: the answer table '{name}' has no columns."
        if len(df) == 0:
            return f"EmptyResult: the answer table '{name}' has 0 rows."
        result.table = df.copy()
        result.pipeline = list(pipeline)
        result.pipeline_source = "\n".join(op.to_source() for op in pipeline)
        return None

    def _record_fallback(
        self,
        task: ADPTask,
        candidates: Sequence[tuple[TableSet, Sequence[OperatorCall]]],
        result: SolveResult,
    ) -> None:
        """Keep the most target-schema-like intermediate table as a fallback.

        Not counted by Completion Rate, but it gives the Eq. (8) partial
        similarity something to score, so a baseline that got most of the way
        there is not reported as indistinguishable from one that did nothing.
        """
        best: tuple[float, Any, Sequence[OperatorCall]] | None = None
        for i, (state, pipeline) in enumerate(candidates):
            name = select_answer_table(state, task.target_schema)
            if name is None:
                continue
            df = state[name].df
            if len(df) == 0 or len(df.columns) == 0:
                continue
            # Depth is a weak tiebreak toward the more processed candidate.
            score = schema_alignment(df.columns, task.target_schema) + 1e-6 * i
            if best is None or score > best[0]:
                best = (score, df, pipeline)
        if best is not None:
            result.fallback_table = best[1].copy()
            result.pipeline = list(best[2])
            result.pipeline_source = "\n".join(op.to_source() for op in best[2])
