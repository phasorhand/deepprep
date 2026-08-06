"""The execution environment (paper Sec 3).

    "DeepPrep iteratively interacts with an environment that executes data
     transformations and provides feedback.  The environment maintains an
     execution state represented by a set of intermediate tables, initialized
     with the source tables S.  Operator instances are executed within the
     environment through concrete execution backends (e.g. a Python interpreter),
     which apply transformations to the current state and produce updated tables
     along with execution feedback."

Sec 4.2 defines the semantics of ``<execute>`` precisely:

    "the environment executes its operators *sequentially* over the materialized
     tables at the selected parent node.  After each operator is successfully
     applied, a new intermediate state node is created ...  If execution fails
     due to a runtime exception (e.g. column mismatch or type error), no new
     state node is created; instead, the parent node is annotated with a failure
     state that records the error trace."

So a partially-successful pipeline keeps the states of its successful prefix —
which is exactly what makes the tree's "reuse valid operator prefixes" property
work.
"""

from __future__ import annotations

import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..operators import OperatorCall, OperatorError, ParseError, parse_pipeline
from ..serialize import serialize_table_set
from ..types import TableSet

__all__ = ["ExecutionResult", "Environment", "EnvironmentLimits"]


@dataclass
class EnvironmentLimits:
    """Guards against pathological model output.

    These are *environment* limits, not paper hyper-parameters; they exist so a
    single bad ``ExeCode`` cannot wedge a whole evaluation run.
    """

    #: Wall-clock seconds allowed for one operator.
    op_timeout_s: float = 30.0
    #: Maximum operators in a single ``<expand>`` pipeline.
    max_ops_per_pipeline: int = 20
    #: Reject a state whose tables exceed this many rows in total.
    max_total_rows: int = 5_000_000
    #: Maximum tables that may coexist in a state.
    max_tables: int = 64


class _Timeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Best-effort wall-clock limit around a single operator.

    Uses ``SIGALRM``, which requires the main thread of a Unix process.  When
    unavailable (Windows, worker threads) execution proceeds unbounded rather
    than failing — the limit is a safety net, not a correctness requirement.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    try:
        previous = signal.getsignal(signal.SIGALRM)
    except ValueError:  # pragma: no cover - not the main thread
        yield
        return

    def handler(signum: int, frame: object) -> None:
        raise _Timeout(f"operator exceeded the {seconds:g}s execution time limit")

    try:
        signal.signal(signal.SIGALRM, handler)
    except ValueError:  # pragma: no cover - not the main thread
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class ExecutionResult:
    """Outcome of executing one candidate pipeline ``P_t`` from a parent state."""

    #: Operators that were applied successfully, in order.
    applied: list[OperatorCall] = field(default_factory=list)
    #: ``states[i]`` is the state produced by ``applied[i]``.  Same length as ``applied``.
    states: list[TableSet] = field(default_factory=list)
    #: The operator that failed, if any.
    failed_op: OperatorCall | None = None
    #: Source text of the failed operator (available even when parsing failed).
    failed_source: str = ""
    #: The error trace surfaced to the agent.
    error: str | None = None
    #: Seconds spent inside the environment.
    elapsed_s: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def final_state(self) -> TableSet | None:
        return self.states[-1] if self.states else None

    @property
    def n_applied(self) -> int:
        return len(self.applied)


class Environment:
    """Executes operator pipelines over materialized table states."""

    def __init__(
        self,
        sources: TableSet,
        limits: EnvironmentLimits | None = None,
    ) -> None:
        self.sources = sources
        self.limits = limits or EnvironmentLimits()
        # Instrumentation used by the cost metric (Sec 6.1) and by the tests.
        self.n_operator_executions = 0
        self.n_failures = 0
        self.total_exec_time_s = 0.0

    # -- state ------------------------------------------------------------- #
    def initial_state(self) -> TableSet:
        """``T_0 = S`` — the root node's state (Sec 2.1)."""
        return self.sources.copy()

    # -- execution --------------------------------------------------------- #
    def execute(
        self, state: TableSet, pipeline: list[OperatorCall] | str
    ) -> ExecutionResult:
        """Apply ``pipeline`` sequentially from ``state``.

        Returns an :class:`ExecutionResult` holding the state produced by *each*
        successful operator, so the caller can materialize one tree node per
        operator (Sec 4.2: "Repeating this process for all operators in P_t
        yields a chain of new nodes").
        """
        result = ExecutionResult()
        t0 = time.perf_counter()

        if isinstance(pipeline, str):
            try:
                ops = parse_pipeline(pipeline)
            except ParseError as e:
                result.error = f"ParseError: {e}"
                result.elapsed_s = time.perf_counter() - t0
                self.n_failures += 1
                return result
        else:
            ops = list(pipeline)

        if not ops:
            result.error = (
                "EmptyPipeline: the <expand> block contained no operator call. "
                "Emit at least one operator, e.g. SelectColumn(table, [col])."
            )
            result.elapsed_s = time.perf_counter() - t0
            self.n_failures += 1
            return result

        if len(ops) > self.limits.max_ops_per_pipeline:
            result.error = (
                f"TooManyOperators: a single expansion may contain at most "
                f"{self.limits.max_ops_per_pipeline} operators, got {len(ops)}. "
                f"Split the plan across several turns."
            )
            result.elapsed_s = time.perf_counter() - t0
            self.n_failures += 1
            return result

        current = state.copy()
        for op in ops:
            self.n_operator_executions += 1
            try:
                with _time_limit(self.limits.op_timeout_s):
                    current = op.execute(current)
                self._check_limits(current)
            except (_Timeout, OperatorError) as e:
                result.failed_op = op
                result.failed_source = op.to_source()
                result.error = str(e) if isinstance(e, OperatorError) else f"Timeout: {e}"
                self.n_failures += 1
                break
            except Exception as e:  # noqa: BLE001 - never let the env crash the run
                result.failed_op = op
                result.failed_source = op.to_source()
                result.error = f"{type(e).__name__}: {e}"
                self.n_failures += 1
                break
            result.applied.append(op)
            result.states.append(current)
            current = current.copy()

        result.elapsed_s = time.perf_counter() - t0
        self.total_exec_time_s += result.elapsed_s
        return result

    def _check_limits(self, state: TableSet) -> None:
        if len(state) > self.limits.max_tables:
            raise OperatorError(
                f"StateTooLarge: the state now holds {len(state)} tables (limit "
                f"{self.limits.max_tables}). Drop intermediate tables you no longer need."
            )
        total = sum(t.n_rows for t in state)
        if total > self.limits.max_total_rows:
            raise OperatorError(
                f"StateTooLarge: the state now holds {total:,} rows in total (limit "
                f"{self.limits.max_total_rows:,}). This usually means a Join exploded — "
                f"check that the join keys are unique on at least one side."
            )

    # -- feedback ---------------------------------------------------------- #
    def render_feedback(
        self,
        result: ExecutionResult,
        max_rows: int = 5,
        max_tables_shown: int = 8,
    ) -> str:
        """Render the ``<execute>`` observation returned to the agent.

        Per Sec 3 the environment returns "execution feedback, such as sample
        outputs or error traces" — so success shows the materialized tables and
        failure shows the trace *plus* the prefix that did succeed, which is what
        lets the agent attribute the error to the right decision point.
        """
        lines: list[str] = []
        for i, (op, st) in enumerate(zip(result.applied, result.states, strict=True), start=1):
            lines.append(f"[{i}/{len(result.applied)}] OK  {op.to_source()}")
            if i == len(result.applied):
                lines.append(self._render_state(st, max_rows, max_tables_shown))

        if result.error is not None:
            idx = len(result.applied) + 1
            src = result.failed_source or "<unparsed operator>"
            lines.append(f"[{idx}] FAILED  {src}")
            lines.append(f"ERROR: {result.error}")
            if result.applied:
                lines.append(
                    f"The first {len(result.applied)} operator(s) were applied and their "
                    f"states are kept in the tree; only this operator was rejected."
                )
            else:
                lines.append(
                    "No operator was applied, so the tree is unchanged. Revise the plan "
                    "or expand from a different parent node."
                )
        elif not result.applied:
            lines.append("(nothing executed)")

        return "\n".join(lines).strip()

    @staticmethod
    def _render_state(state: TableSet, max_rows: int, max_tables_shown: int) -> str:
        """Serialize a state, listing (rather than materializing) surplus tables."""
        tables = list(state)
        if len(tables) <= max_tables_shown:
            return "Current state:\n" + serialize_table_set(
                state, max_rows=max_rows, with_description=False
            )
        shown = TableSet(tables[-max_tables_shown:])
        hidden = [t.name for t in tables[:-max_tables_shown]]
        return (
            "Current state:\n"
            + serialize_table_set(shown, max_rows=max_rows, with_description=False)
            + f"\n\n(also in scope, not shown: {hidden})"
        )
