"""The DeepPrep agent: tree-based agentic inference (paper Sec 4.2).

Implements the factorization of Eq. (3)::

    Pr(R_t | R_<t, S, Sigma*) =  Pr_LLM(psi_t | psi_<t, G_{t-1}, Sigma*)   <plan>
                               * Pr_LLM(P_t  | G_{t-1}, psi_t)             <expand>
                               * Pr_Env(G_t  | G_{t-1}, P_t)               <execute>

and the answer generation ``Pr(A | R, S, Sigma*)`` of Eq. (2) via ``<answer>``.

The trajectory recorded here is the training signal for both cold-start Stage 2
(Eq. 5) and multi-turn GRPO (Sec 5.2): each step separates *agent-generated*
tokens from *environment-generated* tokens so the loss/gradient mask can be
applied exactly as the paper specifies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..env import Environment, EnvironmentLimits
from ..operators import OperatorCall, ParseError, parse_pipeline, split_calls
from ..operators.program import ANSWER_ATTR
from ..serialize import serialize_task_input
from ..tree import Node, ReasoningTree
from ..types import ADPTask, TableSchema, TableSet
from .actions import AgentTurn, parse_agent_output
from .llm import LLMClient, Usage
from .prompts import (
    build_final_answer_prompt,
    build_system_prompt,
    build_task_prompt,
    build_turn_prompt,
)

__all__ = ["DeepPrepAgent", "SolveResult", "TrajectoryStep", "select_answer_table"]

Message = dict[str, str]


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
@dataclass
class TrajectoryStep:
    """One reasoning step ``r_t``.

    ``response`` holds the policy's tokens (``<plan>``, ``<expand>``, ``<answer>``)
    and ``observation``/``feedback`` hold the environment's.  Sec 5.2: "we apply a
    binary token mask that removes gradients on tokens inside <execute> blocks,
    and optimize the policy only over agent decision tokens".
    """

    turn: int
    observation: str
    response: str
    feedback: str = ""
    parent_node_id: str | None = None
    created_node_ids: list[str] = field(default_factory=list)
    failed_node_id: str | None = None
    n_ops_applied: int = 0
    error: str | None = None
    is_backtrack: bool = False
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "observation": self.observation,
            "response": self.response,
            "feedback": self.feedback,
            "parent_node_id": self.parent_node_id,
            "created_node_ids": self.created_node_ids,
            "failed_node_id": self.failed_node_id,
            "n_ops_applied": self.n_ops_applied,
            "error": self.error,
            "is_backtrack": self.is_backtrack,
            "usage": self.usage,
        }


@dataclass
class SolveResult:
    """Outcome of solving one ADP task."""

    task_id: str
    #: The produced target table ``T_hat``, or ``None`` if the agent never answered.
    table: pd.DataFrame | None = None
    #: Best-effort table when no ``<answer>`` was issued.  Used for the partial
    #: reward during RL; NOT counted by the Completion Rate metric.
    fallback_table: pd.DataFrame | None = None
    pipeline: list[OperatorCall] = field(default_factory=list)
    pipeline_source: str = ""
    tree: ReasoningTree | None = None
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    n_turns: int = 0
    #: "answered" | "max_turns" | "llm_error"
    stop_reason: str = ""
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def answered(self) -> bool:
        return self.table is not None

    @property
    def completed(self) -> bool:
        """Completion Rate criterion (Sec 6.1).

        "A case is counted as incomplete if execution exceeds the maximum
         interaction limit, triggers runtime errors, or produces an empty result."
        """
        return (
            self.stop_reason == "answered"
            and self.table is not None
            and len(self.table) > 0
            and len(self.table.columns) > 0
        )

    # -- diagnostics -------------------------------------------------------- #
    @property
    def n_backtracks(self) -> int:
        """How often the agent expanded from a node other than the newest leaf."""
        return sum(1 for s in self.trajectory if s.is_backtrack)

    @property
    def n_failed_expansions(self) -> int:
        return sum(1 for s in self.trajectory if s.failed_node_id is not None)

    def to_dict(self, with_tables: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_id": self.task_id,
            "pipeline": [op.to_source() for op in self.pipeline],
            "pipeline_source": self.pipeline_source,
            "n_turns": self.n_turns,
            "stop_reason": self.stop_reason,
            "answered": self.answered,
            "completed": self.completed,
            "n_backtracks": self.n_backtracks,
            "n_failed_expansions": self.n_failed_expansions,
            "usage": self.usage.to_dict(),
            "elapsed_s": round(self.elapsed_s, 4),
            "error": self.error,
            "trajectory": [s.to_dict() for s in self.trajectory],
        }
        if with_tables and self.table is not None:
            d["table"] = self.table.to_dict(orient="split")
        return d


# --------------------------------------------------------------------------- #
# Answer-table selection
# --------------------------------------------------------------------------- #
def select_answer_table(state: TableSet, target: TableSchema | None) -> str | None:
    """Pick which table of a state is ``T_hat``.

    ``Terminate`` names it explicitly.  When it is absent (a malformed answer, or
    a fallback at turn exhaustion) we score every table by Jaccard overlap
    between its column names and ``Sigma*``'s, breaking ties toward the most
    recently created table.
    """
    marked = getattr(state, ANSWER_ATTR, None)
    if marked and marked in state:
        return marked
    if len(state) == 0:
        return None

    want = {c.name.strip().lower() for c in (target.columns if target else [])}
    if not want:
        return state.names[-1]

    best_name, best_score = None, -1.0
    for i, t in enumerate(state):
        have = {c.strip().lower() for c in t.columns}
        inter = len(want & have)
        union = len(want | have) or 1
        # Recency is a weak tiebreak, so a genuinely better match always wins.
        score = inter / union + 1e-6 * i
        if score > best_score:
            best_name, best_score = t.name, score
    return best_name


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class DeepPrepAgent:
    """Constructs data preparation pipelines by tree-based agentic reasoning."""

    def __init__(
        self,
        llm: LLMClient,
        max_turns: int = 5,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        include_exemplar: bool = False,
        max_rows_in_prompt: int = 5,
        limits: EnvironmentLimits | None = None,
        verbose: bool = False,
        final_answer_turn: bool = True,
    ) -> None:
        # "The maximum exploration turns of DeepPrep are set to 5." (Sec 6.1)
        self.llm = llm
        self.max_turns = max_turns
        # "For all prompting methods, the temperature is set to 0.01." (Sec 6.1)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.include_exemplar = include_exemplar
        self.max_rows_in_prompt = max_rows_in_prompt
        self.limits = limits or EnvironmentLimits()
        self.verbose = verbose
        # Sec 6.1 caps exploration; committing an answer is not exploration.
        self.final_answer_turn = final_answer_turn

    # -- main loop ---------------------------------------------------------- #
    def solve(self, task: ADPTask) -> SolveResult:
        t_start = time.perf_counter()
        env = Environment(task.sources, limits=self.limits)
        tree = ReasoningTree(env.initial_state())
        result = SolveResult(task_id=task.task_id, tree=tree)

        system = build_system_prompt(
            include_exemplar=self.include_exemplar, max_turns=self.max_turns
        )
        task_msg = build_task_prompt(
            serialize_task_input(task, max_rows=self.max_rows_in_prompt)
        )
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task_msg},
        ]

        feedback: str | None = None
        last_expanded: Node = tree.root

        for turn in range(self.max_turns):
            observation = build_turn_prompt(
                tree.render(max_rows=self.max_rows_in_prompt), feedback, turn, self.max_turns
            )
            # Turn 0's observation is folded into the task message: the tree is
            # just the root, whose state is already serialized there.
            if turn == 0:
                messages[-1]["content"] = task_msg + "\n\n" + tree.render()
            else:
                messages.append({"role": "user", "content": observation})

            try:
                resp = self.llm.generate(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:  # noqa: BLE001 - a dead endpoint must not lose the run
                result.error = f"LLM call failed: {type(e).__name__}: {e}"
                result.stop_reason = "llm_error"
                result.n_turns = turn
                result.elapsed_s = time.perf_counter() - t_start
                result.messages = messages
                return result

            result.usage.add(resp.usage)
            messages.append({"role": "assistant", "content": resp.text})
            parsed = parse_agent_output(resp.text)

            step = TrajectoryStep(
                turn=turn,
                observation=observation if turn > 0 else messages[1]["content"],
                response=resp.text,
                usage=resp.usage.to_dict(),
            )

            if self.verbose:
                self._log(turn, parsed)

            # ---- <answer>: terminate -------------------------------------- #
            if parsed.answer is not None:
                feedback = self._finalize(task, env, tree, parsed, result, step)
                result.trajectory.append(step)
                if result.table is not None:
                    result.stop_reason = "answered"
                    result.n_turns = turn + 1
                    break
                # The claimed answer did not execute; report why and keep going.
                messages.append({"role": "user", "content": f"<execute>\n{feedback}\n</execute>"})
                result.n_turns = turn + 1
                continue

            # ---- <expand>: grow the tree ---------------------------------- #
            if parsed.expand is None:
                step.error = parsed.error
                feedback = parsed.error or "MissingAction"
                result.trajectory.append(step)
                result.n_turns = turn + 1
                continue

            feedback = self._expand(env, tree, parsed, step, turn, last_expanded)
            if step.parent_node_id:
                parent = tree.get(step.parent_node_id)
                if parent is not None:
                    last_expanded = (
                        tree.get(step.created_node_ids[-1]) if step.created_node_ids else parent
                    ) or parent
            result.trajectory.append(step)
            result.n_turns = turn + 1

        # ---- exhausted the turn budget ------------------------------------ #
        # One closing turn to commit an answer. The tree is frozen: only
        # <answer> is honoured, so this cannot buy an extra expansion.
        if result.stop_reason == "" and self.final_answer_turn:
            self._closing_answer_turn(task, env, tree, result, messages, feedback)

        if result.stop_reason == "":
            result.stop_reason = "max_turns"
            self._set_fallback(task, tree, result)

        result.messages = messages
        result.elapsed_s = time.perf_counter() - t_start
        return result

    # -- <expand> + <execute> ---------------------------------------------- #
    def _expand(
        self,
        env: Environment,
        tree: ReasoningTree,
        parsed: AgentTurn,
        step: TrajectoryStep,
        turn: int,
        last_expanded: Node,
    ) -> str:
        assert parsed.expand is not None
        parent, warning = tree.resolve_parent(parsed.expand.from_prefix)
        step.parent_node_id = parent.id
        # A backtrack is an expansion from anywhere other than the frontier the
        # previous turn produced -- the non-local revision of Sec 1.
        step.is_backtrack = turn > 0 and parent.id != last_expanded.id

        assert parent.state is not None, "ok nodes always carry a materialized state"
        exec_result = env.execute(parent.state, parsed.expand.ops_source)

        cursor = parent
        for op, state in zip(exec_result.applied, exec_result.states, strict=True):
            cursor = tree.add_state(cursor, op, state, turn=turn)
            step.created_node_ids.append(cursor.id)
        step.n_ops_applied = exec_result.n_applied

        if exec_result.error is not None:
            failed = tree.add_failure(
                cursor,
                exec_result.failed_op,
                exec_result.failed_source or parsed.expand.ops_source,
                exec_result.error,
                turn=turn,
            )
            step.failed_node_id = failed.id
            step.error = exec_result.error

        fb = env.render_feedback(exec_result, max_rows=self.max_rows_in_prompt)
        step.feedback = fb
        return f"{warning}\n\n{fb}" if warning else fb

    # -- <answer> ----------------------------------------------------------- #
    def _finalize(
        self,
        task: ADPTask,
        env: Environment,
        tree: ReasoningTree,
        parsed: AgentTurn,
        result: SolveResult,
        step: TrajectoryStep,
    ) -> str:
        """Realize ``<answer>``: extract ``P*`` and materialize ``T_hat``."""
        assert parsed.answer is not None
        steps = split_calls(parsed.answer.pipeline_source)
        if not steps:
            msg = (
                "EmptyAnswer: the <answer> block contained no operator call. It must list "
                "the full operator sequence from n0 to the leaf, ending with "
                "Terminate([<table>])."
            )
            step.error = msg
            step.feedback = msg
            return msg

        # Reuse the materialized prefix; execute only what the tree does not have.
        node, remaining = tree.resolve_longest_prefix(steps)
        assert node.state is not None
        state = node.state
        pipeline = list(node.path_ops())

        if remaining:
            try:
                ops = parse_pipeline("\n".join(remaining))
            except ParseError as e:
                msg = f"ParseError in <answer>: {e}"
                step.error = msg
                step.feedback = msg
                return msg
            exec_result = env.execute(state, ops)
            cursor = node
            for op, st in zip(exec_result.applied, exec_result.states, strict=True):
                cursor = tree.add_state(cursor, op, st, turn=step.turn)
                step.created_node_ids.append(cursor.id)
            pipeline += list(exec_result.applied)
            if exec_result.error is not None:
                failed = tree.add_failure(
                    cursor, exec_result.failed_op, exec_result.failed_source,
                    exec_result.error, turn=step.turn,
                )
                step.failed_node_id = failed.id
                msg = (
                    f"The answer pipeline failed to execute:\n"
                    f"{env.render_feedback(exec_result, max_rows=self.max_rows_in_prompt)}"
                )
                step.error = exec_result.error
                step.feedback = msg
                return msg
            state = exec_result.final_state or state

        name = select_answer_table(state, task.target_schema)
        if name is None:
            msg = "EmptyAnswer: the final state contains no table."
            step.error = msg
            step.feedback = msg
            return msg

        table = state[name].df
        if len(table.columns) == 0:
            msg = f"EmptyAnswer: the answer table '{name}' has no columns."
            step.error = msg
            step.feedback = msg
            return msg
        if len(table) == 0:
            # Sec 6.1 counts an empty result as incomplete, so this is worth one
            # more turn rather than an immediate failure.
            msg = (
                f"EmptyResult: the answer table '{name}' has 0 rows. A filter or a join "
                f"key is probably wrong. Backtrack to the node before the operator that "
                f"emptied it and expand a corrected branch."
            )
            step.error = msg
            step.feedback = msg
            return msg

        result.table = table.copy()
        result.pipeline = pipeline
        result.pipeline_source = "\n".join(op.to_source() for op in pipeline)
        step.feedback = f"Answer accepted: table '{name}' with shape {table.shape}."
        return step.feedback

    # -- fallback ----------------------------------------------------------- #
    # -- closing turn ------------------------------------------------------- #
    def _closing_answer_turn(
        self,
        task: ADPTask,
        env: Environment,
        tree: ReasoningTree,
        result: SolveResult,
        messages: list[Message],
        feedback: str | None,
    ) -> None:
        """Ask once for an ``<answer>`` after the expansion budget is spent."""
        observation = build_final_answer_prompt(
            tree.render(max_rows=self.max_rows_in_prompt), feedback
        )
        messages.append({"role": "user", "content": observation})
        try:
            resp = self.llm.generate(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
        except Exception as e:  # noqa: BLE001 - a dead endpoint must not lose the run
            result.error = f"LLM call failed: {type(e).__name__}: {e}"
            result.stop_reason = "llm_error"
            return

        result.usage.add(resp.usage)
        messages.append({"role": "assistant", "content": resp.text})
        parsed = parse_agent_output(resp.text)
        turn = self.max_turns
        step = TrajectoryStep(
            turn=turn,
            observation=observation,
            response=resp.text,
            usage=resp.usage.to_dict(),
        )
        if self.verbose:
            self._log(turn, parsed)

        if parsed.answer is None:
            # An <expand> here is ignored by construction: the budget is spent.
            step.error = parsed.error or "MissingAnswer"
            step.feedback = step.error
            result.trajectory.append(step)
            result.n_turns = turn + 1
            return

        self._finalize(task, env, tree, parsed, result, step)
        result.trajectory.append(step)
        result.n_turns = turn + 1
        if result.table is not None:
            result.stop_reason = "answered"

    def _set_fallback(self, task: ADPTask, tree: ReasoningTree, result: SolveResult) -> None:
        """Record the most promising leaf when the agent never answered.

        Not counted as completed (Sec 6.1), but it gives the partial reward of
        Eq. (8) something to score during RL instead of a flat zero.
        """
        best: tuple[float, pd.DataFrame, list[OperatorCall]] | None = None
        want = {c.name.strip().lower() for c in task.target_schema.columns}
        for node in tree.ok_nodes():
            if node.state is None:
                continue
            name = select_answer_table(node.state, task.target_schema)
            if name is None:
                continue
            df = node.state[name].df
            if len(df) == 0 or len(df.columns) == 0:
                continue
            have = {c.strip().lower() for c in df.columns}
            score = len(want & have) / (len(want | have) or 1) + 1e-4 * node.depth
            if best is None or score > best[0]:
                best = (score, df, node.path_ops())
        if best is not None:
            result.fallback_table = best[1].copy()
            result.pipeline = best[2]
            result.pipeline_source = "\n".join(op.to_source() for op in best[2])

    # -- logging ------------------------------------------------------------ #
    @staticmethod
    def _log(turn: int, parsed: AgentTurn) -> None:  # pragma: no cover - diagnostics
        head = f"[turn {turn}]"
        if parsed.plan:
            print(f"{head} plan: {parsed.plan.text[:160]}")
        if parsed.expand:
            print(f"{head} expand from {parsed.expand.from_prefix or 'n0'}:")
            print("    " + parsed.expand.ops_source.replace("\n", "\n    ")[:400])
        if parsed.answer:
            print(f"{head} answer:\n    " + parsed.answer.pipeline_source.replace("\n", "\n    "))
        if parsed.error:
            print(f"{head} parse error: {parsed.error}")
