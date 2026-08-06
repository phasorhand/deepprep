"""MCTS-OP baseline (paper Sec 6.1, "Prompting Baselines").

    "**MCTS-OP** applies Monte Carlo Tree Search, using local node expansion and
     scalar rewards to guide search. We follow the framework of [22] (Alpha-SQL)."

A standard UCT loop over environment states — selection by UCB1, local expansion
by sampling ``n_expand`` candidate operators from the LLM, a rollout that
completes the pipeline, and backpropagation of a **scalar** reward.

Sec 7 states exactly why this is a baseline and not the method:

    "MCTS-style approaches typically rely on rollout-based value estimates and
     scalar reward signals to guide node selection and backpropagation. However,
     data preparation pipelines expose structured execution feedback, including
     intermediate table states, schema changes, and runtime errors ... DeepPrep is
     designed to operate directly over such execution-grounded states ... rather
     than scalar rollout statistics."

That contrast is implemented literally: a rollout's structured outcome — the
tables it produced, the schema it reached, the error it hit — is **collapsed into
one float** before being backpropagated, and the rollout's own states are then
discarded rather than added to the tree.  The search therefore remembers "this
subtree scored 0.62", not "this subtree lost the ``values`` column at depth 3".

The reward never sees ``task.target_table``: the gold table "is used only for
evaluation" (Sec 2.1).  It combines

* a schema-alignment term against the *target schema* ``Sigma*``
  (:func:`~deepprep.baselines.common.schema_alignment`, the ``S_sch`` term of
  Eq. 8 computed against the task input), and
* an LLM self-assessment, i.e. the value model of the Alpha-SQL framework,

which is the strongest scalar an agent may legitimately compute for itself.
"""

from __future__ import annotations

import itertools
import math
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pandas as pd

from ..agent.actions import extract_block, strip_think
from ..agent.agent import SolveResult, TrajectoryStep, select_answer_table
from ..agent.llm import LLMClient
from ..env import Environment, EnvironmentLimits
from ..operators import OperatorCall, ParseError, parse_operator_call, split_calls
from ..serialize import serialize_table_set, serialize_task_input
from ..types import ADPTask, TableSet
from .common import (
    BaselineAgent,
    LLMCallError,
    Message,
    operator_space_section,
    schema_alignment,
)

__all__ = ["MCTSOperatorAgent", "MCTSNode"]

_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


SYSTEM_PROMPT = f"""
You are a data preparation assistant searching for an operator pipeline that
turns a set of source tables into a table conforming to a target schema Sigma*.

{operator_space_section()}

# Rules
1. The target table's contents are NOT given to you; derive them from the sources.
2. Every operator must reference tables and columns that exist in the state shown.
3. Prefer the predefined operators; use `ExeCode` only for transformations none
   of them can express.
""".strip()


EXPAND_PROMPT = """
Propose {k} DIFFERENT candidate operators to apply next to the state above. They
are alternatives, not a sequence: each one is explored as a separate branch, so
they should embody genuinely different next steps.

Reply with <candidates>...</candidates> containing one operator per line, and
nothing else.
""".strip()


ROLLOUT_PROMPT = """
Complete the pipeline from the state above: write the remaining operators, one
per line, ending with Terminate([<table_name>]) naming the table that holds the
final result. Then judge how promising this branch is.

Reply with exactly two blocks and nothing else:
<pipeline>
the remaining operators
</pipeline>
<score>
a single number in [0, 1]: 1 means this branch reaches the target schema exactly,
0 means it is a dead end
</score>
""".strip()


# --------------------------------------------------------------------------- #
# Search tree
# --------------------------------------------------------------------------- #
@dataclass
class MCTSNode:
    """One node of the UCT tree: an environment state and its scalar statistics.

    Note what is *not* stored: any record of what went wrong below this node.
    ``value_sum`` and ``visits`` are the entire memory of the search, which is the
    representational limitation Sec 7 attributes to MCTS-style methods.
    """

    id: str
    state: TableSet
    parent: MCTSNode | None = None
    op: OperatorCall | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    #: True once the node has been given its one round of local expansion.
    expanded: bool = False

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def path_ops(self) -> list[OperatorCall]:
        ops: list[OperatorCall] = []
        node: MCTSNode | None = self
        while node is not None and node.op is not None:
            ops.append(node.op)
            node = node.parent
        return list(reversed(ops))

    def ucb1(self, exploration_c: float) -> float:
        """``Q + c * sqrt(ln N_parent / N_child)`` — unvisited children come first."""
        if self.visits == 0:
            return math.inf
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.value + exploration_c * math.sqrt(
            math.log(max(parent_visits, 1)) / self.visits
        )


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class MCTSOperatorAgent(BaselineAgent):
    """UCT over operator sequences, guided by a scalar value estimate."""

    METHOD = "MCTS-OP"

    def __init__(
        self,
        llm: LLMClient,
        n_iterations: int = 8,
        n_expand: int = 3,
        exploration_c: float = math.sqrt(2),
        max_depth: int = 8,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        max_rows_in_prompt: int = 5,
        limits: EnvironmentLimits | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            llm,
            temperature=temperature,
            max_tokens=max_tokens,
            max_rows_in_prompt=max_rows_in_prompt,
            limits=limits,
            verbose=verbose,
        )
        self.n_iterations = n_iterations
        self.n_expand = n_expand
        self.exploration_c = exploration_c
        self.max_depth = max_depth

    # -- main loop ---------------------------------------------------------- #
    def solve(self, task: ADPTask) -> SolveResult:
        t_start = time.perf_counter()
        env = Environment(task.sources, limits=self.limits)
        result = SolveResult(task_id=task.task_id)

        root = MCTSNode(id="n0", state=env.initial_state())
        task_input = serialize_task_input(task, max_rows=self.max_rows_in_prompt)
        # Node ids live in this closure, not on ``self``: ``evaluate`` shares one
        # solver across a thread pool, so per-solve state must stay per-solve.
        ids: Iterator[int] = itertools.count(1)

        def next_id() -> str:
            return f"n{next(ids)}"

        #: Best complete pipeline any rollout has produced, by scalar reward.
        best: tuple[float, pd.DataFrame, list[OperatorCall]] | None = None

        for it in range(self.n_iterations):
            step = TrajectoryStep(turn=it, observation="", response="")
            result.trajectory.append(step)
            result.n_turns = it + 1

            # 1. Selection -- descend by UCB1 until an unexpanded node is reached.
            node = self._select(root)
            step.parent_node_id = node.id
            step.observation = self._render_node(task_input, node)

            try:
                # 2. Expansion -- local, k candidate operators from the LLM.
                if not node.expanded and node.depth < self.max_depth:
                    self._expand(env, node, task_input, result, step, next_id)
                target = self._best_child(node) or node
                step.created_node_ids = [c.id for c in node.children]

                # 3. Simulation -- one rollout completing the pipeline.
                reward, rollout = self._simulate(task, env, target, task_input, result, step)
            except LLMCallError as e:
                return self._abort_llm_error(result, e, t_start, it)

            if rollout is not None and (best is None or reward > best[0]):
                best = (reward, rollout[0], rollout[1])

            # 4. Backpropagation -- a single float travels back up the path.
            self._backpropagate(target, reward)
            step.feedback = f"reward={reward:.4f} at {target.id} (depth {target.depth})"

        # ---- answer with the best-scoring rollout --------------------------- #
        if best is not None:
            result.table = best[1].copy()
            result.pipeline = list(best[2])
            result.pipeline_source = "\n".join(op.to_source() for op in best[2])
            result.stop_reason = "answered"
            result.elapsed_s = time.perf_counter() - t_start
            return result

        result.stop_reason = "max_turns"
        result.error = "No rollout produced a table conforming to the target schema."
        self._record_fallback(
            task, [(n.state, n.path_ops()) for n in self._walk(root)], result
        )
        result.elapsed_s = time.perf_counter() - t_start
        return result

    # -- 1. selection ------------------------------------------------------- #
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Descend by UCB1 while the node is fully expanded and has children."""
        node = root
        while node.expanded and node.children and node.depth < self.max_depth:
            node = max(node.children, key=lambda c: c.ucb1(self.exploration_c))
        return node

    # -- 2. expansion ------------------------------------------------------- #
    def _expand(
        self,
        env: Environment,
        node: MCTSNode,
        task_input: str,
        result: SolveResult,
        step: TrajectoryStep,
        next_id: Callable[[], str],
    ) -> None:
        """Sample ``n_expand`` candidate operators and keep the ones that execute."""
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{self._render_node(task_input, node)}\n\n"
                    f"{EXPAND_PROMPT.format(k=self.n_expand)}"
                ),
            },
        ]
        resp = self._generate(messages, result)
        step.response += resp.text
        node.expanded = True

        body = extract_block(strip_think(resp.text), "candidates") or resp.text
        errors: list[str] = []
        for src in split_calls(body)[: self.n_expand]:
            try:
                op = parse_operator_call(src)
            except ParseError as e:
                errors.append(f"{src}: {e}")
                continue
            exec_result = env.execute(node.state, [op])
            if exec_result.error is not None or exec_result.final_state is None:
                # A failed candidate contributes nothing but a lost sample: MCTS
                # has no mechanism to turn the error trace into a better proposal.
                errors.append(f"{src}: {exec_result.error}")
                continue
            child = MCTSNode(
                id=next_id(),
                state=exec_result.final_state,
                parent=node,
                op=op,
            )
            node.children.append(child)
        if errors:
            step.error = "; ".join(errors)[:2000]

    def _best_child(self, node: MCTSNode) -> MCTSNode | None:
        """The child to run the rollout from: least-visited first, then UCB1."""
        if not node.children:
            return None
        return max(node.children, key=lambda c: c.ucb1(self.exploration_c))

    # -- 3. simulation ------------------------------------------------------ #
    def _simulate(
        self,
        task: ADPTask,
        env: Environment,
        node: MCTSNode,
        task_input: str,
        result: SolveResult,
        step: TrajectoryStep,
    ) -> tuple[float, tuple[pd.DataFrame, list[OperatorCall]] | None]:
        """Roll out to a terminal state and collapse the outcome to one float.

        Returns ``(reward, (table, pipeline) | None)``.  The rollout's own
        intermediate states are *not* added to the tree — this is the "rollout-based
        value estimate" of Sec 7, and the discarded structure is precisely what
        DeepPrep's reasoning tree would have retained.
        """
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{self._render_node(task_input, node)}\n\n{ROLLOUT_PROMPT}",
            },
        ]
        resp = self._generate(messages, result)
        step.response += ("\n" if step.response else "") + resp.text
        cleaned = strip_think(resp.text)

        self_score = _parse_score(extract_block(cleaned, "score"))
        body = extract_block(cleaned, "pipeline") or ""

        # A rollout that cannot even be parsed still yields a reward: the current
        # state's own schema alignment, which is what the branch is worth so far.
        base = schema_alignment(self._answer_columns(task, node.state), task.target_schema)
        ops = parse_pipeline_safe(body)
        if not ops:
            return self._blend(base, self_score), None

        exec_result = env.execute(node.state, ops)
        if exec_result.error is not None or exec_result.final_state is None:
            # Structured failure information exists here and is thrown away.
            return self._blend(base, self_score * 0.5), None

        final = exec_result.final_state
        name = select_answer_table(final, task.target_schema)
        if name is None:
            return self._blend(base, self_score * 0.5), None

        df = final[name].df
        aligned = schema_alignment(df.columns, task.target_schema)
        reward = self._blend(aligned, self_score)
        if len(df) == 0 or len(df.columns) == 0:
            # Sec 6.1 counts an empty result as incomplete, so it must never win.
            return reward * 0.5, None
        return reward, (df, node.path_ops() + list(exec_result.applied))

    @staticmethod
    def _blend(alignment: float, self_score: float) -> float:
        """The scalar reward: half grounded in ``Sigma*``, half self-assessed."""
        return 0.5 * alignment + 0.5 * self_score

    # -- 4. backpropagation -------------------------------------------------- #
    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        cursor: MCTSNode | None = node
        while cursor is not None:
            cursor.visits += 1
            cursor.value_sum += reward
            cursor = cursor.parent

    # -- rendering ----------------------------------------------------------- #
    def _render_node(self, task_input: str, node: MCTSNode) -> str:
        applied = "\n".join(op.to_source() for op in node.path_ops()) or "(none yet)"
        state = serialize_table_set(
            node.state, max_rows=self.max_rows_in_prompt, with_description=False
        )
        return (
            f"{task_input}\n\n"
            f"## Operators applied so far\n{applied}\n\n"
            f"## Current state\n{state}"
        )

    @staticmethod
    def _answer_columns(task: ADPTask, state: TableSet) -> list[str]:
        name = select_answer_table(state, task.target_schema)
        return list(state[name].columns) if name else []

    @staticmethod
    def _walk(root: MCTSNode) -> list[MCTSNode]:
        out: list[MCTSNode] = []
        stack = [root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out


def parse_pipeline_safe(text: str) -> list[OperatorCall]:
    """Parse a rollout, dropping operators that do not parse.

    A rollout is a heuristic estimate, not a committed decision, so one malformed
    line should degrade the estimate rather than void it.
    """
    ops: list[OperatorCall] = []
    for src in split_calls(text):
        try:
            ops.append(parse_operator_call(src))
        except ParseError:
            continue
    return ops


def _parse_score(body: str | None) -> float:
    """Read the LLM's self-assessment, clamped to [0, 1].

    Defaults to 0.5 rather than 0 so an unparseable score is neutral and does not
    silently prune an otherwise promising branch.
    """
    if not body:
        return 0.5
    m = _SCORE_RE.search(body)
    if not m:
        return 0.5
    return max(0.0, min(1.0, float(m.group(0))))
