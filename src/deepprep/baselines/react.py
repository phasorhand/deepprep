"""ReAct baseline (paper Sec 6.1, "Prompting Baselines").

    "**ReAct** [2] is a linear reasoning agent that iteratively predicts the next
     operator based on previous execution results."

ReAct is the ablation the paper's whole argument rests on, so its limitation is
implemented *structurally* rather than merely described.

The agent keeps **one** variable, ``state``.  Each turn it emits a thought and
one operator (or a short group of operators), the environment executes them, and
the resulting state **overwrites** the previous one.  There is no tree, no set of
frontier nodes, and no addressing scheme for earlier states — therefore no
backtracking is expressible, even in principle.  Sec 1 states the consequence:

    "existing agentic methods typically follow a linear reasoning paradigm ...
     Such linear reasoning makes it difficult to revise earlier decisions, since
     the effects of previous operators are already reflected in the current
     state."

Concretely: if turn 1 runs ``SelectColumn(ratings, [movie, rating])`` and turn 3
discovers that the dropped ``values`` column was needed, the column is simply
gone.  Re-emitting ``SelectColumn(ratings, [movie, values])`` fails against the
mutated state, and the run is lost.  DeepPrep's ``<expand><from>`` prefix
addressing exists precisely to make that recovery possible; ReAct has no
equivalent, and this class deliberately provides none.

``n_backtracks`` on the returned :class:`~deepprep.agent.agent.SolveResult` is
therefore always 0 — that is a property of the method, not a missing feature.
"""

from __future__ import annotations

import time

from ..agent.actions import extract_block, strip_think
from ..agent.agent import SolveResult, TrajectoryStep
from ..agent.llm import LLMClient
from ..env import Environment, EnvironmentLimits
from ..operators import OperatorCall, ParseError, parse_pipeline
from ..serialize import serialize_task_input
from ..types import ADPTask, TableSet
from .common import BaselineAgent, LLMCallError, Message, operator_space_section

__all__ = ["ReActAgent"]


SYSTEM_PROMPT = f"""
You are a data preparation assistant. You are given a set of source tables and a
target schema Sigma* describing the table an analyst needs. You reach it by
interacting with an execution environment one step at a time.

{operator_space_section()}

# Interaction protocol

The environment holds a single current state: a set of intermediate tables,
initialized with the source tables. At every turn you emit exactly two blocks.

<thought>
Read the current state and the last execution result. State what the current
state still lacks with respect to the target schema, and what the next step is.
</thought>

<action>
The next operator to apply, one per line. Keep this short -- one operator, or a
few that clearly belong together. They execute sequentially and each one rewrites
the current state.
</action>

When the current state already contains a table conforming to the target schema,
emit <answer>Terminate([<table_name>])</answer> instead of <action>.

The environment replies with an <observation> block containing the operators it
applied, the resulting tables, and any error trace. You never write it yourself.

# Rules
1. The state is overwritten by every operator you apply. There is no undo and no
   way to return to an earlier state, so do not discard a column or a row you may
   still need.
2. Ground every decision in the observed data: check the sample rows before
   choosing keys, separators or formats.
3. The target table's contents are NOT given to you; derive them from the sources.
4. Prefer the predefined operators; use `ExeCode` only for transformations none
   of them can express.
""".strip()


class ReActAgent(BaselineAgent):
    """Linear thought/action/observation loop over a single mutable state.

    The single-trajectory restriction is enforced by the data structure: the
    only state this class holds is ``state``, rebound in place after every
    successful expansion.  Nothing anywhere retains an earlier state, so an early
    wrong decision cannot be revised — only worked around within the state it
    left behind.
    """

    METHOD = "ReAct"

    def __init__(
        self,
        llm: LLMClient,
        max_turns: int = 5,
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
        # Matched to DeepPrep's budget: "The maximum exploration turns of DeepPrep
        # are set to 5." (Sec 6.1) -- the comparison must not turn on turn count.
        self.max_turns = max_turns

    # -- main loop ---------------------------------------------------------- #
    def solve(self, task: ADPTask) -> SolveResult:
        t_start = time.perf_counter()
        env = Environment(task.sources, limits=self.limits)
        result = SolveResult(task_id=task.task_id)

        # THE trajectory. One state, one pipeline, no alternatives retained.
        state = env.initial_state()
        pipeline: list[OperatorCall] = []

        task_msg = (
            f"{serialize_task_input(task, max_rows=self.max_rows_in_prompt)}\n\n"
            "Reply with <thought> followed by <action> (or <answer>)."
        )
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_msg},
        ]

        for turn in range(self.max_turns):
            if turn > 0:
                messages.append({"role": "user", "content": self._turn_prompt(turn)})

            try:
                resp = self._generate(messages, result)
            except LLMCallError as e:
                result.messages = messages
                return self._abort_llm_error(result, e, t_start, turn)

            messages.append({"role": "assistant", "content": resp.text})
            result.n_turns = turn + 1
            step = TrajectoryStep(
                turn=turn,
                observation=messages[-2]["content"],
                response=resp.text,
                usage=resp.usage.to_dict(),
            )
            result.trajectory.append(step)
            cleaned = strip_think(resp.text)

            # ---- <answer>: finish from the current state -------------------- #
            answer = extract_block(cleaned, "answer")
            if answer:
                error = self._answer(task, env, state, pipeline, answer, result, step)
                if error is None:
                    result.stop_reason = "answered"
                    result.messages = messages
                    result.elapsed_s = time.perf_counter() - t_start
                    return result
                messages.append({"role": "user", "content": f"<observation>\n{error}\n</observation>"})
                continue

            # ---- <action>: apply operators to the one current state --------- #
            action = extract_block(cleaned, "action")
            if not action:
                msg = (
                    "MissingAction: the reply contained neither <action> nor <answer>. "
                    "Emit <thought>...</thought> followed by <action>...</action>."
                )
                step.error, step.feedback = msg, msg
                messages.append({"role": "user", "content": f"<observation>\n{msg}\n</observation>"})
                continue

            try:
                ops = parse_pipeline(action)
            except ParseError as e:
                msg = f"ParseError: {e}"
                step.error, step.feedback = msg, msg
                messages.append({"role": "user", "content": f"<observation>\n{msg}\n</observation>"})
                continue

            exec_result = env.execute(state, ops)
            step.n_ops_applied = exec_result.n_applied
            feedback = env.render_feedback(exec_result, max_rows=self.max_rows_in_prompt)
            step.feedback = feedback
            step.error = exec_result.error

            if exec_result.final_state is not None:
                # *** The linear commitment. ***
                # The pre-operator state is dropped here and is unreachable
                # afterwards: nothing holds a reference to it and the protocol
                # offers no way to name it. This single line is what makes ReAct
                # unable to backtrack, and it is intentional -- a tree of states
                # is exactly the capability DeepPrep adds (Sec 4.2).
                state = exec_result.final_state
                pipeline.extend(exec_result.applied)

            messages.append(
                {"role": "user", "content": f"<observation>\n{feedback}\n</observation>"}
            )

        # ---- turn budget exhausted ----------------------------------------- #
        result.stop_reason = "max_turns"
        result.error = "Reached the maximum number of interaction turns without an answer."
        self._record_fallback(task, [(state, pipeline)], result)
        result.messages = messages
        result.elapsed_s = time.perf_counter() - t_start
        return result

    # -- helpers ------------------------------------------------------------ #
    def _answer(
        self,
        task: ADPTask,
        env: Environment,
        state: TableSet,
        pipeline: list[OperatorCall],
        answer_body: str,
        result: SolveResult,
        step: TrajectoryStep,
    ) -> str | None:
        """Realize ``<answer>`` against the current state.

        Unlike DeepPrep, there is no root-to-leaf path to re-materialize: the
        answer can only ever be a table of the single state ReAct has arrived at.
        Any operators inside the block (normally just ``Terminate``) are applied
        to that state.
        """
        try:
            ops = parse_pipeline(answer_body)
        except ParseError as e:
            msg = f"ParseError in <answer>: {e}"
            step.error, step.feedback = msg, msg
            return msg

        final = state
        applied = list(pipeline)
        if ops:
            exec_result = env.execute(state, ops)
            step.n_ops_applied = exec_result.n_applied
            if exec_result.error is not None:
                msg = env.render_feedback(exec_result, max_rows=self.max_rows_in_prompt)
                step.error, step.feedback = exec_result.error, msg
                return msg
            final = exec_result.final_state or state
            applied += list(exec_result.applied)

        error = self._accept_answer(task, final, applied, result)
        if error is not None:
            step.error, step.feedback = error, error
            return error
        step.feedback = "Answer accepted."
        return None

    def _turn_prompt(self, turn: int) -> str:
        remaining = self.max_turns - turn
        if remaining <= 1:
            return (
                "This is your LAST turn. If the current state already contains a table "
                "conforming to the target schema, emit <answer>Terminate([<table_name>])"
                "</answer> now; otherwise make this action the one that completes it."
            )
        return (
            f"Turn {turn + 1} of {self.max_turns}. Reply with <thought> followed by "
            f"<action> (or <answer>)."
        )
