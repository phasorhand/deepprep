"""Plan-and-Solve baseline (paper Sec 6.1, "Prompting Baselines").

    "**Plan-and-Solve (PaS)** [37] prompts an LLM using a two-stage strategy that
     first produces a high-level plan and then generates a complete operator
     sequence to obtain the target table."

Two generations, in this order:

1. ``<plan>`` — a high-level, natural-language decomposition of the task.
2. ``<pipeline>`` — the *entire* operator sequence ``P``, emitted in one shot.

The pipeline is then executed once.  There is no third call: whatever the
environment reports is the outcome.  PaS therefore has the decomposition that
CodeGen lacks but still, in the words of Sec 1, "make[s] decisions without
systematic grounding in intermediate execution results" — the plan is written
before a single operator has been run, and the operator sequence is written
before a single intermediate table has been seen.

When execution fails halfway, the successful prefix's state is kept as a
fallback table (it feeds the Eq. (8) partial similarity) but the case is *not*
completed: Sec 6.1 counts a run that "triggers runtime errors" as incomplete.
"""

from __future__ import annotations

import time

from ..agent.actions import extract_block, strip_think
from ..agent.agent import SolveResult, TrajectoryStep
from ..env import Environment
from ..operators import ParseError, parse_pipeline
from ..serialize import serialize_task_input
from ..types import ADPTask
from .common import BaselineAgent, LLMCallError, Message, operator_space_section

__all__ = ["PlanAndSolveAgent"]


SYSTEM_PROMPT = f"""
You are a data preparation assistant. You are given a set of source tables and a
target schema Sigma* describing the table an analyst needs. You solve the task in
two stages: first you devise a plan, then you carry it out as one operator
sequence.

{operator_space_section()}

# Rules
1. The target table's contents are NOT given to you; derive them from the source
   samples and the target schema.
2. The pipeline runs once, without interaction. Anticipate the data's problems
   (duplicates, padded or miscased keys, multi-valued cells, values packed into a
   single column) from the samples you are shown.
3. Every operator must reference tables and columns that exist at the point it
   runs. Operators execute sequentially and each one rewrites the state.
4. Prefer the predefined operators; use `ExeCode` only for transformations none
   of them can express.
5. The last operator must be `Terminate([<table_name>])`, naming the table that
   holds the final result.
""".strip()


PLAN_PROMPT = """
Stage 1 of 2. Devise a plan: understand the problem and lay out a step-by-step
plan of the transformations needed to turn the source tables into a table
conforming to the target schema. Name the tables and columns involved at each
step. Do not write operators yet.

Reply with <plan>...</plan> and nothing else.
""".strip()


SOLVE_PROMPT = """
Stage 2 of 2. Carry out the plan: write the complete operator sequence, one
operator per line, from the source tables to the final result. It will be
executed exactly as written, with no opportunity to revise it. End with
Terminate([<table_name>]).

Reply with <pipeline>...</pipeline> and nothing else.
""".strip()


class PlanAndSolveAgent(BaselineAgent):
    """Plan once, emit the whole pipeline once, execute once.

    The absence of a feedback loop is the defining property of this baseline and
    is enforced structurally: :meth:`solve` makes exactly two LLM calls.
    """

    METHOD = "Plan-and-Solve"

    def solve(self, task: ADPTask) -> SolveResult:
        t_start = time.perf_counter()
        env = Environment(task.sources, limits=self.limits)
        state = env.initial_state()
        result = SolveResult(task_id=task.task_id)

        task_msg = serialize_task_input(task, max_rows=self.max_rows_in_prompt)
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{task_msg}\n\n{PLAN_PROMPT}"},
        ]

        # ---- stage 1: plan ------------------------------------------------- #
        try:
            plan_resp = self._generate(messages, result)
        except LLMCallError as e:
            result.messages = messages
            return self._abort_llm_error(result, e, t_start, 0)

        messages.append({"role": "assistant", "content": plan_resp.text})
        result.trajectory.append(
            TrajectoryStep(
                turn=0,
                observation=messages[1]["content"],
                response=plan_resp.text,
                feedback="(plan recorded; nothing executed)",
                usage=plan_resp.usage.to_dict(),
            )
        )

        # ---- stage 2: solve ------------------------------------------------ #
        messages.append({"role": "user", "content": SOLVE_PROMPT})
        try:
            solve_resp = self._generate(messages, result)
        except LLMCallError as e:
            result.messages = messages
            result.n_turns = 1
            return self._abort_llm_error(result, e, t_start, 1)

        messages.append({"role": "assistant", "content": solve_resp.text})
        result.messages = messages
        result.n_turns = 2

        step = TrajectoryStep(
            turn=1,
            observation=SOLVE_PROMPT,
            response=solve_resp.text,
            usage=solve_resp.usage.to_dict(),
        )
        result.trajectory.append(step)

        body = extract_block(strip_think(solve_resp.text), "pipeline")
        if not body:
            # Tolerate a model that answers in <answer> or with a bare listing;
            # a formatting slip should cost accuracy, not be scored as a crash.
            body = extract_block(strip_think(solve_resp.text), "answer") or solve_resp.text

        try:
            pipeline = parse_pipeline(body)
        except ParseError as e:
            msg = f"ParseError in <pipeline>: {e}"
            step.error, step.feedback = msg, msg
            return self._finish(result, t_start, msg)

        if not pipeline:
            msg = "EmptyPipeline: the reply contained no operator call."
            step.error, step.feedback = msg, msg
            return self._finish(result, t_start, msg)

        # ---- execute the whole sequence, once ------------------------------ #
        exec_result = env.execute(state, pipeline)
        step.n_ops_applied = exec_result.n_applied
        step.feedback = env.render_feedback(exec_result, max_rows=self.max_rows_in_prompt)

        if exec_result.error is not None:
            step.error = exec_result.error
            # No repair turn: the feedback exists in the trajectory for analysis,
            # but PaS never gets to read it.
            if exec_result.final_state is not None:
                self._record_fallback(
                    task, [(exec_result.final_state, exec_result.applied)], result
                )
            return self._finish(result, t_start, exec_result.error)

        final = exec_result.final_state or state
        error = self._accept_answer(task, final, exec_result.applied, result)
        if error is not None:
            step.error = error
            self._record_fallback(task, [(final, exec_result.applied)], result)
            return self._finish(result, t_start, error)

        result.stop_reason = "answered"
        result.elapsed_s = time.perf_counter() - t_start
        return result

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _finish(result: SolveResult, t_start: float, error: str) -> SolveResult:
        """Close out a run that never produced a usable table.

        ``stop_reason`` is "max_turns" because PaS's budget of two calls is
        exhausted the moment the single pipeline fails — there is no turn left in
        which the model could react to the error.
        """
        result.stop_reason = "max_turns"
        result.error = error
        result.elapsed_s = time.perf_counter() - t_start
        return result
