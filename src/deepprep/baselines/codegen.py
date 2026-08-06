"""CodeGen baseline (paper Sec 6.1, "Prompting Baselines").

    "**CodeGen** [28, 32] prompts an LLM to generate data preparation code from
     the source tables and target schema."

This is the weakest form of the task: the model sees ``phi(S)`` and ``Sigma*``
once and must emit a complete program.  Sec 1 names the defect precisely — such
methods "make decisions without systematic grounding in intermediate execution
results", so a wrong assumption about a separator, a join key or a duplicated id
is never discovered, because no intermediate table is ever observed.

The implementation keeps that property intact:

* exactly **one** generation produces the whole program;
* the program is run through the ``ExeCode`` operator (Sec 2.2.8) so it goes
  through the same sandbox and the same environment as every other method;
* ``max_repairs`` optional retries are allowed on a *hard execution error*, and
  the retry prompt carries **only the error trace** — never the intermediate
  tables.  Showing the model a materialized state would turn CodeGen into an
  interactive agent and destroy the comparison this baseline exists to support.
"""

from __future__ import annotations

import re
import time

from ..agent.actions import extract_block, strip_think
from ..agent.agent import SolveResult, TrajectoryStep
from ..agent.llm import LLMClient
from ..env import Environment, EnvironmentLimits
from ..operators import OperatorCall, parse_operator_call
from ..operators.program import ExeCode
from ..serialize import serialize_task_input
from ..types import ADPTask
from .common import BaselineAgent, LLMCallError, Message

__all__ = ["CodeGenAgent"]

#: ``ExeCode`` is stateless, so one shared instance is enough.
_EXECODE = ExeCode()

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


SYSTEM_PROMPT = """
You are a data preparation assistant. You are given a set of source tables and a
target schema Sigma* describing the table an analyst needs. Write a single
self-contained Python program that transforms the source tables into a table
conforming to the target schema.

# Execution environment
- Each source table is already bound as a pandas DataFrame under its own name.
  Do not load, read or create any file.
- `pd` (pandas) and `np` (numpy) are available, as are `re`, `math`, `datetime`
  and `statistics`. No other imports are permitted.
- The program must end by assigning the final table to a variable named exactly
  `{target}`. Its columns must be the target schema's columns, with the target
  schema's names.
- The target table's contents are NOT given to you; derive them from the source
  data shown below.

# Output format
Reply with the program inside <code>...</code> and nothing else. No explanation,
no markdown outside the tags.
""".strip()


def _extract_code(text: str) -> str:
    """Pull the program out of a generation, tolerating fences instead of tags."""
    cleaned = strip_think(text or "")
    body = extract_block(cleaned, "code")
    if body:
        # A model that wraps the code in both a tag and a fence is still correct.
        fenced = _CODE_FENCE.search(body)
        return (fenced.group(1) if fenced else body).strip()
    fenced = _CODE_FENCE.search(cleaned)
    if fenced:
        return fenced.group(1).strip()
    return cleaned.strip()


class CodeGenAgent(BaselineAgent):
    """One-shot program synthesis from ``phi(S)`` and ``Sigma*``.

    No intermediate state is ever fed back to the model, by construction.
    """

    METHOD = "CodeGen"

    def __init__(
        self,
        llm: LLMClient,
        max_repairs: int = 1,
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
        #: Retries granted on a hard execution error.  These see the error trace
        #: only, so they do not constitute grounding in intermediate results.
        self.max_repairs = max_repairs

    # -- main --------------------------------------------------------------- #
    def solve(self, task: ADPTask) -> SolveResult:
        t_start = time.perf_counter()
        env = Environment(task.sources, limits=self.limits)
        state = env.initial_state()
        result = SolveResult(task_id=task.task_id)

        source_names = list(task.sources.names)
        # Guard against a source table already occupying the answer name.
        target_name = task.sources.unique_name("result_table")

        task_msg = (
            f"{serialize_task_input(task, max_rows=self.max_rows_in_prompt)}\n\n"
            f"Tables bound in the namespace: {', '.join(source_names)}\n"
            f"Assign the final table to `{target_name}`."
        )
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(target=target_name)},
            {"role": "user", "content": task_msg},
        ]

        error: str | None = None
        for attempt in range(self.max_repairs + 1):
            if attempt > 0:
                messages.append({"role": "user", "content": self._repair_prompt(error, target_name)})

            try:
                resp = self._generate(messages, result)
            except LLMCallError as e:
                result.messages = messages
                return self._abort_llm_error(result, e, t_start, attempt)

            messages.append({"role": "assistant", "content": resp.text})
            code = _extract_code(resp.text)
            step = TrajectoryStep(
                turn=attempt,
                observation=messages[-2]["content"],
                response=resp.text,
                usage=resp.usage.to_dict(),
            )

            if not code:
                error = "EmptyProgram: the reply contained no code."
                step.error = error
                step.feedback = error
                result.trajectory.append(step)
                result.n_turns = attempt + 1
                continue

            pipeline = self._build_pipeline(source_names, target_name, code)
            exec_result = env.execute(state, pipeline)
            step.n_ops_applied = exec_result.n_applied

            if exec_result.error is not None:
                # Only the trace is retained; the successful prefix's tables are
                # deliberately not rendered back into the prompt.
                error = exec_result.error
                step.error = error
                step.feedback = error
                result.trajectory.append(step)
                result.n_turns = attempt + 1
                continue

            final = exec_result.final_state or state
            error = self._accept_answer(task, final, exec_result.applied, result)
            step.feedback = error or f"Answer accepted: table '{target_name}'."
            step.error = error
            result.trajectory.append(step)
            result.n_turns = attempt + 1
            if error is None:
                result.stop_reason = "answered"
                result.messages = messages
                result.elapsed_s = time.perf_counter() - t_start
                return result
            # The program ran but produced nothing usable; that is still a hard
            # failure worth one repair, and the message says only what went wrong.
            self._record_fallback(task, [(final, exec_result.applied)], result)

        result.stop_reason = "max_turns"
        result.error = error
        result.messages = messages
        result.elapsed_s = time.perf_counter() - t_start
        return result


    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _build_pipeline(
        source_names: list[str], target_name: str, code: str
    ) -> list[OperatorCall]:
        """Wrap the generated program as ``ExeCode`` followed by ``Terminate``.

        Routing through the operator keeps CodeGen inside the same sandbox and
        the same ``Environment`` as the agentic methods, so an unsafe program or
        a runaway join is caught by the identical limits.
        """
        exe = OperatorCall(
            op=_EXECODE,
            params={"tables": list(source_names), "target": target_name, "func": code},
            raw_params={"func": "```\n" + code + "\n```"},
        )
        return [exe, parse_operator_call(f"Terminate([{target_name}])")]

    @staticmethod
    def _repair_prompt(error: str | None, target_name: str) -> str:
        return (
            "The program failed:\n\n"
            f"{error or 'unknown error'}\n\n"
            "Rewrite the whole program so it runs. You are not shown the "
            "intermediate tables; reason from the source samples and the target "
            f"schema above. Reply with <code>...</code> assigning `{target_name}`."
        )
