"""Prompting baselines (paper Sec 6.1, "Prompting Baselines").

    "**CodeGen** [28, 32] prompts an LLM to generate data preparation code from
     the source tables and target schema. **Plan-and-Solve (PaS)** [37] prompts
     an LLM using a two-stage strategy that first produces a high-level plan and
     then generates a complete operator sequence to obtain the target table.
     **ReAct** [2] is a linear reasoning agent that iteratively predicts the next
     operator based on previous execution results. **MCTS-OP** applies Monte
     Carlo Tree Search, using local node expansion and scalar rewards to guide
     search."

Every class here exposes the same interface as
:class:`~deepprep.agent.agent.DeepPrepAgent` — an ``llm`` attribute and
``solve(task) -> SolveResult`` — so :func:`deepprep.eval.evaluate` produces
Table 2 for all five methods without a special case::

    >>> from deepprep import ADPTask                        # doctest: +SKIP
    >>> from deepprep.baselines import ReActAgent           # doctest: +SKIP
    >>> from deepprep.eval import evaluate                  # doctest: +SKIP
    >>> evaluate(ReActAgent(llm), tasks, method="ReAct")    # doctest: +SKIP

The baselines share the operator space, the execution environment and the
serialization function ``phi(.)`` with DeepPrep; what differs is only the
reasoning structure, which is what Sec 6.2's comparison is meant to isolate.
"""

from .codegen import CodeGenAgent
from .common import BaselineAgent, LLMCallError, schema_alignment
from .mcts_op import MCTSNode, MCTSOperatorAgent
from .plan_and_solve import PlanAndSolveAgent
from .react import ReActAgent

__all__ = [
    "BaselineAgent",
    "CodeGenAgent",
    "LLMCallError",
    "MCTSNode",
    "MCTSOperatorAgent",
    "PlanAndSolveAgent",
    "ReActAgent",
    "schema_alignment",
]
