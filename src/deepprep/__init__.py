"""DeepPrep — an LLM-powered agentic system for autonomous data preparation.

Reimplementation of Fan et al., "DeepPrep: An LLM-Powered Agentic System for
Autonomous Data Preparation" (arXiv:2602.07371, PVLDB Vol. 19).

    >>> from deepprep import ADPTask, DeepPrepAgent
    >>> from deepprep.agent.llm import OpenAIClient
    >>> task = ADPTask.load("examples/movies_demo/task.json")
    >>> result = DeepPrepAgent(llm=OpenAIClient(model="gpt-4o-mini")).solve(task)
    >>> result.pipeline_source                                    # doctest: +SKIP
"""

from .agent import DeepPrepAgent, SolveResult, TrajectoryStep
from .env import Environment, EnvironmentLimits, ExecutionResult
from .operators import OPERATORS, OperatorCall, operator_manual, parse_pipeline
from .tree import Node, ReasoningTree
from .types import ADPTask, ColumnSpec, Table, TableSchema, TableSet, load_tasks

__version__ = "0.1.0"

__all__ = [
    "ADPTask",
    "ColumnSpec",
    "DeepPrepAgent",
    "Environment",
    "EnvironmentLimits",
    "ExecutionResult",
    "Node",
    "OPERATORS",
    "OperatorCall",
    "ReasoningTree",
    "SolveResult",
    "Table",
    "TableSchema",
    "TableSet",
    "TrajectoryStep",
    "__version__",
    "load_tasks",
    "operator_manual",
    "parse_pipeline",
]
