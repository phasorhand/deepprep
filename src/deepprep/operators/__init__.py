"""The operator space ``O`` of DeepPrep (paper Sec 2.2).

    "In total, we consider 31 operators, organized into eight categories that
     cover the major transformation patterns in data preparation pipelines,
     including data cleaning, normalization, structural reshaping, table
     combination, and aggregation."

The registry is the single source of truth: the prompt's operator manual, the
parser, and the training-data generator all read from it, so adding an operator
("DeepPrep ... is designed to be extensible") requires only registering it here.
"""

from __future__ import annotations

from .aggregation import CalculateStatistic, Count, GroupBy
from .base import (
    Operator,
    OperatorCall,
    OperatorError,
    ParamSpec,
    ParseError,
    parse_operator_call,
    parse_pipeline,
    split_calls,
)
from .cleaning import (
    Deduplicate,
    DropNA,
    ErrorDetection,
    MissingValueImputation,
    OutlierDetection,
)
from .combination import Append, Join, Union
from .normalization import CastType, StandardizeDatetime, ValueTransform
from .program import ExeCode, Terminate, answer_table_name
from .reshaping import Explode, Pivot, Stack, Transpose, WideToLong
from .row_select import Filter, Sort, TopK
from .schema_edit import (
    AddNewColumn,
    Concatenate,
    DropColumn,
    RenameColumn,
    SelectColumn,
    SplitColumn,
    Subtitle,
)

__all__ = [
    "CATEGORY_ORDER",
    "OPERATORS",
    "Operator",
    "OperatorCall",
    "OperatorError",
    "ParamSpec",
    "ParseError",
    "answer_table_name",
    "get_operator",
    "operator_manual",
    "operator_names",
    "operators_by_category",
    "parse_operator_call",
    "parse_pipeline",
    "split_calls",
]

#: Human-readable names for the eight categories of Sec 2.2, in paper order.
CATEGORY_ORDER: dict[str, str] = {
    "cleaning": "2.2.1 Data Cleaning",
    "normalization": "2.2.2 Value Normalization",
    "schema_edit": "2.2.3 Schema Editing",
    "row_select": "2.2.4 Row Selection",
    "aggregation": "2.2.5 Aggregation",
    "combination": "2.2.6 Table Combination",
    "reshaping": "2.2.7 Table Reshaping",
    "program": "2.2.8 Program Synthesis",
    "control": "Control",
}

_OPERATOR_CLASSES: tuple[type[Operator], ...] = (
    # 2.2.1 Data Cleaning (5)
    DropNA, MissingValueImputation, Deduplicate, ErrorDetection, OutlierDetection,
    # 2.2.2 Value Normalization (3)
    ValueTransform, StandardizeDatetime, CastType,
    # 2.2.3 Schema Editing (7)
    RenameColumn, AddNewColumn, DropColumn, SplitColumn, Concatenate, SelectColumn, Subtitle,
    # 2.2.4 Row Selection (3)
    Filter, Sort, TopK,
    # 2.2.5 Aggregation (3)
    GroupBy, Count, CalculateStatistic,
    # 2.2.6 Table Combination (3)
    Join, Union, Append,
    # 2.2.7 Table Reshaping (5)
    Pivot, Stack, WideToLong, Transpose, Explode,
    # 2.2.8 Program Synthesis (1)
    ExeCode,
    # Control (1) -- Figure 4 shows `Terminate([result_table])` closing the pipeline.
    Terminate,
)

#: name -> singleton instance.  Operators are stateless, so one instance suffices.
OPERATORS: dict[str, Operator] = {cls.NAME: cls() for cls in _OPERATOR_CLASSES}

assert len(OPERATORS) == len(_OPERATOR_CLASSES) == 31, (
    f"expected the paper's 31 operators, registry has {len(OPERATORS)}"
)


def get_operator(name: str) -> Operator | None:
    """Look up an operator type by name (case-sensitive, then case-insensitive)."""
    op = OPERATORS.get(name)
    if op is not None:
        return op
    lowered = {k.lower(): v for k, v in OPERATORS.items()}
    return lowered.get(name.lower())


def operator_names() -> list[str]:
    return list(OPERATORS)


def operators_by_category() -> dict[str, list[Operator]]:
    out: dict[str, list[Operator]] = {c: [] for c in CATEGORY_ORDER}
    for op in OPERATORS.values():
        out.setdefault(type(op).CATEGORY, []).append(op)
    return {k: v for k, v in out.items() if v}


def operator_manual(categories: list[str] | None = None) -> str:
    """Render the operator manual injected into the agent's system prompt."""
    lines: list[str] = []
    for cat, ops in operators_by_category().items():
        if categories and cat not in categories:
            continue
        lines.append(f"\n### {CATEGORY_ORDER.get(cat, cat)}")
        lines.extend(type(op).manual() for op in ops)
    return "\n".join(lines).strip()
