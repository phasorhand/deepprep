"""The serialization function ``phi(.)`` from the paper (Sec 5.1).

    "Let phi(.) denote the serialization function. It serializes P_sub by
     concatenating all serialized operators and intermediate table set T by
     serializing it into markdown format."

``phi`` is used in three places and must be *identical* across them, otherwise
training and inference see different token distributions:

* Eq. (4) — operator syntax learning:  ``phi(P_sub) | phi(T_in), phi(T_out)``
* Eq. (5) — reasoning procedure learning: ``phi(r_t) | phi(r_<t), phi(S), Sigma*``
* Inference — the ``<execute>`` observation written back into the tree.

Everything here is deterministic and side-effect free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from .types import ADPTask, Table, TableSchema, TableSet

if TYPE_CHECKING:  # pragma: no cover
    from .operators.base import OperatorCall

__all__ = [
    "serialize_operator",
    "serialize_pipeline",
    "serialize_schema",
    "serialize_table",
    "serialize_table_set",
    "serialize_task_input",
]

# Row budget when materializing a table into the prompt.  The agent reasons over
# *schema + a sample*, never the full tuple set: the paper's environment returns
# "sample outputs or error traces" as feedback (Sec 3).
DEFAULT_MAX_ROWS = 5
DEFAULT_MAX_COLS = 30
DEFAULT_CELL_WIDTH = 60


def _fmt_cell(v: Any, width: int = DEFAULT_CELL_WIDTH) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NaN"
    try:
        if pd.isna(v):
            return "NaN"
    except (TypeError, ValueError):
        pass
    s = str(v).replace("|", "\\|").replace("\n", " ")
    if len(s) > width:
        s = s[: width - 3] + "..."
    return s


def _markdown_table(df: pd.DataFrame, max_rows: int, max_cols: int) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    cols = [str(c) for c in df.columns]
    truncated_cols = len(cols) > max_cols
    shown_cols = cols[:max_cols]

    header = "| " + " | ".join(shown_cols) + (" | ..." if truncated_cols else "") + " |"
    sep = "| " + " | ".join("---" for _ in shown_cols) + (" | ---" if truncated_cols else "") + " |"
    lines = [header, sep]

    body = df.head(max_rows)
    for _, row in body.iterrows():
        cells = [_fmt_cell(row[c]) for c in shown_cols]
        lines.append("| " + " | ".join(cells) + (" | ..." if truncated_cols else "") + " |")

    if len(df) > max_rows:
        lines.append(
            "| " + " | ".join("..." for _ in shown_cols) + (" | ..." if truncated_cols else "") + " |"
        )
    return "\n".join(lines)


def serialize_schema(schema: TableSchema | None, indent: str = "") -> str:
    """Serialize ``Sigma = (tau, C)``: table-level then column-level spec."""
    if schema is None:
        return ""
    parts: list[str] = []
    if schema.description:
        parts.append(f"{indent}Description: {schema.description}")
    if schema.columns:
        parts.append(f"{indent}Columns:")
        for c in schema.columns:
            bits = [f"{indent}  - {c.name}"]
            if c.dtype:
                bits.append(f"({c.dtype})")
            if c.description:
                bits.append(f": {c.description}")
            parts.append(" ".join(bits))
    return "\n".join(parts)


def serialize_table(
    table: Table,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
    with_description: bool = True,
) -> str:
    """Serialize one table ``(Sigma, D)`` to markdown."""
    n_rows, n_cols = table.df.shape
    head = f"### Table: {table.name}  ({n_rows} rows x {n_cols} columns)"
    lines = [head]
    if with_description and table.description:
        lines.append(f"Description: {table.description}")
    # Column descriptions are only emitted when they carry semantic information
    # beyond the name, to keep the context budget under control.
    described = [
        c
        for c in (table.schema.columns if table.schema else [])
        if c.description and str(c.name) in set(table.columns)
    ]
    if with_description and described:
        lines.append("Column semantics:")
        lines.extend(f"  - {c.name}: {c.description}" for c in described)
    if n_cols == 0:
        lines.append("(table has no columns)")
    elif n_rows == 0:
        lines.append("| " + " | ".join(table.columns[:max_cols]) + " |")
        lines.append("| " + " | ".join("---" for _ in table.columns[:max_cols]) + " |")
        lines.append("(table is empty: 0 rows)")
    else:
        lines.append(_markdown_table(table.df, max_rows, max_cols))
    return "\n".join(lines)


def serialize_table_set(
    tables: TableSet,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
    with_description: bool = True,
) -> str:
    """``phi(T)`` — serialize an intermediate table set into markdown."""
    if len(tables) == 0:
        return "(no tables in the current state)"
    return "\n\n".join(
        serialize_table(t, max_rows, max_cols, with_description) for t in tables
    )


def serialize_operator(op: OperatorCall) -> str:
    """Serialize a single operator instance, e.g. ``Deduplicate(movies, [id], first)``."""
    return op.to_source()


def serialize_pipeline(pipeline: list[OperatorCall] | list[str], joiner: str = "\n") -> str:
    """``phi(P)`` — "concatenating all serialized operators"."""
    out = []
    for op in pipeline:
        out.append(op if isinstance(op, str) else op.to_source())
    return joiner.join(out)


def serialize_task_input(task: ADPTask, max_rows: int = DEFAULT_MAX_ROWS) -> str:
    """``phi(S), Sigma*`` — the immutable part of the agent's context.

    Deliberately excludes ``task.target_table``: the ground-truth ``T*`` "is used
    only for evaluation" (Sec 2.1).
    """
    parts = [
        "## Source Tables",
        serialize_table_set(task.sources, max_rows=max_rows),
        "",
        "## Target Schema (Sigma*)",
        serialize_schema(task.target_schema) or "(unspecified)",
    ]
    return "\n".join(parts)
