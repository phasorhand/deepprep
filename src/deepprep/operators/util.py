"""Shared helpers for operator implementations.

Every validation failure raises :class:`OperatorError` with a message naming the
offending table/column *and* listing the valid alternatives.  This matters: the
message is the ``<execute>`` feedback the agent uses to backtrack (paper Sec 4.2
shows exactly this, e.g. "MissingColumn: No column named 'year' in the table
ratings"), so vague errors directly cost accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..types import Table, TableSet
from .base import OperatorError

__all__ = [
    "as_bool",
    "as_list",
    "callable_param",
    "require_columns",
    "require_table",
]


def require_table(tables: TableSet, name: Any) -> Table:
    """Fetch a table by name, or raise a ``MissingTable`` feedback error."""
    if not isinstance(name, str):
        raise OperatorError(
            f"MissingTable: table reference must be a name, got {name!r}"
        )
    t = tables.get(name)
    if t is None:
        raise OperatorError(
            f"MissingTable: No table named '{name}'. Available tables: {tables.names}"
        )
    return t


def require_columns(table: Table, columns: Any) -> list[str]:
    """Validate that ``columns`` exist in ``table``; return them as a list of str."""
    cols = as_list(columns)
    out: list[str] = []
    have = set(table.columns)
    for c in cols:
        c = str(c)
        if c not in have:
            raise OperatorError(
                f"MissingColumn: No column named '{c}' in the table {table.name}. "
                f"Available columns: {table.columns}"
            )
        out.append(c)
    return out


def as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    if isinstance(v, pd.Index):
        return list(v)
    return [v]


def as_bool(v: Any, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "asc", "ascending")
    return bool(v)


def callable_param(func: Any, op: str, param: str) -> Callable[..., Any]:
    """Coerce a function parameter into a callable, or raise actionable feedback."""
    if callable(func):
        return func
    if isinstance(func, str):
        # Allow a bare string that is itself source, e.g. func="lambda x: x.strip()".
        from .sandbox import SandboxError, compile_callable

        try:
            fn = compile_callable(func)
        except SandboxError as e:
            raise OperatorError(f"{op}: invalid {param}: {e}") from e
        if callable(fn):
            return fn
    raise OperatorError(
        f"{op}: parameter '{param}' must be a function (e.g. lambda x: ...), got {func!r}"
    )


def apply_elementwise(series: pd.Series, fn: Callable[[Any], Any], op: str) -> pd.Series:
    """Apply ``fn`` to each value, converting a raised error into agent feedback."""
    try:
        return series.map(fn)
    except Exception as e:  # noqa: BLE001
        raise OperatorError(
            f"{op}: the provided function raised {type(e).__name__}: {e}. "
            f"Make sure it handles every value in the column (including NaN)."
        ) from e
