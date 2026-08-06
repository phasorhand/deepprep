"""Sec 2.2.5 — Aggregation operators (3).

    "Aggregation operators summarize data by grouping records and computing
     statistics, transforming row-level data into compact, analysis-ready
     representations."

``Count`` and ``CalculateStatistic`` are described as returning scalars.  Since
every operator must be a function ``T -> T`` (Sec 2.1), a scalar result is
materialized as a 1x1 table so it remains addressable by downstream operators.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..types import Table, TableSet
from .base import Operator, OperatorError, ParamSpec
from .util import callable_param, require_columns, require_table

__all__ = ["CalculateStatistic", "Count", "GroupBy"]

#: Aggregation function names accepted by :class:`GroupBy`.
_AGG_ALIASES = {
    "mean": "mean", "avg": "mean", "average": "mean",
    "sum": "sum", "total": "sum",
    "count": "count", "size": "size",
    "min": "min", "max": "max",
    "median": "median", "std": "std", "var": "var",
    "first": "first", "last": "last",
    "nunique": "nunique",
    "list": list, "unique": "unique",
}


def _resolve_agg(spec: Any, op: str) -> Any:
    """Map an aggregation spec (string alias, callable, or list) to a pandas agg."""
    if callable(spec):
        return spec
    if isinstance(spec, str):
        fn = _AGG_ALIASES.get(spec.strip().lower())
        if fn is None:
            raise OperatorError(
                f"{op}: unknown aggregation {spec!r}. Supported: {sorted(_AGG_ALIASES)}"
            )
        return fn
    if isinstance(spec, (list, tuple)):
        return [_resolve_agg(s, op) for s in spec]
    raise OperatorError(f"{op}: unsupported aggregation specification {spec!r}")


class GroupBy(Operator):
    NAME = "GroupBy"
    CATEGORY = "aggregation"
    DOC = "Group rows by columns and apply aggregation functions."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("by", kind="columns", doc="grouping key columns"),
        ParamSpec("agg", doc="dict mapping column -> aggregation, e.g. {'rating': 'mean'}"),
        ParamSpec("rename", required=False, default=None,
                  doc="optional dict renaming aggregated columns"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        by = require_columns(t, p["by"])
        agg_spec = p["agg"]
        if not isinstance(agg_spec, dict):
            raise OperatorError(
                f"GroupBy: 'agg' must be a dict {{column: function}}, got {agg_spec!r}. "
                f"Example: {{'rating': 'mean'}}"
            )
        require_columns(t, list(agg_spec))
        resolved = {str(c): _resolve_agg(f, self.NAME) for c, f in agg_spec.items()}

        df = t.df.copy()
        # Numeric aggregations on object columns silently produce garbage in pandas;
        # coerce up-front so the failure (if any) is explicit and actionable.
        for col, fn in resolved.items():
            if fn in ("mean", "median", "sum", "std", "var") and not pd.api.types.is_numeric_dtype(df[col]):
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().sum() == 0 and len(df) > 0:
                    raise OperatorError(
                        f"GroupBy: column '{col}' of table {t.name} is not numeric, so "
                        f"aggregation '{fn}' cannot be applied. Cast it first with "
                        f"CastType({t.name}, {col}, float). Sample values: {df[col].head(3).tolist()}"
                    )
                df[col] = coerced

        try:
            out = df.groupby(by, dropna=False, sort=True, as_index=False).agg(resolved)
        except Exception as e:  # noqa: BLE001
            raise OperatorError(f"GroupBy: {type(e).__name__}: {e}") from e

        out.columns = [
            "_".join(str(x) for x in c if x != "") if isinstance(c, tuple) else str(c)
            for c in out.columns
        ]
        if p["rename"]:
            if not isinstance(p["rename"], dict):
                raise OperatorError("GroupBy: 'rename' must be a dict {old: new}.")
            out = out.rename(columns={str(k): str(v) for k, v in p["rename"].items()})
        return tables.replace(t.with_df(out.reset_index(drop=True)))


class Count(Operator):
    NAME = "Count"
    CATEGORY = "aggregation"
    DOC = (
        "Return the total number of rows of the table as a single-cell table. "
        "A special case of GroupBy without grouping keys."
    )
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("target", kind="table", required=False, default=None,
                  doc="name of the produced table (default: <table>_count)"),
        ParamSpec("column", kind="column", required=False, default="count",
                  doc="name of the produced column"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        name = str(p["target"]) if p["target"] else f"{t.name}_count"
        col = str(p["column"] or "count")
        out = pd.DataFrame({col: [len(t.df)]})
        return tables.replace(Table(name=name, df=out))


class CalculateStatistic(Operator):
    NAME = "CalculateStatistic"
    CATEGORY = "aggregation"
    DOC = (
        "Compute a global statistic over the table with a user-defined function, "
        "returning a single-cell table summarizing a table-level property."
    )
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("stat", doc="name of the statistic; also the produced column name"),
        ParamSpec("func", kind="func",
                  doc="DataFrame -> scalar, e.g. lambda df: df['rating'].mean()"),
        ParamSpec("target", kind="table", required=False, default=None,
                  doc="name of the produced table (default: <table>_<stat>)"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        fn = callable_param(p["func"], self.NAME, "func")
        stat = str(p["stat"])
        try:
            value = fn(t.df)
        except Exception as e:  # noqa: BLE001
            raise OperatorError(
                f"CalculateStatistic: the function raised {type(e).__name__}: {e}. It receives "
                f"the whole DataFrame, e.g. lambda df: df['x'].mean(). "
                f"Available columns: {t.columns}"
            ) from e
        if isinstance(value, (pd.Series, pd.DataFrame)):
            raise OperatorError(
                f"CalculateStatistic: the function must return a scalar, got a "
                f"{type(value).__name__}. Use GroupBy for per-group statistics."
            )
        name = str(p["target"]) if p["target"] else f"{t.name}_{stat}"
        return tables.replace(Table(name=name, df=pd.DataFrame({stat: [value]})))
