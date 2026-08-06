"""Sec 2.2.4 — Row Selection operators (3).

    "Row selection operators manipulate tables by filtering, ordering, or
     selecting subsets of rows."
"""

from __future__ import annotations

from typing import Any

from ..types import TableSet
from .base import Operator, OperatorError, ParamSpec
from .util import as_bool, callable_param, require_columns, require_table

__all__ = ["Filter", "Sort", "TopK"]


class Filter(Operator):
    NAME = "Filter"
    CATEGORY = "row_select"
    DOC = "Keep only the rows satisfying a predicate."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("func", kind="func",
                  doc="row -> bool; the row is a pandas Series, e.g. lambda r: r['year'] == 2019"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        fn = callable_param(p["func"], self.NAME, "func")
        df = t.df
        if len(df) == 0:
            return tables.replace(t.with_df(df.copy()))
        try:
            mask = df.apply(fn, axis=1)
        except Exception as e:  # noqa: BLE001
            raise OperatorError(
                f"Filter: the predicate raised {type(e).__name__}: {e}. It receives a row "
                f"Series; index it by column name, e.g. lambda r: r['x'] > 0. "
                f"Available columns: {t.columns}"
            ) from e
        try:
            mask = mask.astype(bool)
        except Exception as e:  # noqa: BLE001
            raise OperatorError(
                f"Filter: the predicate must return a boolean per row, got values of type "
                f"{mask.dtype}. ({e})"
            ) from e
        return tables.replace(t.with_df(df[mask.values].reset_index(drop=True)))


class Sort(Operator):
    NAME = "Sort"
    CATEGORY = "row_select"
    DOC = "Order rows by the specified columns."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("by", kind="columns", doc="columns to sort by, in priority order"),
        ParamSpec("ascending", required=False, default=True,
                  doc="True/False, or a list of booleans matching 'by'"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        by = require_columns(t, p["by"])
        asc_raw = p["ascending"]
        if isinstance(asc_raw, (list, tuple)):
            ascending: Any = [as_bool(a) for a in asc_raw]
            if len(ascending) != len(by):
                raise OperatorError(
                    f"Sort: 'ascending' has {len(ascending)} entries but 'by' has {len(by)} columns."
                )
        else:
            ascending = as_bool(asc_raw, default=True)
        df = t.df.sort_values(by=by, ascending=ascending, kind="mergesort")
        return tables.replace(t.with_df(df.reset_index(drop=True)))


class TopK(Operator):
    NAME = "TopK"
    CATEGORY = "row_select"
    DOC = "Retain the top-k rows of the table (apply Sort first to define the order)."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("k", doc="number of rows to keep"),
        ParamSpec("by", kind="columns", required=False, default=None,
                  doc="optional: sort by these columns before taking the top k"),
        ParamSpec("ascending", required=False, default=False,
                  doc="sort direction when 'by' is given"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        try:
            k = int(p["k"])
        except (TypeError, ValueError) as e:
            raise OperatorError(f"TopK: 'k' must be an integer, got {p['k']!r}") from e
        if k < 0:
            raise OperatorError(f"TopK: 'k' must be non-negative, got {k}")
        df = t.df
        if p["by"]:
            by = require_columns(t, p["by"])
            asc = p["ascending"]
            ascending: Any = (
                [as_bool(a) for a in asc] if isinstance(asc, (list, tuple)) else as_bool(asc, False)
            )
            df = df.sort_values(by=by, ascending=ascending, kind="mergesort")
        return tables.replace(t.with_df(df.head(k).reset_index(drop=True)))
