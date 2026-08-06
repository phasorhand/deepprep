"""Sec 2.2.7 — Table Reshaping operators (5).

    "Table reshaping operators reorganize the structural layout of tables, such
     as converting between wide and long formats or expanding nested fields."

``Explode`` additionally handles *string-encoded* lists, which is what the
paper's running example requires: the ``genres`` column holds ``"Sci-Fi, Action"``
and the correct target table demands single-valued categories (Figure 2).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..types import TableSet
from .base import Operator, OperatorError, ParamSpec
from .util import as_list, require_columns, require_table

__all__ = ["Explode", "Pivot", "Stack", "Transpose", "WideToLong"]


class Pivot(Operator):
    NAME = "Pivot"
    CATEGORY = "reshaping"
    DOC = (
        "Reshape a table from long to wide format by aggregating values at the "
        "intersection of the index and column identifiers."
    )
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("index", kind="columns", doc="columns that become the row identifier"),
        ParamSpec("columns", kind="columns", doc="column whose values become new column names"),
        ParamSpec("values", kind="columns", doc="column(s) supplying the cell values"),
        ParamSpec("aggfunc", required=False, default="first",
                  doc="aggregation applied to collisions, e.g. 'mean', 'first', 'sum'"),
        ParamSpec("dropna", required=False, default=True,
                  doc="drop index combinations that have no observation; set False to keep "
                      "the full cartesian product of the index levels"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        index = require_columns(t, p["index"])
        cols = require_columns(t, p["columns"])
        values = require_columns(t, p["values"])
        aggfunc = p["aggfunc"] or "first"

        df = t.df.copy()
        if aggfunc in ("mean", "sum", "median", "std", "var"):
            for v in values:
                df[v] = pd.to_numeric(df[v], errors="coerce")
        try:
            # dropna=True is the default: with a MultiIndex, pandas' dropna=False
            # materializes the full cartesian product of the index levels, which
            # invents rows for group combinations that never occur in the data.
            out = pd.pivot_table(
                df, index=index, columns=cols, values=values,
                aggfunc=aggfunc, dropna=bool(p["dropna"]), observed=False,
            )
        except Exception as e:  # noqa: BLE001
            raise OperatorError(
                f"Pivot: {type(e).__name__}: {e}. index={index}, columns={cols}, "
                f"values={values}, aggfunc={aggfunc!r}"
            ) from e

        out = out.reset_index()
        # Flatten the MultiIndex header.  With a single `values` column the value
        # name is redundant, so we drop it and keep only the pivoted labels —
        # this is what produces `2019` / `2020` columns in the paper's example.
        if isinstance(out.columns, pd.MultiIndex):
            flat = []
            for c in out.columns:
                parts = [str(x) for x in c if str(x) != ""]
                if len(values) == 1 and len(parts) > 1 and parts[0] == values[0]:
                    parts = parts[1:]
                flat.append("_".join(parts))
            out.columns = flat
        else:
            out.columns = [str(c) for c in out.columns]
        out.columns.name = None
        return tables.replace(t.with_df(out.reset_index(drop=True)))


class Stack(Operator):
    NAME = "Stack"
    CATEGORY = "reshaping"
    DOC = "Reshape to long format by collapsing multiple columns into key-value pairs."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("id_vars", kind="columns", doc="columns held fixed"),
        ParamSpec("value_vars", kind="columns", required=False, default=None,
                  doc="columns to collapse; omit to use all remaining columns"),
        ParamSpec("var_name", required=False, default="variable",
                  doc="name of the produced key column"),
        ParamSpec("value_name", required=False, default="value",
                  doc="name of the produced value column"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        id_vars = require_columns(t, p["id_vars"]) if p["id_vars"] else []
        value_vars = require_columns(t, p["value_vars"]) if p["value_vars"] else None
        try:
            out = t.df.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=str(p["var_name"] or "variable"),
                value_name=str(p["value_name"] or "value"),
            )
        except Exception as e:  # noqa: BLE001
            raise OperatorError(f"Stack: {type(e).__name__}: {e}") from e
        return tables.replace(t.with_df(out.reset_index(drop=True)))


class WideToLong(Operator):
    NAME = "WideToLong"
    CATEGORY = "reshaping"
    DOC = "Reshape wide-format data into long format by parsing column naming patterns."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("stubnames", kind="columns", doc="common prefixes of the wide columns"),
        ParamSpec("i", kind="columns", doc="column(s) uniquely identifying a row"),
        ParamSpec("j", doc="name of the produced column holding the parsed suffix"),
        ParamSpec("sep", required=False, default="", doc="separator between stub and suffix"),
        ParamSpec("suffix", required=False, default=r"\w+", doc="regex matching the suffix"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        stubs = [str(s) for s in as_list(p["stubnames"])]
        i = require_columns(t, p["i"])
        j = str(p["j"])
        try:
            out = pd.wide_to_long(
                t.df.copy(), stubnames=stubs, i=i, j=j,
                sep=str(p["sep"] or ""), suffix=str(p["suffix"] or r"\w+"),
            ).reset_index()
        except Exception as e:  # noqa: BLE001
            raise OperatorError(
                f"WideToLong: {type(e).__name__}: {e}. stubnames={stubs}, i={i}, j={j!r}. "
                f"Available columns: {t.columns}"
            ) from e
        out.columns = [str(c) for c in out.columns]
        return tables.replace(t.with_df(out.reset_index(drop=True)))


class Transpose(Operator):
    NAME = "Transpose"
    CATEGORY = "reshaping"
    DOC = "Swap the rows and columns of the table."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("header", kind="column", required=False, default=None,
                  doc="use this column's values as the new column names"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        df = t.df.copy()
        if p["header"]:
            (h,) = require_columns(t, p["header"])
            df = df.set_index(h)
            out = df.T.reset_index().rename(columns={"index": "column"})
        else:
            out = df.T.reset_index().rename(columns={"index": "column"})
        out.columns = [str(c) for c in out.columns]
        return tables.replace(t.with_df(out.reset_index(drop=True)))


class Explode(Operator):
    NAME = "Explode"
    CATEGORY = "reshaping"
    DOC = (
        "Expand a column containing list-valued entries into multiple rows. "
        "String-encoded lists (e.g. 'Sci-Fi, Action') are split on 'sep' first."
    )
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("column", kind="column"),
        ParamSpec("sep", required=False, default=",",
                  doc="separator used to split string-encoded lists; None disables splitting"),
        ParamSpec("strip", required=False, default=True,
                  doc="strip whitespace around the produced values"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        (col,) = require_columns(t, p["column"])
        df = t.df.copy()
        sep = p["sep"]
        strip = bool(p["strip"])

        def to_list(v: Any) -> Any:
            if isinstance(v, (list, tuple, set)):
                items = list(v)
            elif isinstance(v, str) and sep not in (None, "None", ""):
                items = v.split(str(sep))
            else:
                return v
            if strip:
                items = [x.strip() if isinstance(x, str) else x for x in items]
            return items

        df[col] = df[col].map(to_list)
        try:
            out = df.explode(col, ignore_index=True)
        except Exception as e:  # noqa: BLE001
            raise OperatorError(f"Explode: {type(e).__name__}: {e}") from e
        return tables.replace(t.with_df(out.reset_index(drop=True)))
