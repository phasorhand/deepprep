"""Sec 2.2.6 — Table Combination operators (3).

    "Table combination operators integrate data from multiple tables into a
     unified representation, enabling multi-source data preparation pipelines."

Output naming follows the paper's Example 2: "applying Join merges the movies and
directors tables ... yielding a table named ``movies_directors_join``".  All three
operators additionally accept an optional ``target`` name so a pipeline can pick
an explicit output table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..types import Table, TableSet
from .base import Operator, OperatorError, ParamSpec
from .util import as_list, require_columns, require_table

__all__ = ["Append", "Join", "Union"]

_HOW_JOIN = ("inner", "left", "right", "outer", "cross")


def _split_on(on: Any) -> tuple[list[str], list[str]]:
    """Interpret the ``on`` parameter as (left keys, right keys).

    Accepts ``id`` / ``[a, b]`` for shared key names, and ``(dir_id, id)`` or
    ``{'left': ..., 'right': ...}`` for differently-named keys.
    """
    if isinstance(on, dict):
        left = [str(c) for c in as_list(on.get("left") or on.get("left_on"))]
        right = [str(c) for c in as_list(on.get("right") or on.get("right_on"))]
        if not left or not right:
            raise OperatorError(
                "Join: dict form of 'on' must provide both 'left' and 'right' keys."
            )
        return left, right
    items = as_list(on)
    if len(items) == 2 and all(isinstance(i, (str, int)) for i in items):
        # Ambiguous: `[a, b]` could be two shared keys or a (left, right) pair.
        # Resolution happens in Join.apply() where both schemas are known.
        return [str(items[0])], [str(items[1])]
    cols = [str(c) for c in items]
    return cols, cols


class Join(Operator):
    NAME = "Join"
    CATEGORY = "combination"
    DOC = "Combine two tables on key columns using the specified join type."
    PARAMS = (
        ParamSpec("left", kind="table"),
        ParamSpec("right", kind="table"),
        ParamSpec("on", doc="shared key name(s), or (left_key, right_key) when they differ"),
        ParamSpec("how", required=False, default="inner", choices=_HOW_JOIN),
        ParamSpec("target", kind="table", required=False, default=None,
                  doc="name of the produced table (default: <left>_<right>_join)"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        lt = require_table(tables, p["left"])
        rt = require_table(tables, p["right"])
        how = p["how"] or "inner"
        name = str(p["target"]) if p["target"] else f"{lt.name}_{rt.name}_join"

        if how == "cross":
            out = lt.df.merge(rt.df, how="cross", suffixes=("", "_right"))
            return tables.replace(Table(name=name, df=out.reset_index(drop=True)))

        left_keys, right_keys = _split_on(p["on"])
        items = as_list(p["on"])
        # Disambiguate `[a, b]`: prefer "two shared keys" when both exist on both sides.
        if len(items) == 2 and all(isinstance(i, (str, int)) for i in items):
            both = [str(i) for i in items]
            if all(c in lt.columns for c in both) and all(c in rt.columns for c in both):
                left_keys = right_keys = both

        require_columns(lt, left_keys)
        require_columns(rt, right_keys)
        if len(left_keys) != len(right_keys):
            raise OperatorError(
                f"Join: got {len(left_keys)} left key(s) but {len(right_keys)} right key(s); "
                f"they must pair up one-to-one."
            )

        ldf, rdf = lt.df.copy(), rt.df.copy()
        # Merging across mismatched dtypes (a very common cause of silently empty
        # joins) is normalised to string when the raw dtypes are incompatible.
        for lk, rk in zip(left_keys, right_keys, strict=True):
            if ldf[lk].dtype != rdf[rk].dtype:
                ldf[lk] = ldf[lk].astype(str)
                rdf[rk] = rdf[rk].astype(str)

        try:
            out = ldf.merge(
                rdf, left_on=left_keys, right_on=right_keys, how=how, suffixes=("", "_right")
            )
        except Exception as e:  # noqa: BLE001
            raise OperatorError(f"Join: {type(e).__name__}: {e}") from e

        if len(out) == 0 and len(ldf) > 0 and len(rdf) > 0 and how == "inner":
            raise OperatorError(
                f"EmptyResult: the inner join of {lt.name}.{left_keys} with "
                f"{rt.name}.{right_keys} produced 0 rows, so no key values match. "
                f"Sample left keys: {ldf[left_keys[0]].head(3).tolist()}; "
                f"sample right keys: {rdf[right_keys[0]].head(3).tolist()}. "
                f"Normalise the key values first (e.g. ValueTransform to strip/lowercase) "
                f"or use how='left'."
            )
        return tables.replace(Table(name=name, df=out.reset_index(drop=True)))


class Union(Operator):
    NAME = "Union"
    CATEGORY = "combination"
    DOC = "Vertically combine multiple tables with compatible schemas."
    PARAMS = (
        ParamSpec("tables", kind="tables", doc="list of table names to combine"),
        ParamSpec("how", required=False, default="distinct", choices=("distinct", "all"),
                  doc="'distinct' removes duplicate rows (SQL UNION), 'all' retains them"),
        ParamSpec("target", kind="table", required=False, default=None,
                  doc="name of the produced table (default: <t1>_<t2>_union)"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        names = [str(n) for n in as_list(p["tables"])]
        if len(names) < 2:
            raise OperatorError(
                f"Union: needs at least two tables, got {names}. "
                f"Available tables: {tables.names}"
            )
        parts = [require_table(tables, n) for n in names]
        out = pd.concat([t.df for t in parts], ignore_index=True, sort=False)
        if (p["how"] or "distinct") == "distinct":
            out = out.drop_duplicates()
        name = str(p["target"]) if p["target"] else "_".join(names[:2]) + "_union"
        return tables.replace(Table(name=name, df=out.reset_index(drop=True)))


class Append(Operator):
    NAME = "Append"
    CATEGORY = "combination"
    DOC = (
        "Append one table to another by vertically concatenating their rows, "
        "without the duplicate elimination performed by Union."
    )
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("other", kind="table"),
        ParamSpec("target", kind="table", required=False, default=None,
                  doc="name of the produced table (default: overwrite <table>)"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        a = require_table(tables, p["table"])
        b = require_table(tables, p["other"])
        out = pd.concat([a.df, b.df], ignore_index=True, sort=False)
        name = str(p["target"]) if p["target"] else a.name
        return tables.replace(Table(name=name, df=out.reset_index(drop=True), schema=None))
