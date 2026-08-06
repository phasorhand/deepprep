"""Sec 2.2.3 — Schema Editing operators (7).

    "Schema editing operators modify table structure and column semantics,
     enabling column selection, renaming, derivation, and restructuring to align
     raw data with the target schema."
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..types import TableSet
from .base import Operator, OperatorError, ParamSpec
from .util import as_list, callable_param, require_columns, require_table

__all__ = [
    "AddNewColumn",
    "Concatenate",
    "DropColumn",
    "RenameColumn",
    "SelectColumn",
    "SplitColumn",
    "Subtitle",
]


def _is_missing(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


class RenameColumn(Operator):
    NAME = "RenameColumn"
    CATEGORY = "schema_edit"
    DOC = "Rename columns according to a mapping from original to target names."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("rename_map", doc="dict mapping old name -> new name, e.g. {'name': 'director_name'}"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        mapping = p["rename_map"]
        if not isinstance(mapping, dict):
            raise OperatorError(
                f"RenameColumn: rename_map must be a dict {{old: new}}, got {mapping!r}"
            )
        mapping = {str(k): str(v) for k, v in mapping.items()}
        require_columns(t, list(mapping))
        df = t.df.rename(columns=mapping)
        new = t.with_df(df)
        # Carry column descriptions across the rename.
        if t.schema:
            old_desc = {c.name: c.description for c in t.schema.columns}
            for c in new.schema.columns:
                src = next((o for o, n in mapping.items() if n == c.name), c.name)
                if c.description is None:
                    c.description = old_desc.get(src)
        return tables.replace(new)


class AddNewColumn(Operator):
    NAME = "AddNewColumn"
    CATEGORY = "schema_edit"
    DOC = "Add a new column computed from existing data using a row-wise function."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("name", kind="column", doc="name of the column to create"),
        ParamSpec("func", kind="func",
                  doc="row -> value; the row is a pandas Series, e.g. lambda r: r['a'] + r['b']"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        fn = callable_param(p["func"], self.NAME, "func")
        df = t.df.copy()
        col = str(p["name"])
        if len(df) == 0:
            df[col] = pd.Series(dtype="object")
            return tables.replace(t.with_df(df))
        try:
            df[col] = df.apply(fn, axis=1)
        except Exception as e:  # noqa: BLE001
            raise OperatorError(
                f"AddNewColumn: the row-wise function raised {type(e).__name__}: {e}. "
                f"It receives a row Series; index it by column name, e.g. lambda r: r['x']. "
                f"Available columns: {t.columns}"
            ) from e
        return tables.replace(t.with_df(df))


class DropColumn(Operator):
    NAME = "DropColumn"
    CATEGORY = "schema_edit"
    DOC = "Remove the specified columns."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("columns", kind="columns"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        cols = require_columns(t, p["columns"])
        return tables.replace(t.with_df(t.df.drop(columns=cols)))


class SplitColumn(Operator):
    NAME = "SplitColumn"
    CATEGORY = "schema_edit"
    DOC = "Split one column into multiple columns using a user-defined splitting function."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("source", kind="column", doc="column to split"),
        ParamSpec("target", kind="columns", doc="names of the produced columns"),
        ParamSpec("func", kind="func", required=False, default=None,
                  doc="value -> sequence of parts; omit to split on 'sep'"),
        ParamSpec("sep", required=False, default=None,
                  doc="separator used when 'func' is omitted"),
        ParamSpec("drop", required=False, default=True,
                  doc="drop the source column afterwards"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        (src,) = require_columns(t, p["source"])
        targets = [str(c) for c in as_list(p["target"])]
        if not targets:
            raise OperatorError("SplitColumn: 'target' must name at least one output column.")

        if p["func"] is not None:
            fn = callable_param(p["func"], self.NAME, "func")
        elif p["sep"] is not None:
            sep = str(p["sep"])
            fn = lambda v: str(v).split(sep)  # noqa: E731
        else:
            raise OperatorError(
                "SplitColumn: provide either 'func' (value -> list of parts) or 'sep'."
            )

        df = t.df.copy()
        rows: list[list[Any]] = []
        for v in df[src]:
            try:
                parts = fn(v)
            except Exception as e:  # noqa: BLE001
                raise OperatorError(
                    f"SplitColumn: the split function raised {type(e).__name__}: {e} on value "
                    f"{v!r}. Make sure it handles every value in '{src}' (including NaN)."
                ) from e
            parts = list(parts) if isinstance(parts, (list, tuple)) else [parts]
            parts = parts[: len(targets)] + [None] * max(0, len(targets) - len(parts))
            rows.append(parts)

        for i, name in enumerate(targets):
            df[name] = [r[i] for r in rows] if rows else pd.Series(dtype="object")
        if p["drop"] and src not in targets:
            df = df.drop(columns=[src])
        return tables.replace(t.with_df(df))


class Concatenate(Operator):
    NAME = "Concatenate"
    CATEGORY = "schema_edit"
    DOC = "Combine multiple columns into a single column using a formatting function."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("columns", kind="columns", doc="columns to combine, in order"),
        ParamSpec("target", kind="column", doc="name of the produced column"),
        ParamSpec("func", kind="func", required=False, default=None,
                  doc="(*values) -> combined value; omit to join with 'sep'"),
        ParamSpec("sep", required=False, default=" ",
                  doc="separator used when 'func' is omitted"),
        ParamSpec("drop", required=False, default=False,
                  doc="drop the source columns afterwards"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        cols = require_columns(t, p["columns"])
        target = str(p["target"])
        df = t.df.copy()

        if p["func"] is not None:
            fn = callable_param(p["func"], self.NAME, "func")
            values = []
            for _, row in df.iterrows():
                vals = [row[c] for c in cols]
                try:
                    # Accept both `lambda a, b: ...` and `lambda row: ...` styles.
                    values.append(fn(*vals))
                except TypeError:
                    values.append(fn(row))
                except Exception as e:  # noqa: BLE001
                    raise OperatorError(
                        f"Concatenate: the formatting function raised {type(e).__name__}: {e}"
                    ) from e
            df[target] = values if len(df) else pd.Series(dtype="object")
        else:
            sep = str(p["sep"]) if p["sep"] is not None else " "
            # A plain `.astype(str)` leaves NaN as a float under pandas' StringDtype,
            # which then breaks str.join. Render missing values as empty, matching
            # SQL's CONCAT_WS, so a null in one column does not destroy the row.
            rendered = [
                df[c].map(lambda v: "" if _is_missing(v) else str(v)) for c in cols
            ]
            df[target] = (
                pd.concat(rendered, axis=1).agg(sep.join, axis=1)
                if len(df)
                else pd.Series(dtype="object")
            )

        if p["drop"]:
            df = df.drop(columns=[c for c in cols if c != target])
        return tables.replace(t.with_df(df))


class SelectColumn(Operator):
    NAME = "SelectColumn"
    CATEGORY = "schema_edit"
    DOC = "Retain only the specified columns, in the given order."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("columns", kind="columns"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        cols = require_columns(t, p["columns"])
        return tables.replace(t.with_df(t.df[cols].copy()))


class Subtitle(Operator):
    NAME = "Subtitle"
    CATEGORY = "schema_edit"
    DOC = "Add a new column populated with a constant title/subtitle value."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("title", doc="the constant value to broadcast"),
        ParamSpec("target_col", kind="column", doc="name of the produced column"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        df = t.df.copy()
        df[str(p["target_col"])] = p["title"]
        return tables.replace(t.with_df(df))
