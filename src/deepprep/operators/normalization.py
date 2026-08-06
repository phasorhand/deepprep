"""Sec 2.2.2 — Value Normalization operators (3).

    "Value normalization operators standardize field values and data types to
     ensure semantic consistency."
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..types import TableSet
from .base import Operator, OperatorError, ParamSpec
from .util import apply_elementwise, callable_param, require_columns, require_table

__all__ = ["CastType", "StandardizeDatetime", "ValueTransform"]

#: Aliases accepted for the ``dtype`` parameter of :class:`CastType`.
_DTYPE_ALIASES = {
    "int": "int64", "integer": "int64", "int64": "int64", "int32": "int32",
    "float": "float64", "double": "float64", "float64": "float64", "float32": "float32",
    "str": "string", "string": "string", "text": "string", "object": "object",
    "bool": "boolean", "boolean": "boolean",
    "datetime": "datetime64[ns]", "date": "datetime64[ns]", "datetime64[ns]": "datetime64[ns]",
    "category": "category",
}


class ValueTransform(Operator):
    NAME = "ValueTransform"
    CATEGORY = "normalization"
    DOC = "Apply a user-defined transformation to every value of a column to standardize its format."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("column", kind="column"),
        ParamSpec("func", kind="func", doc="value -> new value, e.g. lambda x: str(x).strip().lower()"),
        ParamSpec("target", kind="column", required=False, default=None,
                  doc="write into this column instead of overwriting 'column'"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        (col,) = require_columns(t, p["column"])
        fn = callable_param(p["func"], self.NAME, "func")
        df = t.df.copy()
        df[str(p["target"]) if p["target"] else col] = apply_elementwise(df[col], fn, self.NAME)
        return tables.replace(t.with_df(df))


class StandardizeDatetime(Operator):
    NAME = "StandardizeDatetime"
    CATEGORY = "normalization"
    DOC = "Parse a column into datetimes and render it in a target format."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("column", kind="column"),
        ParamSpec("format", required=False, default="%Y-%m-%d",
                  doc="strftime output format; use None to keep native datetime objects"),
        ParamSpec("input_format", required=False, default=None,
                  doc="strptime format of the raw values; omit to infer per value"),
        ParamSpec("errors", required=False, default="coerce", choices=("coerce", "raise"),
                  doc="'coerce' turns unparseable values into NaN"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        (col,) = require_columns(t, p["column"])
        df = t.df.copy()
        errors = p["errors"] or "coerce"
        try:
            parsed = pd.to_datetime(
                df[col], format=p["input_format"], errors=errors, utc=False
            )
        except Exception as e:  # noqa: BLE001
            raise OperatorError(
                f"StandardizeDatetime: could not parse column '{col}' of table {t.name}: {e}. "
                f"Pass input_format=... or normalise the values with ValueTransform first."
            ) from e
        if errors == "coerce" and parsed.notna().sum() == 0 and len(df) > 0:
            raise OperatorError(
                f"StandardizeDatetime: no value in column '{col}' of table {t.name} could be "
                f"parsed as a date. Sample values: {df[col].head(3).tolist()}"
            )
        fmt = p["format"]
        df[col] = parsed if fmt in (None, "None") else parsed.dt.strftime(fmt)
        return tables.replace(t.with_df(df))


class CastType(Operator):
    NAME = "CastType"
    CATEGORY = "normalization"
    DOC = "Cast the values of a column to a target data type."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("column", kind="column"),
        ParamSpec("dtype", doc=f"one of {sorted(set(_DTYPE_ALIASES))}"),
        ParamSpec("errors", required=False, default="coerce", choices=("coerce", "raise"),
                  doc="'coerce' turns uncastable values into NaN"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        (col,) = require_columns(t, p["column"])
        raw = str(p["dtype"]).lower()
        dtype = _DTYPE_ALIASES.get(raw)
        if dtype is None:
            raise OperatorError(
                f"CastType: unsupported dtype {p['dtype']!r}. Supported: {sorted(set(_DTYPE_ALIASES))}"
            )
        df = t.df.copy()
        s = df[col]
        errors = p["errors"] or "coerce"
        try:
            if dtype.startswith("datetime"):
                df[col] = pd.to_datetime(s, errors=errors)
            elif dtype.startswith(("int", "float")):
                num = pd.to_numeric(s, errors=errors)
                if dtype.startswith("int"):
                    # Nullable integer keeps NaN representable after coercion.
                    df[col] = num.round().astype("Int64") if num.isna().any() else num.astype(dtype)
                else:
                    df[col] = num.astype(dtype)
            elif dtype == "boolean":
                df[col] = s.map(_to_bool).astype("boolean")
            else:
                df[col] = s.astype(dtype)
        except Exception as e:  # noqa: BLE001
            raise OperatorError(
                f"CastType: cannot cast column '{col}' of table {t.name} to {p['dtype']!r}: "
                f"{type(e).__name__}: {e}. Sample values: {s.head(3).tolist()}"
            ) from e
        return tables.replace(t.with_df(df))


def _to_bool(v: Any) -> Any:
    if pd.isna(v):
        return pd.NA
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    return pd.NA
