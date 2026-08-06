"""Sec 2.2.1 — Data Cleaning operators (5).

    "Data cleaning operators improve data quality by detecting, fixing, or
     removing erroneous, missing, or duplicate records."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..types import TableSet
from .base import Operator, OperatorError, ParamSpec
from .util import apply_elementwise, callable_param, require_columns, require_table

__all__ = [
    "Deduplicate",
    "DropNA",
    "ErrorDetection",
    "MissingValueImputation",
    "OutlierDetection",
]


def _fill_missing(df: pd.DataFrame, col: str, missing: pd.Series, value: Any) -> None:
    """Write ``value`` into the missing cells of ``col``, in place.

    Under pandas' StringDtype an assignment of a number into a text column raises
    rather than upcasting, so a mixed column is widened to ``object`` first.  The
    alternative -- coercing the column -- would destroy the non-missing values
    this operator is not allowed to touch.
    """
    try:
        df.loc[missing, col] = value
    except (TypeError, ValueError):
        df[col] = df[col].astype(object)
        df.loc[missing, col] = value


class DropNA(Operator):
    NAME = "DropNA"
    CATEGORY = "cleaning"
    DOC = "Remove rows with missing values in the specified columns."
    PARAMS = (
        ParamSpec("table", kind="table", doc="table to clean"),
        ParamSpec("subset", kind="columns", required=False, default=None,
                  doc="columns to check; omit to check all columns"),
        ParamSpec("how", required=False, default="any", choices=("any", "all"),
                  doc="'any' drops a row if any checked value is missing; "
                      "'all' only if every checked value is missing"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        subset = require_columns(t, p["subset"]) if p["subset"] else None
        df = t.df.dropna(subset=subset, how=p["how"] or "any")
        return tables.replace(t.with_df(df.reset_index(drop=True)))


class MissingValueImputation(Operator):
    NAME = "MissingValueImputation"
    CATEGORY = "cleaning"
    DOC = "Impute missing values in a column using a statistical strategy."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("column", kind="column", doc="column to impute"),
        ParamSpec("mode", required=False, default="mean",
                  choices=("mean", "median", "mode", "ffill", "bfill", "zero", "constant"),
                  doc="imputation strategy"),
        ParamSpec("value", required=False, default=None,
                  doc="fill value; required when mode='constant'"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        (col,) = require_columns(t, p["column"])
        df = t.df.copy()
        s = df[col]
        mode = (p["mode"] or "mean").lower()

        # Sec 2.2.1: this operator "imputes MISSING values". Only cells that are
        # actually missing may be written; a value that merely fails to parse as a
        # number is data, not a hole, and overwriting it would silently destroy it.
        missing = s.isna()

        if mode in ("mean", "median"):
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().sum() == 0:
                raise OperatorError(
                    f"MissingValueImputation: column '{col}' of table {t.name} has no numeric "
                    f"values, so mode='{mode}' cannot be applied. Use mode='mode' for "
                    f"categorical columns."
                )
            fill = numeric.mean() if mode == "mean" else numeric.median()
            _fill_missing(df, col, missing, fill)
        elif mode == "mode":
            m = s.mode(dropna=True)
            if len(m) == 0:
                raise OperatorError(
                    f"MissingValueImputation: column '{col}' of table {t.name} is entirely "
                    f"missing, so there is no mode to impute with."
                )
            df[col] = s.fillna(m.iloc[0])
        elif mode == "ffill":
            df[col] = s.ffill()
        elif mode == "bfill":
            df[col] = s.bfill()
        elif mode == "zero":
            _fill_missing(df, col, missing, 0)
        else:  # constant
            if p["value"] is None:
                raise OperatorError(
                    "MissingValueImputation: mode='constant' requires the 'value' parameter."
                )
            df[col] = s.fillna(p["value"])
        return tables.replace(t.with_df(df))


class Deduplicate(Operator):
    NAME = "Deduplicate"
    CATEGORY = "cleaning"
    DOC = "Remove duplicate rows based on the specified columns."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("subset", kind="columns", required=False, default=None,
                  doc="columns defining duplication; omit to use all columns"),
        ParamSpec("keep", required=False, default="first", choices=("first", "last"),
                  doc="which occurrence of a duplicate group to retain"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        subset = require_columns(t, p["subset"]) if p["subset"] else None
        keep = p["keep"] or "first"
        df = t.df.drop_duplicates(subset=subset, keep=keep)
        return tables.replace(t.with_df(df.reset_index(drop=True)))


class ErrorDetection(Operator):
    NAME = "ErrorDetection"
    CATEGORY = "cleaning"
    DOC = (
        "Identify invalid records in a column using a user-defined predicate. "
        "The predicate returns True for an INVALID value."
    )
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("column", kind="column"),
        ParamSpec("func", kind="func",
                  doc="predicate over a cell value; True means the value is erroneous"),
        ParamSpec("action", required=False, default="flag",
                  choices=("flag", "remove", "null"),
                  doc="'flag' adds a <column>_is_error boolean column (the default: "
                      "Sec 2.2.1 says this operator *identifies* invalid records), "
                      "'remove' drops the offending rows, 'null' blanks the values"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        (col,) = require_columns(t, p["column"])
        fn = callable_param(p["func"], self.NAME, "func")
        df = t.df.copy()
        bad = apply_elementwise(df[col], fn, self.NAME).astype(bool)
        action = p["action"] or "flag"
        if action == "remove":
            df = df[~bad].reset_index(drop=True)
        elif action == "flag":
            df[f"{col}_is_error"] = bad.values
        else:  # null
            df.loc[bad.values, col] = np.nan
        return tables.replace(t.with_df(df))


class OutlierDetection(Operator):
    NAME = "OutlierDetection"
    CATEGORY = "cleaning"
    DOC = "Detect statistical outliers in a numerical column."
    PARAMS = (
        ParamSpec("table", kind="table"),
        ParamSpec("column", kind="column"),
        ParamSpec("action", required=False, default="remove",
                  choices=("remove", "flag", "clip"),
                  doc="'remove' drops outlier rows, 'flag' adds <column>_is_outlier, "
                      "'clip' winsorizes to the bounds"),
        ParamSpec("method", required=False, default="iqr", choices=("iqr", "zscore"),
                  doc="detection rule"),
        ParamSpec("threshold", required=False, default=None,
                  doc="IQR multiplier (default 1.5) or z-score cutoff (default 3.0)"),
    )

    def apply(self, tables: TableSet, /, **p: Any) -> TableSet:
        t = require_table(tables, p["table"])
        (col,) = require_columns(t, p["column"])
        s = pd.to_numeric(t.df[col], errors="coerce")
        if s.notna().sum() == 0:
            raise OperatorError(
                f"OutlierDetection: column '{col}' of table {t.name} contains no numeric "
                f"values. Cast it with CastType first, or pick a numeric column."
            )
        method = (p["method"] or "iqr").lower()
        if method == "iqr":
            k = float(p["threshold"]) if p["threshold"] is not None else 1.5
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
        else:
            k = float(p["threshold"]) if p["threshold"] is not None else 3.0
            mu, sd = s.mean(), s.std(ddof=0)
            sd = sd if sd and not pd.isna(sd) else 0.0
            lo, hi = mu - k * sd, mu + k * sd

        out = (s < lo) | (s > hi)
        out = out.fillna(False)
        df = t.df.copy()
        action = p["action"] or "remove"
        if action == "remove":
            df = df[~out.values].reset_index(drop=True)
        elif action == "flag":
            df[f"{col}_is_outlier"] = out.values
        else:  # clip
            df[col] = s.clip(lower=lo, upper=hi)
        return tables.replace(t.with_df(df))
