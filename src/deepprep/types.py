"""Core data types for DeepPrep (paper Section 2.1, "Problem Formulation").

The paper models a source table as ``S_i = (Sigma_i, D_i)`` where the schema
``Sigma_i = (tau_i, C_i)`` carries a *table-level* textual description ``tau_i``
and a set of *column-level* specifications ``C_i``.  ``D_i`` is the tuple set.

We keep that split explicit:

* :class:`ColumnSpec`  -> one element of ``C_i`` (name, dtype, description)
* :class:`TableSchema` -> ``Sigma_i = (tau_i, C_i)``
* :class:`Table`       -> ``S_i = (Sigma_i, D_i)``; ``D_i`` is a pandas DataFrame
* :class:`TableSet`    -> the environment state ``T`` (an ordered set of tables)
* :class:`ADPTask`     -> ``(S, Sigma*, T*)``: sources, target schema, and the
  ground-truth target table which is used *only for evaluation*.

Per the paper, "the schema Sigma_i of a source table may be partially specified
or entirely unavailable.  In such cases, table- and column-level specifications
can be inferred from the raw data in D_i."  :meth:`TableSchema.infer` does that.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "ADPTask",
    "ColumnSpec",
    "Table",
    "TableSchema",
    "TableSet",
]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class ColumnSpec:
    """A column-level specification ``c_i^j`` (paper Sec 2.1).

    Includes "column-level metadata such as column name, data type, and semantic
    description".
    """

    name: str
    dtype: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.dtype:
            d["dtype"] = self.dtype
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: Any) -> ColumnSpec:
        if isinstance(d, str):
            return cls(name=d)
        return cls(
            name=str(d["name"]),
            dtype=d.get("dtype"),
            description=d.get("description") or d.get("desc"),
        )


@dataclass
class TableSchema:
    """A schema ``Sigma = (tau, C)`` (paper Sec 2.1)."""

    description: str | None = None
    columns: list[ColumnSpec] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def get(self, name: str) -> ColumnSpec | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    @classmethod
    def infer(cls, df: pd.DataFrame, description: str | None = None) -> TableSchema:
        """Infer ``Sigma`` from raw data ``D`` when the schema is unavailable."""
        return cls(
            description=description,
            columns=[ColumnSpec(name=str(c), dtype=str(df[c].dtype)) for c in df.columns],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "columns": [c.to_dict() for c in self.columns],
        }

    @classmethod
    def from_dict(cls, d: Any) -> TableSchema:
        if d is None:
            return cls()
        if isinstance(d, str):
            return cls(description=d)
        return cls(
            description=d.get("description") or d.get("table_description"),
            columns=[ColumnSpec.from_dict(c) for c in d.get("columns", [])],
        )


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
@dataclass
class Table:
    """A table ``(Sigma, D)`` with a name used to reference it in operators."""

    name: str
    df: pd.DataFrame
    schema: TableSchema | None = None

    def __post_init__(self) -> None:
        if self.schema is None:
            self.schema = TableSchema.infer(self.df)
        # Column names are always strings so operator parameters can reference them.
        self.df = self.df.rename(columns={c: str(c) for c in self.df.columns})

    # -- convenience ------------------------------------------------------- #
    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.df.columns]

    @property
    def n_rows(self) -> int:
        return int(len(self.df))

    @property
    def description(self) -> str | None:
        return self.schema.description if self.schema else None

    def copy(self, name: str | None = None) -> Table:
        """Deep-copy the tuple set.  Node states must never alias each other."""
        return Table(
            name=name or self.name,
            df=self.df.copy(deep=True),
            schema=TableSchema(
                description=self.schema.description if self.schema else None,
                columns=list(self.schema.columns) if self.schema else [],
            ),
        )

    def with_df(self, df: pd.DataFrame, name: str | None = None) -> Table:
        """Return a new table carrying ``df``, refreshing inferred column dtypes.

        Column *descriptions* from the old schema are preserved for columns that
        survive the transformation, so semantic metadata is not lost across a
        pipeline.
        """
        old = {c.name: c for c in (self.schema.columns if self.schema else [])}
        cols = [
            ColumnSpec(
                name=str(c),
                dtype=str(df[c].dtype),
                description=old[str(c)].description if str(c) in old else None,
            )
            for c in df.columns
        ]
        return Table(
            name=name or self.name,
            df=df,
            schema=TableSchema(
                description=self.schema.description if self.schema else None,
                columns=cols,
            ),
        )


# --------------------------------------------------------------------------- #
# Table set  (= one environment state / one node of the reasoning tree)
# --------------------------------------------------------------------------- #
class TableSet:
    """An ordered set of tables ``T`` (paper Sec 2.1: ``o_theta : T -> T``).

    This *is* the environment state.  Operators consume a ``TableSet`` and
    produce a new one; a node of the agentic reasoning tree stores exactly one.
    """

    def __init__(self, tables: Iterable[Table] | None = None) -> None:
        self._tables: OrderedDict[str, Table] = OrderedDict()
        for t in tables or []:
            self._tables[t.name] = t

    # -- mapping protocol -------------------------------------------------- #
    def __contains__(self, name: object) -> bool:
        return name in self._tables

    def __getitem__(self, name: str) -> Table:
        if name not in self._tables:
            raise KeyError(
                f"No table named {name!r}. Available tables: {sorted(self._tables)}"
            )
        return self._tables[name]

    def __setitem__(self, name: str, table: Table) -> None:
        self._tables[name] = table

    def __iter__(self) -> Iterator[Table]:
        return iter(self._tables.values())

    def __len__(self) -> int:
        return len(self._tables)

    def __repr__(self) -> str:
        parts = [f"{t.name}{tuple(t.df.shape)}" for t in self]
        return f"TableSet({', '.join(parts)})"

    # -- accessors --------------------------------------------------------- #
    @property
    def names(self) -> list[str]:
        return list(self._tables)

    def get(self, name: str, default: Table | None = None) -> Table | None:
        return self._tables.get(name, default)

    def require(self, name: str) -> Table:
        """Fetch a table, raising a *feedback-friendly* error if it is missing."""
        if name not in self._tables:
            raise KeyError(
                f"No table named {name!r}. Available tables: {sorted(self._tables)}"
            )
        return self._tables[name]

    def add(self, table: Table) -> None:
        self._tables[table.name] = table

    def drop(self, name: str) -> None:
        self._tables.pop(name, None)

    def copy(self) -> TableSet:
        """Deep copy — required so a tree node's materialized state is immutable."""
        return TableSet(t.copy() for t in self)

    def replace(self, table: Table) -> TableSet:
        """Return a *new* TableSet with ``table`` replacing/adding by name.

        Operators are pure functions ``T -> T``; the paper's example notes that
        after ``Deduplicate(movies, ...)`` the "tables ratings and directors
        remain unchanged", i.e. untouched tables are carried over.
        """
        new = TableSet()
        for t in self:
            new.add(table if t.name == table.name else t)
        if table.name not in new:
            new.add(table)
        return new

    def unique_name(self, base: str) -> str:
        """Derive a fresh table name (used by Join/Union/ExeCode outputs)."""
        if base not in self._tables:
            return base
        i = 2
        while f"{base}_{i}" in self._tables:
            i += 1
        return f"{base}_{i}"

    # -- fingerprinting ---------------------------------------------------- #
    def fingerprint(self) -> str:
        """Stable content hash used to detect duplicate states in the tree."""
        import hashlib

        h = hashlib.sha256()
        for t in self:
            h.update(t.name.encode())
            h.update(str(list(t.df.columns)).encode())
            h.update(
                pd.util.hash_pandas_object(t.df.astype(str), index=False).values.tobytes()
            )
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
@dataclass
class ADPTask:
    """An autonomous data preparation instance (paper Sec 2.1, Problem Statement).

    Given source tables ``S``, a target schema ``Sigma*`` and operator types
    ``O``, find a pipeline ``P`` transforming ``S`` into ``T_hat = (Sigma*, D_hat)``.

    ``target_table`` holds the ground-truth ``T* = (Sigma*, D*)``, "which is used
    *only for evaluation*" — the agent never sees it.
    """

    task_id: str
    sources: TableSet
    target_schema: TableSchema
    target_table: pd.DataFrame | None = None
    gold_pipeline: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- (de)serialization -------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sources": [
                {
                    "name": t.name,
                    "schema": t.schema.to_dict() if t.schema else None,
                    "data": t.df.to_dict(orient="split"),
                }
                for t in self.sources
            ],
            "target_schema": self.target_schema.to_dict(),
            "target_table": (
                self.target_table.to_dict(orient="split")
                if self.target_table is not None
                else None
            ),
            "gold_pipeline": list(self.gold_pipeline),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ADPTask:
        sources = TableSet()
        for s in d["sources"]:
            df = _df_from_split(s["data"])
            sources.add(Table(name=s["name"], df=df, schema=TableSchema.from_dict(s.get("schema"))))
        target = d.get("target_table")
        return cls(
            task_id=str(d["task_id"]),
            sources=sources,
            target_schema=TableSchema.from_dict(d["target_schema"]),
            target_table=_df_from_split(target) if target is not None else None,
            gold_pipeline=list(d.get("gold_pipeline", [])),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ADPTask:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _df_from_split(d: Any) -> pd.DataFrame:
    """Rebuild a DataFrame from ``to_dict(orient="split")`` output."""
    if isinstance(d, pd.DataFrame):
        return d
    return pd.DataFrame(d["data"], columns=[str(c) for c in d["columns"]])


def load_tasks(path: str | Path) -> list[ADPTask]:
    """Load a JSONL/JSON file of ADP tasks."""
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl":
        return [ADPTask.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    return [ADPTask.from_dict(d) for d in data]
