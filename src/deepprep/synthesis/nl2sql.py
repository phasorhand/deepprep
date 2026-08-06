"""ADP task construction from NL2SQL benchmarks (paper Sec 5.3).

    "To obtain realistic ADP tasks, we instantiate them from NL2SQL benchmarks
     [26,42], which provide real databases paired with analytical SQL queries.
     For each ground-truth query q, we execute it to produce the target table T*
     and use an LLM to generate the corresponding target schema specification."

This module covers the first half of that sentence: it loads a Spider/BIRD-style
benchmark (a JSON list of ``{db_id, question, query}`` plus one SQLite file per
database), executes the gold query to obtain ``T*``, loads the source tables the
query references, and produces the target schema specification ``Sigma*``.

The LLM is *optional*.  Without one, ``Sigma*`` is derived deterministically: the
natural-language question already **is** a table-level specification ``tau*``
(that is precisely the setting of Sec 2.1, "a natural language description of the
target table"), and the column-level specifications are read off the executed
result together with the select-list expression that produced each column.  With
an LLM we ask for the richer specification the paper describes, and fall back to
the deterministic one whenever the response cannot be parsed — synthesis must
never fail because of a malformed generation.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..agent.llm import LLMClient
from ..types import ColumnSpec, Table, TableSchema, TableSet

__all__ = [
    "CleanInstance",
    "NL2SQLCase",
    "build_clean_instance",
    "database_path",
    "execute_sql",
    "extract_json",
    "infer_target_schema",
    "load_benchmark",
    "load_sources",
    "referenced_tables",
    "sqlite_schema",
]


# --------------------------------------------------------------------------- #
# Benchmark records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NL2SQLCase:
    """One ``(database, question, gold query)`` triple of an NL2SQL benchmark."""

    case_id: str
    db_id: str
    question: str
    query: str
    #: BIRD ships an "evidence" field with external knowledge; Spider does not.
    evidence: str | None = None
    #: Difficulty label when the benchmark provides one; kept for task metadata.
    difficulty: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "db_id": self.db_id,
            "question": self.question,
            "query": self.query,
            "evidence": self.evidence,
            "difficulty": self.difficulty,
        }


def load_benchmark(
    path: str | Path,
    *,
    limit: int | None = None,
    db_ids: Iterable[str] | None = None,
) -> list[NL2SQLCase]:
    """Load a Spider- or BIRD-format benchmark file (``.json`` or ``.jsonl``).

    The two benchmarks differ only in the name of the SQL field (``query`` vs
    ``SQL``) and in the presence of ``evidence``/``question_id``, so one reader
    handles both.
    """
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    if isinstance(records, dict):
        records = [records]

    keep = set(db_ids) if db_ids is not None else None
    cases: list[NL2SQLCase] = []
    for i, r in enumerate(records):
        db_id = str(r.get("db_id", ""))
        if keep is not None and db_id not in keep:
            continue
        query = r.get("query") or r.get("SQL") or r.get("sql") or ""
        if not isinstance(query, str) or not query.strip():
            # Spider's dev set stores a parsed `sql` dict alongside the string;
            # a record without executable SQL cannot become an ADP task.
            continue
        raw_id = r.get("question_id", r.get("id", i))
        cases.append(
            NL2SQLCase(
                case_id=f"{db_id}_{raw_id}",
                db_id=db_id,
                question=str(r.get("question", "")).strip(),
                query=query.strip().rstrip(";"),
                evidence=(str(r["evidence"]).strip() or None) if r.get("evidence") else None,
                difficulty=str(r["difficulty"]) if r.get("difficulty") else None,
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


# --------------------------------------------------------------------------- #
# SQLite access
# --------------------------------------------------------------------------- #
def database_path(db_root: str | Path, db_id: str) -> Path:
    """Locate the SQLite file of ``db_id`` under ``db_root``.

    Spider lays databases out as ``database/<db_id>/<db_id>.sqlite``; BIRD uses
    ``<split>_databases/<db_id>/<db_id>.sqlite``.  Both, plus the flat layout, are
    accepted so a caller only has to point at the directory that holds them.
    """
    root = Path(db_root)
    candidates = [
        root / db_id / f"{db_id}.sqlite",
        root / db_id / f"{db_id}.db",
        root / f"{db_id}.sqlite",
        root / f"{db_id}.db",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"no SQLite database for db_id={db_id!r} under {root}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _connect(path: str | Path) -> sqlite3.Connection:
    # `uri=True` + `mode=ro` guarantees synthesis never mutates a benchmark file.
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [str(r[0]) for r in rows]


def sqlite_schema(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> TableSchema:
    """Build ``Sigma_i`` for a source table from SQLite's catalogue.

    Primary/foreign keys are surfaced in the column descriptions because they are
    exactly the "key fields or relationships" the paper warns must survive noise
    injection — an agent that can see them can also avoid breaking them.
    """
    info = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
    fks = conn.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()
    fk_by_col = {str(r[3]): (str(r[2]), str(r[4])) for r in fks}

    columns: list[ColumnSpec] = []
    for row in info:
        col = str(row[1])
        if col not in df.columns:
            continue
        decl = str(row[2] or "").strip()
        is_pk = bool(row[5])
        bits: list[str] = []
        if is_pk:
            bits.append(f"Primary key of {name}.")
        if col in fk_by_col:
            ref_table, ref_col = fk_by_col[col]
            bits.append(f"Foreign key referencing {ref_table}.{ref_col}.")
        if decl:
            bits.append(f"Declared SQL type {decl}.")
        columns.append(
            ColumnSpec(
                name=col,
                dtype=str(df[col].dtype),
                description=" ".join(bits) or None,
            )
        )
    # Columns present in the frame but absent from PRAGMA (should not happen, but
    # a corrupt catalogue must not silently drop data from the schema).
    known = {c.name for c in columns}
    for c in df.columns:
        if str(c) not in known:
            columns.append(ColumnSpec(name=str(c), dtype=str(df[c].dtype)))
    return TableSchema(description=f"Table '{name}' of the source database.", columns=columns)


def load_sources(
    conn: sqlite3.Connection,
    tables: Sequence[str] | None = None,
) -> TableSet:
    """Load the requested tables (default: all) as the source table set ``S``."""
    names = list(tables) if tables is not None else table_names(conn)
    out = TableSet()
    for name in names:
        df = pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
        df = _dedupe_columns(df)
        out.add(Table(name=name, df=df, schema=sqlite_schema(conn, name, df)))
    return out


def execute_sql(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Execute the gold query ``q`` to produce the target table ``T*``."""
    return _dedupe_columns(pd.read_sql_query(query, conn))


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names unique.

    ``SELECT T1.id, T2.id`` yields two columns called ``id``; duplicate names make
    the permutation-invariant matcher fall back to signature matching and make
    operator parameters ambiguous, so they are disambiguated up front.
    """
    names = [str(c) for c in df.columns]
    if len(set(names)) == len(names):
        df.columns = names
        return df
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    df = df.copy()
    df.columns = out
    return df


# --------------------------------------------------------------------------- #
# Which source tables does the query touch?
# --------------------------------------------------------------------------- #
#: Everything between a FROM/JOIN and the next clause boundary.  Capturing the
#: whole clause (rather than a single identifier) is what makes the comma-join
#: form ``FROM a AS T1, b AS T2 WHERE ...`` — very common in Spider — work.
_FROM_CLAUSE_RE = re.compile(
    r"\b(?:from|join)\s+(.*?)"
    r"(?=\b(?:where|group|having|order|limit|union|intersect|except|on|using|"
    r"inner|left|right|full|outer|cross|join)\b|\)|$)",
    re.I | re.S,
)
_LEADING_IDENT_RE = re.compile(r'^\s*("[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)')


def referenced_tables(query: str, available: Sequence[str]) -> list[str]:
    """Return the tables of ``available`` that ``query`` reads, in catalogue order.

    Only the referenced tables become sources: handing the agent every table of a
    30-table database would bury the signal, and the paper's ADP instance is
    "source tables, a target schema specification, a target table".
    """
    lowered = {t.lower(): t for t in available}
    found: list[str] = []
    for clause in _FROM_CLAUSE_RE.findall(query):
        for part in clause.split(","):
            m = _LEADING_IDENT_RE.match(part)
            if m is None:
                continue
            real = lowered.get(m.group(1).strip('"`[]').lower())
            if real is not None and real not in found:
                found.append(real)
    if not found:
        # Fall back to word matching (covers `FROM (SELECT ...)` and odd quoting).
        words = {w.lower() for w in re.findall(r"[A-Za-z_][\w$]*", query)}
        found = [t for t in available if t.lower() in words]
    # Preserve catalogue order for reproducible task serialization.
    order = {t: i for i, t in enumerate(available)}
    return sorted(found, key=lambda t: order.get(t, 0))


# --------------------------------------------------------------------------- #
# Target schema specification Sigma*
# --------------------------------------------------------------------------- #
_SELECT_ITEM_RE = re.compile(r"^\s*select\s+(?:distinct\s+)?(.*?)\s+from\s", re.I | re.S)

_AGG_GLOSS = {
    "count": "the number of",
    "sum": "the total",
    "avg": "the average",
    "min": "the minimum",
    "max": "the maximum",
}


def _select_expressions(query: str) -> list[str]:
    """Split the top-level select list of ``query`` into its expressions."""
    m = _SELECT_ITEM_RE.search(query)
    if not m:
        return []
    body, depth, buf = m.group(1), 0, []
    items: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if "".join(buf).strip():
        items.append("".join(buf).strip())
    return items


def _column_gloss(expr: str) -> str | None:
    """Turn one select-list expression into a column-level description."""
    expr = expr.strip()
    m = re.match(r"^(count|sum|avg|min|max)\s*\(\s*(distinct\s+)?(.*?)\s*\)$", expr, re.I)
    if m:
        fn, distinct, arg = m.group(1).lower(), bool(m.group(2)), m.group(3).strip()
        arg = arg.rsplit(".", 1)[-1]
        if fn == "count" and arg == "*":
            return "The number of matching records."
        subject = f"distinct {arg}" if distinct else arg
        return f"{_AGG_GLOSS[fn].capitalize()} {subject}."
    if "." in expr and " " not in expr:
        table, col = expr.rsplit(".", 1)
        return f"Column '{col}' taken from table '{table}'."
    return None


def _deterministic_target_schema(case: NL2SQLCase, target: pd.DataFrame) -> TableSchema:
    """``Sigma*`` derived from the question and the executed result, no LLM."""
    description = case.question or "The table produced by the analytical query."
    if case.evidence:
        description = f"{description} ({case.evidence})"
    exprs = _select_expressions(case.query)
    columns: list[ColumnSpec] = []
    for i, c in enumerate(target.columns):
        gloss = _column_gloss(exprs[i]) if i < len(exprs) else None
        columns.append(
            ColumnSpec(name=str(c), dtype=str(target[c].dtype), description=gloss)
        )
    return TableSchema(description=description, columns=columns)


_SCHEMA_PROMPT = """\
You write target schema specifications for data preparation tasks.

A data preparation task asks an agent to transform raw source tables into one
target table. The target table is specified by a *target schema* Sigma* = (tau, C):
a table-level description tau saying what one row represents, and one entry per
column giving its name, data type and semantic meaning.

The target table below was produced by running an analytical SQL query over the
source database. Write the target schema an analyst would have written *before*
seeing the query — describe the intent, never mention SQL, tables joins or the
query itself.

## Analytical question
{question}

## Target table (first rows)
{preview}

## Column names and inferred types
{columns}

Reply with a single JSON object and nothing else:
{{"description": "<tau: what one row of the target table represents>",
  "columns": [{{"name": "<exact column name>", "dtype": "<data type>",
               "description": "<what this column means>"}}]}}
"""


def extract_json(text: str) -> Any | None:
    """Pull the first JSON object out of an LLM response.

    Models routinely wrap JSON in prose or a ``` fence; a strict parse would
    discard otherwise usable generations, so we scan for the first balanced
    object instead.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def infer_target_schema(
    case: NL2SQLCase,
    target: pd.DataFrame,
    llm: LLMClient | None = None,
    *,
    max_preview_rows: int = 5,
) -> TableSchema:
    """"use an LLM to generate the corresponding target schema specification".

    Falls back to the deterministic specification when no LLM is available or the
    generation is unusable.
    """
    fallback = _deterministic_target_schema(case, target)
    if llm is None:
        return fallback

    preview = target.head(max_preview_rows).to_markdown(index=False) if len(target) else "(empty)"
    columns = "\n".join(f"- {c} ({target[c].dtype})" for c in target.columns)
    prompt = _SCHEMA_PROMPT.format(
        question=case.question or "(no question text available)",
        preview=preview,
        columns=columns,
    )
    try:
        raw = llm.generate([{"role": "user", "content": prompt}], max_tokens=1024).text
    except Exception:  # noqa: BLE001 - a flaky endpoint must not abort synthesis
        return fallback

    data = extract_json(raw)
    if not isinstance(data, dict):
        return fallback

    described = {
        str(c.get("name")): c
        for c in data.get("columns", [])
        if isinstance(c, dict) and c.get("name")
    }
    # The column *set* is dictated by T*, not by the model: a hallucinated or
    # missing column would make Sigma* unsatisfiable.  Only the prose is adopted.
    columns = [
        ColumnSpec(
            name=str(c),
            dtype=str(target[c].dtype),
            description=(
                str(described[str(c)].get("description"))
                if str(c) in described and described[str(c)].get("description")
                else fallback.columns[i].description
            ),
        )
        for i, c in enumerate(target.columns)
    ]
    description = str(data.get("description") or "").strip() or fallback.description
    return TableSchema(description=description, columns=columns)


# --------------------------------------------------------------------------- #
# The clean task instance
# --------------------------------------------------------------------------- #
@dataclass
class CleanInstance:
    """"a clean task instance consisting of source tables, a target schema
    specification, a target table, and a task pipeline" — everything but the
    pipeline, which :mod:`deepprep.synthesis.pipeline_search` derives.
    """

    case: NL2SQLCase
    sources: TableSet
    target_table: pd.DataFrame
    target_schema: TableSchema
    metadata: dict[str, Any] = field(default_factory=dict)


def build_clean_instance(
    case: NL2SQLCase,
    db_root: str | Path,
    *,
    llm: LLMClient | None = None,
    max_rows_per_table: int | None = None,
    max_target_rows: int | None = None,
    require_non_empty: bool = True,
) -> CleanInstance | None:
    """Execute ``q``, load the tables it reads, and specify ``Sigma*``.

    Returns ``None`` when the case cannot become a usable ADP instance (the query
    errors, the result is empty, or the tables are too large to serialize into a
    training example).  Rejecting is deliberate: a synthesized task whose target
    is empty carries no supervision.
    """
    path = database_path(db_root, case.db_id)
    conn = _connect(path)
    try:
        try:
            target = execute_sql(conn, case.query)
        except Exception:  # noqa: BLE001 - benchmark queries do fail on some DBs
            return None
        if require_non_empty and (len(target) == 0 or target.shape[1] == 0):
            return None
        if max_target_rows is not None and len(target) > max_target_rows:
            return None

        available = table_names(conn)
        used = referenced_tables(case.query, available)
        if not used:
            return None
        sources = load_sources(conn, used)
        if max_rows_per_table is not None and any(
            t.n_rows > max_rows_per_table for t in sources
        ):
            return None
        schema = infer_target_schema(case, target, llm)
    finally:
        conn.close()

    return CleanInstance(
        case=case,
        sources=sources,
        target_table=target,
        target_schema=schema,
        metadata={"db_path": str(path), "source_tables": list(sources.names)},
    )
