"""Translating a gold SQL query into an operator pipeline (paper Sec 5.3).

    "We also translate q into an operator pipeline by generating candidate
     pipelines with an LLM and selecting the shortest one that exactly reproduces
     T* through execution."

Two candidate generators feed the same *execution-verified* selection procedure:

* :func:`translate_sql` — a deterministic, rule-based translator for the
  ``SELECT / DISTINCT / JOIN / WHERE / GROUP BY / HAVING / ORDER BY / LIMIT``
  subset that dominates Spider and BIRD.  It exists so the whole synthesis
  pipeline runs offline with no API key, and it emits *several* variants of the
  same query (e.g. with and without a ``Sort`` that no longer affects the
  permutation-invariant target) so that "shortest" is a real choice and not a
  single fixed answer.
* :func:`propose_llm_pipelines` — the paper's generator, used for the queries the
  rule-based translator refuses (correlated sub-queries, set operations, window
  functions, ...).

Selection is identical for both: every candidate is executed on the source tables
and kept only if :func:`deepprep.eval.metrics.table_match` says it reproduces
``T*`` exactly; the shortest survivor wins.  A candidate that merely *looks*
right is never accepted, which is what makes the synthesized supervision sound.

The rule-based translator builds the pipeline *while executing it*.  Static
analysis of post-join column names (pandas suffixes collide with SQL aliases) is
where a translator of this kind normally breaks; carrying the materialized state
along removes that whole class of bug at no cost, since verification has to
execute the pipeline anyway.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..agent.llm import LLMClient
from ..eval.metrics import MatchOptions, table_match
from ..operators import OperatorError, operator_manual, parse_operator_call, parse_pipeline
from ..operators.program import answer_table_name
from ..serialize import serialize_table_set
from ..types import TableSet

__all__ = [
    "PipelineSearchResult",
    "TranslationError",
    "TranslationResult",
    "execute_pipeline",
    "join_key_columns",
    "propose_llm_pipelines",
    "search_pipeline",
    "translate_sql",
    "verify_pipeline",
]


class TranslationError(Exception):
    """Raised when the rule-based translator does not cover a query."""


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #
def execute_pipeline(
    pipeline: Sequence[str], sources: TableSet
) -> tuple[pd.DataFrame | None, str | None]:
    """Run a pipeline on ``sources``; return ``(result table, error)``.

    The result is the table named by ``Terminate``; without a terminator we fall
    back to the table the last operator wrote, which keeps LLM candidates that
    forgot the control operator usable.
    """
    state = sources.copy()
    last: str | None = None
    try:
        for source in pipeline:
            call = parse_operator_call(source)
            before = set(state.names)
            state = call.execute(state)
            new = [n for n in state.names if n not in before]
            if new:
                last = new[-1]
            elif call.name != "Terminate":
                table_param = call.params.get("table") or call.params.get("left")
                if isinstance(table_param, str):
                    last = table_param
    except Exception as e:  # noqa: BLE001 - candidate pipelines are untrusted
        return None, f"{type(e).__name__}: {e}"

    name = answer_table_name(state) or last
    if name is None or name not in state:
        return None, "the pipeline did not produce an identifiable result table"
    return state[name].df, None


def verify_pipeline(
    pipeline: Sequence[str],
    sources: TableSet,
    target: pd.DataFrame,
    options: MatchOptions | None = None,
) -> bool:
    """"selecting the shortest one that exactly reproduces T* through execution"."""
    produced, error = execute_pipeline(pipeline, sources)
    if error is not None:
        return False
    return table_match(produced, target, options)


# --------------------------------------------------------------------------- #
# SQL tokenizer
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<str>'(?:[^']|'')*')
    | (?P<qname>"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\])
    | (?P<num>\d+\.\d*|\.\d+|\d+)
    | (?P<op><>|!=|<=|>=|\|\||[=<>+\-*/%(),.;])
    | (?P<name>[A-Za-z_@#][\w$@#]*)
    """,
    re.X,
)

_KEYWORDS = {
    "select", "distinct", "from", "as", "join", "inner", "left", "right", "full",
    "outer", "cross", "on", "where", "group", "by", "having", "order", "limit",
    "offset", "and", "or", "not", "in", "like", "between", "is", "null", "asc",
    "desc", "union", "intersect", "except", "case", "when", "then", "else", "end",
    "exists", "all", "any", "using", "with",
}

_AGG_FUNCS = {"count", "sum", "avg", "min", "max", "total"}

#: Constructs the rule-based translator deliberately does not model.  They are
#: handed to the LLM generator instead of being mistranslated.
_UNSUPPORTED = {"union", "intersect", "except", "case", "exists", "with", "using", "offset"}


@dataclass(frozen=True)
class _Tok:
    kind: str  # kw | name | num | str | op
    value: str

    @property
    def low(self) -> str:
        return self.value.lower()


def _tokenize(sql: str) -> list[_Tok]:
    toks: list[_Tok] = []
    pos = 0
    sql = sql.strip().rstrip(";")
    while pos < len(sql):
        m = _TOKEN_RE.match(sql, pos)
        if m is None:
            raise TranslationError(f"unexpected character {sql[pos]!r} at offset {pos}")
        pos = m.end()
        kind = m.lastgroup or ""
        text = m.group()
        if kind == "ws":
            continue
        if kind == "qname":
            toks.append(_Tok("name", text.strip('"`[]')))
        elif kind == "str":
            toks.append(_Tok("str", text[1:-1].replace("''", "'")))
        elif kind == "num":
            toks.append(_Tok("num", text))
        elif kind == "op":
            toks.append(_Tok("op", text))
        else:
            toks.append(_Tok("kw" if text.lower() in _KEYWORDS else "name", text))
    return toks


def _split_top_level(toks: Sequence[_Tok], sep: str = ",") -> list[list[_Tok]]:
    parts: list[list[_Tok]] = []
    buf: list[_Tok] = []
    depth = 0
    for t in toks:
        if t.kind == "op" and t.value == "(":
            depth += 1
        elif t.kind == "op" and t.value == ")":
            depth -= 1
        if depth == 0 and t.kind == "op" and t.value == sep:
            parts.append(buf)
            buf = []
            continue
        buf.append(t)
    if buf:
        parts.append(buf)
    return parts


# --------------------------------------------------------------------------- #
# Parsed query
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ColRef:
    qualifier: str | None
    column: str

    def key(self) -> tuple[str | None, str]:
        return ((self.qualifier or "").lower() or None, self.column.lower())


@dataclass(frozen=True)
class _SelectItem:
    agg: str | None
    distinct: bool
    ref: _ColRef | None  # None means COUNT(*) or SELECT *
    star: bool
    alias: str | None

    def agg_key(self) -> tuple[str, str, bool]:
        col = f"{self.ref.qualifier}.{self.ref.column}" if self.ref else "*"
        return (self.agg or "", col.lower(), self.distinct)


@dataclass(frozen=True)
class _TableRef:
    name: str
    alias: str


@dataclass
class _Query:
    distinct: bool
    items: list[_SelectItem]
    tables: list[_TableRef]
    joins: list[tuple[_ColRef, _ColRef]]  # (left ref, right ref) per joined table
    where: list[_Tok]
    group_by: list[_ColRef]
    having: list[_Tok]
    order_by: list[tuple[_SelectItem, bool]]  # (item, ascending)
    limit: int | None


_CLAUSE_STARTS = ("from", "where", "group", "having", "order", "limit")


def _clause_bounds(toks: Sequence[_Tok]) -> dict[str, list[_Tok]]:
    """Split a flat SELECT statement into its top-level clauses."""
    marks: list[tuple[int, str]] = []
    depth = 0
    for i, t in enumerate(toks):
        if t.kind == "op" and t.value == "(":
            depth += 1
        elif t.kind == "op" and t.value == ")":
            depth -= 1
        elif depth == 0 and t.kind == "kw" and t.low in _CLAUSE_STARTS:
            if t.low in ("group", "order") and not (
                i + 1 < len(toks) and toks[i + 1].low == "by"
            ):
                continue
            marks.append((i, t.low))
    if not marks or marks[0][1] != "from":
        raise TranslationError("query has no top-level FROM clause")

    out: dict[str, list[_Tok]] = {"select": list(toks[1 : marks[0][0]])}
    for j, (i, name) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(toks)
        skip = 2 if name in ("group", "order") else 1
        out[name] = list(toks[i + skip : end])
    return out


def _parse_col_ref(toks: Sequence[_Tok]) -> _ColRef:
    if len(toks) == 1 and toks[0].kind == "name":
        return _ColRef(None, toks[0].value)
    if (
        len(toks) == 3
        and toks[0].kind == "name"
        and toks[1].kind == "op"
        and toks[1].value == "."
        and toks[2].kind == "name"
    ):
        return _ColRef(toks[0].value, toks[2].value)
    raise TranslationError(
        "expected a column reference, got " + " ".join(t.value for t in toks)
    )


def _parse_select_item(toks: list[_Tok]) -> _SelectItem:
    alias: str | None = None
    if len(toks) >= 2 and toks[-2].low == "as":
        alias = toks[-1].value
        toks = toks[:-2]
    if len(toks) == 1 and toks[0].kind == "op" and toks[0].value == "*":
        return _SelectItem(None, False, None, True, alias)

    if (
        len(toks) >= 3
        and toks[0].kind in ("name", "kw")
        and toks[0].low in _AGG_FUNCS
        and toks[1].value == "("
        and toks[-1].value == ")"
    ):
        inner = toks[2:-1]
        distinct = bool(inner and inner[0].low == "distinct")
        if distinct:
            inner = inner[1:]
        fn = "sum" if toks[0].low == "total" else toks[0].low
        if len(inner) == 1 and inner[0].value == "*":
            return _SelectItem(fn, distinct, None, False, alias)
        return _SelectItem(fn, distinct, _parse_col_ref(inner), False, alias)

    return _SelectItem(None, False, _parse_col_ref(toks), False, alias)


def _parse_from(toks: list[_Tok]) -> tuple[list[_TableRef], list[tuple[_ColRef, _ColRef]]]:
    """Parse the FROM clause into table references plus equi-join conditions."""
    tables: list[_TableRef] = []
    joins: list[tuple[_ColRef, _ColRef]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "op" and t.value == ",":
            i += 1
            continue
        if t.kind == "kw" and t.low in ("join", "inner", "left", "right", "full", "outer", "cross"):
            if t.low in ("left", "right", "full", "outer", "cross"):
                raise TranslationError(f"{t.value.upper()} JOIN is not translated")
            i += 1
            continue
        if t.kind == "kw" and t.low == "on":
            cond = []
            i += 1
            while i < len(toks) and not (
                toks[i].kind == "kw" and toks[i].low in ("join", "inner", "cross")
            ):
                cond.append(toks[i])
                i += 1
            joins.append(_parse_on(cond))
            continue
        if t.kind != "name":
            raise TranslationError(f"unexpected token {t.value!r} in FROM clause")
        name = t.value
        alias = name
        i += 1
        if i < len(toks) and toks[i].kind == "kw" and toks[i].low == "as":
            i += 1
            alias = toks[i].value
            i += 1
        elif i < len(toks) and toks[i].kind == "name":
            alias = toks[i].value
            i += 1
        tables.append(_TableRef(name, alias))
    if not tables:
        raise TranslationError("FROM clause names no table")
    return tables, joins


def _parse_on(toks: Sequence[_Tok]) -> tuple[_ColRef, _ColRef]:
    parts = _split_on_keyword(toks, "and")
    if len(parts) != 1:
        raise TranslationError("composite ON conditions are not translated")
    eq = [i for i, t in enumerate(toks) if t.kind == "op" and t.value == "="]
    if len(eq) != 1:
        raise TranslationError("only equi-joins are translated")
    return _parse_col_ref(toks[: eq[0]]), _parse_col_ref(toks[eq[0] + 1 :])


def _split_on_keyword(toks: Sequence[_Tok], kw: str) -> list[list[_Tok]]:
    parts: list[list[_Tok]] = []
    buf: list[_Tok] = []
    depth = 0
    for t in toks:
        if t.kind == "op" and t.value == "(":
            depth += 1
        elif t.kind == "op" and t.value == ")":
            depth -= 1
        if depth == 0 and t.kind == "kw" and t.low == kw:
            parts.append(buf)
            buf = []
            continue
        buf.append(t)
    parts.append(buf)
    return parts


def _parse_order_by(toks: list[_Tok], items: list[_SelectItem]) -> list[tuple[_SelectItem, bool]]:
    out: list[tuple[_SelectItem, bool]] = []
    for chunk in _split_top_level(toks):
        if not chunk:
            continue
        ascending = True
        if chunk[-1].kind == "kw" and chunk[-1].low in ("asc", "desc"):
            ascending = chunk[-1].low == "asc"
            chunk = chunk[:-1]
        if len(chunk) == 1 and chunk[0].kind == "num":
            idx = int(chunk[0].value) - 1
            if not 0 <= idx < len(items):
                raise TranslationError("ORDER BY ordinal out of range")
            out.append((items[idx], ascending))
            continue
        out.append((_parse_select_item(list(chunk)), ascending))
    return out


def _parse_query(sql: str) -> _Query:
    toks = _tokenize(sql)
    if not toks or toks[0].low != "select":
        raise TranslationError("only plain SELECT statements are translated")
    lowered = {t.low for t in toks if t.kind == "kw"}
    bad = lowered & _UNSUPPORTED
    if bad:
        raise TranslationError(f"unsupported SQL construct: {sorted(bad)[0].upper()}")
    for i, t in enumerate(toks[:-1]):
        if t.kind == "op" and t.value == "(" and toks[i + 1].low == "select":
            raise TranslationError("sub-queries are not translated")

    clauses = _clause_bounds(toks)
    select_toks = clauses["select"]
    distinct = bool(select_toks and select_toks[0].low == "distinct")
    if distinct:
        select_toks = select_toks[1:]
    items = [_parse_select_item(list(c)) for c in _split_top_level(select_toks) if c]
    if not items:
        raise TranslationError("empty select list")

    tables, joins = _parse_from(clauses["from"])
    if len({t.name for t in tables}) != len(tables):
        raise TranslationError("self-joins are not translated")

    group_by = [_parse_col_ref(list(c)) for c in _split_top_level(clauses.get("group", [])) if c]
    order_by = _parse_order_by(clauses.get("order", []), items)

    limit: int | None = None
    limit_toks = clauses.get("limit", [])
    if limit_toks:
        if len(limit_toks) != 1 or limit_toks[0].kind != "num":
            raise TranslationError("only a constant LIMIT is translated")
        limit = int(limit_toks[0].value)

    return _Query(
        distinct=distinct,
        items=items,
        tables=tables,
        joins=joins,
        where=clauses.get("where", []),
        group_by=group_by,
        having=clauses.get("having", []),
        order_by=order_by,
        limit=limit,
    )


# --------------------------------------------------------------------------- #
# Predicate compilation (WHERE / HAVING -> a Python lambda body)
# --------------------------------------------------------------------------- #
class _PredicateCompiler:
    """Compile a boolean SQL expression into the body of ``lambda r: ...``."""

    def __init__(self, resolve: Any) -> None:
        self.resolve = resolve  # (qualifier, column) -> current dataframe column
        self.toks: list[_Tok] = []
        self.i = 0

    def compile(self, toks: Sequence[_Tok]) -> str:
        self.toks, self.i = list(toks), 0
        expr = self._or()
        if self.i != len(self.toks):
            raise TranslationError(
                "could not parse predicate near " + " ".join(t.value for t in self.toks[self.i :])
            )
        return expr

    # -- grammar ---------------------------------------------------------- #
    def _peek(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _eat(self, value: str) -> bool:
        t = self._peek()
        if t is not None and t.low == value:
            self.i += 1
            return True
        return False

    def _or(self) -> str:
        parts = [self._and()]
        while self._eat("or"):
            parts.append(self._and())
        return parts[0] if len(parts) == 1 else "(" + " or ".join(parts) + ")"

    def _and(self) -> str:
        parts = [self._unary()]
        while self._eat("and"):
            parts.append(self._unary())
        return parts[0] if len(parts) == 1 else "(" + " and ".join(parts) + ")"

    def _unary(self) -> str:
        if self._eat("not"):
            return f"(not {self._unary()})"
        t = self._peek()
        if t is not None and t.kind == "op" and t.value == "(" and self._is_group():
            self.i += 1
            inner = self._or()
            if not (self._peek() and self._peek().value == ")"):  # type: ignore[union-attr]
                raise TranslationError("unbalanced parenthesis in predicate")
            self.i += 1
            return f"({inner})"
        return self._predicate()

    def _is_group(self) -> bool:
        """Distinguish ``(a AND b)`` from an ``IN (...)`` value list."""
        depth = 0
        for t in self.toks[self.i :]:
            if t.kind == "op" and t.value == "(":
                depth += 1
            elif t.kind == "op" and t.value == ")":
                depth -= 1
                if depth == 0:
                    return False
            elif depth == 1 and t.kind == "kw" and t.low in ("and", "or", "not"):
                return True
        return False

    def _predicate(self) -> str:
        left = self._operand()
        negate = self._eat("not")
        t = self._peek()
        if t is None:
            raise TranslationError("predicate ends after an operand")

        if t.low == "is":
            self.i += 1
            neg = self._eat("not")
            if not self._eat("null"):
                raise TranslationError("only IS [NOT] NULL is translated")
            expr = f"pd.isna({left})"
            return f"(not {expr})" if neg != negate else expr

        if t.low == "in":
            self.i += 1
            values = self._value_list()
            expr = f"({left} in ({', '.join(values)}{',' if len(values) == 1 else ''}))"
            return f"(not {expr})" if negate else expr

        if t.low == "like":
            self.i += 1
            pat = self._peek()
            if pat is None or pat.kind != "str":
                raise TranslationError("LIKE requires a string pattern")
            self.i += 1
            regex = _like_to_regex(pat.value)
            expr = f"bool(re.fullmatch({regex!r}, str({left}), re.IGNORECASE))"
            return f"(not {expr})" if negate else expr

        if t.low == "between":
            self.i += 1
            lo = self._operand()
            if not self._eat("and"):
                raise TranslationError("BETWEEN without AND")
            hi = self._operand()
            expr = f"({lo} <= {left} <= {hi})"
            return f"(not {expr})" if negate else expr

        if t.kind == "op" and t.value in ("=", "!=", "<>", "<", ">", "<=", ">="):
            self.i += 1
            op = {"=": "==", "<>": "!="}.get(t.value, t.value)
            right = self._operand()
            expr = f"({left} {op} {right})"
            return f"(not {expr})" if negate else expr

        raise TranslationError(f"unsupported comparison operator {t.value!r}")

    def _value_list(self) -> list[str]:
        if not (self._peek() and self._peek().value == "("):  # type: ignore[union-attr]
            raise TranslationError("IN must be followed by a value list")
        self.i += 1
        values: list[str] = []
        while self._peek() is not None and self._peek().value != ")":  # type: ignore[union-attr]
            if self._peek().value == ",":  # type: ignore[union-attr]
                self.i += 1
                continue
            values.append(self._operand())
        if self._peek() is None:
            raise TranslationError("unterminated IN list")
        self.i += 1
        return values

    def _operand(self) -> str:
        t = self._peek()
        if t is None:
            raise TranslationError("predicate ended unexpectedly")
        if t.kind == "str":
            self.i += 1
            return repr(t.value)
        if t.kind == "num":
            self.i += 1
            return repr(float(t.value) if "." in t.value else int(t.value))
        if t.kind == "op" and t.value == "-":
            self.i += 1
            return f"(-{self._operand()})"
        if t.low in _AGG_FUNCS and self.i + 1 < len(self.toks) and self.toks[self.i + 1].value == "(":
            depth, j = 0, self.i
            while j < len(self.toks):
                if self.toks[j].value == "(":
                    depth += 1
                elif self.toks[j].value == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            item = _parse_select_item(list(self.toks[self.i : j + 1]))
            self.i = j + 1
            return f"r[{self.resolve(item)!r}]"
        if t.kind == "name":
            end = self.i + 3 if (
                self.i + 2 < len(self.toks) and self.toks[self.i + 1].value == "."
            ) else self.i + 1
            ref = _parse_col_ref(self.toks[self.i : end])
            self.i = end
            return f"r[{self.resolve(ref)!r}]"
        raise TranslationError(f"unsupported operand {t.value!r}")


def _like_to_regex(pattern: str) -> str:
    """SQL ``LIKE`` -> a Python regex (``%`` = any run, ``_`` = one character)."""
    out = []
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Rule-based translation
# --------------------------------------------------------------------------- #
@dataclass
class TranslationResult:
    """Candidate pipelines for one query, plus the keys they depend on."""

    candidates: list[list[str]] = field(default_factory=list)
    #: ``{source table: {column}}`` used as join keys — the "key fields or
    #: relationships" Sec 5.3 warns noise injection must not break.
    key_columns: dict[str, set[str]] = field(default_factory=dict)


#: Name of the constant helper column used to express ``COUNT(*)`` and global
#: aggregates through ``GroupBy``.
_HELPER = "_dp_count"

_AGG_TO_PANDAS = {
    "count": "count",
    "sum": "sum",
    "avg": "mean",
    "min": "min",
    "max": "max",
}


class _Builder:
    """Emits operator calls and keeps the materialized state in step."""

    def __init__(self, sources: TableSet) -> None:
        self.state = sources.copy()
        self.ops: list[str] = []

    def run(self, source: str) -> None:
        call = parse_operator_call(source)
        self.state = call.execute(self.state)
        self.ops.append(source)

    def columns(self, table: str) -> list[str]:
        return self.state[table].columns


class _Translator:
    def __init__(self, query: _Query, sources: TableSet) -> None:
        self.q = query
        self.b = _Builder(sources)
        #: current dataframe column -> (alias, original column)
        self.owner: dict[str, tuple[str, str]] = {}
        self.work = ""
        self.post_agg: dict[tuple[str, str, bool], str] = {}
        self.key_columns: dict[str, set[str]] = {}
        self._counter = 0

    # -- naming ------------------------------------------------------------ #
    def _fresh(self, base: str) -> str:
        self._counter += 1
        return self.b.state.unique_name(f"{base}{self._counter}")

    # -- column resolution ------------------------------------------------- #
    def resolve(self, item: _ColRef | _SelectItem) -> str:
        if isinstance(item, _SelectItem):
            if item.agg:
                name = self.post_agg.get(item.agg_key())
                if name is None:
                    raise TranslationError(
                        f"aggregate {item.agg}() is referenced before it is computed"
                    )
                return name
            if item.ref is None:
                raise TranslationError("'*' cannot be used here")
            return self.resolve(item.ref)

        qualifier, column = item.key()
        matches = [
            cur
            for cur, (alias, orig) in self.owner.items()
            if orig.lower() == column and (qualifier is None or alias.lower() == qualifier)
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            # After an aggregation the working table holds output names only.
            direct = [c for c in self.b.columns(self.work) if c.lower() == column]
            if len(direct) == 1:
                return direct[0]
            raise TranslationError(f"cannot resolve column {item.qualifier}.{item.column}")
        raise TranslationError(f"column {item.column!r} is ambiguous across the joined tables")

    # -- FROM / JOIN ------------------------------------------------------- #
    def _build_from(self) -> None:
        first = self.q.tables[0]
        self.work = first.name
        self.owner = {c: (first.alias, c) for c in self.b.columns(self.work)}

        on_by_alias: dict[str, tuple[_ColRef, _ColRef]] = {}
        for left, right in self.q.joins:
            # The ON condition may be written in either direction.
            for ref in (left, right):
                if ref.qualifier:
                    on_by_alias.setdefault(ref.qualifier.lower(), (left, right))

        for tref in self.q.tables[1:]:
            prev_cols = self.b.columns(self.work)
            right_cols = self.b.columns(tref.name)
            target = self._fresh("join")
            pair = on_by_alias.get(tref.alias.lower())
            if pair is None:
                # Comma-joins express the predicate in WHERE; a cross product is
                # the faithful translation and the WHERE Filter restores it.
                self.b.run(f"Join({self.work}, {tref.name}, on=[], how=cross, target={target})")
            else:
                left_ref, right_ref = pair
                if (right_ref.qualifier or "").lower() != tref.alias.lower():
                    left_ref, right_ref = right_ref, left_ref
                lk = self.resolve(left_ref)
                rk = self._original_column(tref.name, right_ref.column)
                self._note_key(left_ref, right_ref, tref)
                self.b.run(
                    f"Join({self.work}, {tref.name}, "
                    f"on={{'left': [{lk!r}], 'right': [{rk!r}]}}, how=inner, target={target})"
                )
            new_cols = self.b.columns(target)
            # pandas.merge emits the left frame's columns first, then the right
            # frame's (suffixed on collision), so positions map provenance back.
            owner: dict[str, tuple[str, str]] = {}
            for i, c in enumerate(new_cols):
                if i < len(prev_cols):
                    owner[c] = self.owner[prev_cols[i]]
                else:
                    owner[c] = (tref.alias, right_cols[i - len(prev_cols)])
            self.owner = owner
            self.work = target

    def _original_column(self, table: str, column: str) -> str:
        for c in self.b.columns(table):
            if c.lower() == column.lower():
                return c
        raise TranslationError(f"table {table!r} has no column {column!r}")

    def _note_key(self, left_ref: _ColRef, right_ref: _ColRef, tref: _TableRef) -> None:
        alias_to_table = {t.alias.lower(): t.name for t in self.q.tables}
        for ref in (left_ref, right_ref):
            table = alias_to_table.get((ref.qualifier or "").lower())
            if table is None and len(self.q.tables) == 1:
                table = self.q.tables[0].name
            if table:
                self.key_columns.setdefault(table, set()).add(
                    self._original_column(table, ref.column)
                )
        self.key_columns.setdefault(tref.name, set())

    # -- WHERE ------------------------------------------------------------- #
    def _build_where(self) -> None:
        if not self.q.where:
            return
        body = _PredicateCompiler(self.resolve).compile(self.q.where)
        self.b.run(f"Filter({self.work}, lambda r: {body})")

    # -- aggregation ------------------------------------------------------- #
    def _aggregate_items(self) -> list[_SelectItem]:
        seen: dict[tuple[str, str, bool], _SelectItem] = {}
        pools: list[Sequence[_SelectItem]] = [
            self.q.items,
            [it for it, _ in self.q.order_by],
            self._having_aggregates(),
        ]
        for pool in pools:
            for it in pool:
                if it.agg:
                    seen.setdefault(it.agg_key(), it)
        return list(seen.values())

    def _having_aggregates(self) -> list[_SelectItem]:
        out: list[_SelectItem] = []
        toks = self.q.having
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.low in _AGG_FUNCS and i + 1 < len(toks) and toks[i + 1].value == "(":
                depth, j = 0, i
                while j < len(toks):
                    if toks[j].value == "(":
                        depth += 1
                    elif toks[j].value == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                out.append(_parse_select_item(list(toks[i : j + 1])))
                i = j + 1
                continue
            i += 1
        return out

    def _count_helper(self, by: Sequence[str]) -> str:
        """A column that is safe to aggregate with ``size`` (never a group key)."""
        free = [c for c in self.b.columns(self.work) if c not in by]
        if free:
            return free[0]
        self.b.run(f"Subtitle({self.work}, 1, {_HELPER})")
        return _HELPER

    def _build_aggregation(self) -> None:
        aggs = self._aggregate_items()
        if not aggs and not self.q.group_by:
            return
        if not aggs:
            # `GROUP BY x` with no aggregate is a DISTINCT over the keys.
            by = [self.resolve(g) for g in self.q.group_by]
            self.b.run(f"SelectColumn({self.work}, {by!r})")
            self.b.run(f"Deduplicate({self.work})")
            self.owner = {c: self.owner[c] for c in self.b.columns(self.work)}
            return

        by = [self.resolve(g) for g in self.q.group_by]
        if not by:
            self._build_global_aggregation(aggs)
            return

        spec: dict[str, list[str]] = {}
        names: dict[tuple[str, str, bool], tuple[str, str]] = {}
        for item in aggs:
            fn, col = self._agg_pandas(item, by)
            spec.setdefault(col, [])
            if fn not in spec[col]:
                spec[col].append(fn)
            names[item.agg_key()] = (col, fn)

        agg_param = {c: (fns[0] if len(fns) == 1 else fns) for c, fns in spec.items()}
        rename: dict[str, str] = {}
        for key, (col, fn) in names.items():
            raw = col if len(spec[col]) == 1 else f"{col}_{fn}"
            canonical = self._canonical_agg_name(key, by, rename)
            if raw != canonical:
                rename[raw] = canonical
            self.post_agg[key] = canonical

        call = f"GroupBy({self.work}, by={by!r}, agg={agg_param!r}"
        if rename:
            call += f", rename={rename!r}"
        self.b.run(call + ")")
        self.owner = {
            c: self.owner[c] for c in self.b.columns(self.work) if c in self.owner
        }

    def _canonical_agg_name(
        self, key: tuple[str, str, bool], by: Sequence[str], taken: dict[str, str]
    ) -> str:
        fn, col, distinct = key
        base = f"{fn}_{col.replace('.', '_').replace('*', 'star')}"
        if distinct:
            base += "_distinct"
        name = base
        i = 2
        while name in by or name in taken.values():
            name = f"{base}_{i}"
            i += 1
        return name

    def _agg_pandas(self, item: _SelectItem, by: Sequence[str]) -> tuple[str, str]:
        """Map one aggregate to a ``(pandas function, column)`` pair."""
        if item.agg == "count" and (item.ref is None or item.distinct is False):
            if item.ref is None:
                return "size", self._count_helper(by)
            col = self.resolve(item.ref)
            if col in by:
                # COUNT(k) over the grouping key k equals the group size.
                return "size", self._count_helper(by)
            return "count", col
        col = self.resolve(item.ref) if item.ref else self._count_helper(by)
        if item.distinct:
            if item.agg != "count":
                raise TranslationError(f"DISTINCT inside {item.agg}() is not translated")
            return "nunique", col
        if col in by:
            raise TranslationError("aggregating a grouping key is not translated")
        fn = _AGG_TO_PANDAS.get(item.agg or "")
        if fn is None:
            raise TranslationError(f"unsupported aggregate {item.agg!r}")
        return fn, col

    def _build_global_aggregation(self, aggs: list[_SelectItem]) -> None:
        """Aggregates without GROUP BY collapse the table to a single row."""
        if len(aggs) == 1 and not self.q.having:
            item = aggs[0]
            name = self._canonical_agg_name(item.agg_key(), (), {})
            target = self._fresh("agg")
            if item.agg == "count" and item.ref is None:
                self.b.run(f"Count({self.work}, target={target}, column={name!r})")
            else:
                col = self.resolve(item.ref) if item.ref else ""
                self.b.run(
                    f"CalculateStatistic({self.work}, {name!r}, "
                    f"lambda df: {self._stat_expr(item, col)}, target={target})"
                )
            self.work = target
            self.owner = {}
            self.post_agg[item.agg_key()] = name
            return

        # Several global aggregates: group by a constant column, then drop it.
        self.b.run(f"Subtitle({self.work}, 1, {_HELPER})")
        self.q.group_by = [_ColRef(None, _HELPER)]
        self.owner[_HELPER] = ("", _HELPER)
        self._build_aggregation()
        self.b.run(f"DropColumn({self.work}, [{_HELPER!r}])")

    @staticmethod
    def _stat_expr(item: _SelectItem, col: str) -> str:
        if item.agg == "count":
            return f"df[{col!r}].nunique()" if item.distinct else f"df[{col!r}].notna().sum()"
        fn = _AGG_TO_PANDAS[item.agg or ""]
        if fn in ("mean", "sum"):
            # Mirror GroupBy's coercion so a numeric column stored as text works.
            return f"pd.to_numeric(df[{col!r}], errors='coerce').{fn}()"
        return f"df[{col!r}].{fn}()"

    # -- HAVING / ORDER BY / LIMIT ----------------------------------------- #
    def _build_having(self) -> None:
        if not self.q.having:
            return
        body = _PredicateCompiler(self.resolve).compile(self.q.having)
        self.b.run(f"Filter({self.work}, lambda r: {body})")

    def _build_order(self, emit: bool) -> None:
        if not self.q.order_by or not emit:
            return
        by = [self.resolve(item) for item, _ in self.q.order_by]
        ascending = [asc for _, asc in self.q.order_by]
        self.b.run(f"Sort({self.work}, by={by!r}, ascending={ascending!r})")

    def _build_projection(self, target_columns: Sequence[str]) -> None:
        if any(it.star for it in self.q.items):
            cols = list(self.b.columns(self.work))
        else:
            cols = [self.resolve(it) for it in self.q.items]
        if len(set(cols)) != len(cols):
            raise TranslationError("the select list projects the same column twice")
        if cols != self.b.columns(self.work):
            self.b.run(f"SelectColumn({self.work}, {cols!r})")
        if self.q.distinct:
            self.b.run(f"Deduplicate({self.work})")
        if self.q.limit is not None:
            self.b.run(f"TopK({self.work}, {self.q.limit})")
        # T*'s column names come from SQLite; Sigma* demands exactly those, so the
        # last step aligns names positionally with the target schema.
        if len(target_columns) == len(cols):
            rename = {c: str(t) for c, t in zip(cols, target_columns, strict=True) if c != str(t)}
            if rename:
                self.b.run(f"RenameColumn({self.work}, {rename!r})")

    # -- driver ------------------------------------------------------------ #
    def build(self, target_columns: Sequence[str], emit_sort: bool) -> list[str]:
        self._build_from()
        self._build_where()
        self._build_aggregation()
        self._build_having()
        self._build_order(emit_sort)
        self._build_projection(target_columns)
        self.b.run(f"Terminate([{self.work}])")
        return list(self.b.ops)


def translate_sql(
    sql: str,
    sources: TableSet,
    target: pd.DataFrame | None = None,
) -> TranslationResult:
    """Rule-based ``q -> P`` translation, returning several candidate variants.

    Raises :class:`TranslationError` when the query falls outside the covered
    subset — the caller then falls back to the LLM generator, exactly as the
    paper does for every query.
    """
    query = _parse_query(sql)
    target_columns = [str(c) for c in target.columns] if target is not None else []

    result = TranslationResult()
    # Variant 2 drops ORDER BY when no LIMIT depends on it: the paper's matching
    # metric is row-permutation invariant, so the shorter pipeline is equivalent
    # and "the shortest one that exactly reproduces T*" should prefer it.
    variants = [True] if query.limit is not None else [False, True]
    for emit_sort in variants:
        translator = _Translator(_parse_query(sql), sources)
        try:
            result.candidates.append(translator.build(target_columns, emit_sort))
        except (TranslationError, OperatorError):
            continue
        # Key columns are variant-independent; record them from any success.
        result.key_columns = {k: set(v) for k, v in translator.key_columns.items()}
    if not result.candidates:
        raise TranslationError(f"no variant of the query could be executed: {sql[:160]}")
    return result


def join_key_columns(sql: str, sources: TableSet) -> dict[str, set[str]]:
    """Best-effort ``{table: {join key columns}}`` for noise protection.

    Used when the pipeline came from the LLM (so no translation state exists);
    falls back to a textual scan of the ON conditions.
    """
    try:
        return translate_sql(sql, sources).key_columns
    except (TranslationError, OperatorError):
        pass
    keys: dict[str, set[str]] = {}
    alias_re = re.compile(r"\b(?:from|join)\s+([\w$]+)(?:\s+(?:as\s+)?([\w$]+))?", re.I)
    aliases = {
        (m.group(2) or m.group(1)).lower(): m.group(1)
        for m in alias_re.finditer(sql)
        if m.group(1).lower() in {n.lower() for n in sources.names}
    }
    real = {n.lower(): n for n in sources.names}
    for m in re.finditer(r"([\w$]+)\.([\w$]+)\s*=\s*([\w$]+)\.([\w$]+)", sql):
        for alias, col in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
            table = real.get(aliases.get(alias.lower(), alias).lower())
            if table and col in sources[table].columns:
                keys.setdefault(table, set()).add(col)
    return keys


# --------------------------------------------------------------------------- #
# LLM candidate generation (the paper's generator)
# --------------------------------------------------------------------------- #
_PIPELINE_PROMPT = """\
You translate an analytical SQL query into a data preparation pipeline built from
a fixed operator set. The pipeline is executed over the source tables and must
reproduce the query result *exactly*.

## Source tables
{tables}

## SQL query to translate
```sql
{sql}
```

## Expected result (first rows)
{preview}

## Operator manual
{manual}

Write {n} alternative pipelines, from the one you believe is shortest to the
longest. Each pipeline is one operator per line, wrapped in its own block:

<pipeline>
Filter(table, lambda r: r['x'] > 0)
SelectColumn(table, [a, b])
Terminate([table])
</pipeline>

Rules:
- The last operator of every pipeline is Terminate([<result table>]).
- The result must have exactly the column names shown in the expected result.
- Prefer the fewest operators that still reproduce the result exactly.
- Emit nothing except the <pipeline> blocks.
"""

_PIPELINE_BLOCK = re.compile(r"<pipeline\s*>(.*?)</pipeline\s*>", re.S | re.I)


def propose_llm_pipelines(
    sql: str,
    sources: TableSet,
    target: pd.DataFrame,
    llm: LLMClient,
    *,
    n: int = 4,
    max_tokens: int = 2048,
) -> list[list[str]]:
    """"generating candidate pipelines with an LLM" (Sec 5.3)."""
    preview = target.head(5).to_markdown(index=False) if len(target) else "(empty result)"
    prompt = _PIPELINE_PROMPT.format(
        tables=serialize_table_set(sources, max_rows=5),
        sql=sql,
        preview=preview,
        manual=operator_manual(),
        n=n,
    )
    try:
        raw = llm.generate([{"role": "user", "content": prompt}], max_tokens=max_tokens).text
    except Exception:  # noqa: BLE001 - a failed generation is simply no candidate
        return []

    out: list[list[str]] = []
    for block in _PIPELINE_BLOCK.findall(raw):
        try:
            ops = [call.to_source() for call in parse_pipeline(block)]
        except Exception:  # noqa: BLE001 - malformed candidates are dropped
            continue
        if ops:
            out.append(ops)
    return out


# --------------------------------------------------------------------------- #
# Verified selection
# --------------------------------------------------------------------------- #
@dataclass
class PipelineSearchResult:
    """Outcome of "selecting the shortest one that exactly reproduces T*"."""

    pipeline: list[str] = field(default_factory=list)
    verified: bool = False
    origin: str = "none"  # rule | llm | none
    n_candidates: int = 0
    n_verified: int = 0
    key_columns: dict[str, set[str]] = field(default_factory=dict)
    error: str | None = None


def search_pipeline(
    sql: str,
    sources: TableSet,
    target: pd.DataFrame,
    *,
    llm: LLMClient | None = None,
    n_llm_candidates: int = 4,
    use_llm_always: bool = False,
    match_options: MatchOptions | None = None,
) -> PipelineSearchResult:
    """Generate candidate pipelines and keep the shortest verified one.

    The rule-based translator runs first because it is free; the LLM is queried
    when it produced nothing (or when ``use_llm_always`` asks for the paper's
    behaviour of always sampling candidates).
    """
    result = PipelineSearchResult()
    candidates: list[tuple[str, list[str]]] = []

    try:
        translated = translate_sql(sql, sources, target)
        result.key_columns = translated.key_columns
        candidates.extend(("rule", c) for c in translated.candidates)
    except (TranslationError, OperatorError) as e:
        result.error = str(e)

    if llm is not None and (use_llm_always or not candidates):
        candidates.extend(
            ("llm", c) for c in propose_llm_pipelines(sql, sources, target, llm, n=n_llm_candidates)
        )
        if not result.key_columns:
            result.key_columns = join_key_columns(sql, sources)

    result.n_candidates = len(candidates)
    verified = [
        (origin, pipe)
        for origin, pipe in candidates
        if verify_pipeline(pipe, sources, target, match_options)
    ]
    result.n_verified = len(verified)
    if not verified:
        result.error = result.error or "no candidate pipeline reproduced the target table"
        return result

    # "selecting the shortest one" — ties keep generation order, so the
    # deterministic rule-based candidate wins over an equally short LLM one.
    origin, best = min(verified, key=lambda item: len(item[1]))
    result.pipeline = best
    result.origin = origin
    result.verified = True
    result.error = None
    return result
