"""Operator framework: types, parameter binding, and the call parser.

Paper Sec 2.1:

    "an operator ``o`` is characterized by its type ``kappa`` and parameters
     ``theta``, and defines a function ``o_theta : T -> T``"

:class:`Operator` models the *type* ``kappa`` (a registered, stateless
transformation), and :class:`OperatorCall` models an *instance* ``o_theta``
(a type bound to concrete parameters).

Parsing
-------
The LLM emits operator instances in the paper's notation, e.g.::

    Deduplicate(movies, [id], first)
    ValueTransform(movies, title, lambda x: x.strip().lower())
    Join(movies, directors, on=(dir_id, id), how=left)

Note that ``movies``, ``id`` and ``first`` are bare identifiers, not Python
strings.  We therefore parse the call with :mod:`ast` and evaluate it under a
*symbolic* reader where a bare ``Name`` denotes the string of its identifier.
This accepts the paper's notation verbatim while also accepting fully quoted
Python, so the model is never penalised for either style.
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..types import TableSet

__all__ = [
    "OperatorCall",
    "OperatorError",
    "Operator",
    "ParamSpec",
    "ParseError",
    "parse_operator_call",
    "parse_pipeline",
]


class OperatorError(Exception):
    """Raised when an operator cannot be applied to the current state.

    The message becomes the ``<execute>`` failure feedback surfaced to the agent
    (paper Sec 4.2: "the parent node is annotated with a failure state that
    records the error trace"), so messages must be actionable.
    """


class ParseError(OperatorError):
    """Raised when an operator call is syntactically invalid."""


# --------------------------------------------------------------------------- #
# Parameter specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParamSpec:
    """One entry of ``theta``."""

    name: str
    kind: str = "value"  # value | table | tables | column | columns | func | code
    required: bool = True
    default: Any = None
    choices: tuple[str, ...] | None = None
    doc: str = ""

    def render(self) -> str:
        s = self.name
        if not self.required:
            s += f"={self.default!r}"
        return s


# --------------------------------------------------------------------------- #
# Operator type (kappa)
# --------------------------------------------------------------------------- #
class Operator(ABC):
    """A registered operator *type*.  Stateless: all state lives in the TableSet."""

    NAME: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = ""
    PARAMS: ClassVar[tuple[ParamSpec, ...]] = ()
    DOC: ClassVar[str] = ""

    @abstractmethod
    def apply(self, tables: TableSet, /, **params: Any) -> TableSet:
        """``o_theta : T -> T``.  Must not mutate ``tables`` in place."""

    # -- signature helpers -------------------------------------------------- #
    @classmethod
    def signature(cls) -> str:
        return f"{cls.NAME}({', '.join(p.render() for p in cls.PARAMS)})"

    @classmethod
    def manual(cls) -> str:
        """One entry of the operator manual injected into the system prompt."""
        lines = [f"- {cls.signature()}", f"    {cls.DOC.strip()}"]
        for p in cls.PARAMS:
            bits = [f"      * {p.name}"]
            if p.choices:
                bits.append(f"[{ '|'.join(p.choices) }]")
            if not p.required:
                bits.append(f"(optional, default={p.default!r})")
            if p.doc:
                bits.append(f"- {p.doc}")
            lines.append(" ".join(bits))
        return "\n".join(lines)

    # -- parameter binding -------------------------------------------------- #
    @classmethod
    def bind(cls, args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Bind positional/keyword arguments to ``PARAMS``.

        Raises :class:`ParseError` with a message the agent can act on.
        """
        specs = cls.PARAMS
        by_name = {p.name: p for p in specs}

        for k in kwargs:
            if k not in by_name:
                raise ParseError(
                    f"{cls.NAME}() got an unexpected parameter {k!r}. "
                    f"Expected signature: {cls.signature()}"
                )
        if len(args) > len(specs):
            raise ParseError(
                f"{cls.NAME}() takes at most {len(specs)} positional parameters "
                f"but {len(args)} were given. Expected signature: {cls.signature()}"
            )

        bound: dict[str, Any] = {}
        for i, spec in enumerate(specs):
            if i < len(args):
                if spec.name in kwargs:
                    raise ParseError(
                        f"{cls.NAME}() got multiple values for parameter {spec.name!r}."
                    )
                bound[spec.name] = args[i]
            elif spec.name in kwargs:
                bound[spec.name] = kwargs[spec.name]
            elif spec.required:
                raise ParseError(
                    f"{cls.NAME}() is missing required parameter {spec.name!r}. "
                    f"Expected signature: {cls.signature()}"
                )
            else:
                bound[spec.name] = spec.default

        # Light normalisation so the agent may write either `[a]` or `a`.
        for spec in specs:
            v = bound[spec.name]
            if spec.kind in ("columns", "tables") and v is not None and not isinstance(v, list):
                bound[spec.name] = [v]
            if spec.choices and isinstance(v, str) and v not in spec.choices:
                raise ParseError(
                    f"{cls.NAME}() parameter {spec.name!r} must be one of "
                    f"{list(spec.choices)}, got {v!r}."
                )
        return bound


# --------------------------------------------------------------------------- #
# Operator instance (o_theta)
# --------------------------------------------------------------------------- #
@dataclass
class OperatorCall:
    """An operator instance: a type plus concrete parameters, and its source."""

    op: Operator
    params: dict[str, Any]
    source: str = ""
    #: Raw source text of callable/code parameters, kept for faithful re-serialization.
    raw_params: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return type(self.op).NAME

    @property
    def category(self) -> str:
        return type(self.op).CATEGORY

    def execute(self, tables: TableSet) -> TableSet:
        """Apply ``o_theta`` to a state, wrapping failures as :class:`OperatorError`."""
        try:
            out = self.op.apply(tables, **self.params)
        except OperatorError:
            raise
        except KeyError as e:  # missing table / column
            raise OperatorError(f"{self.name}: {e.args[0] if e.args else e}") from e
        except Exception as e:  # noqa: BLE001 - surfaced to the agent as feedback
            raise OperatorError(f"{self.name}: {type(e).__name__}: {e}") from e
        if not isinstance(out, TableSet):
            raise OperatorError(
                f"{self.name}: operator returned {type(out).__name__}, expected a TableSet"
            )
        return out

    def to_source(self) -> str:
        """Canonical serialization used by ``phi(.)``."""
        if self.source:
            return self.source
        parts = []
        for spec in type(self.op).PARAMS:
            v = self.params.get(spec.name)
            if not spec.required and v == spec.default:
                continue
            if spec.name in self.raw_params:
                parts.append(f"{spec.name}={self.raw_params[spec.name]}")
            else:
                parts.append(f"{spec.name}={v!r}")
        return f"{self.name}({', '.join(parts)})"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<OperatorCall {self.to_source()}>"


# --------------------------------------------------------------------------- #
# Symbolic argument reader
# --------------------------------------------------------------------------- #
_ALLOWED_UNARY = {ast.UAdd: lambda v: +v, ast.USub: lambda v: -v, ast.Not: lambda v: not v}


def _read_arg(node: ast.AST, source: str, sandbox_factory: Callable[[str], Any]) -> Any:
    """Evaluate one argument node under the symbolic reader.

    * ``Constant``            -> the literal value
    * ``Name``                -> its identifier as a ``str`` (paper's bare notation)
    * ``List``/``Tuple``/``Set``/``Dict`` -> recursively read
    * ``Lambda``/``FunctionDef`` -> a sandboxed callable
    * ``UnaryOp`` on a number -> the signed number
    * anything else           -> compiled in the sandbox (e.g. ``str.strip``)
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_read_arg(e, source, sandbox_factory) for e in node.elts]
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values, strict=True):
            if k is None:
                raise ParseError("dict unpacking (**) is not supported in operator parameters")
            out[_read_arg(k, source, sandbox_factory)] = _read_arg(v, source, sandbox_factory)
        return out
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_read_arg(node.operand, source, sandbox_factory))
    # Lambdas, attribute access (str.lower), calls, comprehensions: compile them.
    seg = ast.get_source_segment(source, node)
    if seg is None:  # pragma: no cover - only if source mapping is unavailable
        raise ParseError(f"could not read parameter of type {type(node).__name__}")
    return sandbox_factory(seg)


_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def _normalise_code_fences(text: str) -> str:
    """Turn ```` ```code``` ```` blocks into Python string literals.

    Figure 4 of the paper shows ``ExeCode(..., code=```merged = pd.merge(...)```)``.
    Triple-backtick fences are not Python, so we rewrite them to ``\"\"\"...\"\"\"``
    before handing the call to :func:`ast.parse`.
    """

    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        # Escape backslashes and any embedded triple quotes.
        body = body.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        if body.endswith('"'):
            body += "\n"
        return '"""' + body + '"""'

    return _CODE_FENCE.sub(repl, text)


def _split_top_level(args: str) -> list[str] | None:
    """Split an argument list on top-level commas, respecting brackets and strings."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    i = 0
    n = len(args)
    while i < n:
        ch = args[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(args[i + 1])
                i += 2
                continue
            if args.startswith(quote, i):
                buf.extend(args[i + 1 : i + len(quote)])
                i += len(quote)
                quote = None
                continue
            i += 1
            continue
        if args.startswith('"""', i) or args.startswith("'''", i):
            quote = args[i : i + 3]
            buf.append(quote)
            i += 3
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if quote is not None or depth != 0:
        return None
    parts.append("".join(buf))
    return parts


_CALL_HEAD = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.DOTALL)
_KWARG_HEAD = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(?!=)(.*)$", re.DOTALL)


def _quote_unparsable_args(src: str) -> str | None:
    """Repair a call whose bare arguments are not valid Python expressions.

    The paper writes parameters as bare tokens (``Deduplicate(movies, [id], first)``),
    and most such tokens happen to be valid Python names.  Some are not: format
    strings (``%Y-%m-%d``), regexes, separators and units all appear unquoted in
    model output and would otherwise be a hard parse error.  Rather than reject
    them — which costs the agent a whole turn — quote the offending arguments and
    reparse.

    Returns ``None`` when the call is malformed in a way quoting cannot fix, so
    the caller still reports a real syntax error.
    """
    m = _CALL_HEAD.match(src)
    if not m:
        return None
    name, arglist = m.group(1), m.group(2)
    if not arglist.strip():
        return None
    parts = _split_top_level(arglist)
    if parts is None:
        return None

    repaired: list[str] = []
    changed = False
    for part in parts:
        stripped = part.strip()
        if not stripped:
            repaired.append(part)
            continue
        kw = _KWARG_HEAD.match(stripped)
        key, value = (kw.group(1), kw.group(2)) if kw else (None, stripped)
        try:
            ast.parse(value, mode="eval")
        except SyntaxError:
            literal = repr(value.strip())
            repaired.append(f"{key}={literal}" if key else literal)
            changed = True
            continue
        repaired.append(part)

    if not changed:
        return None
    return f"{name}({', '.join(p.strip() for p in repaired)})"


def parse_operator_call(text: str, registry: dict[str, Operator] | None = None) -> OperatorCall:
    """Parse a single operator call in the paper's notation.

    Raises :class:`ParseError` on malformed input; the message is fed back to the
    agent as execution feedback rather than crashing the run.
    """
    from . import get_operator, operator_names  # local import: avoids a cycle

    src = _normalise_code_fences(text.strip())
    if not src:
        raise ParseError("empty operator call")

    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        repaired = _quote_unparsable_args(src)
        if repaired is None:
            raise ParseError(
                f"invalid operator syntax near line {e.lineno}, offset {e.offset}: {e.msg}. "
                f"Got: {text.strip()[:200]}"
            ) from e
        try:
            tree = ast.parse(repaired, mode="eval")
            src = repaired
        except SyntaxError as e2:
            raise ParseError(
                f"invalid operator syntax near line {e2.lineno}, offset {e2.offset}: {e2.msg}. "
                f"Got: {text.strip()[:200]}"
            ) from e2

    node = tree.body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ParseError(
            f"expected a call of the form OperatorName(...), got: {text.strip()[:200]}"
        )

    op_name = node.func.id
    op = get_operator(op_name) if registry is None else registry.get(op_name)
    if op is None:
        raise ParseError(
            f"unknown operator {op_name!r}. Available operators: {', '.join(operator_names())}"
        )

    from .sandbox import compile_callable  # local import: avoids a cycle

    raw: dict[str, str] = {}

    def factory(segment: str) -> Any:
        return compile_callable(segment)

    args: list[Any] = []
    for a in node.args:
        if isinstance(a, ast.Starred):
            raise ParseError("argument unpacking (*) is not supported in operator parameters")
        args.append(_read_arg(a, src, factory))

    kwargs: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise ParseError("keyword unpacking (**) is not supported in operator parameters")
        kwargs[kw.arg] = _read_arg(kw.value, src, factory)
        seg = ast.get_source_segment(src, kw.value)
        if seg is not None:
            raw[kw.arg] = seg

    # Record raw source for positional callable/code params too.
    specs = type(op).PARAMS
    for i, a in enumerate(node.args):
        if i < len(specs) and specs[i].kind in ("func", "code"):
            seg = ast.get_source_segment(src, a)
            if seg is not None:
                raw[specs[i].name] = seg

    params = type(op).bind(args, kwargs)
    return OperatorCall(op=op, params=params, source=text.strip(), raw_params=raw)


def split_calls(text: str) -> list[str]:
    """Split a multi-operator block into individual call sources.

    Splitting is bracket- and string-aware so that lambdas, embedded newlines and
    fenced code blocks inside ``ExeCode`` do not break the pipeline apart.
    """
    text = _normalise_code_fences(text)
    calls: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    quote: str | None = None
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            if text.startswith(quote, i):
                i += len(quote)
                buf.extend(text[i - len(quote) + 1 : i])
                quote = None
                continue
            i += 1
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            quote = text[i : i + 3]
            buf.append(quote)
            i += 3
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth <= 0 and ch in "\n;":
            chunk = "".join(buf).strip()
            if chunk:
                calls.append(chunk)
            buf = []
            depth = 0
            i += 1
            continue
        buf.append(ch)
        i += 1
    chunk = "".join(buf).strip()
    if chunk:
        calls.append(chunk)
    # Drop decorative lines (bullets, numbering, prose without a call).
    cleaned = []
    for c in calls:
        c = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", c).strip()
        c = c.rstrip(",")
        if c and re.match(r"^[A-Za-z_]\w*\s*\(", c):
            cleaned.append(c)
    return cleaned


def parse_pipeline(text: str) -> list[OperatorCall]:
    """Parse a whole ``<expand>`` block into a pipeline ``P``.

    A parse failure on any operator aborts the pipeline: the paper's environment
    executes operators "sequentially", so a malformed operator must not be
    silently skipped.
    """
    return [parse_operator_call(c) for c in split_calls(text)]
