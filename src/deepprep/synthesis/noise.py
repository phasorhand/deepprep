"""Reversible noise injection (paper Sec 5.3).

    "Since NL2SQL data is typically clean, we introduce controlled noise to
     increase data diversity for training. We inject noise by applying inverse
     transformations of data-cleaning operators, so that each corruption
     corresponds to a valid cleaning step. Specifically, we sample an operator and
     use an LLM to generate its inverse transformation logic. For example, the
     inverse of a date-standardization operator converts a normalized date (e.g.
     '2023-01-01') into heterogeneous formats (e.g. '01/01/23'). After each
     corruption step, we verify executability by applying the corresponding
     cleaning operator and checking that the previous table state is restored.
     Only reversible corruptions are kept. Repeating this process produces dirty
     source tables together with a matching cleaning pipeline."

The whole module hangs off one invariant, enforced in :func:`try_corruption`:

    ``clean(corrupt(T)) == T``  — checked by *executing* both.

Nothing else is trusted.  A corruption whose paired cleaning operator does not
restore the previous state bit-for-bit is discarded, which is what guarantees the
concatenated pipeline (cleaning ++ task) still produces ``T*``.

State equality is deliberately stricter than the paper's evaluation metric: it
compares table names, column order, **dtypes** and cell values *positionally*
(via :meth:`TableSet.fingerprint`).  Row-permutation invariance would let a
corruption that merely reorders rows pass, and dtype-blindness would let a
``CastType`` that never restored ``int64`` pass; both would silently poison
downstream operators such as ``TopK`` or ``GroupBy``.

Two proposers feed the same gate:

* :data:`BUILTIN_NOISE_PAIRS` — deterministic (corruption, cleaning operator)
  pairs covering every operator of Sec 2.2.1 (Data Cleaning) and Sec 2.2.2 (Value
  Normalization).  These make the synthesizer runnable with no API key.
* :class:`LLMInverseProposer` — the paper's method: sample an operator and ask an
  LLM for its inverse transformation logic.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..agent.llm import LLMClient
from ..operators import parse_operator_call
from ..operators.sandbox import SandboxError, compile_callable
from ..serialize import serialize_table
from ..types import Table, TableSet
from .nl2sql import extract_json

__all__ = [
    "BUILTIN_NOISE_PAIRS",
    "Corruption",
    "LLMInverseProposer",
    "NoiseConfig",
    "NoisePair",
    "NoiseResult",
    "apply_cleaning",
    "inject_noise",
    "state_signature",
    "try_corruption",
]


# --------------------------------------------------------------------------- #
# The reversibility gate
# --------------------------------------------------------------------------- #
@dataclass
class Corruption:
    """One "inverse transformation" together with the cleaning step it inverts."""

    kind: str
    table: str
    column: str | None
    #: ``D -> D'``: the inverse of ``cleaning``, applied to one table's tuple set.
    corrupt: Callable[[pd.DataFrame], pd.DataFrame]
    #: Source of the operator instance that must restore the previous state.
    cleaning: str
    #: Name of the cleaning operator type, for dataset statistics.
    cleaning_operator: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "table": self.table,
            "column": self.column,
            "cleaning": self.cleaning,
            "cleaning_operator": self.cleaning_operator,
            "note": self.note,
        }


def state_signature(state: TableSet) -> str:
    """A content+dtype signature used to decide "the previous table state is restored".

    ``TableSet.fingerprint`` already hashes table names, column order and cell
    values positionally.  Column dtypes are appended because the fingerprint casts
    to ``str`` first, so a corruption that turned ``int64`` into text would
    otherwise look perfectly reversible while leaving the source table typed
    differently from before.
    """
    parts = [state.fingerprint()]
    for t in state:
        dtypes = ",".join(f"{c}:{t.df[c].dtype}" for c in t.columns)
        parts.append(f"{t.name}[{dtypes}]")
    return "|".join(parts)


def try_corruption(state: TableSet, corruption: Corruption) -> TableSet | None:
    """Apply a corruption and keep it only if its cleaning operator undoes it.

        "After each corruption step, we verify executability by applying the
         corresponding cleaning operator and checking that the previous table
         state is restored.  Only reversible corruptions are kept."

    Returns the corrupted state, or ``None`` when the corruption is rejected
    (it failed, changed nothing, or was not reversible).
    """
    table = state.get(corruption.table)
    if table is None:
        return None
    before = state_signature(state)

    try:
        dirty_df = corruption.corrupt(table.df.copy(deep=True))
    except Exception:  # noqa: BLE001 - proposers are untrusted, including the LLM's
        return None
    if not isinstance(dirty_df, pd.DataFrame):
        return None

    dirty = state.replace(table.with_df(dirty_df.reset_index(drop=True)))
    if state_signature(dirty) == before:
        # A corruption that changes nothing would add a no-op cleaning operator to
        # the gold pipeline, teaching the model to emit operators without reason.
        return None

    try:
        restored = parse_operator_call(corruption.cleaning).execute(dirty)
    except Exception:  # noqa: BLE001 - parse and operator errors alike
        return None
    if state_signature(restored) != before:
        return None
    return dirty


# --------------------------------------------------------------------------- #
# Built-in (corruption, cleaning operator) pairs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoisePair:
    """A named corruption family and the operator that repairs it."""

    name: str
    #: Operator type of Sec 2.2.1 / 2.2.2 whose inverse this pair implements.
    cleaning_operator: str
    #: ``(table, column, rng) -> Corruption | None``; ``None`` = precondition unmet.
    propose: Callable[[Table, str, random.Random], Corruption | None]
    #: Whether the corruption appends rows (disabled by ``NoiseConfig``).
    inserts_rows: bool = False


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _values(s: pd.Series) -> list[Any]:
    return s.tolist()


def _has_null(s: pd.Series) -> bool:
    return bool(s.isna().any())


def _set_column(df: pd.DataFrame, column: str, values: Any) -> pd.DataFrame:
    out = df.copy()
    out[column] = values
    return out


def _append_rows(df: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, rows], ignore_index=True)


# -- 2.2.2 StandardizeDatetime ---------------------------------------------- #
_ISO = "%Y-%m-%d"
_ALT_DATE_FORMATS = ("%m/%d/%Y", "%d-%b-%Y", "%m/%d/%y", "%B %d, %Y", "%Y/%m/%d")


def _propose_date_reformat(
    table: Table, column: str, rng: random.Random
) -> Corruption | None:
    """Inverse of date standardization: ISO dates -> a heterogeneous format.

    This is the paper's own example ("'2023-01-01' into ... '01/01/23'").
    """
    s = table.df[column]
    if not _is_text(s) or _has_null(s) or len(s) == 0:
        return None
    try:
        pd.to_datetime(s, format=_ISO)
    except (ValueError, TypeError):
        return None
    fmt = rng.choice(_ALT_DATE_FORMATS)

    def corrupt(df: pd.DataFrame) -> pd.DataFrame:
        return _set_column(df, column, pd.to_datetime(df[column], format=_ISO).dt.strftime(fmt))

    return Corruption(
        kind="date_reformat",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=(
            f"StandardizeDatetime({table.name}, {column!r}, "
            f"format={_ISO!r}, input_format={fmt!r})"
        ),
        cleaning_operator="StandardizeDatetime",
        note=f"dates rewritten as {fmt}",
    )


# -- 2.2.1 Deduplicate ------------------------------------------------------- #
def _propose_duplicate_rows(
    table: Table, column: str, rng: random.Random
) -> Corruption | None:
    """Inverse of Deduplicate: re-insert copies of existing records."""
    df = table.df
    if len(df) < 2 or df.duplicated().any():
        return None
    k = rng.randint(1, min(3, len(df)))
    picks = [rng.randrange(len(df)) for _ in range(k)]

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        return _append_rows(d, d.iloc[picks])

    return Corruption(
        kind="duplicate_rows",
        table=table.name,
        column=None,
        corrupt=corrupt,
        cleaning=f"Deduplicate({table.name}, keep=first)",
        cleaning_operator="Deduplicate",
        note=f"{k} duplicate record(s) appended",
    )


# -- 2.2.1 DropNA ------------------------------------------------------------ #
def _propose_null_rows(table: Table, column: str, rng: random.Random) -> Corruption | None:
    """Inverse of DropNA: append records whose value in ``column`` is missing."""
    df = table.df
    s = df[column]
    # An integer column cannot hold NaN without silently widening to float, and
    # DropNA does not narrow it back, so such columns are skipped.
    if len(df) == 0 or _has_null(s) or pd.api.types.is_integer_dtype(s):
        return None
    k = rng.randint(1, min(2, len(df)))
    picks = [rng.randrange(len(df)) for _ in range(k)]

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        rows = d.iloc[picks].copy()
        rows[column] = None
        return _append_rows(d, rows)

    return Corruption(
        kind="null_rows",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=f"DropNA({table.name}, subset=[{column!r}])",
        cleaning_operator="DropNA",
        note=f"{k} record(s) with a missing '{column}' appended",
    )


# -- 2.2.1 MissingValueImputation -------------------------------------------- #
def _propose_null_ffill(table: Table, column: str, rng: random.Random) -> Corruption | None:
    """Inverse of forward-fill imputation: blank out repeated values.

    A cell equal to the one above it can be blanked and recovered exactly, which
    is the shape of the real-world "value only written when it changes" defect.
    """
    s = table.df[column]
    if not _is_text(s) or _has_null(s) or len(s) < 2:
        return None
    vals = _values(s)
    repeats = [i for i in range(1, len(vals)) if vals[i] == vals[i - 1]]
    if not repeats:
        return None
    rng.shuffle(repeats)
    chosen = sorted(repeats[: max(1, len(repeats) // 2)])

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        col = d[column].copy()
        col.iloc[chosen] = None
        return _set_column(d, column, col)

    return Corruption(
        kind="null_ffill",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=f"MissingValueImputation({table.name}, {column!r}, mode=ffill)",
        cleaning_operator="MissingValueImputation",
        note=f"{len(chosen)} repeated value(s) blanked out",
    )


def _propose_null_placeholder(
    table: Table, column: str, rng: random.Random
) -> Corruption | None:
    """Inverse of constant imputation: drop the most frequent categorical value."""
    s = table.df[column]
    if not _is_text(s) or _has_null(s) or len(s) < 3:
        return None
    counts = s.value_counts()
    if counts.empty or int(counts.iloc[0]) < 2:
        return None
    value = str(counts.index[0])

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        col = d[column].copy()
        col[col == value] = None
        return _set_column(d, column, col)

    return Corruption(
        kind="null_placeholder",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=(
            f"MissingValueImputation({table.name}, {column!r}, mode=constant, value={value!r})"
        ),
        cleaning_operator="MissingValueImputation",
        note=f"occurrences of {value!r} replaced by NULL",
    )


# -- 2.2.1 ErrorDetection ---------------------------------------------------- #
_SENTINELS = ("N/A", "n/a", "unknown", "-", "???")


def _propose_error_sentinel(
    table: Table, column: str, rng: random.Random
) -> Corruption | None:
    """Inverse of error detection: append records holding an invalid marker."""
    df = table.df
    s = df[column]
    if len(df) == 0 or not _is_text(s) or _has_null(s):
        return None
    sentinel = rng.choice(_SENTINELS)
    if (s.astype(str) == sentinel).any():
        return None
    k = rng.randint(1, min(2, len(df)))
    picks = [rng.randrange(len(df)) for _ in range(k)]

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        rows = d.iloc[picks].copy()
        rows[column] = sentinel
        return _append_rows(d, rows)

    return Corruption(
        kind="error_sentinel",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=(
            f"ErrorDetection({table.name}, {column!r}, "
            f"lambda x: str(x) == {sentinel!r}, action=remove)"
        ),
        cleaning_operator="ErrorDetection",
        note=f"{k} record(s) with the invalid marker {sentinel!r} appended",
    )


# -- 2.2.1 OutlierDetection -------------------------------------------------- #
def _propose_outlier_rows(table: Table, column: str, rng: random.Random) -> Corruption | None:
    """Inverse of outlier removal: append records with an implausible value."""
    df = table.df
    s = df[column]
    if len(df) < 4 or not _is_numeric(s) or _has_null(s):
        return None
    numeric = pd.to_numeric(s, errors="coerce")
    spread = float(numeric.max() - numeric.min())
    magnitude = max(abs(float(numeric.max())), 1.0)
    extreme = float(numeric.max()) + 50.0 * (spread + magnitude)
    if pd.api.types.is_integer_dtype(s):
        extreme = int(round(extreme))
    pick = rng.randrange(len(df))

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        rows = d.iloc[[pick]].copy()
        rows[column] = extreme
        return _append_rows(d, rows)

    return Corruption(
        kind="outlier_row",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=(
            f"OutlierDetection({table.name}, {column!r}, action=remove, "
            f"method=iqr, threshold=1.5)"
        ),
        cleaning_operator="OutlierDetection",
        note=f"one record with {column}={extreme} appended",
    )


# -- 2.2.2 CastType ---------------------------------------------------------- #
def _propose_numeric_as_text(
    table: Table, column: str, rng: random.Random
) -> Corruption | None:
    """Inverse of CastType: store numbers as text, as a CSV export would."""
    s = table.df[column]
    if not _is_numeric(s) or _has_null(s) or len(s) == 0:
        return None
    dtype = "int" if pd.api.types.is_integer_dtype(s) else "float"

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        return _set_column(d, column, d[column].map(str).astype(object))

    return Corruption(
        kind="numeric_as_text",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=f"CastType({table.name}, {column!r}, {dtype})",
        cleaning_operator="CastType",
        note=f"numeric column stored as text (target dtype {dtype})",
    )


# -- 2.2.2 ValueTransform ---------------------------------------------------- #
def _propose_case_noise(table: Table, column: str, rng: random.Random) -> Corruption | None:
    """Inverse of case normalization: upper-case an otherwise lower-case column."""
    s = table.df[column]
    if not _is_text(s) or _has_null(s) or len(s) == 0:
        return None
    vals = [str(v) for v in _values(s)]
    if not any(v != v.upper() for v in vals) or any(v != v.lower() for v in vals):
        return None

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        return _set_column(d, column, d[column].map(lambda v: str(v).upper()))

    return Corruption(
        kind="case_noise",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=f"ValueTransform({table.name}, {column!r}, lambda x: str(x).lower())",
        cleaning_operator="ValueTransform",
        note="values upper-cased",
    )


def _propose_whitespace_noise(
    table: Table, column: str, rng: random.Random
) -> Corruption | None:
    """Inverse of whitespace normalization: pad values with stray blanks."""
    s = table.df[column]
    if not _is_text(s) or _has_null(s) or len(s) == 0:
        return None
    vals = [str(v) for v in _values(s)]
    if any(v != v.strip() or not v for v in vals):
        return None
    # The padding is drawn once, at proposal time: a corruption must be a pure
    # function of the table, otherwise re-applying it would not match the state
    # the reversibility gate accepted.
    pads = ("  ", " ", "\t")
    left, right = rng.choice(pads), rng.choice(pads)

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        return _set_column(d, column, d[column].map(lambda v: left + str(v) + right))

    return Corruption(
        kind="whitespace_noise",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=f"ValueTransform({table.name}, {column!r}, lambda x: str(x).strip())",
        cleaning_operator="ValueTransform",
        note="values padded with leading/trailing whitespace",
    )


def _propose_unit_decoration(
    table: Table, column: str, rng: random.Random
) -> Corruption | None:
    """Inverse of unit stripping: decorate values with a unit/currency marker."""
    s = table.df[column]
    if not _is_text(s) or _has_null(s) or len(s) == 0:
        return None
    marker = rng.choice(("$", "#", "~", "* "))
    vals = [str(v) for v in _values(s)]
    if any(v.startswith(marker) for v in vals):
        return None

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        return _set_column(d, column, d[column].map(lambda v: marker + str(v)))

    return Corruption(
        kind="unit_decoration",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=(
            f"ValueTransform({table.name}, {column!r}, "
            f"lambda x: str(x).removeprefix({marker!r}))"
        ),
        cleaning_operator="ValueTransform",
        note=f"values prefixed with {marker!r}",
    )


def _propose_separator_typo(
    table: Table, column: str, rng: random.Random
) -> Corruption | None:
    """Inverse of format normalization: replace the word separator by an underscore."""
    s = table.df[column]
    if not _is_text(s) or _has_null(s) or len(s) == 0:
        return None
    vals = [str(v) for v in _values(s)]
    if not any(" " in v for v in vals) or any("_" in v for v in vals):
        return None

    def corrupt(d: pd.DataFrame) -> pd.DataFrame:
        return _set_column(d, column, d[column].map(lambda v: str(v).replace(" ", "_")))

    return Corruption(
        kind="separator_typo",
        table=table.name,
        column=column,
        corrupt=corrupt,
        cleaning=(
            f"ValueTransform({table.name}, {column!r}, lambda x: str(x).replace('_', ' '))"
        ),
        cleaning_operator="ValueTransform",
        note="spaces replaced by underscores",
    )


#: The offline corruption library.  Every operator of Sec 2.2.1 and 2.2.2 is the
#: target of at least one pair, so the synthesized cleaning pipelines exercise the
#: full data-cleaning / value-normalization vocabulary.
BUILTIN_NOISE_PAIRS: tuple[NoisePair, ...] = (
    NoisePair("date_reformat", "StandardizeDatetime", _propose_date_reformat),
    NoisePair("duplicate_rows", "Deduplicate", _propose_duplicate_rows, inserts_rows=True),
    NoisePair("null_rows", "DropNA", _propose_null_rows, inserts_rows=True),
    NoisePair("null_ffill", "MissingValueImputation", _propose_null_ffill),
    NoisePair("null_placeholder", "MissingValueImputation", _propose_null_placeholder),
    NoisePair("error_sentinel", "ErrorDetection", _propose_error_sentinel, inserts_rows=True),
    NoisePair("outlier_row", "OutlierDetection", _propose_outlier_rows, inserts_rows=True),
    NoisePair("numeric_as_text", "CastType", _propose_numeric_as_text),
    NoisePair("case_noise", "ValueTransform", _propose_case_noise),
    NoisePair("whitespace_noise", "ValueTransform", _propose_whitespace_noise),
    NoisePair("unit_decoration", "ValueTransform", _propose_unit_decoration),
    NoisePair("separator_typo", "ValueTransform", _propose_separator_typo),
)


# --------------------------------------------------------------------------- #
# LLM-generated inverse transformations (the paper's method)
# --------------------------------------------------------------------------- #
_INVERSE_PROMPT = """\
You corrupt a clean table so that a data preparation agent has something to clean.

The corruption must be the exact *inverse* of one data-cleaning operator: after
applying the operator below to the corrupted column, the column must be restored
to its current values, bit for bit. Example: the inverse of date standardization
rewrites '2023-01-01' as '01/01/23', and StandardizeDatetime turns it back.

## Cleaning operator to invert
{operator}

## Table `{table}` (column to corrupt: `{column}`)
{preview}

## Current values of `{column}` (sample)
{values}

Reply with a single JSON object and nothing else:
{{"kind": "<short snake_case name for this corruption>",
  "corrupt": "lambda x: <expression corrupting ONE cell value>",
  "cleaning": "{operator_name}({table}, {column!r}, ...)"}}

Constraints:
- "corrupt" is a one-argument Python lambda applied to every cell of `{column}`.
- "cleaning" is one operator call in the syntax shown above; it must fully undo
  "corrupt" for every value listed, including edge cases.
- Use only the standard library plus `pd`, `np` and `re`.
"""

#: Cleaning operators the LLM may be asked to invert (Sec 2.2.1 + 2.2.2, minus
#: the ones whose inverse changes the row count, which a cell-wise lambda cannot
#: express).
_INVERTIBLE_OPERATORS = {
    "ValueTransform": "ValueTransform(table, column, func) — normalize every value of a column.",
    "StandardizeDatetime": (
        "StandardizeDatetime(table, column, format='%Y-%m-%d', input_format=None) — "
        "parse a column into dates and render it in a target format."
    ),
    "CastType": "CastType(table, column, dtype) — cast a column to int/float/str/bool/datetime.",
    "MissingValueImputation": (
        "MissingValueImputation(table, column, mode, value) — fill missing values; "
        "mode is one of mean/median/mode/ffill/bfill/zero/constant."
    ),
}


@dataclass
class LLMInverseProposer:
    """"we sample an operator and use an LLM to generate its inverse transformation".

    The generated inverse goes through exactly the same reversibility gate as the
    built-in pairs, so a hallucinated inverse costs one rejected attempt and
    nothing else.
    """

    llm: LLMClient
    max_tokens: int = 512
    max_values_shown: int = 12

    def propose(self, table: Table, column: str, rng: random.Random) -> Corruption | None:
        operator = rng.choice(sorted(_INVERTIBLE_OPERATORS))
        prompt = _INVERSE_PROMPT.format(
            operator=_INVERTIBLE_OPERATORS[operator],
            operator_name=operator,
            table=table.name,
            column=column,
            preview=serialize_table(table, max_rows=5, with_description=False),
            values=", ".join(repr(v) for v in table.df[column].head(self.max_values_shown)),
        )
        try:
            raw = self.llm.generate([{"role": "user", "content": prompt}], max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001 - a failed generation is just no proposal
            return None

        data = extract_json(raw.text)
        if not isinstance(data, dict):
            return None
        lambda_src = str(data.get("corrupt") or "")
        cleaning = str(data.get("cleaning") or "").strip()
        if not lambda_src or not cleaning:
            return None
        try:
            fn = compile_callable(lambda_src)
        except SandboxError:
            return None
        if not callable(fn):
            return None

        def corrupt(df: pd.DataFrame) -> pd.DataFrame:
            return _set_column(df, column, df[column].map(fn))

        return Corruption(
            kind=str(data.get("kind") or "llm_inverse"),
            table=table.name,
            column=column,
            corrupt=corrupt,
            cleaning=cleaning,
            cleaning_operator=_operator_name(cleaning) or operator,
            note="inverse transformation generated by the LLM",
        )


def _operator_name(call: str) -> str | None:
    m = re.match(r"\s*([A-Za-z_]\w*)\s*\(", call)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# The injection loop
# --------------------------------------------------------------------------- #
@dataclass
class NoiseConfig:
    """Knobs for "repeating this process produces dirty source tables"."""

    #: Number of accepted corruptions to aim for.  Together with the task pipeline
    #: this sets the total pipeline length (Table 1: Synth-Spider 1~28 operators).
    max_steps: int = 5
    #: Proposal attempts per accepted step before giving up on that step.
    max_attempts_per_step: int = 12
    seed: int = 0
    #: Corruptions that append rows.  They are always reversible when accepted,
    #: but they change table cardinality, so they can be switched off.
    allow_row_insertion: bool = True
    #: Probability of asking the LLM for the inverse instead of using a built-in
    #: pair (only when an ``LLMClient`` is supplied).
    llm_share: float = 0.5
    #: Never corrupt these ``{table: {column}}``.  Defaults to the join keys of the
    #: task pipeline: "excessive noise injection may break key fields or
    #: relationships, making it difficult to construct any feasible pipeline".
    protect: Mapping[str, Iterable[str]] | None = None


@dataclass
class NoiseResult:
    """Dirty sources plus "a matching cleaning pipeline"."""

    sources: TableSet
    #: Operator calls that turn ``sources`` back into the clean tables, in the
    #: order they must be executed (i.e. reverse order of corruption).
    cleaning_pipeline: list[str] = field(default_factory=list)
    accepted: list[Corruption] = field(default_factory=list)
    n_rejected: int = 0

    @property
    def kinds(self) -> list[str]:
        return [c.kind for c in self.accepted]


def _corruptible_columns(
    table: Table, protected: Mapping[str, Iterable[str]] | None
) -> list[str]:
    blocked = {str(c) for c in (protected or {}).get(table.name, ())}
    return [c for c in table.columns if c not in blocked]


def inject_noise(
    sources: TableSet,
    config: NoiseConfig | None = None,
    llm: LLMClient | None = None,
) -> NoiseResult:
    """Corrupt ``sources`` with reversible noise; return the dirty tables + cleaning.

    Each accepted corruption prepends its cleaning operator to the pipeline: the
    corruptions compose as ``c_n(...c_1(S))``, so the repair must run in the
    reverse order ``clean_n, ..., clean_1``.
    """
    cfg = config or NoiseConfig()
    rng = random.Random(cfg.seed)
    pairs = [p for p in BUILTIN_NOISE_PAIRS if cfg.allow_row_insertion or not p.inserts_rows]
    llm_proposer = LLMInverseProposer(llm) if llm is not None else None

    state = sources.copy()
    result = NoiseResult(sources=state)
    candidates_tables = [t.name for t in state if t.n_rows > 0 and len(t.columns) > 0]
    if not candidates_tables or not pairs:
        return result

    for _ in range(cfg.max_steps):
        accepted = False
        for _ in range(cfg.max_attempts_per_step):
            table = state[rng.choice(candidates_tables)]
            columns = _corruptible_columns(table, cfg.protect)
            if not columns:
                continue
            column = rng.choice(columns)

            use_llm = llm_proposer is not None and rng.random() < cfg.llm_share
            corruption: Corruption | None
            if use_llm and llm_proposer is not None:
                corruption = llm_proposer.propose(table, column, rng)
            else:
                corruption = rng.choice(pairs).propose(table, column, rng)
            if corruption is None:
                continue

            dirty = try_corruption(state, corruption)
            if dirty is None:
                result.n_rejected += 1
                continue
            state = dirty
            result.accepted.append(corruption)
            # Prepend: the cleaning pipeline undoes the corruptions in reverse.
            result.cleaning_pipeline.insert(0, corruption.cleaning)
            accepted = True
            break
        if not accepted:
            break

    result.sources = state
    return result


def apply_cleaning(sources: TableSet, cleaning: Sequence[str]) -> TableSet:
    """Execute a cleaning pipeline; used by tests and by the final gold check."""
    state = sources.copy()
    for call in cleaning:
        state = parse_operator_call(call).execute(state)
    return state
