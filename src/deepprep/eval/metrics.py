"""Evaluation metrics (paper Sec 6.1, "Evaluation Metrics").

    "(1) **Exact Match Accuracy (Acc.)** measures the percentage of test cases
     where the produced table T_hat matches the ground-truth table T*. We follow
     the table matching method in [12], which is *invariant to row and column
     permutations* but requires exact cell-value equality.

     (2) **Completion Rate (Comp.)** measures the percentage of test cases that
     produce a final output table. A case is counted as incomplete if execution
     exceeds the maximum interaction limit, triggers runtime errors, or produces
     an empty result.

     (3) **Inference Cost (Cost)** measures the average monetary cost per case.
     For closed-source LLMs, we compute API cost using official pricing with
     caching enabled. For open-source models, we measure per-case end-to-end
     runtime under parallel requests and convert it to cost using the hourly
     price of the GPU instance. Specifically, the cost of case i is computed as
     c_i = p_gpu * t_i / 3600 ... We use an A800-80G GPU priced at $0.91/hour."
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "A800_USD_PER_HOUR",
    "API_PRICING",
    "ApiPricing",
    "MatchOptions",
    "api_cost",
    "gpu_cost",
    "partial_similarity",
    "table_match",
]


# --------------------------------------------------------------------------- #
# (1) Exact match
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatchOptions:
    """Knobs for the permutation-invariant table comparison.

    ``float_tol`` exists because "exact cell-value equality" cannot be taken
    literally for floating point: two pipelines that compute the same mean in a
    different operator order can differ in the last ULP.  Quantizing to
    ``float_tol`` makes the comparison order-independent without accepting
    genuinely different numbers.
    """

    float_tol: float = 1e-6
    strip_strings: bool = True
    case_sensitive: bool = True
    #: Require the produced column *names* to equal the gold ones.  The paper's
    #: metric is permutation-invariant over columns; when names disagree entirely
    #: we fall back to matching columns by their value signature.
    require_column_names: bool = False


_NULL = "\x00<NULL>"


def _normalize_cell(v: Any, opt: MatchOptions) -> Hashable:
    """Map a cell to a hashable canonical form."""
    if v is None or v is pd.NaT or v is pd.NA:
        return _NULL
    if isinstance(v, float) and np.isnan(v):
        return _NULL
    try:
        if pd.isna(v):
            return _NULL
    except (TypeError, ValueError):
        pass

    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return float(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if opt.float_tol > 0:
            # Quantize so that 8.699999999999999 and 8.7 collapse together.
            return round(f / opt.float_tol) * opt.float_tol
        return f
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(v))
    if isinstance(v, (list, tuple, set)):
        return tuple(_normalize_cell(x, opt) for x in v)

    s = str(v)
    if opt.strip_strings:
        s = s.strip()
    if not opt.case_sensitive:
        s = s.lower()
    # A numeric-looking string must compare equal to the number it denotes,
    # otherwise a CastType left undone would be scored as a content error rather
    # than the type error it is.
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    if opt.float_tol > 0:
        return round(f / opt.float_tol) * opt.float_tol
    return f


def _normalize_frame(df: pd.DataFrame, opt: MatchOptions) -> list[list[Hashable]]:
    """Column-major normalized values."""
    return [[_normalize_cell(v, opt) for v in df[c].tolist()] for c in df.columns]


def _rows_equal(a_cols: list[list[Hashable]], b_cols: list[list[Hashable]]) -> bool:
    """Row-permutation-invariant comparison of two aligned column lists."""
    a_rows = Counter(zip(*a_cols, strict=True)) if a_cols else Counter()
    b_rows = Counter(zip(*b_cols, strict=True)) if b_cols else Counter()
    return a_rows == b_rows


def table_match(
    pred: pd.DataFrame | None,
    gold: pd.DataFrame | None,
    options: MatchOptions | None = None,
) -> bool:
    """Row- and column-permutation-invariant exact match (Sec 6.1, metric 1)."""
    opt = options or MatchOptions()
    if pred is None or gold is None:
        return False
    if pred.shape != gold.shape:
        return False
    if pred.shape[1] == 0:
        return pred.shape[0] == gold.shape[0]

    p_names = [str(c) for c in pred.columns]
    g_names = [str(c) for c in gold.columns]

    p_cols = _normalize_frame(pred, opt)
    g_cols = _normalize_frame(gold, opt)

    # Preferred alignment: by column name (a permutation of the gold order).
    if set(p_names) == set(g_names) and len(set(p_names)) == len(p_names):
        index = {n: i for i, n in enumerate(p_names)}
        aligned = [p_cols[index[n]] for n in g_names]
        return _rows_equal(aligned, g_cols)

    if opt.require_column_names:
        return False

    # Fallback: match columns by their value signature.  Permutation invariance
    # over columns is only decidable this way when the names disagree.
    return _match_by_signature(p_cols, g_cols)


def _match_by_signature(
    p_cols: list[list[Hashable]], g_cols: list[list[Hashable]]
) -> bool:
    """Find a column bijection making the row multisets equal.

    Candidate pairings are restricted to columns with an identical value
    multiset, which collapses the search from ``n!`` to the product of the
    duplicate-group sizes; a bounded backtracking search then verifies the rows.
    """
    n = len(g_cols)
    if len(p_cols) != n:
        return False

    p_sig = [Counter(c) for c in p_cols]
    g_sig = [Counter(c) for c in g_cols]

    candidates: list[list[int]] = []
    for gs in g_sig:
        cand = [i for i, ps in enumerate(p_sig) if ps == gs]
        if not cand:
            return False
        candidates.append(cand)

    # Try the most constrained gold column first.
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    assignment: list[int | None] = [None] * n
    used: set[int] = set()
    steps = [0]
    MAX_STEPS = 200_000

    def backtrack(k: int) -> bool:
        steps[0] += 1
        if steps[0] > MAX_STEPS:  # pragma: no cover - pathological width
            return False
        if k == n:
            aligned = [p_cols[assignment[i]] for i in range(n)]  # type: ignore[index]
            return _rows_equal(aligned, g_cols)
        gi = order[k]
        for pi in candidates[gi]:
            if pi in used:
                continue
            assignment[gi] = pi
            used.add(pi)
            if backtrack(k + 1):
                return True
            used.discard(pi)
            assignment[gi] = None
        return False

    return backtrack(0)


# --------------------------------------------------------------------------- #
# Partial similarity (used by the RL partial reward, Eq. 8)
# --------------------------------------------------------------------------- #
def partial_similarity(
    pred: pd.DataFrame | None,
    gold: pd.DataFrame | None,
    options: MatchOptions | None = None,
) -> dict[str, float]:
    """Schema / shape / content similarity, Eq. (8) of the paper.

        S_sch = |C_That ∩ C_T*| / |C_That ∪ C_T*|
        S_shp = exp( -| |D_hat| - |D*| | / |D*| )
        S_cnt = (1/|C_m|) * sum_{c in C_m} ( sum_i 1[D_hat[c]_i = D*[c]_i] / max(|D_hat|,|D*|) )

    where ``C_m = C_That ∩ C_T*`` is the set of matched columns.

    Note the cell term is *positional* (index ``i``), unlike the exact-match
    metric which is permutation-invariant.  To keep the signal meaningful under
    row reordering we compare each matched column after sorting the rows by the
    matched columns — this preserves the formula's value when the row order
    already agrees, and stops a pure permutation from scoring ~0.
    """
    opt = options or MatchOptions()
    out = {"schema": 0.0, "shape": 0.0, "content": 0.0, "partial": 0.0}
    if pred is None or gold is None:
        return out

    p_names = {str(c) for c in pred.columns}
    g_names = {str(c) for c in gold.columns}

    # S_sch -- Jaccard over column names.
    union = p_names | g_names
    out["schema"] = (len(p_names & g_names) / len(union)) if union else 0.0

    # S_shp -- exponential penalty on the row-count deviation.
    n_p, n_g = len(pred), len(gold)
    if n_g == 0:
        out["shape"] = 1.0 if n_p == 0 else 0.0
    else:
        out["shape"] = float(np.exp(-abs(n_p - n_g) / n_g))

    # S_cnt -- cell agreement over the matched columns.
    matched = sorted(p_names & g_names)
    if matched and max(n_p, n_g) > 0:
        p_sorted = _sort_for_alignment(pred, matched, opt)
        g_sorted = _sort_for_alignment(gold, matched, opt)
        denom = max(n_p, n_g)
        total = 0.0
        for c in matched:
            pc = [_normalize_cell(v, opt) for v in p_sorted[c].tolist()]
            gc = [_normalize_cell(v, opt) for v in g_sorted[c].tolist()]
            # Row counts may differ; the denominator is max(|D_hat|,|D*|), so pairing
            # only the overlapping prefix is what Eq. (8) intends.
            hits = sum(1 for a, b in zip(pc, gc, strict=False) if a == b)
            total += hits / denom
        out["content"] = total / len(matched)

    out["partial"] = (out["schema"] + out["shape"] + out["content"]) / 3.0
    return out


def _sort_for_alignment(
    df: pd.DataFrame, cols: list[str], opt: MatchOptions
) -> pd.DataFrame:
    """Canonically order rows so the positional cell comparison is stable."""
    if len(df) == 0:
        return df
    key = pd.DataFrame(
        {c: [str(_normalize_cell(v, opt)) for v in df[c].tolist()] for c in cols},
        index=df.index,
    )
    order = key.sort_values(by=cols, kind="mergesort").index
    return df.loc[order].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# (3) Inference cost
# --------------------------------------------------------------------------- #
#: "We use an A800-80G GPU priced at $0.91/hour (https://cloud.vast.ai/)." (Sec 6.1)
A800_USD_PER_HOUR = 0.91


@dataclass(frozen=True)
class ApiPricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    #: Cached-input rate. The paper computes API cost "with caching enabled".
    cached_input_per_mtok: float | None = None


#: Published list prices for the backbones the paper evaluates, plus common
#: alternatives.  These change over time — override via ``api_cost(pricing=...)``.
API_PRICING: dict[str, ApiPricing] = {
    # Sec 6.1: "GPT-5 (i.e. gpt-5-mini-2025-08-07)"
    "gpt-5-mini-2025-08-07": ApiPricing(0.25, 2.00, 0.025),
    "gpt-5-mini": ApiPricing(0.25, 2.00, 0.025),
    "gpt-5": ApiPricing(1.25, 10.00, 0.125),
    # Sec 6.1: "Claude-4 (i.e. claude-sonnet-4-20250514)"
    "claude-sonnet-4-20250514": ApiPricing(3.00, 15.00, 0.30),
    "claude-sonnet-4-5": ApiPricing(3.00, 15.00, 0.30),
    "gpt-4o": ApiPricing(2.50, 10.00, 1.25),
    "gpt-4o-mini": ApiPricing(0.15, 0.60, 0.075),
    "deepseek-chat": ApiPricing(0.27, 1.10, 0.07),
    "deepseek-reasoner": ApiPricing(0.55, 2.19, 0.14),
}


def api_cost(usage: Any, model: str | None = None, pricing: ApiPricing | None = None) -> float:
    """Monetary cost of one case for a closed-source backbone, caching enabled."""
    if pricing is None:
        pricing = API_PRICING.get(model or "", None)
    if pricing is None:
        return 0.0
    prompt = float(getattr(usage, "prompt_tokens", 0) or 0)
    completion = float(getattr(usage, "completion_tokens", 0) or 0)
    cached = float(getattr(usage, "cached_prompt_tokens", 0) or 0)
    cached = min(cached, prompt)
    fresh = prompt - cached
    rate_cached = (
        pricing.cached_input_per_mtok
        if pricing.cached_input_per_mtok is not None
        else pricing.input_per_mtok
    )
    return (
        fresh * pricing.input_per_mtok
        + cached * rate_cached
        + completion * pricing.output_per_mtok
    ) / 1e6


def gpu_cost(wall_time_s: float, usd_per_hour: float = A800_USD_PER_HOUR) -> float:
    """``c_i = p_gpu * t_i / 3600`` — cost for a self-hosted open-source backbone."""
    return usd_per_hour * float(wall_time_s) / 3600.0
