"""Regression tests for Sec 6.1 metrics and Eq. (8).

Each test here pins a defect found in adversarial review and since fixed, so a
regression fails loudly.  Docstrings quote the paper clause the behaviour follows
from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepprep.agent.agent import SolveResult
from deepprep.eval import MatchOptions, evaluate, partial_similarity, table_match
from deepprep.types import ADPTask, ColumnSpec, Table, TableSchema, TableSet


# --------------------------------------------------------------------------- #
# BUG 1 (CRITICAL): table_match raises on inf / "nan" / "inf" cells.
# --------------------------------------------------------------------------- #
def test_non_finite_cells_compare_instead_of_raising():
    """Sec 6.1: exact match "requires exact cell-value equality".

    Quantizing with ``round(f / float_tol)`` returns an int and cannot convert
    NaN/inf, so an unguarded cell used to raise instead of returning a bool.
    """
    assert table_match(pd.DataFrame({"a": [np.inf]}), pd.DataFrame({"a": [np.inf]}))
    assert not table_match(pd.DataFrame({"a": [np.inf]}), pd.DataFrame({"a": [-np.inf]}))
    assert not table_match(pd.DataFrame({"a": [np.inf]}), pd.DataFrame({"a": [1.0]}))
    # The *string* "nan"/"inf" survives the pd.isna() guard and reaches float().
    assert table_match(pd.DataFrame({"a": ["nan"]}), pd.DataFrame({"a": ["nan"]}))
    assert table_match(pd.DataFrame({"a": ["inf"]}), pd.DataFrame({"a": ["inf"]}))


def test_one_infinite_cell_does_not_abort_the_evaluation_run():
    """``table_match`` is called outside ``run_one``'s try/except (which wraps
    only ``solver.solve``), so a raise here would kill an entire benchmark."""

    class Solver:
        def solve(self, task):
            r = SolveResult(task_id=task.task_id)
            r.table = pd.DataFrame({"a": [np.inf]})
            r.stop_reason = "answered"
            return r

    task = ADPTask(
        task_id="t1",
        sources=TableSet([Table("s", pd.DataFrame({"a": [1]}))]),
        target_schema=TableSchema("x", [ColumnSpec("a")]),
        target_table=pd.DataFrame({"a": [np.inf]}),
    )
    report = evaluate(Solver(), [task], max_workers=1, verbose=False)
    assert report.n_cases == 1
    assert report.accuracy == 100.0  # inf == inf


# --------------------------------------------------------------------------- #
# BUG 2 (MAJOR): the signature fallback accepts tables with WRONG column names.
# --------------------------------------------------------------------------- #
def test_exact_match_rejects_wrong_column_names():
    """Sec 6.1: the metric is "invariant to row and column *permutations*".

    A rename is not a permutation, and the task is to produce a table conforming
    to Sigma*, whose names are given.  Accepting arbitrary names would also make
    the metric blind to the exact reward hack Sec 5.2 names -- "renaming columns
    without performing the required data cleaning".
    """
    gold = pd.DataFrame({"director_name": ["a", "b"], "genre": ["Sci-Fi", "Action"]})
    pred = pd.DataFrame({"col1": ["a", "b"], "col2": ["Sci-Fi", "Action"]})
    assert not table_match(pred, gold)

    # A single misnamed column is enough to fail.
    gold2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    pred2 = pd.DataFrame({"a": [1, 2], "WRONG": [3, 4]})
    assert not table_match(pred2, gold2)

    # The value-signature fallback remains available, but only on request.
    assert table_match(pred, gold, MatchOptions(require_column_names=False))


def test_exact_match_rejects_a_semantically_wrong_column_permutation():
    """With names required, two columns whose contents happen to be each other's
    permutation can no longer be swapped silently."""
    gold = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
    pred = pd.DataFrame({"p": [2, 1], "q": [1, 2]})  # p<->y, q<->x
    assert not table_match(pred, gold)


# --------------------------------------------------------------------------- #
# BUG 3 (MAJOR): numeric-looking strings are coerced, breaking "exact equality".
# --------------------------------------------------------------------------- #
def test_numeric_string_coercion_is_narrow():
    """Sec 6.1 "requires exact cell-value equality".

    A numeric string must equal the number it denotes, so an undone CastType is
    scored as the type error it is -- but the coercion has to be narrower than
    ``float()``, which happily eats identifiers.
    """
    # Intended: a plain numeric rendering still matches its number.
    assert table_match(pd.DataFrame({"a": ["123"]}), pd.DataFrame({"a": [123]}))
    assert table_match(pd.DataFrame({"a": ["8.7"]}), pd.DataFrame({"a": [8.7]}))
    # Rejected: zero-padded identifiers, postcodes, phone numbers.
    assert not table_match(pd.DataFrame({"a": ["0123"]}), pd.DataFrame({"a": [123]}))
    # Rejected: Python's underscore separators are not a data format.
    assert not table_match(pd.DataFrame({"a": ["1_0"]}), pd.DataFrame({"a": [10]}))


# --------------------------------------------------------------------------- #
# BUG 4 (MINOR): duplicate column labels crash both metrics.
# --------------------------------------------------------------------------- #
def test_duplicate_column_labels_do_not_crash():
    """A malformed prediction should score 0, not abort the run."""
    df = pd.DataFrame([[1, 1], [2, 2]], columns=["a", "a"])
    assert isinstance(table_match(df, df), bool)
    assert isinstance(partial_similarity(df, df)["partial"], float)


# --------------------------------------------------------------------------- #
# BUG 5 (MAJOR): partial_similarity crashes on non-string column labels.
# --------------------------------------------------------------------------- #
def test_partial_similarity_handles_integer_column_labels():
    """``matched`` is built from ``str(c)`` but then used to index the frame.

    ``table_match`` normalizes with ``str(c)`` throughout and works fine, so the
    two metrics disagree on the same input.  A gold table produced by a raw
    ``pivot`` on an integer ``year`` column (the paper's Figure-2 example
    produces ``2019``/``2020`` columns) hits this.
    """
    gold = pd.DataFrame({"director": ["a", "b"], 2019: [7.6, 8.9]})
    pred = pd.DataFrame({"director": ["a", "b"], "2019": [7.6, 8.9]})
    assert table_match(pred, gold)
    # The two metrics must agree on the same input.
    assert partial_similarity(pred, gold)["partial"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Eq. (8) — the formulas themselves are correct; documented for the record.
# --------------------------------------------------------------------------- #
def test_eq8_formulas_match_the_paper():
    """S_sch / S_shp / S_cnt, incl. denominator max(|D_hat|,|D*|) and C_m."""
    gold = pd.DataFrame({"k": [1, 2], "g": ["x", "y"]})
    pred = pd.DataFrame({"k": [1, 2, 3], "p": ["x", "y", "z"]})
    s = partial_similarity(pred, gold)
    # C_That = {k,p}, C_T* = {k,g}  ->  1/3
    assert s["schema"] == pytest.approx(1 / 3)
    # exp(-| |D_hat| - |D*| | / |D*|) = exp(-1/2)
    assert s["shape"] == pytest.approx(np.exp(-0.5))
    # C_m = {k}; 2 of the overlapping rows agree; denominator max(3,2) = 3
    assert s["content"] == pytest.approx(2 / 3)
    assert s["partial"] == pytest.approx((1 / 3 + np.exp(-0.5) + 2 / 3) / 3)


def test_deviation_eq8_content_is_not_positional():
    """Eq. (8) indexes cells positionally (``D_hat[c]_i == D*[c]_i``).

    The implementation canonically *sorts* both frames by the matched columns
    first, so a pure row permutation scores 1.0 where the paper's formula would
    score ~0.  Deliberate and documented, but it is a numeric deviation from
    Eq. (8) that would change every reported R_part.
    """
    gold = pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]})
    pred = gold.iloc[::-1].reset_index(drop=True)
    assert partial_similarity(pred, gold)["content"] == pytest.approx(1.0)
