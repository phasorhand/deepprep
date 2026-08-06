"""Regression tests for Sec 4.1/4.2 tree semantics and Sec 2.2 operators.

Each test pins a defect found in adversarial review and since fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deepprep.env import Environment
from deepprep.operators import parse_operator_call
from deepprep.tree import ReasoningTree
from deepprep.types import Table, TableSet


def _run(src: str, ts: TableSet) -> TableSet:
    return parse_operator_call(src).execute(ts)


# --------------------------------------------------------------------------- #
# BUG A (CRITICAL): prefix matching silently resolves to the WRONG node.
# --------------------------------------------------------------------------- #
def test_coarse_prefix_match_never_resolves_silently():
    """Sec 4.2 <expand>: "the parent node is specified using a prefix-matching
    constraint: the agent must reference the full operator sequence from the root
    to that node.  This constraint *avoids ambiguous* ... node references."

    The coarse fallback keys on operator name plus target table only, so it can
    land on a different operator instance.  It stays available -- it recovers a
    turn when the agent reformats an operator -- but it must never resolve
    *silently*, or the agent reasons about a state it did not name.
    """
    st = TableSet([Table("ratings", pd.DataFrame({"movie": ["a"], "rating": [8.2], "values": ["8.2 (2022)"]}))])
    tree = ReasoningTree(st)
    op = parse_operator_call("SelectColumn(ratings, [movie, rating])")
    tree.add_state(tree.root, op, op.execute(st.copy()))

    # The agent references a DIFFERENT SelectColumn (Figure 4's corrected one).
    node, warning = tree.resolve_parent("SelectColumn(ratings, [movie, values])")
    assert node.id == "n1"
    assert warning is not None and "InexactPrefix" in warning
    # The warning has to name both what was asked for and what was resolved.
    assert "movie, values" in warning and "movie, rating" in warning


def test_answer_never_reuses_a_node_built_by_a_different_operator():
    """Sec 4.2: "The root-to-leaf operator path associated with that node is
    extracted as the final pipeline P*".

    Answer resolution must match exactly.  Reusing a materialized node under a
    fuzzy match would substitute a state built by a different operator, yielding
    a wrong T_hat while reporting a pipeline the agent never wrote.  A
    non-matching operator is simply re-executed instead.
    """
    st = TableSet([Table("ratings", pd.DataFrame({"movie": ["a", "b"], "rating": [8.2, 9.0], "values": ["x", "y"]}))])
    tree = ReasoningTree(st)
    env = Environment(st)
    res = env.execute(tree.root.state, "SelectColumn(ratings,[movie,values])")
    assert res.error is None
    tree.add_state(tree.root, res.applied[0], res.states[0])

    node, remaining = tree.resolve_longest_prefix(
        ["SelectColumn(ratings, [movie, rating])", "Terminate([ratings])"]
    )
    # No reuse: resolution stops at the root and the whole answer is re-executed.
    assert node.id == "n0"
    assert remaining == ["SelectColumn(ratings, [movie, rating])", "Terminate([ratings])"]

    # An exactly-matching prefix IS still reused -- that is the point of the tree.
    node, remaining = tree.resolve_longest_prefix(
        ["SelectColumn(ratings,[movie,values])", "Terminate([ratings])"]
    )
    assert node.id == "n1"
    assert remaining == ["Terminate([ratings])"]


# --------------------------------------------------------------------------- #
# VERIFIED CORRECT: node states never alias each other.
# --------------------------------------------------------------------------- #
def test_node_states_are_independent():
    """Sec 4.1: every node is a *materialized* state.  Mutating one must not
    change another, or backtracking would be meaningless."""
    st = TableSet([Table("t", pd.DataFrame({"a": [1, 2, 3]})), Table("u", pd.DataFrame({"b": [9]}))])
    env = Environment(st)
    tree = ReasoningTree(env.initial_state())
    res = env.execute(tree.root.state, "Filter(t, lambda r: r['a'] > 1)\nDeduplicate(u)")
    n1 = tree.add_state(tree.root, res.applied[0], res.states[0])
    n2 = tree.add_state(n1, res.applied[1], res.states[1])

    n2.state["u"].df.loc[0, "b"] = 999          # 'u' is untouched by both operators
    n2.state["t"].df.loc[0, "a"] = -1
    assert n1.state["u"].df["b"].tolist() == [9]
    assert tree.root.state["u"].df["b"].tolist() == [9]
    assert n1.state["t"].df["a"].tolist() == [2, 3]
    assert tree.root.state["t"].df["a"].tolist() == [1, 2, 3]
    # and the task's own source tables are untouched
    assert st["t"].df["a"].tolist() == [1, 2, 3]


# --------------------------------------------------------------------------- #
# BUG B (CRITICAL): MissingValueImputation rewrites NON-missing values.
# --------------------------------------------------------------------------- #
def test_missing_value_imputation_only_writes_missing_cells():
    """Sec 2.2.1: "MissingValueImputation(table, column, mode) *imputes missing
    values* in a specified column using a statistical strategy."

    A value that merely fails to parse as a number is data, not a hole; coercing
    the whole column would turn it into NaN and then overwrite it with the mean.
    """
    ts = TableSet([Table("t", pd.DataFrame({"v": ["abc", 1, 2, None]}))])
    out = _run("MissingValueImputation(t, v, mean)", ts)
    assert out["t"].df["v"].tolist() == ["abc", 1, 2, 1.5]

    ts = TableSet([Table("t", pd.DataFrame({"v": ["abc", "1", None]}))])
    out = _run("MissingValueImputation(t, v, zero)", ts)
    assert out["t"].df["v"].tolist() == ["abc", "1", 0]


# --------------------------------------------------------------------------- #
# BUG C (CRITICAL): Join stringifies keys on any dtype mismatch.
# --------------------------------------------------------------------------- #
def test_join_aligns_int_and_float_keys_numerically():
    """Sec 2.2.6: "Join(left, right, on, how) combines two tables based on
    specified key columns".

    Casting both keys to str on a dtype mismatch makes this WORSE, not better:
    int64 101 renders "101" but float64 101.0 renders "101.0", so the join matches
    nothing.  A nullable foreign key read back as float is extremely common.
    """
    left = pd.DataFrame({"dir_id": [101, 102], "title": ["a", "b"]})
    right = pd.DataFrame({"id": [101.0, 102.0], "name": ["S", "J"]})
    assert len(left.merge(right, left_on="dir_id", right_on="id")) == 2  # pandas is fine

    ts = TableSet([Table("movies", left), Table("directors", right)])
    out = _run("Join(movies, directors, on=(dir_id, id), how=inner)", ts)
    assert out["movies_directors_join"].df["name"].tolist() == ["S", "J"]

    # A null in the float key must not break the surviving matches either.
    right2 = pd.DataFrame({"id": [101.0, 102.0, np.nan], "name": ["S", "J", "X"]})
    ts = TableSet([Table("movies", left), Table("directors", right2)])
    out = _run("Join(movies, directors, on=(dir_id, id), how=left)", ts)
    assert out["movies_directors_join"].df["name"].tolist() == ["S", "J"]

    # A numeric key against its own text rendering joins without a spurious ".0".
    ts = TableSet([
        Table("a", pd.DataFrame({"k": [101, 102]})),
        Table("b", pd.DataFrame({"k": ["101", "102"], "v": ["x", "y"]})),
    ])
    out = _run("Join(a, b, on=k, how=inner, target=j)", ts)
    assert out["j"].df["v"].tolist() == ["x", "y"]


# --------------------------------------------------------------------------- #
# BUG D (MINOR): ErrorDetection *removes* by default.
# --------------------------------------------------------------------------- #
def test_error_detection_identifies_rather_than_removes_by_default():
    """Sec 2.2.1: "ErrorDetection(table, column, func) *identifies* invalid
    records in a specified column based on a user-defined function."

    The paper gives ErrorDetection no ``action`` parameter (unlike
    OutlierDetection, which explicitly has one), so the 3-argument call must not
    be destructive.  ``action`` remains available as an extension.
    """
    ts = TableSet([Table("t", pd.DataFrame({"v": [1, -1, 3]}))])
    out = _run("ErrorDetection(t, v, lambda x: x < 0)", ts)
    assert out["t"].df["v"].tolist() == [1, -1, 3]        # nothing destroyed
    assert out["t"].df["v_is_error"].tolist() == [False, True, False]

    out = _run("ErrorDetection(t, v, lambda x: x < 0, remove)", ts)
    assert out["t"].df["v"].tolist() == [1, 3]


# --------------------------------------------------------------------------- #
# BUG E (MINOR): Explode splits plain strings by default.
# --------------------------------------------------------------------------- #
def test_explode_does_not_split_plain_strings_by_default():
    """Sec 2.2.7: "Explode(table, column) expands a column containing
    *list-valued* entries into multiple rows."

    A ``sep`` defaulting to "," would silently shred a scalar text column, so
    string splitting is opt-in; genuine list values always explode.
    """
    ts = TableSet([Table("t", pd.DataFrame({"name": ["Smith, John"], "id": [1]}))])
    assert _run("Explode(t, name)", ts)["t"].df["name"].tolist() == ["Smith, John"]

    # Opt in, as the paper's Figure-2 genres column requires.
    out = _run("Explode(t, name, sep=',')", ts)
    assert out["t"].df["name"].tolist() == ["Smith", "John"]

    # A genuine list-valued column needs no separator.
    ts = TableSet([Table("u", pd.DataFrame({"g": [["Sci-Fi", "Action"]]}))])
    assert _run("Explode(u, g)", ts)["u"].df["g"].tolist() == ["Sci-Fi", "Action"]
