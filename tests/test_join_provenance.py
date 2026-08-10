"""Column provenance across a Join in the SQL -> pipeline translator (Sec 5.3).

The translator tracks, for every column of the working table, which source table
and original column it came from, so that ``d.dept_name`` in the SELECT list can
be resolved to a concrete column after an arbitrary chain of joins.

It used to reconstruct that map positionally, assuming the merged frame is
exactly ``left columns ++ right columns``.  The Join operator merges keys that
share a name, so the right side is one column short and every right-hand
provenance entry shifts by one: ``d.dept_name`` silently resolved to
``building``.  The translation still executed, so nothing raised -- it just
produced a *different table* and got dropped by the verifier, which is why
Spider-style joins yielded almost no synthesized tasks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deepprep.synthesis.pipeline_search import execute_pipeline, translate_sql
from deepprep.types import Table, TableSet


@pytest.fixture
def university() -> TableSet:
    instructor = pd.DataFrame(
        {
            "inst_id": [1, 2, 3],
            "name": ["Ada", "Alan", "Emmy"],
            "dept_id": [1, 1, 2],
            "salary": [145000.0, 162000.0, 138000.0],
        }
    )
    department = pd.DataFrame(
        {
            "dept_id": [1, 2],
            "dept_name": ["Computer Science", "Mathematics"],
            "building": ["Turing Hall", "Euler Hall"],
            "budget": [1850000.0, 920000.0],
        }
    )
    return TableSet([Table(name="instructor", df=instructor), Table(name="department", df=department)])


def _translate_and_run(sql: str, sources: TableSet, target: pd.DataFrame) -> pd.DataFrame:
    result = translate_sql(sql, sources, target)
    produced, error = execute_pipeline(result.candidates[0], sources)
    assert error is None, error
    assert produced is not None
    return produced


def test_shared_key_name_does_not_shift_right_hand_provenance(university):
    """`ON i.dept_id = d.dept_id` coalesces one column; the map must account for it."""
    sql = (
        "SELECT i.name, d.dept_name, i.salary FROM instructor i "
        "JOIN department d ON i.dept_id = d.dept_id"
    )
    target = pd.DataFrame(
        {
            "name": ["Ada", "Alan", "Emmy"],
            "dept_name": ["Computer Science", "Computer Science", "Mathematics"],
            "salary": [145000.0, 162000.0, 138000.0],
        }
    )
    produced = _translate_and_run(sql, university, target)
    pd.testing.assert_frame_equal(produced.reset_index(drop=True), target)


def test_distinct_key_names_keep_both_columns(university):
    """No coalescing here -- the map must not over-correct."""
    renamed = university.replace(
        Table(name="department", df=university["department"].df.rename(columns={"dept_id": "id"}))
    )
    sql = (
        "SELECT i.name, d.building, d.budget FROM instructor i "
        "JOIN department d ON i.dept_id = d.id"
    )
    target = pd.DataFrame(
        {
            "name": ["Ada", "Alan", "Emmy"],
            "building": ["Turing Hall", "Turing Hall", "Euler Hall"],
            "budget": [1850000.0, 1850000.0, 920000.0],
        }
    )
    produced = _translate_and_run(sql, renamed, target)
    pd.testing.assert_frame_equal(produced.reset_index(drop=True), target)


def test_a_colliding_non_key_column_is_suffixed_not_shifted(university):
    """The right frame's `name` becomes `name_right`; provenance must follow it.

    `d.name` must resolve to the department's name, not the instructor's, and
    not -- as the positional map had it -- to `building`.
    """
    collided = university.replace(
        Table(
            name="department",
            df=university["department"].df.rename(columns={"dept_name": "name"}),
        )
    )
    sql = (
        "SELECT d.name, d.building FROM instructor i "
        "JOIN department d ON i.dept_id = d.dept_id"
    )
    target = pd.DataFrame(
        {
            "name": ["Computer Science", "Computer Science", "Mathematics"],
            "building": ["Turing Hall", "Turing Hall", "Euler Hall"],
        }
    )
    produced = _translate_and_run(sql, collided, target)
    pd.testing.assert_frame_equal(produced.reset_index(drop=True), target)


def test_three_way_join_provenance_survives_two_coalesced_keys():
    """Errors of this kind compound: two joins shift the last table by two."""
    a = pd.DataFrame({"k": [1, 2], "a1": ["a", "b"], "a2": [10, 20]})
    b = pd.DataFrame({"k": [1, 2], "b1": ["x", "y"], "b2": [30, 40]})
    c = pd.DataFrame({"k": [1, 2], "c1": ["p", "q"], "c2": [50, 60]})
    sources = TableSet(
        [Table(name="a", df=a), Table(name="b", df=b), Table(name="c", df=c)]
    )
    sql = "SELECT ta.a1, tb.b1, tc.c1 FROM a ta JOIN b tb ON ta.k = tb.k JOIN c tc ON ta.k = tc.k"
    target = pd.DataFrame({"a1": ["a", "b"], "b1": ["x", "y"], "c1": ["p", "q"]})
    produced = _translate_and_run(sql, sources, target)
    pd.testing.assert_frame_equal(produced.reset_index(drop=True), target)
