"""Join must survive being applied to the same table twice (Sec 2.2, combination).

Joining one dimension table in under two roles -- ``airport`` as both origin and
destination, ``person`` as both author and reviewer -- is ordinary relational
work and appears throughout Spider.  The operator used a hardcoded ``_right``
suffix, so the second join tried to create a column name the first join had
already taken and pandas raised ``MergeError: Passing 'suffixes' which cause
duplicate columns``.  The failure surfaced as an unusable operator, not a wrong
answer, but it made a whole class of gold pipelines unexpressible.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deepprep.operators import parse_operator_call
from deepprep.types import Table, TableSet


@pytest.fixture
def flights() -> TableSet:
    """Mirrors the shape that broke: `country` lives on both airline and airport,
    so the first join already consumes the `_right` suffix for it."""
    flight = pd.DataFrame(
        {
            "flight_id": [1, 2],
            "airline_id": [1, 1],
            "origin": ["SFO", "JFK"],
            "destination": ["JFK", "ZRH"],
        }
    )
    airline = pd.DataFrame(
        {"airline_id": [1], "airline_name": ["Blue Sky"], "country": ["USA"]}
    )
    airport = pd.DataFrame(
        {
            "airport_code": ["SFO", "JFK", "ZRH"],
            "city": ["San Francisco", "New York", "Zurich"],
            "country": ["USA", "USA", "Switzerland"],
        }
    )
    return TableSet(
        [
            Table(name="flight", df=flight),
            Table(name="airline", df=airline),
            Table(name="airport", df=airport),
        ]
    )


def _run(state: TableSet, source: str) -> TableSet:
    return parse_operator_call(source).execute(state)


def test_the_same_table_can_be_joined_twice_under_two_roles(flights):
    state = _run(
        flights, "Join(flight, airline, on='airline_id', how=inner, target=with_airline)"
    )
    state = _run(
        state,
        "Join(with_airline, airport, on=('origin', 'airport_code'), how=inner, target=with_origin)",
    )
    state = _run(
        state,
        "Join(with_origin, airport, on=('destination', 'airport_code'), how=inner, target=both)",
    )
    df = state["both"].df

    assert len(df) == 2
    assert len(df.columns) == len(set(df.columns)), f"duplicate columns: {list(df.columns)}"
    # Both roles are present and distinguishable.
    cities = [c for c in df.columns if c.startswith("city")]
    assert len(cities) == 2
    assert sorted(df[cities[0]]) == ["New York", "San Francisco"]
    assert sorted(df[cities[1]]) == ["New York", "Zurich"]


def test_the_plain_right_suffix_is_still_used_when_it_is_free(flights):
    """The escalation must not churn column names in the common single-join case."""
    state = _run(
        flights,
        "Join(flight, airport, on=('origin', 'airport_code'), how=inner, target=j)",
    )
    state = _run(state, "RenameColumn(airport, {'city': 'origin'})")
    state = _run(state, "Join(j, airport, on=('destination', 'airport_code'), how=inner, target=k)")
    assert "origin_right" in state["k"].df.columns


def test_a_repeated_cross_join_also_stays_unique(flights):
    state = _run(flights, "Join(flight, airport, on=[], how=cross, target=x1)")
    state = _run(state, "Join(x1, airport, on=[], how=cross, target=x2)")
    df = state["x2"].df
    assert len(df.columns) == len(set(df.columns)), f"duplicate columns: {list(df.columns)}"
    assert len(df) == 2 * 3 * 3


def test_three_roles_of_one_table_stay_distinct(flights):
    """Two escalations deep: the suffix counter must keep going."""
    state = _run(flights, "AddNewColumn(flight, via, lambda r: 'ZRH')")
    state = _run(
        state, "Join(flight, airport, on=('origin', 'airport_code'), how=inner, target=j1)"
    )
    state = _run(
        state, "Join(j1, airport, on=('destination', 'airport_code'), how=inner, target=j2)"
    )
    state = _run(state, "Join(j2, airport, on=('via', 'airport_code'), how=inner, target=j3)")
    df = state["j3"].df
    assert len(df.columns) == len(set(df.columns)), f"duplicate columns: {list(df.columns)}"
    assert len([c for c in df.columns if c.startswith("city")]) == 3
