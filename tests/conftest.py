"""Shared fixtures.

Everything here is offline: no network, no API key, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "examples" / "movies_demo"
sys.path.insert(0, str(DEMO_DIR))

from deepprep.types import ADPTask, ColumnSpec, Table, TableSchema, TableSet  # noqa: E402


@pytest.fixture
def demo_task() -> ADPTask:
    """The paper's Figure-2 running example."""
    from build_task import build_task  # type: ignore

    return build_task()


@pytest.fixture
def figure4_trajectory() -> list[str]:
    """The canned agent responses reproducing Figure 4."""
    from scripted_trajectory import FIGURE_4_TRAJECTORY  # type: ignore

    return list(FIGURE_4_TRAJECTORY)


@pytest.fixture
def simple_tables() -> TableSet:
    """A small, deliberately messy table set for operator unit tests."""
    people = pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 4],
            "name": [" Alice ", "BOB", "BOB", "carol", "dave"],
            "dept": ["eng", "eng", "eng", "sales", None],
            "salary": [100.0, 200.0, 200.0, None, 5000.0],
            "joined": ["2021-01-02", "01/03/2021", "01/03/2021", "2021-05-06", "2021-07-08"],
            "tags": ["a,b", "c", "c", "b,c", "a"],
        }
    )
    depts = pd.DataFrame({"code": ["eng", "sales", "hr"], "label": ["Engineering", "Sales", "HR"]})
    return TableSet(
        [
            Table("people", people, TableSchema("Employee records.", [
                ColumnSpec("id", "int64", "Employee id."),
                ColumnSpec("name", "object", "Name, inconsistently formatted."),
                ColumnSpec("dept", "object", "Department code."),
                ColumnSpec("salary", "float64", "Annual salary."),
                ColumnSpec("joined", "object", "Join date, mixed formats."),
                ColumnSpec("tags", "object", "Comma-separated tags."),
            ])),
            Table("depts", depts),
        ]
    )
