"""Operator space and call parser (paper Sec 2.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepprep.operators import (
    CATEGORY_ORDER,
    OPERATORS,
    OperatorError,
    ParseError,
    operator_manual,
    operators_by_category,
    parse_operator_call,
    parse_pipeline,
    split_calls,
)
from deepprep.types import Table, TableSet


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_has_the_papers_31_operators():
    assert len(OPERATORS) == 31


@pytest.mark.parametrize(
    "category,expected",
    [
        ("cleaning", 5),       # Sec 2.2.1
        ("normalization", 3),  # Sec 2.2.2
        ("schema_edit", 7),    # Sec 2.2.3
        ("row_select", 3),     # Sec 2.2.4
        ("aggregation", 3),    # Sec 2.2.5
        ("combination", 3),    # Sec 2.2.6
        ("reshaping", 5),      # Sec 2.2.7
        ("program", 1),        # Sec 2.2.8
        ("control", 1),        # Terminate (Figure 4)
    ],
)
def test_operator_counts_per_category(category, expected):
    assert len(operators_by_category()[category]) == expected


def test_every_operator_appears_in_the_manual():
    manual = operator_manual()
    for name in OPERATORS:
        assert f"{name}(" in manual, f"{name} is missing from the prompt manual"


def test_every_category_is_named():
    for op in OPERATORS.values():
        assert type(op).CATEGORY in CATEGORY_ORDER


# --------------------------------------------------------------------------- #
# Parser -- the paper's bare-identifier notation
# --------------------------------------------------------------------------- #
def test_parses_bare_identifier_notation_from_the_paper():
    # Sec 2.1: "Consider operator Deduplicate(movies, [id], first) as an example."
    call = parse_operator_call("Deduplicate(movies, [id], first)")
    assert call.name == "Deduplicate"
    assert call.params == {"table": "movies", "subset": ["id"], "keep": "first"}


def test_parses_quoted_python_style_equivalently():
    a = parse_operator_call("Deduplicate(movies, [id], first)")
    b = parse_operator_call("Deduplicate('movies', ['id'], keep='last')")
    assert a.params["table"] == b.params["table"]
    assert a.params["subset"] == b.params["subset"]


def test_parses_lambda_parameters():
    call = parse_operator_call("ValueTransform(t, c, lambda x: str(x).strip().lower())")
    assert callable(call.params["func"])
    assert call.params["func"]("  AB ") == "ab"


def test_parses_tuple_and_dict_parameters():
    call = parse_operator_call("Join(a, b, on=(x, y), how=left)")
    assert call.params["on"] == ["x", "y"]
    call = parse_operator_call("RenameColumn(t, {'a': 'b'})")
    assert call.params["rename_map"] == {"a": "b"}


def test_parses_triple_backtick_code_block():
    # Figure 4 writes ExeCode(..., code=```...```).
    src = 'ExeCode([a, b], out, ```\nout = a\n```)'
    call = parse_operator_call(src)
    assert call.params["tables"] == ["a", "b"]
    assert "out = a" in call.params["func"]


def test_split_calls_is_bracket_and_string_aware():
    text = (
        'Deduplicate(movies, [id], first)\n'
        'ValueTransform(movies, title, lambda x: x.split("\\n")[0])\n'
        'ExeCode([a], out, ```\nx = 1\ny = 2\nout = a\n```)\n'
    )
    calls = split_calls(text)
    assert len(calls) == 3
    assert calls[0].startswith("Deduplicate")
    assert "y = 2" in calls[2]


def test_split_calls_strips_list_markers_and_prose():
    calls = split_calls("1. Deduplicate(t)\n- Sort(t, [a])\nThis is prose, not a call.\n")
    assert calls == ["Deduplicate(t)", "Sort(t, [a])"]


# --------------------------------------------------------------------------- #
# Parser -- error messages are agent feedback, so they must be actionable
# --------------------------------------------------------------------------- #
def test_unknown_operator_lists_the_available_ones():
    with pytest.raises(ParseError) as e:
        parse_operator_call("Frobnicate(t)")
    assert "unknown operator" in str(e.value)
    assert "Deduplicate" in str(e.value)


def test_missing_required_parameter_names_it_and_shows_the_signature():
    with pytest.raises(ParseError) as e:
        parse_operator_call("Join(a, b)")
    assert "'on'" in str(e.value)
    assert "Join(left, right, on" in str(e.value)


def test_unexpected_parameter_is_rejected():
    with pytest.raises(ParseError) as e:
        parse_operator_call("Deduplicate(t, nonsense=1)")
    assert "nonsense" in str(e.value)


def test_invalid_choice_is_rejected():
    with pytest.raises(ParseError) as e:
        parse_operator_call("DropNA(t, [a], sometimes)")
    assert "must be one of" in str(e.value)


def test_bare_non_python_tokens_are_repaired_not_rejected():
    """Format strings, regexes and separators appear unquoted in model output.

    They are not valid Python, but rejecting them costs the agent a whole turn.
    """
    call = parse_operator_call("StandardizeDatetime(t, c, %Y-%m-%d)")
    assert call.params["format"] == "%Y-%m-%d"

    call = parse_operator_call("SplitColumn(t, c, [a, b], sep=- )")
    assert call.params["sep"] == "-"


def test_syntax_error_reports_position():
    with pytest.raises(ParseError) as e:
        parse_operator_call("Deduplicate(movies, [id,)")
    assert "invalid operator syntax" in str(e.value)


# --------------------------------------------------------------------------- #
# Positional-only state argument
# --------------------------------------------------------------------------- #
def test_operators_whose_param_is_named_tables_still_bind(simple_tables):
    """`ExeCode`/`Union` have a parameter literally called `tables`.

    The state is passed positionally, so a same-named parameter must land in
    kwargs rather than colliding with it.
    """
    call = parse_operator_call('ExeCode([people], out, "out = people[[\'id\']]")')
    out = call.execute(simple_tables)
    assert "out" in out
    assert list(out["out"].columns) == ["id"]

    call = parse_operator_call("Union([depts, depts], all, target=u)")
    out = call.execute(simple_tables)
    assert len(out["u"].df) == 6


# --------------------------------------------------------------------------- #
# Purity: operators are functions T -> T
# --------------------------------------------------------------------------- #
def test_operators_do_not_mutate_the_input_state(simple_tables):
    before = simple_tables.fingerprint()
    parse_operator_call("Deduplicate(people, [id], first)").execute(simple_tables)
    parse_operator_call("DropColumn(people, [tags])").execute(simple_tables)
    assert simple_tables.fingerprint() == before


def test_untouched_tables_are_carried_through(simple_tables):
    # Sec 2.1: after Deduplicate(movies, ...) "tables ratings and directors remain unchanged".
    out = parse_operator_call("Deduplicate(people, [id], first)").execute(simple_tables)
    assert out.names == ["people", "depts"]
    assert out["depts"].df.equals(simple_tables["depts"].df)


# --------------------------------------------------------------------------- #
# Per-category behaviour
# --------------------------------------------------------------------------- #
def test_cleaning_operators(simple_tables):
    t = simple_tables
    assert len(parse_operator_call("Deduplicate(people, [id], first)").execute(t)["people"].df) == 4
    assert len(parse_operator_call("DropNA(people, [salary])").execute(t)["people"].df) == 4

    out = parse_operator_call("MissingValueImputation(people, salary, mean)").execute(t)
    assert out["people"].df["salary"].notna().all()

    # Sec 2.2.1 says ErrorDetection *identifies* invalid records, so the paper's
    # 3-argument form flags rather than deletes.
    out = parse_operator_call("ErrorDetection(people, salary, lambda v: v is not None and v > 1000)").execute(t)
    assert out["people"].df["salary_is_error"].tolist() == [False, False, False, False, True]
    out = parse_operator_call(
        "ErrorDetection(people, salary, lambda v: v is not None and v > 1000, remove)"
    ).execute(t)
    assert len(out["people"].df) == 4

    out = parse_operator_call("OutlierDetection(people, salary, flag)").execute(t)
    assert "salary_is_outlier" in out["people"].columns


def test_normalization_operators(simple_tables):
    t = simple_tables
    out = parse_operator_call("ValueTransform(people, name, lambda x: str(x).strip().lower())").execute(t)
    assert out["people"].df["name"].tolist() == ["alice", "bob", "bob", "carol", "dave"]

    out = parse_operator_call("CastType(people, id, str)").execute(t)
    assert out["people"].df["id"].dtype == "string"

    out = parse_operator_call("StandardizeDatetime(people, joined, %Y-%m-%d)").execute(t)
    assert out["people"].df["joined"].iloc[0] == "2021-01-02"


def test_schema_editing_operators(simple_tables):
    t = simple_tables
    out = parse_operator_call("RenameColumn(people, {'name': 'full_name'})").execute(t)
    assert "full_name" in out["people"].columns

    out = parse_operator_call("SelectColumn(people, [id, name])").execute(t)
    assert out["people"].columns == ["id", "name"]

    out = parse_operator_call("AddNewColumn(people, initial, lambda r: str(r['name']).strip()[0])").execute(t)
    assert out["people"].df["initial"].iloc[0] == "A"

    out = parse_operator_call("SplitColumn(people, tags, [t1, t2], sep=',')").execute(t)
    assert out["people"].df["t1"].iloc[0] == "a" and out["people"].df["t2"].iloc[0] == "b"

    out = parse_operator_call("Concatenate(people, [name, dept], who, sep='-')").execute(t)
    assert out["people"].df["who"].iloc[0] == " Alice -eng"

    out = parse_operator_call("Subtitle(people, 2024, year)").execute(t)
    assert set(out["people"].df["year"]) == {2024}


def test_row_selection_operators(simple_tables):
    t = simple_tables
    out = parse_operator_call("Filter(people, lambda r: r['dept'] == 'eng')").execute(t)
    assert len(out["people"].df) == 3

    out = parse_operator_call("Sort(people, [id], False)").execute(t)
    assert out["people"].df["id"].tolist() == [4, 3, 2, 2, 1]

    out = parse_operator_call("TopK(people, 2, by=[id], ascending=False)").execute(t)
    assert out["people"].df["id"].tolist() == [4, 3]


def test_aggregation_operators(simple_tables):
    t = simple_tables
    out = parse_operator_call("GroupBy(people, [dept], {'salary': 'mean'})").execute(t)
    eng = out["people"].df.set_index("dept").loc["eng", "salary"]
    assert eng == pytest.approx((100 + 200 + 200) / 3)

    out = parse_operator_call("Count(people)").execute(t)
    assert out["people_count"].df["count"].iloc[0] == 5

    out = parse_operator_call("CalculateStatistic(people, total, lambda df: float(df['salary'].sum()))").execute(t)
    assert out["people_total"].df["total"].iloc[0] == pytest.approx(5500.0)


def test_combination_operators(simple_tables):
    t = simple_tables
    out = parse_operator_call("Join(people, depts, on=(dept, code), how=inner)").execute(t)
    assert "people_depts_join" in out
    assert "label" in out["people_depts_join"].columns

    out = parse_operator_call("Union([depts, depts], distinct, target=u)").execute(t)
    assert len(out["u"].df) == 3

    out = parse_operator_call("Append(depts, depts, target=a)").execute(t)
    assert len(out["a"].df) == 6


def test_reshaping_operators(simple_tables):
    t = simple_tables
    out = parse_operator_call("Explode(people, tags, sep=',')").execute(t)
    assert len(out["people"].df) == 7  # a,b / c / c / b,c / a
    # Without an explicit separator a plain string column is left intact.
    assert len(parse_operator_call("Explode(people, tags)").execute(t)["people"].df) == 5

    out = parse_operator_call("Stack(people, [id], [salary])").execute(t)
    assert set(out["people"].columns) == {"id", "variable", "value"}

    out = parse_operator_call("Transpose(depts)").execute(t)
    assert out["depts"].df.shape[0] == 2

    out = parse_operator_call(
        "Pivot(people, index=[dept], columns=[name], values=[salary], aggfunc=mean)"
    ).execute(t)
    assert "dept" in out["people"].columns


def test_pivot_does_not_invent_absent_group_combinations():
    """`pivot_table(dropna=False)` materializes the full cartesian product of the
    index levels, inventing rows for combinations that never occur."""
    df = pd.DataFrame(
        {"a": ["x", "x", "y"], "b": ["p", "q", "p"], "c": ["2019", "2020", "2019"], "v": [1.0, 2.0, 3.0]}
    )
    ts = TableSet([Table("t", df)])
    out = parse_operator_call(
        "Pivot(t, index=[a, b], columns=[c], values=[v], aggfunc=mean)"
    ).execute(ts)
    # (x,p) (x,q) (y,p) occur; (y,q) does not.
    assert len(out["t"].df) == 3


def test_explode_splits_string_encoded_lists():
    """The paper's example has genres = "Sci-Fi, Action" as a *string*, and the
    target schema demands single-valued categories."""
    df = pd.DataFrame({"m": ["a", "b"], "genres": ["Sci-Fi, Action", "Sci-Fi"]})
    out = parse_operator_call("Explode(t, genres, sep=',')").execute(TableSet([Table("t", df)]))
    assert out["t"].df["genres"].tolist() == ["Sci-Fi", "Action", "Sci-Fi"]


# --------------------------------------------------------------------------- #
# Execution error messages
# --------------------------------------------------------------------------- #
def test_missing_table_error_lists_available_tables(simple_tables):
    with pytest.raises(OperatorError) as e:
        parse_operator_call("Deduplicate(nope, [id])").execute(simple_tables)
    assert "MissingTable" in str(e.value)
    assert "people" in str(e.value)


def test_missing_column_error_matches_the_papers_figure_4_shape(simple_tables):
    # Figure 4: "MissingColumn: No column named 'year' in the table ratings"
    with pytest.raises(OperatorError) as e:
        parse_operator_call("SelectColumn(people, [year])").execute(simple_tables)
    msg = str(e.value)
    assert "MissingColumn" in msg
    assert "'year'" in msg and "people" in msg
    assert "Available columns" in msg


def test_empty_inner_join_is_reported_with_sample_keys():
    left = TableSet([Table("l", pd.DataFrame({"k": ["a"]})), Table("r", pd.DataFrame({"k": ["b"]}))])
    with pytest.raises(OperatorError) as e:
        parse_operator_call("Join(l, r, on=k, how=inner)").execute(left)
    assert "EmptyResult" in str(e.value)
    assert "Sample left keys" in str(e.value)


def test_execode_missing_target_variable_says_what_to_do(simple_tables):
    with pytest.raises(OperatorError) as e:
        parse_operator_call('ExeCode([people], out, "z = people")').execute(simple_tables)
    assert "did not assign a variable named 'out'" in str(e.value)


def test_groupby_on_non_numeric_column_explains_the_fix(simple_tables):
    with pytest.raises(OperatorError) as e:
        parse_operator_call("GroupBy(people, [dept], {'name': 'mean'})").execute(simple_tables)
    assert "CastType" in str(e.value)


# --------------------------------------------------------------------------- #
# Terminate
# --------------------------------------------------------------------------- #
def test_terminate_marks_the_answer_table(simple_tables):
    from deepprep.operators import answer_table_name

    out = parse_operator_call("Terminate([people])").execute(simple_tables)
    assert answer_table_name(out) == "people"


def test_terminate_requires_exactly_one_table(simple_tables):
    with pytest.raises(OperatorError) as e:
        parse_operator_call("Terminate([people, depts])").execute(simple_tables)
    assert "exactly one result table" in str(e.value)


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "code",
    [
        '"import os\\nout = people"',
        '"out = open(\'/etc/passwd\').read()"',
        '"out = eval(\'1+1\')"',
        '"out = ().__class__.__bases__"',
    ],
)
def test_sandbox_rejects_dangerous_code(simple_tables, code):
    with pytest.raises(OperatorError):
        parse_operator_call(f"ExeCode([people], out, {code})").execute(simple_tables)


def test_exec_can_be_disabled_entirely(monkeypatch, simple_tables):
    monkeypatch.setenv("DEEPPREP_ALLOW_EXEC", "0")
    with pytest.raises((OperatorError, ParseError)):
        parse_operator_call('ExeCode([people], out, "out = people")').execute(simple_tables)


# --------------------------------------------------------------------------- #
# Round-tripping
# --------------------------------------------------------------------------- #
def test_pipeline_round_trips_through_serialization(demo_task):
    """`phi(P)` must reparse: training data and the tree both depend on it."""
    ops = parse_pipeline("\n".join(demo_task.gold_pipeline))
    reparsed = parse_pipeline("\n".join(op.to_source() for op in ops))
    assert [o.name for o in ops] == [o.name for o in reparsed]

    state = demo_task.sources.copy()
    for op in reparsed:
        state = op.execute(state)
    assert len(state["joined"].df) == len(demo_task.target_table)


def test_gold_pipeline_reproduces_the_target_table(demo_task):
    from deepprep.eval import table_match

    state = demo_task.sources.copy()
    for op in parse_pipeline("\n".join(demo_task.gold_pipeline)):
        state = op.execute(state)
    assert table_match(state["joined"].df, demo_task.target_table)


def test_numpy_types_survive_serialization(simple_tables):
    df = pd.DataFrame({"a": np.array([1, 2], dtype=np.int32)})
    out = parse_operator_call("Sort(t, [a])").execute(TableSet([Table("t", df)]))
    assert out["t"].df["a"].tolist() == [1, 2]
