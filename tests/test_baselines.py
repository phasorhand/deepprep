"""Tests for the prompting baselines of paper Sec 6.1.

Everything runs offline through :class:`~deepprep.agent.llm.ScriptedClient`, so
no API key and no network are needed.  Each baseline is exercised end-to-end on
the ``movies_demo`` task — the Figure 4 / Example 4 instance — with a canned
trajectory that a real backbone could plausibly have produced.

The most important test here is
:func:`test_react_cannot_recover_from_an_early_wrong_decision`: it encodes the
claim of Sec 1 that "linear reasoning makes it difficult to revise earlier
decisions, since the effects of previous operators are already reflected in the
current state", which is the gap DeepPrep's reasoning tree is designed to close.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deepprep.agent.agent import SolveResult
from deepprep.agent.llm import ScriptedClient
from deepprep.baselines import (
    CodeGenAgent,
    MCTSOperatorAgent,
    PlanAndSolveAgent,
    ReActAgent,
)
from deepprep.eval import evaluate
from deepprep.eval.metrics import table_match
from deepprep.types import ADPTask

TASK_PATH = Path(__file__).resolve().parents[1] / "examples" / "movies_demo" / "task.json"

#: A value that occurs only in the ground-truth table T* (it is the mean of two
#: source ratings), used to prove no baseline is shown the gold answer.
GOLD_ONLY_VALUE = "8.875"


@pytest.fixture()
def task() -> ADPTask:
    return ADPTask.load(TASK_PATH)


# --------------------------------------------------------------------------- #
# Canned model outputs
# --------------------------------------------------------------------------- #
#: A pandas program solving the demo task, as CodeGen would emit it in one shot.
CODEGEN_PROGRAM = """
m = movies.drop_duplicates(subset=['id'], keep='first').copy()
m['title'] = m['title'].astype(str).str.strip().str.lower()
m['genre'] = m['genres'].astype(str).str.split(',')
m = m.explode('genre')
m['genre'] = m['genre'].str.strip()

r = ratings.copy()
parts = r['values'].astype(str).str.split(' (', regex=False, expand=True)
r['rating_value'] = parts[0].astype(float)
r['year'] = parts[1].str.rstrip(')')
r = r[r['year'].isin(['2019', '2020'])]

md = m.merge(directors, left_on='dir_id', right_on='id', suffixes=('', '_dir'))
j = md.merge(r, left_on='title', right_on='movie')
j = j.rename(columns={'name': 'director_name'})

pivoted = j.pivot_table(index=['director_name', 'genre'], columns='year',
                        values='rating_value', aggfunc='mean').reset_index()
pivoted.columns = [str(c) for c in pivoted.columns]
result_table = pivoted[['director_name', 'genre', '2019', '2020']]
"""

#: The same program with the classic one-shot mistake: `values` is assumed to be
#: a plain float column rather than "8.9 (2020)".
CODEGEN_BROKEN = """
r = ratings.copy()
r['year'] = r['year_column']
result_table = r
"""

REACT_TURN_1 = """<thought>
`movies` has a duplicate id and padded, miscased titles while `ratings.movie` is
lowercase, and the year is packed inside `ratings.values`. Clean both first.
</thought>
<action>
Deduplicate(movies, [id], first)
ValueTransform(movies, title, lambda x: str(x).strip().lower())
SelectColumn(ratings, [movie, values])
SplitColumn(ratings, values, [rating_value, year], sep=" (")
ValueTransform(ratings, year, lambda x: str(x).rstrip(")"))
CastType(ratings, rating_value, float)
Filter(ratings, lambda r: r["year"] in ("2019", "2020"))
</action>"""

REACT_TURN_2 = """<thought>
`genres` is still multi-valued; explode it, attach director names, join on the
normalized title and pivot the years into columns.
</thought>
<action>
Explode(movies, genres, sep=",")
Join(movies, directors, on=(dir_id, id), how=inner, target=movies_directors_join)
Join(movies_directors_join, ratings, on=(title, movie), how=inner, target=joined)
RenameColumn(joined, {"name": "director_name", "genres": "genre"})
Pivot(joined, index=[director_name, genre], columns=[year], values=[rating_value], aggfunc=mean)
</action>"""

REACT_TURN_3 = """<thought>
`joined` now has director_name, genre, 2019 and 2020. Done.
</thought>
<answer>Terminate([joined])</answer>"""

#: The Figure 4 mistake, made by an agent with no way to undo it: turn 1 drops
#: `ratings.values`, which every later turn needs.
REACT_WRONG_TURN_1 = """<thought>
I only need the movie and its rating, so narrow `ratings` down now.
</thought>
<action>
Deduplicate(movies, [id], first)
ValueTransform(movies, title, lambda x: str(x).strip().lower())
SelectColumn(ratings, [movie, rating])
</action>"""

REACT_WRONG_TURN_2 = """<thought>
The target needs one column per year, and the year lives inside `values`.
</thought>
<action>
SplitColumn(ratings, values, [rating_value, year], sep=" (")
</action>"""

REACT_WRONG_TURN_3 = """<thought>
My earlier SelectColumn dropped `values`. I need to undo that decision and keep
`values` instead.
</thought>
<action>
SelectColumn(ratings, [movie, values])
</action>"""

REACT_WRONG_TURN_4 = """<thought>
Try to recover `values` from the sources again.
</thought>
<action>
SplitColumn(ratings, values, [rating_value, year], sep=" (")
</action>"""

REACT_WRONG_TURN_5 = """<thought>
There is no way back to the state before the SelectColumn.
</thought>
<action>
SelectColumn(ratings, [movie, values])
</action>"""


def scripted(*responses: str) -> ScriptedClient:
    return ScriptedClient(list(responses))


def prompts(llm: ScriptedClient) -> str:
    """All prompt text the client was ever shown, concatenated."""
    return "\n".join(m.get("content", "") for call in llm.calls for m in call)


# --------------------------------------------------------------------------- #
# Interface conformance
# --------------------------------------------------------------------------- #
BASELINES = [CodeGenAgent, PlanAndSolveAgent, ReActAgent, MCTSOperatorAgent]


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.METHOD)
def test_baseline_matches_the_deepprep_agent_interface(cls, task: ADPTask) -> None:
    """`evaluate()` only needs `.llm` and `.solve()`; every baseline must have both."""
    llm = scripted()  # exhausted script: the agent gets junk on every call
    agent = cls(llm)

    assert agent.llm is llm
    result = agent.solve(task)

    assert isinstance(result, SolveResult)
    assert result.task_id == task.task_id
    # Junk in, no answer out -- but never a crash, and never a false completion.
    assert result.completed is False
    assert result.table is None
    assert result.stop_reason in {"answered", "max_turns", "llm_error"}
    assert result.usage.n_calls >= 1
    assert result.elapsed_s >= 0.0


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.METHOD)
def test_baseline_reports_llm_failure_without_crashing(cls, task: ADPTask) -> None:
    class DeadClient:
        model = "dead"

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            raise ConnectionError("endpoint unreachable")

    result = cls(DeadClient()).solve(task)

    assert result.stop_reason == "llm_error"
    assert result.completed is False
    assert result.error is not None and "ConnectionError" in result.error


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.METHOD)
def test_baseline_never_sees_the_gold_target_table(cls, task: ADPTask) -> None:
    """T* "is used only for evaluation" (Sec 2.1)."""
    assert GOLD_ONLY_VALUE in task.target_table.to_string()

    llm = scripted()
    cls(llm).solve(task)

    assert GOLD_ONLY_VALUE not in prompts(llm)


# --------------------------------------------------------------------------- #
# CodeGen
# --------------------------------------------------------------------------- #
def test_codegen_solves_the_demo_task_in_one_call(task: ADPTask) -> None:
    llm = scripted(f"<code>\n{CODEGEN_PROGRAM}\n</code>")
    result = CodeGenAgent(llm).solve(task)

    assert result.completed is True
    assert result.stop_reason == "answered"
    assert table_match(result.table, task.target_table)
    # One generation, no interaction: the defining property of this baseline.
    assert llm.usage.n_calls == 1
    assert result.n_turns == 1


def test_codegen_runs_through_the_execode_operator(task: ADPTask) -> None:
    llm = scripted(f"<code>\n{CODEGEN_PROGRAM}\n</code>")
    result = CodeGenAgent(llm).solve(task)

    assert [op.name for op in result.pipeline] == ["ExeCode", "Terminate"]


def test_codegen_accepts_a_fenced_program_without_tags(task: ADPTask) -> None:
    llm = scripted(f"```python\n{CODEGEN_PROGRAM}\n```")
    result = CodeGenAgent(llm).solve(task)

    assert result.completed is True


def test_codegen_repair_sees_only_the_error_trace(task: ADPTask) -> None:
    """A repair turn must not become grounding in intermediate results (Sec 1)."""
    llm = scripted(
        f"<code>\n{CODEGEN_BROKEN}\n</code>",
        f"<code>\n{CODEGEN_PROGRAM}\n</code>",
    )
    result = CodeGenAgent(llm, max_repairs=1).solve(task)

    assert result.completed is True
    assert result.n_turns == 2

    repair_prompt = llm.calls[1][-1]["content"]
    assert "year_column" in repair_prompt  # the error trace is shown
    assert "### Table:" not in repair_prompt  # ... but no materialized state is
    assert "| director_name |" not in repair_prompt


def test_codegen_without_repairs_fails_on_a_broken_program(task: ADPTask) -> None:
    llm = scripted(f"<code>\n{CODEGEN_BROKEN}\n</code>")
    result = CodeGenAgent(llm, max_repairs=0).solve(task)

    assert result.completed is False
    assert result.table is None
    assert result.stop_reason == "max_turns"
    assert llm.usage.n_calls == 1


# --------------------------------------------------------------------------- #
# Plan-and-Solve
# --------------------------------------------------------------------------- #
def test_plan_and_solve_uses_exactly_two_stages(task: ADPTask) -> None:
    llm = scripted(
        "<plan>Clean movies, unpack ratings.values, join, pivot.</plan>",
        "<pipeline>\n" + "\n".join(task.gold_pipeline) + "\n</pipeline>",
    )
    result = PlanAndSolveAgent(llm).solve(task)

    assert result.completed is True
    assert result.stop_reason == "answered"
    assert table_match(result.table, task.target_table)
    # Stage 1 = plan, stage 2 = the whole operator sequence. Nothing else.
    assert llm.usage.n_calls == 2
    assert result.n_turns == 2
    assert len(result.trajectory) == 2


def test_plan_and_solve_has_no_feedback_loop(task: ADPTask) -> None:
    """A failed pipeline is not repaired: PaS never reads the execution result."""
    llm = scripted(
        "<plan>Join movies and ratings, then select.</plan>",
        "<pipeline>\n"
        "Join(movies, ratings, on=(title, movie), how=inner, target=joined)\n"
        "SelectColumn(joined, [does_not_exist])\n"
        "Terminate([joined])\n"
        "</pipeline>",
    )
    result = PlanAndSolveAgent(llm).solve(task)

    assert result.completed is False
    assert result.table is None
    assert llm.usage.n_calls == 2  # no third, corrective call
    assert result.error is not None and "does_not_exist" in result.error
    # The successful prefix still feeds the Eq. (8) partial reward.
    assert result.fallback_table is not None


def test_plan_and_solve_reports_an_unparseable_pipeline(task: ADPTask) -> None:
    llm = scripted("<plan>...</plan>", "<pipeline>\nthis is not an operator\n</pipeline>")
    result = PlanAndSolveAgent(llm).solve(task)

    assert result.completed is False
    assert result.stop_reason == "max_turns"


# --------------------------------------------------------------------------- #
# ReAct
# --------------------------------------------------------------------------- #
def test_react_solves_the_demo_task_iteratively(task: ADPTask) -> None:
    llm = scripted(REACT_TURN_1, REACT_TURN_2, REACT_TURN_3)
    result = ReActAgent(llm).solve(task)

    assert result.completed is True
    assert result.stop_reason == "answered"
    assert table_match(result.table, task.target_table)
    assert result.n_turns == 3
    # Linear by construction: there is no tree, so nothing can be a backtrack.
    assert result.n_backtracks == 0


def test_react_is_grounded_in_execution_feedback(task: ADPTask) -> None:
    """Unlike CodeGen/PaS, ReAct does observe intermediate states."""
    llm = scripted(REACT_TURN_1, REACT_TURN_2, REACT_TURN_3)
    ReActAgent(llm).solve(task)

    second_turn_context = "\n".join(m["content"] for m in llm.calls[1])
    assert "<observation>" in second_turn_context
    assert "rating_value" in second_turn_context  # a column only the env could know


def test_react_cannot_recover_from_an_early_wrong_decision(task: ADPTask) -> None:
    """The paper's core claim about linear agents (Sec 1).

    Turn 1 drops ``ratings.values``.  Turn 2 needs it.  Turns 3-5 try to get it
    back and cannot: the state that still had the column is gone, and the
    protocol offers no way to name it.  A tree-based agent would re-expand from
    the node *before* the ``SelectColumn`` (Figure 4); ReAct has no such node.
    """
    llm = scripted(
        REACT_WRONG_TURN_1,
        REACT_WRONG_TURN_2,
        REACT_WRONG_TURN_3,
        REACT_WRONG_TURN_4,
        REACT_WRONG_TURN_5,
    )
    result = ReActAgent(llm, max_turns=5).solve(task)

    assert result.completed is False
    assert result.table is None
    assert result.stop_reason == "max_turns"
    assert result.n_backtracks == 0

    # Turn 1 succeeded; every later turn failed on the column it discarded.
    assert result.trajectory[0].error is None
    for step in result.trajectory[1:]:
        assert step.error is not None
        assert "values" in step.error

    # Even the explicit attempt to undo the decision (turn 3) fails, because the
    # operator is applied to the already-mutated state rather than to a parent.
    assert "No column named 'values'" in result.trajectory[2].error
    assert "'movie', 'rating'" in result.trajectory[2].error


def test_react_stops_at_the_turn_budget(task: ADPTask) -> None:
    llm = scripted(REACT_TURN_1, REACT_TURN_1, REACT_TURN_1)
    result = ReActAgent(llm, max_turns=2).solve(task)

    assert result.n_turns == 2
    assert llm.usage.n_calls == 2
    assert result.stop_reason == "max_turns"
    assert result.completed is False


# --------------------------------------------------------------------------- #
# MCTS-OP
# --------------------------------------------------------------------------- #
def _mcts_expand(*ops: str) -> str:
    return "<candidates>\n" + "\n".join(ops) + "\n</candidates>"


def _mcts_rollout(ops: list[str], score: float) -> str:
    return "<pipeline>\n" + "\n".join(ops) + f"\n</pipeline>\n<score>{score}</score>"


def test_mcts_op_solves_the_demo_task(task: ADPTask) -> None:
    gold = task.gold_pipeline
    llm = scripted(_mcts_expand(gold[0]), _mcts_rollout(gold[1:], 0.9))
    result = MCTSOperatorAgent(llm, n_iterations=1, n_expand=1).solve(task)

    assert result.completed is True
    assert result.stop_reason == "answered"
    assert table_match(result.table, task.target_table)
    # One expansion call + one rollout call per iteration.
    assert llm.usage.n_calls == 2
    assert result.n_turns == 1


def test_mcts_op_keeps_the_highest_scoring_rollout(task: ADPTask) -> None:
    """UCB1 visits the unvisited sibling next, and the best scalar wins the answer.

    Iteration 1 rolls out the poor branch (``DropColumn``) and scores it low;
    iteration 2 is therefore steered to the untried sibling, whose rollout
    reaches the target schema and replaces the incumbent answer.
    """
    gold = task.gold_pipeline
    llm = scripted(
        _mcts_expand("DropColumn(movies, [genres])", gold[0]),
        _mcts_rollout(["Terminate([movies])"], 0.1),
        _mcts_expand("SelectColumn(movies, [no_such_column])"),
        _mcts_rollout(gold[1:], 0.9),
    )
    result = MCTSOperatorAgent(llm, n_iterations=2, n_expand=2).solve(task)

    assert result.completed is True
    assert table_match(result.table, task.target_table)
    assert result.n_turns == 2
    assert result.trajectory[0].parent_node_id == "n0"
    assert result.trajectory[0].created_node_ids == ["n1", "n2"]
    # The low-reward branch is abandoned in favour of its untried sibling.
    assert result.trajectory[1].parent_node_id == "n2"


def test_mcts_op_backpropagates_a_scalar(task: ADPTask) -> None:
    """Sec 7: node statistics are "scalar rollout statistics", nothing richer."""
    gold = task.gold_pipeline
    llm = scripted(_mcts_expand(gold[0]), _mcts_rollout(gold[1:], 1.0))
    agent = MCTSOperatorAgent(llm, n_iterations=1, n_expand=1)
    result = agent.solve(task)

    assert result.trajectory[0].feedback.startswith("reward=")
    assert result.trajectory[0].parent_node_id == "n0"
    assert result.trajectory[0].created_node_ids == ["n1"]


def test_mcts_op_falls_back_when_no_rollout_reaches_the_target(task: ADPTask) -> None:
    llm = scripted(
        _mcts_expand("Deduplicate(movies, [id], first)"),
        "<pipeline>\n</pipeline>\n<score>0.2</score>",
    )
    result = MCTSOperatorAgent(llm, n_iterations=1, n_expand=1).solve(task)

    assert result.completed is False
    assert result.table is None
    assert result.stop_reason == "max_turns"
    assert result.fallback_table is not None


def test_mcts_op_discards_candidates_that_do_not_execute(task: ADPTask) -> None:
    gold = task.gold_pipeline
    llm = scripted(
        _mcts_expand("SelectColumn(movies, [no_such_column])", gold[0]),
        _mcts_rollout(gold[1:], 0.8),
    )
    result = MCTSOperatorAgent(llm, n_iterations=1, n_expand=2).solve(task)

    # The broken candidate produced no node; the valid one did.
    assert result.trajectory[0].created_node_ids == ["n1"]
    assert result.trajectory[0].error is not None
    assert result.completed is True


# --------------------------------------------------------------------------- #
# Integration with the evaluation harness
# --------------------------------------------------------------------------- #
def test_evaluate_scores_every_baseline_unchanged(task: ADPTask) -> None:
    """`deepprep.eval.evaluate` must work on the baselines with no special case."""
    gold = task.gold_pipeline
    solvers = {
        "CodeGen": CodeGenAgent(scripted(f"<code>\n{CODEGEN_PROGRAM}\n</code>")),
        "Plan-and-Solve": PlanAndSolveAgent(
            scripted("<plan>...</plan>", "<pipeline>\n" + "\n".join(gold) + "\n</pipeline>")
        ),
        "ReAct": ReActAgent(scripted(REACT_TURN_1, REACT_TURN_2, REACT_TURN_3)),
        "MCTS-OP": MCTSOperatorAgent(
            scripted(_mcts_expand(gold[0]), _mcts_rollout(gold[1:], 0.9)),
            n_iterations=1,
            n_expand=1,
        ),
    }

    for method, solver in solvers.items():
        report = evaluate(
            solver, [task], method=method, dataset="movies_demo",
            max_workers=1, verbose=False,
        )
        assert report.n_cases == 1
        assert report.accuracy == 100.0, method
        assert report.completion_rate == 100.0, method
        assert report.cases[0].partial["partial"] == pytest.approx(1.0)


def test_evaluate_reports_an_incomplete_react_run(task: ADPTask) -> None:
    solver = ReActAgent(
        scripted(
            REACT_WRONG_TURN_1,
            REACT_WRONG_TURN_2,
            REACT_WRONG_TURN_3,
            REACT_WRONG_TURN_4,
            REACT_WRONG_TURN_5,
        )
    )
    report = evaluate(solver, [task], method="ReAct", max_workers=1, verbose=False)

    assert report.accuracy == 0.0
    assert report.completion_rate == 0.0
    assert report.cases[0].stop_reason == "max_turns"
