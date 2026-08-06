"""Cross-module integration.

The unit suites verify each module against the paper in isolation.  This file
verifies that the modules compose: a task synthesized by Sec 5.3 must be solvable
by the Sec 4 agent, scorable by the Sec 6.1 metrics, and convertible into the
Sec 5.1 training data — which is the actual end-to-end claim of the system.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from deepprep.agent import DeepPrepAgent
from deepprep.agent.llm import ScriptedClient
from deepprep.baselines import CodeGenAgent, MCTSOperatorAgent, PlanAndSolveAgent, ReActAgent
from deepprep.cli import main
from deepprep.eval import evaluate, table_match
from deepprep.operators import parse_pipeline
from deepprep.training import build_op_syntax_dataset, materialize_states


# --------------------------------------------------------------------------- #
# Synthesis -> agent -> metrics -> training data
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    db = tmp_path / "shop.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER,
                             amount REAL, status TEXT);
        INSERT INTO customer VALUES (1,'Ann','Berlin'),(2,'Bo','Paris'),(3,'Cy','Berlin');
        INSERT INTO orders VALUES (1,1,10.0,'paid'),(2,1,20.0,'paid'),
                                  (3,2,30.0,'open'),(4,3,40.0,'paid'),(5,3,5.0,'open');
        """
    )
    con.commit()
    con.close()
    return db


@pytest.fixture
def synth_tasks(tiny_db: Path, tmp_path: Path):
    import json

    from deepprep.synthesis import SynthesisConfig, load_benchmark, synthesize_dataset

    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            [
                {
                    "db_id": "shop",
                    "question": "What is the total paid amount per city?",
                    "query": "SELECT c.city, SUM(o.amount) FROM customer AS c "
                             "JOIN orders AS o ON c.id = o.customer_id "
                             "WHERE o.status = 'paid' GROUP BY c.city",
                },
                {
                    "db_id": "shop",
                    "question": "List customer names in Berlin.",
                    "query": "SELECT name FROM customer WHERE city = 'Berlin'",
                },
                {
                    "db_id": "shop",
                    "question": "How many orders are there?",
                    "query": "SELECT COUNT(*) FROM orders",
                },
            ]
        )
    )
    # The synthesis loader expects <db_root>/<db_id>/<db_id>.sqlite.
    dbdir = tmp_path / "dbs" / "shop"
    dbdir.mkdir(parents=True)
    (dbdir / "shop.sqlite").write_bytes(tiny_db.read_bytes())

    cases = load_benchmark(spec)
    tasks, _ = synthesize_dataset(
        cases, db_root=tmp_path / "dbs", config=SynthesisConfig(seed=3), llm=None
    )
    return tasks


def test_synthesis_produces_solvable_tasks(synth_tasks):
    assert synth_tasks, "synthesis produced no tasks"
    for task in synth_tasks:
        assert task.gold_pipeline
        assert task.target_table is not None
        assert task.target_schema.columns


def test_synthesized_gold_pipelines_reproduce_their_targets(synth_tasks):
    """The property the entire training set rests on: if the gold pipeline does
    not reproduce T* on the *dirty* sources, the supervision is wrong."""
    for task in synth_tasks:
        state = task.sources.copy()
        for op in parse_pipeline("\n".join(task.gold_pipeline)):
            state = op.execute(state)
        produced = state[list(state.names)[-1]].df
        candidates = [t.df for t in state]
        assert any(table_match(df, task.target_table) for df in candidates), (
            f"task {task.task_id}: no table in the final state matches T*; "
            f"last table shape {produced.shape}, target {task.target_table.shape}"
        )


def test_an_agent_replaying_the_gold_pipeline_scores_a_perfect_case(synth_tasks):
    """Closes the loop: synthesis -> agent -> Sec 6.1 metrics."""
    task = synth_tasks[0]
    answer_table = _answer_table_name(task)
    responses = [
        "<plan>Execute the known-good pipeline.</plan>\n"
        "<answer>\n" + "\n".join(task.gold_pipeline) + f"\nTerminate([{answer_table}])\n</answer>"
    ]
    result = DeepPrepAgent(llm=ScriptedClient(responses), max_turns=2).solve(task)
    assert result.completed, result.trajectory[-1].error
    assert table_match(result.table, task.target_table)

    report = evaluate(
        DeepPrepAgent(llm=ScriptedClient(responses), max_turns=2),
        [task],
        max_workers=1,
        verbose=False,
    )
    assert report.accuracy == 100.0
    assert report.completion_rate == 100.0


def _answer_table_name(task) -> str:
    state = task.sources.copy()
    for op in parse_pipeline("\n".join(task.gold_pipeline)):
        state = op.execute(state)
    for t in state:
        if table_match(t.df, task.target_table):
            return t.name
    return state.names[-1]


def test_stage1_training_data_builds_from_synthesized_tasks(synth_tasks):
    """Sec 5.3 exists to feed Sec 5.1; the handoff must actually work."""
    examples = build_op_syntax_dataset(synth_tasks, max_span=2, verbose=False)
    assert examples

    for task in synth_tasks:
        ops = parse_pipeline("\n".join(task.gold_pipeline))
        states = materialize_states(task.sources, ops)
        assert len(states) == len(ops) + 1

    for ex in examples:
        assert ex.trainable == [False, False, True]
        assert parse_pipeline(ex.messages[-1]["content"])


def test_synthesized_sources_are_actually_dirty(synth_tasks):
    """Sec 5.3 injects noise so the pipelines exercise the cleaning operators.
    A corpus with no corruptions would silently train only on the analytical half."""
    cleaning = {
        "DropNA", "MissingValueImputation", "Deduplicate", "ErrorDetection",
        "OutlierDetection", "ValueTransform", "StandardizeDatetime", "CastType",
    }
    used = {
        op.name
        for task in synth_tasks
        for op in parse_pipeline("\n".join(task.gold_pipeline))
    }
    assert used & cleaning, f"no cleaning operator appears in any gold pipeline: {used}"


# --------------------------------------------------------------------------- #
# All five methods share one harness (Table 2)
# --------------------------------------------------------------------------- #
def test_every_method_conforms_to_the_solver_interface(demo_task):
    """Sec 6.1 compares DeepPrep against the baselines on the same metrics, so
    they must all satisfy the same interface with no special-casing."""
    for cls in (DeepPrepAgent, CodeGenAgent, PlanAndSolveAgent, ReActAgent, MCTSOperatorAgent):
        solver = cls(llm=ScriptedClient(["no useful output"] * 12))
        report = evaluate(
            solver, [demo_task], method=cls.__name__, max_workers=1, verbose=False
        )
        assert report.n_cases == 1
        assert 0.0 <= report.accuracy <= 100.0
        assert 0.0 <= report.completion_rate <= 100.0
        assert report.cases[0].stop_reason != "solver_error"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_demo_reproduces_the_paper_example(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "exact match=True" in out
    assert "backtracks=1" in out


def test_cli_operators_lists_all_31(capsys):
    assert main(["operators"]) == 0
    out = capsys.readouterr().out
    assert "31 operators" in out
    assert out.count("\n- ") == 31


def test_cli_build_op_syntax_writes_a_dataset(tmp_path, demo_task, capsys):
    import json

    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(demo_task.to_dict()) + "\n")
    out = tmp_path / "d.jsonl"
    assert main(["build-op-syntax", "--tasks", str(tasks_path), "--out", str(out)]) == 0
    assert out.exists()
    assert len(out.read_text().strip().splitlines()) > 0
