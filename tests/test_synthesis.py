"""Tests for the data synthesis module (paper Sec 5.3, "Data Synthesis for Training").

Everything here runs offline: the "NL2SQL benchmark" is a handful of SQLite
tables built in a temporary directory, and the LLM is
:class:`~deepprep.agent.llm.ScriptedClient`.

Three properties carry the module:

1. **Reversibility.**  "After each corruption step, we verify executability by
   applying the corresponding cleaning operator and checking that the previous
   table state is restored.  Only reversible corruptions are kept."  A corruption
   whose cleaning operator does *not* restore the state must be rejected.
2. **Gold-pipeline soundness.**  "The final ground-truth pipeline is formed by
   concatenating the cleaning pipeline with the task pipeline" — that pipeline,
   executed on the *dirty* synthesized sources, must reproduce ``T*`` exactly.
   Every training example depends on this, so it is checked on every task.
3. **Translation.**  ``q`` must become an operator pipeline that reproduces the
   SQLite result of ``q``.
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from deepprep.agent.llm import ScriptedClient
from deepprep.eval.metrics import table_match
from deepprep.synthesis import (
    NL2SQLCase,
    NoiseConfig,
    SynthesisConfig,
    build_clean_instance,
    execute_pipeline,
    infer_target_schema,
    inject_noise,
    load_benchmark,
    search_pipeline,
    state_signature,
    synthesize_dataset,
    synthesize_task,
    translate_sql,
    try_corruption,
    verify_pipeline,
    write_jsonl,
)
from deepprep.synthesis.nl2sql import referenced_tables
from deepprep.synthesis.noise import Corruption, LLMInverseProposer, apply_cleaning
from deepprep.synthesis.pipeline_search import TranslationError
from deepprep.types import Table, TableSet, load_tasks

DB_ID = "concerts"

_SCHEMA = """
CREATE TABLE singer (
    id       INTEGER PRIMARY KEY,
    name     TEXT,
    country  TEXT,
    age      INTEGER,
    debut    TEXT
);
CREATE TABLE concert (
    cid       INTEGER PRIMARY KEY,
    singer_id INTEGER,
    city      TEXT,
    year      INTEGER,
    revenue   REAL,
    FOREIGN KEY (singer_id) REFERENCES singer(id)
);
INSERT INTO singer VALUES
    (1, 'ada lovelace', 'uk',      36, '2019-01-05'),
    (2, 'bob marley',   'jamaica', 36, '2019-02-11'),
    (3, 'cleo parker',  'usa',     41, '2020-03-21'),
    (4, 'dina ray',     'usa',     29, '2020-07-04'),
    (5, 'eli stone',    'uk',      52, '2021-11-30');
INSERT INTO concert VALUES
    (10, 1, 'london',   2019, 120.5),
    (11, 1, 'leeds',    2020,  80.0),
    (12, 2, 'kingston', 2019, 200.0),
    (13, 3, 'austin',   2021,  95.25),
    (14, 4, 'dallas',   2021,  60.0),
    (15, 5, 'york',     2020,  45.5),
    (16, 3, 'boston',   2022, 150.0);
"""

#: Representative queries covering the SELECT/WHERE/JOIN/GROUP BY/ORDER BY/
#: LIMIT/DISTINCT subset the rule-based translator claims to handle.
QUERIES: list[str] = [
    "SELECT name FROM singer WHERE age > 30",
    "SELECT * FROM singer WHERE age < 40",
    "SELECT DISTINCT country FROM singer ORDER BY country",
    "SELECT name FROM singer ORDER BY age DESC LIMIT 1",
    "SELECT country, COUNT(*) FROM singer GROUP BY country",
    "SELECT AVG(age) FROM singer",
    "SELECT MIN(age), MAX(age) FROM singer",
    "SELECT COUNT(DISTINCT country) FROM singer",
    "SELECT name, age FROM singer WHERE country IN ('uk', 'usa') AND name LIKE 'a%'",
    "SELECT city FROM concert WHERE revenue BETWEEN 50 AND 100 ORDER BY revenue",
    "SELECT debut FROM singer WHERE country = 'uk' ORDER BY debut DESC",
    "SELECT T1.name, T2.city FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id",
    "SELECT T2.city, T1.name FROM singer AS T1, concert AS T2"
    " WHERE T1.id = T2.singer_id AND T2.year = 2019",
    "SELECT T1.country, SUM(T2.revenue) FROM singer AS T1 JOIN concert AS T2"
    " ON T1.id = T2.singer_id GROUP BY T1.country",
    "SELECT T1.name, COUNT(*) FROM singer AS T1 JOIN concert AS T2"
    " ON T1.id = T2.singer_id GROUP BY T1.name HAVING COUNT(*) > 1",
    "SELECT country, AVG(age), MAX(age) FROM singer GROUP BY country ORDER BY AVG(age) DESC",
]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def db_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Spider-shaped database directory: ``<root>/<db_id>/<db_id>.sqlite``."""
    root = tmp_path_factory.mktemp("databases")
    db_dir = root / DB_ID
    db_dir.mkdir()
    conn = sqlite3.connect(db_dir / f"{DB_ID}.sqlite")
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    return root


@pytest.fixture(scope="session")
def cases() -> list[NL2SQLCase]:
    return [
        NL2SQLCase(case_id=f"q{i}", db_id=DB_ID, question=f"Analytical question {i}.", query=q)
        for i, q in enumerate(QUERIES)
    ]


@pytest.fixture(scope="session")
def instances(db_root: Path, cases: list[NL2SQLCase]) -> list:
    out = [build_clean_instance(c, db_root) for c in cases]
    assert all(i is not None for i in out), "every fixture query must execute"
    return out


@pytest.fixture(scope="session")
def synthesized(db_root: Path, cases: list[NL2SQLCase]):
    config = SynthesisConfig(
        dataset="Synth-Test",
        split="train",
        seed=11,
        noise=NoiseConfig(max_steps=4, seed=11),
    )
    return synthesize_dataset(cases, db_root, config=config)


@pytest.fixture
def toy_tables() -> TableSet:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["ann", "bob", "bob", "dee"],
            "score": [1.5, 2.5, 3.5, 4.5],
            "day": ["2023-01-01", "2023-02-05", "2023-03-09", "2023-04-17"],
        }
    )
    return TableSet([Table("t", df)])


# --------------------------------------------------------------------------- #
# 1. Benchmark loading
# --------------------------------------------------------------------------- #
def test_load_benchmark_accepts_spider_and_bird_records(tmp_path: Path) -> None:
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            [
                {"db_id": "a", "question": "spider style", "query": "SELECT 1"},
                {"db_id": "b", "question": "bird style", "SQL": "SELECT 2;", "evidence": "hint"},
                {"db_id": "c", "question": "no sql at all"},
            ]
        )
    )
    loaded = load_benchmark(path)
    assert [c.db_id for c in loaded] == ["a", "b"]  # the record without SQL is dropped
    assert loaded[1].query == "SELECT 2"  # the trailing semicolon is stripped
    assert loaded[1].evidence == "hint"


def test_referenced_tables_covers_join_and_comma_forms() -> None:
    available = ["singer", "concert", "stadium"]
    assert referenced_tables(
        "SELECT * FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id", available
    ) == ["singer", "concert"]
    assert referenced_tables(
        "SELECT * FROM singer AS T1, concert AS T2 WHERE T1.id = T2.singer_id", available
    ) == ["singer", "concert"]
    assert referenced_tables("SELECT age FROM singer", available) == ["singer"]


def test_clean_instance_carries_sources_target_and_schema(db_root: Path) -> None:
    case = NL2SQLCase(
        case_id="one",
        db_id=DB_ID,
        question="What are the names of singers older than 30?",
        query="SELECT name FROM singer WHERE age > 30",
    )
    instance = build_clean_instance(case, db_root)
    assert instance is not None
    # Only the referenced table becomes a source.
    assert instance.sources.names == ["singer"]
    assert list(instance.target_table.columns) == ["name"]
    assert instance.target_schema.column_names == ["name"]
    # Without an LLM the question itself is the table-level specification tau*.
    assert instance.target_schema.description == case.question


def test_clean_instance_rejects_empty_results(db_root: Path) -> None:
    case = NL2SQLCase("empty", DB_ID, "no rows", "SELECT name FROM singer WHERE age > 1000")
    assert build_clean_instance(case, db_root) is None


# --------------------------------------------------------------------------- #
# 2. Rule-based SQL -> pipeline translation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("index", range(len(QUERIES)), ids=lambda i: f"q{i}")
def test_rule_based_translation_reproduces_the_sql_result(instances: list, index: int) -> None:
    """"translate q into an operator pipeline" — verified by execution."""
    instance = instances[index]
    result = translate_sql(instance.case.query, instance.sources, instance.target_table)
    assert result.candidates, "the translator produced no candidate"
    verified = [
        c
        for c in result.candidates
        if verify_pipeline(c, instance.sources, instance.target_table)
    ]
    assert verified, f"no candidate reproduced T* for: {instance.case.query}"
    # Every pipeline ends with the control operator that names the answer table.
    assert all(c[-1].startswith("Terminate(") for c in verified)


def test_translation_records_join_keys(instances: list, cases: list[NL2SQLCase]) -> None:
    """Join keys are what noise injection must not break (Sec 5.3)."""
    index = QUERIES.index(
        "SELECT T1.name, T2.city FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id"
    )
    result = translate_sql(
        cases[index].query, instances[index].sources, instances[index].target_table
    )
    assert result.key_columns["singer"] == {"id"}
    assert result.key_columns["concert"] == {"singer_id"}


def test_translation_refuses_sql_outside_the_covered_subset(instances: list) -> None:
    sources = instances[0].sources
    for sql in (
        "SELECT name FROM singer WHERE id IN (SELECT singer_id FROM concert)",
        "SELECT name FROM singer UNION SELECT city FROM concert",
        "SELECT CASE WHEN age > 40 THEN 1 ELSE 0 END FROM singer",
    ):
        with pytest.raises(TranslationError):
            translate_sql(sql, sources)


def test_search_prefers_the_shortest_verified_candidate(instances: list) -> None:
    """"selecting the shortest one that exactly reproduces T*"."""
    instance = instances[QUERIES.index("SELECT DISTINCT country FROM singer ORDER BY country")]
    result = search_pipeline(instance.case.query, instance.sources, instance.target_table)
    assert result.verified and result.origin == "rule"
    assert result.n_verified >= 2, "the translator should offer more than one valid variant"
    # The ORDER BY is unobservable through the permutation-invariant metric, so
    # the Sort-free variant must win on length.
    assert not any(op.startswith("Sort(") for op in result.pipeline)


# --------------------------------------------------------------------------- #
# 3. The reversibility invariant
# --------------------------------------------------------------------------- #
def test_non_reversible_corruption_is_rejected(toy_tables: TableSet) -> None:
    """The central gate: a corruption its cleaning operator cannot undo is dropped.

    Deleting a record is *not* the inverse of ``Deduplicate``: re-running
    ``Deduplicate`` cannot bring the record back, so the previous table state is
    not restored and the corruption must be refused.
    """
    corruption = Corruption(
        kind="delete_first_row",
        table="t",
        column=None,
        corrupt=lambda df: df.iloc[1:].reset_index(drop=True),
        cleaning="Deduplicate(t, keep=first)",
        cleaning_operator="Deduplicate",
    )
    assert try_corruption(toy_tables, corruption) is None


def test_corruption_with_a_partially_restoring_cleaner_is_rejected(
    toy_tables: TableSet,
) -> None:
    """Even a *nearly* right cleaner is refused; restoration must be exact."""
    corruption = Corruption(
        kind="upper_and_pad",
        table="t",
        column="name",
        # Upper-casing AND padding, but the cleaner only lower-cases.
        corrupt=lambda df: df.assign(name=df["name"].map(lambda v: f" {v.upper()} ")),
        cleaning="ValueTransform(t, 'name', lambda x: str(x).lower())",
        cleaning_operator="ValueTransform",
    )
    assert try_corruption(toy_tables, corruption) is None


def test_dtype_changes_count_as_a_failed_restoration(toy_tables: TableSet) -> None:
    """Values alone are not enough: the source dtype must come back too.

    Rendering integers as text and "cleaning" with a no-op ``ValueTransform``
    leaves cells that *print* identically while the column is now text — which
    would silently break a later numeric aggregation.
    """
    corruption = Corruption(
        kind="int_to_text",
        table="t",
        column="id",
        corrupt=lambda df: df.assign(id=df["id"].map(str)),
        cleaning="ValueTransform(t, 'id', lambda x: str(x))",
        cleaning_operator="ValueTransform",
    )
    assert try_corruption(toy_tables, corruption) is None


def test_no_op_corruption_is_rejected(toy_tables: TableSet) -> None:
    """A corruption that changes nothing would add a pointless cleaning step."""
    corruption = Corruption(
        kind="identity",
        table="t",
        column="name",
        corrupt=lambda df: df.copy(),
        cleaning="ValueTransform(t, 'name', lambda x: str(x).lower())",
        cleaning_operator="ValueTransform",
    )
    assert try_corruption(toy_tables, corruption) is None


def test_reversible_corruption_is_accepted_and_undone(toy_tables: TableSet) -> None:
    before = state_signature(toy_tables)
    corruption = Corruption(
        kind="case_noise",
        table="t",
        column="name",
        corrupt=lambda df: df.assign(name=df["name"].map(lambda v: str(v).upper())),
        cleaning="ValueTransform(t, 'name', lambda x: str(x).lower())",
        cleaning_operator="ValueTransform",
    )
    dirty = try_corruption(toy_tables, corruption)
    assert dirty is not None
    assert dirty["t"].df["name"].tolist() == ["ANN", "BOB", "BOB", "DEE"]
    assert state_signature(apply_cleaning(dirty, [corruption.cleaning])) == before


def test_injected_noise_is_exactly_undone_by_the_cleaning_pipeline(
    toy_tables: TableSet,
) -> None:
    """"produces dirty source tables together with a matching cleaning pipeline"."""
    before = state_signature(toy_tables)
    result = inject_noise(toy_tables, NoiseConfig(max_steps=6, seed=3))
    assert result.accepted, "no corruption was accepted on the toy table"
    assert len(result.cleaning_pipeline) == len(result.accepted)
    assert state_signature(result.sources) != before
    assert state_signature(apply_cleaning(result.sources, result.cleaning_pipeline)) == before


def test_noise_covers_the_cleaning_and_normalization_operators(toy_tables: TableSet) -> None:
    """The built-in library must reach Sec 2.2.1 and Sec 2.2.2, not one operator."""
    seen: set[str] = set()
    for seed in range(25):
        result = inject_noise(toy_tables, NoiseConfig(max_steps=4, seed=seed))
        seen.update(c.cleaning_operator for c in result.accepted)
    assert {"Deduplicate", "ValueTransform", "CastType", "StandardizeDatetime"} <= seen
    assert len(seen) >= 6


def test_noise_never_touches_protected_key_columns(toy_tables: TableSet) -> None:
    """"excessive noise injection may break key fields or relationships"."""
    result = inject_noise(
        toy_tables, NoiseConfig(max_steps=8, seed=5, protect={"t": {"id", "name"}})
    )
    assert result.accepted
    assert all(c.column not in ("id", "name") for c in result.accepted)


def test_row_insertion_can_be_disabled(toy_tables: TableSet) -> None:
    result = inject_noise(
        toy_tables, NoiseConfig(max_steps=6, seed=2, allow_row_insertion=False)
    )
    assert result.accepted
    assert all(len(t.df) == 4 for t in result.sources)


def test_llm_generated_inverse_passes_through_the_same_gate(toy_tables: TableSet) -> None:
    """"we sample an operator and use an LLM to generate its inverse transformation"."""
    good = json.dumps(
        {
            "kind": "prefix_marker",
            "corrupt": "lambda x: '>> ' + str(x)",
            "cleaning": "ValueTransform(t, 'name', lambda x: str(x).removeprefix('>> '))",
        }
    )
    proposal = LLMInverseProposer(ScriptedClient([good])).propose(
        toy_tables["t"], "name", random.Random(0)
    )
    assert proposal is not None
    dirty = try_corruption(toy_tables, proposal)
    assert dirty is not None
    assert dirty["t"].df["name"].tolist()[0] == ">> ann"

    # A hallucinated inverse (destroys the values) costs one rejected attempt.
    bogus = json.dumps(
        {
            "kind": "wipe",
            "corrupt": "lambda x: 'ZZZ'",
            "cleaning": "ValueTransform(t, 'name', lambda x: str(x).lower())",
        }
    )
    bad = LLMInverseProposer(ScriptedClient([bogus])).propose(
        toy_tables["t"], "name", random.Random(0)
    )
    assert bad is not None
    assert try_corruption(toy_tables, bad) is None


# --------------------------------------------------------------------------- #
# 4. End-to-end synthesis
# --------------------------------------------------------------------------- #
def test_synthesis_produces_a_task_for_every_fixture_query(synthesized) -> None:
    tasks, stats = synthesized
    assert stats.n_cases == len(QUERIES)
    assert len(tasks) == len(QUERIES)
    assert stats.n_gold_mismatch == 0
    # Table 1 reports 31 operator types across the synthesized sets; a fixture of
    # this size cannot reach that, but it must exercise more than a handful.
    assert len(stats.operator_types) >= 8


def test_gold_pipeline_reproduces_the_target_on_the_dirty_sources(synthesized) -> None:
    """The property the whole training set depends on, checked on every task.

    ``gold_pipeline`` = cleaning pipeline ++ task pipeline, executed on the
    corrupted sources stored in the task, must reproduce ``T*`` exactly.
    """
    tasks, _ = synthesized
    assert len(tasks) >= 3
    for task in tasks:
        produced, error = execute_pipeline(task.gold_pipeline, task.sources)
        assert error is None, f"{task.task_id}: {error}"
        assert table_match(produced, task.target_table), f"{task.task_id} did not match T*"


def test_gold_pipeline_is_cleaning_then_task(synthesized) -> None:
    """"formed by concatenating the cleaning pipeline with the task pipeline"."""
    tasks, _ = synthesized
    noisy = [t for t in tasks if t.metadata["n_cleaning_ops"] > 0]
    assert noisy, "no task received any noise"
    for task in noisy:
        n_clean = task.metadata["n_cleaning_ops"]
        n_task = task.metadata["n_task_ops"]
        assert n_clean + n_task == len(task.gold_pipeline)
        # The prefix alone restores the clean sources, which is precisely why the
        # suffix (translated from the SQL) still produces T*.
        cleaned = apply_cleaning(task.sources, task.gold_pipeline[:n_clean])
        produced, error = execute_pipeline(task.gold_pipeline[n_clean:], cleaned)
        assert error is None
        assert table_match(produced, task.target_table)


def test_sources_are_actually_dirty(synthesized, instances: list) -> None:
    """The emitted sources must differ from the clean database."""
    tasks, _ = synthesized
    clean = {i.case.query: i.sources for i in instances}
    changed = 0
    for task in tasks:
        if task.metadata["n_cleaning_ops"] == 0:
            continue
        changed += 1
        assert state_signature(task.sources) != state_signature(clean[task.metadata["sql"]])
    assert changed >= 3


def test_synthesis_is_reproducible(db_root: Path, cases: list[NL2SQLCase]) -> None:
    """The same config must yield byte-identical tasks across runs.

    Seeding is derived from a stable checksum of the case id rather than
    ``hash()``, which Python salts per interpreter run.
    """
    config = SynthesisConfig(seed=4, noise=NoiseConfig(max_steps=3, seed=4))
    first, _ = synthesize_dataset(cases[:4], db_root, config=config)
    second, _ = synthesize_dataset(cases[:4], db_root, config=config)
    assert [t.gold_pipeline for t in first] == [t.gold_pipeline for t in second]
    assert [state_signature(t.sources) for t in first] == [
        state_signature(t.sources) for t in second
    ]


def test_task_pipeline_alone_fails_on_the_dirty_sources(synthesized) -> None:
    """Noise must be *load bearing*: without cleaning, the task pipeline breaks.

    Some corruptions (e.g. padding a column that is only projected) survive the
    task pipeline unchanged, so this is asserted over the population rather than
    per task.
    """
    tasks, _ = synthesized
    broken = 0
    for task in tasks:
        n_clean = task.metadata["n_cleaning_ops"]
        if n_clean == 0:
            continue
        produced, error = execute_pipeline(task.gold_pipeline[n_clean:], task.sources)
        if error is not None or not table_match(produced, task.target_table):
            broken += 1
    assert broken >= 1


def test_task_metadata_records_provenance(synthesized) -> None:
    tasks, _ = synthesized
    task = tasks[0]
    assert task.metadata["dataset"] == "Synth-Test"
    assert task.metadata["db_id"] == DB_ID
    assert task.metadata["sql"] in QUERIES
    assert task.metadata["pipeline_origin"] in ("rule", "llm")
    assert task.target_table is not None


def test_synthesis_without_noise_yields_the_task_pipeline_only(
    db_root: Path, cases: list[NL2SQLCase]
) -> None:
    instance = build_clean_instance(cases[0], db_root)
    assert instance is not None
    config = SynthesisConfig(noise=NoiseConfig(max_steps=0))
    task = synthesize_task(instance, config=config)
    assert task is not None
    assert task.metadata["n_cleaning_ops"] == 0
    assert state_signature(task.sources) == state_signature(instance.sources)
    produced, error = execute_pipeline(task.gold_pipeline, task.sources)
    assert error is None
    assert table_match(produced, task.target_table)


def test_jsonl_round_trip_preserves_the_supervision(synthesized, tmp_path: Path) -> None:
    """One ``task.to_dict()`` per line, reloadable by ``deepprep.types.load_tasks``."""
    tasks, _ = synthesized
    path = tmp_path / "synth.jsonl"
    assert write_jsonl(tasks, path) == len(tasks)
    assert len(path.read_text().splitlines()) == len(tasks)

    reloaded = load_tasks(path)
    assert [t.task_id for t in reloaded] == [t.task_id for t in tasks]
    for task in reloaded:
        assert task.gold_pipeline
        produced, error = execute_pipeline(task.gold_pipeline, task.sources)
        assert error is None, f"{task.task_id}: {error}"
        assert table_match(produced, task.target_table)


# --------------------------------------------------------------------------- #
# 5. The optional LLM paths
# --------------------------------------------------------------------------- #
def test_llm_candidate_pipeline_is_used_when_the_translator_bails(instances: list) -> None:
    """The paper's generator takes over for SQL the rule-based path refuses."""
    instance = instances[QUERIES.index("SELECT name FROM singer WHERE age > 30")]
    sql = "SELECT name FROM singer WHERE id IN (SELECT id FROM singer WHERE age > 30)"
    response = """
<pipeline>
Filter(singer, lambda r: r['age'] > 30)
SelectColumn(singer, [name])
Sort(singer, by=[name])
Terminate([singer])
</pipeline>
<pipeline>
Filter(singer, lambda r: r['age'] > 30)
SelectColumn(singer, [name])
Terminate([singer])
</pipeline>
<pipeline>
SelectColumn(singer, [nonexistent])
Terminate([singer])
</pipeline>
"""
    result = search_pipeline(
        sql, instance.sources, instance.target_table, llm=ScriptedClient([response])
    )
    assert result.verified and result.origin == "llm"
    assert result.n_candidates == 3
    assert result.n_verified == 2  # the third candidate fails to execute
    assert len(result.pipeline) == 3  # and the shortest verified one is selected


def test_search_reports_failure_when_no_candidate_matches(instances: list) -> None:
    instance = instances[0]
    response = "<pipeline>\nSelectColumn(singer, [country])\nTerminate([singer])\n</pipeline>"
    result = search_pipeline(
        "SELECT name FROM singer WHERE id IN (SELECT 1)",
        instance.sources,
        instance.target_table,
        llm=ScriptedClient([response]),
    )
    assert not result.verified
    assert result.pipeline == []
    assert result.error


def test_llm_target_schema_keeps_the_columns_of_the_target_table(instances: list) -> None:
    """The model supplies prose; the column set stays dictated by ``T*``."""
    instance = instances[QUERIES.index("SELECT country, COUNT(*) FROM singer GROUP BY country")]
    response = json.dumps(
        {
            "description": "One row per country with how many singers come from it.",
            "columns": [
                {"name": "country", "description": "Country of origin."},
                {"name": "hallucinated", "description": "Not in the target table."},
            ],
        }
    )
    schema = infer_target_schema(
        instance.case, instance.target_table, ScriptedClient([response])
    )
    assert schema.description.startswith("One row per country")
    assert schema.column_names == [str(c) for c in instance.target_table.columns]
    assert schema.get("country").description == "Country of origin."


def test_llm_failures_fall_back_to_the_deterministic_schema(instances: list) -> None:
    instance = instances[0]
    schema = infer_target_schema(
        instance.case, instance.target_table, ScriptedClient(["not json at all"])
    )
    assert schema.description == instance.case.question
    assert schema.column_names == [str(c) for c in instance.target_table.columns]
