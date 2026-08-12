"""The documented examples actually run.

Every other test exercises the library through Python imports.  That leaves the
one path a new reader takes first -- copy the command out of the README, paste
it into a shell -- completely uncovered, and it is the path most likely to rot:
a renamed flag or a moved asset breaks it without failing a single test.

These tests run the two documented entry points as a user would:

* ``deepprep demo`` -- the offline Figure-4 replay, no API key,
* ``python examples/mini_spider/build_db.py`` then ``synthesize`` -- the whole
  Sec 5.3 path on a fixture that ships with the repo.

They are deliberately end-to-end and therefore slow-ish (a couple of seconds);
that is the point.  Nothing here touches the network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from deepprep.cli import main
from deepprep.eval.metrics import table_match
from deepprep.synthesis import (
    SynthesisConfig,
    execute_pipeline,
    load_benchmark,
    synthesize_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DB = REPO_ROOT / "examples" / "mini_spider" / "build_db.py"


def _build_db(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BUILD_DB), *argv],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


# --------------------------------------------------------------------------- #
# the fixture builder's own CLI
# --------------------------------------------------------------------------- #
def test_the_fixture_builder_accepts_an_out_flag(tmp_path):
    """`--out DIR` is the obvious spelling, so it must not be taken as a path.

    Reading `sys.argv[1]` unconditionally turned a mistyped flag into a
    *directory literally named `--out`*, silently written next to the sources,
    while the intended destination stayed empty.
    """
    out = tmp_path / "mini_spider"
    proc = _build_db("--out", str(out))

    assert proc.returncode == 0, proc.stderr
    assert (out / "spec.json").exists()
    assert not (REPO_ROOT / "--out").exists(), "wrote a directory named after the flag"


def test_the_fixture_builder_still_accepts_a_positional_path(tmp_path):
    """The README documents the positional form; it must keep working."""
    out = tmp_path / "mini_spider"
    proc = _build_db(str(out))

    assert proc.returncode == 0, proc.stderr
    assert (out / "spec.json").exists()


def test_an_unknown_flag_is_an_error_not_a_directory_name(tmp_path):
    proc = _build_db("--nonsense", str(tmp_path / "x"))

    assert proc.returncode != 0
    assert not (REPO_ROOT / "--nonsense").exists()


# --------------------------------------------------------------------------- #
# the fixture itself
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mini_spider(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("mini_spider")
    proc = _build_db(str(out))
    assert proc.returncode == 0, proc.stderr
    return out


def test_the_fixture_has_spider_layout(mini_spider):
    """`db_root/<db_id>/<db_id>.sqlite` is what `synthesize --db-root` expects."""
    spec = json.loads((mini_spider / "spec.json").read_text())
    assert spec

    for case in spec:
        db_id = case["db_id"]
        assert (mini_spider / "database" / db_id / f"{db_id}.sqlite").exists()
        assert case["question"] and case["query"]


def test_the_documented_synthesis_command_produces_sound_tasks(mini_spider):
    """The README's second command, end to end, with the paper's own guarantee.

    Sec 5.3: the gold pipeline is the cleaning pipeline concatenated with the
    task pipeline, and executing it on the *dirty* sources must reproduce `T*`.
    A task that fails this is unusable as training data.
    """
    cases = load_benchmark(mini_spider / "spec.json")
    tasks, _ = synthesize_dataset(
        cases,
        db_root=mini_spider / "database",
        config=SynthesisConfig(dataset="mini_spider", seed=0),
    )

    assert len(tasks) >= 8, f"only {len(tasks)}/{len(cases)} cases synthesized"
    for task in tasks:
        produced, error = execute_pipeline(task.gold_pipeline, task.sources)
        assert error is None, f"{task.task_id}: {error}"
        assert table_match(produced, task.target_table), f"{task.task_id} did not match T*"


# --------------------------------------------------------------------------- #
# the quickstart command
# --------------------------------------------------------------------------- #
def test_the_demo_command_solves_the_paper_example(capsys):
    """`deepprep demo` is the first thing the README tells you to run."""
    code = main(["demo"])
    out = capsys.readouterr().out

    assert code == 0
    assert "completed=True" in out
    assert "exact match=True" in out
