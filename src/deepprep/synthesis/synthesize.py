"""The synthesis driver (paper Sec 5.3, "Data Synthesis for Training").

    "To train our agentic model, we require supervision in the form of executable
     data preparation pipelines together with their source and target tables.
     Since such ADP task instances are not available at scale, we construct them
     through synthesis by converting SQL benchmarks into ADP tasks consisting of
     source tables, a target table, and a corresponding transformation pipeline."

This module chains the three stages into :class:`~deepprep.types.ADPTask` objects:

1. :mod:`.nl2sql` — execute ``q`` to get ``T*``, load the sources, specify ``Sigma*``.
2. :mod:`.pipeline_search` — "translate q into an operator pipeline ... selecting
   the shortest one that exactly reproduces T* through execution".
3. :mod:`.noise` — reversible corruption of the sources.

    "The final ground-truth pipeline is formed by concatenating the cleaning
     pipeline with the task pipeline."

Every emitted task is re-verified end to end: the concatenated pipeline is
executed on the *dirty* sources and must reproduce ``T*`` under the paper's
exact-match metric.  Reversibility already implies this, so the check is cheap
insurance against a bug in any of the three stages leaking into training data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..agent.llm import LLMClient
from ..eval.metrics import MatchOptions, table_match
from ..operators import parse_operator_call
from ..types import ADPTask
from .nl2sql import CleanInstance, NL2SQLCase, build_clean_instance, load_benchmark
from .noise import NoiseConfig, NoiseResult, inject_noise
from .pipeline_search import PipelineSearchResult, execute_pipeline, search_pipeline

__all__ = [
    "SynthesisConfig",
    "SynthesisStats",
    "main",
    "synthesize_dataset",
    "synthesize_task",
    "write_jsonl",
]


@dataclass
class SynthesisConfig:
    """Knobs of the synthesis run."""

    #: Dataset label; the paper builds "Synth-Spider" and "Synth-Bird" (Table 1).
    dataset: str = "Synth-Spider"
    split: str = "train"
    #: Skip cases whose sources or target are too large to serialize as a task.
    max_rows_per_table: int = 20_000
    max_target_rows: int = 2_000
    #: Number of pipelines to ask the LLM for, per case.
    n_llm_candidates: int = 4
    #: Query the LLM even when the rule-based translator succeeded, so the
    #: "shortest verified candidate" is chosen from both generators.
    use_llm_always: bool = False
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    #: Drop cases where noise injection accepted nothing (they add no cleaning
    #: supervision).  Off by default: clean cases are still valid ADP tasks.
    require_noise: bool = False
    seed: int = 0
    match_options: MatchOptions | None = None


@dataclass
class SynthesisStats:
    """Counters explaining why cases were dropped — the run's audit trail."""

    n_cases: int = 0
    n_instances: int = 0
    n_pipelines: int = 0
    n_tasks: int = 0
    n_no_instance: int = 0
    n_no_pipeline: int = 0
    n_gold_mismatch: int = 0
    n_no_noise: int = 0
    n_corruptions: int = 0
    n_corruptions_rejected: int = 0
    pipeline_lengths: list[int] = field(default_factory=list)
    operator_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        lengths = sorted(self.pipeline_lengths)
        return {
            "n_cases": self.n_cases,
            "n_instances": self.n_instances,
            "n_pipelines": self.n_pipelines,
            "n_tasks": self.n_tasks,
            "n_no_instance": self.n_no_instance,
            "n_no_pipeline": self.n_no_pipeline,
            "n_gold_mismatch": self.n_gold_mismatch,
            "n_no_noise": self.n_no_noise,
            "n_corruptions": self.n_corruptions,
            "n_corruptions_rejected": self.n_corruptions_rejected,
            "pipeline_length_min": lengths[0] if lengths else 0,
            "pipeline_length_max": lengths[-1] if lengths else 0,
            "pipeline_length_mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "n_operator_types": len(self.operator_types),
            "operator_types": dict(sorted(self.operator_types.items())),
        }

    def summary(self) -> str:
        d = self.to_dict()
        return (
            f"{d['n_tasks']}/{d['n_cases']} cases synthesized  "
            f"(no instance {d['n_no_instance']}, no pipeline {d['n_no_pipeline']}, "
            f"gold mismatch {d['n_gold_mismatch']}, no noise {d['n_no_noise']})\n"
            f"  pipeline length {d['pipeline_length_min']}~{d['pipeline_length_max']} "
            f"(mean {d['pipeline_length_mean']:.1f}), {d['n_operator_types']} operator types\n"
            f"  corruptions accepted {d['n_corruptions']}, "
            f"rejected {d['n_corruptions_rejected']}"
        )


def _operator_types(pipeline: Sequence[str]) -> list[str]:
    names: list[str] = []
    for src in pipeline:
        try:
            names.append(parse_operator_call(src).name)
        except Exception:  # noqa: BLE001 - stats must never break a run
            continue
    return names


def synthesize_task(
    instance: CleanInstance,
    *,
    config: SynthesisConfig | None = None,
    llm: LLMClient | None = None,
    stats: SynthesisStats | None = None,
) -> ADPTask | None:
    """Turn one clean instance into an ADP task with a dirty source table set.

    Returns ``None`` when no verified task pipeline exists for the query, or when
    the concatenated gold pipeline fails the final end-to-end check.
    """
    cfg = config or SynthesisConfig()
    st = stats or SynthesisStats()

    search: PipelineSearchResult = search_pipeline(
        instance.case.query,
        instance.sources,
        instance.target_table,
        llm=llm,
        n_llm_candidates=cfg.n_llm_candidates,
        use_llm_always=cfg.use_llm_always,
        match_options=cfg.match_options,
    )
    if not search.verified:
        st.n_no_pipeline += 1
        return None
    st.n_pipelines += 1

    # "Excessive noise injection may break key fields or relationships": the join
    # keys the task pipeline depends on are shielded from corruption.
    noise_cfg = replace(cfg.noise)
    if noise_cfg.protect is None:
        noise_cfg.protect = search.key_columns
    noise_cfg.seed = _case_seed(cfg.seed, instance.case)

    noise: NoiseResult = inject_noise(instance.sources, noise_cfg, llm=llm)
    st.n_corruptions += len(noise.accepted)
    st.n_corruptions_rejected += noise.n_rejected
    if cfg.require_noise and not noise.accepted:
        st.n_no_noise += 1
        return None
    if not noise.accepted:
        st.n_no_noise += 1

    # "The final ground-truth pipeline is formed by concatenating the cleaning
    #  pipeline with the task pipeline."
    gold = list(noise.cleaning_pipeline) + list(search.pipeline)

    produced, error = execute_pipeline(gold, noise.sources)
    if error is not None or not table_match(produced, instance.target_table, cfg.match_options):
        st.n_gold_mismatch += 1
        return None

    ops = _operator_types(gold)
    for name in ops:
        st.operator_types[name] = st.operator_types.get(name, 0) + 1
    st.pipeline_lengths.append(len(gold))
    st.n_tasks += 1

    case = instance.case
    return ADPTask(
        task_id=f"{cfg.dataset.lower()}_{case.case_id}",
        sources=noise.sources,
        target_schema=instance.target_schema,
        target_table=instance.target_table,
        gold_pipeline=gold,
        metadata={
            "dataset": cfg.dataset,
            "split": cfg.split,
            "db_id": case.db_id,
            "question": case.question,
            "sql": case.query,
            "evidence": case.evidence,
            "difficulty": case.difficulty,
            "source_tables": list(noise.sources.names),
            "pipeline_origin": search.origin,
            "n_candidates": search.n_candidates,
            "n_verified_candidates": search.n_verified,
            "n_cleaning_ops": len(noise.cleaning_pipeline),
            "n_task_ops": len(search.pipeline),
            "pipeline_length": len(gold),
            "operator_types": sorted(set(ops)),
            "noise": [c.to_dict() for c in noise.accepted],
            "protected_columns": {k: sorted(v) for k, v in (search.key_columns or {}).items()},
        },
    )


def _case_seed(base: int, case: NL2SQLCase) -> int:
    """Per-case deterministic seed, so one task's noise never depends on the others.

    ``hash()`` on a ``str`` is salted per interpreter run, which would make two
    invocations of the synthesizer emit different corruptions for the same case;
    CRC32 keeps the dataset reproducible.
    """
    return (base * 1_000_003 + zlib.crc32(case.case_id.encode())) % (2**31)


def synthesize_dataset(
    benchmark: str | Path | Sequence[NL2SQLCase],
    db_root: str | Path,
    *,
    config: SynthesisConfig | None = None,
    llm: LLMClient | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    on_task: Any = None,
    verbose: bool = False,
) -> tuple[list[ADPTask], SynthesisStats]:
    """Synthesize a whole ADP dataset from an NL2SQL benchmark.

    ``benchmark`` is either a path to a Spider/BIRD JSON file or an already
    loaded list of cases (which is what the offline tests use).
    """
    cfg = config or SynthesisConfig()
    cases: list[NL2SQLCase] = (
        list(benchmark)
        if isinstance(benchmark, (list, tuple))
        else load_benchmark(benchmark)  # type: ignore[arg-type]
    )
    if shuffle:
        random.Random(cfg.seed).shuffle(cases)
    if limit is not None:
        cases = cases[:limit]

    stats = SynthesisStats(n_cases=len(cases))
    tasks: list[ADPTask] = []
    for case in cases:
        instance = build_clean_instance(
            case,
            db_root,
            llm=llm,
            max_rows_per_table=cfg.max_rows_per_table,
            max_target_rows=cfg.max_target_rows,
        )
        if instance is None:
            stats.n_no_instance += 1
            continue
        stats.n_instances += 1
        task = synthesize_task(instance, config=cfg, llm=llm, stats=stats)
        if task is None:
            continue
        tasks.append(task)
        if on_task is not None:
            on_task(task)
        if verbose:
            print(f"  [{len(tasks)}] {task.task_id}  |P|={len(task.gold_pipeline)}", flush=True)
    return tasks, stats


def write_jsonl(tasks: Iterable[ADPTask], path: str | Path) -> int:
    """Write one ``ADPTask.to_dict()`` per line; returns the number of tasks."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task.to_dict(), ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """``python -m deepprep.synthesis --benchmark ... --db-root ... --out ...``"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True, help="Spider/BIRD JSON or JSONL file")
    parser.add_argument("--db-root", required=True, help="directory holding the SQLite databases")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--dataset", default="Synth-Spider")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-steps", type=int, default=5)
    parser.add_argument("--require-noise", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI-compatible model to use for schema/pipeline/inverse generation; "
        "omit to run fully offline",
    )
    parser.add_argument("--stats", default=None, help="optional path for the run statistics JSON")
    args = parser.parse_args(argv)

    llm: LLMClient | None = None
    if args.model:
        from ..agent.llm import OpenAIClient

        llm = OpenAIClient(model=args.model)

    cfg = SynthesisConfig(
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        require_noise=args.require_noise,
        noise=NoiseConfig(max_steps=args.noise_steps, seed=args.seed),
    )
    tasks, stats = synthesize_dataset(
        args.benchmark,
        args.db_root,
        config=cfg,
        llm=llm,
        limit=args.limit,
        shuffle=args.shuffle,
        verbose=True,
    )
    write_jsonl(tasks, args.out)
    print(stats.summary())
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
