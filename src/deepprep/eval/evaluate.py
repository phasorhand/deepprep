"""Evaluation harness (paper Sec 6.1-6.2).

Runs any solver — :class:`~deepprep.agent.DeepPrepAgent` or one of the baselines
in :mod:`deepprep.baselines` — over a task set and reports the paper's three
metrics: Exact Match Accuracy, Completion Rate, and average Inference Cost.

Cases are executed with a thread pool because the paper measures open-source cost
as "per-case end-to-end runtime *under parallel requests*"; running them serially
would inflate the GPU-hour cost estimate.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from ..agent.agent import SolveResult
from ..types import ADPTask
from .metrics import (
    A800_USD_PER_HOUR,
    API_PRICING,
    ApiPricing,
    MatchOptions,
    api_cost,
    gpu_cost,
    partial_similarity,
    table_match,
)

__all__ = ["CaseResult", "EvalReport", "Solver", "evaluate"]


class Solver(Protocol):
    """Anything that turns an :class:`ADPTask` into a :class:`SolveResult`."""

    def solve(self, task: ADPTask) -> SolveResult: ...


@dataclass
class CaseResult:
    task_id: str
    correct: bool
    completed: bool
    cost_usd: float
    n_turns: int
    elapsed_s: float
    stop_reason: str
    n_backtracks: int = 0
    n_failed_expansions: int = 0
    pipeline: list[str] = field(default_factory=list)
    partial: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "correct": self.correct,
            "completed": self.completed,
            "cost_usd": self.cost_usd,
            "n_turns": self.n_turns,
            "elapsed_s": round(self.elapsed_s, 4),
            "stop_reason": self.stop_reason,
            "n_backtracks": self.n_backtracks,
            "n_failed_expansions": self.n_failed_expansions,
            "pipeline": self.pipeline,
            "partial": {k: round(v, 4) for k, v in self.partial.items()},
            "error": self.error,
            "usage": self.usage,
        }


@dataclass
class EvalReport:
    """Aggregated results in the shape of the paper's Table 2."""

    method: str
    model: str
    dataset: str
    n_cases: int
    #: "Exact Match Accuracy (Acc.)" — reported on a 0-100 scale, as in Table 2.
    accuracy: float
    #: "Completion Rate (Comp.)" — percentage.
    completion_rate: float
    #: "Inference Cost (Cost)" — mean USD per case.
    avg_cost_usd: float
    avg_turns: float
    avg_elapsed_s: float
    avg_backtracks: float
    #: Mean of the Eq. (8) partial similarity — useful for ablation diagnosis.
    avg_partial: float
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self, with_cases: bool = True) -> dict[str, Any]:
        d = {
            "method": self.method,
            "model": self.model,
            "dataset": self.dataset,
            "n_cases": self.n_cases,
            "accuracy": round(self.accuracy, 2),
            "completion_rate": round(self.completion_rate, 2),
            "avg_cost_usd": self.avg_cost_usd,
            "avg_turns": round(self.avg_turns, 3),
            "avg_elapsed_s": round(self.avg_elapsed_s, 3),
            "avg_backtracks": round(self.avg_backtracks, 3),
            "avg_partial": round(self.avg_partial, 4),
        }
        if with_cases:
            d["cases"] = [c.to_dict() for c in self.cases]
        return d

    def summary(self) -> str:
        return (
            f"{self.method} / {self.model} on {self.dataset} (n={self.n_cases})\n"
            f"  Acc.   {self.accuracy:6.2f}\n"
            f"  Comp.  {self.completion_rate:6.2f}%\n"
            f"  Cost   ${self.avg_cost_usd:.6f} per case\n"
            f"  turns  {self.avg_turns:.2f}   backtracks {self.avg_backtracks:.2f}   "
            f"partial {self.avg_partial:.3f}"
        )

    def save(self, path: str | Path, with_cases: bool = True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(with_cases), indent=2, default=str))


def _case_cost(
    result: SolveResult,
    cost_mode: str,
    model: str,
    pricing: ApiPricing | None,
    gpu_usd_per_hour: float,
) -> float:
    if cost_mode == "api":
        return api_cost(result.usage, model=model, pricing=pricing)
    if cost_mode == "gpu":
        # Sec 6.1 uses end-to-end runtime; LLM wall time is the dominant term but
        # environment execution is part of the case, so charge the whole case.
        return gpu_cost(result.elapsed_s, gpu_usd_per_hour)
    return 0.0


def evaluate(
    solver: Solver,
    tasks: Sequence[ADPTask],
    *,
    method: str = "DeepPrep",
    model: str = "",
    dataset: str = "",
    cost_mode: str = "auto",
    pricing: ApiPricing | None = None,
    gpu_usd_per_hour: float = A800_USD_PER_HOUR,
    match_options: MatchOptions | None = None,
    max_workers: int = 4,
    on_case: Callable[[CaseResult], None] | None = None,
    verbose: bool = True,
) -> EvalReport:
    """Evaluate ``solver`` on ``tasks``.

    ``cost_mode``:
      * ``"api"``  — official token pricing with caching (closed-source backbones)
      * ``"gpu"``  — ``c_i = p_gpu * t_i / 3600`` (self-hosted open-source backbones)
      * ``"auto"`` — ``api`` when the model has a published price, else ``gpu``
    """
    model = model or getattr(getattr(solver, "llm", None), "model", "") or ""
    if cost_mode == "auto":
        cost_mode = "api" if (pricing or model in API_PRICING) else "gpu"

    opts = match_options or MatchOptions()
    # Keyed by position, not task_id: duplicate ids in a task file would otherwise
    # collapse into a single case and silently shrink the denominator.
    results: dict[int, CaseResult] = {}
    t0 = time.perf_counter()

    def run_one(task: ADPTask) -> CaseResult:
        try:
            res = solver.solve(task)
        except Exception as e:  # noqa: BLE001 - one bad case must not kill the run
            return CaseResult(
                task_id=task.task_id, correct=False, completed=False, cost_usd=0.0,
                n_turns=0, elapsed_s=0.0, stop_reason="solver_error",
                error=f"{type(e).__name__}: {e}",
            )

        produced = res.table if res.table is not None else res.fallback_table
        correct = bool(
            res.completed and table_match(res.table, task.target_table, opts)
        )
        return CaseResult(
            task_id=task.task_id,
            correct=correct,
            completed=res.completed,
            cost_usd=_case_cost(res, cost_mode, model, pricing, gpu_usd_per_hour),
            n_turns=res.n_turns,
            elapsed_s=res.elapsed_s,
            stop_reason=res.stop_reason,
            n_backtracks=res.n_backtracks,
            n_failed_expansions=res.n_failed_expansions,
            pipeline=[op.to_source() for op in res.pipeline],
            partial=partial_similarity(produced, task.target_table, opts),
            error=res.error,
            usage=res.usage.to_dict(),
        )

    if max_workers <= 1:
        for i, task in enumerate(tasks):
            cr = run_one(task)
            results[i] = cr
            if on_case:
                on_case(cr)
            if verbose:
                _progress(i + 1, len(tasks), results.values())
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_one, t): i for i, t in enumerate(tasks)}
            for done, fut in enumerate(as_completed(futures)):
                cr = fut.result()
                results[futures[fut]] = cr
                if on_case:
                    on_case(cr)
                if verbose:
                    _progress(done + 1, len(tasks), results.values())
    if verbose:
        print()

    # Preserve the input order for reproducible reports.
    cases = [results[i] for i in range(len(tasks)) if i in results]
    n = len(cases) or 1
    report = EvalReport(
        method=method,
        model=model,
        dataset=dataset,
        n_cases=len(cases),
        accuracy=100.0 * sum(c.correct for c in cases) / n,
        completion_rate=100.0 * sum(c.completed for c in cases) / n,
        avg_cost_usd=sum(c.cost_usd for c in cases) / n,
        avg_turns=sum(c.n_turns for c in cases) / n,
        avg_elapsed_s=sum(c.elapsed_s for c in cases) / n,
        avg_backtracks=sum(c.n_backtracks for c in cases) / n,
        avg_partial=sum(c.partial.get("partial", 0.0) for c in cases) / n,
        cases=cases,
    )
    report.to_dict(with_cases=False)["wall_time_s"] = time.perf_counter() - t0
    return report


def _progress(done: int, total: int, cases: Any) -> None:  # pragma: no cover
    cases = list(cases)
    acc = 100.0 * sum(c.correct for c in cases) / max(len(cases), 1)
    comp = 100.0 * sum(c.completed for c in cases) / max(len(cases), 1)
    print(f"\r  {done}/{total}  acc={acc:5.2f}  comp={comp:5.2f}%", end="", flush=True)


def report_table(reports: Sequence[EvalReport]) -> pd.DataFrame:
    """Assemble several reports into a Table-2-shaped DataFrame."""
    return pd.DataFrame(
        [
            {
                "Method": r.method,
                "Model": r.model,
                "Dataset": r.dataset,
                "Acc.": round(r.accuracy, 2),
                "Comp.": f"{r.completion_rate:.2f}%",
                "Cost": f"${r.avg_cost_usd:.6f}",
            }
            for r in reports
        ]
    )
