"""Evaluation metrics and harness (paper Sec 6.1)."""

from .evaluate import CaseResult, EvalReport, Solver, evaluate, report_table
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

__all__ = [
    "A800_USD_PER_HOUR",
    "API_PRICING",
    "ApiPricing",
    "CaseResult",
    "EvalReport",
    "MatchOptions",
    "Solver",
    "api_cost",
    "evaluate",
    "gpu_cost",
    "partial_similarity",
    "report_table",
    "table_match",
]
