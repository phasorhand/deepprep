"""Evaluation metrics (Sec 6.1) and the hybrid reward (Sec 5.2, Eq. 6-8)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from deepprep.agent import DeepPrepAgent
from deepprep.agent.llm import ScriptedClient, Usage
from deepprep.eval import (
    A800_USD_PER_HOUR,
    ApiPricing,
    MatchOptions,
    api_cost,
    evaluate,
    gpu_cost,
    partial_similarity,
    table_match,
)
from deepprep.training.rewards import (
    HeuristicProcessJudge,
    LLMProcessJudge,
    RewardConfig,
    compute_reward,
)

# --------------------------------------------------------------------------- #
# Exact match: "invariant to row and column permutations but requires exact
# cell-value equality" (Sec 6.1)
# --------------------------------------------------------------------------- #
BASE = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_identical_tables_match():
    assert table_match(BASE.copy(), BASE.copy())


def test_row_permutation_does_not_matter():
    assert table_match(BASE.iloc[[2, 0, 1]].reset_index(drop=True), BASE)


def test_column_permutation_does_not_matter():
    assert table_match(BASE[["b", "a"]], BASE)


def test_row_and_column_permutation_together():
    assert table_match(BASE[["b", "a"]].iloc[[1, 2, 0]].reset_index(drop=True), BASE)


def test_a_single_wrong_cell_fails():
    bad = BASE.copy()
    bad.loc[1, "b"] = "WRONG"
    assert not table_match(bad, BASE)


def test_wrong_shape_fails():
    assert not table_match(BASE.head(2), BASE)
    assert not table_match(BASE.assign(c=1), BASE)


def test_duplicate_rows_are_counted_as_multisets():
    a = pd.DataFrame({"a": [1, 1, 2]})
    b = pd.DataFrame({"a": [1, 2, 2]})
    assert not table_match(a, b)
    assert table_match(a, pd.DataFrame({"a": [1, 2, 1]}))


def test_int_and_float_representations_of_the_same_number_match():
    assert table_match(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1.0, 2.0]}))


def test_numeric_string_matches_the_number_it_denotes():
    """A CastType left undone should be scored as the type error it is, not as a
    content error on every row."""
    assert table_match(pd.DataFrame({"a": ["1", "2"]}), pd.DataFrame({"a": [1, 2]}))


def test_float_noise_below_tolerance_matches():
    a = pd.DataFrame({"a": [8.699999999999999, 0.1 + 0.2]})
    b = pd.DataFrame({"a": [8.7, 0.3]})
    assert table_match(a, b)


def test_float_difference_above_tolerance_fails():
    assert not table_match(pd.DataFrame({"a": [8.7]}), pd.DataFrame({"a": [8.8]}))


def test_nulls_of_different_flavours_are_equivalent():
    a = pd.DataFrame({"a": [None, 1.0]})
    b = pd.DataFrame({"a": [np.nan, 1.0]})
    assert table_match(a, b)


def test_columns_are_matched_by_name_when_names_agree():
    """Two columns holding the same values must not be swapped when their names
    say otherwise."""
    gold = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
    swapped_values = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
    assert not table_match(swapped_values, gold)


def test_column_bijection_is_found_when_names_differ():
    gold = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    renamed = pd.DataFrame({"zzz": ["x", "y"], "qqq": [1, 2]})
    assert table_match(renamed, gold)
    assert not table_match(renamed, gold, MatchOptions(require_column_names=True))


def test_none_tables_never_match():
    assert not table_match(None, BASE)
    assert not table_match(BASE, None)


# --------------------------------------------------------------------------- #
# Partial similarity (Eq. 8)
# --------------------------------------------------------------------------- #
def test_partial_similarity_is_one_for_an_exact_match():
    s = partial_similarity(BASE.copy(), BASE.copy())
    assert s["schema"] == pytest.approx(1.0)
    assert s["shape"] == pytest.approx(1.0)
    assert s["content"] == pytest.approx(1.0)
    assert s["partial"] == pytest.approx(1.0)


def test_schema_term_is_the_jaccard_of_column_names():
    # S_sch = |C_That n C_T*| / |C_That u C_T*|
    pred = pd.DataFrame({"a": [1], "c": [1]})
    gold = pd.DataFrame({"a": [1], "b": [1]})
    assert partial_similarity(pred, gold)["schema"] == pytest.approx(1 / 3)


def test_shape_term_follows_the_exponential_formula():
    # S_shp = exp(-| |D_hat| - |D*| | / |D*|)
    pred = pd.DataFrame({"a": [1, 2, 3, 4]})
    gold = pd.DataFrame({"a": [1, 2]})
    assert partial_similarity(pred, gold)["shape"] == pytest.approx(math.exp(-1.0))


def test_content_term_counts_matching_cells_over_matched_columns():
    pred = pd.DataFrame({"a": [1, 2, 99]})
    gold = pd.DataFrame({"a": [1, 2, 3]})
    assert partial_similarity(pred, gold)["content"] == pytest.approx(2 / 3)


def test_partial_similarity_survives_a_row_permutation():
    """The formula is positional, so without canonical ordering a pure shuffle
    would score near zero despite being an exact match."""
    shuffled = BASE.iloc[[2, 0, 1]].reset_index(drop=True)
    assert partial_similarity(shuffled, BASE)["content"] == pytest.approx(1.0)


def test_partial_similarity_of_a_disjoint_table_is_low():
    other = pd.DataFrame({"zzz": ["p", "q", "r", "s", "t", "u"]})
    assert partial_similarity(other, BASE)["partial"] < 0.3


# --------------------------------------------------------------------------- #
# Cost (Sec 6.1)
# --------------------------------------------------------------------------- #
def test_gpu_cost_follows_the_papers_formula():
    # c_i = p_gpu * t_i / 3600, with an A800-80G at $0.91/hour.
    assert gpu_cost(3600.0) == pytest.approx(A800_USD_PER_HOUR)
    assert gpu_cost(60.0) == pytest.approx(0.91 / 60)


def test_api_cost_applies_the_cached_input_rate():
    """Sec 6.1: "we compute API cost using official pricing with caching enabled"."""
    pricing = ApiPricing(input_per_mtok=1.0, output_per_mtok=10.0, cached_input_per_mtok=0.1)
    usage = Usage(prompt_tokens=1_000_000, cached_prompt_tokens=1_000_000, completion_tokens=0)
    assert api_cost(usage, pricing=pricing) == pytest.approx(0.1)

    uncached = Usage(prompt_tokens=1_000_000, cached_prompt_tokens=0, completion_tokens=0)
    assert api_cost(uncached, pricing=pricing) == pytest.approx(1.0)


def test_api_cost_is_zero_for_an_unpriced_model():
    assert api_cost(Usage(prompt_tokens=10), model="some-local-model") == 0.0


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def test_evaluate_reports_the_papers_three_metrics(demo_task, figure4_trajectory):
    agent = DeepPrepAgent(llm=ScriptedClient(figure4_trajectory), max_turns=5)
    report = evaluate(
        agent, [demo_task], method="DeepPrep", dataset="demo", max_workers=1, verbose=False
    )
    assert report.n_cases == 1
    assert report.accuracy == 100.0
    assert report.completion_rate == 100.0
    assert report.avg_backtracks == 1.0
    assert "Acc." in report.summary()


def test_evaluate_isolates_a_crashing_solver(demo_task):
    class Exploding:
        llm = None

        def solve(self, task):
            raise RuntimeError("boom")

    report = evaluate(Exploding(), [demo_task], max_workers=1, verbose=False)
    assert report.accuracy == 0.0
    assert report.cases[0].stop_reason == "solver_error"
    assert "boom" in report.cases[0].error


# --------------------------------------------------------------------------- #
# Hybrid reward (Eq. 6-8)
# --------------------------------------------------------------------------- #
def _solve(task, trajectory, max_turns=5):
    return DeepPrepAgent(llm=ScriptedClient(trajectory), max_turns=max_turns).solve(task)


def test_correct_trajectory_earns_the_outcome_reward(demo_task, figure4_trajectory):
    r = compute_reward(demo_task, _solve(demo_task, figure4_trajectory), judge=HeuristicProcessJudge())
    assert r.r_out == 1.0
    assert r.r_part == 0.0  # Eq. 8 is the fallback for R_out == 0
    assert r.total > 0.7


def test_incorrect_trajectory_falls_back_to_the_partial_reward(demo_task):
    responses = [
        "<plan>p</plan><expand><from></from><ops>Deduplicate(movies, [id], first)</ops></expand>"
    ] * 3
    r = compute_reward(demo_task, _solve(demo_task, responses, max_turns=3))
    assert r.r_out == 0.0
    assert 0.0 < r.r_part < 1.0


def test_a_correct_trajectory_always_outranks_an_incorrect_one(demo_task, figure4_trajectory):
    """Otherwise the process term would itself be hackable: a wrong answer with
    beautiful reasoning must never beat a right one."""
    cfg = RewardConfig()
    good = compute_reward(demo_task, _solve(demo_task, figure4_trajectory), cfg, HeuristicProcessJudge())
    bad = compute_reward(
        demo_task,
        _solve(demo_task, ["<plan>p</plan><expand><from></from><ops>Sort(movies,[id])</ops></expand>"] * 3, 3),
        cfg,
        HeuristicProcessJudge(),
    )
    # Even granting the bad trajectory a perfect process score:
    best_possible_bad = cfg.beta * 1.0 + cfg.gamma * 1.0
    assert cfg.alpha * 1.0 > best_possible_bad
    assert good.total > bad.total


def test_format_penalty_applies_to_unparseable_turns(demo_task):
    r = compute_reward(demo_task, _solve(demo_task, ["not a tagged response"] * 2, 2))
    assert r.format_penalty > 0


def test_heuristic_judge_rewards_a_justified_backtrack(demo_task, figure4_trajectory):
    score = HeuristicProcessJudge().score(demo_task, _solve(demo_task, figure4_trajectory))
    assert score.backtracking_justification == 1.0
    assert score.feedback_responsiveness > 0.5


def test_heuristic_judge_penalizes_an_unjustified_branch_switch(demo_task):
    """Sec 5.2 criterion 3: a parent-node switch must be "supported by recorded
    failure evidence from the current branch"."""
    responses = [
        "<plan>start</plan><expand><from></from><ops>Deduplicate(movies, [id], first)</ops></expand>",
        # Jumps back to the root with no failure to justify it and no explanation.
        "<plan>let me try something else</plan><expand><from></from><ops>Sort(movies, [id])</ops></expand>",
    ]
    result = _solve(demo_task, responses, max_turns=2)
    assert result.n_backtracks >= 1
    score = HeuristicProcessJudge().score(demo_task, result)
    assert score.backtracking_justification == 0.0


def test_llm_judge_parses_a_json_verdict(demo_task, figure4_trajectory):
    verdict = (
        '{"plan_action_consistency": 0.9, "feedback_responsiveness": 0.8, '
        '"backtracking_justification": 1.0, "rationale": "clean recovery"}'
    )
    judge = LLMProcessJudge(ScriptedClient([verdict]))
    score = judge.score(demo_task, _solve(demo_task, figure4_trajectory))
    assert score.plan_action_consistency == 0.9
    assert score.value == pytest.approx((0.9 + 0.8 + 1.0) / 3)


def test_llm_judge_falls_back_when_the_judge_misbehaves(demo_task, figure4_trajectory):
    """A judge outage must not silently zero R_llm for a whole training batch."""
    judge = LLMProcessJudge(ScriptedClient(["I refuse to answer in JSON."]))
    score = judge.score(demo_task, _solve(demo_task, figure4_trajectory))
    assert score.rationale == "heuristic structural judge"
    assert score.value > 0


def test_llm_judge_does_not_see_the_ground_truth(demo_task, figure4_trajectory):
    canary = "ZZ_CANARY_ZZ"
    demo_task.target_table = demo_task.target_table.copy()
    demo_task.target_table["director_name"] = canary
    client = ScriptedClient(['{"plan_action_consistency": 1, "feedback_responsiveness": 1, '
                            '"backtracking_justification": 1, "rationale": "ok"}'])
    LLMProcessJudge(client).score(demo_task, _solve(demo_task, figure4_trajectory))
    assert canary not in "\n".join(m["content"] for c in client.calls for m in c)
