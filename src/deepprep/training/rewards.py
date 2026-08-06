"""Hybrid reward (paper Sec 5.2, Eq. 6-8).

    "Using only final pipeline correctness as the reward, however, results in
     sparse and delayed learning signals. We therefore introduce a hybrid reward
     design ...

        R(tau) = alpha * R_out(tau) + beta * R_part(tau) + gamma * R_llm(tau)   (6)

     where R_out, R_part, and R_llm correspond to strict correctness, partial
     progress, and reasoning process signals, respectively."

**Outcome reward** (Eq. 7) — ``R_out(tau) = I(T_hat == T*)``, evaluated with the
permutation-invariant table match of Sec 6.1.

**Partial reward** (Eq. 8) — active only when ``R_out(tau) = 0``:
``R_part = avg(S_sch, S_shp, S_cnt)``.

**Process reward** ``R_llm`` — an LLM judge over the trajectory.  Its purpose is
anti-reward-hacking:

    "Relying solely on data similarity rewards may lead to reward hacking ... and
     can also arise in ADP scenarios (e.g. renaming columns without performing
     the required data cleaning). To discourage such 'shortcut' strategies ... we
     introduce a process reward that evaluates the quality of the reasoning
     trajectory. We use an instruction-tuned LLM (e.g. GPT-4o) as a judge to
     score each trajectory based on the actions defined in Section 4.

     Specifically, the judge evaluates three criteria: (1) *Plan-action
     consistency*, measuring whether generated pipeline (from <expand>) correctly
     implements the plan (from <plan>). (2) *Feedback responsiveness*, checking
     whether subsequent <plan> outputs explicitly respond to the prior execution
     feedback (e.g. reported error traces). (3) *Backtracking justification*,
     verifying that parent-node switches are supported by recorded failure
     evidence from the current branch."

``R_llm`` is used twice: as the third reward term here, and as the filter that
selects teacher trajectories for cold-start Stage 2 (Sec 5.1).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from ..agent.agent import SolveResult
from ..agent.llm import LLMClient
from ..eval.metrics import MatchOptions, partial_similarity, table_match
from ..types import ADPTask

__all__ = [
    "HeuristicProcessJudge",
    "LLMProcessJudge",
    "ProcessJudge",
    "ProcessScore",
    "RewardConfig",
    "RewardBreakdown",
    "compute_reward",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    """Weights of Eq. (6).

    The paper does not publish alpha/beta/gamma.  The defaults keep the reward in
    [0, 1] and preserve the intended ordering: a strictly correct trajectory must
    outrank any incorrect one no matter how good its process score, otherwise the
    process term would itself become hackable.  With these values a correct run
    scores >= 0.7 and an incorrect one at most 0.3 + 0.15 = 0.45.
    """

    alpha: float = 0.7   # R_out  -- strict correctness
    beta: float = 0.3    # R_part -- partial progress (only when R_out == 0)
    gamma: float = 0.15  # R_llm  -- reasoning process
    #: Eq. (8) is defined as the fallback for R_out == 0; set False to always add it.
    partial_only_when_incorrect: bool = True
    #: Score the fallback table when the agent never emitted <answer>.  Keeps a
    #: gradient on trajectories that made real progress but ran out of turns.
    score_fallback_table: bool = True
    #: Small negative shaping for unparseable actions, which are pure waste.
    format_penalty: float = 0.05
    match_options: MatchOptions = field(default_factory=MatchOptions)


@dataclass
class ProcessScore:
    """The three criteria of ``R_llm``, each in [0, 1]."""

    plan_action_consistency: float = 0.0
    feedback_responsiveness: float = 0.0
    backtracking_justification: float = 0.0
    rationale: str = ""

    @property
    def value(self) -> float:
        return (
            self.plan_action_consistency
            + self.feedback_responsiveness
            + self.backtracking_justification
        ) / 3.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_action_consistency": round(self.plan_action_consistency, 4),
            "feedback_responsiveness": round(self.feedback_responsiveness, 4),
            "backtracking_justification": round(self.backtracking_justification, 4),
            "value": round(self.value, 4),
            "rationale": self.rationale,
        }


@dataclass
class RewardBreakdown:
    """All terms of Eq. (6), retained for logging and ablation (Table 3b)."""

    r_out: float = 0.0
    r_part: float = 0.0
    r_llm: float = 0.0
    total: float = 0.0
    format_penalty: float = 0.0
    similarity: dict[str, float] = field(default_factory=dict)
    process: ProcessScore | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_out": round(self.r_out, 4),
            "r_part": round(self.r_part, 4),
            "r_llm": round(self.r_llm, 4),
            "format_penalty": round(self.format_penalty, 4),
            "total": round(self.total, 4),
            "similarity": {k: round(v, 4) for k, v in self.similarity.items()},
            "process": self.process.to_dict() if self.process else None,
        }


# --------------------------------------------------------------------------- #
# Process judges
# --------------------------------------------------------------------------- #
class ProcessJudge(Protocol):
    """Scores a trajectory on the three criteria of Sec 5.2."""

    def score(self, task: ADPTask, result: SolveResult) -> ProcessScore: ...


_ANTI_HACK_PATTERNS = (
    # "renaming columns without performing the required data cleaning" -- the
    # exact shortcut the paper names.
    re.compile(r"\brename\b", re.IGNORECASE),
)


class HeuristicProcessJudge:
    """A deterministic approximation of ``R_llm``.

    Not a substitute for the paper's LLM judge — it cannot assess *semantic*
    plan/action agreement.  It exists so that RL, trajectory filtering and the
    tests all run without an API key, and as a cheap prefilter that removes
    obviously bad trajectories before spending judge calls on the rest.

    The three criteria are approximated structurally:

    * **plan-action consistency** — did the turn emit a plan at all, and did the
      pipeline it proposed actually execute?
    * **feedback responsiveness** — after a failed turn, does the next plan
      mention the error (its type, the named column/table, or a repair word)?
    * **backtracking justification** — is every parent-node switch preceded by a
      recorded failure on the branch being abandoned?
    """

    _ERROR_WORDS = re.compile(
        r"\b(error|fail|failed|missing|empty|invalid|because|since|instead|"
        r"backtrack|rollback|revert|revise|wrong|caused)\b",
        re.IGNORECASE,
    )

    def score(self, task: ADPTask, result: SolveResult) -> ProcessScore:
        steps = result.trajectory
        if not steps:
            return ProcessScore(rationale="empty trajectory")

        # (1) Plan-action consistency.
        consistent = 0.0
        for s in steps:
            if s.error is not None and s.n_ops_applied == 0:
                continue  # nothing executed: neither consistent nor inconsistent
            has_plan = "<plan>" in s.response.lower()
            acted = s.n_ops_applied > 0 or "<answer>" in s.response.lower()
            consistent += 1.0 if (has_plan and acted) else (0.5 if acted else 0.0)
        c1 = consistent / len(steps)

        # (2) Feedback responsiveness: only defined after a turn that produced an error.
        responsive: list[float] = []
        for prev, cur in zip(steps, steps[1:], strict=False):  # deliberately offset by one
            if prev.error is None:
                continue
            plan = _plan_text(cur.response).lower()
            if not plan:
                responsive.append(0.0)
                continue
            hit = bool(self._ERROR_WORDS.search(plan))
            # Naming the specific offending identifier is stronger evidence than
            # a generic "something failed".
            tokens = set(re.findall(r"[A-Za-z_]\w{2,}", prev.error))
            named = any(tok.lower() in plan for tok in tokens if len(tok) > 3)
            responsive.append(1.0 if (hit and named) else (0.6 if hit else 0.0))
        c2 = sum(responsive) / len(responsive) if responsive else 1.0

        # (3) Backtracking justification: a switch must follow recorded failure
        #     evidence on the branch being left.
        justified: list[float] = []
        for i, s in enumerate(steps):
            if not s.is_backtrack:
                continue
            had_failure = any(p.error is not None for p in steps[:i])
            plan = _plan_text(s.response).lower()
            explains = bool(self._ERROR_WORDS.search(plan))
            justified.append(1.0 if (had_failure and explains) else (0.3 if had_failure else 0.0))
        c3 = sum(justified) / len(justified) if justified else 1.0

        return ProcessScore(
            plan_action_consistency=c1,
            feedback_responsiveness=c2,
            backtracking_justification=c3,
            rationale="heuristic structural judge",
        )


_JUDGE_SYSTEM = """
You are a strict evaluator of data preparation agent trajectories.

The agent builds a pipeline by interacting with an execution environment over a
search tree. Each turn it emits <plan> (what it intends to do and from which node)
and <expand> (the operators to apply); the environment replies with <execute>
(the resulting tables or an error trace).

Score the trajectory on three criteria, each from 0.0 to 1.0:

1. plan_action_consistency: does the operator pipeline in <expand> actually
   implement what the <plan> said it would? Penalize plans that describe one
   transformation while the operators perform another.
2. feedback_responsiveness: after an execution error, does the NEXT <plan>
   explicitly engage with that error - naming the cause and how it will be
   fixed - rather than ignoring it or blindly retrying?
3. backtracking_justification: when the agent switches to a different parent
   node, is that switch supported by failure evidence recorded on the branch it
   is abandoning? Unmotivated jumping between branches scores low.

Be skeptical of SHORTCUTS. An agent that reaches a plausible-looking table by
renaming columns, hard-coding values, or dropping rows to satisfy a shape
constraint - rather than performing the required cleaning and transformation
logic - must score low on criterion 1 regardless of how the table looks.

Respond with ONLY a JSON object:
{"plan_action_consistency": <float>, "feedback_responsiveness": <float>,
 "backtracking_justification": <float>, "rationale": "<one sentence>"}
""".strip()


class LLMProcessJudge:
    """``R_llm`` as specified: "an instruction-tuned LLM (e.g. GPT-4o) as a judge"."""

    def __init__(
        self,
        llm: LLMClient,
        max_turns_shown: int = 8,
        max_chars_per_turn: int = 2500,
        fallback: ProcessJudge | None = None,
    ) -> None:
        self.llm = llm
        self.max_turns_shown = max_turns_shown
        self.max_chars_per_turn = max_chars_per_turn
        # A judge outage must not silently zero the reward term for a whole batch.
        self.fallback = fallback or HeuristicProcessJudge()

    def score(self, task: ADPTask, result: SolveResult) -> ProcessScore:
        prompt = self._render(task, result)
        try:
            resp = self.llm.generate(
                [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            parsed = _parse_judge_json(resp.text)
        except Exception:  # noqa: BLE001
            return self.fallback.score(task, result)
        if parsed is None:
            return self.fallback.score(task, result)
        return parsed

    def _render(self, task: ADPTask, result: SolveResult) -> str:
        parts = [
            "# Target schema",
            task.target_schema.description or "(no description)",
            "Columns: " + ", ".join(c.name for c in task.target_schema.columns),
            "",
            "# Trajectory",
        ]
        steps = result.trajectory[: self.max_turns_shown]
        for s in steps:
            parts.append(f"--- turn {s.turn} ---")
            parts.append(_truncate(s.response, self.max_chars_per_turn))
            parts.append("<execute>")
            parts.append(_truncate(s.feedback or "(no feedback)", self.max_chars_per_turn))
            parts.append("</execute>")
            if s.is_backtrack:
                parts.append(f"[the agent switched parent node to {s.parent_node_id}]")
        if len(result.trajectory) > self.max_turns_shown:
            parts.append(f"... ({len(result.trajectory) - self.max_turns_shown} more turns)")
        parts += ["", "Score this trajectory. Respond with the JSON object only."]
        return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} chars elided] ...\n{text[-half:]}"


def _plan_text(response: str) -> str:
    m = re.search(r"<plan\s*>(.*?)</plan\s*>", response or "", re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"<plan\s*>(.*)", response or "", re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else ""


def _parse_judge_json(text: str) -> ProcessScore | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

    def clamp(key: str) -> float:
        try:
            return max(0.0, min(1.0, float(d.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    return ProcessScore(
        plan_action_consistency=clamp("plan_action_consistency"),
        feedback_responsiveness=clamp("feedback_responsiveness"),
        backtracking_justification=clamp("backtracking_justification"),
        rationale=str(d.get("rationale", ""))[:500],
    )


# --------------------------------------------------------------------------- #
# Eq. (6)
# --------------------------------------------------------------------------- #
def compute_reward(
    task: ADPTask,
    result: SolveResult,
    config: RewardConfig | None = None,
    judge: ProcessJudge | None = None,
) -> RewardBreakdown:
    """Evaluate ``R(tau) = alpha*R_out + beta*R_part + gamma*R_llm``."""
    cfg = config or RewardConfig()
    out = RewardBreakdown()

    produced: pd.DataFrame | None = result.table
    if produced is None and cfg.score_fallback_table:
        produced = result.fallback_table

    # R_out (Eq. 7): strict correctness requires a real <answer>, so a fallback
    # table can earn partial credit but never the outcome reward.
    out.r_out = (
        1.0
        if (result.completed and table_match(result.table, task.target_table, cfg.match_options))
        else 0.0
    )

    # R_part (Eq. 8).
    sim = partial_similarity(produced, task.target_table, cfg.match_options)
    out.similarity = sim
    if out.r_out == 0.0 or not cfg.partial_only_when_incorrect:
        out.r_part = sim["partial"]

    # R_llm.
    if judge is not None and cfg.gamma > 0:
        score = judge.score(task, result)
        out.process = score
        out.r_llm = score.value

    # Format shaping: unparseable actions waste a turn and teach nothing.
    n_bad = sum(1 for s in result.trajectory if s.error and s.error.startswith("MissingAction"))
    out.format_penalty = cfg.format_penalty * n_bad / max(len(result.trajectory), 1)

    out.total = (
        cfg.alpha * out.r_out
        + cfg.beta * out.r_part
        + cfg.gamma * out.r_llm
        - out.format_penalty
    )
    return out
