"""Teacher trajectory distillation for cold-start Stage 2 (paper Sec 5.1).

    "This stage aligns the model with the Agentic Tree-based Reasoning mechanism
     (Section 4.2). To achieve this, we synthesize a dataset of high-quality
     reasoning trajectories by distilling knowledge from strong teacher models
     (e.g., DeepSeek-R1). Specifically, for each ADP task defined by source tables
     S and a target schema Sigma*, we aim to generate a ground-truth trajectory
     tau. **To enhance robustness, we apply input perturbations Phi, such as
     randomly shuffling rows and columns in the source tables.** Next, we employ
     In-Context Learning [8] to guide the teacher model. By utilizing elaborate
     prompts that exemplify the tree-based reasoning process defined in Eq. (2),
     we generate a set of candidate trajectories U. To unify the training
     objective, we represent each trajectory as a sequence tau = (r_1, ..., r_m).
     Here, the initial subsequence (r_1, ..., r_{m-1}) corresponds to the
     iterative reasoning steps (R_1, ..., R_T) defined in Eq. (3), while the final
     element r_m corresponds to the answer generation A in Eq. (2). These
     candidates are filtered using a process reward function R_llm(tau) (detailed
     in Section 5.2), and only the top-ranked trajectories are retained to form
     the final dataset U*."

Filtering is two-stage here.  A "ground-truth trajectory" must first be *correct*
— a trajectory that reasons beautifully toward the wrong table is not supervision
— and the surviving candidates are then ranked by ``R_llm`` and truncated to the
top ``k``.  Distilling incorrect-but-eloquent trajectories is precisely the
reward-hacking failure that ``R_llm`` exists to prevent.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..agent.agent import DeepPrepAgent, SolveResult
from ..agent.llm import LLMClient
from ..agent.prompts import build_system_prompt, build_task_prompt
from ..serialize import serialize_task_input
from ..types import ADPTask, TableSet
from .rewards import HeuristicProcessJudge, ProcessJudge, RewardConfig, compute_reward
from .sft_data import SFTExample

__all__ = [
    "DistillConfig",
    "DistillStats",
    "distill_trajectories",
    "perturb_task",
    "trajectory_to_sft_example",
]


@dataclass
class DistillConfig:
    #: Candidate trajectories |U| generated per task.
    n_candidates: int = 4
    #: How many of the ranked survivors to keep per task (|U*| per task).
    top_k: int = 1
    #: Sampling temperature for the teacher.  Must be > 0 or the candidates are
    #: identical and the filter has nothing to choose between.
    temperature: float = 0.7
    max_turns: int = 5
    max_tokens: int = 3072
    #: Apply the input perturbations Phi.
    perturb: bool = True
    #: Keep only trajectories whose produced table exactly matches T*.
    require_correct: bool = True
    #: Drop trajectories whose R_llm falls below this even if they are correct.
    min_process_score: float = 0.0
    seed: int = 0


@dataclass
class DistillStats:
    n_tasks: int = 0
    n_candidates: int = 0
    n_correct: int = 0
    n_kept: int = 0
    n_tasks_with_output: int = 0
    #: Trajectories that backtracked at least once — the behaviour Stage 2 exists
    #: to teach, so a corpus with none of them has not captured the mechanism.
    n_with_backtrack: int = 0

    def summary(self) -> str:
        return (
            f"distilled {self.n_kept} trajectories from {self.n_candidates} candidates "
            f"over {self.n_tasks} tasks "
            f"({self.n_tasks_with_output} tasks produced at least one; "
            f"{self.n_correct} candidates were correct; "
            f"{self.n_with_backtrack} kept trajectories contain a backtrack)"
        )


# --------------------------------------------------------------------------- #
# Input perturbation Phi
# --------------------------------------------------------------------------- #
def perturb_task(task: ADPTask, rng: random.Random | None = None) -> ADPTask:
    """Apply ``Phi``: "randomly shuffling rows and columns in the source tables".

    This is a *robustness* perturbation, not a corruption: it must not change the
    task's answer.  Row order is irrelevant to the metric, and column order is
    likewise permutation-invariant, so ``T*`` is left untouched.

    Column *descriptions* travel with their columns, so shuffling cannot silently
    re-associate a description with the wrong column.
    """
    rng = rng or random.Random(0)
    sources = TableSet()
    for t in task.sources:
        df = t.df
        if len(df) > 1:
            order = list(range(len(df)))
            rng.shuffle(order)
            df = df.iloc[order].reset_index(drop=True)
        cols = list(df.columns)
        if len(cols) > 1:
            rng.shuffle(cols)
            df = df[cols]
        new = t.with_df(df)
        # with_df preserves descriptions by name, so the schema follows the shuffle.
        sources.add(new)

    return ADPTask(
        task_id=task.task_id,
        sources=sources,
        target_schema=task.target_schema,
        target_table=task.target_table,
        gold_pipeline=list(task.gold_pipeline),
        metadata={**task.metadata, "perturbed": True},
    )


# --------------------------------------------------------------------------- #
# Trajectory -> training example
# --------------------------------------------------------------------------- #
def trajectory_to_sft_example(
    task: ADPTask,
    result: SolveResult,
    max_turns: int = 5,
    max_rows: int = 5,
    include_exemplar_in_target: bool = False,
) -> SFTExample | None:
    """Serialize ``tau = (r_1, ..., r_m)`` into a masked chat example (Eq. 5).

    The *student's* system prompt is used, not the teacher's: the teacher needed
    the in-context exemplar to produce the trajectory, but training the student on
    a prompt it will never see at inference would create a train/test mismatch.
    After Stage 2 the student is expected to have internalized the protocol.
    """
    if not result.trajectory:
        return None

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(
                include_exemplar=include_exemplar_in_target, max_turns=max_turns
            ),
        }
    ]
    trainable: list[bool] = [False]

    for i, step in enumerate(result.trajectory):
        if i == 0:
            # phi(S), Sigma* -- Eq. (5) conditions on these throughout.
            user = build_task_prompt(serialize_task_input(task, max_rows=max_rows))
            user += "\n\n" + _root_view(result)
        else:
            user = step.observation
        messages.append({"role": "user", "content": user})
        trainable.append(False)
        messages.append({"role": "assistant", "content": step.response})
        trainable.append(True)

    return SFTExample(
        messages=messages,
        trainable=trainable,
        meta={
            "stage": "reasoning",
            "task_id": task.task_id,
            "n_turns": result.n_turns,
            "n_backtracks": result.n_backtracks,
            "n_failed_expansions": result.n_failed_expansions,
            "stop_reason": result.stop_reason,
        },
    )


def _root_view(result: SolveResult) -> str:
    if result.tree is None:
        return ""
    # Re-render the root only; the tree at turn 0 is exactly n0.
    from ..tree import ReasoningTree

    root_state = result.tree.root.state
    if root_state is None:  # pragma: no cover - the root always has a state
        return ""
    return ReasoningTree(root_state).render()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def distill_trajectories(
    tasks: Sequence[ADPTask],
    teacher: LLMClient,
    config: DistillConfig | None = None,
    judge: ProcessJudge | None = None,
    reward_config: RewardConfig | None = None,
    verbose: bool = True,
) -> tuple[list[SFTExample], DistillStats]:
    """Generate ``U``, filter it, and return ``U*`` as masked SFT examples."""
    cfg = config or DistillConfig()
    judge = judge or HeuristicProcessJudge()
    rcfg = reward_config or RewardConfig()
    stats = DistillStats(n_tasks=len(tasks))
    rng = random.Random(cfg.seed)

    # The teacher is guided by ICL over "elaborate prompts that exemplify the
    # tree-based reasoning process" -- hence include_exemplar=True.
    agent = DeepPrepAgent(
        llm=teacher,
        max_turns=cfg.max_turns,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        include_exemplar=True,
    )

    out: list[SFTExample] = []
    for t_i, task in enumerate(tasks):
        scored: list[tuple[float, SolveResult, ADPTask]] = []
        for _c in range(cfg.n_candidates):
            variant = (
                perturb_task(task, random.Random(rng.randrange(1 << 30)))
                if cfg.perturb
                else task
            )
            try:
                res = agent.solve(variant)
            except Exception:  # noqa: BLE001 - a teacher failure must not stop the sweep
                continue
            stats.n_candidates += 1
            breakdown = compute_reward(variant, res, rcfg, judge)
            if breakdown.r_out > 0:
                stats.n_correct += 1
            if cfg.require_correct and breakdown.r_out <= 0:
                continue
            if breakdown.r_llm < cfg.min_process_score:
                continue
            # Rank by the process reward, as the paper specifies.
            scored.append((breakdown.r_llm, res, variant))

        if scored:
            stats.n_tasks_with_output += 1
        # Prefer higher R_llm; break ties toward the shorter trajectory, which is
        # the less noisy demonstration of the same behaviour.
        scored.sort(key=lambda x: (-x[0], x[1].n_turns))
        for _, res, variant in scored[: cfg.top_k]:
            ex = trajectory_to_sft_example(
                variant, res, max_turns=cfg.max_turns
            )
            if ex is None:
                continue
            out.append(ex)
            stats.n_kept += 1
            if res.n_backtracks > 0:
                stats.n_with_backtrack += 1

        if verbose:
            print(
                f"\r  distilling {t_i + 1}/{len(tasks)}  kept={stats.n_kept}",
                end="",
                flush=True,
            )
    if verbose:
        print()
        print(stats.summary())
        if stats.n_kept and stats.n_with_backtrack == 0:
            print(
                "  WARNING: no kept trajectory contains a backtrack. Stage 2 will not "
                "demonstrate non-local revision, which is the behaviour it exists to "
                "teach. Consider harder tasks or more candidates per task."
            )
    return out, stats
