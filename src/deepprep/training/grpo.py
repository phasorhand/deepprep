"""Multi-turn RL with Group Relative Policy Optimization (paper Sec 5.2).

    "Data preparation pipeline generation is a sequential decision process with
     execution feedback, where each action affects subsequent states and future
     operator choices. Optimizing such behavior cannot be reduced to single-shot
     generation. Instead, the policy should be trained over multi-turn
     interaction trajectories. In each turn, the agent produces structured
     actions (e.g. <plan> and <expand>), the environment executes the proposed
     operators, and feedback is returned. Then, subsequent decisions are
     conditioned on these execution outcomes. ...

     We optimize the policy using Group Relative Policy Optimization (GRPO) [34],
     which estimates baselines from group-level reward statistics instead of
     training a separate value network, supporting more stable optimization over
     multi-turn trajectories."

Three things make this *multi-turn* GRPO rather than the usual single-response
variant:

1. **The rollout is an episode.**  One sample is a whole plan/expand/execute
   interaction with the environment, not one completion.  Sampling therefore runs
   the real :class:`~deepprep.agent.DeepPrepAgent` against the real environment.
2. **The gradient is masked to policy tokens.**  Sec 5.2: "we apply a binary
   token mask that removes gradients on tokens inside <execute> blocks, and
   optimize the policy only over agent decision tokens within <plan>, <expand>,
   and <answer>."  Serialized intermediate tables are deterministic environment
   output; treating them as actions would train the model to predict its own
   observations.
3. **The reward is the hybrid of Eq. (6)**, computed once per episode and
   broadcast to every action token of that episode.

The group baseline is formed over ``group_size`` rollouts **of the same task**,
which is what makes the relative advantage meaningful.

    "we use the AdamW optimizer with a learning rate of ... 1 x 10^-6 for RL" (Sec 6.1)
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..agent.agent import DeepPrepAgent, SolveResult
from ..agent.llm import HFClient
from ..types import ADPTask
from .masking import MaskedSequence, build_masked_sequence
from .rewards import (
    HeuristicProcessJudge,
    ProcessJudge,
    RewardBreakdown,
    RewardConfig,
    compute_reward,
)

__all__ = ["GRPOConfig", "GRPOTrainer", "Rollout"]


@dataclass
class GRPOConfig:
    model_name_or_path: str = "checkpoints/sft-stage2"
    output_dir: str = "checkpoints/grpo"
    #: Sec 6.1: "1 x 10^-6 for RL".
    learning_rate: float = 1e-6
    #: G -- rollouts per task. The group baseline needs G >= 2.
    group_size: int = 8
    #: Tasks per optimization batch.
    tasks_per_batch: int = 4
    n_epochs: int = 1
    #: Inner PPO-style epochs over each collected batch.
    inner_epochs: int = 1
    micro_batch_size: int = 1
    max_length: int = 8192
    #: PPO clipping range.
    clip_eps: float = 0.2
    #: KL penalty toward the frozen reference policy (the Stage-2 checkpoint).
    kl_coef: float = 0.04
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    #: Rollout sampling temperature. Must be > 0: identical rollouts give a
    #: zero-variance group and therefore zero advantage for every sample.
    rollout_temperature: float = 1.0
    max_turns: int = 5
    max_new_tokens: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    enable_thinking: bool | None = False
    seed: int = 0
    device: str | None = None
    log_every: int = 1
    save_every: int | None = None
    #: Skip groups whose rewards are all equal — their advantages are all zero,
    #: so they contribute nothing but still cost a backward pass.
    skip_degenerate_groups: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Rollout:
    """One sampled episode with its reward and masked token sequence."""

    task_id: str
    result: SolveResult
    reward: RewardBreakdown
    sequence: MaskedSequence | None = None
    advantage: float = 0.0
    old_logprobs: Any = None
    ref_logprobs: Any = None

    @property
    def scalar_reward(self) -> float:
        return self.reward.total


@dataclass
class GRPOStats:
    step: int = 0
    mean_reward: float = 0.0
    mean_r_out: float = 0.0
    mean_r_part: float = 0.0
    mean_r_llm: float = 0.0
    mean_completion: float = 0.0
    mean_turns: float = 0.0
    mean_backtracks: float = 0.0
    policy_loss: float = 0.0
    kl: float = 0.0
    n_groups_used: int = 0
    n_groups_skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}


class GRPOTrainer:
    """Multi-turn GRPO over agentic data preparation trajectories."""

    def __init__(
        self,
        config: GRPOConfig,
        reward_config: RewardConfig | None = None,
        judge: ProcessJudge | None = None,
    ) -> None:
        self.cfg = config
        self.reward_config = reward_config or RewardConfig()
        # The paper uses an instruction-tuned LLM judge; the heuristic keeps RL
        # runnable without a second model in the loop.
        self.judge = judge or HeuristicProcessJudge()
        self._torch: Any = None
        self.model: Any = None
        self.ref_model: Any = None
        self.tokenizer: Any = None
        self.history: list[dict[str, Any]] = []

    # -- setup -------------------------------------------------------------- #
    def load(self) -> None:
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "training requires the optional dependencies: pip install 'deepprep[train]'"
            ) from e
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if cfg.bf16 and torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path, torch_dtype=dtype, trust_remote_code=True
        )
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=cfg.lora_r, lora_alpha=cfg.lora_alpha, task_type="CAUSAL_LM"
                ),
            )

        # Frozen reference for the KL term. With LoRA the base weights already
        # serve as the reference, so a second copy would only waste memory.
        if cfg.kl_coef > 0 and not cfg.use_lora:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path, torch_dtype=dtype, trust_remote_code=True
            )
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)

    # -- rollout ------------------------------------------------------------ #
    def collect_group(self, task: ADPTask) -> list[Rollout]:
        """Sample ``G`` episodes of one task with the current policy."""
        cfg = self.cfg
        self.model.eval()
        # use_cache is disabled for gradient checkpointing but generation wants it.
        prev_cache = getattr(self.model.config, "use_cache", None)
        self.model.config.use_cache = True

        client = HFClient(
            self.model, self.tokenizer, device=self.device,
            enable_thinking=bool(cfg.enable_thinking),
        )
        agent = DeepPrepAgent(
            llm=client,
            max_turns=cfg.max_turns,
            temperature=cfg.rollout_temperature,
            max_tokens=cfg.max_new_tokens,
            include_exemplar=False,
        )

        rollouts: list[Rollout] = []
        for _ in range(cfg.group_size):
            try:
                res = agent.solve(task)
            except Exception:  # noqa: BLE001 - a broken sample must not kill the step
                continue
            breakdown = compute_reward(task, res, self.reward_config, self.judge)
            seq = build_masked_sequence(
                self.tokenizer,
                res.messages,
                trainable=[m.get("role") == "assistant" for m in res.messages],
                enable_thinking=cfg.enable_thinking,
                max_length=cfg.max_length,
            )
            if seq.n_action_tokens == 0:
                continue
            rollouts.append(
                Rollout(task_id=task.task_id, result=res, reward=breakdown, sequence=seq)
            )

        if prev_cache is not None:
            self.model.config.use_cache = prev_cache
        self.model.train()
        return rollouts

    @staticmethod
    def assign_advantages(group: Sequence[Rollout], skip_degenerate: bool) -> bool:
        """Group-relative advantage: ``A_i = (r_i - mean(r)) / std(r)``.

        Returns ``False`` when the group is degenerate (no reward variance), in
        which case every advantage is zero and the group carries no signal.
        """
        rewards = [r.scalar_reward for r in group]
        if len(rewards) < 2:
            return False
        mean = statistics.fmean(rewards)
        std = statistics.pstdev(rewards)
        if std < 1e-8:
            if skip_degenerate:
                return False
            for r in group:
                r.advantage = 0.0
            return True
        for r in group:
            r.advantage = (r.scalar_reward - mean) / (std + 1e-8)
        return True

    # -- log probabilities -------------------------------------------------- #
    def _token_logprobs(self, model: Any, seq: MaskedSequence, grad: bool) -> Any:
        """Per-position log-prob of the realized token, aligned to ``input_ids[1:]``."""
        torch = self._torch
        ids = torch.tensor([seq.input_ids], dtype=torch.long, device=self.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = model(input_ids=ids).logits
            # Predict token t+1 from position t.
            logits = logits[:, :-1, :].float()
            targets = ids[:, 1:]
            logp = torch.log_softmax(logits, dim=-1)
            return torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)[0]

    def _action_mask_tensor(self, seq: MaskedSequence) -> Any:
        torch = self._torch
        # Shifted to match the log-prob alignment above: position t of the
        # log-prob vector scores input_ids[t+1], so the mask must be too.
        return torch.tensor(seq.action_mask[1:], dtype=torch.float32, device=self.device)

    # -- optimization ------------------------------------------------------- #
    def train(self, tasks: Sequence[ADPTask]) -> list[dict[str, Any]]:
        if self.model is None:
            self.load()
        torch = self._torch
        cfg = self.cfg
        rng = random.Random(cfg.seed)

        optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        pool = list(tasks)
        step = 0
        for epoch in range(cfg.n_epochs):
            rng.shuffle(pool)
            for i in range(0, len(pool), cfg.tasks_per_batch):
                batch_tasks = pool[i : i + cfg.tasks_per_batch]
                stats = GRPOStats(step=step)
                usable: list[Rollout] = []
                all_rollouts: list[Rollout] = []

                for task in batch_tasks:
                    group = self.collect_group(task)
                    all_rollouts.extend(group)
                    if self.assign_advantages(group, cfg.skip_degenerate_groups):
                        usable.extend(group)
                        stats.n_groups_used += 1
                    else:
                        stats.n_groups_skipped += 1

                self._fill_stats(stats, all_rollouts)
                if not usable:
                    self.history.append(stats.to_dict())
                    step += 1
                    continue

                # Freeze the behaviour policy's log-probs for the ratio, and the
                # reference's for the KL, before any parameter update.
                for r in usable:
                    r.old_logprobs = self._token_logprobs(self.model, r.sequence, grad=False).detach()
                    if self.ref_model is not None:
                        r.ref_logprobs = self._token_logprobs(
                            self.ref_model, r.sequence, grad=False
                        ).detach()
                    elif cfg.kl_coef > 0 and cfg.use_lora:
                        with self.model.disable_adapter():
                            r.ref_logprobs = self._token_logprobs(
                                self.model, r.sequence, grad=False
                            ).detach()

                for _ in range(cfg.inner_epochs):
                    rng.shuffle(usable)
                    optim.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    total_kl = 0.0
                    n = 0
                    for r in usable:
                        loss, kl = self._rollout_loss(r)
                        (loss / len(usable)).backward()
                        total_loss += float(loss.detach())
                        total_kl += kl
                        n += 1
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optim.step()
                    stats.policy_loss = total_loss / max(n, 1)
                    stats.kl = total_kl / max(n, 1)

                self.history.append(stats.to_dict())
                if step % cfg.log_every == 0:
                    print(
                        f"  epoch {epoch} step {step}  reward={stats.mean_reward:.4f} "
                        f"r_out={stats.mean_r_out:.3f} comp={stats.mean_completion:.3f} "
                        f"backtracks={stats.mean_backtracks:.2f} loss={stats.policy_loss:.4f} "
                        f"kl={stats.kl:.4f} groups={stats.n_groups_used}"
                        f"(+{stats.n_groups_skipped} degenerate)"
                    )
                step += 1
                if cfg.save_every and step % cfg.save_every == 0:
                    self.save(Path(cfg.output_dir) / f"step-{step}")

        self.save(cfg.output_dir)
        return self.history

    def _rollout_loss(self, r: Rollout) -> tuple[Any, float]:
        """Token-level clipped surrogate with a KL penalty, masked to actions."""
        torch = self._torch
        cfg = self.cfg
        assert r.sequence is not None

        logprobs = self._token_logprobs(self.model, r.sequence, grad=True)
        mask = self._action_mask_tensor(r.sequence)
        denom = mask.sum().clamp(min=1.0)

        ratio = torch.exp(logprobs - r.old_logprobs)
        adv = torch.tensor(r.advantage, dtype=torch.float32, device=self.device)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
        policy_loss = -(torch.min(unclipped, clipped) * mask).sum() / denom

        kl_value = 0.0
        if cfg.kl_coef > 0 and r.ref_logprobs is not None:
            # k3 estimator: unbiased and non-negative, unlike the naive difference.
            delta = r.ref_logprobs - logprobs
            kl = (torch.exp(delta) - delta - 1.0)
            kl_term = (kl * mask).sum() / denom
            policy_loss = policy_loss + cfg.kl_coef * kl_term
            kl_value = float(kl_term.detach())

        return policy_loss, kl_value

    @staticmethod
    def _fill_stats(stats: GRPOStats, rollouts: Sequence[Rollout]) -> None:
        if not rollouts:
            return
        n = len(rollouts)
        stats.mean_reward = sum(r.scalar_reward for r in rollouts) / n
        stats.mean_r_out = sum(r.reward.r_out for r in rollouts) / n
        stats.mean_r_part = sum(r.reward.r_part for r in rollouts) / n
        stats.mean_r_llm = sum(r.reward.r_llm for r in rollouts) / n
        stats.mean_completion = sum(1.0 for r in rollouts if r.result.completed) / n
        stats.mean_turns = sum(r.result.n_turns for r in rollouts) / n
        stats.mean_backtracks = sum(r.result.n_backtracks for r in rollouts) / n

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        (p / "deepprep_grpo_config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        (p / "deepprep_grpo_history.json").write_text(json.dumps(self.history, indent=2))
        print(f"  saved checkpoint to {p}")
