"""Progressive Agentic Training (paper Sec 5).

    "we propose a Progressive Agentic Training framework that (i) initializes the
     model with basic operator usage and tree-based interaction patterns, (ii)
     refines the policy through multi-turn reinforcement learning with a hybrid
     reward that considers both outcome-level and process-level signals, and
     (iii) leverages data synthesis to construct diverse and complex ADP tasks
     for training."

Stage layout:

  Stage 1  :mod:`deepprep.training.sft_data`  -> ``build_op_syntax_dataset``  (Eq. 4)
  Stage 2  :mod:`deepprep.training.distill`   -> ``distill_trajectories``     (Eq. 5)
           :mod:`deepprep.training.sft`       -> ``SFTTrainer`` for both stages
  Stage 3  :mod:`deepprep.training.grpo`      -> ``GRPOTrainer``              (Eq. 6)

The heavy dependencies (``torch``, ``transformers``, ``peft``) are only imported
when a trainer is instantiated, so importing this package is cheap.
"""

from .distill import (
    DistillConfig,
    DistillStats,
    distill_trajectories,
    perturb_task,
    trajectory_to_sft_example,
)
from .masking import MaskedSequence, build_masked_sequence, refine_mask_to_action_tags
from .rewards import (
    HeuristicProcessJudge,
    LLMProcessJudge,
    ProcessJudge,
    ProcessScore,
    RewardBreakdown,
    RewardConfig,
    compute_reward,
)
from .sft_data import (
    SFTExample,
    build_op_syntax_dataset,
    build_op_syntax_examples,
    materialize_states,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    "DistillConfig",
    "DistillStats",
    "HeuristicProcessJudge",
    "LLMProcessJudge",
    "MaskedSequence",
    "ProcessJudge",
    "ProcessScore",
    "RewardBreakdown",
    "RewardConfig",
    "SFTExample",
    "build_masked_sequence",
    "build_op_syntax_dataset",
    "build_op_syntax_examples",
    "compute_reward",
    "distill_trajectories",
    "materialize_states",
    "perturb_task",
    "read_jsonl",
    "refine_mask_to_action_tags",
    "trajectory_to_sft_example",
    "write_jsonl",
]


def __getattr__(name: str) -> object:
    """Expose the trainers lazily so ``import deepprep.training`` needs no torch."""
    if name in ("SFTConfig", "SFTTrainer", "train_sft"):
        from . import sft

        return getattr(sft, name)
    if name in ("GRPOConfig", "GRPOTrainer", "Rollout"):
        from . import grpo

        return getattr(grpo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
