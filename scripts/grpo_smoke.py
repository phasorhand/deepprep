"""One CPU-sized GRPO step (Sec 5.2), for checking the loop actually runs.

The paper's settings (group 8, 4 tasks/batch, 2048 new tokens, bf16) assume a
GPU.  This shrinks every dimension until a step fits on a laptop CPU, which is
enough to exercise rollout -> reward -> masked policy gradient end to end.

    python scripts/grpo_smoke.py data/tasks/synth_train.jsonl
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from deepprep.training.grpo import GRPOConfig, GRPOTrainer
from deepprep.types import load_tasks

BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def main(argv: list[str]) -> int:
    tasks_path = argv[1] if len(argv) > 1 else "data/tasks/synth_train.jsonl"
    tasks = load_tasks(tasks_path)[:1]
    print(f"{len(tasks)} task(s) from {tasks_path}")

    cfg = GRPOConfig(
        model_name_or_path=argv[2] if len(argv) > 2 else BASE,
        output_dir="checkpoints/grpo-smoke",
        # A group whose rewards are all equal has zero advantage, so the step is
        # a mathematically correct no-op -- raise GRPO_GROUP to actually exercise
        # the masked policy gradient.
        group_size=int(os.environ.get("GRPO_GROUP", "2")),
        tasks_per_batch=1,
        n_epochs=1,
        max_turns=2,
        max_new_tokens=int(os.environ.get("GRPO_NEW_TOKENS", "160")),
        max_length=2048,
        bf16=False,               # CPU bf16 matmuls are painfully slow
        gradient_checkpointing=False,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        skip_degenerate_groups=False,  # a tiny model rarely differentiates
    )
    GRPOTrainer(cfg).train(tasks)
    print(f"checkpoint dir: {Path(cfg.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
