"""Supervised fine-tuning for the cold-start curriculum (paper Sec 5.1).

One trainer serves both stages, because Eq. (4) and Eq. (5) differ only in *what
is masked*:

* Stage 1 (Eq. 4) — one trainable message: the operator sequence ``phi(P_sub)``.
* Stage 2 (Eq. 5) — one trainable message per agent turn, with the mask ``M_t``.

Both reduce to a causal-LM cross-entropy over the masked positions, which is what
:class:`~deepprep.training.masking.MaskedSequence` produces.

    "For all training methods, we use the AdamW optimizer with a learning rate of
     6 x 10^-5 for SFT and 1 x 10^-6 for RL." (Sec 6.1)

``torch``/``transformers`` are imported lazily so the inference and evaluation
paths stay installable without the training stack.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .masking import build_masked_sequence
from .sft_data import SFTExample

__all__ = ["SFTConfig", "SFTTrainer", "train_sft"]


@dataclass
class SFTConfig:
    model_name_or_path: str = "Qwen/Qwen3-8B"
    output_dir: str = "checkpoints/sft"
    #: Sec 6.1: "a learning rate of 6 x 10^-5 for SFT".
    learning_rate: float = 6e-5
    n_epochs: int = 2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 8192
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"  # cosine | linear | constant
    seed: int = 0
    bf16: bool = True
    gradient_checkpointing: bool = True
    #: Parameter-efficient tuning. The paper fine-tunes fully on 16 A800s; LoRA is
    #: offered so the curriculum is reproducible on a single GPU.
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    enable_thinking: bool | None = False
    log_every: int = 10
    save_every: int | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Batch:
    input_ids: Any
    attention_mask: Any
    labels: Any
    n_action_tokens: int


class SFTTrainer:
    """Minimal, explicit SFT loop over masked chat sequences."""

    def __init__(self, config: SFTConfig) -> None:
        self.cfg = config
        self._torch = None
        self.model: Any = None
        self.tokenizer: Any = None

    # -- setup -------------------------------------------------------------- #
    def _lazy_imports(self) -> Any:
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "training requires the optional dependencies: pip install 'deepprep[train]'"
            ) from e
        self._torch = torch
        return torch

    def load(self) -> None:
        torch = self._lazy_imports()
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=list(cfg.lora_target_modules),
                    task_type="CAUSAL_LM",
                ),
            )
            self.model.print_trainable_parameters()

        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.device = device

    # -- data --------------------------------------------------------------- #
    def encode(self, examples: Sequence[SFTExample]) -> list[Any]:
        """Tokenize examples into masked sequences, dropping degenerate ones."""
        out = []
        n_empty = 0
        for ex in examples:
            seq = build_masked_sequence(
                self.tokenizer,
                ex.messages,
                ex.trainable,
                enable_thinking=self.cfg.enable_thinking,
                max_length=self.cfg.max_length,
            )
            # Left-truncation can push every trainable token out of the window;
            # such an example contributes nothing but still costs a forward pass.
            if seq.n_action_tokens == 0:
                n_empty += 1
                continue
            out.append(seq)
        if n_empty:
            print(
                f"  dropped {n_empty}/{len(examples)} examples with no trainable tokens "
                f"after truncation to {self.cfg.max_length} tokens"
            )
        return out

    def _collate(self, seqs: Sequence[Any]) -> _Batch:
        torch = self._torch
        pad_id = self.tokenizer.pad_token_id
        width = max(len(s) for s in seqs)
        input_ids, attn, labels = [], [], []
        for s in seqs:
            n_pad = width - len(s)
            input_ids.append(s.input_ids + [pad_id] * n_pad)
            attn.append([1] * len(s) + [0] * n_pad)
            labels.append(s.labels() + [-100] * n_pad)
        return _Batch(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=self.device),
            attention_mask=torch.tensor(attn, dtype=torch.long, device=self.device),
            labels=torch.tensor(labels, dtype=torch.long, device=self.device),
            n_action_tokens=sum(s.n_action_tokens for s in seqs),
        )

    # -- training ----------------------------------------------------------- #
    def train(self, examples: Sequence[SFTExample]) -> dict[str, Any]:
        if self.model is None:
            self.load()
        torch = self._torch
        cfg = self.cfg

        seqs = self.encode(examples)
        if not seqs:
            raise ValueError("no trainable examples after encoding")

        # Length-bucketed batching keeps padding waste low without a sampler.
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        batches = [
            [seqs[j] for j in order[i : i + cfg.per_device_batch_size]]
            for i in range(0, len(order), cfg.per_device_batch_size)
        ]

        steps_per_epoch = math.ceil(len(batches) / cfg.gradient_accumulation_steps)
        total_steps = steps_per_epoch * cfg.n_epochs

        optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = _make_scheduler(torch, optim, cfg, total_steps)

        history: list[dict[str, float]] = []
        self.model.train()
        step = 0
        rng = __import__("random").Random(cfg.seed)

        for epoch in range(cfg.n_epochs):
            rng.shuffle(batches)
            accum_loss = 0.0
            for bi, batch_seqs in enumerate(batches):
                batch = self._collate(batch_seqs)
                out = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )
                loss = out.loss / cfg.gradient_accumulation_steps
                loss.backward()
                accum_loss += float(loss.detach())

                if (bi + 1) % cfg.gradient_accumulation_steps == 0 or bi == len(batches) - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    if step % cfg.log_every == 0:
                        rec = {
                            "epoch": epoch,
                            "step": step,
                            "loss": accum_loss,
                            "lr": scheduler.get_last_lr()[0],
                        }
                        history.append(rec)
                        print(
                            f"  epoch {epoch} step {step}/{total_steps} "
                            f"loss={accum_loss:.4f} lr={rec['lr']:.2e}"
                        )
                    accum_loss = 0.0
                    if cfg.save_every and step % cfg.save_every == 0:
                        self.save(Path(cfg.output_dir) / f"step-{step}")

        self.save(cfg.output_dir)
        return {"steps": step, "history": history, "n_examples": len(seqs)}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        (p / "deepprep_train_config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        print(f"  saved checkpoint to {p}")


def _make_scheduler(torch: Any, optim: Any, cfg: SFTConfig, total_steps: int) -> Any:
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        if cfg.lr_scheduler == "constant":
            return 1.0
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if cfg.lr_scheduler == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def train_sft(
    examples: Iterable[SFTExample], config: SFTConfig | None = None
) -> dict[str, Any]:
    """Convenience entry point used by the CLI."""
    cfg = config or SFTConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer = SFTTrainer(cfg)
    return trainer.train(list(examples))
