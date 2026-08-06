"""LLM backends.

The paper implements DeepPrep on both closed-source backbones (GPT-5, Claude-4)
and open-source ones (Qwen2.5 0.5B-14B, Qwen3 0.6B-14B), with "temperature ...
set to 0.01" for prompting methods (Sec 6.1).

Three clients are provided:

* :class:`OpenAIClient` — any OpenAI-compatible endpoint.  This covers the API
  models *and* locally served open-source checkpoints, since the paper serves
  them for evaluation (a vLLM server exposes the same protocol).
* :class:`HFClient` — in-process ``transformers`` generation, used by the RL
  rollout worker where the policy weights change every step.
* :class:`ScriptedClient` — deterministic canned responses for tests and for
  ``deepprep demo``, so the whole pipeline is exercisable with no network.

Every client accumulates :class:`Usage`, which feeds the Cost metric of Sec 6.1.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "HFClient",
    "LLMClient",
    "LLMResponse",
    "OpenAIClient",
    "ScriptedClient",
    "Usage",
]

Message = dict[str, str]


@dataclass
class Usage:
    """Token and wall-clock accounting used by the Cost metric (Sec 6.1)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    n_calls: int = 0
    wall_time_s: float = 0.0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_prompt_tokens += other.cached_prompt_tokens
        self.n_calls += other.n_calls
        self.wall_time_s += other.wall_time_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "n_calls": self.n_calls,
            "wall_time_s": round(self.wall_time_s, 4),
        }


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Minimal generation interface the agent depends on."""

    model: str

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoints (OpenAI / DeepSeek / vLLM / SGLang / ...)
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """Chat-completions client for any OpenAI-compatible server.

    ``base_url`` lets the same class drive a local vLLM deployment of a Qwen
    checkpoint, which is how the paper evaluates open-source backbones.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 4,
        timeout: float = 180.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OpenAIClient requires the 'openai' package: pip install 'deepprep[llm]'"
            ) from e
        self.model = model
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.usage = Usage()
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            timeout=timeout,
            max_retries=0,  # retries are handled below so backoff is observable
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=list(stop) if stop else None,
                    **self.extra_body,
                )
            except Exception as e:  # noqa: BLE001 - transient API failures
                last_err = e
                time.sleep(min(2**attempt, 20))
                continue

            dt = time.perf_counter() - t0
            u = resp.usage
            cached = 0
            if u is not None:
                details = getattr(u, "prompt_tokens_details", None)
                cached = int(getattr(details, "cached_tokens", 0) or 0)
            usage = Usage(
                prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
                cached_prompt_tokens=cached,
                n_calls=1,
                wall_time_s=dt,
            )
            self.usage.add(usage)
            choice = resp.choices[0]
            return LLMResponse(
                text=choice.message.content or "",
                usage=usage,
                finish_reason=choice.finish_reason,
            )
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Local transformers
# --------------------------------------------------------------------------- #
class HFClient:
    """In-process ``transformers`` generation.

    Used by the GRPO rollout worker, where the policy is the object being
    trained and cannot be reached over HTTP.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self._model = model
        self._tok = tokenizer
        self.model = getattr(getattr(model, "config", None), "_name_or_path", "hf-model")
        self.device = device or next(model.parameters()).device
        self.enable_thinking = enable_thinking
        self.usage = Usage()

    def apply_template(self, messages: Sequence[Message], add_generation_prompt: bool = True) -> str:
        kwargs: dict[str, Any] = {}
        # Qwen3 exposes a thinking toggle; DeepPrep's structured actions do not
        # need a separate reasoning phase, so it defaults to off.
        try:
            return self._tok.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self._tok.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
            )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        import torch

        prompt = self.apply_template(messages)
        enc = self._tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_tokens,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        dt = time.perf_counter() - t0
        gen = out[0][enc["input_ids"].shape[1] :]
        text = self._tok.decode(gen, skip_special_tokens=True)
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[: idx + len(s)]
        usage = Usage(
            prompt_tokens=int(enc["input_ids"].shape[1]),
            completion_tokens=int(gen.shape[0]),
            n_calls=1,
            wall_time_s=dt,
        )
        self.usage.add(usage)
        return LLMResponse(text=text, usage=usage, finish_reason="stop")


# --------------------------------------------------------------------------- #
# Scripted (tests / offline demo)
# --------------------------------------------------------------------------- #
class ScriptedClient:
    """Replays a fixed list of responses.

    Makes the full plan/expand/execute/answer loop testable — including the
    backtracking path of Figure 4 — without any network access.
    """

    def __init__(self, responses: Sequence[str], model: str = "scripted") -> None:
        self.responses = list(responses)
        self.model = model
        self.calls: list[list[Message]] = []
        self.usage = Usage()
        self._i = 0

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if self._i >= len(self.responses):
            text = "<plan>The script is exhausted.</plan>"
        else:
            text = self.responses[self._i]
            self._i += 1
        # Rough token accounting keeps the cost metric exercisable offline.
        usage = Usage(
            prompt_tokens=sum(len(m.get("content", "")) for m in messages) // 4,
            completion_tokens=len(text) // 4,
            n_calls=1,
            wall_time_s=0.0,
        )
        self.usage.add(usage)
        return LLMResponse(text=text, usage=usage, finish_reason="stop")
