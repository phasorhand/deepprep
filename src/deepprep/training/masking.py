"""Token-level loss / gradient masking.

Two places in the paper require a mask over a multi-turn token sequence, and both
reduce to the same operation — mark exactly the tokens the *policy* produced.

Eq. (5), cold-start Stage 2:

    "we introduce a loss mask M_t to ensure that gradients are only calculated
     for tokens corresponding to the agent's outputs."

Sec 5.2, multi-turn RL:

    "each trajectory contains both (i) agent-generated tokens (actions and their
     contents) and (ii) environment-generated tokens, such as serialized
     intermediate tables and execution traces. In particular, since
     environment-generated tokens are deterministic execution outputs rather than
     policy decisions, they are excluded from policy gradient optimization. That
     is, we apply a binary token mask that removes gradients on tokens inside
     <execute> blocks, and optimize the policy only over agent decision tokens
     within <plan>, <expand>, and <answer>."

In this implementation the environment's ``<execute>`` output is delivered as a
**user** message and the policy only ever writes assistant messages, so
"assistant tokens only" and "everything except <execute>" denote the same set.
:func:`build_masked_sequence` therefore masks by role, and
:func:`refine_mask_to_action_tags` is available when a caller serializes
``<execute>`` inline into an assistant turn instead.

Correctness note: masks are derived by *incremental re-tokenization* of the chat
template and verified to be prefix-consistent.  Guessing token spans from string
offsets silently misaligns on templates that merge or split tokens at role
boundaries, which would train the model on the wrong positions — a bug that does
not crash and does not show up in the loss curve.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MaskedSequence",
    "build_masked_sequence",
    "refine_mask_to_action_tags",
]

Message = dict[str, str]


@dataclass
class MaskedSequence:
    """A tokenized conversation with a binary trainable mask.

    ``action_mask[i] == 1`` means token ``input_ids[i]`` was produced by the
    policy and therefore contributes to the loss / policy gradient.
    """

    input_ids: list[int]
    action_mask: list[int]
    #: (start, end) half-open token spans of each trainable message.
    spans: list[tuple[int, int]]

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def n_action_tokens(self) -> int:
        return sum(self.action_mask)

    def labels(self, ignore_index: int = -100) -> list[int]:
        """Causal-LM labels: masked positions become ``ignore_index``.

        No shifting is applied — ``transformers`` models shift internally.
        """
        return [
            tok if m else ignore_index for tok, m in zip(self.input_ids, self.action_mask, strict=True)
        ]


def _encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _template(
    tokenizer: Any,
    messages: Sequence[Message],
    add_generation_prompt: bool,
    enable_thinking: bool | None,
) -> str:
    if not messages:
        # The context preceding the first message is the empty string.  Asking
        # the template is not an option: transformers >= 5 raises on an empty
        # conversation rather than returning the bare prefix.
        return ""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                list(messages), enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(list(messages), **kwargs)


def build_masked_sequence(
    tokenizer: Any,
    messages: Sequence[Message],
    trainable: Sequence[bool] | None = None,
    enable_thinking: bool | None = False,
    max_length: int | None = None,
) -> MaskedSequence:
    """Tokenize a conversation and mark the trainable spans.

    ``trainable`` defaults to "every assistant message".  Pass it explicitly for
    :class:`~deepprep.training.sft_data.SFTExample`, which carries its own mask.

    The generation prompt (the ``<|im_start|>assistant`` header and friends) is
    treated as *context*, not as a policy decision: it is emitted by the template,
    not sampled, so training on it teaches nothing and biases the loss toward
    boilerplate.
    """
    msgs = list(messages)
    if trainable is None:
        trainable = [m.get("role") == "assistant" for m in msgs]
    if len(trainable) != len(msgs):
        raise ValueError("trainable must be the same length as messages")

    input_ids: list[int] = []
    action_mask: list[int] = []
    spans: list[tuple[int, int]] = []

    for i, msg in enumerate(msgs):
        is_assistant = msg.get("role") == "assistant"
        # Context up to (but excluding) this message. For an assistant message we
        # include the generation prompt so the boundary is where sampling began.
        prefix_text = _template(tokenizer, msgs[:i], is_assistant, enable_thinking)
        full_text = _template(tokenizer, msgs[: i + 1], False, enable_thinking)

        prefix_ids = _encode(tokenizer, prefix_text)
        full_ids = _encode(tokenizer, full_text)

        if not _is_prefix(prefix_ids, full_ids):
            # Some templates re-tokenize across the boundary. Fall back to the
            # longest common token prefix so the mask stays inside the message.
            k = _common_prefix_len(prefix_ids, full_ids)
        else:
            k = len(prefix_ids)

        # Re-anchor: input_ids already holds tokens for msgs[:i] (without the
        # generation prompt), so splice in everything from the common prefix on.
        base = _common_prefix_len(input_ids, full_ids)
        new_tokens = full_ids[base:]
        start_in_new = max(k - base, 0)

        input_ids = full_ids[:base] + new_tokens
        action_mask = action_mask[:base] + [0] * len(new_tokens)

        if trainable[i]:
            s, e = base + start_in_new, len(input_ids)
            if e > s:
                for p in range(s, e):
                    action_mask[p] = 1
                spans.append((s, e))

    if max_length is not None and len(input_ids) > max_length:
        # Truncate from the LEFT: the most recent turns carry the decisions being
        # learned, whereas the oldest observation is the most expendable context.
        cut = len(input_ids) - max_length
        input_ids = input_ids[cut:]
        action_mask = action_mask[cut:]
        spans = [(max(s - cut, 0), e - cut) for s, e in spans if e - cut > 0]

    return MaskedSequence(input_ids=input_ids, action_mask=action_mask, spans=spans)


def _is_prefix(a: list[int], b: list[int]) -> bool:
    return len(a) <= len(b) and b[: len(a)] == a


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


_EXECUTE_BLOCK = re.compile(r"<execute\s*>.*?</execute\s*>", re.DOTALL | re.IGNORECASE)


def refine_mask_to_action_tags(
    tokenizer: Any,
    seq: MaskedSequence,
    keep_tags: Sequence[str] = ("plan", "expand", "answer"),
) -> MaskedSequence:
    """Restrict an existing mask to the paper's action tags.

    Only needed when a caller inlines ``<execute>`` output into an assistant
    message.  With the default transport (environment output as a user message)
    :func:`build_masked_sequence` already excludes it, and this is a no-op.

    Works by decoding each trainable span and zeroing the tokens that fall inside
    an ``<execute>`` block or outside every kept tag.
    """
    mask = list(seq.action_mask)
    for start, end in seq.spans:
        # The character mask must be built over the SAME string the per-token
        # offsets are measured against.  Decoding the span in one call and then
        # advancing a cursor by `len(decode([tok]))` drifts whenever the batch
        # decode is not the exact concatenation of the single-token decodes --
        # the normal case for SentencePiece's word marker and for any tokenizer
        # that re-inserts separators.  The drift accumulates, so the later tokens
        # of a turn receive the mask of an entirely different region: exactly
        # inverting Sec 5.2 by training on <execute> and dropping <plan>.
        pieces = [
            tokenizer.decode([seq.input_ids[t]], skip_special_tokens=False)
            for t in range(start, end)
        ]
        keep = _char_keep_mask("".join(pieces), keep_tags)
        if all(keep):
            continue
        # A token survives only if every character it contributes is kept.
        pos = 0
        for offset, piece in enumerate(pieces):
            nxt = pos + len(piece)
            window = keep[pos:nxt]
            if window and not all(window):
                mask[start + offset] = 0
            pos = nxt
    return MaskedSequence(input_ids=seq.input_ids, action_mask=mask, spans=seq.spans)


def _char_keep_mask(text: str, keep_tags: Sequence[str]) -> list[bool]:
    """Per-character keep/drop decision for one assistant turn."""
    keep = [True] * len(text)
    for m in _EXECUTE_BLOCK.finditer(text):
        for i in range(m.start(), m.end()):
            keep[i] = False
    if not keep_tags:
        return keep
    # If any kept tag is present, drop everything outside those tags.
    inside = [False] * len(text)
    found = False
    for tag in keep_tags:
        for m in re.finditer(
            rf"<{tag}\s*>.*?</{tag}\s*>", text, re.DOTALL | re.IGNORECASE
        ):
            found = True
            for i in range(m.start(), m.end()):
                inside[i] = True
    if not found:
        return keep
    return [k and i for k, i in zip(keep, inside, strict=True)]
