"""The loss mask must not template an empty conversation (Eq. 5).

``build_masked_sequence`` finds each message's token span by templating the
conversation twice -- once up to the message, once including it -- and diffing.
For the first message the "up to" slice is empty, and transformers >= 5 raises
``ValueError: Cannot apply chat template to an empty conversation``.  The
project's own FakeTokenizer answers empty input happily, so the whole suite
passed while `deepprep train-sft` died on its first example against a real
Qwen tokenizer.
"""

from __future__ import annotations

import pytest

from deepprep.training.masking import build_masked_sequence


class StrictTokenizer:
    """A chat template with the real library's empty-conversation contract."""

    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab))

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [self._id(t) for t in text.split()]}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kw
    ) -> str:
        if not messages:
            raise ValueError(
                "Cannot apply chat template to an empty conversation. "
                "Provide at least one message."
            )
        parts = [f"<|{m['role']}|> {m['content']} <|end|>" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def decode(self, ids, skip_special_tokens=False) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "") for i in ids)


@pytest.fixture
def tok() -> StrictTokenizer:
    return StrictTokenizer()


def test_a_strict_chat_template_does_not_break_masking(tok):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER ONE"},
        {"role": "assistant", "content": "AGENT ONE"},
    ]
    seq = build_masked_sequence(tok, messages)
    assert seq.input_ids
    assert len(seq.action_mask) == len(seq.input_ids)


def test_only_the_assistant_tokens_are_trained_under_a_strict_template(tok):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER ONE"},
        {"role": "assistant", "content": "AGENT ONE"},
        {"role": "user", "content": "ENV FEEDBACK"},
        {"role": "assistant", "content": "AGENT TWO"},
    ]
    seq = build_masked_sequence(tok, messages)
    trained = tok.decode(
        [t for t, m in zip(seq.input_ids, seq.action_mask, strict=False) if m]
    ).split()

    assert "AGENT" in trained and "ONE" in trained and "TWO" in trained
    for leaked in ("SYS", "USER", "ENV", "FEEDBACK"):
        assert leaked not in trained, f"{leaked!r} leaked into the trained span"


def test_a_single_message_conversation_is_handled(tok):
    """The degenerate case: the empty prefix is the whole context."""
    seq = build_masked_sequence(tok, [{"role": "user", "content": "ONLY"}])
    assert seq.input_ids
    assert sum(seq.action_mask) == 0
