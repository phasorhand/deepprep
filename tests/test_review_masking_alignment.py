"""Sec 5.2 gradient masking / logit-target alignment.

Sec 5.2:

    "we apply a binary token mask that removes gradients on tokens inside
     <execute> blocks, and optimize the policy only over agent decision tokens
     within <plan>, <expand>, and <answer>."

``torch`` is not installed in this environment, so the alignment between
``GRPOTrainer._token_logprobs`` (which returns one entry per *target*, i.e.
``input_ids[1:]``) and ``GRPOTrainer._action_mask_tensor`` (``action_mask[1:]``)
is proven against a numpy stand-in that implements exactly the tensor ops those
two methods use.  A shift error would be silent and catastrophic, so the test is
constructive: it corrupts the prediction of one specific token at a time and
checks that the masked sum reacts iff that token is an action token.

The second test is a regression test for a char-to-token cursor drift in
``refine_mask_to_action_tags`` found in review and since fixed.
"""

from __future__ import annotations

import contextlib
import types

import numpy as np
import pytest

from deepprep.training import build_masked_sequence, refine_mask_to_action_tags
from deepprep.training.grpo import GRPOTrainer
from deepprep.training.masking import MaskedSequence

VOCAB = 24


# --------------------------------------------------------------------------- #
# numpy stand-in for the handful of torch ops used by _token_logprobs
# --------------------------------------------------------------------------- #
class _T(np.ndarray):
    def float(self):
        return np.asarray(self, dtype=np.float64).view(_T)

    def unsqueeze(self, d):
        return np.expand_dims(self, d).view(_T)

    def squeeze(self, d=None):
        return np.asarray(self).squeeze(axis=d).view(_T)


def _fake_torch() -> types.SimpleNamespace:
    def log_softmax(x, dim=-1):
        x = np.asarray(x, dtype=np.float64)
        z = x - x.max(axis=dim, keepdims=True)
        return (z - np.log(np.exp(z).sum(axis=dim, keepdims=True))).view(_T)

    return types.SimpleNamespace(
        long=np.int64,
        float32=np.float64,
        tensor=lambda x, dtype=None, device=None: np.array(x, dtype=dtype).view(_T),
        enable_grad=contextlib.nullcontext,
        no_grad=contextlib.nullcontext,
        log_softmax=log_softmax,
        gather=lambda src, dim, index: np.take_along_axis(
            np.asarray(src), np.asarray(index), axis=dim
        ).view(_T),
    )


class _DeltaModel:
    """Logits at position ``t`` put (almost) all mass on ``pred[t]``."""

    def __init__(self, pred: list[int]) -> None:
        self.pred = pred

    def __call__(self, input_ids=None):
        length = np.asarray(input_ids).shape[1]
        logits = np.full((1, length, VOCAB), -50.0)
        for t in range(length):
            logits[0, t, self.pred[t]] = 50.0
        return types.SimpleNamespace(logits=logits.view(_T))


def _trainer() -> GRPOTrainer:
    tr = GRPOTrainer.__new__(GRPOTrainer)
    tr._torch = _fake_torch()
    tr.device = "cpu"
    return tr


# --------------------------------------------------------------------------- #
def test_logprobs_and_action_mask_are_aligned_with_no_off_by_one():
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    mask = [0, 0, 0, 1, 1, 0, 0, 0]  # policy produced input_ids[3] and input_ids[4]
    seq = MaskedSequence(input_ids=ids, action_mask=mask, spans=[(3, 5)])
    tr = _trainer()

    mask_t = np.asarray(tr._action_mask_tensor(seq))
    # One entry per *target* position: len(ids) - 1.
    assert len(mask_t) == len(ids) - 1
    assert mask_t.tolist() == [float(m) for m in mask[1:]]

    # A model that predicts every next token perfectly -> logprob ~ 0 everywhere.
    perfect = [ids[min(t + 1, len(ids) - 1)] for t in range(len(ids))]
    lp = np.asarray(tr._token_logprobs(_DeltaModel(perfect), seq, grad=False))
    assert len(lp) == len(ids) - 1
    assert np.allclose(lp, 0.0, atol=1e-6)

    # Now corrupt the prediction of exactly one token and check that the masked
    # sum moves iff that token is an action token.  This pins the alignment in
    # BOTH directions: a +1 or -1 shift would light up the neighbouring index.
    for target_idx in range(1, len(ids)):
        pred = list(perfect)
        pred[target_idx - 1] = VOCAB - 1  # mispredict input_ids[target_idx]
        lp = np.asarray(tr._token_logprobs(_DeltaModel(pred), seq, grad=False))
        masked_sum = float((lp * mask_t).sum())
        if mask[target_idx]:
            assert masked_sum < -50.0, f"action token {target_idx} not counted"
        else:
            assert masked_sum == pytest.approx(0.0), (
                f"context token {target_idx} leaked into the policy gradient"
            )


# --------------------------------------------------------------------------- #
# Regression: refine_mask_to_action_tags must not drift chars -> tokens.
# --------------------------------------------------------------------------- #
def test_refine_mask_to_action_tags_survives_a_long_execute_prefix():
    """The per-character keep mask has to be built over the SAME string the
    per-token offsets are measured against.

    Decoding the span in one call and then advancing a cursor by
    ``len(decode([tok]))`` drifts whenever the batch decode is not the exact
    concatenation of the single-token decodes -- the normal case for
    SentencePiece's word marker and for any tokenizer that re-inserts a
    separator.  The drift accumulates, so with a long enough ``<execute>`` block
    the result inverts Sec 5.2 exactly: the ``<plan>`` decision tokens are
    dropped and environment tokens survive.
    """
    from test_training import FakeTokenizer  # noqa: PLC0415

    tok = FakeTokenizer()
    messages = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": "<execute> A B C D E F G H I J </execute> <plan> KEEPME </plan>",
        },
    ]
    seq = build_masked_sequence(tok, messages)
    refined = refine_mask_to_action_tags(tok, seq)
    kept = tok.decode(
        [t for t, m in zip(refined.input_ids, refined.action_mask, strict=False) if m]
    ).split()

    # The decision tokens survive, however long the preceding <execute> block.
    assert "KEEPME" in kept
    assert "<plan>" in kept and "</plan>" in kept
    # ... and not one environment token leaks into the policy gradient.
    for env_token in ("<execute>", "</execute>", *"ABCDEFGHIJ"):
        assert env_token not in kept
