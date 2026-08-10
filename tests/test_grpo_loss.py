"""The GRPO surrogate objective (Sec 5.2, Eq. 6-8).

`assign_advantages` is already covered by pure-arithmetic tests. The loss itself
was not: it needs a model to produce log-probs, so it fell outside the suite's
torch-free design and only ran during real training. That left the parts most
likely to be silently wrong unchecked -- the sign of the advantage term, the
clip range, the +1 shift that aligns the action mask with the log-prob vector,
and the fact that environment tokens contribute nothing (which is the whole
point of Sec 5.2's gradient masking).

Real runs do not cover it either: a small policy produces identically-scored
rollouts, every advantage is 0, and the loss is a no-op.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from deepprep.agent.agent import SolveResult  # noqa: E402
from deepprep.training.grpo import GRPOConfig, GRPOTrainer, Rollout  # noqa: E402
from deepprep.training.masking import MaskedSequence  # noqa: E402
from deepprep.training.rewards import RewardBreakdown  # noqa: E402

VOCAB = 128
SEQ = [3, 9, 14, 5, 27, 8, 11, 2]
#: Two context tokens, then six sampled by the policy.
MASK = [0, 0, 1, 1, 1, 1, 1, 1]


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )


@pytest.fixture
def trainer():
    t = GRPOTrainer(GRPOConfig(model_name_or_path="<injected>", clip_eps=0.2, kl_coef=0.0))
    t._torch = torch
    t.device = "cpu"
    t.model = _tiny_model()
    return t


def _rollout(trainer, advantage: float, *, mask=None) -> Rollout:
    seq = MaskedSequence(input_ids=list(SEQ), action_mask=list(mask or MASK), spans=[(2, 8)])
    r = Rollout(
        task_id="t",
        result=SolveResult(task_id="t"),
        reward=RewardBreakdown(similarity=None),
        sequence=seq,
        advantage=advantage,
    )
    # On-policy: the behaviour policy is the current one, so every ratio is 1.
    r.old_logprobs = trainer._token_logprobs(trainer.model, seq, grad=False).detach()
    return r


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def test_the_action_mask_is_shifted_to_match_the_logprob_vector(trainer):
    seq = MaskedSequence(input_ids=list(SEQ), action_mask=list(MASK), spans=[(2, 8)])
    mask = trainer._action_mask_tensor(seq)
    logprobs = trainer._token_logprobs(trainer.model, seq, grad=False)

    assert mask.shape == logprobs.shape, "an off-by-one here trains on the wrong tokens"
    assert mask.tolist() == [float(m) for m in MASK[1:]]


# --------------------------------------------------------------------------- #
# the surrogate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("advantage", [1.0, -1.0, 0.5])
def test_on_policy_loss_is_the_negated_advantage(trainer, advantage):
    """With ratio == 1 the clipped surrogate collapses to -A."""
    loss, kl = trainer._rollout_loss(_rollout(trainer, advantage))
    assert float(loss.detach()) == pytest.approx(-advantage, abs=1e-5)
    assert kl == 0.0


def test_a_zero_advantage_makes_the_step_a_no_op(trainer):
    loss, _ = trainer._rollout_loss(_rollout(trainer, 0.0))
    loss.backward()
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)
    assert all(float(g.abs().max()) == pytest.approx(0.0, abs=1e-6) for g in grads)


def test_a_positive_advantage_is_clipped_from_above(trainer):
    """An improved policy must not be rewarded without bound (clip_eps = 0.2)."""
    r = _rollout(trainer, 1.0)
    r.old_logprobs = r.old_logprobs - 5.0  # ratio = e^5, far outside the trust region
    loss, _ = trainer._rollout_loss(r)
    assert float(loss.detach()) == pytest.approx(-(1.0 + 0.2), abs=1e-4)


def test_a_negative_advantage_is_clipped_from_below(trainer):
    r = _rollout(trainer, -1.0)
    r.old_logprobs = r.old_logprobs - 5.0
    loss, _ = trainer._rollout_loss(r)
    # min(ratio * A, clip(ratio) * A) with A < 0 picks the *unclipped* branch.
    assert float(loss.detach()) > 1.0


# --------------------------------------------------------------------------- #
# gradient masking (Sec 5.2)
# --------------------------------------------------------------------------- #
def test_environment_tokens_do_not_enter_the_objective(trainer):
    """The defining property: masked-out positions must not move the loss."""
    all_on = _rollout(trainer, 1.0, mask=[1] * len(SEQ))
    actions_only = _rollout(trainer, 1.0)

    loss_all, _ = trainer._rollout_loss(all_on)
    loss_actions, _ = trainer._rollout_loss(actions_only)
    # Both are -A on policy, so equality alone proves nothing; the denominators
    # must differ, which is what shows the mask is actually applied.
    assert trainer._action_mask_tensor(all_on.sequence).sum() == len(SEQ) - 1
    assert trainer._action_mask_tensor(actions_only.sequence).sum() == sum(MASK[1:])
    assert float(loss_all.detach()) == pytest.approx(float(loss_actions.detach()), abs=1e-5)


def test_an_all_zero_mask_does_not_divide_by_zero(trainer):
    r = _rollout(trainer, 1.0, mask=[0] * len(SEQ))
    loss, _ = trainer._rollout_loss(r)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# KL
# --------------------------------------------------------------------------- #
def test_the_kl_penalty_is_zero_against_an_identical_reference(trainer):
    trainer.cfg.kl_coef = 0.04
    r = _rollout(trainer, 1.0)
    r.ref_logprobs = r.old_logprobs.clone()
    loss, kl = trainer._rollout_loss(r)
    assert kl == pytest.approx(0.0, abs=1e-6)
    assert float(loss.detach()) == pytest.approx(-1.0, abs=1e-5)


def test_the_k3_kl_estimator_is_non_negative(trainer):
    """k3 is used precisely because the naive difference can go negative."""
    trainer.cfg.kl_coef = 0.04
    for shift in (-2.0, -0.5, 0.5, 2.0):
        r = _rollout(trainer, 0.0)
        r.ref_logprobs = r.old_logprobs + shift
        _, kl = trainer._rollout_loss(r)
        assert kl >= 0.0, f"k3 estimator went negative at shift={shift}"


def test_the_kl_penalty_raises_the_loss(trainer):
    trainer.cfg.kl_coef = 0.5
    r = _rollout(trainer, 1.0)
    r.ref_logprobs = r.old_logprobs + 1.0
    with_kl, kl = trainer._rollout_loss(r)

    trainer.cfg.kl_coef = 0.0
    without_kl, _ = trainer._rollout_loss(_rollout(trainer, 1.0))
    assert kl > 0.0
    assert float(with_kl.detach()) > float(without_kl.detach())
