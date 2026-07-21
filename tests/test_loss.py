"""Tests for crisp_nam.utils.loss.

Covers:
- Regression guard for the pre-existing cause-specific NLL loss functions
  (just confirm they still run and return finite values).
- The new Fine-Gray subdistribution hazard loss functions, with a hand
  constructed scenario that pins down the exact semantic difference from
  the cause-specific risk set: subjects who already had a *competing*
  event remain in the Fine-Gray risk set for other causes.
"""

import torch

from crisp_nam.utils.loss import (
    fine_gray_negative_log_likelihood,
    negative_log_likelihood_loss,
    weighted_fine_gray_negative_log_likelihood,
    weighted_negative_log_likelihood_loss,
)


def _small_batch():
    """A small fixed synthetic batch with 2 competing risks."""
    torch.manual_seed(0)
    batch_size = 6
    times = torch.tensor([1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
    events = torch.tensor([1, 0, 2, 1, 0, 2])
    risk_scores = [
        torch.randn(batch_size, 1),
        torch.randn(batch_size, 1),
    ]
    return risk_scores, times, events


def test_negative_log_likelihood_loss_runs_and_is_finite():
    risk_scores, times, events = _small_batch()
    loss = negative_log_likelihood_loss(risk_scores, times, events, num_competing_risks=2)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss).item()


def test_weighted_negative_log_likelihood_loss_runs_and_is_finite():
    risk_scores, times, events = _small_batch()
    event_weights = torch.tensor([1.0, 2.0])
    sample_weights = torch.ones(times.shape[0])
    loss = weighted_negative_log_likelihood_loss(
        risk_scores,
        times,
        events,
        num_competing_risks=2,
        event_weights=event_weights,
        sample_weights=sample_weights,
    )
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss).item()


def _fine_gray_scenario():
    """Hand constructed 2-subject, 2-risk scenario.

    Subject 0 (A): competing event (cause 2) at an early time (t=1).
    Subject 1 (B): cause-1 event at a later time (t=5).

    Under the cause-specific hazard, A already left the risk set (due to
    A's own event, regardless of type) by the time B's cause-1 event
    occurs, so A must be excluded from cause-1's risk set at t=5.

    Under Fine-Gray, A's competing event does not remove A from cause-1's
    subdistribution risk set, so A must be INCLUDED at t=5.
    """
    times = torch.tensor([1.0, 5.0])
    events = torch.tensor([2, 1])
    risk_scores_1 = torch.tensor([[0.5], [1.2]])
    risk_scores_2 = torch.tensor([[0.3], [-0.7]])
    risk_scores = [risk_scores_1, risk_scores_2]
    return risk_scores, times, events


def test_fine_gray_differs_from_cause_specific():
    risk_scores, times, events = _fine_gray_scenario()

    fg_loss = fine_gray_negative_log_likelihood(risk_scores, times, events, num_competing_risks=2)
    cs_loss = negative_log_likelihood_loss(risk_scores, times, events, num_competing_risks=2)

    # This is the load-bearing regression guard: if the Fine-Gray retained
    # risk-set condition regresses to the broken `times > times[i]` variant
    # (a no-op, redundant with `times >= times[i]`), fg_loss would become
    # numerically identical to cs_loss in this scenario.
    assert not torch.allclose(fg_loss, cs_loss)


def test_fine_gray_matches_hand_computed_value():
    risk_scores, times, events = _fine_gray_scenario()
    risk_scores_1 = risk_scores[0].squeeze()
    risk_scores_2 = risk_scores[1].squeeze()

    # Cause 1 event is subject index 1 (time=5). Fine-Gray risk set at t=5
    # retains subject 0 because subject 0 had a competing event (cause 2)
    # at t=1 <= 5. So the risk set is {0, 1} -- everyone.
    term_k1 = torch.logsumexp(risk_scores_1, dim=0) - risk_scores_1[1]

    # Cause 2 event is subject index 0 (time=1). At t=1 both subjects are
    # still at risk (times >= 1 for both), and there's no earlier cause-1
    # event to retain anyone via the Fine-Gray extension, so this term is
    # identical between Fine-Gray and cause-specific.
    term_k2 = torch.logsumexp(risk_scores_2, dim=0) - risk_scores_2[0]

    expected = (term_k1 + term_k2) / 2  # n_events = 2

    fg_loss = fine_gray_negative_log_likelihood(risk_scores, times, events, num_competing_risks=2)
    assert torch.allclose(fg_loss, expected, atol=1e-6)


def test_fine_gray_risk_set_construction_directly():
    """Directly pin the risk-set membership semantics described in the docstring."""
    _, times, events = _fine_gray_scenario()

    # Cause-specific risk set for cause 1's event (i=1, time=5): only
    # subjects with times >= 5 remain -- subject 0 (time=1) is excluded.
    cause_specific_risk_set = times >= times[1]
    assert cause_specific_risk_set.tolist() == [False, True]

    # Fine-Gray risk set for cause 1's event (i=1, time=5): subject 0 is
    # retained because subject 0 had a competing event (cause 2) at
    # time=1 <= 5.
    at_risk = times >= times[1]
    already_had_competing_event = (events == 2) & (times <= times[1])
    fine_gray_risk_set = at_risk | already_had_competing_event
    assert fine_gray_risk_set.tolist() == [True, True]


def test_weighted_fine_gray_ones_matches_unweighted():
    risk_scores, times, events = _fine_gray_scenario()
    event_weights = torch.ones(2)
    sample_weights = torch.ones(times.shape[0])

    weighted = weighted_fine_gray_negative_log_likelihood(
        risk_scores,
        times,
        events,
        num_competing_risks=2,
        event_weights=event_weights,
        sample_weights=sample_weights,
    )
    unweighted = fine_gray_negative_log_likelihood(risk_scores, times, events, num_competing_risks=2)
    assert torch.allclose(weighted, unweighted, atol=1e-6)


def test_weighted_fine_gray_zero_weight_zeroes_out_risk_contribution():
    risk_scores, times, events = _fine_gray_scenario()
    risk_scores_2 = risk_scores[1].squeeze()

    # Zero out cause 1's contribution entirely.
    event_weights = torch.tensor([0.0, 1.0])
    sample_weights = torch.ones(times.shape[0])

    weighted = weighted_fine_gray_negative_log_likelihood(
        risk_scores,
        times,
        events,
        num_competing_risks=2,
        event_weights=event_weights,
        sample_weights=sample_weights,
    )

    # Only cause 2's term should survive, still divided by total n_events (2).
    term_k2 = torch.logsumexp(risk_scores_2, dim=0) - risk_scores_2[0]
    expected = term_k2 / 2
    assert torch.allclose(weighted, expected, atol=1e-6)


def test_fine_gray_all_censored_is_zero():
    torch.manual_seed(1)
    batch_size = 5
    times = torch.arange(1, batch_size + 1, dtype=torch.float32)
    events = torch.zeros(batch_size, dtype=torch.long)
    risk_scores = [torch.randn(batch_size, 1), torch.randn(batch_size, 1)]

    fg_loss = fine_gray_negative_log_likelihood(risk_scores, times, events, num_competing_risks=2)
    weighted_fg_loss = weighted_fine_gray_negative_log_likelihood(
        risk_scores, times, events, num_competing_risks=2
    )

    assert fg_loss.item() == 0.0
    assert weighted_fg_loss.item() == 0.0
