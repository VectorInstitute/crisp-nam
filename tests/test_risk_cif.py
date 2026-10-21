"""Tests for crisp_nam.utils.risk_cif.

Covers the Aalen-Johansen marginal CIF estimator (`compute_baseline_cif`,
`compute_all_baseline_cifs`), the Breslow baseline (`fit_breslow_baseline`,
`compute_breslow_baseline`) and the absolute-risk and cumulative-hazard
predictions built on it (`predict_absolute_risk`, `predict_cif`,
`compute_baseline_cumulative_hazard`, `predict_cumulative_hazard`,
`predict_all_cumulative_hazards`).
"""

import numpy as np
import pytest
import torch
from torch import nn

from crisp_nam.models import CrispNamModel
from crisp_nam.utils.risk_cif import (
    compute_all_baseline_cifs,
    compute_baseline_cif,
    compute_baseline_cumulative_hazard,
    compute_breslow_baseline,
    fit_breslow_baseline,
    predict_absolute_risk,
    predict_all_cumulative_hazards,
    predict_cif,
    predict_cumulative_hazard,
)


def _synthetic_competing_risk_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    times = rng.exponential(scale=5.0, size=n)
    events = rng.integers(0, 3, size=n)  # 0=censored, 1, 2
    return times, events


def _tiny_model(num_features=4, num_competing_risks=2, seed=0):
    torch.manual_seed(seed)
    model = CrispNamModel(
        num_features=num_features,
        num_competing_risks=num_competing_risks,
        hidden_sizes=(8, 8),
        dropout_rate=0.0,
        feature_dropout=0.0,
    )
    model.eval()
    return model


class _ShiftedModel(nn.Module):
    """Adds a constant to every risk score; leaves the partial likelihood unchanged."""

    def __init__(self, base, shift):
        super().__init__()
        self.base = base
        self.shift = shift

    def forward(self, x):
        risk_scores, feature_outputs = self.base(x)
        return [r + self.shift for r in risk_scores], feature_outputs


class _ConstantModel(nn.Module):
    """Gives every subject a risk score of zero for both causes."""

    def forward(self, x):
        return [torch.zeros(len(x), 1), torch.zeros(len(x), 1)], None


def _train_features(model, n, seed=1):
    return np.random.default_rng(seed).normal(size=(n, model.num_features)).astype(np.float32)


def test_compute_baseline_cif_shape_range_monotone():
    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 20)

    cif = compute_baseline_cif(times, events, eval_times, event_type=1)

    assert cif.shape == (len(eval_times),)
    assert np.all(cif >= 0.0) and np.all(cif <= 1.0)
    assert np.all(np.diff(cif) >= -1e-9)


def test_compute_all_baseline_cifs_matches_individual_calls():
    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 15)
    num_competing_risks = 2

    all_cifs = compute_all_baseline_cifs(times, events, eval_times, num_competing_risks)

    assert set(all_cifs.keys()) == {0, 1}
    for k in range(num_competing_risks):
        individual = compute_baseline_cif(times, events, eval_times, event_type=k + 1)
        assert np.allclose(all_cifs[k], individual)


def test_fit_breslow_baseline_rejects_unknown_risk_model():
    times, events = _synthetic_competing_risk_data()
    with pytest.raises(ValueError):
        fit_breslow_baseline(np.zeros((len(times), 2)), times, events, risk_model="cox")


@pytest.mark.parametrize("risk_model", ["cause_specific", "fine_gray"])
def test_absolute_risk_invariant_to_constant_score_shift(risk_model):
    # Regression test: the partial likelihood cannot identify a constant offset in
    # the risk scores, so predicted CIFs must not depend on it either.
    model = _tiny_model()
    times, events = _synthetic_competing_risk_data()
    x_train = _train_features(model, len(times))
    x_test = _train_features(model, 20, seed=2)
    eval_times = np.linspace(0.1, times.max() * 0.9, 8)

    risks = {}
    for shift in (0.0, 3.0):
        shifted = _ShiftedModel(model, shift)
        baseline = compute_breslow_baseline(shifted, x_train, times, events, risk_model)
        risks[shift] = predict_absolute_risk(shifted, x_test, baseline, eval_times)

    assert np.allclose(risks[0.0], risks[3.0], atol=1e-10)


def test_cause_specific_with_constant_scores_matches_aalen_johansen():
    times, events = _synthetic_competing_risk_data(n=500)
    eval_times = np.linspace(0.1, times.max() * 0.9, 10)
    baseline = fit_breslow_baseline(np.zeros((len(times), 2)), times, events)

    # A model with identical scores for everyone reduces to the marginal CIF.
    cif = predict_absolute_risk(
        _ConstantModel(), np.zeros((1, 4), dtype=np.float32), baseline, eval_times
    )

    for k in range(2):
        expected = compute_baseline_cif(times, events, eval_times, event_type=k + 1)
        assert np.allclose(cif[0, k], expected, atol=0.02)


def test_fine_gray_with_constant_scores_and_no_censoring_matches_empirical_cif():
    rng = np.random.default_rng(3)
    times = rng.exponential(scale=5.0, size=500)
    events = rng.integers(1, 3, size=500)  # no censoring
    eval_times = np.linspace(0.1, times.max() * 0.9, 10)
    baseline = fit_breslow_baseline(
        np.zeros((len(times), 2)), times, events, risk_model="fine_gray"
    )

    cumhaz = baseline.cumulative_hazard(eval_times)
    for k in range(2):
        empirical = np.array([np.mean((times <= t) & (events == k + 1)) for t in eval_times])
        assert np.allclose(1.0 - np.exp(-cumhaz[k]), empirical, atol=0.02)


@pytest.mark.parametrize("risk_model", ["cause_specific", "fine_gray"])
def test_predict_absolute_risk_shape_range_monotone(risk_model):
    model = _tiny_model()
    times, events = _synthetic_competing_risk_data()
    x_train = _train_features(model, len(times))
    x = torch.randn(10, model.num_features)
    eval_times = np.linspace(0.1, times.max() * 0.9, 8)

    baseline = compute_breslow_baseline(model, x_train, times, events, risk_model)
    abs_risk = predict_absolute_risk(model, x, baseline, eval_times)

    assert abs_risk.shape == (10, 2, len(eval_times))
    assert np.all(abs_risk >= 0.0) and np.all(abs_risk <= 1.0)
    assert np.all(np.diff(abs_risk, axis=2) >= -1e-12)
    if risk_model == "cause_specific":
        assert np.all(abs_risk.sum(axis=1) <= 1.0 + 1e-12)


def test_predict_absolute_risk_chunking_is_exact():
    model = _tiny_model()
    times, events = _synthetic_competing_risk_data()
    x_train = _train_features(model, len(times))
    x = _train_features(model, 25, seed=4)
    eval_times = np.linspace(0.1, times.max() * 0.9, 6)
    baseline = compute_breslow_baseline(model, x_train, times, events)

    full = predict_absolute_risk(model, x, baseline, eval_times)
    chunked = predict_absolute_risk(model, x, baseline, eval_times, chunk_size=7)

    assert np.array_equal(full, chunked)


def test_predict_cif_matches_absolute_risk_slice():
    model = _tiny_model()
    times, events = _synthetic_competing_risk_data()
    x_train = _train_features(model, len(times))
    x = torch.randn(10, model.num_features)
    eval_times = np.linspace(0.1, times.max() * 0.9, 8)
    baseline = compute_breslow_baseline(model, x_train, times, events)

    cif_pred = predict_cif(model, x, baseline, eval_times, event_of_interest=1)

    assert cif_pred.shape == (10, len(eval_times))
    assert np.allclose(cif_pred, predict_absolute_risk(model, x, baseline, eval_times)[:, 1])


def test_compute_baseline_cumulative_hazard_shape_monotone_and_beyond_followup():
    model = _tiny_model()
    times, events = _synthetic_competing_risk_data()
    baseline = compute_breslow_baseline(model, _train_features(model, len(times)), times, events)
    eval_times = np.linspace(0.1, times.max() * 2.0, 30)

    cumhaz = compute_baseline_cumulative_hazard(baseline, eval_times, event_type=1)

    assert cumhaz.shape == (len(eval_times),)
    assert not np.isnan(cumhaz).any()
    assert np.all(cumhaz >= 0.0)
    assert np.all(np.diff(cumhaz) >= -1e-12)


def test_predict_cumulative_hazard_matches_ph_multiplicative_property():
    model = _tiny_model()
    n_samples = 6
    x = torch.randn(n_samples, model.num_features)

    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 12)
    baseline = compute_breslow_baseline(model, _train_features(model, len(times)), times, events)
    baseline_cumhazard = compute_baseline_cumulative_hazard(baseline, eval_times, event_type=1)
    event_of_interest = 0  # 0-based index into risk_scores list -> cause 1

    predicted = predict_cumulative_hazard(model, x, baseline_cumhazard, event_of_interest)

    with torch.no_grad():
        risk_scores, _ = model(x)
    f_x = risk_scores[event_of_interest].squeeze(1).cpu().numpy()
    expected = baseline_cumhazard.reshape(1, -1) * np.exp(f_x).reshape(-1, 1)

    assert predicted.shape == (n_samples, len(eval_times))
    assert np.allclose(predicted, expected)


def test_predict_all_cumulative_hazards_shape_and_consistency():
    model = _tiny_model()
    n_samples = 6
    x_np = _train_features(model, n_samples, seed=5)

    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 10)
    num_competing_risks = 2
    baseline = compute_breslow_baseline(model, _train_features(model, len(times)), times, events)
    baseline_cumhazards = {
        k: compute_baseline_cumulative_hazard(baseline, eval_times, event_type=k + 1)
        for k in range(num_competing_risks)
    }

    all_hazards = predict_all_cumulative_hazards(model, x_np, baseline_cumhazards, eval_times)

    assert all_hazards.shape == (n_samples, num_competing_risks, len(eval_times))
    assert np.all(all_hazards >= 0.0)

    x_tensor = torch.from_numpy(x_np).float()
    for k in range(num_competing_risks):
        individual = predict_cumulative_hazard(model, x_tensor, baseline_cumhazards[k], k)
        assert np.allclose(all_hazards[:, k, :], individual)
