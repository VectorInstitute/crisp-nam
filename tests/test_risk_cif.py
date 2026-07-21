"""Tests for crisp_nam.utils.risk_cif.

Covers the Aalen-Johansen baseline CIF estimator (`compute_baseline_cif`,
`compute_all_baseline_cifs`), the Nelson-Aalen baseline cumulative hazard
estimator (`compute_baseline_cumulative_hazard`), the proportional-hazards
cumulative-hazard prediction functions (`predict_cumulative_hazard`,
`predict_all_cumulative_hazards`), and regression guards for the
pre-existing `predict_cif` / `predict_absolute_risk` functions.
"""

import numpy as np
import torch

from crisp_nam.models import CrispNamModel
from crisp_nam.utils.risk_cif import (
    compute_all_baseline_cifs,
    compute_baseline_cif,
    compute_baseline_cumulative_hazard,
    predict_absolute_risk,
    predict_cif,
    predict_all_cumulative_hazards,
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


def test_compute_baseline_cumulative_hazard_shape_and_monotone():
    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 20)

    cumhaz = compute_baseline_cumulative_hazard(times, events, eval_times, event_type=1)

    assert cumhaz.shape == (len(eval_times),)
    assert np.all(cumhaz >= 0.0)
    assert np.all(np.diff(cumhaz) >= -1e-9)


def test_compute_baseline_cumulative_hazard_no_nan_beyond_followup():
    times, events = _synthetic_competing_risk_data()

    # Evaluate far beyond the maximum observed follow-up time.
    far_times = np.array([times.max() * 5.0])
    cumhaz = compute_baseline_cumulative_hazard(times, events, far_times, event_type=1)

    assert cumhaz.shape == (1,)
    assert not np.isnan(cumhaz).any()

    # Also check a grid that straddles the max follow-up time.
    eval_times = np.linspace(0.1, times.max() * 2.0, 30)
    cumhaz_grid = compute_baseline_cumulative_hazard(times, events, eval_times, event_type=1)
    assert not np.isnan(cumhaz_grid).any()
    assert np.all(np.diff(cumhaz_grid) >= -1e-9)


def test_predict_cif_regression_guard():
    model = _tiny_model()
    n_samples = 10
    x = torch.randn(n_samples, model.num_features)

    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 8)
    baseline_cif = compute_baseline_cif(times, events, eval_times, event_type=1)

    cif_pred = predict_cif(model, x, baseline_cif, eval_times, event_of_interest=0)

    assert cif_pred.shape == (n_samples, len(eval_times))
    assert np.all(cif_pred >= 0.0) and np.all(cif_pred <= 1.0)


def test_predict_absolute_risk_regression_guard():
    model = _tiny_model()
    n_samples = 10
    x = torch.randn(n_samples, model.num_features)

    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 8)
    baseline_cifs = compute_all_baseline_cifs(times, events, eval_times, num_competing_risks=2)

    abs_risk = predict_absolute_risk(model, x, baseline_cifs, eval_times)

    assert abs_risk.shape == (n_samples, 2, len(eval_times))
    assert np.all(abs_risk >= 0.0) and np.all(abs_risk <= 1.0)


def test_predict_cumulative_hazard_matches_ph_multiplicative_property():
    model = _tiny_model()
    n_samples = 6
    x = torch.randn(n_samples, model.num_features)

    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 12)
    baseline_cumhazard = compute_baseline_cumulative_hazard(
        times, events, eval_times, event_type=1
    )
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
    x_np = np.random.default_rng(1).normal(size=(n_samples, model.num_features)).astype(
        np.float32
    )

    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 10)
    num_competing_risks = 2
    baseline_cumhazards = {
        k: compute_baseline_cumulative_hazard(times, events, eval_times, event_type=k + 1)
        for k in range(num_competing_risks)
    }

    all_hazards = predict_all_cumulative_hazards(model, x_np, baseline_cumhazards, eval_times)

    assert all_hazards.shape == (n_samples, num_competing_risks, len(eval_times))
    assert np.all(all_hazards >= 0.0)

    x_tensor = torch.from_numpy(x_np).float()
    for k in range(num_competing_risks):
        individual = predict_cumulative_hazard(model, x_tensor, baseline_cumhazards[k], k)
        assert np.allclose(all_hazards[:, k, :], individual)
