"""Tests for crisp_nam.utils.plotting.

Uses the non-interactive "Agg" matplotlib backend so tests run headlessly.
Covers the new `plot_cumulative_hazard` function plus light smoke tests
for the pre-existing `plot_feature_importance` and
`plot_coxnam_shape_functions`, since neither previously had any coverage.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from crisp_nam.models import CrispNamModel
from crisp_nam.utils.plotting import (
    plot_coxnam_shape_functions,
    plot_cumulative_hazard,
    plot_feature_importance,
)
from crisp_nam.utils.risk_cif import compute_baseline_cumulative_hazard


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


def _synthetic_competing_risk_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    times = rng.exponential(scale=5.0, size=n)
    events = rng.integers(0, 3, size=n)
    return times, events


def test_plot_cumulative_hazard_runs_without_error():
    model = _tiny_model()
    n_samples = 10
    x_data = torch.randn(n_samples, model.num_features)

    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 10)
    baseline_cumhazard = compute_baseline_cumulative_hazard(
        times, events, eval_times, event_type=1
    )

    fig, ax = plot_cumulative_hazard(model, x_data, baseline_cumhazard, eval_times, risk_idx=1)

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_cumulative_hazard_writes_output_file(tmp_path):
    model = _tiny_model()
    n_samples = 5
    x_data = torch.randn(n_samples, model.num_features)

    times, events = _synthetic_competing_risk_data()
    eval_times = np.linspace(0.1, times.max() * 0.9, 10)
    baseline_cumhazard = compute_baseline_cumulative_hazard(
        times, events, eval_times, event_type=1
    )

    output_file = tmp_path / "cumhazard.png"
    fig, ax = plot_cumulative_hazard(
        model,
        x_data,
        baseline_cumhazard,
        eval_times,
        risk_idx=1,
        output_file=str(output_file),
    )
    plt.close(fig)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_feature_importance_smoke():
    model = _tiny_model()
    n_samples = 20
    x_data = torch.randn(n_samples, model.num_features)
    feature_names = [f"feat_{i}" for i in range(model.num_features)]

    fig, ax, top_pos, top_neg = plot_feature_importance(
        model, x_data, feature_names=feature_names, n_top=2, n_bottom=2, risk_idx=1
    )

    assert fig is not None
    assert ax is not None
    assert isinstance(top_pos, list)
    assert isinstance(top_neg, list)
    plt.close(fig)


def test_plot_coxnam_shape_functions_smoke():
    model = _tiny_model()
    n_samples = 20
    x_data = torch.randn(n_samples, model.num_features)

    fig, axes = plot_coxnam_shape_functions(model, x_data, risk_to_plot=1, ncols=2)

    assert fig is not None
    assert len(axes) == model.num_features
    plt.close(fig)
