"""Risk functions for evaluation.

This module provides functions to compute cumulative incidence functions (CIFs)
and risk scores for competing risk models.
"""

from typing import Any, Dict, List

import numpy as np
import torch
from sksurv.nonparametric import cumulative_incidence_competing_risks


def compute_baseline_cif(
    times: np.ndarray, events: np.ndarray, eval_times: List[Any], event_type: np.ndarray
) -> np.ndarray:
    """
    Compute baseline cumulative incidence function for a specific event type.

    Uses the Aalen-Johansen nonparametric estimator (via
    `sksurv.nonparametric.cumulative_incidence_competing_risks`) rather than a
    naive event-count ratio. The Aalen-Johansen estimator is the standard
    nonparametric estimator of the CIF in the presence of competing risks and
    censoring, and is statistically correct in settings where the naive
    event-count / n_samples ratio would be biased.

    Args:
        times: Numpy array of event times
        events: Numpy array of event indicators (0=censored, 1...K=event types)
        eval_times: Times at which to evaluate the CIF
        event_type: Event type to compute CIF for (1...K)

    Returns
    -------
        Numpy array of baseline CIF values at eval_times
    """
    unique_times, cum_incidence = cumulative_incidence_competing_risks(
        event=events.astype(int), time_exit=times
    )
    cif_for_event = cum_incidence[event_type]  # index 0 is total risk across all events
    baseline_cif = np.interp(eval_times, unique_times, cif_for_event)
    baseline_cif = np.clip(baseline_cif, 0, 1)

    return baseline_cif


def compute_all_baseline_cifs(
    times: np.ndarray, events: np.ndarray, eval_times: List[Any], num_competing_risks: int
) -> Dict[int, np.ndarray]:
    """
    Compute baseline CIFs for every competing risk.

    Convenience wrapper around `compute_baseline_cif` that builds the
    0-based `{event_index: baseline_cif_array}` dict expected by
    `predict_absolute_risk` and by the training scripts (e.g.
    `training_scripts/train.py`), which otherwise construct this dict
    inline via `{k: compute_baseline_cif(..., k + 1) for k in range(...)}`.

    Args:
        times: Numpy array of event times
        events: Numpy array of event indicators (0=censored, 1...K=event types)
        eval_times: Times at which to evaluate the CIF
        num_competing_risks: Number of competing risks, K

    Returns
    -------
        Dict mapping 0-based event index k (0...K-1) to the baseline CIF
        array of shape (len(eval_times),) for event type k + 1.
    """
    return {
        k: compute_baseline_cif(times, events, eval_times, k + 1)
        for k in range(num_competing_risks)
    }


def predict_cif(
    model: torch.nn.Module,
    x: np.ndarray,
    baseline_cif: np.ndarray,
    times: np.ndarray,
    event_of_interest: int,
) -> np.ndarray:
    """
    Predict cumulative incidence function for a specific competing risk.

    Parameters
    ----------
        model: Trained  model.
        x: Input tensor of shape (n_samples, n_features).
        baseline_cif: Array of shape (len(times),) —
        estimated CIF for baseline (e.g. from compute_baseline_cif).
        times: Time points at which CIF is evaluated.
        event_type: Integer, 0-based index of event of interest.

    Returns
    -------
        cif_pred: Array of shape (n_samples, len(times)) — predicted CIF per sample.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)  # list of length num_risks
        f_j_x = logits[event_of_interest].squeeze(1).cpu().numpy()  # (n_samples,)

    baseline_cif = np.asarray(baseline_cif).reshape(1, -1)  # (1, T)
    risk_scores = np.exp(f_j_x).reshape(-1, 1)  # (N, 1)

    # Return Fine-Gray style CIF prediction under PH assumption
    return 1.0 - np.power(1.0 - baseline_cif, risk_scores)  # shape (N, T)


def predict_risk(
    model: torch.nn.Module, x_input: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """
    Predicts relative risk scores for each competing risk.

    Args:
        model : Trained model.
        x_input (np.ndarray or torch.Tensor): Input features of
        shape (n_samples, n_features).
        device (str): Device to run the computation on.

    Returns
    -------
        np.ndarray: Array of shape (n_samples, num_risks) with relative risk scores.
    """
    model.eval()

    if isinstance(x_input, np.ndarray):
        x_tensor = torch.from_numpy(x_input).float().to(device)
    else:
        x_tensor = x_input.to(device).float()

    with torch.no_grad():
        risk_outputs, _ = model(x_tensor)  # List of [batch_size, 1] tensors
        risks = torch.cat(risk_outputs, dim=1)  # Shape: [batch_size, num_risks]

    return risks.cpu().numpy()


def predict_absolute_risk(
    model: torch.nn.Module,
    x_input: np.ndarray,
    baseline_cifs: List[Any],
    times: List[Any],
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict absolute risk (CIF) for each competing event by given time points.

    Parameters
    ----------
        model: Trained  model.
        x_input (np.ndarray or Tensor): Input features, shape (n_samples, n_features).
        baseline_cifs (dict): Mapping of event index to baseline CIF
        array of shape (n_times,).
        times (np.ndarray): Time grid used for baseline_cifs.
        device: CPU or CUDA.

    Returns
    -------
        np.ndarray: Shape (n_samples, num_events, n_times) with predicted CIFs.
    """
    rel_risks = predict_risk(model, x_input, device)  # shape (n_samples, num_events)
    n_samples, num_events = rel_risks.shape
    n_times = len(times)

    abs_risks = np.zeros((n_samples, num_events, n_times))

    for k in range(num_events):
        base_cif = np.clip(baseline_cifs[k], 1e-10, 0.9999)  # avoid edge cases
        for i in range(n_samples):
            abs_risks[i, k, :] = 1 - np.power(1 - base_cif, np.exp(rel_risks[i, k]))

    return abs_risks


def compute_baseline_cumulative_hazard(
    times: np.ndarray, events: np.ndarray, eval_times: List[Any], event_type: int
) -> np.ndarray:
    """
    Compute the cause-specific Nelson-Aalen baseline cumulative hazard.

    Treats occurrences of `event_type` as the event of interest and all
    other outcomes (censoring, and competing events of a different type)
    as censored, then fits a Nelson-Aalen estimator (via `lifelines`) to
    get the cause-specific baseline cumulative hazard function.

    Args:
        times: Numpy array of event times
        events: Numpy array of event indicators (0=censored, 1...K=event types)
        eval_times: Times at which to evaluate the cumulative hazard
        event_type: Event type to compute the cause-specific cumulative hazard for (1...K)

    Returns
    -------
        Numpy array of baseline cumulative hazard values at eval_times.
    """
    from lifelines import NelsonAalenFitter

    event_observed = events == event_type
    naf = NelsonAalenFitter()
    naf.fit(durations=times, event_observed=event_observed)
    baseline_cumhazard = naf.cumulative_hazard_at_times(np.asarray(eval_times)).to_numpy()
    baseline_cumhazard = np.clip(baseline_cumhazard, 0, None)
    # enforce monotone non-decreasing as a numerical safety net
    baseline_cumhazard = np.maximum.accumulate(baseline_cumhazard)

    return baseline_cumhazard


def predict_cumulative_hazard(
    model: torch.nn.Module,
    x: np.ndarray,
    baseline_cumhazard: np.ndarray,
    event_of_interest: int,
) -> np.ndarray:
    """
    Predict cause-specific cumulative hazard for a specific competing risk.

    Under the proportional-hazards assumption, the cause-specific cumulative
    hazard is `H_k(t|x) = H0_k(t) * exp(f_k(x))`, where `H0_k` is the
    cause-specific baseline cumulative hazard (e.g. from
    `compute_baseline_cumulative_hazard`) and `f_k(x)` is the model's risk
    score for cause k. Mirrors `predict_cif`'s structure (model.eval(),
    no_grad, extract the risk score for `event_of_interest` as a 0-based
    index into the model's per-risk output list), but combines with the
    baseline multiplicatively instead of via the CIF power-transform used
    in `predict_cif`.

    Parameters
    ----------
        model: Trained model.
        x: Input tensor of shape (n_samples, n_features).
        baseline_cumhazard: Array of shape (len(times),) — estimated
        cause-specific baseline cumulative hazard (e.g. from
        compute_baseline_cumulative_hazard).
        event_of_interest: Integer, 0-based index of event of interest.

    Returns
    -------
        Array of shape (n_samples, len(baseline_cumhazard)) with predicted
        cumulative hazard per sample.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)  # list of length num_risks
        f_j_x = logits[event_of_interest].squeeze(1).cpu().numpy()  # (n_samples,)

    hazard_ratios = np.exp(f_j_x).reshape(-1, 1)  # (N, 1)
    baseline_cumhazard = np.asarray(baseline_cumhazard).reshape(1, -1)  # (1, T)

    return hazard_ratios * baseline_cumhazard  # (N, T)


def predict_all_cumulative_hazards(
    model: torch.nn.Module,
    x_input: np.ndarray,
    baseline_cumhazards: Dict[int, np.ndarray],
    times: List[Any],
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict cumulative hazard for every competing risk.

    Analogous to `predict_absolute_risk` but for cumulative hazard instead
    of CIF: each cause-specific cumulative hazard is combined with its
    baseline multiplicatively (`H_k(t|x) = H0_k(t) * exp(f_k(x))`), not via
    the CIF power-transform used in `predict_absolute_risk`.

    Parameters
    ----------
        model: Trained model.
        x_input (np.ndarray or Tensor): Input features, shape (n_samples, n_features).
        baseline_cumhazards (dict): Mapping of event index to baseline
        cumulative hazard array of shape (n_times,).
        times (np.ndarray): Time grid used for baseline_cumhazards.
        device: CPU or CUDA.

    Returns
    -------
        np.ndarray: Shape (n_samples, num_events, n_times) with predicted
        cumulative hazards.
    """
    rel_risks = predict_risk(model, x_input, device)  # (n_samples, num_events)
    n_samples, num_events = rel_risks.shape
    n_times = len(times)

    cumhaz = np.zeros((n_samples, num_events, n_times))
    for k in range(num_events):
        baseline = np.asarray(baseline_cumhazards[k]).reshape(1, -1)
        cumhaz[:, k, :] = np.exp(rel_risks[:, k]).reshape(-1, 1) * baseline

    return cumhaz
