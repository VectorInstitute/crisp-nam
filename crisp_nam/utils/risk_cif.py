"""Risk functions for evaluation.

This module provides functions to compute cumulative incidence functions (CIFs)
and risk scores for competing risk models.

Absolute risks are built from a Breslow baseline fitted to the model's own
training-set risk scores (`compute_breslow_baseline`). The partial likelihood
used for training is invariant to adding a constant to a cause's risk score,
so the scores have no meaningful absolute level; a baseline estimated from the
same scores absorbs that constant, which makes the predicted CIFs (and hence
calibration metrics such as the Brier score) independent of it.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from sksurv.nonparametric import cumulative_incidence_competing_risks

RISK_MODELS = ("cause_specific", "fine_gray")


@dataclass
class BreslowBaseline:
    """Breslow baseline hazard for each competing risk.

    Attributes
    ----------
        event_times: Sorted unique training times at which any event occurred, shape (G,).
        hazard_increments: Baseline hazard jumps at `event_times`, shape (K, G).
            Cause-specific hazards for `risk_model="cause_specific"`,
            subdistribution hazards for `risk_model="fine_gray"`.
        risk_model: Which hazard the increments describe.
    """

    event_times: np.ndarray
    hazard_increments: np.ndarray
    risk_model: str = "cause_specific"

    @property
    def num_competing_risks(self) -> int:
        return self.hazard_increments.shape[0]

    def cumulative_hazard(self, times: List[Any]) -> np.ndarray:
        """Baseline cumulative hazard at `times`, shape (K, len(times))."""
        cumhaz = np.concatenate(
            [
                np.zeros((self.num_competing_risks, 1)),
                np.cumsum(self.hazard_increments, axis=1),
            ],
            axis=1,
        )
        return cumhaz[:, np.searchsorted(self.event_times, np.asarray(times), side="right")]


def fit_breslow_baseline(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    risk_model: str = "cause_specific",
) -> BreslowBaseline:
    """
    Fit Breslow baseline hazards from training-set risk scores.

    The baseline hazard jump for cause k at event time s is
    `d_k(s) / sum_{j in R_k(s)} exp(f_k(x_j))`, using the same risk set
    `R_k(s)` as the training loss: subjects with `t_j >= s` for the
    cause-specific model, plus subjects who had a competing event before `s`
    for the Fine-Gray model.

    Args:
        risk_scores: Training-set risk scores f_k(x), shape (n_samples, K).
        times: Training event/censoring times, shape (n_samples,).
        events: Training event indicators (0=censored, 1...K=event types).
        risk_model: "cause_specific" or "fine_gray".

    Returns
    -------
        BreslowBaseline fitted to the given scores.
    """
    if risk_model not in RISK_MODELS:
        raise ValueError(f"risk_model must be one of {RISK_MODELS}, got {risk_model!r}")

    risk_scores = np.asarray(risk_scores, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events).astype(int)
    num_risks = risk_scores.shape[1]

    event_times = np.unique(times[events > 0])
    order = np.argsort(times)
    first_at_risk = np.searchsorted(times[order], event_times, side="left")

    increments = np.zeros((num_risks, len(event_times)))
    for k in range(num_risks):
        # Shift scores by their max before exponentiating to avoid overflow.
        shift = risk_scores[:, k].max()
        weights = np.exp(risk_scores[:, k] - shift)

        # sum of weights over {j: t_j >= s}
        tail_sums = np.concatenate([np.cumsum(weights[order][::-1])[::-1], [0.0]])
        risk_set_sums = tail_sums[first_at_risk]

        if risk_model == "fine_gray":
            # Subjects with a competing event strictly before s stay in the risk set.
            competing = (events != 0) & (events != k + 1)
            comp_order = np.argsort(times[competing])
            comp_cumsum = np.concatenate([[0.0], np.cumsum(weights[competing][comp_order])])
            n_before = np.searchsorted(times[competing][comp_order], event_times, side="left")
            risk_set_sums = risk_set_sums + comp_cumsum[n_before]

        n_events = np.bincount(
            np.searchsorted(event_times, times[events == k + 1]),
            minlength=len(event_times),
        )
        increments[k] = n_events / risk_set_sums * np.exp(-shift)

    return BreslowBaseline(event_times, increments, risk_model)


def compute_breslow_baseline(
    model: torch.nn.Module,
    x_train: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    risk_model: str = "cause_specific",
    device: str = "cpu",
) -> BreslowBaseline:
    """
    Fit Breslow baseline hazards for a trained model on its training data.

    Args:
        model: Trained model.
        x_train: Training features the model was fitted on, shape (n_samples, n_features).
        t_train: Training event/censoring times.
        e_train: Training event indicators (0=censored, 1...K=event types).
        risk_model: "cause_specific" or "fine_gray"; must match the loss the
            model was trained with.
        device: CPU or CUDA.

    Returns
    -------
        BreslowBaseline for use with `predict_absolute_risk` and
        `compute_baseline_cumulative_hazard`.
    """
    risk_scores = predict_risk(model, x_train, device)
    return fit_breslow_baseline(risk_scores, t_train, e_train, risk_model)


def compute_baseline_cif(
    times: np.ndarray, events: np.ndarray, eval_times: List[Any], event_type: np.ndarray
) -> np.ndarray:
    """
    Compute the marginal (population) cumulative incidence function for an event type.

    This describes the training population as a whole, not a subject with
    risk score zero, so it is not a valid baseline for `predict_absolute_risk`;
    use `compute_breslow_baseline` for that.

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
    Compute marginal CIFs for every competing risk.

    Convenience wrapper around `compute_baseline_cif` that builds a 0-based
    `{event_index: cif_array}` dict.

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
    baseline: BreslowBaseline,
    times: np.ndarray,
    event_of_interest: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict cumulative incidence function for a specific competing risk.

    Under the cause-specific model the CIF of one cause depends on the
    hazards of all causes, so this is a slice of `predict_absolute_risk`.

    Parameters
    ----------
        model: Trained  model.
        x: Input tensor of shape (n_samples, n_features).
        baseline: Breslow baseline from `compute_breslow_baseline`.
        times: Time points at which CIF is evaluated.
        event_of_interest: Integer, 0-based index of event of interest.
        device: CPU or CUDA.

    Returns
    -------
        cif_pred: Array of shape (n_samples, len(times)) — predicted CIF per sample.
    """
    return predict_absolute_risk(model, x, baseline, times, device)[:, event_of_interest, :]


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
    baseline: BreslowBaseline,
    times: List[Any],
    device: str = "cpu",
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Predict absolute risk (CIF) for each competing event by given time points.

    Cause-specific model: Aalen-Johansen combination of the per-cause
    hazards, `CIF_k(t|x) = sum_{s<=t} S(s-|x) dH_k(s|x)` with
    `dH_k(s|x) = dH0_k(s) exp(f_k(x))` and `S = exp(-sum_k H_k)`.
    Fine-Gray model: `CIF_k(t|x) = 1 - exp(-H0_k(t) exp(f_k(x)))` with
    `H0_k` the subdistribution baseline cumulative hazard.

    Parameters
    ----------
        model: Trained  model.
        x_input (np.ndarray or Tensor): Input features, shape (n_samples, n_features).
        baseline: Breslow baseline from `compute_breslow_baseline`, fitted
            on the model's training data.
        times (np.ndarray): Time points at which to evaluate the CIFs.
        device: CPU or CUDA.
        chunk_size: Number of samples processed at once (bounds memory use).

    Returns
    -------
        np.ndarray: Shape (n_samples, num_events, n_times) with predicted CIFs.
    """
    rel_risks = predict_risk(model, x_input, device)  # shape (n_samples, num_events)
    hazard_ratios = np.exp(rel_risks)

    if baseline.risk_model == "fine_gray":
        cumhaz = baseline.cumulative_hazard(times)  # (K, T)
        return 1.0 - np.exp(-hazard_ratios[:, :, None] * cumhaz[None])

    time_idx = np.searchsorted(baseline.event_times, np.asarray(times), side="right")
    abs_risks = np.zeros((len(rel_risks), baseline.num_competing_risks, len(time_idx)))
    for start in range(0, len(rel_risks), chunk_size):
        hr = hazard_ratios[start : start + chunk_size]
        d_hazard = baseline.hazard_increments[None] * hr[:, :, None]  # (N, K, G)
        d_total = d_hazard.sum(axis=1)  # (N, G)
        surv_before = np.exp(-(np.cumsum(d_total, axis=1) - d_total))  # S(s-|x)
        cif = np.cumsum(surv_before[:, None, :] * d_hazard, axis=2)
        cif = np.concatenate([np.zeros(cif.shape[:2] + (1,)), cif], axis=2)
        abs_risks[start : start + chunk_size] = cif[:, :, time_idx]

    return abs_risks


def compute_baseline_cumulative_hazard(
    baseline: BreslowBaseline, eval_times: List[Any], event_type: int
) -> np.ndarray:
    """
    Baseline cumulative hazard for one event type from a Breslow baseline.

    Combine with `predict_cumulative_hazard` / `plot_cumulative_hazard` to get
    `H_k(t|x) = H0_k(t) exp(f_k(x))`. The baseline must come from the same
    model's training scores; a population-level estimate (e.g. Nelson-Aalen)
    does not correspond to `f_k(x) = 0` and gives an arbitrary scale.

    Args:
        baseline: Breslow baseline from `compute_breslow_baseline`.
        eval_times: Times at which to evaluate the cumulative hazard
        event_type: Event type to return the cumulative hazard for (1...K)

    Returns
    -------
        Numpy array of baseline cumulative hazard values at eval_times.
    """
    return baseline.cumulative_hazard(eval_times)[event_type - 1]


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
    score for cause k. The baseline must be fitted on the same model's
    training scores for the result to have a meaningful scale.

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
    of CIF: each cumulative hazard is combined with its baseline
    multiplicatively (`H_k(t|x) = H0_k(t) * exp(f_k(x))`).

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
