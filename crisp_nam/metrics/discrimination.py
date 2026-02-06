"""Discrimination metrics for time-to-event models with competing risks.

This module contains functions to compute the cumulative and single time-dependent AUC and time-dependent C-index for ealuating competing risks.
"""

import numpy as np

from .ipcw import estimate_ipcw


epsilon = 1e-10


def auc_td(e_test, t_test, risk_predicted_test, times, t, km=None, primary_risk=1):
    """
    Compute the time-dependent AUC for a given competing risk using predicted CIFs.

    Parameters
    ----------
        e_test : ndarray of shape (n_samples,)
            Event indicator (0=censored, 1=event of interest, 2=competing event, etc.)
        t_test : ndarray of shape (n_samples,)
            Observed time to event or censoring.
        risk_predicted_test : ndarray of shape (n_samples, n_times)
            Predicted cumulative incidence for the event of interest across time.
        times : ndarray of shape (n_times,)
            Evaluation time grid (same axis as second dim of risk_predicted_test).
        t : float
            Specific evaluation time point.
        km : KaplanMeierFitter or tuple of (e_train, t_train), optional
            For IPCW adjustment; can be None to skip weighting.
        primary_risk : int
            The event label to treat as the event of interest.

    Returns
    -------
        auc_value : float
            AUC estimate at time t (always between 0 and 1)
        km : Updated Kaplan-Meier estimator
    """
    index = np.argmin(np.abs(times - t))
    preds = risk_predicted_test[:, index]

    # Define event group and at-risk group
    event_mask = (e_test == primary_risk) & (t_test <= t)
    control_mask = t_test > t  # those still at risk

    if event_mask.sum() == 0 or control_mask.sum() == 0:
        return np.nan, km

    event_scores = preds[event_mask]
    control_scores = preds[control_mask]

    # Compute IPCW weights
    if km is None:
        weights_event = np.ones_like(event_scores)
        weights_control = np.ones_like(control_scores)
    else:
        km = estimate_ipcw(km)
        weights_event = 1.0 / np.clip(km.predict(t_test[event_mask]), epsilon, 1.0)
        weights_control = 1.0 / np.clip(km.predict(t_test[control_mask]), epsilon, 1.0)

    # Compute pairwise AUC: compare each (event, control) pair
    auc_numerator = 0.0
    auc_denominator = 0.0
    for _i, (score_i, w_i) in enumerate(zip(event_scores, weights_event)):
        for _j, (score_j, w_j) in enumerate(zip(control_scores, weights_control)):
            weight = w_i * w_j
            auc_denominator += weight
            if score_i > score_j:
                auc_numerator += weight
            elif np.isclose(score_i, score_j):
                auc_numerator += 0.5 * weight
            # else: no increment

    auc_value = auc_numerator / auc_denominator if auc_denominator > 0 else np.nan
    return auc_value, km


def cumulative_dynamic_auc(
    e_test, t_test, risk_predicted_test, times, t_eval=None, km=None, primary_risk=1
):
    """
    Compute the cumulative dynamic AUC by numerically integrating the
    time-dependent AUC over a range of time points.

    Parameters
    ----------
        e_test, t_test, risk_predicted_test, times, km, primary_risk:
            Same as in auc_td.
        t_eval: ndarray, optional
            Specific time points to evaluate. If None, uses times.

    Returns
    -------
        auc_integral: float
            The cumulative dynamic AUC.
        km: object
            Updated Kaplan-Meier estimator.
    """
    km = estimate_ipcw(km)
    t_eval = times if t_eval is None else t_eval
    aucs = [
        auc_td(e_test, t_test, risk_predicted_test, times, t, km, primary_risk)[0]
        for t in t_eval
    ]
    t_eval, aucs = t_eval[~np.isnan(aucs)], np.array(aucs)[~np.isnan(aucs)]
    if t_eval.shape[0] < 2:
        raise ValueError("At least two time points must be given")
    auc_integral = np.trapz(aucs, t_eval) / (t_eval[-1] - t_eval[0])
    return auc_integral, km


def truncated_concordance_td(
    e_test,
    t_test,
    risk_predicted_test,
    times,
    t,
    km=None,
    primary_risk=1,
    tied_tol=1e-8,
):
    """
    Compute the truncated time-dependent concordance index (C-index).

    Parameters
    ----------
        e_test : ndarray
            Event indicator (0=censored, 1=event of interest, etc.)
        t_test : ndarray
            Time-to-event or censoring
        risk_predicted_test : ndarray
            Predicted cumulative incidence (n_samples, n_timepoints)
        times : ndarray
            Time grid
        t : float
            Specific evaluation time
        km : KaplanMeierFitter or (e_train, t_train), optional
            For IPCW weighting
        primary_risk : int
            Risk of interest
        tied_tol : float
            Tolerance to assign 0.5 score for ties

    Returns
    -------
        c_index : float
        km : Updated km object
    """
    epsilon = 1e-10
    index = np.argmin(np.abs(times - t))

    # IPCW
    if km is not None:
        km = estimate_ipcw(km)
        weights_event = np.clip(km.predict(t_test), epsilon, None)
    else:
        weights_event = np.ones_like(t_test)

    # Event of interest occurred before t
    event_mask = (e_test == primary_risk) & (t_test <= t)
    if event_mask.sum() == 0:
        return np.nan, km

    nominator = 0.0
    denominator = 0.0

    for i in np.where(event_mask)[0]:
        t_i = t_test[i]
        r_i = risk_predicted_test[i, index]
        w_i = weights_event[i]

        # Define other subjects at risk
        after_mask = t_test > t_i
        before_mask = (t_test <= t_i) & (e_test != primary_risk) & (e_test != 0)

        weights_after = weights_event[after_mask] / (w_i**2)
        weights_before = weights_event[before_mask] / (w_i * weights_event[before_mask])

        risks_after = risk_predicted_test[after_mask, index]
        risks_before = risk_predicted_test[before_mask, index]

        concordant_after = (risks_after < r_i).astype(float)
        concordant_before = (risks_before < r_i).astype(float)

        concordant_after[np.abs(risks_after - r_i) <= tied_tol] = 0.5
        concordant_before[np.abs(risks_before - r_i) <= tied_tol] = 0.5

        nominator += (concordant_after * weights_after).sum()
        nominator += (concordant_before * weights_before).sum()

        denominator += weights_after.sum()
        denominator += weights_before.sum()

    if denominator == 0:
        return np.nan, km

    c_index = nominator / denominator
    return c_index, km
