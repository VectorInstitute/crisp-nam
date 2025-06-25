import numpy as np
from .ipcw import estimate_ipcw

# A small constant to avoid division by zero
epsilon = 1e-4

def brier_score(e_test, t_test, risk_predicted_test, times, t, km=None, primary_risk=1):
    """
    Compute the corrected Brier score for a given competing risk.
    
    This implementation is based on the work of Schoop et al. on quantifying the
    predictive accuracy of time-to-event models in the presence of competing risks.
    
    Parameters:
        e_test (ndarray): Array of event indicators (0 = censored; positive integers for different events).
        t_test (ndarray): Array of event/censoring times.
        risk_predicted_test (ndarray): Predicted risk matrix with shape (n_samples, n_times).
        times (ndarray): Array of time points corresponding to columns in risk_predicted_test.
        t (float): Time at which to evaluate the Brier score.
        km (object, optional): Kaplan–Meier estimator or data to estimate the censoring distribution.
        primary_risk (int, optional): The event label for which to compute the score.
        
    Returns:
        brier (float): The corrected Brier score evaluated at time t.
        km (object): Updated Kaplan–Meier estimator (if applicable).
    """
    # Binary truth: True if event of interest (primary_risk) occurs by time t.
    truth = (e_test == primary_risk) & (t_test <= t)
    # Find index of the time horizon closest to t.
    index = np.argmin(np.abs(times - t))
    km = estimate_ipcw(km)

    if truth.sum() == 0:
        return np.nan, km

    # If no KM is provided, compute unweighted Brier score.
    if km is None:
        return ((truth - risk_predicted_test[:, index]) ** 2).mean(), km

    # Initialize weights for IPCW correction.
    weights = np.zeros_like(e_test, dtype=float)
    # For subjects with events (or censoring) before t (excluding those censored exactly at 0 event label), use KM weights.
    mask = (t_test <= t) & (e_test != 0)
    weights[mask] = 1. / np.clip(km.survival_function_at_times(t_test[mask]), epsilon, None)
    # For subjects still at risk at time t, assign constant weight based on KM at time t.
    weights[t_test > t] = 1. / np.clip(km.survival_function_at_times(t), epsilon, None)

    brier = (weights * (truth - risk_predicted_test[:, index]) ** 2).mean()
    return brier, km


def integrated_brier_score(e_test, t_test, risk_predicted_test, times, t_eval=None, km=None, primary_risk=1):
    """
    Compute the integrated Brier score for competing risks over a range of time points.
    
    The integrated Brier score is computed by numerically integrating the Brier score over the evaluation times.
    
    Parameters:
        e_test (ndarray): Event indicators.
        t_test (ndarray): Event/censoring times.
        risk_predicted_test (ndarray): Predicted risk matrix with shape (n_samples, n_times).
        times (ndarray): Array of time points corresponding to the predictions.
        t_eval (ndarray, optional): Specific time points at which to compute the score. Defaults to using 'times'.
        km (object, optional): Kaplan–Meier estimator for IPCW.
        primary_risk (int, optional): The event label for which to compute the score.
        
    Returns:
        ibs (float): Integrated Brier score.
        km (object): Updated Kaplan–Meier estimator.
    """
    km = estimate_ipcw(km)
    t_eval = times if t_eval is None else t_eval
    # Compute Brier scores at each time point.
    brier_scores = [brier_score(e_test, t_test, risk_predicted_test, times, t_val, km, primary_risk)[0] 
                    for t_val in t_eval]
    # Remove NaN values if any.
    t_eval = t_eval[~np.isnan(brier_scores)]
    brier_scores = np.array(brier_scores)[~np.isnan(brier_scores)]
    
    if t_eval.shape[0] < 2:
        raise ValueError("At least two time points must be provided for integration.")
    
    ibs = np.trapz(brier_scores, t_eval) / (t_eval[-1] - t_eval[0])
    return ibs, km

