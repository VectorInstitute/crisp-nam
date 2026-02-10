"""IPCW estimation for time-to-event models with competing risks.

This module provides a function to estimate the inverse probability of censoring weights (IPCW) using a
Kaplan-Meier estimator.
"""

from lifelines import KaplanMeierFitter


def estimate_ipcw(km: tuple | KaplanMeierFitter) -> KaplanMeierFitter:
    """Estimate the inverse probability of censoring weights (IPCW)
    using a Kaplan-Meier estimator.

    Parameters
    ----------
    km : tuple or KaplanMeierFitter

    Returns
    -------
    kmf : KaplanMeierFitter
        A KaplanMeierFitter instance fitted to the provided data or
        the input instance if already fitted.
    """

    if isinstance(km, tuple):
        kmf = KaplanMeierFitter()
        e_train, t_train = km
        # For IPCW, we need to reverse the event indicator
        # For censoring distribution, events are when subject is censored (e_train == 0)
        c_train = (e_train == 0).astype(int)  # Convert boolean to int
        kmf.fit(t_train, event_observed=c_train)
    else:
        kmf = km
    return kmf
