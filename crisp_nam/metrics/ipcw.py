from lifelines import KaplanMeierFitter

def estimate_ipcw(km):
    """
    Estimate the inverse probability of censoring weights (IPCW) using a Kaplan-Meier estimator.

    Parameters:
    -----------
    km : tuple or KaplanMeierFitter
        If `km` is a tuple, it should contain two elements:
        - e_train: array-like, event indicators (1 if the event occurred, 0 if censored).
        - t_train: array-like, corresponding event or censoring times.
        If `km` is already a fitted KaplanMeierFitter instance, it will be used directly.

    Returns:
    --------
    kmf : KaplanMeierFitter
        A KaplanMeierFitter instance fitted to the provided data or the input instance if already fitted.
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


