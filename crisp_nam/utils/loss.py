"""Loss functions for competing risks.

This module implements weighted and un-weighted
negative log-likelihood loss, L2 penalty loss functions.
"""

import torch


def _masked_log_risk_sums(risk_k, at_risk, log_weights=None):
    """
    Compute per-row logsumexp of `risk_k` restricted to each row's risk set.

    Parameters
    ----------
        risk_k: Tensor of shape (batch_size,) with risk scores for one competing risk
        at_risk: Boolean tensor of shape (batch_size, batch_size); at_risk[i, j] is
        True if sample j is in sample i's risk set
        log_weights: Optional tensor of shape (batch_size, batch_size) with the log
        weight of sample j in sample i's risk set

    Returns
    -------
        Tensor of shape (batch_size,): log_risk_sum for each row i
    """
    neg_inf = torch.finfo(risk_k.dtype).min
    masked_scores = risk_k.unsqueeze(0).expand(at_risk.shape[0], -1)
    if log_weights is not None:
        masked_scores = masked_scores + log_weights
    masked_scores = torch.where(
        at_risk, masked_scores, torch.full_like(masked_scores, neg_inf)
    )
    return torch.logsumexp(masked_scores, dim=1)


def _censoring_survival_before(times, events):
    """
    Kaplan-Meier estimate of the censoring survival G(t-) at each sample's own time.

    Estimated from the batch itself; subjects with an event at a censoring
    time count as still at risk of censoring at that time (matching
    `crisp_nam.utils.risk_cif.censoring_survival_before`).

    Parameters
    ----------
        times: Event/censoring times (batch_size,)
        events: Event indicators (0=censored, 1...K=event types) (batch_size,)

    Returns
    -------
        Tensor of shape (batch_size,) with G(times[i]-)
    """
    cens_times, n_cens = torch.unique(times[events == 0], return_counts=True)
    if cens_times.numel() == 0:
        return torch.ones_like(times, dtype=torch.float)
    n_at_risk = (times.unsqueeze(0) >= cens_times.unsqueeze(1)).sum(dim=1)
    log_factors = torch.log1p(-n_cens.float() / n_at_risk.float())  # (C,)
    before = cens_times.unsqueeze(0) < times.unsqueeze(1)  # (batch, C)
    log_surv = torch.where(before, log_factors, torch.zeros_like(log_factors)).sum(dim=1)
    return torch.exp(log_surv)


def _fine_gray_risk_set(times, events, k):
    """
    Subdistribution risk set and IPCW log-weights for cause k (Fine & Gray, 1999).

    Sample j is in sample i's risk set if `times[j] >= times[i]` (weight 1),
    or if j had a competing event (type != 0, != k) before `times[i]`, with
    weight `G(times[i]-) / G(times[j]-)`, where G is the censoring survival
    function. The weight accounts for competing-event subjects who would
    have been censored by `times[i]` had they not had their event.

    Returns
    -------
        at_risk: Boolean tensor (batch_size, batch_size)
        log_weights: Tensor (batch_size, batch_size); 0 for subjects still at risk
    """
    log_cens_surv = torch.log(
        torch.clamp(_censoring_survival_before(times, events), min=1e-12)
    )
    times_i = times.unsqueeze(1)  # (batch, 1)
    times_j = times.unsqueeze(0)  # (1, batch)
    still_at_risk = times_j >= times_i
    is_competing_for_k = (events != 0) & (events != k)  # (batch,)
    retained = is_competing_for_k.unsqueeze(0) & (times_j < times_i)
    log_weights = torch.where(
        retained,
        log_cens_surv.unsqueeze(1) - log_cens_surv.unsqueeze(0),
        torch.zeros_like(retained, dtype=log_cens_surv.dtype),
    )
    return still_at_risk | retained, log_weights


def weighted_negative_log_likelihood_loss(
    risk_scores,
    times,
    events,
    num_competing_risks,
    event_weights=None,
    sample_weights=None,
    eps=1e-8,
) -> torch.Tensor:
    """
    Compute the weighted negative log-likelihood loss for competing risks Cox model.

    Vectorized: for each risk, builds a single (batch_size, batch_size)
    risk-set matrix (sample j is at risk for sample i's event iff
    times[j] >= times[i]) and reduces it with one batched `logsumexp`,
    instead of looping over the batch in Python and calling `logsumexp`
    once per event.

    Parameters
    ----------
        risk_scores: List of tensors with shape (batch_size, 1) for each competing risk
        times: Event/censoring times (batch_size,)
        events: Event indicators (0=censored, 1...K=event types) (batch_size,)
        num_competing_risks: Number of competing risks
        event_weights: Tensor of weights for each competing risk type
        (size: num_competing_risks)
        sample_weights: Tensor of weights for each sample (size: batch_size)
        eps: Small constant for numerical stability (unused, kept for API parity)

    Returns
    -------
        Weighted negative log partial likelihood loss
    """
    device = times.device
    batch_size = times.shape[0]

    # Initialize loss
    loss = torch.tensor(0.0, device=device)

    # Set default weights if not provided
    if event_weights is None:
        event_weights = torch.ones(num_competing_risks, device=device)
    if sample_weights is None:
        sample_weights = torch.ones(batch_size, device=device)

    # Count number of events
    n_events = (events > 0).sum().item()
    if n_events == 0:
        return loss

    # Risk set: sample j is at risk for the event at sample i iff times[j] >= times[i].
    at_risk = times.unsqueeze(1) <= times.unsqueeze(0)  # at_risk[i, j]

    # Process each competing risk separately
    for k in range(1, num_competing_risks + 1):
        # Find samples with this event type
        event_mask = events == k
        if event_mask.sum().item() == 0:
            continue

        # Get risk scores for this competing risk
        risk_k = risk_scores[k - 1].squeeze(-1)

        # Get weight for this event type
        event_weight = event_weights[k - 1]

        # Calculate log sum of exp of risk scores in each event's risk set
        log_risk_sum = _masked_log_risk_sums(risk_k, at_risk)

        # Subtract individual risk score from log sum and apply weights
        individual_loss = log_risk_sum - risk_k
        weighted_individual_loss = individual_loss * event_weight * sample_weights
        loss = loss + weighted_individual_loss[event_mask].sum()

    # Return average loss
    return loss / max(n_events, 1)


def negative_log_likelihood_loss(
    risk_scores: float,
    times: torch.Tensor,
    events: torch.Tensor,
    num_competing_risks: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the negative log-likelihood loss for competing risks Cox model.

    Vectorized: for each risk, builds a single (batch_size, batch_size)
    risk-set matrix (sample j is at risk for sample i's event iff
    times[j] >= times[i]) and reduces it with one batched `logsumexp`,
    instead of looping over the batch in Python and calling `logsumexp`
    once per event.

    Parameters
    ----------
        risk_scores: List of tensors with shape (batch_size, 1) for each competing risk
        times: Event/censoring times (batch_size,)
        events: Event indicators (0=censored, 1...K=event types) (batch_size,)
        num_competing_risks: Number of competing risks
        eps: Small constant for numerical stability (unused, kept for API parity)

    Returns
    -------
        Negative log partial likelihood loss
    """
    device = times.device

    # Initialize loss
    loss = torch.tensor(0.0, device=device)

    # Count number of events
    n_events = (events > 0).sum().item()
    if n_events == 0:
        return loss

    # Risk set: sample j is at risk for the event at sample i iff times[j] >= times[i].
    at_risk = times.unsqueeze(1) <= times.unsqueeze(0)  # at_risk[i, j]

    # Process each competing risk separately
    for k in range(1, num_competing_risks + 1):
        # Find samples with this event type
        event_mask = events == k
        if event_mask.sum().item() == 0:
            continue

        # Get risk scores for this competing risk
        risk_k = risk_scores[k - 1].squeeze(-1)

        # Calculate log sum of exp of risk scores in each event's risk set
        log_risk_sum = _masked_log_risk_sums(risk_k, at_risk)

        # Subtract individual risk score from log sum
        individual_loss = log_risk_sum - risk_k
        loss = loss + individual_loss[event_mask].sum()

    # Return average loss
    return loss / max(n_events, 1)


def compute_l2_penalty(
    model: torch.nn.Module,
    include_bias: bool = False
    ) -> torch.Tensor:
    """
    Compute L2 regularization penalty on model parameters.

    Parameters
    ----------
        model: Neural network model
        include_bias: Whether to include bias terms in regularization

    Returns
    -------
        L2 penalty term
    """
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Skip bias parameters if specified
            if not include_bias and "bias" in name:
                continue
            l2_reg += torch.sum(param**2)
    return l2_reg


def fine_gray_negative_log_likelihood(
    risk_scores,
    times,
    events,
    num_competing_risks,
    eps=1e-8,
) -> torch.Tensor:
    """
    Compute the negative log-likelihood loss for the Fine-Gray subdistribution model.

    Unlike `negative_log_likelihood_loss`, which models the cause-specific hazard
    and removes a subject from cause k's risk set as soon as any event occurs
    (competing or otherwise), this function models the subdistribution hazard:
    subjects who experience a competing event (event type j != k, j != 0) are
    NOT removed from cause k's risk set. Instead they remain at risk for cause k
    after their competing event time, down-weighted by the probability of
    having remained uncensored, `G(t_i-) / G(t_j-)`, per Fine & Gray (1999),
    with G the batch Kaplan-Meier censoring survival function. This
    retained-risk-set construction is what distinguishes the subdistribution
    hazard from the cause-specific hazard.

    Vectorized: for each risk, builds a single (batch_size, batch_size)
    risk-set matrix (the standard `times[j] >= times[i]` condition, extended
    with the competing-event retention condition above) and reduces it with
    one batched `logsumexp`, instead of looping over the batch (and, for the
    retention condition, over the other risk types) in Python. A sample j is
    "already-competing" for cause k if `events[j]` is neither censored (0)
    nor cause k itself, which is equivalent to looping over the other cause
    types but avoids the inner Python loop entirely.

    Parameters
    ----------
        risk_scores: List of tensors with shape (batch_size, 1) for each competing risk
        times: Event/censoring times (batch_size,)
        events: Event indicators (0=censored, 1...K=event types) (batch_size,)
        num_competing_risks: Number of competing risks
        eps: Small constant for numerical stability (unused, kept for API parity)

    Returns
    -------
        Negative log subdistribution partial likelihood loss
    """
    device = times.device

    # Initialize loss
    loss = torch.tensor(0.0, device=device)

    # Count number of events
    n_events = (events > 0).sum().item()
    if n_events == 0:
        return loss

    # Process each competing risk separately
    for k in range(1, num_competing_risks + 1):
        # Find samples with this event type
        event_mask = events == k
        if event_mask.sum().item() == 0:
            continue

        # Fine-Gray extension: subjects who already had a competing event
        # (event type j != k, j != 0) remain in the subdistribution risk set
        # for cause k, with IPCW weights.
        at_risk, log_weights = _fine_gray_risk_set(times, events, k)

        # Get risk scores for this competing risk
        risk_k = risk_scores[k - 1].squeeze(-1)

        # Calculate log sum of exp of risk scores in each event's risk set
        log_risk_sum = _masked_log_risk_sums(risk_k, at_risk, log_weights)
        individual_loss = log_risk_sum - risk_k
        loss = loss + individual_loss[event_mask].sum()

    # Return average loss
    return loss / max(n_events, 1)


def weighted_fine_gray_negative_log_likelihood(
    risk_scores,
    times,
    events,
    num_competing_risks,
    event_weights=None,
    sample_weights=None,
    eps=1e-8,
) -> torch.Tensor:
    """
    Compute the weighted negative log-likelihood loss for the Fine-Gray subdistribution model.

    Unlike `weighted_negative_log_likelihood_loss`, which models the cause-specific
    hazard and removes a subject from cause k's risk set as soon as any event
    occurs (competing or otherwise), this function models the subdistribution
    hazard: subjects who experience a competing event (event type j != k, j != 0)
    are NOT removed from cause k's risk set. Instead they remain at risk for
    cause k after their competing event time with IPCW weight
    `G(t_i-) / G(t_j-)`, per Fine & Gray (1999).
    This retained-risk-set construction is what distinguishes the subdistribution
    hazard from the cause-specific hazard.

    Vectorized: see `fine_gray_negative_log_likelihood` for the risk-set
    construction, applied here with per-event and per-sample weighting.

    Parameters
    ----------
        risk_scores: List of tensors with shape (batch_size, 1) for each competing risk
        times: Event/censoring times (batch_size,)
        events: Event indicators (0=censored, 1...K=event types) (batch_size,)
        num_competing_risks: Number of competing risks
        event_weights: Tensor of weights for each competing risk type
        (size: num_competing_risks)
        sample_weights: Tensor of weights for each sample (size: batch_size)
        eps: Small constant for numerical stability (unused, kept for API parity)

    Returns
    -------
        Weighted negative log subdistribution partial likelihood loss
    """
    device = times.device
    batch_size = times.shape[0]

    # Initialize loss
    loss = torch.tensor(0.0, device=device)

    # Set default weights if not provided
    if event_weights is None:
        event_weights = torch.ones(num_competing_risks, device=device)
    if sample_weights is None:
        sample_weights = torch.ones(batch_size, device=device)

    # Count number of events
    n_events = (events > 0).sum().item()
    if n_events == 0:
        return loss

    # Process each competing risk separately
    for k in range(1, num_competing_risks + 1):
        # Find samples with this event type
        event_mask = events == k
        if event_mask.sum().item() == 0:
            continue

        # Fine-Gray extension: subjects who already had a competing event
        # (event type j != k, j != 0) remain in the subdistribution risk set
        # for cause k, with IPCW weights.
        at_risk, log_weights = _fine_gray_risk_set(times, events, k)

        # Get risk scores for this competing risk
        risk_k = risk_scores[k - 1].squeeze(-1)

        # Get weight for this event type
        event_weight = event_weights[k - 1]

        # Calculate log sum of exp of risk scores in each event's risk set
        log_risk_sum = _masked_log_risk_sums(risk_k, at_risk, log_weights)

        # Subtract individual risk score from log sum and apply weights
        individual_loss = log_risk_sum - risk_k
        weighted_individual_loss = individual_loss * event_weight * sample_weights
        loss = loss + weighted_individual_loss[event_mask].sum()

    # Return average loss
    return loss / max(n_events, 1)
