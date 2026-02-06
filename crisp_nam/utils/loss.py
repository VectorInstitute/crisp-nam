import torch


def weighted_negative_log_likelihood_loss(
    risk_scores,
    times,
    events,
    num_competing_risks,
    event_weights=None,
    sample_weights=None,
    eps=1e-8,
) -> float:
    """
    Computes the weighted negative log-likelihood loss for competing risks Cox model.

    Args:
        risk_scores: List of tensors with shape (batch_size, 1) for each competing risk
        times: Event/censoring times (batch_size,)
        events: Event indicators (0=censored, 1...K=event types) (batch_size,)
        num_competing_risks: Number of competing risks
        event_weights: Tensor of weights for each competing risk type (size: num_competing_risks)
        sample_weights: Tensor of weights for each sample (size: batch_size)
        eps: Small constant for numerical stability

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

    # Process each competing risk separately
    for k in range(1, num_competing_risks + 1):
        # Find samples with this event type
        event_mask = events == k
        n_events_k = event_mask.sum().item()

        if n_events_k == 0:
            continue

        # Get risk scores for this competing risk
        risk_k = risk_scores[k - 1].squeeze()

        # Get weight for this event type
        event_weight = event_weights[k - 1]

        # For each event of type k
        for i in range(batch_size):
            if event_mask[i]:
                # Find samples in risk set (samples with time >= event time)
                risk_set = times >= times[i]

                # Calculate log sum of exp of risk scores in risk set
                risk_set_scores = risk_k[risk_set]
                log_risk_sum = torch.logsumexp(risk_set_scores, dim=0)

                # Subtract individual risk score from log sum and apply weights
                individual_loss = log_risk_sum - risk_k[i]
                weighted_individual_loss = (
                    individual_loss * event_weight * sample_weights[i]
                )
                loss += weighted_individual_loss

    # Return average loss
    return loss / max(n_events, 1)


def negative_log_likelihood_loss(
    risk_scores, times, events, num_competing_risks, eps=1e-8
):
    """
    Computes the negative log-likelihood loss for competing risks Cox model.

    Args:
        risk_scores: List of tensors with shape (batch_size, 1) for each competing risk
        times: Event/censoring times (batch_size,)
        events: Event indicators (0=censored, 1...K=event types) (batch_size,)
        num_competing_risks: Number of competing risks
        eps: Small constant for numerical stability

    Returns
    -------
        Negative log partial likelihood loss
    """
    device = times.device
    batch_size = times.shape[0]

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
        n_events_k = event_mask.sum().item()

        if n_events_k == 0:
            continue

        # Get risk scores for this competing risk
        risk_k = risk_scores[k - 1].squeeze()

        # For each event of type k
        for i in range(batch_size):
            if event_mask[i]:
                # Find samples in risk set (samples with time >= event time)
                risk_set = times >= times[i]

                # Calculate log sum of exp of risk scores in risk set
                risk_set_scores = risk_k[risk_set]
                log_risk_sum = torch.logsumexp(risk_set_scores, dim=0)

                # Subtract individual risk score from log sum
                loss += log_risk_sum - risk_k[i]

    # Return average loss
    return loss / max(n_events, 1)


def compute_l2_penalty(model, include_bias=False) -> int:
    """
    Compute L2 regularization penalty on model parameters

    Args:
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
