import torch
import numpy as np
from typing import List, Any

def compute_baseline_cif(times:np.ndarray,
                         events:np.ndarray,
                         eval_times:List[Any],
                         event_type:np.ndarray) -> np.ndarray:
    """
    Compute baseline cumulative incidence function for a specific event type
    
    Args:
        times: Numpy array of event times
        events: Numpy array of event indicators (0=censored, 1...K=event types)
        eval_times: Times at which to evaluate the CIF
        event_type: Event type to compute CIF for (1...K)
        
    Returns:
        Numpy array of baseline CIF values at eval_times
    """
    # Sort times and corresponding events
    sort_idx = np.argsort(times)
    sorted_times = times[sort_idx]
    sorted_events = events[sort_idx]
    
    # Initialize cumulative hazard
    n_samples = len(times)
    baseline_cif = np.zeros(len(eval_times))
    
    # For each evaluation time
    for i, t in enumerate(eval_times):
        cif_t = 0.0
        # Count number of events of the specified type before time t
        event_count = np.sum((sorted_events == event_type) & (sorted_times <= t))
        if event_count > 0:
            # Simple Aalen-Johansen estimator
            cif_t = event_count / n_samples
        baseline_cif[i] = cif_t
        
    return baseline_cif


def predict_cif(model:torch.Module,
                x:np.ndarray,
                baseline_cif:np.ndarray,
                times:np.ndarray,
                event_of_interest:int) -> np.ndarray:
    """
    Predict cumulative incidence function for a specific competing risk.

    Args:
        model: Trained  model.
        x: Input tensor of shape (n_samples, n_features).
        baseline_cif: Array of shape (len(times),) — estimated CIF for baseline (e.g. from compute_baseline_cif).
        times: Time points at which CIF is evaluated.
        event_type: Integer, 0-based index of event of interest.

    Returns:
        cif_pred: Array of shape (n_samples, len(times)) — predicted CIF per sample.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)  # list of length num_risks
        f_j_x = logits[event_of_interest].squeeze(1).cpu().numpy()  # (n_samples,)

    baseline_cif = np.asarray(baseline_cif).reshape(1, -1)  # (1, T)
    risk_scores = np.exp(f_j_x).reshape(-1, 1)               # (N, 1)
    
    # Fine-Gray style CIF prediction under PH assumption
    cif_pred = 1.0 - np.power(1.0 - baseline_cif, risk_scores)  # shape (N, T)
    
    return cif_pred

def predict_risk(model:np.ndarray,
                 x_input:np.ndarray,
                 device:str = 'cpu'):
    """
    Predicts relative risk scores for each competing risk.

    Args:
        model : Trained model.
        x_input (np.ndarray or torch.Tensor): Input features of shape (n_samples, n_features).
        device (str): Device to run the computation on.

    Returns:
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

def predict_absolute_risk(model:torch.Tensor,
                          x_input:np.ndarray,
                          baseline_cifs:List[Any],
                          times:List[Any],
                          device:str = 'cpu') -> np.ndarray:
    """
    Predict absolute risk (CIF) for each competing event by given time points.

    Args:
        model: Trained  model.
        x_input (np.ndarray or Tensor): Input features, shape (n_samples, n_features).
        baseline_cifs (dict): Mapping of event index to baseline CIF array of shape (n_times,).
        times (np.ndarray): Time grid used for baseline_cifs.
        device: CPU or CUDA.

    Returns:
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
