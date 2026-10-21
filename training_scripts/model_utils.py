import random

import numpy as np
import torch


DEVICE_CHOICES = ["auto", "cpu", "cuda", "mps"]
DEVICE_HELP = (
    "Device to train on. 'auto' uses CUDA if available, else CPU. 'mps' (Apple GPU) "
    "is supported but usually slower than CPU for these small per-feature networks."
)


def get_device(name="auto"):
    """Resolve a --device choice to a torch.device, failing loudly if unavailable."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps requested but MPS is not available")
    return torch.device(name)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True


# Utility functions to create masks for DeepHit
def create_fc_mask1(k, t, num_Event, num_Category, device=None):
    """Create mask1 for loss calculation - for uncensored loss"""
    N = len(k)
    mask = torch.zeros((N, num_Event, num_Category), device=device)

    for i in range(N):
        if k[i] > 0:  # Not censored
            event_idx = int(k[i] - 1)
            time_idx = int(t[i])
            if time_idx < num_Category:
                mask[i, event_idx, time_idx] = 1.0

    return mask


def create_fc_mask2(t, num_Category, device=None):
    """Create mask2 for loss calculation - for censored loss"""
    N = len(t)
    mask = torch.zeros((N, num_Category), device=device)

    for i in range(N):
        time_idx = int(t[i])
        for j in range(time_idx, num_Category):
            mask[i, j] = 1.0

    return mask


# Pre-create masks for DeepHit on GPU
def create_fc_mask1_gpu(e, t_disc, num_Event, num_Category, device):
    """
    Create first mask for DeepHit loss computation
    Optimized version that keeps operations on GPU
    """
    batch_size = e.size(0)
    mask1 = torch.zeros(batch_size, num_Event, num_Category, device=device)

    for i in range(batch_size):
        if e[i] > 0:  # if not censored
            event_idx = int(e[i].item()) - 1
            t_idx = int(t_disc[i].item())
            mask1[i, event_idx, t_idx] = 1

    return mask1


def create_fc_mask2_gpu(t_disc, num_Category, device):
    """
    Create second mask for DeepHit loss computation
    Optimized version that keeps operations on GPU
    """
    batch_size = t_disc.size(0)
    mask2 = torch.zeros(batch_size, num_Category, device=device)

    for i in range(batch_size):
        t_idx = int(t_disc[i].item())
        mask2[i, t_idx:] = 1

    return mask2
