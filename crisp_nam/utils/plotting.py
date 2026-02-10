"""Utility functions for plotting.

This module provides functions to visualize feature importance
and shape functions for both crisp-nam and deephit models.
"""

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_feature_importance(
    model: torch.nn.Module,
    x_data: Union[np.ndarray, torch.Tensor],
    feature_names=None,
    n_top: int = 5,
    n_bottom: int = 5,
    risk_idx: int = 1,
    figsize: tuple = (8, 6),
    output_file: str = None,
    color_positive: str = "#2196F3",
    color_negative: str = "#F44336",
) -> tuple:
    """Plot feature importance with both top positive and negative influences,
    handling both CPU and CUDA devices automatically.

    Args:
    - model: A trained CoxNAM model (torch.nn.Module)
    - x_data: Input data (numpy array or torch tensor) to compute contributions
    - feature_names: Optional list of feature names (default: generic names)
    - n_top: Number of top positive features to display
    - n_bottom: Number of top negative features to display
    - risk_idx: Index of the competing risk to analyze
    - figsize: Size of the plot (width, height)
    - output_file: Optional path to save the plot image (e.g., "feature_importance.png")
    - color_positive: Color for positive contributions (default: blue)
    - color_negative: Color for negative contributions (default: red)

    Returns
    -------
    - fig: Matplotlib figure object
    - ax: Matplotlib axes object
    - top_pos: List of top positive feature names
    - top_neg: List of top negative feature names
    """
    # determine model device
    device = next(model.parameters()).device
    model.eval()

    # prepare feature names
    num_features: int = model.num_features
    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(num_features)]

    # convert x_data to tensor on the model device
    if not isinstance(x_data, torch.Tensor):
        x = torch.tensor(x_data, dtype=torch.float32, device=device)
    else:
        x = x_data.to(device)

    feature_contribs = {}
    risk_idx0 = risk_idx - 1

    with torch.no_grad():
        for i in range(num_features):
            vals = x[:, i].unsqueeze(1)  # shape (N,1)
            if torch.var(vals) <= 1e-8:
                feature_contribs[feature_names[i]] = 0.0
                continue

            # forward through the feature net and projection
            rep = model.feature_nets[i](vals)
            proj = model.risk_projections[i][risk_idx0](rep)
            # mean contribution as a Python float
            contrib = proj.mean().item()
            feature_contribs[feature_names[i]] = contrib

    # build a DataFrame for sorting
    df = pd.DataFrame(
        {
            "feature": list(feature_contribs.keys()),
            "contribution": list(feature_contribs.values()),
        }
    )
    df["abs_contrib"] = df["contribution"].abs()
    df = df.sort_values("abs_contrib", ascending=False)

    pos = df[df["contribution"] > 0].head(n_top).sort_values("contribution")
    neg = (
        df[df["contribution"] < 0]
        .head(n_bottom)
        .sort_values("contribution", ascending=False)
    )

    top_pos = pos["feature"].tolist()
    top_neg = neg["feature"].tolist()

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(pos["feature"], pos["contribution"], color=color_positive, alpha=0.8)
    ax.barh(neg["feature"], neg["contribution"], color=color_negative, alpha=0.8)
    ax.axvline(0, color="black", linestyle="-", alpha=0.3)

    ax.set_xlabel("Contribution to Risk Score")
    ax.set_title(
        f"Top {n_top} Positive & {n_bottom} Negative Features for risk_{risk_idx}"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)

    return fig, ax, top_pos, top_neg


def plot_coxnam_shape_functions(
    model: torch.nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    risk_to_plot: int = 1,
    feature_names: List[str] = None,
    top_features: List[str] = None,
    ncols: int = 3,
    figsize: tuple = (12, 8),
    output_file: str = "",
) -> list[float]:
    """Plot shape functions for each feature in a CoxNAM model,
    automatically handling CPU vs CUDA inputs.

    Args:
    - model: A trained CoxNAM model (torch.nn.Module)
    - X: Input data (numpy array or torch tensor) to compute shape functions
    - risk_to_plot: Index of the competing risk to visualize
    - feature_names: Optional list of feature names (default: generic names)
    - top_features: Optional list of feature names to plot features)
    - ncols: Number of columns in the subplot grid
    - figsize: Size of the entire figure (width, height)
    - output_file: Optional path to save the plot image (e.g., "shape_functions.png")

    Returns
    -------
    - fig: Matplotlib figure object
    - axes: List of Matplotlib axes objects for each plotted feature
    """
    device = next(model.parameters()).device
    model.eval()
    risk_idx = risk_to_plot - 1

    # ensure X is a numpy array
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else np.array(X, dtype=float)

    # derive feature list
    num_features = model.num_features
    if feature_names:
        feature_names = [f"Feature {i + 1}" for i in range(num_features)]
    if top_features:
        # map names back to indices
        idx_map = {name: i for i, name in enumerate(feature_names)}
        selected = [(idx_map.get(name), name) for name in top_features]
        selected = [(i, name) for i, name in selected if i is not None]
    else:
        selected = list(zip(range(num_features), feature_names))

    n_selected = len(selected)
    nrows = int(np.ceil(n_selected / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize))
    axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for ax, (f_idx, fname) in zip(axes, selected):
            vals = X_np[:, f_idx]
            if vals.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue

            # choose evaluation points
            if np.issubdtype(vals.dtype, np.integer) or len(np.unique(vals)) <= 10:
                pts = np.unique(vals)
            else:
                pts = np.linspace(vals.min(), vals.max(), 100)

            # convert to tensor on correct device
            t_pts = torch.tensor(pts, dtype=torch.float32, device=device).unsqueeze(1)

            # compute shape values
            rep = model.feature_nets[f_idx](t_pts)
            proj = model.risk_projections[f_idx][risk_idx](rep)
            shp = proj.squeeze(-1).cpu().numpy()

            # plot
            ax.plot(pts, shp, linewidth=2)
            ax.axhline(0, linestyle="--", alpha=0.5)
            ax.set_title(fname)
            ax.set_xlabel("Value")
            ax.set_ylabel("Contribution")
            # rug plot
            ax.plot(vals, np.zeros_like(vals) - 0.1, "|", alpha=0.3)

    # turn off any extra axes
    for ax in axes[n_selected:]:
        ax.axis("off")

    fig.suptitle(f"Shape Functions for Risk {risk_to_plot}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig, axes[:n_selected]
