import configargparse
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tabulate import tabulate
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

from crisp_nam.models import CrispNamModel
from crisp_nam.utils import (
    weighted_negative_log_likelihood_loss,
    negative_log_likelihood_loss,
    compute_l2_penalty
)
from data_utils import *
from model_utils import EarlyStopping, set_seed
from crisp_nam.metrics import brier_score, auc_td
from crisp_nam.utils import predict_absolute_risk, compute_baseline_cif
from crisp_nam.utils import plot_coxnam_shape_functions, plot_feature_importance

def parse_args():
    parser = configargparse.ArgumentParser(
        description="Training script for MultiTaskCoxNAM model",
        default_config_files=["config.yml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    
    parser.add_argument("-c", "--config", is_config_file=True,
                      help="Path to config file")

    parser.add_argument("--dataset", type=str, default="framingham", 
                      help="Dataset to use: (framingham, support, pbc, synthetic)")

    parser.add_argument("--scaling", type=str, default="standard", choices=["minmax", "standard", "none"],
                      help="Data scaling method for continuous features")

    parser.add_argument("--num_epochs", type=int, default=500, 
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, 
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                      help="Learning rate for optimizer")
    parser.add_argument("--l2_reg", type=float, default=1e-3, 
                      help="L2 regularization weight")
    parser.add_argument("--patience", type=int, default=10, 
                      help="Patience for early stopping")

    parser.add_argument("--dropout_rate", type=float, default=0.5, 
                      help="Dropout rate for model")
    parser.add_argument("--feature_dropout", type=float, default=0.1, 
                      help="Feature dropout rate")
    parser.add_argument("--hidden_dimensions", type=str, default="64,64", 
                      help="Hidden layer dimensions (comma-separated)")
    parser.add_argument("--batch_norm", type=str, default="False", choices=["True", "False"],
                      help="Whether to use batch normalization")

    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    parser.add_argument("--n_folds", type=int, default=5, 
                      help="Number of folds for cross-validation")
    
    return parser.parse_args()




def train_model(model, train_loader, val_loader=None, num_epochs=500, learning_rate=1e-3, 
                l2_reg=0.01, patience=10, event_weights=None, verbose=True):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(patience=patience)
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, t, e, _ in train_loader:
            x, t, e = x.to(device), t.to(device), e.to(device)
            risk_scores, _ = model(x)

            # Use weighted loss if event_weights is provided
            if event_weights is not None:
                loss = weighted_negative_log_likelihood_loss(risk_scores, t, e, 
                                                           model.num_competing_risks,
                                                           event_weights=event_weights)
            else:
                loss = negative_log_likelihood_loss(risk_scores, t, e, model.num_competing_risks)
                
            reg = compute_l2_penalty(model) * l2_reg
            total = loss + reg

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            total_loss += total.item()

        avg_train_loss = total_loss / len(train_loader)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, t, e, _ in val_loader:
                    x, t, e = x.to(device), t.to(device), e.to(device)
                    risk_scores, _ = model(x)
                    
                    # Use same loss function as in training
                    if event_weights is not None:
                        loss = weighted_negative_log_likelihood_loss(risk_scores, t, e, 
                                                                   model.num_competing_risks,
                                                                   event_weights=event_weights)
                    else:
                        loss = negative_log_likelihood_loss(risk_scores, t, e, model.num_competing_risks)
                        
                    reg = compute_l2_penalty(model) * l2_reg
                    val_loss += (loss + reg).item()
            avg_val_loss = val_loss / len(val_loader)
            early_stopper.step(avg_val_loss)

            if verbose:
                print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if early_stopper.should_stop:
                if verbose:
                    print("Early stopping triggered.")
                break
        elif verbose:
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")


def evaluate_model(model, x_val, t_val, e_val, t_train, e_train, abs_risks, times):
    """
    Evaluate the model using time-dependent AUC, Brier score, and td-C-index.

    Args:
        model: Trained model (not directly used here).
        x_val: Validation input features.
        t_val: Validation times.
        e_val: Validation event indicators.
        t_train: Training times.
        e_train: Training event indicators.
        abs_risks: Array of shape (n_samples, n_events, n_times) with predicted absolute risks.
        times: List of time points at which to evaluate.

    Returns:
        metrics: dict of evaluation metrics by event and time.
    """

    survival_train = Surv.from_arrays(e_train != 0, t_train)
    survival_val = Surv.from_arrays(e_val != 0, t_val)

    metrics = defaultdict(list)
    n_events = abs_risks.shape[1]

    for k in range(n_events):
        for i, time in enumerate(times):
            risk_preds = abs_risks[:, k, i]
            
        
            try:
                risk_preds_2d = np.zeros((len(risk_preds), len(times)))
                risk_preds_2d[:, i] = risk_preds 
                
                auc_score, _ = auc_td(
                    e_val,
                    t_val,
                    risk_preds_2d,
                    times,
                    time,
                    km=(e_train, t_train),
                    primary_risk=k+1
                )
                metrics[f"auc_event{k+1}_t{time:.2f}"].append(float(auc_score))
            except Exception as ex:
                print(f"[Warning] AUC failed at t={time:.2f}, event={k+1}: {ex}")
                metrics[f"auc_event{k+1}_t{time:.2f}"].append(np.nan)

            
            try:
                
                brier_score_val, _ = brier_score(
                    e_val,
                    t_val,
                    risk_preds_2d,
                    times,
                    time,
                    km=(e_train, t_train),
                    primary_risk=k+1
                )
                metrics[f"brier_event{k+1}_t{time:.2f}"].append(float(brier_score_val))
            except Exception as ex:
                print(f"[Warning] Brier failed at t={time:.2f}, event={k+1}: {ex}")
                print(f"Debug: surv_probs shape={risk_preds_2d.shape}")
                metrics[f"brier_event{k+1}_t{time:.2f}"].append(np.nan)

           
            try:
                tdci_result = concordance_index_ipcw(
                    survival_train,
                    survival_val,
                    estimate=risk_preds,
                    tau=time
                )
                if isinstance(tdci_result, tuple):
                    tdci_score = tdci_result[0]
                else:
                    tdci_score = tdci_result  
                metrics[f"tdci_event{k+1}_t{time:.2f}"].append(float(tdci_score))
            except Exception as ex:
                print(f"[Warning] td-CI failed at t={time:.2f}, event={k+1}: {ex}")
                metrics[f"tdci_event{k+1}_t{time:.2f}"].append(np.nan)

    return metrics


def display_metrics_table(metrics_dict, n_folds=5, quantiles=[0.25, 0.5, 0.75]):
    """
    Display evaluation metrics summarized across folds for different time quantiles
    """
  
    
    time_points_by_metric = {}
    event_types = set()
    metric_types = ['auc', 'tdci', 'brier']
    
    for key, values in metrics_dict.items():
        if any(metric in key for metric in metric_types):
            parts = key.split('_')
            metric_type = parts[0]
            event_info = parts[1]
            
            # Extract event type
            event_type = int(event_info.replace('event', ''))
            event_types.add(event_type)
            
            # Extract time point
            if len(parts) > 2 and parts[2].startswith('t'):
                time_point = float(parts[2].replace('t', ''))
                
                # Initialize nested dictionaries if needed
                if (event_type, metric_type) not in time_points_by_metric:
                    time_points_by_metric[(event_type, metric_type)] = {}
                
                # Store metrics by time point
                time_points_by_metric[(event_type, metric_type)][time_point] = values
    
    results = []
    
    for event_type in sorted(event_types):
        row = {'Risk': f"Type {event_type}"}
        
        for metric_type in metric_types:
            if (event_type, metric_type) not in time_points_by_metric:
                # Skip metrics that don't exist for this event type
                for q in quantiles:
                    row[f"{metric_type.upper()}_q{q:.2f}"] = "N/A"
                continue
            
            # Get all time points for this event/metric
            time_data = time_points_by_metric[(event_type, metric_type)]
            sorted_times = sorted(time_data.keys())
            
            # For each quantile
            for q in quantiles:
                # Calculate the index for this quantile
                q_idx = max(0, min(len(sorted_times) - 1, int(len(sorted_times) * q)))
                
                # Get the corresponding time point for this quantile
                q_time = sorted_times[q_idx]
                
                # Get the metrics for this time point
                q_values = time_data[q_time]
                
                # Calculate and format statistics
                if q_values:
                    value_array = np.array(q_values)
                    mean_val = np.nanmean(value_array)
                    std_val = np.nanstd(value_array)
                    row[f"{metric_type.upper()}_q{q:.2f}"] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    row[f"{metric_type.upper()}_q{q:.2f}"] = "N/A"
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Define column order
    columns = ['Risk']
    for metric in ['AUC', 'TDCI', 'BRIER']:
        for q in quantiles:
            columns.append(f"{metric}_q{q:.2f}")
    
    # Select columns in the right order (only those that exist)
    df = df[[col for col in columns if col in df.columns]]
    
    print("\nSummary Performance Metrics:")
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\nInterpretation:")
    print("- AUC: 0.5=random, >0.7=good, >0.8=excellent")
    print("- TDCI (Time-Dependent C-Index): 0.5=random, >0.7=good, >0.8=excellent") 
    print("- Brier Score: 0=perfect, <0.25=good, >0.25=poor")


def main():
    def parse_args():
        parser = configargparse.ArgumentParser(
            description="Training script for MultiTaskCoxNAM model",
            default_config_files=["config.yaml"],
            config_file_parser_class=configargparse.YAMLConfigFileParser
        )
        
        # Config file option
        parser.add_argument("-c", "--config", is_config_file=True, 
                        help="Path to config file")
        
        # Dataset
        parser.add_argument("--dataset", type=str, default="framingham", choices=["framingham", "support", "pbc", "synthetic"],
                        help="Dataset to use")
        
        # Data scaling
        parser.add_argument("--scaling", type=str, default="standard", choices=["minmax", "standard", "none"],
                        help="Data scaling method for continuous features")
        
        # Training parameters
        parser.add_argument("--num_epochs", type=int, default=500, 
                        help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size for training")
        parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate for optimizer")
        parser.add_argument("--l2_reg", type=float, default=1e-3, 
                        help="L2 regularization weight")
        parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping")
        
        
        parser.add_argument("--dropout_rate", type=float, default=0.5, 
                        help="Dropout rate for model")
        parser.add_argument("--feature_dropout", type=float, default=0.1, 
                        help="Feature dropout rate")
        parser.add_argument("--hidden_dimensions", type=str, default="64,64", 
                        help="Hidden layer dimensions (comma-separated)")
        parser.add_argument("--batch_norm", type=str, default="False", choices=["True", "False"],
                        help="Whether to use batch normalization")
        
        
        parser.add_argument("--event_weighting", type=str, default="none", 
                        choices=["none", "balanced", "custom"],
                        help="Event weighting strategy (none, balanced, custom)")
        parser.add_argument("--custom_event_weights", type=str, default=None,
                        help="Custom weights for events (comma-separated, e.g., '1.0,2.0')")
        
        #
        parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
        parser.add_argument("--n_folds", type=int, default=5, 
                        help="Number of folds for cross-validation")
    
        return parser.parse_args()
    
    args = parse_args()
    print(args)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load the dataset
    if args.dataset.lower() == "framingham":
        x, t, e, feature_names, n_cont, _ = load_framingham()
    elif args.dataset.lower() == "support":
        x, t, e, feature_names, n_cont, _ = load_support_dataset()
    elif args.dataset.lower() == "pbc":
        x, t, e, feature_names, n_cont, _ = load_pbc2_dataset()
    elif args.dataset.lower() == "synthetic":
        x, t, e, feature_names, n_cont, _ = load_synthetic_dataset()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    
    # Note: Scaling will be done inside cross-validation loop to prevent data leakage

    # Compute weights based on event distribution
    num_competing_risks = len(np.unique(e)) - 1  # Excluding censoring (0)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    
    event_weights = None
    
    if args.event_weighting != "none":
        if args.event_weighting == "balanced":
            # Compute balanced weights (inverse of class frequencies)
            event_counts = np.zeros(num_competing_risks)
            for k in range(1, num_competing_risks + 1):
                event_counts[k-1] = np.sum(e == k)
            

            event_counts = np.maximum(event_counts, 1)
            
            # Inverse frequency weighting
            event_weights = 1.0 / event_counts
            
            # Normalize weights to sum to num_competing_risks
            event_weights = event_weights * (num_competing_risks / event_weights.sum())
            
            print(f"Computed balanced event weights: {event_weights}")
        
        elif args.event_weighting == "custom":
            if args.custom_event_weights is None:
                raise ValueError("Custom event weights must be provided when using custom weighting")
            
            custom_weights = [float(w) for w in args.custom_event_weights.split(",")]
            if len(custom_weights) != num_competing_risks:
                raise ValueError(f"Expected {num_competing_risks} weights, got {len(custom_weights)}")
            
            event_weights = np.array(custom_weights)
            print(f"Using custom event weights: {event_weights}")
        
        event_weights = torch.tensor(event_weights, dtype=torch.float32, device=device)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    all_metrics = defaultdict(list)
    quantiles = [0.25, 0.5, 0.75]
    

    safe_max = 0.99 * np.max(t)
    eval_times = np.quantile(t[t <= safe_max], quantiles)
    
    print(f"Evaluation times: {eval_times}")
    
    # Parse hidden dimensions string to list of integers
    hidden_dimensions = [int(dim) for dim in args.hidden_dimensions.split(",")]
    
    # Convert batch_norm string to boolean
    batch_norm = args.batch_norm.lower() == "true"

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, e)):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")

        # Split data for this fold
        x_train, x_val = x[train_idx].copy(), x[val_idx].copy()
        t_train, t_val = t[train_idx], t[val_idx]
        e_train, e_val = e[train_idx], e[val_idx]

        # Apply scaling within this fold to prevent data leakage
        if args.scaling.lower() == "standard":
            scaler = StandardScaler()
            x_train[:, -n_cont:] = scaler.fit_transform(x_train[:, -n_cont:])
            x_val[:, -n_cont:] = scaler.transform(x_val[:, -n_cont:])
        elif args.scaling.lower() == "minmax":
            scaler = MinMaxScaler()
            x_train[:, -n_cont:] = scaler.fit_transform(x_train[:, -n_cont:])
            x_val[:, -n_cont:] = scaler.transform(x_val[:, -n_cont:])
        elif args.scaling.lower() == "none":
            pass
        else:
            raise ValueError(f"Scaling method {args.scaling} not supported")

        # Calculate fold-specific weights if using balanced weighting
        fold_event_weights = event_weights
        if args.event_weighting == "balanced":
            # Recalculate weights based on training set for this fold
            fold_event_counts = np.zeros(num_competing_risks)
            for k in range(1, num_competing_risks + 1):
                fold_event_counts[k-1] = np.sum(e_train == k)
            
            # Avoid division by zero
            fold_event_counts = np.maximum(fold_event_counts, 1)
            
            # Inverse frequency weighting
            fold_event_weights = 1.0 / fold_event_counts
            
            # Normalize weights to sum to num_competing_risks
            fold_event_weights = fold_event_weights * (num_competing_risks / fold_event_weights.sum())
            fold_event_weights = torch.tensor(fold_event_weights, dtype=torch.float32, device=device)
            
            print(f"Fold {fold+1} event weights: {fold_event_weights.cpu().numpy()}")

        # Create datasets with properly normalized data
        train_dataset = SurvivalDataset(x_train, t_train, e_train)
        val_dataset = SurvivalDataset(x_val, t_val, e_val)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = CrispNamModel(
            num_features=x.shape[1],
            num_competing_risks=num_competing_risks,
            hidden_sizes=hidden_dimensions,
            dropout_rate=args.dropout_rate,
            feature_dropout=args.feature_dropout,
            batch_norm=batch_norm
        ).to(device)

       
        train_model(model, train_loader, val_loader, 
                    num_epochs=args.num_epochs, 
                    learning_rate=args.learning_rate,
                    l2_reg=args.l2_reg, 
                    patience=args.patience,
                    event_weights=fold_event_weights,
                    verbose=True)
        
        # Calculate baseline CIFs using the same eval times for all folds
        baseline_cifs = {k: compute_baseline_cif(t_train, e_train, eval_times, k + 1) 
                        for k in range(num_competing_risks)}

        abs_risks = predict_absolute_risk(model, x_val, baseline_cifs, eval_times, device=device)
        
        
        fold_metrics = evaluate_model(model, x_val, t_val, e_val,
                                      t_train, e_train, abs_risks, eval_times)
        
        for k, v in fold_metrics.items():
            all_metrics[k].extend(v)

    
    display_metrics_table(all_metrics, n_folds=args.n_folds)
    
    #create figs subdirectory if not present
    import os
    if not os.path.exists("figs"):
        os.makedirs("figs")
        

    for risk in range(1, num_competing_risks + 1):
    
        fig, _, top_positive, top_negative = plot_feature_importance(
            model=model,
            x_data=x,
            feature_names=feature_names,
            n_top=5,  # Show top 5 positive contributors
            n_bottom=5,  # Show top 5 negative contributors
            risk_idx=risk,  
            figsize=(6, 4),
            output_file=f"figs/feature_importance_risk_new_{risk}_{args.dataset}.png"
            )
    
        top_features = top_positive + top_negative
        
        fig, axes = plot_coxnam_shape_functions(
            model=model,
            X=x,  
            risk_to_plot=risk,
            feature_names=feature_names,  
            top_features=top_features,    
            ncols=5,
            figsize=(12,6),
            output_file=f"figs/shape_functions_top_features_risk{risk}_{args.dataset}.png"
        )
        plt.close(fig)


if __name__ == "__main__":
    main()