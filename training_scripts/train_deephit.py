import os
import configargparse
from collections import defaultdict

import torch
import numpy as np
import torch.optim as optim
from sksurv.util import Surv
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sksurv.metrics import concordance_index_ipcw

from data_utils import *
from model_utils import (
    set_seed,
    EarlyStopping,
    create_fc_mask1,
    create_fc_mask2
)
from crisp_nam.metrics import brier_score, auc_td
from crisp_nam.models import DeepHit

def parse_args():
    parser = configargparse.ArgumentParser(
        description="Training script for DeepHit model",
        default_config_files=["config.yml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    
    parser.add_argument("-c", "--config", is_config_file=True,
                      help="Path to config file")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="framingham", 
                      help="Dataset to use: (framingham, support, pbc, synthetic)")
    
    # Data scaling
    parser.add_argument("--scaling", type=str, default="standard", choices=["minmax", "standard", "none"],
                      help="Data scaling method for continuous features")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=250, 
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, 
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                      help="Learning rate for optimizer")
    parser.add_argument("--l2_reg", type=float, default=1e-5, 
                      help="L2 regularization weight")
    parser.add_argument("--patience", type=int, default=10, 
                      help="Patience for early stopping")
    
    # DeepHit specific parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                      help="Weight for log-likelihood loss")
    parser.add_argument("--beta", type=float, default=0.0,
                      help="Weight for ranking loss")
    parser.add_argument("--gamma", type=float, default=0.0,
                      help="Weight for calibration loss")
    parser.add_argument("--h_dim_shared", type=int, default=128,
                      help="Hidden dimension for shared network")
    parser.add_argument("--h_dim_CS", type=int, default=32,
                      help="Hidden dimension for cause-specific networks")
    parser.add_argument("--num_layers_shared", type=int, default=1,
                      help="Number of layers in shared network")
    parser.add_argument("--num_layers_CS", type=int, default=2,
                      help="Number of layers in cause-specific networks")
    parser.add_argument("--num_categories", type=int, default=100,
                      help="Number of time categories for discretization")
    parser.add_argument("--active_fn", type=str, default="tanh", choices=["relu", "elu", "tanh"],
                      help="Activation function")
    
    # General parameters
    parser.add_argument("--dropout_rate", type=float, default=0.3, 
                      help="Dropout rate for model")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    parser.add_argument("--n_folds", type=int, default=5, 
                      help="Number of folds for cross-validation")
    
    return parser.parse_args()

def train_deephit_model(model, train_loader, val_loader=None, alpha=1.0, beta=1.0, gamma=1.0,
                         num_epochs=500, learning_rate=1e-3, l2_reg=0.01, patience=10, verbose=True):
    """
    Train the DeepHit model using the three loss components
    """
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    early_stopper = EarlyStopping(patience=patience)
    
    # Get model dimensions for mask creation
    num_Event = model.num_Event
    num_Category = model.num_Category
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for x, t, e, t_disc in train_loader:
            x, t, e, t_disc = x.to(device), t.to(device), e.to(device), t_disc.to(device)
            
            # Create masks for loss computation
            mask1 = create_fc_mask1(e, t_disc, num_Event, num_Category, device)
            mask2 = create_fc_mask2(t_disc, num_Category, device)
            
            # Forward pass
            out, _ = model(x)
            
            # Compute loss components
            loss = model.compute_loss(out, t, e, mask1, mask2, alpha, beta, gamma)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation if provided
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, t, e, t_disc in val_loader:
                    x, t, e, t_disc = x.to(device), t.to(device), e.to(device), t_disc.to(device)
                    
                    # Create masks for validation
                    mask1 = create_fc_mask1(e, t_disc, num_Event, num_Category, device)
                    mask2 = create_fc_mask2(t_disc, num_Category, device)
                    
                    # Forward pass
                    out, _ = model(x)
                    
                    # Compute loss
                    loss = model.compute_loss(out, t, e, mask1, mask2, alpha, beta, gamma)
                    val_loss += loss.item()
                    
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


def predict_absolute_risk_deephit(model, x_test, times, t_max=None, device="cpu"):
    """
    Predict absolute risks using DeepHit model
    
    Args:
        model: Trained DeepHit model
        x_test: Test features
        times: Evaluation times (continuous)
        t_max: Maximum observed time in training data (for scaling)
        device: Device to perform prediction on
    
    Returns:
        abs_risks: Array of shape (n_samples, n_events, n_times) with CIF at each time
    """
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        preds, _ = model(x_test_tensor)  # shape: (batch_size, num_Event, num_Category)
        
    # Convert to numpy for further processing
    preds_np = preds.cpu().numpy()
    
    # Get model dimensions
    n_samples = preds_np.shape[0]
    n_events = preds_np.shape[1]
    n_categories = preds_np.shape[2]
    n_times = len(times)
    
    # Set t_max if not provided (should be the same max time used when training)
    if t_max is None:
        t_max = max(times)  # This should ideally be the max time from training
    
    # Initialize the output array
    abs_risks = np.zeros((n_samples, n_events, n_times))
    
    # For each evaluation time, find the corresponding bin or interpolate
    for t_idx, t in enumerate(times):
        # Scale time to [0, num_categories-1]
        scaled_t = (t / t_max) * (n_categories - 1)
        
        # Find lower and upper bin indices
        lower_bin = int(np.floor(scaled_t))
        upper_bin = min(int(np.ceil(scaled_t)), n_categories-1)
        
        # Handle boundary cases
        if lower_bin == upper_bin or upper_bin >= n_categories:
            # Exactly at a bin boundary or at/beyond the last bin
            for k in range(n_events):
                # Get cumulative probability up to this bin
                abs_risks[:, k, t_idx] = np.sum(preds_np[:, k, :lower_bin+1], axis=1)
        else:
            # Need to interpolate between bins
            weight_upper = scaled_t - lower_bin
            weight_lower = 1 - weight_upper
            
            for k in range(n_events):
                # Cumulative probability up to lower bin
                cum_prob_lower = np.sum(preds_np[:, k, :lower_bin+1], axis=1)
                
                # Cumulative probability up to upper bin
                cum_prob_upper = np.sum(preds_np[:, k, :upper_bin+1], axis=1)
                
                # Linear interpolation
                abs_risks[:, k, t_idx] = weight_lower * cum_prob_lower + weight_upper * cum_prob_upper
    
    return abs_risks

def evaluate_model(model, x_val, t_val, e_val, t_train, e_train, times, max_time, device="cpu"):
    """
    Evaluate the DeepHit model using time-dependent AUC, Brier score, and td-C-index
    """
    # Predict absolute risks for validation set
    abs_risks = predict_absolute_risk_deephit(model, x_val, times, t_max=max_time, device=device)
    
    # Format data for scikit-survival functions
    survival_train = Surv.from_arrays(e_train != 0, t_train)
    survival_val = Surv.from_arrays(e_val != 0, t_val)
    
    metrics = defaultdict(list)
    n_events = abs_risks.shape[1]
    
    for k in range(n_events):
        for i, time in enumerate(times):
            risk_preds = abs_risks[:, k, i]
            
            try:
                # Reshape to format needed by custom function (n_samples, n_times)
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
    import pandas as pd
    from tabulate import tabulate
    
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
    
    # Scale features if needed
    if args.scaling.lower() == "standard":
        scaler = StandardScaler()
        x[:, -n_cont:] = scaler.fit_transform(x[:, -n_cont:])
    elif args.scaling.lower() == "minmax":
        scaler = MinMaxScaler()
        x[:, -n_cont:] = scaler.fit_transform(x[:, -n_cont:])
    elif args.scaling.lower() == "none":
        pass
    else:
        raise ValueError(f"Scaling method {args.scaling} not supported")
    
    # Get number of competing risks
    num_competing_risks = len(np.unique(e)) - 1  # Excluding censoring (0)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare dataset with discretized times
    # Maximum time in the dataset
    max_time = np.max(t)
    
    # Create dataset with discretized times
    dataset = SurvivalDatasetDeepHit(x, t, e, args.num_categories)
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Metrics collection and evaluation times
    all_metrics = defaultdict(list)
    quantiles = [0.25, 0.5, 0.75]
    
    # Calculate global evaluation times
    safe_max = 0.99 * np.max(t)
    eval_times = np.quantile(t[t <= safe_max], quantiles)
    
    print(f"Evaluation times: {eval_times}")
    
    # Create directory for figures if it doesn't exist
    if not os.path.exists("figs"):
        os.makedirs("figs")
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(x, e)):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
        
        # Create data loaders
        train_loader = DataLoader(
            Subset(dataset, train_idx), 
            batch_size=args.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), 
            batch_size=args.batch_size
        )
        
        # Initialize DeepHit model
        input_dims = {
            'x_dim': x.shape[1],
            'num_Event': num_competing_risks,
            'num_Category': args.num_categories
        }
        
        network_settings = {
            'h_dim_shared': args.h_dim_shared,
            'h_dim_CS': args.h_dim_CS,
            'num_layers_shared': args.num_layers_shared,
            'num_layers_CS': args.num_layers_CS,
            'active_fn': args.active_fn,
            'keep_prob': 1.0 - args.dropout_rate
        }
        
        model = DeepHit(input_dims, network_settings).to(device)
        
        # Train the model
        train_deephit_model(
            model, 
            train_loader, 
            val_loader,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            l2_reg=args.l2_reg,
            patience=args.patience,
            verbose=True
        )
        
        # Evaluate model
        fold_metrics = evaluate_model(
            model, 
            x[val_idx], 
            t[val_idx], 
            e[val_idx],
            t[train_idx], 
            e[train_idx], 
            eval_times, 
            max_time,
            device
        )
        
        # Collect metrics
        for k, v in fold_metrics.items():
            all_metrics[k].extend(v)
        
        # Save model for this fold if needed
        # torch.save(model.state_dict(), f"models/deephit_fold{fold}_{args.dataset}.pt")
    
    # Display final metrics
    display_metrics_table(all_metrics, n_folds=args.n_folds)


if __name__ == "__main__":
    main()