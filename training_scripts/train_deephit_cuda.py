import os
import configargparse
from collections import defaultdict

import torch
import numpy as np
import torch.optim as optim
from sksurv.util import Surv
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sksurv.metrics import concordance_index_ipcw

# Import the DeepHit model implementation
from data_utils import *
from model_utils import (
    set_seed,
    EarlyStopping,
    create_fc_mask1_gpu,
    create_fc_mask2_gpu
)
from crisp_nam.models import DeepHit
from crisp_nam.metrics import brier_score, auc_td

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
    parser.add_argument("--num_epochs", type=int, default=500, 
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, 
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                      help="Learning rate for optimizer")
    parser.add_argument("--l2_reg", type=float, default=1e-4, 
                      help="L2 regularization weight")
    parser.add_argument("--patience", type=int, default=10, 
                      help="Patience for early stopping")
    
    # DeepHit specific parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                      help="Weight for log-likelihood loss")
    parser.add_argument("--beta", type=float, default=0.5,
                      help="Weight for ranking loss")
    parser.add_argument("--gamma", type=float, default=0.5,
                      help="Weight for calibration loss")
    parser.add_argument("--h_dim_shared", type=int, default=64,
                      help="Hidden dimension for shared network")
    parser.add_argument("--h_dim_CS", type=int, default=16,
                      help="Hidden dimension for cause-specific networks")
    parser.add_argument("--num_layers_shared", type=int, default=2,
                      help="Number of layers in shared network")
    parser.add_argument("--num_layers_CS", type=int, default=2,
                      help="Number of layers in cause-specific networks")
    parser.add_argument("--num_categories", type=int, default=100,
                      help="Number of time categories for discretization")
    parser.add_argument("--active_fn", type=str, default="relu", choices=["relu", "elu", "tanh"],
                      help="Activation function")
    
    # General parameters
    parser.add_argument("--dropout_rate", type=float, default=0.2, 
                      help="Dropout rate for model")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    parser.add_argument("--n_folds", type=int, default=5, 
                      help="Number of folds for cross-validation")
    parser.add_argument("--num_workers", type=int, default=8,
                      help="Number of workers for data loading")
    parser.add_argument("--eval_freq", type=int, default=10,
                      help="Evaluate every N epochs during training")
    parser.add_argument("--use_amp", action="store_true",
                      help="Use Automatic Mixed Precision for training")
    
    return parser.parse_args()

def train_deephit_model(model, train_loader, val_loader=None, alpha=1.0, beta=1.0, gamma=1.0,
                         num_epochs=500, learning_rate=1e-3, l2_reg=0.01, patience=10, 
                         eval_freq=10, use_amp=False, verbose=True):
    """
    Train the DeepHit model using the three loss components
    Optimized with AMP support, prefetching, and CUDA streams
    """
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    early_stopper = EarlyStopping(patience=patience)
    
    # Get model dimensions for mask creation
    num_Event = model.num_Event
    num_Category = model.num_Category
    
    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Enable asynchronous CUDA execution
    torch.backends.cudnn.benchmark = True
    
    # Increase batch size if it's too small (optional)
    # if train_loader.batch_size < 512 and device == 'cuda':
    #    print("Warning: Small batch size. Consider increasing for better GPU utilization.")
    
    # Prefetch data using CUDA streams for async data loading
    prefetch_stream = torch.cuda.Stream() if device == 'cuda' else None
    
    # Precomputed masks for common events (optimization)
    precomputed_masks2 = {}
    for t_val in range(num_Category):
        mask = torch.zeros(1, num_Category, device=device)
        mask[0, t_val:] = 1
        precomputed_masks2[t_val] = mask
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # Prefetch first batch
        if prefetch_stream is not None:
            try:
                # Get first batch
                batch_iter = iter(train_loader)
                prefetch_batch = next(batch_iter)
                # Prefetch to GPU asynchronously
                with torch.cuda.stream(prefetch_stream):
                    prefetch_x, prefetch_t, prefetch_e, prefetch_t_disc = [t.to(device, non_blocking=True) for t in prefetch_batch]
                    # Precompute masks
                    prefetch_mask1 = create_fc_mask1_gpu(prefetch_e, prefetch_t_disc, num_Event, num_Category, device)
                    prefetch_mask2 = torch.cat([precomputed_masks2[int(t_val.item())] for t_val in prefetch_t_disc], dim=0)
            except StopIteration:
                prefetch_batch = None
        else:
            prefetch_batch = None
        
        # Main training loop with prefetching
        batch_iter = iter(train_loader)
        more_batches = True
        
        while more_batches:
            # Synchronize with prefetched data 
            if prefetch_stream is not None and prefetch_batch is not None:
                torch.cuda.current_stream().wait_stream(prefetch_stream)
                x, t, e, t_disc = prefetch_x, prefetch_t, prefetch_e, prefetch_t_disc
                mask1, mask2 = prefetch_mask1, prefetch_mask2
                
                # Prefetch next batch
                try:
                    prefetch_batch = next(batch_iter)
                    with torch.cuda.stream(prefetch_stream):
                        prefetch_x, prefetch_t, prefetch_e, prefetch_t_disc = [t.to(device, non_blocking=True) for t in prefetch_batch]
                        # Precompute masks
                        prefetch_mask1 = create_fc_mask1_gpu(prefetch_e, prefetch_t_disc, num_Event, num_Category, device)
                        prefetch_mask2 = torch.cat([precomputed_masks2[int(t_val.item())] for t_val in prefetch_t_disc], dim=0)
                except StopIteration:
                    prefetch_batch = None
                    more_batches = False
            else:
                # Regular data loading if no prefetching
                try:
                    batch = next(batch_iter)
                    x, t, e, t_disc = [t.to(device, non_blocking=True) for t in batch]
                    # Create masks for loss computation
                    mask1 = create_fc_mask1_gpu(e, t_disc, num_Event, num_Category, device)
                    mask2 = torch.cat([precomputed_masks2[int(t_val.item())] for t_val in t_disc], dim=0)
                except StopIteration:
                    more_batches = False
                    continue
            
            # Automatic mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    out, _ = model(x)
                    # Compute loss components
                    loss = model.compute_loss(out, t, e, mask1, mask2, alpha, beta, gamma)
                
                # Optimization step with gradient scaling
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                out, _ = model(x)
                # Compute loss components
                loss = model.compute_loss(out, t, e, mask1, mask2, alpha, beta, gamma)
                
                # Regular optimization step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation if provided, but only every eval_freq epochs to save time
        if val_loader and (epoch % eval_freq == 0 or epoch == num_epochs - 1):
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, t, e, t_disc in val_loader:
                    x, t, e, t_disc = x.to(device, non_blocking=True), t.to(device, non_blocking=True), \
                                      e.to(device, non_blocking=True), t_disc.to(device, non_blocking=True)
                    
                    # Create masks for validation (using precomputed masks)
                    mask1 = create_fc_mask1(e, t_disc, num_Event, num_Category, device)
                    mask2 = torch.cat([precomputed_masks2[int(t_val.item())] for t_val in t_disc], dim=0)
                    
                    # Forward pass
                    out, _ = model(x)
                    
                    # Compute loss
                    loss = model.compute_loss(out, t, e, mask1, mask2, alpha, beta, gamma)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            early_stopper.step(avg_val_loss)
            
            if verbose and (epoch % eval_freq == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                
            if early_stopper.should_stop:
                if verbose:
                    print("Early stopping triggered.")
                break
        elif verbose and (epoch % eval_freq == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")


def predict_absolute_risk_deephit(model, x_test, times, t_max=None, device="cpu"):
    """
    Predict absolute risks using DeepHit model
    Optimized to keep more operations on GPU
    """
    model.eval()
    
    # Convert x_test to tensor and move to device
    if not isinstance(x_test, torch.Tensor):
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
    else:
        x_test_tensor = x_test.to(device)
    
    # Process in batches for large datasets
    batch_size = 1024  # Can be adjusted based on available GPU memory
    n_samples = x_test_tensor.shape[0]
    
    # Get model dimensions
    n_events = model.num_Event
    n_categories = model.num_Category
    n_times = len(times)
    
    # Set t_max if not provided
    if t_max is None:
        t_max = max(times)
    
    # Initialize the output tensor on GPU
    abs_risks = torch.zeros(n_samples, n_events, n_times, device=device)
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = x_test_tensor[i:end_idx]
            
            # Get model predictions
            preds, _ = model(batch)  # shape: (batch_size, num_Event, num_Category)
            
            # For each evaluation time, calculate CIF
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
                        abs_risks[i:end_idx, k, t_idx] = torch.sum(preds[:, k, :lower_bin+1], dim=1)
                else:
                    # Need to interpolate between bins
                    weight_upper = scaled_t - lower_bin
                    weight_lower = 1 - weight_upper
                    
                    for k in range(n_events):
                        # Cumulative probability up to lower bin
                        cum_prob_lower = torch.sum(preds[:, k, :lower_bin+1], dim=1)
                        
                        # Cumulative probability up to upper bin
                        cum_prob_upper = torch.sum(preds[:, k, :upper_bin+1], dim=1)
                        
                        # Linear interpolation
                        abs_risks[i:end_idx, k, t_idx] = weight_lower * cum_prob_lower + weight_upper * cum_prob_upper
    
    # Move results back to CPU and convert to numpy only if needed
    return abs_risks.cpu().numpy()


def evaluate_model(model, x_val, t_val, e_val, t_train, e_train, times, max_time, device="cpu"):
    """
    Evaluate the DeepHit model using time-dependent AUC, Brier score, and td-C-index
    """
    # Predict absolute risks for validation set
    abs_risks_tensor = predict_absolute_risk_deephit(model, x_val, times, t_max=max_time, device=device)
    abs_risks = abs_risks_tensor  # Now returns numpy array directly
    
    # Format data for scikit-survival functions
    survival_train = Surv.from_arrays(e_train != 0, t_train)
    survival_val = Surv.from_arrays(e_val != 0, t_val)
    
    metrics = defaultdict(list)
    n_events = abs_risks.shape[1]
    
    # Calculate metrics in parallel using joblib if available
    try:
        from joblib import Parallel, delayed
        
        def process_metric(k, i, time):
            result = {}
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
                result[f"auc_event{k+1}_t{time:.2f}"] = float(auc_score)
            except Exception as ex:
                result[f"auc_event{k+1}_t{time:.2f}"] = np.nan
            
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
                result[f"brier_event{k+1}_t{time:.2f}"] = float(brier_score_val)
            except Exception as ex:
                result[f"brier_event{k+1}_t{time:.2f}"] = np.nan
            
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
                result[f"tdci_event{k+1}_t{time:.2f}"] = float(tdci_score)
            except Exception as ex:
                result[f"tdci_event{k+1}_t{time:.2f}"] = np.nan
            
            return result
        
        # Collect all tasks
        tasks = [(k, i, time) for k in range(n_events) for i, time in enumerate(times)]
        
        # Run in parallel
        results = Parallel(n_jobs=-1)(delayed(process_metric)(k, i, time) for k, i, time in tasks)
        
        # Combine results
        for result in results:
            for k, v in result.items():
                metrics[k].append(v)
                
    except ImportError:
        # Fall back to sequential computation
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
    
    # Configure CUDA for maximum performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Allow TF32 on Ampere GPUs for potential 2-3x speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Print CUDA device info
        device_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"Using CUDA device: {device_name} with compute capability {compute_capability[0]}.{compute_capability[1]}")
        
        # Set GPU memory usage strategy
        torch.cuda.empty_cache()
        
        # Manual memory management
        if hasattr(torch.cuda, 'memory_stats'):
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
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
    
    # Convert data to tensors immediately before scaling
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scale features if needed (more efficiently)
    if args.scaling.lower() == "standard":
        scaler = StandardScaler()
        if n_cont > 0:
            cont_features = x[:, -n_cont:].copy()
            x[:, -n_cont:] = scaler.fit_transform(cont_features)
    elif args.scaling.lower() == "minmax":
        scaler = MinMaxScaler()
        if n_cont > 0:
            cont_features = x[:, -n_cont:].copy()
            x[:, -n_cont:] = scaler.fit_transform(cont_features)
    elif args.scaling.lower() == "none":
        pass
    else:
        raise ValueError(f"Scaling method {args.scaling} not supported")
    
    # Get number of competing risks
    num_competing_risks = len(np.unique(e)) - 1  # Excluding censoring (0)
    
    # Maximum time in the dataset
    max_time = np.max(t)
    
    # Convert data to tensors early and move to device if small enough
    # For large datasets, keep on CPU and transfer in batches
    dataset_size = x.nbytes / (1024 * 1024)  # Size in MB
    if dataset_size < 1000:  # If dataset is less than 1GB
        try:
            # Try to move entire dataset to GPU
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
            e_tensor = torch.tensor(e, dtype=torch.long, device=device)
            print(f"Entire dataset moved to {device} (Size: {dataset_size:.2f} MB)")
            # Create dataset with GPU tensors
            dataset = SurvivalDatasetDeepHit(x_tensor.cpu().numpy(), t_tensor.cpu().numpy(), 
                                           e_tensor.cpu().numpy(), args.num_categories)
        except RuntimeError:
            # If OOM, fall back to CPU
            print(f"Dataset too large for GPU memory, keeping on CPU (Size: {dataset_size:.2f} MB)")
            x_tensor = torch.tensor(x, dtype=torch.float32)
            t_tensor = torch.tensor(t, dtype=torch.float32)
            e_tensor = torch.tensor(e, dtype=torch.long)
            dataset = SurvivalDatasetDeepHit(x, t, e, args.num_categories)
    else:
        # Large dataset, keep on CPU
        print(f"Large dataset detected ({dataset_size:.2f} MB), keeping on CPU and using efficient batch loading")
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
        
        # Calculate optimal batch size based on device
        if device == "cuda":
            # Dynamically adjust batch size based on available GPU memory
            total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            # Use heuristic - larger batch sizes for larger GPUs
            dynamic_batch_size = min(2048, max(512, int(args.batch_size * (total_gpu_mem / 8))))
            if dynamic_batch_size != args.batch_size:
                print(f"Adjusting batch size from {args.batch_size} to {dynamic_batch_size} based on GPU memory")
                batch_size = dynamic_batch_size
            else:
                batch_size = args.batch_size
        else:
            batch_size = args.batch_size
        
        # Create data loaders with optimized settings
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Use different prefetch factors based on data size and complexity
        prefetch_factor = 2
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device == "cuda" else False,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=prefetch_factor if args.num_workers > 0 else None,
            drop_last=True  # Drop last incomplete batch for better performance
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=True if device == "cuda" else False,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=prefetch_factor if args.num_workers > 0 else None
        )
        
        # Initialize DeepHit model with optimized size
        input_dims = {
            'x_dim': x.shape[1],
            'num_Event': num_competing_risks,
            'num_Category': args.num_categories
        }
        
        # Adjust network size based on available GPU memory
        if device == "cuda":
            h_dim_shared = args.h_dim_shared
            h_dim_CS = args.h_dim_CS
            
            # For GPUs with more memory, can use larger networks
            if total_gpu_mem > 16:  # For high-end GPUs
                h_dim_shared = max(args.h_dim_shared, 128)
                h_dim_CS = max(args.h_dim_CS, 32)
                print(f"Using larger network dimensions for high-memory GPU: {h_dim_shared}/{h_dim_CS}")
        else:
            h_dim_shared = args.h_dim_shared
            h_dim_CS = args.h_dim_CS
        
        network_settings = {
            'h_dim_shared': h_dim_shared,
            'h_dim_CS': h_dim_CS,
            'num_layers_shared': args.num_layers_shared,
            'num_layers_CS': args.num_layers_CS,
            'active_fn': args.active_fn,
            'keep_prob': 1.0 - args.dropout_rate
        }
        
        # Create model
        model = DeepHit(input_dims, network_settings).to(device)
        
        # Optional: print model summary for debugging
        if fold == 0:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model has {num_params:,} parameters")
            
            # Print estimated memory usage per batch
            # 4 bytes per float32 parameter, multiply by 4 for activations, gradients, optimizer state
            param_memory_mb = num_params * 4 * 4 / (1024 * 1024)
            print(f"Estimated model memory usage: {param_memory_mb:.2f} MB per batch")

        # Use profile first batch to analyze performance bottlenecks
        if fold == 0 and device == "cuda":
            try:
                print("Profiling first batch for performance analysis...")
                # Get a sample batch for profiling
                sample_batch = next(iter(train_loader))
                x_sample, t_sample, e_sample, t_disc_sample = [t.to(device, non_blocking=True) for t in sample_batch]
                
                # Simple profiling of forward and backward pass
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    # Create masks
                    mask1 = create_fc_mask1_gpu(e_sample, t_disc_sample, model.num_Event, model.num_Category, device)
                    mask2 = create_fc_mask2_gpu(t_disc_sample, model.num_Category, device)
                    
                    # Forward pass
                    out, _ = model(x_sample)
                    
                    # Compute loss
                    loss = model.compute_loss(out, t_sample, e_sample, mask1, mask2, args.alpha, args.beta, args.gamma)
                    
                    # Backward pass
                    loss.backward()
                
                # Print profiling results
                profile_sorted = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
                print("Profile of most expensive CUDA operations:")
                print(profile_sorted)
                
                # Identify bottlenecks
                cpu_pct = prof.self_cpu_time_total / prof.total_cuda_time_total * 100
                if cpu_pct > 20:
                    print(f"WARNING: High CPU overhead ({cpu_pct:.1f}%). Consider optimizing CPU-GPU transfers.")
                
            except Exception as e:
                print(f"Profiling skipped due to error: {e}")
        
        # Train the model with optimizations
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
            eval_freq=args.eval_freq,
            use_amp=args.use_amp,
            verbose=True
        )
        
        # Evaluate model (less frequently during training)
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
        
        # Free memory after each fold
        if device == "cuda":
            del model
            torch.cuda.empty_cache()
    
    # Display final metrics
    display_metrics_table(all_metrics, n_folds=args.n_folds)


if __name__ == "__main__":
    main()