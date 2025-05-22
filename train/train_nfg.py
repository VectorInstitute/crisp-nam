import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
from sksurv.util import Surv
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

import configargparse
import random
from tqdm import tqdm
from copy import deepcopy


from nfg.dsm.dsm_api import DSMBase
from nfg.nfg_torch import NeuralFineGrayTorch
from nfg.losses import total_loss, total_loss_cs, weighted_total_loss
from nfg.utilities import train_nfg
from nfg.nfg_api import NeuralFineGray


from datasets.SurvivalDataset import SurvivalDataset
from datasets.framingham_dataset import load_framingham
from datasets.support_dataset import load_support_dataset
from datasets.pbc_dataset import load_pbc2_dataset
from datasets.synthetic_dataset import load_synthetic_dataset
from nfg.metrics.calibration import brier_score
from nfg.metrics.discrimination import auc_td
from utils.risk_cif import predict_absolute_risk, compute_baseline_cif
from utils.plotting import plot_feature_importance



def parse_args():
    parser = configargparse.ArgumentParser(
        description="Training script for Neural Fine Gray model",
        default_config_files=["config.yml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    
    parser.add_argument("-c", "--config", is_config_file=True,
                      help="Path to config file")

    parser.add_argument("--dataset", type=str, default="framingham", 
                      help="Dataset to use: (framingham, support, pbc, synthetic)")
    

    parser.add_argument("--scaling", type=str, default="standard", choices=["minmax", "standard", "none"],
                      help="Data scaling method for continuous features")
    

    parser.add_argument("--num_epochs", type=int, default=1500, 
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, 
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                      help="Learning rate for optimizer")
    parser.add_argument("--l2_reg", type=float, default=1e-5, 
                      help="L2 regularization weight")
   
    
  
    parser.add_argument("--dropout_rate", type=float, default=0.5, 
                      help="Dropout rate for model")
    parser.add_argument("--feature_dropout", type=float, default=0.0, 
                      help="Feature dropout rate")
    parser.add_argument("--hidden_dimensions", type=int, nargs="+", default=[100,100],
                  help="Hidden layer dimensions (space-separated list)")
    parser.add_argument("--layers_surv", type=int, nargs="+", default=[100],
                  help="Surv layer dimensions (space-separated list)")
    parser.add_argument("--batch_norm", type=str, default="False", choices=["True", "False"],
                      help="Whether to use batch normalization")
    
    
    parser.add_argument("--cause_specific", type=str, default="False", choices=["True", "False"],
                      help="Whether to use cause-specific loss")
    parser.add_argument("--normalise", type=str, default="None", choices=["None", "uniform", "minmax"],
                      help="Time normalization method")
    parser.add_argument("--multihead", type=str, default="True", choices=["True", "False"],
                      help="Whether to use multihead architecture")
    
  
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "SGD", "RMSProp","AdamW"],
                      help="Optimizer to use for training")
    parser.add_argument("--act", type=str, default="Tanh", choices=["Tanh", "ReLU", "ELU"],
                      help="Actvation function to use in the model")
    
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    parser.add_argument("--n_folds", type=int, default=5, 
                      help="Number of folds for cross-validation")
    
    return parser.parse_args()


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



def evaluate_model(model, x_val, t_val, e_val, t_train, e_train, times):
    """
    Evaluate model performance with correct metrics
    """


    n_events = model.torch_model.risks
    print(f"Number of events/risks: {n_events}")
    abs_risks = np.zeros((len(x_val), n_events, len(times)))
    
  
    
    for k in range(n_events):
        risk_idx = k + 1
        print(f"Predicting survival for risk {risk_idx}")
        
        # Get survival probabilities
        survival_probs = model.predict_survival(x_val, list(times), risk=risk_idx)
        
        # IMPORTANT: Convert survival probabilities to risks (1 - survival)
        # Higher risk = higher likelihood of event
        abs_risks[:, k, :] = 1 - survival_probs
    

    
    # Create scikit-survival format
    dt = np.dtype([('event', bool), ('time', float)])
    survival_train = np.zeros(len(t_train), dtype=dt)
    survival_train['event'] = e_train > 0
    survival_train['time'] = t_train
    
    survival_val = np.zeros(len(t_val), dtype=dt)
    survival_val['event'] = e_val > 0
    survival_val['time'] = t_val
    
    metrics = defaultdict(list)
    
   
    for k in range(n_events):
        risk_idx = k + 1
        for i, time in enumerate(times):
            try:         
                risk_preds = abs_risks[:, k, :]
                
                
                auc_score, _ = auc_td(
                    e_val,
                    t_val,
                    risk_preds,  
                    times,
                    time,
                    km=(e_train, t_train),
                    competing_risk=risk_idx
                )
                metrics[f"auc_event{risk_idx}_t{time:.2f}"].append(float(auc_score))
            except Exception as ex:
                print(f"[Warning] AUC failed: {ex}")
                metrics[f"auc_event{risk_idx}_t{time:.2f}"].append(np.nan)

            try:
                brier_score_val, _ = brier_score(
                    e_val,
                    t_val,
                    risk_preds,  
                    times,
                    time,
                    km=(e_train, t_train),
                    competing_risk=risk_idx
                )
                metrics[f"brier_event{risk_idx}_t{time:.2f}"].append(float(brier_score_val))
            except Exception as ex:
                print(f"[Warning] Brier failed: {ex}")
                metrics[f"brier_event{risk_idx}_t{time:.2f}"].append(np.nan)

            try:
                tdci_result = concordance_index_ipcw(
                    survival_train,
                    survival_val,
                    estimate=risk_preds[:, i],
                    tau=time
                )
                tdci_score = tdci_result[0] if isinstance(tdci_result, tuple) else tdci_result
                metrics[f"tdci_event{risk_idx}_t{time:.2f}"].append(float(tdci_score))
            except Exception as ex:
                print(f"[Warning] td-CI failed: {ex}")
                metrics[f"tdci_event{risk_idx}_t{time:.2f}"].append(np.nan)

    return metrics, abs_risks

def display_metrics_table(metrics_dict, n_folds=5, quantiles=[0.25, 0.5, 0.75]):
    """
    Display evaluation metrics summarized across folds for different time quantiles
    """
    if not metrics_dict:
        print("\nNo evaluation metrics collected.")
        return
        
    
    time_points_by_metric = {}
    event_types = set()
    metric_types = ['auc', 'tdci', 'brier', 'discrimination']
    
    for key, values in metrics_dict.items():
        if any(metric in key for metric in metric_types):
            if '.' in key:
                # Handle metrics with time point info
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
            else:
                # Handle simple metrics without time point info
                print(f"Simple metric found: {key} = {values}")
    
    if not time_points_by_metric and not event_types:
        print("\nUsing simplified metrics during debugging:")
        for key, values in metrics_dict.items():
            mean_val = np.mean(values) if values else np.nan
            std_val = np.std(values) if len(values) > 1 else 0
            print(f"  {key}: {mean_val:.3f} ± {std_val:.3f}")
        return
    
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
    
    if results:
        df = pd.DataFrame(results)
        
        # Define column order
        columns = ['Risk']
        for metric in ['AUC', 'TDCI', 'BRIER']:
            for q in quantiles:
                col = f"{metric}_q{q:.2f}"
                if col in df.columns:
                    columns.append(col)
        
        # Select columns in the right order (only those that exist)
        df = df[[col for col in columns if col in df.columns]]
        
        print("\nSummary Performance Metrics:")
        print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
        
        print("\nInterpretation:")
        print("- AUC: 0.5=random, >0.7=good, >0.8=excellent")
        print("- TDCI (Time-Dependent C-Index): 0.5=random, >0.7=good, >0.8=excellent") 
        print("- Brier Score: 0=perfect, <0.25=good, >0.25=poor")
    else:
        print("\nNo performance metrics to display in tabular format.")
        print("Using simplified metrics during debugging:")
        for key, values in metrics_dict.items():
            mean_val = np.mean(values) if values else np.nan
            std_val = np.std(values) if len(values) > 1 else 0
            print(f"  {key}: {mean_val:.3f} ± {std_val:.3f}")


def main():
    args = parse_args()
    print(args)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Store the original data in a named dictionary to ensure persistence
    dataset_vars = {}
    

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
    
   
    dataset_vars['x'] = x
    dataset_vars['t'] = t
    dataset_vars['e'] = e
    dataset_vars['feature_names'] = feature_names
    dataset_vars['n_cont'] = n_cont
    
    # Ensure all data is numpy arrays
    for key in ['x', 't', 'e']:
        if isinstance(dataset_vars[key], torch.Tensor):
            dataset_vars[key] = dataset_vars[key].detach().cpu().numpy()
    
    
    x = dataset_vars['x']
    t = dataset_vars['t']
    e = dataset_vars['e']
    feature_names = dataset_vars['feature_names']
    n_cont = dataset_vars['n_cont']
    
    if args.scaling.lower() == "standard":
        scaler = StandardScaler()
        # Scale ONLY the continuous features which are at the END of the feature matrix
        x[:, -n_cont:] = scaler.fit_transform(x[:, -n_cont:])
    elif args.scaling.lower() == "minmax":
        scaler = MinMaxScaler()
        x[:, -n_cont:] = scaler.fit_transform(x[:, -n_cont:])
    elif args.scaling.lower() == "none":
        print("No scaling applied to the data.")
    
    # Update the dictionary after scaling
    dataset_vars['x'] = x

   
    num_competing_risks = len(np.unique(e)) - 1  # Excluding censoring (0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_value = 2 if device == "cuda" else 0  # Set cuda value for NeuralFineGray
    
    # Create dataset for cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    all_metrics = defaultdict(list)
    quantiles = [0.25, 0.5, 0.75]
    
    # Calculate global evaluation times once
    safe_max = 0.99 * np.max(t)
    eval_times = np.quantile(t[t <= safe_max], quantiles)
    
    print(f"Evaluation times: {eval_times}")
    

    # Make a copy of the data to avoid reference issues
    x_data = x.copy() 
    t_data = t.copy()
    e_data = e.copy()

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_data, e_data)):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")

        # Use numpy arrays directly (don't convert to tensors)
        x_train = x_data[train_idx]
        t_train = t_data[train_idx]
        e_train = e_data[train_idx]

        # Compute class/event weights to penalize rarer events
        # Exclude censored (e == 0)
        event_labels = e_train[e_train > 0]
        unique_events, counts = np.unique(event_labels, return_counts=True)
        total_events = np.sum(counts)
        event_weights = {}
        for event, count in zip(unique_events, counts):
            # Inverse frequency: more weight for rarer events
            event_weights[event] = total_events / (len(unique_events) * count)
        # For censored, you may set weight to 1 or 0 (not used in loss)
        event_weights[0] = 1.0

        
        x_val = x_data[val_idx]
        t_val = t_data[val_idx]
        e_val = e_data[val_idx]

        # Create the Neural Fine Gray model
        model = NeuralFineGray(
            cuda=cuda_value,
            cause_specific=False,
            normalise=args.normalise,
            layers=args.hidden_dimensions,
            layers_surv = args.layers_surv,
            act=args.act,
            dropout=args.dropout_rate,
            multihead=True, #args.multihead,
        )

        # Fit the model
        model.fit(
            x_train, 
            t_train, 
            e_train,
            vsize=0.10,  # Use 15% for validation during training
            optimizer=args.optimizer,
            random_state=args.seed,
            n_iter=args.num_epochs,
            lr=args.learning_rate, 
            weight_decay=args.l2_reg,
            bs=args.batch_size

        )
        
        # Evaluate the model
        fold_metrics, abs_risks = evaluate_model(
            model, 
            x_val, 
            t_val, 
            e_val,
            t_train, 
            e_train, 
            eval_times
        )
        
        for k, v in fold_metrics.items():
            all_metrics[k].extend(v)

       

    # Display metrics in table format
    display_metrics_table(all_metrics, n_folds=args.n_folds)
    
    # Create figs subdirectory if not present
    import os
    if not os.path.exists("figs"):
        os.makedirs("figs")
        
        
if __name__ == "__main__":
    main()