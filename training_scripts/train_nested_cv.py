import yaml
import configargparse
from collections import defaultdict

import optuna
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tabulate import tabulate
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
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
        description="Nested Cross-Validation for CrispNAM model",
        default_config_files=["config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    
    parser.add_argument("-c", "--config", is_config_file=True,
                      help="Path to config file")

    parser.add_argument("--dataset", type=str, default="framingham", 
                      choices=["framingham", "support", "pbc", "synthetic"],
                      help="Dataset to use")

    parser.add_argument("--scaling", type=str, default="standard", 
                      choices=["minmax", "standard", "none"],
                      help="Data scaling method for continuous features")

    parser.add_argument("--num_epochs", type=int, default=100, 
                      help="Number of training epochs (reduced for nested CV)")
    parser.add_argument("--batch_size", type=int, default=256, 
                      help="Batch size for training")
    parser.add_argument("--patience", type=int, default=10, 
                      help="Patience for early stopping")

    # Nested CV parameters
    parser.add_argument("--outer_folds", type=int, default=5, 
                      help="Number of outer CV folds")
    parser.add_argument("--inner_folds", type=int, default=3, 
                      help="Number of inner CV folds for hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=20, 
                      help="Number of Optuna trials per inner fold")

    # Event weighting
    parser.add_argument("--event_weighting", type=str, default="none", 
                      choices=["none", "balanced", "custom"],
                      help="Event weighting strategy")
    parser.add_argument("--custom_event_weights", type=str, default=None,
                      help="Custom weights for events (comma-separated)")

    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    
    return parser.parse_args()


def train_model(model, train_loader, val_loader=None, num_epochs=100, learning_rate=1e-3, 
                l2_reg=0.01, patience=10, event_weights=None, verbose=False):
    """Train model with early stopping"""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(patience=patience)
    device = next(model.parameters()).device
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, t, e, _ in train_loader:
            x, t, e = x.to(device), t.to(device), e.to(device)
            risk_scores, _ = model(x)

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
                    
                    if event_weights is not None:
                        loss = weighted_negative_log_likelihood_loss(risk_scores, t, e, 
                                                                   model.num_competing_risks,
                                                                   event_weights=event_weights)
                    else:
                        loss = negative_log_likelihood_loss(risk_scores, t, e, model.num_competing_risks)
                        
                    reg = compute_l2_penalty(model) * l2_reg
                    val_loss += (loss + reg).item()
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
            early_stopper.step(avg_val_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if early_stopper.should_stop:
                if verbose:
                    print("Early stopping triggered.")
                break
        elif verbose and epoch % 10 == 0:
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")
            
    return best_val_loss


def hyperparameter_optimization(x_train_inner, t_train_inner, e_train_inner, 
                               x_val_inner, t_val_inner, e_val_inner,
                               num_competing_risks, device, args, event_weights, n_cont):
    """Run Optuna hyperparameter optimization on inner training data"""
    
    def objective(trial):
        # Define hyperparameters to optimize
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-1, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.8)
        feature_dropout = trial.suggest_float('feature_dropout', 0.0, 0.5)
        
        # Hidden dimensions
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dimensions = []
        for i in range(n_layers):
            hidden_dimensions.append(trial.suggest_categorical(f'hidden_dim_{i}', [32, 64, 128, 256]))
        
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        
        # Create data loaders
        train_dataset = SurvivalDataset(x_train_inner, t_train_inner, e_train_inner)
        val_dataset = SurvivalDataset(x_val_inner, t_val_inner, e_val_inner)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Initialize model
        model = CrispNamModel(
            num_features=x_train_inner.shape[1],
            num_competing_risks=num_competing_risks,
            hidden_sizes=hidden_dimensions,
            dropout_rate=dropout_rate,
            feature_dropout=feature_dropout,
            batch_norm=batch_norm
        ).to(device)
        
        # Train model
        best_val_loss = train_model(
            model, train_loader, val_loader, 
            num_epochs=args.num_epochs, 
            learning_rate=learning_rate,
            l2_reg=l2_reg, 
            patience=args.patience,
            event_weights=event_weights,
            verbose=False
        )
        
        return best_val_loss
    
    # Create study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    
    return study.best_params


def evaluate_model(model, x_val, t_val, e_val, t_train, e_train, abs_risks, times):
    """Evaluate model performance"""
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
                    e_val, t_val, risk_preds_2d, times, time,
                    km=(e_train, t_train), primary_risk=k+1
                )
                metrics[f"auc_event{k+1}_t{time:.2f}"].append(float(auc_score))
            except Exception as ex:
                print(f"[Warning] AUC failed at t={time:.2f}, event={k+1}: {ex}")
                metrics[f"auc_event{k+1}_t{time:.2f}"].append(np.nan)

            try:
                brier_score_val, _ = brier_score(
                    e_val, t_val, risk_preds_2d, times, time,
                    km=(e_train, t_train), primary_risk=k+1
                )
                metrics[f"brier_event{k+1}_t{time:.2f}"].append(float(brier_score_val))
            except Exception as ex:
                print(f"[Warning] Brier failed at t={time:.2f}, event={k+1}: {ex}")
                metrics[f"brier_event{k+1}_t{time:.2f}"].append(np.nan)

            try:
                tdci_result = concordance_index_ipcw(
                    survival_train, survival_val, estimate=risk_preds, tau=time
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


def display_metrics_table(metrics_dict, quantiles=[0.25, 0.5, 0.75], dataset_name="dataset"):
    """Display evaluation metrics table and return processed data"""
    time_points_by_metric = {}
    event_types = set()
    metric_types = ['auc', 'tdci', 'brier']
    
    for key, values in metrics_dict.items():
        if any(metric in key for metric in metric_types):
            parts = key.split('_')
            metric_type = parts[0]
            event_info = parts[1]
            
            event_type = int(event_info.replace('event', ''))
            event_types.add(event_type)
            
            if len(parts) > 2 and parts[2].startswith('t'):
                time_point = float(parts[2].replace('t', ''))
                
                if (event_type, metric_type) not in time_points_by_metric:
                    time_points_by_metric[(event_type, metric_type)] = {}
                
                time_points_by_metric[(event_type, metric_type)][time_point] = values
    
    # Create summary table
    summary_results = []
    detailed_results = []
    
    for event_type in sorted(event_types):
        row = {'Risk': f"Type {event_type}"}
        
        for metric_type in metric_types:
            if (event_type, metric_type) not in time_points_by_metric:
                for q in quantiles:
                    row[f"{metric_type.upper()}_q{q:.2f}"] = "N/A"
                continue
            
            time_data = time_points_by_metric[(event_type, metric_type)]
            sorted_times = sorted(time_data.keys())
            
            for q in quantiles:
                q_idx = max(0, min(len(sorted_times) - 1, int(len(sorted_times) * q)))
                q_time = sorted_times[q_idx]
                q_values = time_data[q_time]
                
                if q_values:
                    value_array = np.array(q_values)
                    mean_val = np.nanmean(value_array)
                    std_val = np.nanstd(value_array)
                    row[f"{metric_type.upper()}_q{q:.2f}"] = f"{mean_val:.3f} ± {std_val:.3f}"
                    
                    # Add to detailed results for separate CSV
                    detailed_results.append({
                        'Dataset': dataset_name,
                        'Risk_Type': event_type,
                        'Metric': metric_type.upper(),
                        'Time_Quantile': f"q{q:.2f}",
                        'Time_Value': q_time,
                        'Mean': mean_val,
                        'Std': std_val,
                        'N_Folds': len(q_values),
                        'Raw_Values': ';'.join(map(str, q_values))
                    })
                else:
                    row[f"{metric_type.upper()}_q{q:.2f}"] = "N/A"
        
        summary_results.append(row)
    
    # Create DataFrames
    summary_df = pd.DataFrame(summary_results)
    detailed_df = pd.DataFrame(detailed_results)
    
    columns = ['Risk']
    for metric in ['AUC', 'TDCI', 'BRIER']:
        for q in quantiles:
            columns.append(f"{metric}_q{q:.2f}")
    
    summary_df = summary_df[[col for col in columns if col in summary_df.columns]]
    
    print("\nNested CV Performance Metrics:")
    print(tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\nInterpretation:")
    print("- AUC: 0.5=random, >0.7=good, >0.8=excellent")
    print("- TDCI (Time-Dependent C-Index): 0.5=random, >0.7=good, >0.8=excellent") 
    print("- Brier Score: 0=perfect, <0.25=good, >0.25=poor")
    
    return summary_df, detailed_df


def main():
    args = parse_args()
    print(f"Running Nested Cross-Validation with {args.outer_folds} outer folds and {args.inner_folds} inner folds")
    print(args)
    
    # Set random seed
    set_seed(args.seed)
    
    # Load dataset
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
    
    num_competing_risks = len(np.unique(e)) - 1
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Outer cross-validation loop
    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    all_metrics = defaultdict(list)
    all_best_params = []
    
    quantiles = [0.25, 0.5, 0.75]
    safe_max = 0.99 * np.max(t)
    eval_times = np.quantile(t[t <= safe_max], quantiles)
    
    print(f"Evaluation times: {eval_times}")
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, e)):
        print(f"\n=== Outer Fold {outer_fold + 1}/{args.outer_folds} ===")
        
        # Split data for outer fold
        x_train_outer, x_test_outer = x[train_idx].copy(), x[test_idx].copy()
        t_train_outer, t_test_outer = t[train_idx], t[test_idx]
        e_train_outer, e_test_outer = e[train_idx], e[test_idx]
        
        # Apply scaling within outer fold
        if args.scaling.lower() == "standard":
            scaler = StandardScaler()
            x_train_outer[:, -n_cont:] = scaler.fit_transform(x_train_outer[:, -n_cont:])
            x_test_outer[:, -n_cont:] = scaler.transform(x_test_outer[:, -n_cont:])
        elif args.scaling.lower() == "minmax":
            scaler = MinMaxScaler()
            x_train_outer[:, -n_cont:] = scaler.fit_transform(x_train_outer[:, -n_cont:])
            x_test_outer[:, -n_cont:] = scaler.transform(x_test_outer[:, -n_cont:])
        
        # Inner cross-validation for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)
        inner_scores = []
        inner_params = []
        
        for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(x_train_outer, e_train_outer)):
            print(f"  Inner Fold {inner_fold + 1}/{args.inner_folds}")
            
            # Split inner training data
            x_train_inner = x_train_outer[train_inner_idx].copy()
            x_val_inner = x_train_outer[val_inner_idx].copy()
            t_train_inner, t_val_inner = t_train_outer[train_inner_idx], t_train_outer[val_inner_idx]
            e_train_inner, e_val_inner = e_train_outer[train_inner_idx], e_train_outer[val_inner_idx]
            
            # Calculate event weights for this inner fold
            event_weights = None
            if args.event_weighting == "balanced":
                event_counts = np.zeros(num_competing_risks)
                for k in range(1, num_competing_risks + 1):
                    event_counts[k-1] = np.sum(e_train_inner == k)
                event_counts = np.maximum(event_counts, 1)
                event_weights = 1.0 / event_counts
                event_weights = event_weights * (num_competing_risks / event_weights.sum())
                event_weights = torch.tensor(event_weights, dtype=torch.float32, device=device)
            
            # Hyperparameter optimization
            best_params = hyperparameter_optimization(
                x_train_inner, t_train_inner, e_train_inner,
                x_val_inner, t_val_inner, e_val_inner,
                num_competing_risks, device, args, event_weights, n_cont
            )
            
            inner_params.append(best_params)
        
        # Select best hyperparameters (could use ensemble or average, here we take most common)
        # For simplicity, we'll use the first inner fold's best params
        best_params = inner_params[0]
        all_best_params.append(best_params)
        
        print(f"  Selected hyperparameters for outer fold {outer_fold + 1}: {best_params}")
        
        # Train final model on all outer training data with best hyperparameters
        n_layers = best_params['n_layers']
        hidden_dimensions = [best_params[f'hidden_dim_{i}'] for i in range(n_layers)]
        
        # Calculate event weights for outer training data
        event_weights = None
        if args.event_weighting == "balanced":
            event_counts = np.zeros(num_competing_risks)
            for k in range(1, num_competing_risks + 1):
                event_counts[k-1] = np.sum(e_train_outer == k)
            event_counts = np.maximum(event_counts, 1)
            event_weights = 1.0 / event_counts
            event_weights = event_weights * (num_competing_risks / event_weights.sum())
            event_weights = torch.tensor(event_weights, dtype=torch.float32, device=device)
        
        # Create final model
        final_model = CrispNamModel(
            num_features=x.shape[1],
            num_competing_risks=num_competing_risks,
            hidden_sizes=hidden_dimensions,
            dropout_rate=best_params['dropout_rate'],
            feature_dropout=best_params['feature_dropout'],
            batch_norm=best_params['batch_norm']
        ).to(device)
        
        # Train final model
        train_dataset = SurvivalDataset(x_train_outer, t_train_outer, e_train_outer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        train_model(
            final_model, train_loader, None,
            num_epochs=args.num_epochs, 
            learning_rate=best_params['learning_rate'],
            l2_reg=best_params['l2_reg'], 
            patience=args.patience,
            event_weights=event_weights,
            verbose=True
        )
        
        # Evaluate on test data
        baseline_cifs = {k: compute_baseline_cif(t_train_outer, e_train_outer, eval_times, k + 1) 
                        for k in range(num_competing_risks)}
        
        abs_risks = predict_absolute_risk(final_model, x_test_outer, baseline_cifs, eval_times, device=device)
        
        fold_metrics = evaluate_model(final_model, x_test_outer, t_test_outer, e_test_outer,
                                      t_train_outer, e_train_outer, abs_risks, eval_times)
        
        for k, v in fold_metrics.items():
            all_metrics[k].extend(v)
    
    # Display final results and get DataFrames for saving
    summary_df, detailed_df = display_metrics_table(all_metrics, dataset_name=args.dataset)
    
    # Generate shape plots using the last trained model (from the last outer fold)
    print("\n=== Generating Shape Function Plots ===")
    
    # Create figs subdirectory if not present
    import os
    if not os.path.exists("figs"):
        os.makedirs("figs")
    
    # Use the entire dataset for plotting (with proper scaling)
    x_plot = x.copy()
    if args.scaling.lower() == "standard":
        scaler = StandardScaler()
        x_plot[:, -n_cont:] = scaler.fit_transform(x_plot[:, -n_cont:])
    elif args.scaling.lower() == "minmax":
        scaler = MinMaxScaler()
        x_plot[:, -n_cont:] = scaler.fit_transform(x_plot[:, -n_cont:])
    
    # Generate plots for each risk type
    for risk in range(1, num_competing_risks + 1):
        print(f"Generating plots for risk type {risk}")
        
        # Generate feature importance plot
        fig, _, top_positive, top_negative = plot_feature_importance(
            model=final_model,
            x_data=x_plot,
            feature_names=feature_names,
            n_top=5,  # Show top 5 positive contributors
            n_bottom=5,  # Show top 5 negative contributors
            risk_idx=risk,  
            figsize=(6, 4),
            output_file=f"figs/nested_cv_feature_importance_risk_{risk}_{args.dataset}.png"
        )
        
        # Get top features for shape function plots
        top_features = top_positive + top_negative
        
        # Generate shape function plots for top features
        fig, _ = plot_coxnam_shape_functions(
            model=final_model,
            X=x_plot,  
            risk_to_plot=risk,
            feature_names=feature_names,  
            top_features=top_features,    
            ncols=5,
            figsize=(12, 6),
            output_file=f"figs/nested_cv_shape_functions_risk_{risk}_{args.dataset}.png"
        )
        plt.close(fig)
    
    print(f"Shape function plots saved to figs/ directory")
    
    # Save metrics to files
    print("\n=== Saving Metrics to Files ===")
    
    # Save summary metrics (formatted table)
    summary_filename = f"nested_cv_summary_metrics_{args.dataset}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary metrics saved to: {summary_filename}")
    
    # Save detailed metrics (all individual fold results)
    detailed_filename = f"nested_cv_detailed_metrics_{args.dataset}.csv"
    detailed_df.to_csv(detailed_filename, index=False)
    print(f"Detailed metrics saved to: {detailed_filename}")
    
    # Save to Excel with multiple sheets
    excel_filename = f"nested_cv_metrics_{args.dataset}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
        
        # Create a metadata sheet
        metadata_df = pd.DataFrame([
            {'Parameter': 'Dataset', 'Value': args.dataset},
            {'Parameter': 'Outer Folds', 'Value': args.outer_folds},
            {'Parameter': 'Inner Folds', 'Value': args.inner_folds},
            {'Parameter': 'Number of Trials', 'Value': args.n_trials},
            {'Parameter': 'Number of Epochs', 'Value': args.num_epochs},
            {'Parameter': 'Batch Size', 'Value': args.batch_size},
            {'Parameter': 'Event Weighting', 'Value': args.event_weighting},
            {'Parameter': 'Scaling', 'Value': args.scaling},
            {'Parameter': 'Random Seed', 'Value': args.seed}
        ])
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"Excel file with multiple sheets saved to: {excel_filename}")
    
    # Save raw metrics dictionary as JSON for complete reproducibility
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, values in all_metrics.items():
        serializable_metrics[key] = [float(v) if not np.isnan(v) else None for v in values]
    
    json_filename = f"nested_cv_raw_metrics_{args.dataset}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'metrics': serializable_metrics,
            'metadata': {
                'dataset': args.dataset,
                'outer_folds': args.outer_folds,
                'inner_folds': args.inner_folds,
                'n_trials': args.n_trials,
                'num_epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'event_weighting': args.event_weighting,
                'scaling': args.scaling,
                'seed': args.seed
            }
        }, f, indent=2)
    
    print(f"Raw metrics (JSON) saved to: {json_filename}")
    
    # Save aggregated best parameters
    param_summary = {}
    for param_name in all_best_params[0].keys():
        param_values = [params[param_name] for params in all_best_params]
        
        # Check if all values are numeric
        if all(isinstance(val, (int, float)) for val in param_values):
            param_summary[param_name] = np.mean(param_values)
        else:
            # For categorical parameters, take the most common
            param_summary[param_name] = max(set(param_values), key=param_values.count)
    
    print(f"\nAggregated best hyperparameters across all outer folds:")
    for param, value in param_summary.items():
        print(f"{param}: {value}")
    
    # Save to YAML
    config_dict = {
        "dataset": args.dataset,
        "scaling": args.scaling,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": param_summary['learning_rate'],
        "l2_reg": param_summary['l2_reg'],
        "patience": args.patience,
        "dropout_rate": param_summary['dropout_rate'],
        "feature_dropout": param_summary['feature_dropout'],
        "hidden_dimensions": ",".join(map(str, [param_summary[f'hidden_dim_{i}'] for i in range(int(param_summary['n_layers']))])),
        "batch_norm": str(param_summary['batch_norm']),
        "event_weighting": args.event_weighting,
        "seed": args.seed,
        "n_folds": args.outer_folds
    }
    
    output_file = f"nested_cv_best_params_{args.dataset}.yaml"
    with open(output_file, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)
    
    print(f"\nBest hyperparameters saved to {output_file}")


if __name__ == "__main__":
    main()