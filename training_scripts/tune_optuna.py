import yaml  
import configargparse

import optuna
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from crisp_nam.models import CrispNamModel
from crisp_nam.utils import (
    weighted_negative_log_likelihood_loss,
    negative_log_likelihood_loss,
    compute_l2_penalty
)
from data_utils import *
from model_utils import EarlyStopping, set_seed
from crisp_nam.utils import predict_absolute_risk, compute_baseline_cif

def parse_args():
    parser = configargparse.ArgumentParser(
        description="Training script for MultiTaskCoxNAM model with Optuna",
        default_config_files=["config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    
    
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
    parser.add_argument("--patience", type=int, default=10, 
                      help="Patience for early stopping")
    
    # Optuna parameters
    parser.add_argument("--n_trials", type=int, default=50, 
                      help="Number of Optuna trials")
    
    # Weight parameters
    parser.add_argument("--event_weighting", type=str, default="none", 
                      choices=["none", "balanced", "custom"],
                      help="Event weighting strategy (none, balanced, custom)")
    parser.add_argument("--custom_event_weights", type=str, default=None,
                      help="Custom weights for events (comma-separated, e.g., '1.0,2.0')")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader=None, num_epochs=500, learning_rate=1e-3, 
                l2_reg=0.01, patience=10, event_weights=None, verbose=True):
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
            current_best = early_stopper.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            if verbose:
                print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if early_stopper.should_stop:
                if verbose:
                    print("Early stopping triggered.")
                break
        elif verbose:
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")
            
    return best_val_loss


def evaluate_model(model, x_val, t_val, e_val, t_train, e_train, abs_risks, times):
    # Structured arrays for training and validation
    survival_train = Surv.from_arrays(e_train != 0, t_train)
    survival_val = Surv.from_arrays(e_val != 0, t_val)

    metrics = defaultdict(list)
    n_events = abs_risks.shape[1]

    # Average AUC across all events and times
    avg_auc = 0
    count = 0

    for k in range(n_events):
        for i, time in enumerate(times):
            risk_preds = abs_risks[:, k, i]

            # ---- AUC ----
            try:
                auc_score, _ = cumulative_dynamic_auc(
                    survival_train,
                    survival_val,
                    risk_preds,
                    times=[time]
                )
                metrics[f"auc_event{k+1}_t{time:.2f}"].append(float(auc_score[0]))
                avg_auc += float(auc_score[0])
                count += 1
            except Exception as ex:
                print(f"[Warning] AUC failed at t={time:.2f}, event={k+1}: {ex}")
                metrics[f"auc_event{k+1}_t{time:.2f}"].append(np.nan)

            # ---- Brier Score ----
            try:
                surv_probs = 1.0 - risk_preds  # Survival prob = 1 - risk
                surv_probs = surv_probs.reshape(-1, 1)  # Shape: (n_samples, n_times)

                _, brier_scores = brier_score(
                    survival_train,
                    survival_val,
                    surv_probs,
                    times=np.array([time])
                )
                metrics[f"brier_event{k+1}_t{time:.2f}"].append(float(brier_scores[0]))
            except Exception as ex:
                print(f"[Warning] Brier failed at t={time:.2f}, event={k+1}: {ex}")
                metrics[f"brier_event{k+1}_t{time:.2f}"].append(np.nan)

            # ---- Time-dependent Concordance Index ----
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
                    tdci_score = tdci_result  # fallback
                metrics[f"tdci_event{k+1}_t{time:.2f}"].append(float(tdci_score))
            except Exception as ex:
                print(f"[Warning] td-CI failed at t={time:.2f}, event={k+1}: {ex}")
                metrics[f"tdci_event{k+1}_t{time:.2f}"].append(np.nan)
    
    # Calculate average AUC if we have valid measurements
    if count > 0:
        avg_auc = avg_auc / count
    else:
        avg_auc = 0.0
        
    return metrics, avg_auc


def display_metrics_table(metrics_dict, quantiles=[0.25, 0.5, 0.75]):
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


def objective(trial, x, t, e, x_train, t_train, e_train, x_val, t_val, e_val, feature_names, n_cont,
              num_competing_risks, device, args, event_weights=None):
    
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.8)
    feature_dropout = trial.suggest_float('feature_dropout', 0.0, 0.5)
    
    # For hidden_dimensions
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_dimensions = []
    for i in range(n_layers):
        hidden_dimensions.append(trial.suggest_categorical(f'hidden_dim_{i}', [32, 64, 128, 256]))
    
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    
    # Create data loaders
    train_dataset = SurvivalDataset(x_train, t_train, e_train)
    val_dataset = SurvivalDataset(x_val, t_val, e_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=4, pin_memory=True)
    
    # Initialize model with the hyperparameters
    model = CrispNamModel(
        num_features=x.shape[1],
        num_competing_risks=num_competing_risks,
        hidden_sizes=hidden_dimensions,
        dropout_rate=dropout_rate,
        feature_dropout=feature_dropout,
        batch_norm=batch_norm
    ).to(device)
    
    # Train model
    best_val_loss = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=args.num_epochs, 
        learning_rate=learning_rate,
        l2_reg=l2_reg, 
        patience=args.patience,
        event_weights=event_weights,
        verbose=False  # Set to False to reduce output during Optuna trials
    )
    
    # We can also evaluate with metrics like AUC
    # If needed, uncomment this code to optimize for AUC instead of validation loss
    """
    # Calculate evaluation times
    safe_max = 0.99 * np.max(t_train)
    eval_times = np.quantile(t_train[t_train <= safe_max], [0.25, 0.5, 0.75])
    
    # Calculate baseline CIFs
    baseline_cifs = {k: compute_baseline_cif(t_train, e_train, eval_times, k + 1) 
                    for k in range(num_competing_risks)}
    
    # Predict and evaluate
    abs_risks = predict_absolute_risk(model, x_val, baseline_cifs, eval_times, device=device)
    _, avg_auc = evaluate_model(model, x_val, t_val, e_val, t_train, e_train, abs_risks, eval_times)
    
    return avg_auc  # Maximize AUC
    """
    
    # For now, we'll optimize for validation loss (minimize)
    return best_val_loss  # Minimize validation loss


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
    
    # Scale the data
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

    # Compute number of competing risks
    num_competing_risks = len(np.unique(e)) - 1  # Excluding censoring (0)
    device = ("cuda" if torch.cuda.is_available() else "cpu")



    
    # Split data into train and validation sets
    x_train, x_val, t_train, t_val, e_train, e_val = train_test_split(
        x, t, e, test_size=0.2, random_state=args.seed, stratify=e
    )
    
    # Initialize event weights
    event_weights = None
    
    if args.event_weighting != "none":
        if args.event_weighting == "balanced":
            # Compute balanced weights (inverse of class frequencies)
            event_counts = np.zeros(num_competing_risks)
            for k in range(1, num_competing_risks + 1):
                event_counts[k-1] = np.sum(e_train == k)
            
            # Avoid division by zero
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
        
        # Convert to torch tensor
        event_weights = torch.tensor(event_weights, dtype=torch.float32, device=device)
    
    # Create Optuna study
    study = optuna.create_study(direction="minimize")  # Minimize validation loss
    
    # Run optimization
    print("\n=== Starting Optuna hyperparameter optimization ===")
    study.optimize(lambda trial: objective(
        trial, x, t, e, x_train, t_train, e_train, x_val, t_val, e_val,
        feature_names, n_cont, num_competing_risks, device, args, event_weights
    ), n_trials=args.n_trials)
    
    # Get best parameters
    best_params = study.best_params
    print("\n=== Best Hyperparameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")


    # Extract hidden dimensions from best params
    n_layers = best_params.pop('n_layers')
    hidden_dimensions = []
    for i in range(n_layers):
        hidden_dimensions.append(best_params.pop(f'hidden_dim_{i}'))
    
    # Save best hyperparameters to YAML file
    config_dict = {
        "dataset": args.dataset,
        "scaling": args.scaling,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": best_params['learning_rate'],
        "l2_reg": best_params['l2_reg'],
        "patience": args.patience,
        "dropout_rate": best_params['dropout_rate'],
        "feature_dropout": best_params['feature_dropout'],
        "hidden_dimensions": ",".join(map(str, hidden_dimensions)),  # Format as comma-separated string
        "batch_norm": str(best_params['batch_norm']),  # Convert to string "True" or "False"
        "event_weighting": args.event_weighting,
        "seed": args.seed,
        "n_folds": 5  # Default for k-fold script
    }
    
    # Add custom event weights if applicable
    if args.event_weighting == "custom" and args.custom_event_weights is not None:
        config_dict["custom_event_weights"] = args.custom_event_weights
    
    # Save to YAML file
    output_file = f"best_params_{args.dataset}.yaml"
    with open(output_file, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)
    
    print(f"\nBest hyperparameters saved to {output_file}")
    
    # Train final model with best parameters
    print("\n=== Training final model with best parameters ===")
    
    # Create data loaders with all training data
    train_dataset = SurvivalDataset(x_train, t_train, e_train)
    val_dataset = SurvivalDataset(x_val, t_val, e_val)
    
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=4, pin_memory=True)
    
    
    # Initialize final model with best parameters
    final_model = CrispNamModel(
        num_features=x.shape[1],
        num_competing_risks=num_competing_risks,
        hidden_sizes=hidden_dimensions,
        dropout_rate=best_params['dropout_rate'],
        feature_dropout=best_params['feature_dropout'],
        batch_norm=best_params['batch_norm']
    ).to(device)
    
    # Train final model
    train_model(
        final_model, 
        train_loader, 
        val_loader, 
        num_epochs=args.num_epochs, 
        learning_rate=best_params['learning_rate'],
        l2_reg=best_params['l2_reg'], 
        patience=args.patience,
        event_weights=event_weights,
        verbose=True
    )
    
    # Evaluate final model
    # Calculate evaluation times
    safe_max = 0.99 * np.max(t_train)
    eval_times = np.quantile(t_train[t_train <= safe_max], [0.25, 0.5, 0.75])
    
    print(f"Evaluation times: {eval_times}")
    
    # Calculate baseline CIFs
    baseline_cifs = {k: compute_baseline_cif(t_train, e_train, eval_times, k + 1) 
                    for k in range(num_competing_risks)}
    
    # Predict and evaluate
    abs_risks = predict_absolute_risk(final_model, x_val, baseline_cifs, eval_times, device=device)
    final_metrics, _ = evaluate_model(final_model, x_val, t_val, e_val, t_train, e_train, abs_risks, eval_times)
    
    # Display metrics
    display_metrics_table(final_metrics)
    
    

if __name__ == "__main__":
    main()