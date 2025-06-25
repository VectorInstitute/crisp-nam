import yaml
import configargparse
from collections import defaultdict

import torch
import optuna
import numpy as np
import torch.nn as nn
from sksurv.util import Surv
import torch.optim as optim
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sksurv.metrics import (
    cumulative_dynamic_auc,
    brier_score,
    concordance_index_ipcw
)

from data_utils import *
from model_utils import EarlyStopping, set_seed
from crisp_nam.models import CrispNamModel
from crisp_nam.utils import (
    weighted_negative_log_likelihood_loss,
    negative_log_likelihood_loss,
    compute_l2_penalty,
)
from crisp_nam.utils import (
    predict_absolute_risk,
    compute_baseline_cif
)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs: int = 500,
    lr: float = 1e-3,
    l2_reg: float = 1e-3,
    patience: int = 10,
    event_weights=None,
    verbose: bool = True,
) -> float:
    device = next(model.parameters()).device
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    early_stop = EarlyStopping(patience)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, tb, eb in train_loader:
            xb = xb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            eb = eb.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                scores, _ = model(xb)
                if event_weights is not None:
                    loss = weighted_negative_log_likelihood_loss(
                        scores, tb, eb, model.num_competing_risks, event_weights)
                else:
                    loss = negative_log_likelihood_loss(
                        scores, tb, eb, model.num_competing_risks)
                loss = loss + compute_l2_penalty(model) * l2_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, tb, eb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    tb = tb.to(device, non_blocking=True)
                    eb = eb.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        scores, _ = model(xb)
                        if event_weights is not None:
                            loss = weighted_negative_log_likelihood_loss(
                                scores, tb, eb, model.num_competing_risks, event_weights)
                        else:
                            loss = negative_log_likelihood_loss(
                                scores, tb, eb, model.num_competing_risks)
                        loss = loss + compute_l2_penalty(model) * l2_reg
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            if verbose:
                print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            if early_stop.step(avg_val_loss):
                if verbose: print("Early stopping.")
                return avg_val_loss
        else:
            if verbose:
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

    return avg_val_loss if val_loader is not None else avg_train_loss


def objective(
    trial,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    num_features: int,
    num_competing_risks: int,
    device: torch.device,
    args,
    event_weights,
):

    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    l2 = trial.suggest_float('l2_reg', 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float('dropout_rate', 0.0, 0.8)
    feat_drop = trial.suggest_float('feature_dropout', 0.0, 0.5)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_dims = [trial.suggest_categorical(f'hidden_dim_{i}', [32,64,128,256])
                   for i in range(n_layers)]
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    model = CrispNamModel(
        num_features=num_features,
        num_competing_risks=num_competing_risks,
        hidden_sizes=hidden_dims,
        dropout_rate=dropout,
        feature_dropout=feat_drop,
        batch_norm=batch_norm,
    ).to(device)

    val_loss = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        lr=lr,
        l2_reg=l2,
        patience=args.patience,
        event_weights=event_weights,
        verbose=False,
    )
    return val_loss


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    t_val: np.ndarray,
    e_val: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    baseline_cifs: dict,
    eval_times: np.ndarray,
    device: torch.device,
):
    surv_train = Surv.from_arrays(e_train != 0, t_train)
    surv_val   = Surv.from_arrays(e_val   != 0, t_val)
    abs_risk = predict_absolute_risk(model, X, baseline_cifs, eval_times, device=device)

    metrics = defaultdict(list)
    for k_idx, t0 in enumerate(eval_times, start=1):
        preds = abs_risk[:, k_idx-1, :]
        for ti_idx, ti in enumerate(eval_times):
            r = preds[:, ti_idx]
            try:
                auc, _ = cumulative_dynamic_auc(surv_train, surv_val, r, times=[ti])
                metrics[f"auc_event{k_idx}_t{ti:.2f}"].append(float(auc[0]))
            except:
                metrics[f"auc_event{k_idx}_t{ti:.2f}"].append(np.nan)
            try:
                _, bsc = brier_score(surv_train, surv_val, 1 - r.reshape(-1,1), times=np.array([ti]))
                metrics[f"brier_event{k_idx}_t{ti:.2f}"].append(float(bsc[0]))
            except:
                metrics[f"brier_event{k_idx}_t{ti:.2f}"].append(np.nan)
            try:
                ci = concordance_index_ipcw(surv_train, surv_val, estimate=r, tau=ti)
                td = ci[0] if isinstance(ci, tuple) else ci
                metrics[f"tdci_event{k_idx}_t{ti:.2f}"].append(float(td))
            except:
                metrics[f"tdci_event{k_idx}_t{ti:.2f}"].append(np.nan)
    return metrics


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Training script for CrispNamModel with Optuna",
        default_config_files=["config.yml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument("--dataset", type=str, default="framingham",
                        choices=["framingham","support","pbc","synthetic"] )
    parser.add_argument("--scaling", type=str, default="standard",
                        choices=["minmax","standard","none"] )
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--event_weighting", type=str, default="none",
                        choices=["none","balanced","custom"] )
    parser.add_argument("--custom_event_weights", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# ----------------------- Main ----------------------- #
def main():
    args = parse_args()
    print(args)
    set_seed(args.seed)

    # Load dataset
    if args.dataset == 'framingham':
        x, t, e, feature_names, n_cont, _ = load_framingham()
    elif args.dataset == 'support':
        x, t, e, feature_names, n_cont, _ = load_support_dataset()
    elif args.dataset == 'pbc':
        x, t, e, feature_names, n_cont, _ = load_pbc2_dataset()
    else:
        x, t, e, feature_names, n_cont, _ = load_synthetic_dataset()

    # Scale continuous features
    if args.scaling == 'standard':
        x[:, -n_cont:] = StandardScaler().fit_transform(x[:, -n_cont:])
    elif args.scaling == 'minmax':
        x[:, -n_cont:] = MinMaxScaler().fit_transform(x[:, -n_cont:])

    # Bulk convert to torch tensors on CPU
    X = torch.from_numpy(x.astype('float32'))
    T = torch.from_numpy(t.astype('float32'))
    E = torch.from_numpy(e.astype('int64'))

    # Train/validation split (fixed)
    idx_train, idx_val = train_test_split(
        np.arange(len(e)), test_size=0.2, random_state=args.seed, stratify=e
    )
    train_ds = TensorDataset(X[idx_train], T[idx_train], E[idx_train])
    val_ds   = TensorDataset(X[idx_val],   T[idx_val],   E[idx_val])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_competing_risks = len(np.unique(e)) - 1

    # Event weighting
    event_weights = None
    if args.event_weighting == 'balanced':
        counts = np.array([(e[idx_train]==k).sum() for k in range(1, num_competing_risks+1)])
        counts = np.maximum(counts, 1)
        w = 1.0 / counts
        w *= num_competing_risks / w.sum()
        event_weights = torch.from_numpy(w.astype('float32')).to(device)
    elif args.event_weighting == 'custom':
        w = np.array(list(map(float, args.custom_event_weights.split(','))))
        event_weights = torch.from_numpy(w.astype('float32')).to(device)

    # Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(
            trial, train_ds, val_ds,
            x.shape[1], num_competing_risks,
            device, args, event_weights
        ),
        n_trials=args.n_trials
    )

    best = study.best_params
    print("Best hyperparameters:", best)
    with open(f"best_params_{args.dataset}.yaml", 'w') as f:
        yaml.dump(best, f)

    # Final model training with best params
    n_layers = best['n_layers']
    hidden_dims = [best[f'hidden_dim_{i}'] for i in range(n_layers)]
    final_model = CrispNamModel(
        num_features=x.shape[1],
        num_competing_risks=num_competing_risks,
        hidden_sizes=hidden_dims,
        dropout_rate=best['dropout_rate'],
        feature_dropout=best['feature_dropout'],
        batch_norm=best['batch_norm'],
    ).to(device)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    final_train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    final_val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    train_model(
        final_model,
        final_train_loader,
        val_loader=final_val_loader,
        num_epochs=args.num_epochs,
        lr=best['learning_rate'],
        l2_reg=best['l2_reg'],
        patience=args.patience,
        event_weights=event_weights,
        verbose=True,
    )

    # Evaluation
    safe_max = 0.99 * np.max(t[idx_train])
    eval_times = np.quantile(t[idx_train][t[idx_train] <= safe_max], [0.25, 0.5, 0.75])
    baseline_cifs = {
        k: compute_baseline_cif(
            t[idx_train], e[idx_train], eval_times, k+1
        ) for k in range(num_competing_risks)
    }
    metrics = evaluate_model(
        final_model,
        x[idx_val], t[idx_val], e[idx_val],
        t[idx_train], e[idx_train],
        baseline_cifs, eval_times, device
    )

    # Reporting
    rows = []
    for k in range(1, num_competing_risks+1):
        row = {'Risk': f'Type {k}'}
        for metric in ['auc', 'tdci', 'brier']:
            for q, ti in zip([0.25, 0.5, 0.75], eval_times):
                key = f"{metric}_event{k}_t{ti:.2f}"
                vals = np.array(metrics[key], dtype=float)
                row[f"{metric.upper()}_q{q:.2f}"] = f"{np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}"
        rows.append(row)
    print("\n=== Performance ===")
    print(tabulate(rows, headers='keys', tablefmt='pretty', showindex=False))

if __name__ == '__main__':
    main()
