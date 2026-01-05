# CRISP-NAM: Competing Risks Interpretable Survival Prediction with Neural Additive Models

CRISP-NAM (Competing Risks Interpretable Survival Prediction with Neural Additive Models), an interpretable neural additive model for competing risks survival analysis which extends the neural additive architecture to model cause-specific hazards while preserving feature-level interpretability.

## Overview

This repository provides a comprehensive framework for competing risks survival analysis with interpretable neural additive models. CRISP-NAM combines the predictive power of deep learning with interpretability through feature-level shape functions, making it suitable for clinical and biomedical applications where understanding feature contributions is crucial.

### Key Features

- **Interpretable Architecture**: Neural additive models that provide feature-level interpretability through shape functions
- **Competing Risks Support**: Native handling of multiple competing events in survival analysis
- **Comprehensive Evaluation**: Nested cross-validation with robust performance metrics (AUC, Brier Score, Time-dependent C-index)
- **Hyperparameter Optimization**: Automated tuning using Optuna with customizable search spaces
- **Rich Visualizations**: Automated generation of feature importance plots and shape function visualizations
- **Multiple Training Modes**: Standard training, hyperparameter tuning, and nested cross-validation
- **Baseline Comparisons**: DeepHit implementation for benchmarking against state-of-the-art methods

### Available Datasets

The repository includes four well-established survival analysis datasets:

1. **Framingham Heart Study**: Cardiovascular disease prediction with competing events (CVD vs. death)
   - Features: Demographics, clinical measurements, lifestyle factors
   - Events: Cardiovascular disease, death from other causes

2. **PBC (Primary Biliary Cirrhosis)**: Liver disease progression study
   - Features: Clinical laboratory values, demographic information
   - Events: Death, liver transplantation

3. **SUPPORT**: Study to understand prognoses and preferences for outcomes
   - Features: Comprehensive clinical and demographic variables
   - Events: Cancer death, non-cancer death

4. **Synthetic Dataset**: Controlled simulation for method validation
   - Features: Simulated clinical variables with known ground truth
   - Events: Multiple competing risks with controllable hazard functions

All datasets come with preprocessing pipelines that handle missing values, feature encoding, and proper train/test splitting to prevent data leakage.

### Training Scripts

The repository provides several specialized training scripts:

- **`train.py`**: Standard model training with cross-validation and comprehensive evaluation
- **`train_nested_cv.py`**: Robust nested cross-validation for unbiased performance estimation
- **`tune_optuna.py`**: Hyperparameter optimization using Optuna's advanced algorithms
- **`train_deephit.py`**: DeepHit baseline implementation for comparative studies

Each script supports extensive configuration through command-line arguments and YAML config files, enabling reproducible experiments and easy parameter sweeps.

## Requirements
Python >=3.10

## Repository Structure

```
crisp_nam/
├── crisp_nam/                              # Main package
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── discrimination.py
│   │   └── ipcw.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── crisp_nam_model.py
│   │   └── deephit_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── loss.py
│   │   ├── plotting.py
│   │   └── risk_cif.py
│   └── __init__.py
├── data_utils/                             # Data utilities
│   ├── __init__.py
│   ├── load_datasets.py
│   └── survival_datasets.py
├── datasets/                               # Dataset files and loaders
│   ├── metabric/
│   │   ├── cleaned_features_final.csv
│   │   └── label.csv
│   ├── framingham_dataset.py
│   ├── framingham.csv
│   ├── pbc_dataset.py
│   ├── pbc2.csv
│   ├── support_dataset.py
│   ├── support2.csv
│   ├── SurvivalDataset.py
│   ├── synthetic_comprisk.csv
│   └── synthetic_dataset.py
├── results/                                # Results and outputs
│   ├── best_params/                        # Best parameters for dataset and model combinations
│   │   ├── best_params_framingham_deephit.yaml
│   │   ├── best_params_framingham.yaml
│   │   ├── best_params_pbc_deephit.yaml
│   │   ├── best_params_pbc.yaml
│   │   ├── best_params_support.yaml
│   │   ├── best_params_support2_deephit.yaml
│   │   ├── best_params_synthetic_deephit.yaml
│   │   └── best_params_synthetic.yaml
│   ├── logs/                               # Nested CV results and logs
│   │   ├── nested_cv_best_params_*.yaml
│   │   ├── nested_cv_detailed_metrics_*.csv
│   │   ├── nested_cv_metrics_*.xlsx
│   │   ├── nested_cv_raw_metrics_*.json
│   │   └── nested_cv_summary_metrics_*.csv
│   └── plots/                              # Generated plots
│       ├── nested_cv_feature_importance_risk_*_*.png
│       └── nested_cv_shape_functions_risk_*_*.png
├── training_scripts/                       # Training scripts
│   ├── config.yaml
│   ├── model_utils.py
│   ├── train_deephit_cuda.py
│   ├── train_deephit.py
│   ├── train_nested_cv.py                  # Nested cross-validation script
│   ├── train.py
│   ├── tune_optuna_optimized.py
│   └── tune_optuna.py
└── utils/                                  # Legacy utils (duplicate of crisp_nam/utils)
    ├── __init__.py
    ├── loss.py
    ├── plotting.py
    └── risk_cif.py
```

## Install from source

1. Clone the repository

```bash
git clone git@github.com:VectorInstitute/crisp-nam.git
```

2. Install

via `pip`

```bash
cd crisp-nam
pip install -e
```
via `uv`
```bash
cd crisp-nam
uv sync
```

## Running training scripts

1. Modify training parameters in `training_scripts/train.py`
   OR
   Run either of following commands to see CLI arguments for passing training parameters:

   ```bash
   python training_scripts/train.py --help
   ```

   ```bash
   uv run training_scripts/train.py --help
   ```

2. Run the training script

   via `python`
   ```bash
   source .venv/bin/activate
   python training_scripts/train.py --dataset framingham
   ```

   via `uv`
   ```bash
   uv run training_scripts/train.py --dataset framingham
   ```

## Running Nested Cross-Validation

The nested cross-validation script performs robust model evaluation with hyperparameter optimization using inner and outer cross-validation loops. It automatically generates performance metrics, feature importance plots, and shape function visualizations.

### Basic Usage

```bash
# Using python
python training_scripts/train_nested_cv.py --dataset framingham

# Using uv
uv run training_scripts/train_nested_cv.py --dataset framingham
```

### Configuration Parameters

All parameters can be passed via command line or specified in a YAML config file:

#### Dataset Configuration
- `--dataset` (str): Dataset to use (choices: `framingham`, `support`, `pbc`, `synthetic`, default: `framingham`)
- `--scaling` (str): Data scaling method for continuous features (choices: `minmax`, `standard`, `none`, default: `standard`)

#### Training Parameters
- `--num_epochs` (int): Number of training epochs (default: `250`)
- `--batch_size` (int): Batch size for training (default: `512`)
- `--patience` (int): Patience for early stopping (default: `10`)

#### Cross-Validation Configuration
- `--outer_folds` (int): Number of outer CV folds (default: `5`)
- `--inner_folds` (int): Number of inner CV folds for hyperparameter tuning (default: `3`)
- `--n_trials` (int): Number of Optuna trials per inner fold (default: `20`)

#### Event Weighting
- `--event_weighting` (str): Event weighting strategy (choices: `none`, `balanced`, `custom`, default: `none`)
- `--custom_event_weights` (str): Custom weights for events (comma-separated, default: `None`)

#### Other Parameters
- `--seed` (int): Random seed for reproducibility (default: `42`)
- `--config` (str): Path to YAML config file (default: looks for `config.yaml`)

### Examples

#### Basic nested CV with default parameters:
```bash
python training_scripts/train_nested_cv.py --dataset pbc
```

#### Customized nested CV with specific parameters:
```bash
python training_scripts/train_nested_cv.py \
    --dataset support \
    --outer_folds 10 \
    --inner_folds 5 \
    --n_trials 50 \
    --num_epochs 500 \
    --event_weighting balanced \
    --scaling minmax \
    --seed 123
```

#### Using a config file:
```bash
python training_scripts/train_nested_cv.py --config my_config.yaml
```

### Output Files

The script generates several output files in the current directory:

#### Performance Metrics
- `nested_cv_summary_metrics_{dataset}.csv`: Summary table with mean ± std metrics
- `nested_cv_detailed_metrics_{dataset}.csv`: Detailed results for each fold
- `nested_cv_metrics_{dataset}.xlsx`: Excel file with multiple sheets (Summary, Detailed, Metadata)
- `nested_cv_raw_metrics_{dataset}.json`: Raw metrics dictionary for reproducibility

#### Model Configuration
- `nested_cv_best_params_{dataset}.yaml`: Aggregated best hyperparameters across all folds

#### Visualizations
Results are saved to `results/plots/`:
- `nested_cv_feature_importance_risk_{risk}_{dataset}.png`: Feature importance plots
- `nested_cv_shape_functions_risk_{risk}_{dataset}.png`: Shape function plots for top features

### Evaluation Metrics

The script computes the following metrics at different time quantiles (25%, 50%, 75%):

- **AUC (Area Under the ROC Curve)**: Time-dependent AUC for discrimination
  - 0.5 = random, >0.7 = good, >0.8 = excellent
- **TDCI (Time-Dependent Concordance Index)**: Harrell's C-index adapted for competing risks
  - 0.5 = random, >0.7 = good, >0.8 = excellent
- **Brier Score**: Calibration metric measuring prediction accuracy
  - 0 = perfect, <0.25 = good, >0.25 = poor

> [!NOTE]
> For `uv` installation, please visit follow instructions in their [official page](https://docs.astral.sh/uv/getting-started/installation/).

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.