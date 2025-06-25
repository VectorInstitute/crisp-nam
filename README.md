# CRISP-NAM: Competing Risks Interpretable Survival Prediction with Neural Additive Models

CRISP-NAM (Competing Risks Interpretable Survival Prediction with Neural Additive Models), an interpretable neural additive model for 
competing risks survival analysis which extends the neural additive architecture to model cause-specific hazards while preserving feature-level interpretability.

## Requirements
Python >=3.10

## Repository Structure

```
crisp_nam
├── models
    ├── __init__.py
    ├── crisp_nam_model.py
    └── deephit_model.py
├── metrics
    ├── __init__.py
    ├── calibration.py
    ├── discrimination.py
    └── ipcw.py
└── utils
    ├── __init__.py
    ├── loss.py
    ├── plotting.py
    └── risk_cif.py

results
├── best_params                                  #Best parameters for dataset and model combination.
    ├── best_params_framingham_deephit.yaml
    ├── best_params_framingham.yaml
    ├── best_params_pbc_deephit.yaml
    ├── best_params_pbc.yaml
    ├── best_params_support.yaml
    ├── best_params_support2_deephit.yaml
    ├── best_params_synthetic_deephit.yaml
    └── best_params_synthetic.yaml
├── plots
    ├── feature_importance_risk_new_1_framingham.png
    ├── feature_importance_risk_new_2_framingham.png
    ├── shape_functions_top_features_risk1_framingham.png
    └── shape_functions_top_features_risk2_framingham.png

datasets                                        #Directory with csv files
│   ├── framingham.csv
│   ├── metabric
│   │   ├── cleaned_features_final.csv
│   │   └── label.csv
│   ├── pbc2.csv
│   ├── support2.csv
│   ├── synthetic_comprisk.csv

├── training_scripts                        #Training Scripts
│   ├── config.yaml
│   ├── model_utils.py
│   ├── train_deephit_cuda.py
│   ├── train_deephit.py
│   ├── train_nfg.py
│   ├── train.py
│   ├── tune_optuna_optimized.py
│   └── tune_optuna.py
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

> [!NOTE]
> For `uv` installation, please visit follow instructions in their [official page](https://docs.astral.sh/uv/getting-started/installation/).

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.
