# CRISP-NAM: Competing Risks Interpretable Survival Prediction with Neural Additive Models

CRISP-NAM (Competing Risks Interpretable Survival Prediction with Neural Additive Models), an interpretable neural additive model for 
competing risks survival analysis which extends the neural additive architecture to model cause-specific hazards while preserving feature-level interpretability.

## Project Directory Structure

```text
.
├── best_params
│   ├── best_params_framingham_deephit.yaml
│   ├── best_params_framingham.yaml
│   ├── best_params_pbc_deephit.yaml
│   ├── best_params_pbc.yaml
│   ├── best_params_support.yaml
│   ├── best_params_support2_deephit.yaml
│   ├── best_params_synthetic_deephit.yaml
│   └── best_params_synthetic.yaml
├── datasets
│   ├── framingham_dataset.py
│   ├── framingham.csv
│   ├── metabric
│   │   ├── cleaned_features_final.csv
│   │   └── label.csv
│   ├── pbc_dataset.py
│   ├── pbc2.csv
│   ├── support_dataset.py
│   ├── support2.csv
│   ├── SurvivalDataset.py
│   ├── synthetic_comprisk.csv
│   └── synthetic_dataset.py
├── figs
│   ├── feature_importance_risk_new_1_framingham.png
│   ├── feature_importance_risk_new_2_framingham.png
│   ├── shape_functions_top_features_risk1_framingham.png
│   └── shape_functions_top_features_risk2_framingham.png
├── metrics
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── calibration.cpython-310.pyc
│   │   ├── discrimination.cpython-310.pyc
│   │   └── ipcw.cpython-310.pyc
│   ├── calibration.py
│   ├── discrimination.py
│   └── ipcw.py
├── model
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── coxnam_competing.cpython-310.pyc
│   │   ├── crisp_nam_model.cpython-310.pyc
│   │   └── deephit_model.cpython-310.pyc
│   ├── crisp_nam_model.py
│   └── deephit_model.py
├── nfg
│   ├── __init__.py
│   ├── datasets.py
│   ├── dsm
│   │   ├── __init__.py
│   │   ├── contrib
│   │   │   ├── dcm_api.py
│   │   │   ├── dcm_torch.py
│   │   │   └── dcm_utilities.py
│   │   ├── dsm_api.py
│   │   ├── dsm_torch.py
│   │   ├── losses.py
│   │   └── utilities.py
│   ├── losses.py
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── discrimination.py
│   │   └── utils.py
│   ├── nfg_api.py
│   ├── nfg_torch.py
│   └── utilities.py
├── pyproject.toml
├── README.md
├── train
│   ├── config.yaml
│   ├── train_deephit_cuda.py
│   ├── train_deephit.py
│   ├── train_nfg.py
│   ├── train.py
│   ├── tune_optuna_optimized.py
│   └── tune_optuna.py
└── utils
    ├── __init__.py
    ├── loss.py
    ├── plotting.py
    └── risk_cif.py

```

## Getting Started

```bash
git clone https://github.com/yourusername/crisp-nam.git

cd crisp-nam
# Add setup or installation instructions here
## Setting Up Python Dependencies

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management:

```bash
# Install uv if you don't have it
pip install uv

# Create a virtual environment
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip install -r pyproject.toml
```

## Usage

To run examples:

```
python -m train.train --dataset framingham

```


## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.
