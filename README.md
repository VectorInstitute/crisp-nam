# CRISP-NAM: Competing Risks Interpretable Survival Prediction with Neural Additive Models

CRISP-NAM (Competing Risks Interpretable Survival Prediction with Neural Additive Models), an interpretable neural additive model for competing risks survival analysis which extends the neural additive architecture to model cause-specific hazards while preserving feature-level interpretability.

## Overview

This package provides a comprehensive framework for competing risks survival analysis with interpretable neural additive models. CRISP-NAM combines the predictive power of deep learning with interpretability through feature-level shape functions, making it suitable for clinical and biomedical applications where understanding feature contributions is crucial.

### Key Features

- **Interpretable Architecture**: Neural additive models that provide feature-level interpretability through shape functions
- **Competing Risks Support**: Native handling of multiple competing events in survival analysis
- **Comprehensive Evaluation**: Nested cross-validation with robust performance metrics (AUC, Brier Score, Time-dependent C-index)
- **Hyperparameter Optimization**: Automated tuning using Optuna with customizable search spaces
- **Rich Visualizations**: Automated generation of feature importance plots and shape function visualizations
- **Multiple Training Modes**: Standard training, hyperparameter tuning, and nested cross-validation
- **Baseline Comparisons**: DeepHit implementation for benchmarking against state-of-the-art methods

## Requirements

Python >=3.10

## Install the package

```bash
pip install crisp-nam
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
pip install -e .
```
via `uv`
```bash
cd crisp-nam
uv sync
```
## Research details

For more details regarding the research work, please refer to `datasets.md` and `training.md` within the project repository.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## Citation

If you use our package, kindly acknowledge by citing our research.
```
@inproceedings{ramachandram2025crispnam,
    title={CRISP-NAM: Competing Risks Interpretable Survival Prediction with Neural Additive Models},
    author={Ramachandram, Dhanesh and Raval, Ananya},
    booktitle={EXPLIMED 2025 - Second Workshop on Explainable AI for the Medical Domain},
    year={2025}
}
```

## License

This project is licensed under the MIT License.

## What's New

> [!IMPORTANT]
> The bugs listed below were discovered after the paper was published. We are listing them here to be transparent about how they affect the results. Some of the fixes change numbers reported in the paper, most notably the Brier scores and the PBC and SUPPORT results.
>
> We strongly recommend using the most recent version of this code. The version tied to the publication is tagged [`v0.1.2`](https://github.com/VectorInstitute/crisp-nam/tree/v0.1.2), and is kept only for reproducing the published results.

These changes are not yet in the PyPI release; install from source to use them.

### 2026-10-21

- Fixed the time-dependent AUC failing when a test fold contains only one event of interest.

### 2026-10-20

- **Predicted risks no longer depend on an arbitrary score offset.** The training loss is unaffected by adding a constant to a cause's risk scores, but predicted probabilities were not, so Brier scores could vary widely from fold to fold. Predictions now use a baseline hazard fitted to the model's own training scores. `predict_absolute_risk`, `predict_cif` and `compute_baseline_cumulative_hazard` now take this baseline, created with `compute_breslow_baseline`.
- Fixed cause-specific cumulative incidence occasionally exceeding 1 for high-risk subjects.
- The Fine-Gray model now applies censoring weights to subjects with a competing event, as in the original Fine-Gray method. Without them, cumulative incidence was underestimated when many subjects were censored.
- Fixed the time-dependent AUC weighting controls by their own censoring probability, which inflated AUC values.
- Added a cause-specific competing-risks C-index (CRCI), reported alongside the existing time-dependent C-index. The existing C-index measures ranking against time to *any* event.
- Hyperparameter search is now reproducible: the Optuna sampler is seeded from `--seed`.
- Fixed crashes in nested cross-validation when saving hyperparameters, in `tune_optuna_optimized.py`'s final evaluation, and in `train_deephit_cuda.py` on small datasets.
- Added a `--device {auto,cpu,cuda,mps}` option to the training and tuning scripts.

### 2026-07-21

- Added a `--risk_model` option to train with either cause-specific hazards (the default) or Fine-Gray subdistribution hazards.
- Added cumulative hazard estimation and plotting alongside the cumulative incidence plots.
- The competing-risks loss is now vectorised, so training is much faster and gives the same results.
- PBC now uses one record per patient (312 patients). Previously every clinic visit was treated as a separate patient.
- SUPPORT now labels all deaths of patients with cancer as risk 1. Previously, about 950 of these deaths were counted as other-cause deaths.
- Missing-value imputation and feature scaling are now fitted on the training data of each fold only, so test data no longer influences them.
- Removed unused duplicate dataset loaders from `datasets/`.
