# CRISP-NAM: Competing Risks Interpretable Survival Prediction with Neural Additive Models

CRISP-NAM (Competing Risks Interpretable Survival Prediction with Neural Additive Models), an interpretable neural additive model for competing risks survival analysis which extends the neural additive architecture to model cause-specific hazards while preserving feature-level interpretability.

## Overview

This package provides a comprehensive framework for competing risks survival analysis with interpretable neural additive models. CRISP-NAM combines the predictive power of deep learning with interpretability through feature-level shape functions, making it suitable for clinical and biomedical applications where understanding feature contributions is crucial.

### Key Features

- **Interpretable Architecture**: Neural additive models that provide feature-level interpretability through shape functions
- **Competing Risks Support**: Native handling of multiple competing events in survival analysis
- **Selectable Risk Model**: Choose between cause-specific hazards and Fine-Gray subdistribution hazards for competing risks training
- **Cumulative Hazard Estimation**: Compute and plot cause-specific cumulative hazard functions alongside cumulative incidence
- **Comprehensive Evaluation**: Nested cross-validation with robust performance metrics (AUC, Brier Score, Time-dependent C-index)
- **Hyperparameter Optimization**: Automated tuning using Optuna with customizable search spaces
- **Rich Visualizations**: Automated generation of feature importance plots and shape function visualizations
- **Multiple Training Modes**: Standard training, hyperparameter tuning, and nested cross-validation
- **Baseline Comparisons**: DeepHit implementation for benchmarking against state-of-the-art methods

## Recent Updates

### Bug Fixes
- Fixed PBC2 dataset loader treating every longitudinal visit as an independent sample instead of deduplicating to one row per patient.
- Fixed SUPPORT dataset loader silently misclassifying ~14% of cancer patients as non-cancer deaths due to a fragile string-matching heuristic.
- Fixed PBC2 categorical imputation being a silent no-op that never actually filled missing values.
- Fixed continuous-feature imputation being fit on the full dataset before cross-validation splitting, which leaked validation-fold statistics into training (now fit per-fold, mirroring the existing per-fold feature scaling).
- Fixed `tune_optuna_optimized.py` scaling continuous features on the full dataset before its train/validation split (same leakage issue, now fit per-split).
- Removed unused duplicate dataset-loading files under `datasets/` that were dead code and contained the same bugs listed above.
- Fixed absolute-risk (CIF) prediction depending on an arbitrary offset in the model's risk scores. The Cox partial likelihood cannot identify a constant added to a cause's scores, but `predict_absolute_risk` combined the raw scores with a population-level baseline CIF, so predicted probabilities, and therefore Brier scores, varied wildly between folds (e.g. 0.05 in one fold, 0.8 in another). Predictions now use a Breslow baseline fitted to the model's own training scores (`compute_breslow_baseline`) combined via Aalen-Johansen (cause-specific) or `1 - exp(-H)` (Fine-Gray). AUC and C-index are unaffected; Brier scores reported with earlier versions are not reliable. `compute_baseline_cumulative_hazard` now takes this baseline as well.
- Fixed `train_nested_cv.py` crashing when saving aggregated hyperparameters if outer folds selected different numbers of hidden layers, and writing numpy object tags instead of plain numbers to the YAML.
- Fixed `tune_optuna_optimized.py` final evaluation indexing risks by evaluation time, which raised an `IndexError`.
- Fixed hyperparameter search not being reproducible: the Optuna sampler was unseeded, so `--seed` did not make nested-CV results repeatable (e.g. PBC AUC could differ by more than 0.2 between runs with the same seed). The `TPESampler` is now seeded from `--seed`.

### Other Changes
- Added a `--device {auto,cpu,cuda,mps}` option to the training and tuning scripts (`auto` keeps the previous CUDA-else-CPU behaviour). Apple-GPU (`mps`) training works, but for these small per-feature networks it is typically slower than CPU.

### New Features
- Added a `--risk_model` option to select between cause-specific hazards and Fine-Gray subdistribution hazards during training.
- Added cumulative hazard estimation and visualization (`compute_baseline_cumulative_hazard`, `predict_cumulative_hazard`, `plot_cumulative_hazard`) alongside the existing cumulative incidence function (CIF) visualizations.
- Vectorized the competing-risks loss functions for substantially faster training with numerically identical results.

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
pip install -e
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
