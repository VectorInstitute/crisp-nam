# CRISP-NAM: Competing Risks for Interpretable Survival Analysis using Neural Additive Models

This repository contains research code for the paper: [CRISP-NAM: Competing Risks Interpretable Survival
Prediction with Neural Additive Models](https://ceur-ws.org/Vol-4059/paper5.pdf).
It includes the Python code for the following:

- Models: `CRISP-NAM` and `DeepHIT`
- Data loading utilities for 4 datasets: Framingham, PBC, Support2, Synthetic
- Training scripts: Standard training, Hyperparameter optimization via Optuna, Nested cross validation
- Risk models: Cause-specific hazards (default) and Fine-Gray subdistribution hazards, selectable via `--risk_model`
- Metrics: Loss and risk functions for survival analysis.
- Plotting: Feature importance, Shape functions, and Cumulative hazard curves for interpretability.

## PyPI package
The core files of research: models, metrics and plotting utilities.

## Installation
You can install the package via the following pip command:
```bash
pip install crisp_nam
```

## Citation
> <cite>@inproceedings{ramachandram2025crispnam,<br>
title={CRISP-NAM: Competing Risks Interpretable Survival Prediction with Neural Additive Models},<br>
author={Ramachandram, Dhanesh and Raval, Ananya},<br>
booktitle={EXPLIMED 2025 - Second Workshop on Explainable AI for the Medical Domain},<br>
year={2025}<br>
}</cite> 