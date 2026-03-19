## Datasets overview
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

CSV files for all datasets are available in the repository within `crisp-nam/datasets` folder.

## Data loading scripts
The repository contains preprocessing scripts within `datasets` folder that handle missing values, feature encoding, and proper train/test splitting to prevent data leakage for each dataset.

- **`framingham_dataset.py`**: Preprocess and load Framingham dataset.
- **`pbc_dataset.py`**: Preprocess and load PBC dataset.
- **`support_dataset.py`**: Preprocess and load Support2 dataset.
- **`synthetic_dataset.py`**: Preprocess and load synthetically generated dataset.

## Return format
Each script returns the following values for use within training scripts:
1. `x`: Feature matrix after preprocessing
2. `t`: Array of time to event values
3. `event_type`: Categorical data depicting event types
4. `feature_names`: Array of feature names
5. `n_continuous`: Number of continuous features
6. `feature_ranges`: List of (min, max) ranges for each feature.
```
> [!NOTE]
> If introducing a new dataset, the above mentioned return format is needed to run the training scripts.