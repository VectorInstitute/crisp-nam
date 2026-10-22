## Datasets overview
The repository includes four well-established survival analysis datasets:

1. **Framingham Heart Study**: Cardiovascular disease prediction with competing events (CVD vs. death)
   - Features: Demographics, clinical measurements, lifestyle factors
   - Events: Cardiovascular disease, death from other causes

2. **PBC (Primary Biliary Cirrhosis)**: Liver disease progression study
   - Features: Clinical laboratory values, demographic information
   - Events: Death, liver transplantation
   - `pbc2.csv` has one row per clinic visit; the loader keeps each patient's first visit, giving 312 patients

3. **SUPPORT**: Study to understand prognoses and preferences for outcomes
   - Features: Comprehensive clinical and demographic variables
   - Events: Death of a patient with cancer, death of a patient without cancer
   - The dataset does not record cause of death, so cancer status comes from the `ca` column

4. **Synthetic Dataset**: Controlled simulation for method validation
   - Features: Simulated clinical variables with known ground truth
   - Events: Multiple competing risks with controllable hazard functions

CSV files for all datasets are available in the repository within `crisp-nam/datasets` folder.

## Data loading functions
The loaders are in `data_utils/load_datasets.py`, one function per dataset:

- **`load_framingham()`**: Load the Framingham dataset.
- **`load_pbc2_dataset()`**: Load the PBC dataset.
- **`load_support_dataset()`**: Load the SUPPORT (Support2) dataset.
- **`load_synthetic_dataset()`**: Load the synthetic dataset.

Each loader encodes categorical features and fills their missing values. Missing continuous values are left as `NaN`: the training scripts fill them, and scale features, using statistics from each training fold only, so test data never influences them.

## Return format
Each loader returns the following values for use within training scripts:
1. `x`: Feature matrix, with continuous features unscaled and possibly containing `NaN`
2. `t`: Array of time to event values
3. `event_type`: Array of event types (0 = censored, 1, 2, ... = competing events)
4. `feature_names`: Array of feature names
5. `n_continuous`: Number of continuous features
6. `feature_ranges`: List of (min, max) ranges for each feature, or `None`
7. `continuous_impute_strategy`: How the training scripts should fill missing continuous values (for example `"mean"`), or `None` if there are none

> [!NOTE]
> If introducing a new dataset, the above mentioned return format is needed to run the training scripts.