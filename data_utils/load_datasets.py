from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def load_framingham(sequential=False):
    """
    Load and preprocess the Framingham dataset for competing risks analysis,
    with imputation but no scaling. Feature normalization must be done externally
    after splitting to avoid data leakage.

    Continuous-feature imputation is intentionally NOT done in this function.
    Imputing continuous columns (e.g. mean-fill) on the full dataset before any
    train/validation split leaks validation-fold statistics into the training
    fold and vice versa -- the same class of leakage that per-fold scaling
    already guards against elsewhere in this codebase. Continuous columns are
    therefore returned with NaNs preserved, and callers must fit imputation
    per-fold/per-split (mirroring the existing per-fold scaling pattern) using
    the returned `continuous_impute_strategy`. Categorical columns are still
    imputed here (mode/most-frequent imputation), since that carries much
    lower leakage risk and correctly re-deriving one-hot column vocabulary
    per-fold would be a much larger change for little benefit.

    Returns
    -------
        x (np.ndarray): Feature matrix with one-hot categorical + raw continuous features.
        t (np.ndarray): Time-to-event (with +1 offset).
        e (np.ndarray): Event indicator (0=censored, 1=CVD, 2=death).
        feature_names (np.ndarray): Names of all features (categorical first, continuous after).
        n_continuous (int): Number of continuous features at the end of x.
        feature_ranges (None): Placeholder for backward compatibility.
        continuous_impute_strategy (str): Imputation strategy to use for continuous
            features, applied per-fold by the caller (not applied here) -- see
            data-leakage note above.
    """
    file_path = "datasets/framingham.csv"
    data = pd.read_csv(file_path)

    if not sequential:
        data = data.groupby("RANDID").first()

        cat_cols = [
            "SEX",
            "CURSMOKE",
            "DIABETES",
            "BPMEDS",
            "PREVCHD",
            "PREVAP",
            "PREVMI",
            "PREVSTRK",
            "PREVHYP",
            "educ",
        ]

        # 'HDLC', 'LDLC' - removed to replicate nfg experiments.
        cont_cols = [
            "TOTCHOL",
            "AGE",
            "SYSBP",
            "DIABP",
            "CIGPDAY",
            "BMI",
            "HEARTRTE",
            "GLUCOSE",
        ]

        cat_imputer = SimpleImputer(strategy="most_frequent")
        x_cat = pd.DataFrame(
            cat_imputer.fit_transform(data[cat_cols]),
            columns=cat_cols,
            index=data.index,
        )
        x_cat = pd.get_dummies(x_cat, drop_first=True)

        x_cont = data[cont_cols].values.astype(float)  # raw; continuous imputation is deferred to
                                                         # per-fold in the training scripts to avoid
                                                         # leaking validation-fold statistics (mirrors
                                                         # how per-fold scaling already works)

        x = np.hstack([x_cat.values, x_cont])
        feature_names = np.concatenate([x_cat.columns.values, cont_cols])
        n_continuous = len(cont_cols)
        event = np.zeros(len(data), dtype=int)
        time = (data["TIMEDTH"] - data["TIME"]).values

        # Primary CVD event (risk=1)
        cvd_mask = data["CVD"] == 1
        event[cvd_mask] = 1
        time_cvd = (data["TIMECVD"] - data["TIME"]).values
        time[cvd_mask] = time_cvd[cvd_mask]

        # Competing death event (risk=2), only if CVD did not occur
        death_mask = (data["DEATH"] == 1) & ~cvd_mask
        event[death_mask] = 2

        # Filter out invalid or zero times
        valid = ~np.isnan(time) & (time > 0)
        x = x[valid]
        t = time[valid] + 1
        e = event[valid]

        # Sanity check: only the (globally-imputed) categorical portion should be NaN-free here.
        # Continuous columns intentionally still contain NaN -- see comment above.
        assert not np.isnan(x_cat.values.astype(float)).any(), "NaNs found in categorical feature matrix"

        return x, t, e, feature_names, n_continuous, None, "mean"
    raise NotImplementedError("Sequential mode not yet implemented.")


def load_pbc2_dataset():
    """
    Load and preprocess the PBC2 dataset for survival analysis.

    The dataset is preprocessed to include both continuous and categorical features.
    Categorical features have missing values imputed here (most-frequent/mode
    imputation). Continuous features are intentionally left un-imputed (NaNs
    preserved): imputing them on the full dataset before any train/validation
    split would leak validation-fold statistics into the training fold, so
    imputation is deferred to the caller, which must fit it per-fold/per-split
    (the same fit-on-train-only pattern already used for per-fold scaling).
    The function also constructs the outcome variable for competing risks
    analysis and returns time-to-event data.

    Returns
    -------
        tuple: A tuple containing the following elements:
            - x (numpy.ndarray): Combined feature matrix with categorical features
              one-hot encoded and continuous features raw (NaNs preserved).
            - t (numpy.ndarray): Time-to-event data in days.
            - event_type (numpy.ndarray): Event type array where:
                0 = censored,
                1 = death,
                2 = transplantation.
            - feature_names (numpy.ndarray): Array of feature names.
            - n_continuous (int): Number of continuous features.
            - feature_ranges (list of tuple): List of (min, max) ranges for each feature.
            - continuous_impute_strategy (str): Imputation strategy to use for
              continuous features, applied per-fold by the caller (not applied
              here) -- see data-leakage note above.
    """
    file_path = "datasets/pbc2.csv"
    data = pd.read_csv(file_path)

    # PBC2 is a longitudinal dataset: most patients have multiple visit-rows
    # (mean ~6.2, up to 16) that all share the same outcome (years, status)
    # but differ in their covariate values. Without deduplication each visit
    # is treated as an independent subject, which both violates the i.i.d.
    # assumption the model relies on and lets the same patient's rows land
    # in both the train and validation cross-validation splits. Keep only
    # each patient's earliest (baseline) visit, mirroring load_framingham's
    # per-patient deduplication.
    data = (
        data.sort_values(["id", "year"])
        .drop_duplicates(subset="id", keep="first")
        .reset_index(drop=True)
    )
    data = data.drop(columns=["id", "sno.", "year", "status2"], axis=1)

    event_type = np.where(
        data["status"] == "dead", 1, np.where(data["status"] == "transplanted", 2, 0)
    )

    cont_cols = [
        "age",
        "serBilir",
        "serChol",
        "albumin",
        "alkaline",
        "SGOT",
        "platelets",
        "prothrombin",
        "histologic",
    ]
    x_cont = data[cont_cols].replace("NA", np.nan).astype(float)

    x_cont_raw = x_cont.values  # raw; continuous imputation deferred to per-fold in
                                 # the training scripts to avoid leaking validation-fold
                                 # statistics (mirrors per-fold scaling)

    cont_feature_ranges = list(
        zip(np.nanmin(x_cont_raw, axis=0), np.nanmax(x_cont_raw, axis=0))
    )

    cat_cols = ["sex", "drug", "ascites", "hepatomegaly", "spiders", "edema"]
    # Let the imputer see real NaNs directly -- do not pre-fill with a
    # "missing" placeholder string first, which would make the imputer a
    # no-op (it only fills actual NaNs) and silently collapse missing rows
    # into the one-hot reference category instead of the most-frequent one.
    x_cat = data[cat_cols]

    cat_imputer = SimpleImputer(strategy="most_frequent")
    x_cat_imputed = cat_imputer.fit_transform(x_cat)
    x_cat_df = pd.DataFrame(x_cat_imputed, columns=cat_cols)

    x_cat_encoded = pd.get_dummies(x_cat_df, drop_first=True)

    cat_feature_ranges = [(0.0, 1.0)] * x_cat_encoded.shape[1]

    x = np.hstack([x_cat_encoded.values, x_cont_raw])

    feature_names = np.concatenate([x_cat_encoded.columns, cont_cols])
    n_continuous = len(cont_cols)
    feature_ranges = cat_feature_ranges + cont_feature_ranges

    t = data["years"].astype(float).values * 365.25
    valid = ~np.isnan(t)

    return (
        x[valid],
        t[valid],
        event_type[valid],
        feature_names,
        n_continuous,
        feature_ranges,
        "mean",
    )


def load_support_dataset():
    """
    Load and preprocess the SUPPORT dataset.
    This function reads the SUPPORT dataset from a CSV file, encodes categorical
    features (imputing their missing values here), and constructs the outcome
    variable for survival analysis. Continuous features are intentionally left
    un-imputed (NaNs preserved): imputing them on the full dataset before any
    train/validation split would leak validation-fold statistics into the
    training fold, so imputation is deferred to the caller, which must fit it
    per-fold/per-split (the same fit-on-train-only pattern already used for
    per-fold scaling). It returns the processed features, time-to-event data,
    event types, feature names, the number of continuous features, feature
    ranges, and the recommended continuous-imputation strategy.

    Returns
    -------
        tuple: A tuple containing:
            - x (numpy.ndarray): Combined array of processed categorical (imputed)
              and continuous (raw, NaNs preserved) features.
            - t (numpy.ndarray): Time-to-event data with a +1 offset.
            - event_type (numpy.ndarray): Event type array (0: censored, 1: cancer death, 2: non-cancer death).
            - feature_names (numpy.ndarray): Array of feature names.
            - n_continuous (int): Number of continuous features.
            - feature_ranges (list): List of tuples representing the range (min, max) for each feature.
            - continuous_impute_strategy (str): Imputation strategy to use for
              continuous features, applied per-fold by the caller (not applied
              here) -- see data-leakage note above.

    Notes
    -----
        - The dataset file "support2.csv" must be located in the same directory as this script.
        - Continuous features are NOT imputed here; see data-leakage note above.
        - Categorical features are imputed using the most frequent strategy if missing values are present.
        - One-hot encoding is applied to categorical features, dropping the first level to avoid the dummy variable trap.
        - The outcome variable is constructed using the 'ca' and 'death' columns.
        - Time-to-event data ('d.time') is offset by +1 to avoid zero follow-up times.
    """
    file_path = "datasets/support2.csv"
    data = pd.read_csv(file_path)

    # 'ca' is the authoritative cancer-status field, with values
    # {"no", "yes", "metastatic"}. Checking it directly (rather than a
    # ca/dzgroup substring-matching heuristic) avoids missing non-metastatic
    # cancer patients (ca == "yes") whose dzgroup label doesn't literally
    # contain the word "cancer" (e.g. "MOSF w/Malig", "COPD", "Coma").
    is_cancer = data["ca"].astype(str).str.lower() != "no"
    event_type = np.where(data["death"] == 1, np.where(is_cancer, 1, 2), 0)

    cont_cols = [
        "age",
        "num.co",
        "meanbp",
        "wblc",
        "hrt",
        "resp",
        "temp",
        "pafi",
        "alb",
        "bili",
        "crea",
        "sod",
        "ph",
        "glucose",
        "bun",
        "urine",
        "scoma",
        "aps",
        "sps",
        "adls",
        "adlsc",
        "charges",
        "totcst",
        "totmcst",
        "avtisst",
    ]
    x_cont = data[cont_cols]

    x_cont_raw = x_cont.values  # raw; continuous imputation deferred to per-fold in
                                 # the training scripts to avoid leaking validation-fold
                                 # statistics (mirrors per-fold scaling)
    cont_feature_ranges = list(
        zip(np.nanmin(x_cont_raw, axis=0), np.nanmax(x_cont_raw, axis=0))
    )

    # -- Categorical features --
    # Remove leakage fields 'ca', 'dzgroup', 'dzclass'
    cat_cols = ["sex", "income", "race", "dnr", "dementia", "diabetes"]
    x_cat = data[cat_cols]
    if x_cat.isnull().values.any():
        cat_imputer = SimpleImputer(strategy="most_frequent")
        x_cat_imputed = cat_imputer.fit_transform(x_cat)
    else:
        x_cat_imputed = x_cat.values
    x_cat_df = pd.DataFrame(x_cat_imputed, columns=cat_cols)

    x_cat_encoded = pd.get_dummies(x_cat_df, drop_first=True)
    cat_feature_ranges = [(0.0, 1.0)] * x_cat_encoded.shape[1]

    x = np.hstack([x_cat_encoded.values, x_cont_raw])
    feature_names = np.concatenate([x_cat_encoded.columns, cont_cols])
    n_continuous = len(cont_cols)
    feature_ranges = cat_feature_ranges + cont_feature_ranges

    t = data["d.time"].values
    valid = ~np.isnan(t)

    print("Completed imputation of categorical missing values.")

    return (
        x[valid],
        t[valid] + 1,  # Add +1 offset to follow-up times
        event_type[valid],
        feature_names,
        n_continuous,
        feature_ranges,
        "median",
    )


def load_synthetic_dataset() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, List[str], int, List[tuple], Optional[str]
]:
    """
    Loads a synthetic competing risks dataset from a CSV file.

    The CSV is expected to have a header with the following columns:
      - time: observed time
      - label: event indicator (0 for censored; >0 for event types)
      - true_time: (optional) true time (unused here)
      - true_label: (optional) true event label (unused here)
      - feature1, feature2, ..., featureN: feature values

    This dataset has no missing data, so there is nothing to impute; the
    returned `continuous_impute_strategy` is always `None` to signal to
    callers that no imputation is needed/applicable.

    Returns
    -------
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        T_obs (np.ndarray): Observed times of shape (n_samples,).
        e (np.ndarray): Event indicators of shape (n_samples,).
        feature_names (List[str]): List of feature names.
        n_continuous (int): Total number of continuous features.
        feature_ranges (List[tuple]): List of (min, max) tuples for each feature.
        continuous_impute_strategy (Optional[str]): Always None -- this dataset
            has no missing continuous values, so no imputation is applicable.
    """
    file_path = "datasets/synthetic_comprisk.csv"
    df = pd.read_csv(file_path)

    T_obs = df["time"].values.astype(np.float32)
    e = df["label"].values.astype(np.float32)

    feature_columns = [col for col in df.columns if col.startswith("feature")]
    X = df[feature_columns].values.astype(np.float32)

    feature_names = feature_columns
    n_continuous = X.shape[1]

    feature_ranges = [
        (float(X[:, i].min()), float(X[:, i].max())) for i in range(n_continuous)
    ]

    return X, T_obs, e, feature_names, n_continuous, feature_ranges, None
