import os
import pandas as pd
import numpy as np
# Enable and import IterativeImputer (MICE) from scikit-learn:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer



def load_support_dataset():
    """
    Load and preprocess the SUPPORT dataset.
    This function reads the SUPPORT dataset from a CSV file, imputes missing values,
    encodes categorical features, and constructs the outcome variable for survival analysis.
    It returns the processed features, time-to-event data, event types, feature names, 
    the number of continuous features, and feature ranges.
    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Combined array of processed categorical and continuous features.
            - t (numpy.ndarray): Time-to-event data with a +1 offset.
            - event_type (numpy.ndarray): Event type array (0: censored, 1: cancer death, 2: non-cancer death).
            - feature_names (numpy.ndarray): Array of feature names.
            - n_continuous (int): Number of continuous features.
            - feature_ranges (list): List of tuples representing the range (min, max) for each feature.
    Notes:
        - The dataset file "support2.csv" must be located in the same directory as this script.
        - Continuous features are imputed using the median strategy if missing values are present.
        - Categorical features are imputed using the most frequent strategy if missing values are present.
        - One-hot encoding is applied to categorical features, dropping the first level to avoid the dummy variable trap.
        - The outcome variable is constructed using the 'ca', 'dzgroup', and 'death' columns.
        - Time-to-event data ('d.time') is offset by +1 to avoid zero follow-up times.
    """
   

    file_path = os.path.join(os.path.dirname(__file__), "support2.csv")
    data = pd.read_csv(file_path)
    print(data.columns)
    


    
    is_cancer = data['ca'].astype(str).str.lower().str.contains("meta") | \
                data['dzgroup'].astype(str).str.lower().str.contains("cancer")
    event_type = np.where(data['death'] == 1, np.where(is_cancer, 1, 2), 0)


    
    
    cont_cols = [
        'age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb',
        'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine',
        'scoma', 'aps', 'sps', 'adls', 'adlsc', 'charges', 'totcst', 'totmcst', 'avtisst'
    ]
    x_cont = data[cont_cols]
    
    if x_cont.isnull().values.any():
        simp_imputer = SimpleImputer(strategy='median')
        x_cont_imputed = simp_imputer.fit_transform(x_cont)
    else:
        x_cont_imputed = x_cont.values
    cont_feature_ranges = list(zip(np.nanmin(x_cont_imputed, axis=0),
                                    np.nanmax(x_cont_imputed, axis=0)))

    # -- Categorical features --
    # Remove leakage fields 'ca', 'dzgroup', 'dzclass'
    cat_cols = ['sex', 'income', 'race', 'dnr', 'dementia', 'diabetes']
    x_cat = data[cat_cols]
    if x_cat.isnull().values.any():
        cat_imputer = SimpleImputer(strategy='most_frequent')
        x_cat_imputed = cat_imputer.fit_transform(x_cat)
    else:
        x_cat_imputed = x_cat.values
    x_cat_df = pd.DataFrame(x_cat_imputed, columns=cat_cols)
    
    x_cat_encoded = pd.get_dummies(x_cat_df, drop_first=True)
    cat_feature_ranges = [(0.0, 1.0)] * x_cat_encoded.shape[1]

    
    x = np.hstack([x_cat_encoded.values, x_cont_imputed])
    feature_names = np.concatenate([x_cat_encoded.columns, cont_cols])
    n_continuous = len(cont_cols)
    feature_ranges = cat_feature_ranges + cont_feature_ranges

   
    t = data['d.time'].values
    valid = ~np.isnan(t)
    
    print("Completed imputation of missing values.")

    return (
        x[valid],
        t[valid] + 1,   # Add +1 offset to follow-up times
        event_type[valid],
        feature_names,
        n_continuous,
        feature_ranges
    )

if __name__ == "__main__":
    x, t, e, feature_names, n_continuous, feature_ranges = load_support_dataset()
    print("Feature names:", feature_names)
