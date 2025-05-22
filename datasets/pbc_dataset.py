import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def load_pbc2_dataset():
    """
    Load and preprocess the PBC2 dataset for survival analysis.

    The dataset is preprocessed to include both continuous and categorical features,
    with missing values imputed. The function also constructs the outcome variable
    for competing risks analysis and returns time-to-event data.

    Returns:
        tuple: A tuple containing the following elements:
            - x (numpy.ndarray): Combined feature matrix with categorical features
              one-hot encoded and continuous features imputed.
            - t (numpy.ndarray): Time-to-event data in days.
            - event_type (numpy.ndarray): Event type array where:
                0 = censored,
                1 = death,
                2 = transplantation.
            - feature_names (numpy.ndarray): Array of feature names.
            - n_continuous (int): Number of continuous features.
            - feature_ranges (list of tuple): List of (min, max) ranges for each feature.
    """
  

    file_path = os.path.join(os.path.dirname(__file__), "pbc2.csv")
    data = pd.read_csv(file_path)
    data = data.drop(columns=['id', 'sno.', 'year', 'status2'], axis=1)
    

    event_type = np.where(
        data['status'] == 'dead', 1,
        np.where(data['status'] == 'transplanted', 2, 0)
    )

    
    cont_cols = [
        'age', 'serBilir', 'serChol', 'albumin', 'alkaline',
        'SGOT', 'platelets', 'prothrombin', 'histologic'
    ]
    x_cont = data[cont_cols].replace('NA', np.nan).astype(float)

   
    mean_imputer = SimpleImputer(strategy='mean')
    x_cont_imputed = mean_imputer.fit_transform(x_cont)

    cont_feature_ranges = list(zip(np.nanmin(x_cont_imputed, axis=0),
                                   np.nanmax(x_cont_imputed, axis=0)))

   
    cat_cols = ['sex', 'drug', 'ascites', 'hepatomegaly', 'spiders', 'edema']
    x_cat = data[cat_cols].fillna('missing')

    cat_imputer = SimpleImputer(strategy='most_frequent')
    x_cat_imputed = cat_imputer.fit_transform(x_cat)
    x_cat_df = pd.DataFrame(x_cat_imputed, columns=cat_cols)

    x_cat_encoded = pd.get_dummies(x_cat_df, drop_first=True)

    
    x_cat_encoded = x_cat_encoded.loc[:, ~x_cat_encoded.columns.str.contains('_missing')]

    cat_feature_ranges = [(0.0, 1.0)] * x_cat_encoded.shape[1]

  
    x = np.hstack([x_cat_encoded.values, x_cont_imputed])

    feature_names = np.concatenate([x_cat_encoded.columns, cont_cols])
    n_continuous = len(cont_cols)
    feature_ranges = cat_feature_ranges + cont_feature_ranges

 
    t = data['years'].astype(float).values * 365.25
    valid = ~np.isnan(t)

    return (
        x[valid],
        t[valid],
        event_type[valid],
        feature_names,
        n_continuous,
        feature_ranges
    )


if __name__ == "__main__":
    x, t, e, feature_names, n_continuous, feature_ranges = load_pbc2_dataset()
    