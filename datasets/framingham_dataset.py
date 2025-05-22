import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_framingham(competing=True, sequential=False):
    """
    Load and preprocess the Framingham dataset for competing risks analysis,
    with imputation but no scaling. Feature normalization must be done externally
    after splitting to avoid data leakage.

    Returns:
        x (np.ndarray): Feature matrix with one-hot categorical + raw continuous features.
        t (np.ndarray): Time-to-event (with +1 offset).
        e (np.ndarray): Event indicator (0=censored, 1=CVD, 2=death).
        feature_names (np.ndarray): Names of all features (categorical first, continuous after).
        n_continuous (int): Number of continuous features at the end of x.
        feature_ranges (None): Placeholder for backward compatibility.
    """
    
    file_path = os.path.join(os.path.dirname(__file__), "framingham.csv")
    data = pd.read_csv(file_path)

    if not sequential:
        
        data = data.groupby("RANDID").first()

        
        cat_cols = [
            'SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS',
            'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'educ'
        ]
        
        # 'HDLC', 'LDLC' - removed to replicate nfg experiments.
        cont_cols = [
            'TOTCHOL',  'AGE',
            'SYSBP', 'DIABP', 'CIGPDAY', 'BMI',
            'HEARTRTE', 'GLUCOSE'
        ]

        
        cat_imputer = SimpleImputer(strategy='most_frequent')
        x_cat = pd.DataFrame(
            cat_imputer.fit_transform(data[cat_cols]),
            columns=cat_cols,
            index=data.index
        )
        x_cat = pd.get_dummies(x_cat, drop_first=True)

        
        cont_imputer = SimpleImputer(strategy='mean')
        x_cont = cont_imputer.fit_transform(data[cont_cols])

       
        x = np.hstack([x_cat.values, x_cont])
        feature_names = np.concatenate([x_cat.columns.values, cont_cols])
        n_continuous = len(cont_cols)

        
        event = np.zeros(len(data), dtype=int)
        
        time = (data['TIMEDTH'] - data['TIME']).values

        # Primary CVD event (risk=1)
        cvd_mask = data['CVD'] == 1
        event[cvd_mask] = 1
        time_cvd = (data['TIMECVD'] - data['TIME']).values
        time[cvd_mask] = time_cvd[cvd_mask]

        # Competing death event (risk=2), only if CVD did not occur
        death_mask = (data['DEATH'] == 1) & ~cvd_mask
        event[death_mask] = 2

        # Filter out invalid or zero times
        valid = ~np.isnan(time) & (time > 0)
        x = x[valid]
        t = time[valid] + 1
        e = event[valid]

        # Sanity check
        assert not np.isnan(x).any(), "NaNs found in feature matrix"

        return x, t, e, feature_names, n_continuous, None
    else:
        raise NotImplementedError("Sequential mode not yet implemented.")

if __name__ == "__main__":
    x, t, e, feature_names, n_continuous, feature_ranges = load_framingham()


