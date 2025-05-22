import numpy as np
import pandas as pd
from typing import Tuple, List
import os
import torch

def load_synthetic_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int, List[tuple]]:
    """
    Loads a synthetic competing risks dataset from a CSV file.
    
    The CSV is expected to have a header with the following columns:
      - time: observed time
      - label: event indicator (0 for censored; >0 for event types)
      - true_time: (optional) true time (unused here)
      - true_label: (optional) true event label (unused here)
      - feature1, feature2, ..., featureN: feature values
      
    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        T_obs (np.ndarray): Observed times of shape (n_samples,).
        e (np.ndarray): Event indicators of shape (n_samples,).
        feature_names (List[str]): List of feature names.
        n_continuous (int): Total number of continuous features.
        feature_ranges (List[tuple]): List of (min, max) tuples for each feature.
    """
    
    use_gpu_compatible_dtype = torch.cuda.is_available()
    
   
    
 
    file_path = os.path.join(os.path.dirname(__file__), "synthetic_comprisk.csv")
    df = pd.read_csv(file_path)

    
 
    T_obs = df["time"].values .astype(np.float32)
    e = df["label"].values.astype(np.float32)
    
    
    feature_columns = [col for col in df.columns if col.startswith("feature")]
    X = df[feature_columns].values .astype(np.float32)
    
   
    feature_names = feature_columns
    n_continuous = X.shape[1]
    
    feature_ranges = [(float(X[:, i].min()), float(X[:, i].max())) for i in range(n_continuous)]
    
    return X, T_obs, e, feature_names, n_continuous, feature_ranges


if __name__ == "__main__":
    X, T_obs, e, feature_names, n_continuous, feature_ranges = load_synthetic_dataset()
    print("X shape:", X.shape)
    print("T_obs shape:", T_obs.shape)
    print("e shape:", e.shape)
    print("Feature names:", feature_names)
    print("Number of continuous features:", n_continuous)
    print("Feature ranges:", feature_ranges)
    # For a quick inspection, print the first 5 rows (features, time, event)
    print("First 5 rows:")
    print(np.hstack([X[:5], T_obs[:5, None], e[:5, None]]))







