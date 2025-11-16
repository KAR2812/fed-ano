"""
Common preprocessing utilities used by preprocess_ausgrid.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Tuple, List

def season_from_month(month: int) -> int:
    # Southern hemisphere seasons mapping used in paper
    if month in [12, 1, 2]:
        return 2  # summer
    if month in [3, 4, 5]:
        return 3  # autumn
    if month in [6, 7, 8]:
        return 0  # winter
    return 1      # spring

import os
import numpy as np
import joblib

def save_npz_client(client_id, X, y, out_dir="processed"):
    """Save each client dataset as .npz file."""
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"client_{client_id}.npz")
    np.savez(file_path, X=X, y=y)
    print(f"💾 Saved {file_path}")


def load_npz(path: str):
    data = np.load(path)
    return data['X'], data['y']

def save_scaler(scaler, outpath: str):
    joblib.dump(scaler, outpath)

def load_scaler(path: str):
    return joblib.load(path)
