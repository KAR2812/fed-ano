"""
Optional: simple skeleton to preprocess NSL-KDD dataset into federated partitions.
Customize as needed if you decide to experiment with NSL-KDD.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from preprocessing.utils_preprocessing import save_npz_client

DEFAULT_PATH = "data/nsl_kdd.csv"
OUT_DIR = "processed_nslkdd"

def preprocess_nslkdd(path=DEFAULT_PATH, n_clients=100):
    os.makedirs(OUT_DIR, exist_ok=True)
    # NSL-KDD typically needs one-hot encoding and numeric normalization
    df = pd.read_csv(path)
    # TODO: add proper preprocessing steps when dataset provided
    # Here: placeholder to split into partitions uniformly
    X = df.drop(columns=['label']).values.astype(np.float32)
    y = (df['label'] != 'normal').astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    per_client = int(np.ceil(X_train.shape[0] / n_clients))
    for i in range(n_clients):
        s = i*per_client
        e = min((i+1)*per_client, X_train.shape[0])
        Xi = X_train[s:e]
        yi = y_train[s:e]
        if Xi.shape[0] == 0:
            Xi = X_train[:1]
            yi = y_train[:1]
        save_npz_client(os.path.join(OUT_DIR, f"client_{i}.npz"), Xi, yi)
    np.savez_compressed(os.path.join(OUT_DIR, "test_set.npz"), X=X_test, y=y_test)
    print("NSL-KDD preprocessing finished.")
