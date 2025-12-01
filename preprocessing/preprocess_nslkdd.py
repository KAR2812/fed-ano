#!/usr/bin/env python3
# preprocessing/preprocess_nslkdd.py

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------- PATH / IO ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_PATH = os.path.join(BASE_DIR, "data", "nsl_kdd_train.csv")
DEFAULT_TEST_PATH = os.path.join(BASE_DIR, "data", "nsl_kdd_test.csv")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "processed", "nslkdd")


def load_nslkdd(train_path: str, test_path: str) -> pd.DataFrame:
    """Load NSL-KDD train + test and concatenate."""
    print(f"📂 Loading NSL-KDD train from: {train_path}")
    df_train = pd.read_csv(train_path)

    print(f"📂 Loading NSL-KDD test from:  {test_path}")
    df_test = pd.read_csv(test_path)

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

# try different naming conventions
    label_col = None
    for candidate in ["label", "class"]:
      if candidate in df.columns:
        label_col = candidate
        break

# fallback: use last column
    if label_col is None:
     print("⚠️ No 'label' or 'class' column detected. Using last column as label.")
     label_col = df.columns[-1]

    print(f"🎯 Using label column: {label_col}")
    df = df.rename(columns={label_col: "label"})


    print(f"✅ Loaded NSL-KDD combined: {len(df)} rows")
    return df


# ---------------- LABELS ---------------- #

def build_binary_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    y_bin:
      0 = 'normal'
      1 = any other attack label

    attack_key:
      original string label, used only for Non-IID attack-based split.
    """
    df = df.copy()
    labels = df["label"].astype(str).str.strip()

    is_normal = labels.str.lower() == "normal"
    y_bin = np.where(is_normal, 0, 1).astype(int)

    attack_key = labels.values  # string labels

    n_attacks = int(y_bin.sum())
    print(f"✅ Built binary labels: normal={len(y_bin) - n_attacks}, attack={n_attacks}")
    return y_bin, attack_key


# ---------------- FEATURES ---------------- #

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build 3-D NSL-KDD features:
      - numeric features
      - one-hot encode remaining categorical (protocol_type, service, flag, etc.)
      - StandardScaler + PCA(3)
    """
    df = df.copy()
    # We'll drop the label and optional 'difficulty' before feature engineering
    drop_cols = []
    for c in ["label", "difficulty"]:
        if c in df.columns:
            drop_cols.append(c)
    df_feat = df.drop(columns=drop_cols)

    # Separate numeric and categorical
    cat_cols = df_feat.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df_feat.select_dtypes(exclude=["object"]).columns.tolist()

    # Convert numeric cols safely
    for col in num_cols:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

    df_num = df_feat[num_cols]
    df_cat = df_feat[cat_cols]

    # One-hot encode categoricals
    if len(cat_cols) > 0:
        df_cat_dummies = pd.get_dummies(df_cat, drop_first=False)
        feat_df = pd.concat([df_num, df_cat_dummies], axis=1)
    else:
        feat_df = df_num

    feat_df = feat_df.fillna(0.0)

    print(
        f"⚙️ Building NSL-KDD features: "
        f"{len(num_cols)} numeric + {len(cat_cols)} categorical "
        f"-> {feat_df.shape[1]} dims before PCA"
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    pca = PCA(n_components=3, random_state=42)
    X_3d = pca.fit_transform(X_scaled)

    print(f"✅ NSL-KDD feature matrix shape after PCA: {X_3d.shape}")
    return X_3d


# ---------------- NON-IID ATTACK-BASED SPLIT ---------------- #

def partition_attack_non_iid(
    X: np.ndarray,
    y_bin: np.ndarray,
    attack_key: np.ndarray,
    n_clients: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Non-IID split:
      - Normal (y=0) samples split evenly across all clients.
      - Attack (y=1) samples grouped by attack label (e.g., 'neptune', 'smurf')
        and each client gets a different subset of attack labels.
    """
    rng = np.random.default_rng(123)

    idx_all = np.arange(len(y_bin))
    normal_idx = idx_all[y_bin == 0]
    attack_idx = idx_all[y_bin == 1]

    # Normals evenly
    normal_splits = np.array_split(normal_idx, n_clients)

    # Unique attack labels (strings) excluding 'normal'
    unique_labels = np.unique(attack_key[attack_idx])
    unique_labels = [lbl for lbl in unique_labels if lbl.lower() != "normal"]
    rng.shuffle(unique_labels)

    attack_groups = np.array_split(unique_labels, n_clients)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for cid in range(n_clients):
        normals_c = normal_splits[cid]

        labels_for_client = attack_groups[cid]
        if len(labels_for_client) > 0:
            mask_attack = np.isin(attack_key, labels_for_client) & (y_bin == 1)
            attacks_c = idx_all[mask_attack]
        else:
            attacks_c = np.array([], dtype=int)

        client_idx = np.concatenate([normals_c, attacks_c])
        rng.shuffle(client_idx)

        X_c = X[client_idx]
        y_c = y_bin[client_idx]

        print(
            f"👤 Client {cid:02d}: "
            f"samples={len(X_c)}, attacks={int(y_c.sum())}, "
            f"attack_labels={list(labels_for_client)}"
        )
        splits.append((X_c, y_c))

    return splits


def save_clients_npz(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for cid, (X_c, y_c) in enumerate(splits):
        out_path = os.path.join(out_dir, f"client_{cid}.npz")
        np.savez_compressed(out_path, X=X_c, y=y_c)
        print(
            f"💾 Saved client {cid} data to {out_path} "
            f"(X shape={X_c.shape}, attacks={int(y_c.sum())})"
        )


# ---------------- MAIN ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=DEFAULT_TRAIN_PATH,
        help="Path to NSL-KDD train CSV",
    )
    parser.add_argument(
        "--test",
        default=DEFAULT_TEST_PATH,
        help="Path to NSL-KDD test CSV",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=20,
        help="Number of FL clients (default: 20)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help="Output directory for client .npz files",
    )
    args = parser.parse_args()

    df = load_nslkdd(args.train, args.test)
    y_bin, attack_key = build_binary_labels(df)
    X = build_features(df)

    splits = partition_attack_non_iid(X, y_bin, attack_key, n_clients=args.clients)
    save_clients_npz(splits, out_dir=args.out)

    print("🎉 NSL-KDD preprocessing complete.")


if __name__ == "__main__":
    main()
