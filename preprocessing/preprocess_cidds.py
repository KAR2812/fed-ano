#!/usr/bin/env python3
# preprocessing/preprocess_cidds.py

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------- PATH / IO ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "cidss.csv")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "processed", "cidds")


def load_cidds(path: str) -> pd.DataFrame:
    """Load CIDDS-001 CSV and do light cleaning."""
    print(f"📂 Loading CIDDS-001 data from: {path}")
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    expected_cols = {
        "duration",
        "proto",
        "packets",
        "bytes",
        "flows",
        "tcp_urg",
        "tcp_ack",
        "tcp_psh",
        "tcp_rst",
        "tcp_syn",
        "tcp_fin",
        "tos",
        "label",
        "attack_type",
        "attack_id",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing CIDDS columns: {missing}")

    # Strip string columns
    for col in ["proto", "label", "attack_type"]:
        df[col] = df[col].astype(str).str.strip()

    print(f"✅ Loaded {len(df)} CIDDS rows")
    return df


# ---------------- LABELS ---------------- #

def build_binary_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build binary labels and an attack key for Non-IID partitioning.

    y_bin:
      0 = benign/normal (attack_id == 0 OR attack_type=='benign')
      1 = attack       (attack_id > 0 OR attack_type != 'benign')

    attack_key:
      attack_id (int) used only for splitting (Non-IID).
    """
    df = df.copy()

    # Ensure attack_id is numeric
    df["attack_id"] = pd.to_numeric(df["attack_id"], errors="coerce").fillna(0).astype(int)
    df["attack_type"] = df["attack_type"].astype(str).str.lower()

    benign_mask = (df["attack_id"] == 0) | (df["attack_type"] == "benign")
    y_bin = np.where(benign_mask, 0, 1).astype(int)

    attack_key = df["attack_id"].values  # used for Non-IID attack-based split

    n_attacks = int(y_bin.sum())
    print(f"✅ Built binary labels: normal={len(y_bin) - n_attacks}, attack={n_attacks}")
    return y_bin, attack_key


# ---------------- FEATURES ---------------- #

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build 3-D feature vector using:
      - numeric features
      - one-hot 'proto'
      - StandardScaler + PCA(3)
    """
    df = df.copy()

    # Numeric columns from CIDDS
    num_cols = [
        "duration",
        "packets",
        "bytes",
        "flows",
        "tcp_urg",
        "tcp_ack",
        "tcp_psh",
        "tcp_rst",
        "tcp_syn",
        "tcp_fin",
        "tos",
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One-hot encode proto
    proto_dummies = pd.get_dummies(df["proto"], prefix="proto", drop_first=False)

    feat_df = pd.concat([df[num_cols], proto_dummies], axis=1)
    feat_df = feat_df.fillna(0.0)

    print(f"⚙️ Building features from {feat_df.shape[1]} raw dimensions")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    pca = PCA(n_components=3, random_state=42)
    X_3d = pca.fit_transform(X_scaled)

    print(f"✅ Feature matrix shape after PCA: {X_3d.shape}")
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
      - Normal (y=0) samples are evenly split across all clients.
      - Attack samples are grouped by attack_id and each client
        gets a different subset of attack_ids.
    """
    rng = np.random.default_rng(123)

    idx_all = np.arange(len(y_bin))
    normal_idx = idx_all[y_bin == 0]
    attack_idx = idx_all[y_bin == 1]

    # Split normals evenly
    normal_splits = np.array_split(normal_idx, n_clients)

    # Unique attack ids (excluding 0)
    unique_attacks = np.unique(attack_key[attack_idx])
    unique_attacks = unique_attacks[unique_attacks != 0]
    rng.shuffle(unique_attacks)

    attack_groups = np.array_split(unique_attacks, n_clients)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for cid in range(n_clients):
        normals_c = normal_splits[cid]

        attacks_for_client = attack_groups[cid]
        if len(attacks_for_client) > 0:
            mask_attack = np.isin(attack_key, attacks_for_client) & (y_bin == 1)
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
            f"attack_ids={list(attacks_for_client)}"
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
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to CIDDS-001 CSV",
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

    df = load_cidds(args.data)
    y_bin, attack_key = build_binary_labels(df)
    X = build_features(df)

    splits = partition_attack_non_iid(X, y_bin, attack_key, n_clients=args.clients)
    save_clients_npz(splits, out_dir=args.out)

    print("🎉 CIDDS-001 preprocessing complete.")


if __name__ == "__main__":
    main()
