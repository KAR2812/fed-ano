# preprocessing/preprocess_ausgrid.py

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_ausgrid_summary(path: str) -> pd.DataFrame:
    """
    Load Ausgrid 2011-12 Summary Community Electricity Report CSV (the one you pasted).
    We only need:
      - LGA name
      - 2011-12 Residential Total MWh

    The file is messy, so we:
      - skip top 2 rows of title/header
      - drop 'Total' and notes
    """

    print(f"📄 Loading Ausgrid summary: {path}")
    df_raw = pd.read_csv(path, skiprows=2)

    # First column is LGA, second column is 2011-12 Residential Total MWh as string with commas
    lga_col = df_raw.columns[0]
    mwh_col = df_raw.columns[1]

    df = df_raw[[lga_col, mwh_col]].copy()
    df.columns = ["LGA", "res_mwh_2011"]

    # Drop rows that are NaN or 'Total' or notes
    df = df.dropna(subset=["LGA"])
    df = df[~df["LGA"].astype(str).str.contains("Total", case=False)]
    df = df[~df["LGA"].astype(str).str.contains("Notes", case=False)]

    # Clean MWh (remove commas, convert to float)
    df["res_mwh_2011"] = (
        df["res_mwh_2011"]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
    )
    df = df[df["res_mwh_2011"].str.match(r"^\d+(\.\d+)?$")]  # keep numeric
    df["res_mwh_2011"] = df["res_mwh_2011"].astype(float)

    print(f"✅ Loaded {len(df)} LGAs with annual residential MWh")
    return df


def simulate_daily_series(
    df_lga: pd.DataFrame,
    days: int = 365,
    anomaly_frac: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    For each LGA, simulate 'days' daily consumption samples based on annual MWh.
    Inject A4 anomalies: spikes, seasonal deviations, random outliers.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for _, row in df_lga.iterrows():
        lga = row["LGA"]
        annual_mwh = row["res_mwh_2011"]
        # baseline daily consumption
        base_daily = annual_mwh * 1_000.0 / 365.0  # convert MWh -> kWh (roughly)

        # simulate 365 days
        for day in range(days):
            day_of_year = day + 1

            # Seasonal pattern: more use in winter/summer etc. (approx sinusoid)
            season_factor = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.0)

            # Random noise (~10%)
            noise = rng.normal(loc=0.0, scale=0.1)

            gc = base_daily * season_factor * (1.0 + noise)

            rows.append(
                {
                    "LGA": lga,
                    "day_of_year": day_of_year,
                    "GC": gc,
                    "is_anomaly": 0,  # mark as normal first
                }
            )

    df = pd.DataFrame(rows)
    print(f"✅ Simulated {len(df)} daily samples ({len(df_lga)} LGAs × {days} days)")
    df = inject_anomalies(df, anomaly_frac=anomaly_frac, rng=rng)
    return df


def inject_anomalies(
    df: pd.DataFrame,
    anomaly_frac: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject three anomaly types into the GC column:

    Type 1: Spike anomalies (extreme surge/drop)
    Type 2: Seasonal deviation (big break from expected seasonal pattern)
    Type 3: Random outliers (strong random noise)

    anomaly_type values:
      0 = normal
      1 = spike
      2 = seasonal break
      3 = random outlier
    """
    n_samples = len(df)
    n_anom = max(1, int(n_samples * anomaly_frac))

    print(f"⚙️ Injecting anomalies: target {n_anom} / {n_samples} samples (~{anomaly_frac*100:.1f}%)")

    indices = rng.choice(n_samples, size=n_anom, replace=False)
    anomaly_types = rng.integers(low=0, high=3, size=n_anom)  # 0,1,2 => 3 types

    df = df.copy()
    # init as normal
    df["anomaly_type"] = 0

    for idx, a_type in zip(indices, anomaly_types):
        gc = df.at[idx, "GC"]

        if a_type == 0:
            # Type 1: Spike up/down
            factor = rng.choice([0.2, 0.3, 0.5, 2.0, 3.0, 5.0])
            df.at[idx, "GC"] = gc * factor
            df.at[idx, "anomaly_type"] = 1

        elif a_type == 1:
            # Type 2: Seasonal deviation (break)
            offset = rng.normal(loc=0.0, scale=2.5)
            df.at[idx, "GC"] = gc * (1.0 + offset)
            df.at[idx, "anomaly_type"] = 2

        else:
            # Type 3: Random outlier
            outlier_noise = rng.normal(loc=0.0, scale=3.0)
            df.at[idx, "GC"] = gc * (1.0 + outlier_noise)
            df.at[idx, "anomaly_type"] = 3

    counts = df["anomaly_type"].value_counts().sort_index()
    print("✅ Anomaly type counts:")
    print(f"   0=normal         : {counts.get(0,0)}")
    print(f"   1=spike          : {counts.get(1,0)}")
    print(f"   2=seasonal break : {counts.get(2,0)}")
    print(f"   3=random outlier : {counts.get(3,0)}")

    return df


def build_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and label vector y.

    Features:
      - GC_norm (standardized GC)
      - day_sin, day_cos (cyclical encoding of day_of_year)

    Labels:
      y = anomaly_type (0=normal, 1=spike, 2=season break, 3=outlier)
    """
    df = df.copy()

    # Cyclical encoding for day_of_year
    day = df["day_of_year"].values.astype(float)
    day_sin = np.sin(2 * np.pi * day / 365.0)
    day_cos = np.cos(2 * np.pi * day / 365.0)

    gc = df["GC"].values.reshape(-1, 1)
    scaler = StandardScaler()
    gc_norm = scaler.fit_transform(gc).reshape(-1)

    X = np.stack([gc_norm, day_sin, day_cos], axis=1)

    if "anomaly_type" not in df.columns:
        raise ValueError("Column 'anomaly_type' not found — did you call inject_anomalies()?")

    y = df["anomaly_type"].values.astype(int)

    print(f"✅ Built features X shape {X.shape}, labels y shape {y.shape}")
    values, counts = np.unique(y, return_counts=True)
    print("   Class distribution (label -> count):")
    for v, c in zip(values, counts):
        name = {0: "normal", 1: "spike", 2: "seasonal", 3: "outlier"}.get(int(v), "?")
        print(f"   {v} ({name}): {c}")
    return X, y


def partition_clients(
    X: np.ndarray,
    y: np.ndarray,
    lgas: np.ndarray,
    n_clients: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition the dataset into n_clients by LGA blocks as much as possible.
    """
    unique_lgas = np.unique(lgas)
    rng = np.random.default_rng(123)
    rng.shuffle(unique_lgas)

    client_lgas = np.array_split(unique_lgas, n_clients)
    splits = []

    for cid, lga_group in enumerate(client_lgas):
        mask = np.isin(lgas, lga_group)
        X_c = X[mask]
        y_c = y[mask]
        print(
            f"👤 Client {cid}: LGAs={list(lga_group)}, "
            f"samples={len(X_c)}, anomalies={int(y_c.sum())}"
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
            f"(X shape={X_c.shape}, anomalies={int(y_c.sum())})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ausgrid.csv", help="Path to Ausgrid summary CSV")
    parser.add_argument("--clients", type=int, default=3, help="Number of FL clients")
    parser.add_argument("--out", default="processed", help="Output directory")
    parser.add_argument("--anomaly_frac", type=float, default=0.05, help="Fraction of anomalies")
    args = parser.parse_args()

    df_lga = load_ausgrid_summary(args.data)
    df_daily = simulate_daily_series(
        df_lga,
        days=365,
        anomaly_frac=args.anomaly_frac,
        random_state=42,
    )

    X, y = build_features_labels(df_daily)
    lgas = df_daily["LGA"].values

    splits = partition_clients(X, y, lgas, n_clients=args.clients)
    save_clients_npz(splits, out_dir=args.out)

    print("🎉 Preprocessing complete.")


if __name__ == "__main__":
    main()
