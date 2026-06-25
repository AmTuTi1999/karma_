"""
etth1.py
--------
Data preparation script for the ETT-h1 (Electricity Transformer Temperature, hourly)
dataset for time series forecasting with deep learning.

This script:
1. Reads ETTh1.csv from data/raw/, drops the date column -> numpy (T, 7) float32
2. Splits chronologically 70/10/20 (train/val/test)
3. Fits StandardScaler on train only
4. Creates sliding windows (SEQUENCE_LENGTH, PREDICTION_HORIZON)
5. Saves X_train/y_train/X_val/y_val/X_test/y_test + scaler + metadata

Usage:
    python dataset/etth1.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "/workspaces/KARMA-/data/generated/etth1"
CSV_PATH = "/workspaces/KARMA-/data/raw/ETTh1.csv"

SEQUENCE_LENGTH = 24   # 24 hours input
PREDICTION_HORIZON = 1  # Predict next 1 hour

# Chronological split fractions
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
# TEST_FRAC = 0.20  (remainder)

FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
D = 7


def load_data():
    """Load ETTh1.csv, drop date column, return numpy (T, 7) float32."""
    print("Loading ETTh1 data...")
    df = pd.read_csv(CSV_PATH)
    # Drop date column
    df = df.drop(columns=["date"])
    # Enforce feature order
    df = df[FEATURE_COLS]
    data = df.values.astype(np.float32)
    print(f"Loaded data shape: {data.shape}")
    return data


def create_sequences(data, seq_length, pred_horizon):
    """
    Create sliding window sequences for forecasting.

    Args:
        data        : numpy array of shape (T, D)
        seq_length  : input window length
        pred_horizon: number of steps ahead to predict

    Returns:
        X : (N, seq_length, D)
        y : (N, pred_horizon, D)
    """
    T, D = data.shape
    sequences = []
    targets = []

    for i in range(seq_length, T - pred_horizon + 1):
        sequences.append(data[i - seq_length : i])       # (seq_length, D)
        targets.append(data[i : i + pred_horizon])        # (pred_horizon, D)

    X = np.array(sequences)   # (N, seq_length, D)
    y = np.array(targets)     # (N, pred_horizon, D)
    return X, y


def main():
    """Main data preparation pipeline."""
    print("Starting ETTh1 data preparation...")

    # Load raw time series
    data = load_data()
    T = len(data)

    # ── Chronological split on raw time series ──────────────────────────────
    train_end = int(T * TRAIN_FRAC)
    val_end = int(T * (TRAIN_FRAC + VAL_FRAC))

    data_train = data[:train_end]
    data_val = data[train_end:val_end]
    data_test = data[val_end:]

    print(f"Raw split -> train: {len(data_train)}, val: {len(data_val)}, test: {len(data_test)}")

    # ── Fit scaler on training data only ────────────────────────────────────
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(data_train)

    data_train_sc = scaler.transform(data_train).astype(np.float32)
    data_val_sc = scaler.transform(data_val).astype(np.float32)
    data_test_sc = scaler.transform(data_test).astype(np.float32)

    # ── Create sliding windows ───────────────────────────────────────────────
    print("Creating sliding window sequences...")
    X_train, y_train = create_sequences(data_train_sc, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_val, y_val = create_sequences(data_val_sc, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_test, y_test = create_sequences(data_test_sc, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    print(f"Windows -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving processed data to {OUTPUT_DIR} ...")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    import joblib
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    metadata = {
        "name": "etth1",
        "display": "ETT-h1 (Electricity Transformer Temperature, hourly)",
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "n_features": D,
        "feature_names": FEATURE_COLS,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "data_shape": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
        "format": "(N, seq_length, D)",
        "split_fracs": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": round(1 - TRAIN_FRAC - VAL_FRAC, 2)},
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Data preparation completed successfully!")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
