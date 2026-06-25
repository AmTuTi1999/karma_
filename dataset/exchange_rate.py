"""
exchange_rate.py
----------------
Data preparation script for the Exchange Rate dataset for time series forecasting
with deep learning.

Dataset: Exchange Rate (8 currencies)
Source:  https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz
Shape:   ~7588 daily observations, 8 currency pairs

This script:
1. Downloads exchange_rate.txt.gz if not already present
2. Decompresses and reads the data into numpy (T, 8) float32
3. Splits chronologically 70/10/20 (train/val/test)
4. Fits a StandardScaler on training data only
5. Creates sliding windows (SEQUENCE_LENGTH=14, PREDICTION_HORIZON=1)
6. Saves X_train/y_train/X_val/y_val/X_test/y_test + scaler.pkl + metadata.json

Usage:
    python dataset/exchange_rate.py
"""

import gzip
import json
import os
import urllib.request

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR = "/workspaces/KARMA-/data/raw/exchange_rate"
OUTPUT_DIR = "/workspaces/KARMA-/data/generated/exchange_rate"
DATA_URL = (
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data"
    "/master/exchange_rate/exchange_rate.txt.gz"
)
GZ_FILENAME = "exchange_rate.txt.gz"

SEQUENCE_LENGTH = 14   # 2 weeks of daily data
PREDICTION_HORIZON = 1
D = 8                  # 8 currency pairs

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
# TEST_FRAC = 0.20 (remainder)


# ── Download ──────────────────────────────────────────────────────────────────

def download_data():
    """Download exchange_rate.txt.gz to RAW_DIR if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)
    gz_path = os.path.join(RAW_DIR, GZ_FILENAME)
    if os.path.exists(gz_path):
        print(f"Raw file already exists: {gz_path}")
    else:
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, gz_path)
        print(f"Downloaded to {gz_path}")
    return gz_path


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(gz_path: str) -> np.ndarray:
    """
    Decompress and parse exchange_rate.txt.gz.

    Returns
    -------
    data : np.ndarray, shape (T, 8), dtype float32
    """
    print("Reading gzip-compressed data ...")
    rows = []
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split(",")
            if len(values) != D:
                # skip malformed lines
                continue
            rows.append([float(v) for v in values])

    data = np.array(rows, dtype=np.float32)
    print(f"Loaded data shape: {data.shape}")
    return data


# ── Sliding windows ───────────────────────────────────────────────────────────

def create_sequences(data: np.ndarray, seq_length: int, pred_horizon: int):
    """
    Create sliding-window sequences for forecasting.

    Parameters
    ----------
    data        : (T, D)
    seq_length  : input window length
    pred_horizon: number of steps to predict

    Returns
    -------
    X : (N, seq_length, D)
    y : (N, pred_horizon, D)
    """
    T, D = data.shape
    sequences, targets = [], []

    for i in range(seq_length, T - pred_horizon + 1):
        sequences.append(data[i - seq_length : i])          # (seq_length, D)
        targets.append(data[i : i + pred_horizon])          # (pred_horizon, D)

    X = np.array(sequences)  # (N, seq_length, D)
    y = np.array(targets)    # (N, pred_horizon, D)
    return X, y


# ── Normalise ─────────────────────────────────────────────────────────────────

def normalize_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Fit StandardScaler on train, apply to all splits."""
    print("Normalising data ...")

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    def _scale_X(arr):
        return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)

    def _scale_y(arr):
        return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)

    return (
        _scale_X(X_train),
        _scale_y(y_train),
        _scale_X(X_val),
        _scale_y(y_val),
        _scale_X(X_test),
        _scale_y(y_test),
        scaler,
    )


# ── Save ──────────────────────────────────────────────────────────────────────

def save_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    """Persist processed arrays, scaler, and metadata."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving processed data to {OUTPUT_DIR} ...")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    metadata = {
        "dataset": "exchange_rate",
        "display": "Exchange Rate (8 currencies)",
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "n_features": int(X_train.shape[-1]),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "data_shape": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val":   list(X_val.shape),
            "y_val":   list(y_val.shape),
            "X_test":  list(X_test.shape),
            "y_test":  list(y_test.shape),
        },
        "format": "(N, seq_length, D)",
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Data saved.")
    print(f"Metadata: {metadata}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Starting Exchange Rate data preparation ...")

    gz_path = download_data()
    data = load_data(gz_path)          # (T, 8)

    T = len(data)
    train_end = int(T * TRAIN_FRAC)
    val_end   = int(T * (TRAIN_FRAC + VAL_FRAC))

    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    print(f"Split sizes — train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    # Fit scaler on raw training series before windowing so normalisation
    # uses the full training distribution (not just window sub-sequences).
    scaler = StandardScaler()
    scaler.fit(train_data)

    def _scale(arr):
        return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)

    train_scaled = _scale(train_data)
    val_scaled   = _scale(val_data)
    test_scaled  = _scale(test_data)

    # Create sliding windows
    print("Creating sequences ...")
    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_val,   y_val   = create_sequences(val_scaled,   SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_test,  y_test  = create_sequences(test_scaled,  SEQUENCE_LENGTH, PREDICTION_HORIZON)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    save_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler)

    print("Exchange Rate data preparation completed successfully!")


if __name__ == "__main__":
    main()
