"""
exchange_rate_dataset.py
------------------------
PyTorch Dataset class for the prepared Exchange Rate forecasting data.

Loads the preprocessed sliding-window sequences produced by exchange_rate.py
and provides them as PyTorch tensors for model training.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ExchangeRateDataset(Dataset):
    """
    Dataset for Exchange Rate (8 currencies) forecasting.

    Loads preprocessed sequences from numpy files and exposes them as
    (x, y) pairs of PyTorch tensors.

    Parameters
    ----------
    data_dir : str
        Directory containing X_{split}.npy and y_{split}.npy files.
    split : str
        Which split to load: 'train', 'val', or 'test'.
    """

    def __init__(
        self,
        data_dir: str = "/workspaces/KARMA-/data/generated/exchange_rate",
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.split = split

        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy")).astype(np.float32)
        self.y = np.load(os.path.join(data_dir, f"y_{split}.npy")).astype(np.float32)

        print(f"Loaded {split} data: X shape {self.X.shape}, y shape {self.y.shape}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # (seq_length, D)
        y = torch.from_numpy(self.y[idx])  # (pred_horizon, D)
        return x, y


def get_dataloaders(
    data_dir: str = "/workspaces/KARMA-/data/generated/exchange_rate",
    batch_size: int = 32,
    num_workers: int = 2,
):
    """
    Create PyTorch DataLoaders for train/val/test splits.

    Parameters
    ----------
    data_dir : str
        Directory containing the preprocessed data files.
    batch_size : int
        Batch size for all DataLoaders.
    num_workers : int
        Number of worker processes for data loading.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    train_dataset = ExchangeRateDataset(data_dir, "train")
    val_dataset   = ExchangeRateDataset(data_dir, "val")
    test_dataset  = ExchangeRateDataset(data_dir, "test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = ExchangeRateDataset()
    print(f"Dataset length: {len(dataset)}")

    x, y = dataset[0]
    print(f"Sample X shape: {x.shape}, y shape: {y.shape}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)
    print(
        f"Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)}"
    )

    for batch_x, batch_y in train_loader:
        print(f"Batch X shape: {batch_x.shape}, Batch y shape: {batch_y.shape}")
        break
