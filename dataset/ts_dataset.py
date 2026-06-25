import torch
import numpy as np
from torch.utils.data import Dataset


class MVTSDataset(Dataset):
    """
    Sliding-window dataset for multivariate time series.

    For each valid time point t, extracts:
        X_window : (D, T_input)   — input to TCN (channels-first)
        y        : int            — 1 if X[t+1, target_var] > 0 else 0

    Parameters
    ----------
    data        : np.ndarray  (T, D)
    T_input     : input window length  W (must equal TCN receptive field input)
    target_var  : which variable to predict (0 = SPX)
    """

    def __init__(self, data, T_input=30, target_var=0):
        self.data = data.astype(np.float32)
        self.T_input = T_input
        self.target_var = target_var
        # valid indices: need T_input past + 1 future
        self.indices = list(range(T_input, len(data) - 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        window = self.data[t - self.T_input : t]  # (T_input, D)
        x = torch.from_numpy(window.T)  # (D, T_input)
        y_raw = self.data[t + 1, self.target_var]  # next-step value
        y = torch.tensor(float(y_raw > 0))  # binary label
        return x, y
