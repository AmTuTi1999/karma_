#!/usr/bin/env python
"""
experiments/comparison_realdata.py
────────────────────────────────────────────────────────────────────────────────
AUC and prediction-change comparison on real forecasting datasets.

Methods
-------
  KARMA    – edge-removal AUC (edges ranked by ρ; skipped when D > MAX_KARMA_D)
  FO       – feature-occlusion attribution, feature-time removal AUC
  DynaMask – differentiable mask, MSE loss, feature-time removal AUC
  WinIT    – windowed counterfactual, feature-time removal AUC

Datasets
--------
  etth1, exchange_rate, beijing_pm25

AUC metric
----------
  At step k (k=1..n_steps), remove the top-(k/n_steps) fraction of features
  in descending importance order (replace with training-set mean).  Measure
  mean absolute prediction change.  AUC = trapezoidal integral over that curve.

  For KARMA  : "features" are edges (src_var, lag) ranked by ρ.
  For others : (feature_idx, time_idx) pairs ranked by |attribution|.

Usage
-----
  (poetry run) python -m experiments.comparison_realdata --load_existing
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from captum.attr import IntegratedGradients
from timeshap.explainer.kernel import TimeShapKernel

from karma.causal_recovery.edge_contribution import compute_variable_importance
from karma.markov_approximation.markov_surrogacy import select_K_and_baseline
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import TreeKernelEstimator
from karma.utils.sampling import SuffixPool

_ROOT = Path(__file__).parent.parent
_WINIT = _ROOT / "WinIT"
sys.path.insert(0, str(_WINIT))
sys.path.insert(0, str(_ROOT))


from winit.explainer.attribution.mask_group import MaskGroup
from winit.explainer.dynamaskexplainer import DynamaskExplainer
from winit.explainer.explainers import FOExplainer
from winit.explainer.winitexplainers import WinITExplainer
from winit.models import TorchModel


@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    D: int
    T: int  # sequence_length
    pred_horizon: int
    ckpt_prefix: str = ""  # folder prefix for LSTM/TCN checkpoints; defaults to name
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    num_epochs: int = 50
    # KARMA params
    N: int = 2
    eps: float = 0.05
    lam: float = 0.01
    M: int = 100
    n_pool_min: int = 3
    K_max: int = 3

    def __post_init__(self):
        if not self.ckpt_prefix:
            self.ckpt_prefix = self.name


DATASETS: dict[str, DatasetConfig] = {
    "etth1": DatasetConfig(
        name="etth1",
        data_dir="data/generated/etth1",
        D=7,
        T=24,
        pred_horizon=1,
        M=100,
        K_max=3,
    ),
    "exchange_rate": DatasetConfig(
        name="exchange_rate",
        data_dir="data/generated/exchange_rate",
        D=8,
        T=14,
        pred_horizon=1,
        M=100,
        K_max=10,
    ),
    "beijing_pm25": DatasetConfig(
        name="beijing_pm25",
        data_dir="data/generated/beijing_pm25",
        D=11,
        T=24,
        pred_horizon=1,
        M=100,
        K_max=10,
        ckpt_prefix="bpm25",
    ),
}

MAX_KARMA_D = 101  # skip KARMA for D > this (state space too large)
N_AUC_STEPS = 10  # progressive removal steps
N_TEST_MAX = 200  # max test samples for attribution (capped for speed)
N_TEST_LARGE_D = 50  # further reduced cap when D > 20


class GRUForecastModel(TorchModel):
    """GRU regression model supporting single- and multi-step forecasting.

    forward(x: (B, D, T), return_all=False) → (B, D)       [next-step, all callers]
    forward(x: (B, D, T), return_all=True)  → (B, D, T)    [rolling, WinIT/DynaMask]
    predict_multistep(x: (B, D, T))         → (B, H, D)    [KARMA multi-step oracle]
    """

    def __init__(
        self,
        D: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        device="cpu",
    ):
        """GRU Forecast Model

        Args:
            D (int): input dimension
            hidden_size (int, optional): hidden size. Defaults to 64.
            num_layers (int, optional): number of layers. Defaults to 2.
            dropout (float, optional): dropout rate. Defaults to 0.1.
            forecast_steps (int, optional): number of forecast steps. Defaults to 1.
            device (str, optional): device. Defaults to "cpu".
        """
        super().__init__(
            feature_size=D, num_states=D, hidden_size=hidden_size, device=device
        )
        self.D = D
        self.forecast_steps = forecast_steps
        self.gru = nn.GRU(
            D,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, forecast_steps * D)
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """GRU forward function

        Args:
            x (torch.Tensor): input tensor
            return_all (bool, optional): whether to return all steps. Defaults to False.

        Returns:
            torch.Tensor: predicted output tensor
        """
        x_td = x.permute(0, 2, 1)  # (B, T, D)
        out, _ = self.gru(x_td)  # (B, T, hidden)
        if return_all:
            return self.fc(out)[..., : self.D].permute(0, 2, 1)  # (B, D, T)
        return self.fc(out[:, -1, :])[..., : self.D]  # (B, D)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, forecast_steps, D) from the last hidden state."""
        x_td = x.permute(0, 2, 1)
        out, _ = self.gru(x_td)
        raw = self.fc(out[:, -1, :])  # (B, forecast_steps * D)
        return raw.view(x.shape[0], self.forecast_steps, self.D)


class LSTMForecastModel(TorchModel):
    """Stacked LSTM with LayerNorm head.  Matches checkpoint architecture.

    forward(x: (B, D, T)) → (B, D)  [return_all=False, next-step prediction]
    forward(x: (B, D, T)) → (B, D, T)  [return_all=True, prediction at every t]
    """

    def __init__(
        self,
        D: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        device="cpu",
    ):
        """LSTM Forecast Model

        Args:
            D (int): input dimension
            hidden_size (int, optional): hidden size. Defaults to 64.
            num_layers (int, optional): number of layers. Defaults to 2.
            dropout (float, optional): dropout rate. Defaults to 0.1.
            forecast_steps (int, optional): number of forecast steps. Defaults to 1.
            device (str, optional): device. Defaults to "cpu".
        """
        super().__init__(
            feature_size=D, num_states=D, hidden_size=hidden_size, device=device
        )
        self.forecast_steps = forecast_steps
        self.D = D
        self.lstm = nn.LSTM(
            D,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_steps * D),
        )
        self.activation = nn.Identity()

    def _apply_head(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden) → (..., D)  [first forecast step only]"""
        out = self.head(self.dropout_layer(self.layer_norm(h)))
        return out[..., : self.D]  # take first forecast step's D outputs

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        x_td = x.permute(0, 2, 1)  # (B, T, D)
        lstm_out, _ = self.lstm(x_td)  # (B, T, hidden)
        if return_all:
            preds = self._apply_head(lstm_out)  # (B, T, D)
            return preds.permute(0, 2, 1)  # (B, D, T)
        return self._apply_head(lstm_out[:, -1, :])  # (B, D)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, forecast_steps, D) from the last hidden state."""
        x_td = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_td)
        h = self.dropout_layer(self.layer_norm(lstm_out[:, -1, :]))
        raw = self.head(h)  # (B, forecast_steps * D)
        return raw.view(x.shape[0], self.forecast_steps, self.D)


class _Chomp1d(nn.Module):
    """chomp tensor module to remove right padding for causality

    Args:
        nn : torch.nn.Module
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """TCN temporal block

    Args:
        nn (_type_): torch.nn.Module
    """

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float
    ):
        """TCN block initialization

        Args:
            in_ch (int): in channels
            out_ch (int): out channels
            kernel_size (int): kernel size
            dilation (int): dilation
            dropout (float): dropout rate
        """
        super().__init__()
        pad = (kernel_size - 1) * dilation
        from torch.nn.utils import weight_norm as wn

        self.conv1 = wn(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.conv2 = wn(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.net = nn.Sequential(
            self.conv1,
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)


class TCNForecastModel(TorchModel):
    """Dilated causal TCN.  Matches checkpoint architecture.

    forward(x: (B, D, T)) → (B, D)  [return_all=False]
    forward(x: (B, D, T)) → (B, D, T)  [return_all=True]
    """

    def __init__(
        self,
        D: int,
        num_channels: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        device="cpu",
    ):
        num_channels = num_channels or [64, 64, 128]
        super().__init__(
            feature_size=D, num_states=D, hidden_size=num_channels[-1], device=device
        )
        self.forecast_steps = forecast_steps
        self.D = D
        blocks = []
        for i, out_ch in enumerate(num_channels):
            in_ch = D if i == 0 else num_channels[i - 1]
            blocks.append(
                _TemporalBlock(
                    in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout
                )
            )
        self.network = nn.Sequential(*blocks)
        last_ch = num_channels[-1]
        self.head = nn.Sequential(
            nn.Linear(last_ch, last_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_ch, forecast_steps * D),
        )
        self.activation = nn.Identity()

    def _apply_head(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., channels) → (..., D)"""
        out = self.head(h)
        return out[..., : self.D]

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        y = self.network(x)  # (B, channels, T)
        if return_all:
            h = y.permute(0, 2, 1)  # (B, T, channels)
            preds = self._apply_head(h)  # (B, T, D)
            return preds.permute(0, 2, 1)  # (B, D, T)
        h_last = y[:, :, -1]  # (B, channels)
        return self._apply_head(h_last)  # (B, D)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, forecast_steps, D) from the last temporal position."""
        y = self.network(x)
        raw = self.head(y[:, :, -1])  # (B, forecast_steps * D)
        return raw.view(x.shape[0], self.forecast_steps, self.D)


def _mse_loss_multiple(Y_pred: torch.Tensor, Y_target: torch.Tensor) -> torch.Tensor:
    """MSE per sample: (N_area, B, T, D) → (B,). Mirrors cross_entropy_multiple."""
    return ((Y_pred - Y_target) ** 2).mean(dim=[0, 2, 3])


class MSEDynaMaskExplainer(DynamaskExplainer):
    """DynaMask with MSE loss for regression forecasting tasks."""

    def __init__(self, device, **kwargs):
        kwargs.pop("loss", None)
        super().__init__(device=device, loss="ce", **kwargs)
        self.loss = _mse_loss_multiple
        self.loss_str = "mse"

    def _attribute_multiple(self, x: torch.Tensor) -> np.ndarray:
        # x: (B, D, T)
        orig_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        def f(x_in: torch.Tensor) -> torch.Tensor:
            # x_in: (B, T, D) — DynaMask internal convention
            x_perm = x_in.permute(0, 2, 1)  # (B, D, T)
            out = self.base_model(x_perm, return_all=True).permute(0, 2, 1)  # (B, T, D)
            if self.use_last_timestep_only:
                return out[:, -1:, :]  # (B, 1, D)
            return out  # (B, T, D)

        x_td = x.permute(0, 2, 1)  # (B, T, D)

        mask_group = MaskGroup(
            self.pert, self.device, verbose=False, deletion_mode=self.deletion_mode
        )
        mask_group.fit_multiple(
            X=x_td,
            f=f,
            use_last_timestep_only=self.use_last_timestep_only,
            loss_function_multiple=self.loss,
            area_list=self.area_list,
            learning_rate=1.0,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=self.size_reg_factor_dilation,
            initial_mask_coeff=0.5,
            n_epoch=self.num_epoch,
            momentum=1.0,
            time_reg_factor=self.time_reg_factor,
        )

        y_orig = f(x_td)
        y_zero = f(torch.zeros_like(x_td))
        if self.use_last_timestep_only:
            diff = y_orig[:, -1:, :] - y_zero[:, -1:, :]
        else:
            diff = y_orig - y_zero
        thresh = (diff**2).mean(dim=[-1, -2]) * 0.5  # (B,)

        mask = mask_group.get_extremal_mask_multiple(thresholds=thresh)
        mask_saliency = mask.permute(0, 2, 1)  # (B, D, T)

        torch.backends.cudnn.enabled = orig_cudnn
        return mask_saliency.detach().cpu().numpy()


def load_dataset(cfg: DatasetConfig):
    base = Path(cfg.data_dir)
    X_train = np.load(base / "X_train.npy")  # (N, T, D)
    X_val = np.load(base / "X_val.npy")
    X_test = np.load(base / "X_test.npy")
    y_train = np.load(base / "y_train.npy")  # (N, pred_horizon, D)
    y_val = np.load(base / "y_val.npy")
    y_test = np.load(base / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def reconstruct_raw_series(X_windows: np.ndarray) -> np.ndarray:
    """Reconstruct raw (T_total, D) time series from (N, W, D) windows.

    Windows are assumed to be consecutive (stride=1).  X_windows[0] gives the
    first W values; each subsequent window adds one new observation at the end.
    """
    first = X_windows[0]  # (W, D)
    rest = X_windows[1:, -1, :]  # (N-1, D)
    return np.concatenate([first, rest], axis=0)  # (W + N - 1, D)


def train_gru(
    model: GRUForecastModel,
    X_train: np.ndarray,  # (N, T, D)
    y_train: np.ndarray,  # (N, pred_horizon, D)
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DatasetConfig,
    device,
    ckpt_path: Optional[Path] = None,
) -> GRUForecastModel:
    multistep = int(y_train.shape[1]) > 1  # driven by actual data, not cfg
    X_tr = torch.from_numpy(X_train.transpose(0, 2, 1).astype(np.float32))  # (N, D, T)
    y_tr = torch.from_numpy(
        y_train.astype(np.float32) if multistep else y_train[:, 0, :].astype(np.float32)
    )  # (N, actual_horizon, D) or (N, D)
    X_v = torch.from_numpy(X_val.transpose(0, 2, 1).astype(np.float32))
    y_v = torch.from_numpy(
        y_val.astype(np.float32) if multistep else y_val[:, 0, :].astype(np.float32)
    )

    tr_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=cfg.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    best_val = float("inf")
    best_state = None
    patience = 10

    model = model.to(device)
    for epoch in range(cfg.num_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = (
                model.predict_multistep(xb)
                if multistep
                else model(xb, return_all=False)
            )
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = (
                    model.predict_multistep(xb)
                    if multistep
                    else model(xb, return_all=False)
                )
                val_losses.append(F.mse_loss(pred, yb).item())
        vl = float(np.mean(val_losses))
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 10
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            patience -= 1
            if patience == 0:
                break

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}: val_mse={vl:.5f}")

    model.load_state_dict(best_state)
    return model


def get_or_train_gru(
    cfg: DatasetConfig,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    ckpt_dir: Path,
) -> GRUForecastModel:
    ckpt_path = ckpt_dir / f"gru_{cfg.name}.pt"
    actual_horizon = int(y_train.shape[1])  # ground truth from data, not cfg
    model = GRUForecastModel(
        D=cfg.D,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        forecast_steps=actual_horizon,
        device=device,
    )
    if ckpt_path.exists():
        print(f"  Loading GRU from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
    else:
        print(f"  Training GRU ({cfg.num_epochs} epochs max)…")
        model = train_gru(model, X_train, y_train, X_val, y_val, cfg, device, ckpt_path)
    model.eval()
    return model


def load_pretrained_lstm(cfg: DatasetConfig, device, model_ckpt_dir: Path):
    """Load LSTM from outputs/checkpoints/{ckpt_prefix}_lstm/best.pt."""
    ckpt_path = model_ckpt_dir / f"{cfg.ckpt_prefix}_lstm" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    hidden_size = state["lstm.weight_hh_l0"].shape[1]
    num_layers = sum(1 for k in state if k.startswith("lstm.weight_hh"))
    forecast_steps = state["head.3.weight"].shape[0] // cfg.D
    model = LSTMForecastModel(
        D=cfg.D,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        forecast_steps=forecast_steps,
        device=device,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(
        f"  Loaded LSTM from {ckpt_path}  (hidden={hidden_size}, forecast_steps={forecast_steps})"
    )
    return model


def load_pretrained_tcn(cfg: DatasetConfig, device, model_ckpt_dir: Path):
    """Load TCN from outputs/checkpoints/{ckpt_prefix}_tcn/best.pt."""
    ckpt_path = model_ckpt_dir / f"{cfg.ckpt_prefix}_tcn" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TCN checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    n_blocks = max(int(k.split(".")[1]) for k in state if k.startswith("network.")) + 1
    num_channels = [state[f"network.{i}.conv2.bias"].shape[0] for i in range(n_blocks)]
    uses_weight_norm = any(k.endswith(".weight_v") for k in state)
    wkey = "weight_v" if uses_weight_norm else "weight"
    kernel_size = state[f"network.0.conv1.{wkey}"].shape[2]
    forecast_steps = state["head.3.weight"].shape[0] // cfg.D
    model = TCNForecastModel(
        D=cfg.D,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=0.0,
        forecast_steps=forecast_steps,
        device=device,
    )
    if not uses_weight_norm:
        # Checkpoint was saved without weight_norm; strip it from the model so keys match
        for block in model.network:
            nn.utils.remove_weight_norm(block.conv1)
            nn.utils.remove_weight_norm(block.conv2)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(
        f"  Loaded TCN from {ckpt_path}  (channels={num_channels}, kernel={kernel_size}, forecast_steps={forecast_steps})"
    )
    return model


def _orig_preds(x_test: np.ndarray, model: GRUForecastModel, device) -> np.ndarray:
    with torch.no_grad():
        return (
            model(
                torch.from_numpy(x_test.astype(np.float32)).to(device), return_all=False
            )
            .cpu()
            .numpy()
        )


def select_var_order(x_train: np.ndarray, max_K: int = 8) -> int:
    """BIC-based VAR order selection.  x_train: (N, D, T)."""
    N, D, T = x_train.shape
    rng = np.random.default_rng(0)
    xtr = x_train[rng.choice(N, size=min(N, 500), replace=False)]
    n = len(xtr)
    best_K, best_bic = 1, np.inf
    for K in range(1, min(max_K + 1, T // 2)):
        X_in = np.array(
            [xtr[i, :, t - K : t].T.flatten() for i in range(n) for t in range(K, T)],
            dtype=np.float32,
        )
        X_out = np.array(
            [xtr[i, :, t] for i in range(n) for t in range(K, T)], dtype=np.float32
        )
        lr = Ridge(alpha=1e-3).fit(X_in, X_out)
        rss = float(np.sum((X_out - lr.predict(X_in)) ** 2))
        n_obs = X_in.shape[0] * D
        bic = n_obs * np.log(max(rss / n_obs, 1e-12)) + (K * D * D + D) * np.log(n_obs)
        if bic < best_bic:
            best_bic, best_K = bic, K
    return best_K


def fit_var_imputer(x_train: np.ndarray, K: int) -> Ridge:
    """Fit VAR(K) Ridge regressor for conditional imputation.  x_train: (N, D, T)."""
    N, D, T = x_train.shape
    xtr = x_train[np.random.default_rng(1).choice(N, size=min(N, 2000), replace=False)]
    n = len(xtr)
    X_in = np.array(
        [xtr[i, :, t - K : t].T.flatten() for i in range(n) for t in range(K, T)],
        dtype=np.float32,
    )
    X_out = np.array(
        [xtr[i, :, t] for i in range(n) for t in range(K, T)], dtype=np.float32
    )
    return Ridge(alpha=1e-3).fit(X_in, X_out)


def _impute_col(
    x: np.ndarray,  # (B, D, T) — partially masked window
    t_idx: int,
    var_imputer: Ridge | None,
    var_K: int,
    fallback: np.ndarray,  # (D,) marginal training mean for this column
    karma_b_star: np.ndarray | None = None,  # (T-K*, D) KARMA certified baseline
) -> np.ndarray:  # (B, D)
    """Impute a single time column using the best available method."""
    B, D, _ = x.shape

    if karma_b_star is not None and t_idx < len(karma_b_star):
        return np.broadcast_to(karma_b_star[t_idx], (B, D)).copy()
    if var_imputer is not None and t_idx >= var_K:
        ctx = x[:, :, t_idx - var_K : t_idx]  # (B, D, K)
        ctx_flat = ctx.transpose(0, 2, 1).reshape(B, var_K * D)  # (B, K*D) time-major
        return var_imputer.predict(ctx_flat).astype(np.float32)
    return np.broadcast_to(fallback, (B, D)).copy()


def attr_to_time_scores(attr: np.ndarray) -> np.ndarray:
    """(B, D, T) → (T,): mean |attribution| over batch and features per timestep."""
    return np.abs(attr).mean(axis=(0, 1))


def karma_edges_to_time_scores(edges: list, T: int) -> np.ndarray:
    """Sum ρ values per time index from KARMA edges.

    Edge with lag l maps to time index T-l.  Sums ρ across all edges sharing a
    time index (so a timestep targeted by many causal edges scores higher).
    """
    scores = np.zeros(T, dtype=np.float32)
    for e in edges:
        t_idx = T - e["lag"]
        if 0 <= t_idx < T:
            scores[t_idx] += e["rho"]
    return scores


def timestep_removal_auc(
    x_test: np.ndarray,  # (B, D, T)
    time_scores: np.ndarray,  # (T,) importance per time index
    model,
    x_train_mean: np.ndarray,  # (D, T)
    device,
    n_steps: int = N_AUC_STEPS,
    var_imputer: Ridge | None = None,
    var_K: int = 0,
    karma_b_star: np.ndarray | None = None,  # (T-K*, D) KARMA certified baseline
) -> tuple[float, list[float]]:
    """AUC under progressive whole-timestep removal conditioned on the Markov blanket.

    Each masked timestep is imputed via VAR(K) conditional on its K predecessors
    (preserving temporal correlation) rather than replaced with the marginal mean.
    For KARMA, positions outside the identified blanket use the certified b* baseline.
    """
    _, D, T = x_test.shape
    sorted_t = np.argsort(time_scores)[::-1]  # most important first

    orig = _orig_preds(x_test, model, device)
    changes = []
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            k = max(1, step * T // n_steps)
            x_masked = x_test.copy()
            # Impute in temporal order so each position can condition on
            # already-imputed predecessors (AR-consistent chain).
            for t_idx in sorted(sorted_t[:k]):
                x_masked[:, :, t_idx] = _impute_col(
                    x_masked,
                    t_idx,
                    var_imputer,
                    var_K,
                    x_train_mean[:, t_idx],
                    karma_b_star,
                )
            new_pred = (
                model(
                    torch.from_numpy(x_masked.astype(np.float32)).to(device),
                    return_all=False,
                )
                .cpu()
                .numpy()
            )
            changes.append(float(np.abs(orig - new_pred).mean()))

    xs = np.linspace(1 / n_steps, 1.0, n_steps)
    return float(np.trapz(changes, xs)), changes


def timestep_drop_at_frac(
    x_test: np.ndarray,
    time_scores: np.ndarray,
    model,
    x_train_mean: np.ndarray,
    device,
    frac: float = 0.25,
    var_imputer: Ridge | None = None,
    var_K: int = 0,
    karma_b_star: np.ndarray | None = None,
) -> float:
    """Mean |Δpred| after blanking the top-frac fraction of timesteps."""
    _, D, T = x_test.shape
    k = max(1, int(round(frac * T)))
    sorted_t = np.argsort(time_scores)[::-1]
    x_masked = x_test.copy()
    for t_idx in sorted(sorted_t[:k]):
        x_masked[:, :, t_idx] = _impute_col(
            x_masked,
            t_idx,
            var_imputer,
            var_K,
            x_train_mean[:, t_idx],
            karma_b_star,
        )
    orig = _orig_preds(x_test, model, device)
    with torch.no_grad():
        new_pred = (
            model(
                torch.from_numpy(x_masked.astype(np.float32)).to(device),
                return_all=False,
            )
            .cpu()
            .numpy()
        )
    return float(np.abs(orig - new_pred).mean())


def karma_edges_to_attr(edges: list, D: int, T: int) -> np.ndarray:
    """Convert KARMA edges to a (D, T) attribution matrix.

    Edge {"src": s, "lag": l, "rho": r} maps to window position (s, T-l).
    Multiple edges sharing the same (src, lag) are combined by max ρ.
    Returns a (D, T) array broadcastable to (B, D, T) for feature_removal_auc.
    """
    attr = np.zeros((D, T), dtype=np.float32)
    for e in edges:
        src, lag, rho = e["src"], e["lag"], e["rho"]
        t_idx = T - lag  # lag=1 → last timestep, lag=2 → second-to-last, …
        if 0 <= t_idx < T:
            attr[src, t_idx] = max(attr[src, t_idx], rho)
    return attr


def run_fo(x_test: np.ndarray, model: GRUForecastModel, device) -> np.ndarray:
    """FO attribution → (B, D, T) numpy."""
    fo = FOExplainer(device=device)
    fo.set_model(model)
    attr = fo.attribute(torch.from_numpy(x_test.astype(np.float32)).to(device))
    if isinstance(attr, torch.Tensor):
        attr = attr.detach().cpu().numpy()
    return attr


def run_dynamask(
    x_test: np.ndarray,
    model: GRUForecastModel,
    device,
    num_epoch: int = 100,
) -> np.ndarray:
    """MSE DynaMask attribution → (B, D, T) numpy."""
    dyn = MSEDynaMaskExplainer(
        device=device,
        area_list=np.arange(0.25, 0.36, 0.05),
        num_epoch=num_epoch,
        use_last_timestep_only=True,
    )
    dyn.set_model(model)
    attr = dyn.attribute(torch.from_numpy(x_test.astype(np.float32)).to(device))
    if isinstance(attr, torch.Tensor):
        attr = attr.detach().cpu().numpy()
    return attr


def run_ig(x_test: np.ndarray, model: GRUForecastModel, device) -> np.ndarray:
    """Integrated Gradients → (B, D, T).  Zero baseline, sum over D outputs."""
    model.eval()

    def _scalar_forward(x: torch.Tensor) -> torch.Tensor:
        return model(x, return_all=False).sum(dim=-1)  # (B,)

    ig = IntegratedGradients(_scalar_forward)
    x_t = torch.from_numpy(x_test.astype(np.float32)).to(device)
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    attr = ig.attribute(x_t, baselines=torch.zeros_like(x_t))  # (B, D, T)
    torch.backends.cudnn.enabled = orig_cudnn
    return np.abs(attr.detach().cpu().numpy())


def run_timeshap(
    x_test: np.ndarray,  # (B, D, T)
    model: GRUForecastModel,
    device,
    nsamples: int = 200,
) -> np.ndarray:
    """TimeShap cell-level attribution → (B, D, T)."""
    B, D_feat, T = x_test.shape
    x_td = x_test.transpose(0, 2, 1).astype(np.float32)  # (B, T, D)
    background = np.zeros((1, T, D_feat), dtype=np.float32)
    model.eval()

    # LassoLarsIC (used inside TimeShap's kernel solver) requires
    # nsamples > n_features = T * D.  Silently floor up when needed.
    n_features = T * D_feat
    if nsamples <= n_features:
        nsamples_eff = n_features + max(50, n_features // 4)
        print(
            f"    [TimeShap] nsamples={nsamples} < n_features={n_features}; "
            f"raising to {nsamples_eff}"
        )
        nsamples = nsamples_eff

    def model_fn(x: np.ndarray) -> np.ndarray:
        # x: (N, T, D) numpy → (N, D, T) torch → scalar (N,)
        x_dt = torch.from_numpy(x.transpose(0, 2, 1)).to(device)
        with torch.no_grad():
            return model(x_dt, return_all=False).mean(dim=-1).cpu().numpy()

    varying = (list(range(T)), list(range(D_feat)))
    attr = np.zeros((B, D_feat, T), dtype=np.float32)
    for b in range(B):
        kernel = TimeShapKernel(
            model_fn, background, rs=42, mode="cell", varying=varying
        )
        sv = kernel.shap_values(x_td[b : b + 1], pruning_idx=0, nsamples=nsamples)
        # sv: (T*D,) ordered (t0_d0, t0_d1, …, t1_d0, …) → reshape (T, D) → (D, T)
        attr[b] = np.abs(sv.reshape(T, D_feat).T)
    return attr


def run_winit(
    x_test: np.ndarray,  # (B, D, T)
    X_tr_dt: np.ndarray,  # (N_train, D, T)
    X_val_dt: np.ndarray,  # (N_val, D, T)
    model: GRUForecastModel,
    cfg: DatasetConfig,
    ckpt_dir: Path,
    device,
    num_epochs: int = 100,
    load_existing: bool = False,
) -> np.ndarray:
    """WinIT attribution → (B, D, T) numpy (max over window_size dim)."""
    dummy_tr = torch.zeros(len(X_tr_dt))
    dummy_val = torch.zeros(len(X_val_dt))
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_dt.astype(np.float32)), dummy_tr),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_dt.astype(np.float32)), dummy_val),
        batch_size=64,
    )
    winit = WinITExplainer(
        device=device,
        num_features=cfg.D,
        data_name=cfg.name,
        path=ckpt_dir,
        window_size=min(cfg.T, 5),
        num_samples=3,
        metric="pd",
        random_state=42,
    )
    winit.set_model(model)
    if load_existing:
        try:
            winit.load_generators()
            print("  Loaded pre-trained WinIT generators")
        except Exception:
            print("  No saved generators found — training…")
            winit.train_generators(tr_loader, val_loader, num_epochs=num_epochs)
    else:
        winit.train_generators(tr_loader, val_loader, num_epochs=num_epochs)

    attr = winit.attribute(torch.from_numpy(x_test.astype(np.float32)).to(device))
    if isinstance(attr, torch.Tensor):
        attr = attr.detach().cpu().numpy()
    # WinIT returns (B, D, T, window_size) — collapse last dim to (B, D, T)
    if attr.ndim == 4:
        attr = np.abs(attr).max(axis=-1)
    return attr


def run_karma(
    X_train: np.ndarray,  # (N, T, D) windowed
    X_val: np.ndarray,  # (N, T, D) windowed
    model: GRUForecastModel,
    cfg: DatasetConfig,
    device,
    verbose: bool = True,
) -> tuple[list, int, np.ndarray]:
    """KARMA pipeline.  Returns (edges, K_star, b_star).

    b_star: (W-K*, D) certified baseline for non-blanket prefix positions.
    """
    X_train_raw = reconstruct_raw_series(X_train)  # (T_raw, D)
    X_val_raw = reconstruct_raw_series(X_val)
    W = cfg.T  # use dataset window size as KARMA oracle window

    model.eval()

    def f_oracle(window: np.ndarray) -> np.ndarray:
        is_single = window.ndim == 2
        w = window[np.newaxis] if is_single else window  # (B, W, D)
        x = torch.from_numpy(w.transpose(0, 2, 1).astype(np.float32)).to(device)
        with torch.no_grad():
            if getattr(model, "forecast_steps", 1) > 1:
                pred = (
                    model.predict_multistep(x).cpu().numpy()
                )  # (B, forecast_steps, D)
            else:
                pred = model(x, return_all=False).cpu().numpy()  # (B, D)
        return pred[0] if is_single else pred

    disc = Discretiser(N=cfg.N)
    disc.fit(X_train_raw)

    T_val_raw = len(X_val_raw)
    X_val_windows = np.stack([X_val_raw[t : t + W] for t in range(T_val_raw - W)])

    result = select_K_and_baseline(
        f=f_oracle,
        X_train=X_train_raw,
        X_val=X_val_windows,
        disc=disc,
        W=W,
        eps=cfg.eps,
        K_max=cfg.K_max,
        loss="regression",
        verbose=verbose,
    )
    K_star = result["K_star"]
    b_star = result["b_star"]
    pi_star = result["pi_star"]
    if verbose:
        print(f"  K* = {K_star},  Δ_pred = {result['delta_pred']:.4f}")

    pool = SuffixPool(disc=disc, K=K_star, W=W)
    pool.build(X_train_raw)

    tree = TreeKernelEstimator(
        disc, K=K_star, W=W, M=cfg.M, n_pool=cfg.n_pool_min, pool=pool, b_star=b_star
    )
    tree.fit(f=f_oracle, pi_star=pi_star, verbose=verbose)

    vi = compute_variable_importance(tree, pi_star, disc, lam=cfg.lam)
    return vi["edges"], K_star, b_star


def _run_methods_for_model(
    arch: str,
    model,
    X_test_dt: np.ndarray,
    X_tr_dt: np.ndarray,
    X_val_dt: np.ndarray,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: "DatasetConfig",
    args,
    ckpt_dir: Path,
    x_train_mean: np.ndarray,
    device: torch.device,
    var_imputer: Ridge | None = None,
    var_K: int = 0,
) -> dict:
    """Run all attribution methods for one model; keys are prefixed with arch."""
    r: dict = {}
    p = f"{arch}_"  # key prefix
    T = X_test_dt.shape[2]

    def _lag_auc(ts, karma_b_star=None):
        return timestep_removal_auc(
            X_test_dt,
            ts,
            model,
            x_train_mean,
            device,
            var_imputer=var_imputer,
            var_K=var_K,
            karma_b_star=karma_b_star,
        )

    def _lag_d25(ts, karma_b_star=None):
        return timestep_drop_at_frac(
            X_test_dt,
            ts,
            model,
            x_train_mean,
            device,
            frac=0.25,
            var_imputer=var_imputer,
            var_K=var_K,
            karma_b_star=karma_b_star,
        )

    def _store(m, attr, time_scores):
        lag_auc, lag_ch = _lag_auc(time_scores)
        lag_d25 = _lag_d25(time_scores)
        r.update(
            {
                f"{p}{m}_lag_auc": lag_auc,
                f"{p}{m}_lag_changes": lag_ch,
                f"{p}{m}_lag_drop25": lag_d25,
            }
        )
        msg = f"    lag_AUC={lag_auc:.4f}  lag_drop@25%={lag_d25:.4f}"
        print(msg)

    if not args.skip_fo:
        print("\n  [FO]")
        attr = run_fo(X_test_dt, model, device)
        _store("fo", attr, attr_to_time_scores(attr))

    if not args.skip_dynamask:
        print("\n  [DynaMask]")
        attr = run_dynamask(X_test_dt, model, device, num_epoch=args.dynamask_epochs)
        _store("dynamask", attr, attr_to_time_scores(attr))

    if not args.skip_winit:
        print("\n  [WinIT]")
        winit_ckpt = ckpt_dir / f"winit_{cfg.name}_{arch}"
        attr = run_winit(
            X_test_dt,
            X_tr_dt.astype(np.float32),
            X_val_dt.astype(np.float32),
            model,
            cfg,
            winit_ckpt,
            device,
            num_epochs=args.winit_epochs,
            load_existing=args.load_existing,
        )
        _store("winit", attr, attr_to_time_scores(attr))

    if not args.skip_karma and cfg.D <= MAX_KARMA_D:
        print("\n  [KARMA]")
        edges, K_star, b_star = run_karma(
            X_train, X_val, model, cfg, device, verbose=not args.quiet
        )
        # Use raw edge ρ sums per lag for the temporal metric (purer lag signal).
        # Pass b_star so non-blanket positions use KARMA's certified baseline.
        ts_k = karma_edges_to_time_scores(edges, T)
        lag_auc, lag_ch = _lag_auc(ts_k)
        lag_d25 = _lag_d25(ts_k)
        r.update(
            {
                f"{p}karma_lag_auc": lag_auc,
                f"{p}karma_lag_changes": lag_ch,
                f"{p}karma_lag_drop25": lag_d25,
                f"{p}karma_K_star": K_star,
                f"{p}karma_n_edges": len(edges),
            }
        )
        msg = f"    lag_AUC={lag_auc:.4f}  lag_drop@25%={lag_d25:.4f}  K*={K_star}  edges={len(edges)}"
        print(msg)
    elif cfg.D > MAX_KARMA_D:
        r[f"{p}karma_auc"] = None
        r[f"{p}karma_lag_auc"] = None

    if not args.skip_ig:
        print("\n  [IG]")
        attr = run_ig(X_test_dt, model, device)
        _store("ig", attr, attr_to_time_scores(attr))

    if not args.skip_timeshap:
        print(f"\n  [TimeShap] (nsamples={args.ts_nsamples})")
        attr = run_timeshap(X_test_dt, model, device, nsamples=args.ts_nsamples)
        _store("timeshap", attr, attr_to_time_scores(attr))

    return r


def run_dataset(
    name: str,
    args,
    device: torch.device,
    ckpt_dir: Path,
    out_dir: Path,
    model_ckpt_dir: Path,
) -> dict:
    cfg = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  D={cfg.D}  T={cfg.T}  pred_horizon={cfg.pred_horizon}")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(cfg)
    print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    n_test = min(N_TEST_LARGE_D if cfg.D > 20 else N_TEST_MAX, len(X_test))
    X_test_s = X_test[:n_test]
    X_tr_dt = X_train.transpose(0, 2, 1)
    X_val_dt = X_val.transpose(0, 2, 1)
    X_test_dt = X_test_s.transpose(0, 2, 1).astype(np.float32)  # (B, D, T)
    x_train_mean = X_tr_dt.mean(axis=0)  # (D, T)

    # Fit a single VAR imputer for this dataset (shared across all models/methods)
    print("  Fitting VAR imputer for conditional timestep imputation …")
    var_K = select_var_order(X_tr_dt, max_K=8)
    var_imputer = fit_var_imputer(X_tr_dt, var_K)
    print(f"  VAR order K={var_K}")

    results = {
        "dataset": name,
        "D": cfg.D,
        "T": cfg.T,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": n_test,
        "var_K": var_K,
    }

    if not args.skip_lstm:
        print("\n── LSTM ──")
        try:
            lstm = load_pretrained_lstm(cfg, device, model_ckpt_dir)
            results.update(
                _run_methods_for_model(
                    "lstm",
                    lstm,
                    X_test_dt,
                    X_tr_dt,
                    X_val_dt,
                    X_train,
                    X_val,
                    cfg,
                    args,
                    ckpt_dir,
                    x_train_mean,
                    device,
                    var_imputer,
                    var_K,
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping LSTM: {e}")

    if not args.skip_tcn:
        print("\n── TCN ──")
        try:
            tcn = load_pretrained_tcn(cfg, device, model_ckpt_dir)
            results.update(
                _run_methods_for_model(
                    "tcn",
                    tcn,
                    X_test_dt,
                    X_tr_dt,
                    X_val_dt,
                    X_train,
                    X_val,
                    cfg,
                    args,
                    ckpt_dir,
                    x_train_mean,
                    device,
                    var_imputer,
                    var_K,
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping TCN: {e}")
    print("\n── GRU ──")
    gru = get_or_train_gru(cfg, X_train, y_train, X_val, y_val, device, ckpt_dir)
    results.update(
        _run_methods_for_model(
            "gru",
            gru,
            X_test_dt,
            X_tr_dt,
            X_val_dt,
            X_train,
            X_val,
            cfg,
            args,
            ckpt_dir,
            x_train_mean,
            device,
            var_imputer,
            var_K,
        )
    )

    # Save per-dataset JSON
    out_file = out_dir / f"{name}_results.json"
    with open(out_file, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n  → {out_file}")
    return results


def print_summary(all_results: list[dict]) -> None:
    archs = ["gru", "lstm", "tcn"]
    methods = ["fo", "dynamask", "winit", "ig", "timeshap", "karma"]
    col_w = 10
    ds_w = 18
    arch_w = 6
    width = ds_w + arch_w + col_w * len(methods)
    sep = "=" * width
    header = f"{'Dataset':<{ds_w}}{'Arch':<{arch_w}}" + "".join(
        f"{m.upper():>{col_w}}" for m in methods
    )

    metrics = [
        ("auc", "Cell AUC     — (feature,time) removal  (higher = better)"),
        ("drop25", "Cell drop@25%— (feature,time) removal  (higher = better)"),
        (
            "lag_auc",
            "Lag AUC      — whole-timestep removal  (higher = better, tests temporal structure)",
        ),
        ("lag_drop25", "Lag drop@25% — whole-timestep removal  (higher = better)"),
    ]
    for metric, label in metrics:
        print(f"\n{sep}")
        print(label)
        print(sep)
        print(header)
        print("-" * width)
        for r in all_results:
            for arch in archs:
                if not any(r.get(f"{arch}_{m}_{metric}") is not None for m in methods):
                    continue
                row = f"{r['dataset']:<{ds_w}}{arch:<{arch_w}}"
                for m in methods:
                    val = r.get(f"{arch}_{m}_{metric}")
                    row += f"{'—':>{col_w}}" if val is None else f"{val:>{col_w}.4f}"
                print(row)
        print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="AUC comparison on real forecasting datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ckpt_dir", default="outputs/checkpoints/realdata")
    parser.add_argument("--out_dir", default="results/realdata")
    parser.add_argument("--skip_karma", action="store_true")
    parser.add_argument("--skip_winit", action="store_true")
    parser.add_argument("--skip_fo", action="store_true")
    parser.add_argument("--skip_dynamask", action="store_true")
    parser.add_argument("--skip_ig", action="store_true")
    parser.add_argument("--skip_timeshap", action="store_true")
    parser.add_argument(
        "--ts_nsamples",
        type=int,
        default=200,
        help="KernelSHAP coalitions per window (TimeShap)",
    )
    parser.add_argument(
        "--lag_only",
        action="store_true",
        help="Compute only lag (whole-timestep) metrics; skip cell-level AUC",
    )
    parser.add_argument("--skip_lstm", action="store_true", help="Skip LSTM model")
    parser.add_argument("--skip_tcn", action="store_true", help="Skip TCN model")
    parser.add_argument(
        "--model_ckpt_dir",
        default="outputs/checkpoints",
        help="Root dir containing {prefix}_lstm/ and {prefix}_tcn/ checkpoint folders",
    )
    parser.add_argument(
        "--load_existing",
        action="store_true",
        help="Load pre-trained GRU and WinIT generators when available",
    )
    parser.add_argument("--dynamask_epochs", type=int, default=100)
    parser.add_argument("--winit_epochs", type=int, default=100)
    parser.add_argument("--quiet", action="store_true", help="Suppress KARMA verbosity")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    model_ckpt_dir = Path(args.model_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    all_results = []
    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset '{ds_name}'. Available: {list(DATASETS.keys())}")
            continue
        try:
            r = run_dataset(ds_name, args, device, ckpt_dir, out_dir, model_ckpt_dir)
            all_results.append(r)
        except Exception as e:
            print(f"\nERROR on dataset '{ds_name}': {e}")
            traceback.print_exc()

    if all_results:
        print_summary(all_results)
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as fh:
            json.dump(all_results, fh, indent=2, default=float)
        print(f"\nFull summary → {summary_path}")


if __name__ == "__main__":
    main()
