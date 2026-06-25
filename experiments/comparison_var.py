#!/usr/bin/env python
"""
experiments/comparison_var.py
──────────────────────────────────────────────────────────────────────────────
Comparison experiment on the synthetic VAR(3) dataset (D=4 variables).

All methods explain the same analytical VAR oracle (perfect knowledge of
coefficients).  Ground truth importance G* is derived from the coefficient
magnitudes, so Kendall's τ vs G* is a measure of recovery accuracy.

Methods
-------
  KARMA      – transition-kernel TV attributions (Pillar 3)
  FO         – feature occlusion (WinIT's FOExplainer)
  DynaMask   – differentiable mask, MSE loss (regression variant)
  WinIT      – windowed counterfactual attribution, pd metric

Note: FIT is omitted — it uses KL divergence which requires probability
      distributions.  The analytical VAR oracle outputs raw regression
      values, making FIT's loss undefined.

Usage
-----
  python -m experiments.comparison_var
  python -m experiments.comparison_var --T_train 5000 --seed 42
  python -m experiments.comparison_var --skip_dynamask --skip_winit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from torch.utils.data import DataLoader, TensorDataset

from dataset.var import (
    TRUE_EDGES,
    VAR_NAMES,
    D,
    K_TRUE,
    NOISE_STD,
    COEF_BASE,
    build_coef_tensor,
    simulate_var,
    edge_metrics,
)
from karma.causal_recovery.edge_contribution import compute_variable_importance
from karma.markov_approximation.markov_surrogacy import select_K_and_baseline
from karma.utils import check_design
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import TreeKernelEstimator
from karma.utils.sampling import SuffixPool
from captum.attr import IntegratedGradients
from timeshap.explainer.kernel import TimeShapKernel

# Path setup must precede any local-package imports
_ROOT = Path(__file__).parent.parent
_WINIT = _ROOT / "WinIT"
sys.path.insert(0, str(_WINIT))
sys.path.insert(0, str(_ROOT))

from winit.explainer.attribution.mask_group import MaskGroup
from winit.explainer.dynamaskexplainer import DynamaskExplainer
from winit.explainer.explainers import FOExplainer
from winit.explainer.winitexplainers import WinITExplainer
from winit.models import TorchModel

A_TRUE = build_coef_tensor(TRUE_EDGES, D, K_TRUE, COEF_BASE)


def ground_truth_phi(K_max: int) -> np.ndarray:
    """(D, K_max) coefficient-magnitude matrix — the known true importance."""
    phi = np.zeros((D, K_max))
    phi[:, :K_TRUE] = np.abs(A_TRUE).sum(axis=1)  # sum over tgt dimension
    return phi


class VARTorchModel(TorchModel):
    """
    Differentiable implementation of the VAR(K) process.

    Convention follows WinIT's TorchModel:
        forward(x, return_all=True)  → (B, D, T)
        forward(x, return_all=False) → (B, D)
    where x is (B, D, T).

    Activation is set to nn.Identity() (regression, not classification).
    """

    def __init__(self, A: np.ndarray, device):
        D_, K_ = A.shape[0], A.shape[2]
        super().__init__(feature_size=D_, num_states=D_, hidden_size=0, device=device)
        self.register_buffer("A_coef", torch.tensor(A, dtype=torch.float32))
        self.K_true = K_
        # Override TorchModel's softmax activation with identity for regression
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        # x: (B, D, T)
        B, D_feat, T = x.shape
        K = min(self.K_true, T)

        if not return_all:
            pred = torch.zeros(B, D_feat, device=x.device, dtype=x.dtype)
            for k in range(K):
                # x[:, :, T-1-k] is observations at lag k+1, shape (B, D)
                # A_coef[:, :, k] is (D_src, D_tgt), so (B,D) @ (D,D) = (B,D)
                pred = pred + x[:, :, T - 1 - k] @ self.A_coef[:, :, k]
            return pred  # (B, D)
        else:
            out = torch.zeros(B, D_feat, T, device=x.device, dtype=x.dtype)
            for t in range(K, T):
                p = torch.zeros(B, D_feat, device=x.device, dtype=x.dtype)
                for k in range(K):
                    p = p + x[:, :, t - 1 - k] @ self.A_coef[:, :, k]
                out[:, :, t] = p
            return out  # (B, D, T)


def make_numpy_oracle(A: np.ndarray):
    """
    Numpy oracle for KARMA — equivalent to VARTorchModel but pure numpy.
    Accepts (W, D) or (B, W, D) windows.
    """
    K = A.shape[2]

    def f_oracle(window: np.ndarray) -> np.ndarray:
        single = window.ndim == 2
        w = window[np.newaxis] if single else window
        B, W, _D = w.shape
        pred = np.zeros((B, _D))
        for k in range(min(K, W)):
            pred += w[:, W - 1 - k, :] @ A[:, :, k]
        return pred[0] if single else pred

    return f_oracle


def _mse_loss_multiple(Y_pred: torch.Tensor, Y_target: torch.Tensor) -> torch.Tensor:
    """MSE per sample.  Y_pred/Y_target: (N_area, B, T, D) → (B,).
    Mirrors cross_entropy_multiple which also averages over dim=[0, 2, 3]."""
    return ((Y_pred - Y_target) ** 2).mean(dim=[0, 2, 3])


class MSEDynaMaskExplainer(DynamaskExplainer):
    """
    DynaMask adapted for multi-output regression with MSE loss.

    The original DynamaskExplainer only supports "logloss" and "ce" loss strings
    and formats f(x) for single-class or multi-class *classification*.
    This subclass overrides _attribute_multiple so that:
      - f(x_in) returns (B, T, D) raw regression outputs
      - loss is MSE over all (B, T, D) dimensions per mask area
    """

    def __init__(self, device, **kwargs):
        # Pass a valid loss string to bypass the RuntimeError, then replace
        kwargs.pop("loss", None)
        super().__init__(device=device, loss="ce", **kwargs)
        self.loss = _mse_loss_multiple
        self.loss_str = "mse"

    def _attribute_multiple(self, x: torch.Tensor) -> np.ndarray:
        # x: (B, D, T)
        orig_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        def f(x_in: torch.Tensor) -> torch.Tensor:
            # x_in: (B, T, D)  ← DynaMask internal convention
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

        # Threshold: 50% of signal power at the timestep(s) used in the loss
        y_orig = f(x_td)  # (B, T, D)
        y_zero = f(torch.zeros_like(x_td))
        if self.use_last_timestep_only:
            diff = y_orig[:, -1:, :] - y_zero[:, -1:, :]  # (B, 1, D)
        else:
            diff = y_orig - y_zero  # (B, T, D)
        thresh = (diff**2).mean(dim=[-1, -2]) * 0.5  # (B,)

        mask = mask_group.get_extremal_mask_multiple(thresholds=thresh)
        # mask: (B, T, D)
        mask_saliency = mask.permute(0, 2, 1)  # (B, D, T)

        torch.backends.cudnn.enabled = orig_cudnn
        return mask_saliency.detach().cpu().numpy()


def make_sliding_windows(X: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, D) time series → (N, D, W) windows and (N, D) targets."""
    T, _D = X.shape
    xs, ys = [], []
    for t in range(W, T):
        xs.append(X[t - W : t, :].T)  # (D, W)
        ys.append(X[t, :])  # (D,)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def make_loaders(
    X: np.ndarray, W: int, val_frac: float = 0.1, batch_size: int = 64
) -> tuple[DataLoader, DataLoader]:
    x_all, y_all = make_sliding_windows(X, W)
    N = len(x_all)
    n_val = max(1, int(N * val_frac))
    x_tr, y_tr = x_all[:-n_val], y_all[:-n_val]
    x_va, y_va = x_all[-n_val:], y_all[-n_val:]
    tr = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    va = DataLoader(
        TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
        batch_size=batch_size,
        shuffle=False,
    )
    return tr, va


def aggregate_to_phi(attr: np.ndarray, K_max: int, W: int) -> np.ndarray:
    """
    Collapse (B, D_src, W) attribution → (D_src, K_max) importance.

    attr[:, d, W-k] = importance of feature d at lag k (1-indexed).
    Average absolute values over the batch dimension.
    """
    D_src = attr.shape[1]
    phi = np.zeros((D_src, K_max))
    for k in range(1, K_max + 1):
        t_idx = W - k
        if 0 <= t_idx < W:
            phi[:, k - 1] = np.abs(attr[:, :, t_idx]).mean(axis=0)
    return phi


def aggregate_winit_to_phi(attr: np.ndarray, K_max: int) -> np.ndarray:
    """
    Collapse WinIT's (B, D_src, T, window_size) → (D_src, K_max).

    attr[b, d, T-1, window_size-k] = importance of d at lag k for final step.
    """
    D_src = attr.shape[1]
    window_size = attr.shape[3]
    phi = np.zeros((D_src, K_max))
    for k in range(1, K_max + 1):
        w_idx = window_size - k
        if 0 <= w_idx < window_size:
            phi[:, k - 1] = np.abs(attr[:, :, -1, w_idx]).mean(axis=0)
    return phi


def run_karma(
    X_train, X_val, A, W, N, eps, lam, M, n_pool_min, K_max, seed, verbose
) -> tuple[np.ndarray, list, int]:
    """Run KARMA pipeline.  Returns (phi (D, K_max), retained_edges, K_star)."""
    f_oracle = make_numpy_oracle(A)

    disc = Discretiser(N=N)
    disc.fit(X_train)

    # select_K_and_baseline expects X_val as (N_windows, W, D) — pre-create windows
    T_val = len(X_val)
    X_val_windows = np.stack(
        [X_val[t : t + W] for t in range(T_val - W)]
    )  # (T_val-W, W, D)

    result = select_K_and_baseline(
        f=f_oracle,
        X_train=X_train,
        X_val=X_val_windows,
        disc=disc,
        W=W,
        eps=eps,
        K_max=K_max,
        loss="regression",
        verbose=verbose,
    )
    K_star = result["K_star"]
    b_star = result["b_star"]
    pi_star = result["pi_star"]
    delta_pred = result["delta_pred"]

    if verbose:
        print(f"  K* = {K_star},  Δ_pred = {delta_pred:.4f}")

    check_design(
        disc, K=K_star, T_train=len(X_train), n_pool_min=n_pool_min, lam=lam, W=W
    )

    pool = SuffixPool(disc=disc, K=K_star, W=W)
    pool.build(X_train)

    tree = TreeKernelEstimator(
        disc, K=K_star, W=W, M=M, n_pool=n_pool_min, pool=pool, b_star=b_star
    )
    tree.fit(f=f_oracle, pi_star=pi_star, verbose=verbose)

    vi = compute_variable_importance(tree, pi_star, disc, lam=lam)

    phi = np.zeros((D, K_max))
    phi[:, :K_star] = vi["phi"]
    return phi, vi["edges"], K_star


def run_fo(x_test: np.ndarray, model: VARTorchModel, device) -> np.ndarray:
    """FOExplainer attribution → (B, D, W)."""
    fo = FOExplainer(device=device)
    fo.set_model(model)
    x_t = torch.from_numpy(x_test).to(device)
    return fo.attribute(x_t)  # (B, D, W)


def run_dynamask(
    x_test: np.ndarray,
    model: VARTorchModel,
    device,
    area_list=None,
    num_epoch: int = 100,
) -> np.ndarray:
    """MSEDynaMaskExplainer attribution → (B, D, W)."""
    areas = area_list if area_list is not None else np.arange(0.25, 0.36, 0.05)
    dyn = MSEDynaMaskExplainer(
        device=device,
        area_list=areas,
        num_epoch=num_epoch,
        use_last_timestep_only=True,
    )
    dyn.set_model(model)
    x_t = torch.from_numpy(x_test).to(device)
    return dyn.attribute(x_t)  # (B, D, W)


def run_ig(x_test: np.ndarray, model: VARTorchModel, device) -> np.ndarray:
    """Integrated Gradients attribution → (B, D, W).

    Uses zero baseline.  Sums gradients across all D output dimensions so a
    single (B, D, W) saliency map covers all target variables jointly.
    """
    model.eval()

    def _scalar_forward(x: torch.Tensor) -> torch.Tensor:
        return model(x, return_all=False).sum(dim=-1)  # (B,)

    ig = IntegratedGradients(_scalar_forward)
    x_t = torch.from_numpy(x_test.astype(np.float32)).to(device)

    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    attr = ig.attribute(x_t, baselines=torch.zeros_like(x_t))  # (B, D, W)
    torch.backends.cudnn.enabled = orig_cudnn

    return np.abs(attr.detach().cpu().numpy())


def run_timeshap(
    x_test: np.ndarray,
    model: VARTorchModel,
    device,
    nsamples: int = 200,
) -> np.ndarray:
    """TimeShap cell-level attribution → (B, D, W).

    Explains each test window independently.  The model is wrapped to return a
    scalar (mean over D outputs).  Cell SHAP values are reshaped (T, D) →
    transposed to (D, T) so the result matches the (B, D, W) convention used
    by the other methods.
    """
    B, D_feat, W = x_test.shape

    # TimeShap convention: (B, T, D)
    x_td = x_test.transpose(0, 2, 1).astype(np.float32)  # (B, W, D)
    background = np.zeros((1, W, D_feat), dtype=np.float32)

    model.eval()

    def model_fn(x: np.ndarray) -> np.ndarray:
        # x: (N, W, D) numpy → (N, D, W) torch → (N,) scalar
        x_dt = torch.from_numpy(x.transpose(0, 2, 1)).to(device)
        with torch.no_grad():
            return model(x_dt, return_all=False).mean(dim=-1).cpu().numpy()

    varying = (list(range(W)), list(range(D_feat)))
    attr = np.zeros((B, D_feat, W), dtype=np.float32)
    for b in range(B):
        kernel = TimeShapKernel(
            model_fn, background, rs=42, mode="cell", varying=varying
        )
        sv = kernel.shap_values(x_td[b : b + 1], pruning_idx=0, nsamples=nsamples)
        # sv: (W*D,) ordered as (t0_d0, t0_d1, …, t0_dD, t1_d0, …)
        # reshape (W, D) then transpose to (D, W)
        attr[b] = np.abs(sv.reshape(W, D_feat).T)
    return attr  # (B, D, W)


def run_winit(
    x_test: np.ndarray,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: VARTorchModel,
    ckpt_dir: Path,
    device,
    window_size: int = K_TRUE,
    num_samples: int = 3,
    num_epochs: int = 100,
    load_existing: bool = False,
) -> np.ndarray:
    """WinITExplainer attribution → (B, D, T, window_size)."""
    winit = WinITExplainer(
        device=device,
        num_features=D,
        data_name="var",
        path=ckpt_dir,
        window_size=window_size,
        num_samples=num_samples,
        metric="pd",
        random_state=42,
    )
    winit.set_model(model)
    if load_existing:
        winit.load_generators()
    else:
        winit.train_generators(train_loader, val_loader, num_epochs=num_epochs)
    x_t = torch.from_numpy(x_test).to(device)
    return winit.attribute(x_t)  # (B, D, T, window_size)


def phi_to_rank(phi: np.ndarray) -> np.ndarray:
    """Flatten (D, K) → (D*K,) ordinal ranks (0 = most important)."""
    flat = phi.flatten()
    return flat.argsort()[::-1].argsort().astype(int)


def kendall_tau_table(
    phis: dict[str, np.ndarray],
) -> dict[tuple[str, str], tuple[float, float]]:
    """Compute all pairwise Kendall's τ and p-values."""
    methods = list(phis.keys())
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i:]:
            r1 = phi_to_rank(phis[m1])
            r2 = phi_to_rank(phis[m2])
            tau, p = kendalltau(r1, r2)
            out[(m1, m2)] = (round(float(tau), 4), round(float(p), 4))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="KARMA vs baselines on VAR(3) dataset")
    p.add_argument("--T_train", type=int, default=5000)
    p.add_argument("--T_test", type=int, default=1000)
    p.add_argument("--N", type=int, default=3, help="Discretiser bins (KARMA)")
    p.add_argument("--W", type=int, default=10, help="Oracle window size")
    p.add_argument("--eps", type=float, default=0.05, help="Δ_pred stopping tolerance")
    p.add_argument("--lam", type=float, default=0.025, help="Edge trimming threshold")
    p.add_argument("--M", type=int, default=200, help="MC draws per history (KARMA)")
    p.add_argument("--n_pool_min", type=int, default=5, help="Min pool size (KARMA)")
    p.add_argument("--K_max", type=int, default=5, help="Max K to search (KARMA)")
    p.add_argument(
        "--n_test", type=int, default=100, help="Test windows (WinIT methods)"
    )
    p.add_argument("--gen_epochs", type=int, default=50, help="WinIT generator epochs")
    p.add_argument(
        "--dyn_epochs", type=int, default=100, help="DynaMask optimisation epochs"
    )
    p.add_argument("--skip_karma", action="store_true")
    p.add_argument("--skip_fo", action="store_true")
    p.add_argument("--skip_dynamask", action="store_true")
    p.add_argument("--skip_winit", action="store_true")
    p.add_argument("--skip_ig", action="store_true")
    p.add_argument("--skip_timeshap", action="store_true")
    p.add_argument(
        "--ts_nsamples",
        type=int,
        default=200,
        help="KernelSHAP coalitions per test window (TimeShap)",
    )
    p.add_argument(
        "--load_generators", action="store_true", help="Skip WinIT training, load saved"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="results/comparison_var")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = Path(args.output_dir) / "generators"
    ckpt_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"VAR(D={D}, K={K_TRUE})  True edges: {len(TRUE_EDGES)}")
    print(f"Variables: {VAR_NAMES}")

    # ── Generate data ─────────────────────────────────────────────────────────
    _rng = np.random.default_rng(args.seed)
    T_total = args.T_train + args.T_test
    X = simulate_var(A_TRUE, D, K_TRUE, T_total, NOISE_STD, _rng)
    X_train, X_test = X[: args.T_train], X[args.T_train :]

    # ── Ground truth ──────────────────────────────────────────────────────────
    phi_gt = ground_truth_phi(args.K_max)
    phis: dict[str, np.ndarray] = {"ground_truth": phi_gt}

    # ── Shared model and test windows ─────────────────────────────────────────
    var_model = VARTorchModel(A_TRUE, device=device).to(device)

    x_all, _ = make_sliding_windows(X_test, args.W)
    n_test = min(args.n_test, len(x_all))
    x_test_np = x_all[:n_test]  # (n_test, D, W)

    train_loader, val_loader = make_loaders(X_train, args.W, batch_size=64)

    print(f"Test windows: {n_test} × (D={D}, W={args.W})")

    # ── KARMA ─────────────────────────────────────────────────────────────────
    if not args.skip_karma:
        print("\n" + "=" * 60)
        print("KARMA")
        print("=" * 60)
        phi_karma, karma_edges, K_star = run_karma(
            X_train,
            X_test,
            A_TRUE,
            W=args.W,
            N=args.N,
            eps=args.eps,
            lam=args.lam,
            M=args.M,
            n_pool_min=args.n_pool_min,
            K_max=args.K_max,
            seed=args.seed,
            verbose=args.verbose,
        )
        phis["karma"] = phi_karma
        em = edge_metrics(karma_edges, TRUE_EDGES, D, K_star)
        print(f"  K* = {K_star},  edges retained = {len(karma_edges)}")
        print(
            f"  Precision = {em['precision']:.3f},  Recall = {em['recall']:.3f},  F1 = {em['f1']:.3f}"
        )

    # ── FO ────────────────────────────────────────────────────────────────────
    if not args.skip_fo:
        print("\n" + "=" * 60)
        print("FO (Feature Occlusion)")
        print("=" * 60)
        attr_fo = run_fo(x_test_np, var_model, device)
        phi_fo = aggregate_to_phi(attr_fo, args.K_max, args.W)
        phis["fo"] = phi_fo
        print("  Done.")

    # ── DynaMask ──────────────────────────────────────────────────────────────
    if not args.skip_dynamask:
        print("\n" + "=" * 60)
        print("DynaMask (MSE loss, multi-output)")
        print("=" * 60)
        attr_dyn = run_dynamask(
            x_test_np,
            var_model,
            device,
            area_list=np.array([0.25, 0.30, 0.35]),
            num_epoch=args.dyn_epochs,
        )
        phi_dyn = aggregate_to_phi(attr_dyn, args.K_max, args.W)
        phis["dynamask"] = phi_dyn
        print("  Done.")

    # ── WinIT ─────────────────────────────────────────────────────────────────
    if not args.skip_winit:
        print("\n" + "=" * 60)
        print("WinIT (pd metric, generator trained on VAR data)")
        print("=" * 60)
        attr_winit = run_winit(
            x_test_np,
            train_loader,
            val_loader,
            var_model,
            ckpt_dir=ckpt_dir,
            device=device,
            window_size=K_TRUE,
            num_samples=3,
            num_epochs=args.gen_epochs,
            load_existing=args.load_generators,
        )
        if attr_winit.ndim == 4:
            phi_winit = aggregate_winit_to_phi(attr_winit, args.K_max)
        else:
            phi_winit = aggregate_to_phi(attr_winit, args.K_max, args.W)
        phis["winit"] = phi_winit
        print("  Done.")

    # ── Integrated Gradients ──────────────────────────────────────────────────
    if not args.skip_ig:
        print("\n" + "=" * 60)
        print("Integrated Gradients (zero baseline, sum over D outputs)")
        print("=" * 60)
        attr_ig = run_ig(x_test_np, var_model, device)
        phi_ig = aggregate_to_phi(attr_ig, args.K_max, args.W)
        phis["ig"] = phi_ig
        print("  Done.")

    # ── TimeShap ──────────────────────────────────────────────────────────────
    if not args.skip_timeshap:
        print("\n" + "=" * 60)
        print(f"TimeShap (cell-level, nsamples={args.ts_nsamples})")
        print("=" * 60)
        attr_ts = run_timeshap(x_test_np, var_model, device, nsamples=args.ts_nsamples)
        phi_ts = aggregate_to_phi(attr_ts, args.K_max, args.W)
        phis["timeshap"] = phi_ts
        print("  Done.")

    # ── Importance tables ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Importance  phi[D_src, lag]  (K_max={args.K_max})")
    print("=" * 60)
    lag_hdr = "  ".join(f"lag{k+1}" for k in range(args.K_max))
    for method, phi in phis.items():
        print(f"\n  {method}:")
        print(f"    {'var':>5s}  {lag_hdr}")
        for d, vname in enumerate(VAR_NAMES):
            vals = "  ".join(f"{phi[d, k]:.4f}" for k in range(args.K_max))
            print(f"    {vname:>5s}  {vals}")

    # ── Kendall's Tau ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Kendall's τ  (* = p < 0.05)")
    print("=" * 60)
    tau_table = kendall_tau_table(phis)
    methods = list(phis.keys())

    # Header
    print(f"\n{'':>14s}", end="")
    for m in methods:
        print(f"{m:>14s}", end="")
    print()
    # Rows
    for m1 in methods:
        print(f"{m1:>14s}", end="")
        for m2 in methods:
            key = (m1, m2) if (m1, m2) in tau_table else (m2, m1)
            tau, pv = tau_table.get(key, (float("nan"), float("nan")))
            mark = "*" if pv < 0.05 else " "
            print(f"    {tau:>+6.3f}{mark}     ", end="")
        print()

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "config": vars(args),
        "true_edges": [list(e) for e in TRUE_EDGES],
        "karma": {
            "K_star": K_star,
            "edges": [list(e) for e in karma_edges],
            "edge_metrics": em,
        },
        "phis": {m: phi.tolist() for m, phi in phis.items()},
        "kendall_tau": {
            f"{m1}_vs_{m2}": {"tau": t, "p": pv}
            for (m1, m2), (t, pv) in tau_table.items()
        },
    }
    out_path = Path(args.output_dir) / "comparison_var.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
