#!/usr/bin/env python
"""
experiments/comparison_var_multi.py
──────────────────────────────────────────────────────────────────────────────
Multi-dataset comparison: KARMA vs FO, DynaMask, WinIT on VAR processes of
increasing dimensionality and lag order.

Each dataset uses a randomly generated stationary VAR(K) process with a fixed
seed.  All methods explain the same analytical oracle (true DGP, no model
error).  Comparison metric: Kendall's τ vs G* (ground-truth coefficient
magnitudes) and pairwise between methods.

Configurations
--------------
  tiny    D=2  K=1  n_edges=2   T_train=2_000
  small   D=4  K=2  n_edges=5   T_train=5_000
  medium  D=4  K=3  n_edges=7   T_train=5_000   ← matches comparison_var.py
  large   D=6  K=3  n_edges=9   T_train=15_000
  xlarge  D=8  K=4  n_edges=14  T_train=30_000

Usage
-----
  python -m experiments.comparison_var_multi
  python -m experiments.comparison_var_multi --configs tiny small medium
  python -m experiments.comparison_var_multi --skip_dynamask --skip_winit
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from torch.utils.data import DataLoader, TensorDataset

from dataset.var import (
    build_coef_tensor,
    check_stationarity,
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
class VARConfig:
    name: str
    D: int
    K: int
    n_edges: int
    T_train: int
    T_test: int = 1000
    noise_std: float = 0.3
    coef: float = 0.25
    seed: int = 42
    # Derived KARMA params — set after init
    N: int = 3  # discretiser bins (2 for large D to limit state space)
    W: int = 0  # oracle window; 0 = auto (K*3 + 2, min 8)
    M: int = 200  # MC draws per history
    n_pool_min: int = 5
    K_max: int = 0  # 0 = auto (K + 2)

    def __post_init__(self):
        if self.W == 0:
            self.W = max(self.K * 3 + 2, 8)
        if self.K_max == 0:
            self.K_max = self.K + 2
        # Fewer bins for high-D to keep state space tractable
        if self.D > 4 and self.N == 3:
            self.N = 2
        if self.D >= 6:
            self.M = max(50, 200 // self.D * 2)


ALL_CONFIGS: dict[str, VARConfig] = {
    "tiny": VARConfig("tiny", D=2, K=1, n_edges=2, T_train=2_000, T_test=500),
    "small": VARConfig("small", D=4, K=2, n_edges=5, T_train=5_000, T_test=1000),
    "medium": VARConfig("medium", D=4, K=3, n_edges=7, T_train=5_000, T_test=1000),
    "large": VARConfig("large", D=6, K=3, n_edges=9, T_train=15_000, T_test=2000),
    "xlarge": VARConfig("xlarge", D=8, K=4, n_edges=14, T_train=30_000, T_test=3000),
}


def generate_stationary_var(
    cfg: VARConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Generate a random stationary VAR(K) with D variables and n_edges causal
    edges.  Scales down coefficient magnitude until spectral radius < 0.97.
    Returns (A (D,D,K), true_edges).
    """
    rng = np.random.default_rng(cfg.seed)
    candidates = [
        (s, t, k)
        for s in range(cfg.D)
        for t in range(cfg.D)
        for k in range(1, cfg.K + 1)
    ]

    for attempt in range(50):
        idx = rng.choice(len(candidates), size=cfg.n_edges, replace=False)
        edges = [candidates[i] for i in sorted(idx)]

        coef = cfg.coef
        for _ in range(30):
            A = build_coef_tensor(edges, cfg.D, cfg.K, coef)
            _, stationary = check_stationarity(A, cfg.D, cfg.K)
            if stationary:
                return A, edges
            coef *= 0.85

        rng = np.random.default_rng(cfg.seed + attempt + 1)

    raise RuntimeError(f"Could not generate stationary VAR for config '{cfg.name}'")


class VARTorchModel(TorchModel):
    """Differentiable VAR(K) oracle.  Input (B, D, T), output (B, D) or (B, D, T)."""

    def __init__(self, A: np.ndarray, device):
        D_, K_ = A.shape[0], A.shape[2]
        super().__init__(feature_size=D_, num_states=D_, hidden_size=0, device=device)
        self.register_buffer("A_coef", torch.tensor(A, dtype=torch.float32))
        self.K_true = K_
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        B, D_feat, T = x.shape
        K = min(self.K_true, T)
        if not return_all:
            pred = torch.zeros(B, D_feat, device=x.device, dtype=x.dtype)
            for k in range(K):
                pred = pred + x[:, :, T - 1 - k] @ self.A_coef[:, :, k]
            return pred
        else:
            out = torch.zeros(B, D_feat, T, device=x.device, dtype=x.dtype)
            for t in range(K, T):
                p = torch.zeros(B, D_feat, device=x.device, dtype=x.dtype)
                for k in range(K):
                    p = p + x[:, :, t - 1 - k] @ self.A_coef[:, :, k]
                out[:, :, t] = p
            return out


def make_numpy_oracle(A: np.ndarray):
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
    """MSE per sample: (N_area, B, T, D) → (B,)."""
    return ((Y_pred - Y_target) ** 2).mean(dim=[0, 2, 3])


class MSEDynaMaskExplainer(DynamaskExplainer):
    """DynaMask with MSE loss for multi-output regression, last-timestep only."""

    def __init__(self, device, **kwargs):
        kwargs.pop("loss", None)
        super().__init__(device=device, loss="ce", **kwargs)
        self.loss = _mse_loss_multiple
        self.loss_str = "mse"

    def _attribute_multiple(self, x: torch.Tensor) -> np.ndarray:
        orig_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        def f(x_in: torch.Tensor) -> torch.Tensor:
            x_perm = x_in.permute(0, 2, 1)
            out = self.base_model(x_perm, return_all=True).permute(0, 2, 1)
            if self.use_last_timestep_only:
                return out[:, -1:, :]
            return out

        x_td = x.permute(0, 2, 1)
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
        diff = (
            (y_orig[:, -1:, :] - y_zero[:, -1:, :])
            if self.use_last_timestep_only
            else (y_orig - y_zero)
        )
        thresh = (diff**2).mean(dim=[-1, -2]) * 0.5
        mask = mask_group.get_extremal_mask_multiple(thresholds=thresh)
        mask_saliency = mask.permute(0, 2, 1)
        torch.backends.cudnn.enabled = orig_cudnn
        return mask_saliency.detach().cpu().numpy()


def make_sliding_windows(X: np.ndarray, W: int):
    T, _D = X.shape
    xs = np.stack([X[t - W : t, :].T for t in range(W, T)]).astype(np.float32)
    ys = np.stack([X[t, :] for t in range(W, T)]).astype(np.float32)
    return xs, ys


def make_loaders(X: np.ndarray, W: int, batch_size: int = 64, val_frac: float = 0.1):
    x_all, y_all = make_sliding_windows(X, W)
    n_val = max(1, int(len(x_all) * val_frac))
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


def ground_truth_phi(A: np.ndarray, K_max: int) -> np.ndarray:
    D = A.shape[0]
    phi = np.zeros((D, K_max))
    phi[:, : A.shape[2]] = np.abs(A).sum(axis=1)
    return phi


def aggregate_to_phi(attr: np.ndarray, K_max: int, W: int) -> np.ndarray:
    D_src = attr.shape[1]
    phi = np.zeros((D_src, K_max))
    for k in range(1, K_max + 1):
        t_idx = W - k
        if 0 <= t_idx < W:
            phi[:, k - 1] = np.abs(attr[:, :, t_idx]).mean(axis=0)
    return phi


def aggregate_winit_to_phi(attr: np.ndarray, K_max: int) -> np.ndarray:
    D_src, window_size = attr.shape[1], attr.shape[3]
    phi = np.zeros((D_src, K_max))
    for k in range(1, K_max + 1):
        w_idx = window_size - k
        if 0 <= w_idx < window_size:
            phi[:, k - 1] = np.abs(attr[:, :, -1, w_idx]).mean(axis=0)
    return phi


def tau_vs_gt(phi: np.ndarray, phi_gt: np.ndarray) -> tuple[float, float]:
    r1 = phi.flatten()
    r2 = phi_gt.flatten()
    tau, p = kendalltau(r1, r2)
    return round(float(tau), 3), round(float(p), 4)


def run_karma(cfg, X_train, X_val, A, true_edges, lam, seed, verbose) -> dict:
    f_oracle = make_numpy_oracle(A)
    disc = Discretiser(N=cfg.N)
    disc.fit(X_train)

    T_val = len(X_val)
    X_val_w = np.stack([X_val[t : t + cfg.W] for t in range(T_val - cfg.W)])

    result = select_K_and_baseline(
        f=f_oracle,
        X_train=X_train,
        X_val=X_val_w,
        disc=disc,
        W=cfg.W,
        eps=0.00001,
        K_max=cfg.K_max,
        loss="regression",
        verbose=verbose,
    )
    K_star = result["K_star"]
    b_star, pi_star = result["b_star"], result["pi_star"]
    if verbose:
        print(f"    K* = {K_star},  Δ_pred = {result['delta_pred']:.4f}")

    check_design(
        disc,
        K=K_star,
        T_train=len(X_train),
        n_pool_min=cfg.n_pool_min,
        lam=lam,
        W=cfg.W,
    )

    pool = SuffixPool(disc=disc, K=K_star, W=cfg.W)
    pool.build(X_train)
    tree = TreeKernelEstimator(
        disc,
        K=K_star,
        W=cfg.W,
        M=cfg.M,
        n_pool=cfg.n_pool_min,
        pool=pool,
        b_star=b_star,
    )
    tree.fit(f=f_oracle, pi_star=pi_star, verbose=verbose)

    vi = compute_variable_importance(tree, pi_star, disc, lam=lam)
    phi = np.zeros((cfg.D, cfg.K_max))
    phi[:, :K_star] = vi["phi"]
    em = edge_metrics(vi["edges"], true_edges, cfg.D, K_star)
    return {"phi": phi, "K_star": K_star, "edge_metrics": em}


def run_fo(x_test, model, device) -> np.ndarray:
    fo = FOExplainer(device=device)
    fo.set_model(model)
    return fo.attribute(torch.from_numpy(x_test).to(device))


def run_dynamask(x_test, model, device, num_epoch=100) -> np.ndarray:
    dyn = MSEDynaMaskExplainer(
        device=device,
        area_list=np.array([0.25, 0.30, 0.35]),
        num_epoch=num_epoch,
        use_last_timestep_only=True,
    )
    dyn.set_model(model)
    return dyn.attribute(torch.from_numpy(x_test).to(device))


def run_ig(x_test: np.ndarray, model: VARTorchModel, device) -> np.ndarray:
    """Integrated Gradients (zero baseline, sum over D outputs) → (B, D, W)."""
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

    Explains each test window independently.  Cell SHAP values are reshaped
    (W, D) and transposed to (D, W) to match the (B, D, W) convention.
    """
    B, D_feat, W = x_test.shape
    x_td = x_test.transpose(0, 2, 1).astype(np.float32)  # (B, W, D)
    background = np.zeros((1, W, D_feat), dtype=np.float32)
    model.eval()

    def model_fn(x: np.ndarray) -> np.ndarray:
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
        # sv: (W*D,) ordered (t0_d0, t0_d1, …, tW_dD) → reshape (W, D) → (D, W)
        attr[b] = np.abs(sv.reshape(W, D_feat).T)
    return attr


def run_winit(
    x_test,
    train_loader,
    val_loader,
    model,
    cfg,
    ckpt_dir,
    device,
    num_samples=3,
    num_epochs=50,
    load_existing: bool = False,
) -> np.ndarray:
    winit = WinITExplainer(
        device=device,
        num_features=cfg.D,
        data_name=cfg.name,
        path=ckpt_dir,
        window_size=cfg.K,
        num_samples=num_samples,
        metric="pd",
        random_state=42,
    )
    winit.set_model(model)
    if load_existing:
        winit.load_generators()
    else:
        winit.train_generators(train_loader, val_loader, num_epochs=num_epochs)
    attr = winit.attribute(torch.from_numpy(x_test).to(device))
    return attr


def run_config(
    cfg: VARConfig,
    args,
    device: str,
    output_dir: Path,
) -> dict:
    print(f"\n{'═'*70}")
    print(
        f"  {cfg.name.upper()}   D={cfg.D}  K={cfg.K}  n_edges={cfg.n_edges}"
        f"  T_train={cfg.T_train:,}  N={cfg.N}  W={cfg.W}"
    )
    print(f"{'═'*70}")

    rng = np.random.default_rng(cfg.seed)
    A, true_edges = generate_stationary_var(cfg)

    X = simulate_var(A, cfg.D, cfg.K, cfg.T_train + cfg.T_test, cfg.noise_std, rng)
    X_train, X_test = X[: cfg.T_train], X[cfg.T_train :]

    phi_gt = ground_truth_phi(A, cfg.K_max)
    phis: dict[str, np.ndarray] = {"ground_truth": phi_gt}

    var_model = VARTorchModel(A, device=device).to(device)
    x_all, _ = make_sliding_windows(X_test, cfg.W)
    n_test = min(args.n_test, len(x_all))
    x_test_np = x_all[:n_test]

    train_loader, val_loader = make_loaders(X_train, cfg.W, batch_size=64)
    ckpt_dir = output_dir / cfg.name / "generators"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── KARMA ─────────────────────────────────────────────────────────────────
    if not args.skip_karma:
        print("\n  KARMA")
        karma_out = run_karma(
            cfg,
            X_train,
            X_test,
            A,
            true_edges,
            lam=args.lam,
            seed=args.seed,
            verbose=args.verbose,
        )
        phis["karma"] = karma_out["phi"]
        em = karma_out["edge_metrics"]
        print(f"    P={em['precision']:.2f}  R={em['recall']:.2f}  F1={em['f1']:.2f}")

    # ── FO ────────────────────────────────────────────────────────────────────
    if not args.skip_fo:
        print("\n  FO")
        attr_fo = run_fo(x_test_np, var_model, device)
        phis["fo"] = aggregate_to_phi(attr_fo, cfg.K_max, cfg.W)

    # ── DynaMask ──────────────────────────────────────────────────────────────
    if not args.skip_dynamask:
        print("\n  DynaMask")
        attr_dyn = run_dynamask(x_test_np, var_model, device, num_epoch=args.dyn_epochs)
        phis["dynamask"] = aggregate_to_phi(attr_dyn, cfg.K_max, cfg.W)

    # ── WinIT ─────────────────────────────────────────────────────────────────
    if not args.skip_winit:
        print("\n  WinIT")
        attr_wi = run_winit(
            x_test_np,
            train_loader,
            val_loader,
            var_model,
            cfg,
            ckpt_dir,
            device,
            num_samples=args.winit_samples,
            num_epochs=args.gen_epochs,
            load_existing=args.load_generators,
        )
        phi_wi = (
            aggregate_winit_to_phi(attr_wi, cfg.K_max)
            if attr_wi.ndim == 4
            else aggregate_to_phi(attr_wi, cfg.K_max, cfg.W)
        )
        phis["winit"] = phi_wi

    # ── Integrated Gradients ──────────────────────────────────────────────────
    if not args.skip_ig:
        print("\n  IG")
        attr_ig = run_ig(x_test_np, var_model, device)
        phis["ig"] = aggregate_to_phi(attr_ig, cfg.K_max, cfg.W)

    # ── TimeShap ──────────────────────────────────────────────────────────────
    if not args.skip_timeshap:
        print("\n  TimeShap")
        attr_ts = run_timeshap(x_test_np, var_model, device, nsamples=args.ts_nsamples)
        phis["timeshap"] = aggregate_to_phi(attr_ts, cfg.K_max, cfg.W)

    # ── τ vs G* ───────────────────────────────────────────────────────────────
    taus = {}
    for method, phi in phis.items():
        if method == "ground_truth":
            continue
        tau, p = tau_vs_gt(phi, phi_gt)
        taus[method] = (tau, p)

    return {
        "config": {
            "name": cfg.name,
            "D": cfg.D,
            "K": cfg.K,
            "n_edges": cfg.n_edges,
            "T_train": cfg.T_train,
        },
        "true_edges": [list(e) for e in true_edges],
        "karma_edge_metrics": (
            karma_out["edge_metrics"] if not args.skip_karma else None
        ),
        "karma_K_star": karma_out["K_star"] if not args.skip_karma else None,
        "phis": {m: p.tolist() for m, p in phis.items()},
        "tau_vs_gt": {m: {"tau": t, "p": pv} for m, (t, pv) in taus.items()},
    }


def print_summary(all_results: list[dict]) -> None:
    methods = ["karma", "fo", "dynamask", "winit", "ig", "timeshap"]
    col_w = 14

    print(f"\n{'═'*70}")
    print("  Kendall's τ vs G*   (* p < 0.05)")
    print(f"{'═'*70}")

    hdr = f"  {'config':>8s}  {'D':>3s}  {'K':>3s}  {'T':>7s}"
    for m in methods:
        hdr += f"  {m:>{col_w}s}"
    print(hdr)
    print(f"  {'-'*66}")

    for res in all_results:
        cfg_info = res["config"]
        row = (
            f"  {cfg_info['name']:>8s}  {cfg_info['D']:>3d}  {cfg_info['K']:>3d}"
            f"  {cfg_info['T_train']:>7,d}"
        )
        for m in methods:
            entry = res["tau_vs_gt"].get(m)
            if entry is None:
                row += f"  {'—':>{col_w}s}"
            else:
                tau, p = entry["tau"], entry["p"]
                mark = "*" if p < 0.05 else " "
                row += f"  {tau:>+.3f}{mark}{'':>{col_w-7}s}"
        print(row)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(ALL_CONFIGS.keys()),
        choices=list(ALL_CONFIGS.keys()),
        help="Which configs to run (default: all)",
    )
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--gen_epochs", type=int, default=50)
    p.add_argument("--dyn_epochs", type=int, default=100)
    p.add_argument("--winit_samples", type=int, default=3)
    p.add_argument("--skip_karma", action="store_true")
    p.add_argument("--skip_fo", action="store_true")
    p.add_argument("--skip_dynamask", action="store_true")
    p.add_argument("--skip_winit", action="store_true")
    p.add_argument(
        "--load_generators",
        action="store_true",
        help="Load saved WinIT generators instead of retraining",
    )
    p.add_argument("--skip_ig", action="store_true")
    p.add_argument("--skip_timeshap", action="store_true")
    p.add_argument(
        "--ts_nsamples",
        type=int,
        default=200,
        help="KernelSHAP coalitions per test window (TimeShap)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/comparison_var_multi")
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = []
    for name in args.configs:
        cfg = ALL_CONFIGS[name]
        result = run_config(cfg, args, device, output_dir)
        all_results.append(result)

        # Save per-config JSON immediately
        with open(output_dir / f"{name}.json", "w") as fh:
            json.dump(result, fh, indent=2)

    print_summary(all_results)

    with open(output_dir / "summary.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults → {output_dir}/")


if __name__ == "__main__":
    main()
