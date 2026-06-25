"""
Level 5 — Uncertainty-aware explanation for KARMA.

Computes and visualises two distinct uncertainty components per history h:

    Aleatoric uncertainty  H^d_h  = -sum_s T^hat(s|h) log T^hat(s|h)
        Irreducible uncertainty in the model's predicted distribution.
        High entropy = the model is genuinely uncertain about the next
        state given history h. Cannot be reduced by collecting more data.

    Epistemic uncertainty  Var^d(h) = T^hat(1 - T^hat) / n^f(h)
        Uncertainty in the kernel estimate itself due to finite queries.
        High variance = the kernel estimate is unreliable. Reduced by
        collecting more oracle queries for history h.

    Per-history coverage bar:
        bar(h) = max(0, 1 - rho_floor(h) / (lambda/2))
        Colour thresholds: green < lambda/4 < amber < lambda/2 < red.

Inputs
------
kernel_estimator : FactoredKernelEstimator or BStarKernelEstimator
    Must have predict_marginal(d, h_idx) -> (N,) and n_f_h(h_idx) -> int.
pi_star : dict[int, float]
    Stationary weights hat_pi*(h), keyed by h_idx.
lam : float
    Trimming threshold lambda. Coverage threshold is lambda/2.
delta_pred : float
    Certified prefix sensitivity from Pillar 2 (hat_Delta_pred < eps).
M : int
    Monte Carlo draws used in BStarKernelEstimator (for MC noise floor term).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass
from typing import Protocol


class KernelEstimatorLike(Protocol):
    def predict_marginal(self, d: int, h_idx: int) -> np.ndarray: ...


@dataclass
class HistoryUncertainty:
    """
    All uncertainty statistics for a single history h and target variable d.

    Attributes
    ----------
    h_idx       : int    flat history index
    d           : int    target variable index
    kernel      : (N,)   estimated transition distribution T^hat(.|h)
    aleatoric   : float  entropy H^d_h
    epistemic   : (N,)   per-bin variance Var(T^hat(s|h)) for each s
    """

    h_idx: int
    d: int
    kernel: np.ndarray
    aleatoric: float
    epistemic: np.ndarray


@dataclass
class UncertaintyMatrices:
    """
    Top-k histories ranked by mean aleatoric uncertainty across all D dimensions.

    Attributes
    ----------
    h_idxs          : (top_k,)     history indices, sorted by mean aleatoric desc
    aleatoric       : (top_k, D)   aleatoric entropy H^d_h per (h, d)
    epistemic       : (top_k, D)   mean epistemic variance per (h, d)
    """

    h_idxs: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray


def aleatoric_entropy(kernel: np.ndarray) -> float:
    """
    H^d_h = -sum_{s'} T^hat(s'|h) log T^hat(s'|h)

    Uses natural log. Zero terms (T=0) contribute zero by convention.
    Maximum entropy for N bins = log(N).

    Parameters
    ----------
    kernel : (N,) float   probability simplex

    Returns
    -------
    float  entropy in nats
    """
    k = np.clip(kernel, 1e-12, 1.0)
    return float(-np.sum(k * np.log(k)))


def epistemic_variance(kernel: np.ndarray, n_queries: int) -> np.ndarray:
    """
    Var(T^hat(s|h)) ~= T^hat(s|h) * (1 - T^hat(s|h)) / n^f(h)

    Dirichlet posterior variance for a single marginal probability.
    Derived from the Dirichlet posterior: if alpha_s = n_s + 0.5 and
    alpha_0 = sum alpha_s, then Var(p_s) = alpha_s(alpha_0 - alpha_s)
    / (alpha_0^2 (alpha_0 + 1)), which simplifies to T(1-T)/n for large n.

    Parameters
    ----------
    kernel    : (N,) float   posterior mean kernel
    n_queries : int          n^f(h)

    Returns
    -------
    (N,) float  per-bin variance. Returns inf array if n_queries == 0.
    """
    if n_queries == 0:
        return np.full_like(kernel, np.inf)
    return kernel * (1.0 - kernel) / n_queries


def compute_uncertainty_attributes(
    estimator: KernelEstimatorLike,
    sample_kernels: list[dict],
    D: int,
    top_k: int = 10,
) -> UncertaintyMatrices:
    """
    Compute Level 5 uncertainty statistics and return top-k h_idxs vs D matrices.

    Parameters
    ----------
    estimator    : KernelEstimatorLike
    sample_kernels : list of dicts with keys 'h_idx' and 'pool_size'
    D            : int   number of target variables
    top_k        : int   number of most uncertain histories to return

    Returns
    -------
    UncertaintyMatrices
        h_idxs    : (top_k,)   history indices ranked by mean aleatoric desc
        aleatoric : (top_k, D) aleatoric entropy H^d_h per (h, d)
        epistemic : (top_k, D) mean epistemic variance per (h, d)
    """
    h_idxs = [sk["h_idx"] for sk in sample_kernels]
    H_mat = np.zeros((len(h_idxs), D))  # aleatoric
    V_mat = np.zeros((len(h_idxs), D))  # epistemic (mean per-bin variance)

    for i, sample_kernel in enumerate(sample_kernels):
        h_idx = sample_kernel["h_idx"]
        for d in range(D):
            kernel = estimator.predict_marginal(d, h_idx)
            H_mat[i, d] = aleatoric_entropy(kernel)
            V_mat[i, d] = float(
                np.mean(epistemic_variance(kernel, sample_kernel["pool_size"]))
            )

    # Rank by mean aleatoric across D, descending
    mean_aleatoric = H_mat.mean(axis=1)
    k = min(top_k, len(h_idxs))
    top_indices = np.argsort(mean_aleatoric)[::-1][:k]

    return UncertaintyMatrices(
        h_idxs=np.array(h_idxs)[top_indices],
        aleatoric=H_mat[top_indices],
        epistemic=V_mat[top_indices],
    )


def plot_uncertainty_heatmaps(
    matrices: UncertaintyMatrices,
    d_labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    aleatoric_cmap: str = "YlOrRd",
    epistemic_cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot aleatoric and epistemic uncertainty as side-by-side heatmaps.

    Each heatmap has rows = top-k histories (h_idx) and columns = target
    variables (D). A shared colour bar is added to each panel.

    Parameters
    ----------
    matrices        : UncertaintyMatrices  output of compute_uncertainty_attributes
    d_labels        : list[str] | None     column labels; defaults to "d=0", "d=1", …
    figsize         : (width, height) | None  defaults to (4*D, 0.4*top_k + 2)
    aleatoric_cmap  : matplotlib colormap name for the aleatoric panel
    epistemic_cmap  : matplotlib colormap name for the epistemic panel

    Returns
    -------
    matplotlib.figure.Figure
    """
    top_k, D = matrices.aleatoric.shape
    col_labels = d_labels if d_labels is not None else [f"d={d}" for d in range(D)]
    row_labels = [str(h) for h in matrices.h_idxs]

    if figsize is None:
        figsize = (max(8, 4 * D), max(4, 0.4 * top_k + 2))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    panels = [
        (axes[0], matrices.aleatoric, aleatoric_cmap, "Aleatoric entropy  H(h, d)"),
        (axes[1], matrices.epistemic, epistemic_cmap, "Epistemic variance  Var(h, d)"),
    ]

    for ax, data, cmap, title in panels:
        norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

        # Colour bar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Axis labels
        ax.set_xticks(range(D))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("Target variable d", fontsize=10)
        ax.set_ylabel("History h_idx", fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)

        # Annotate cells with values
        for i in range(top_k):
            for j in range(D):
                val = data[i, j]
                text_color = "white" if norm(val) > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    fig.tight_layout()
    return fig
