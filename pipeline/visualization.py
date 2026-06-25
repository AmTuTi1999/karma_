"""
Unified KARMA Level 1–5 visualization pipeline.

Reads karma_results.json produced by pipeline/karma_pipeline.py and
generates publication-quality figures for all five explanation levels.

Level 1  Variable importance           Φ̃ᵈ   horizontal bar chart
Level 2  Lag profiles                  φᵈₖ   line plot per variable
Level 3  Regime distance matrix        TV(T̂(·|h), T̂(·|h′))  clustered heatmap
Level 4  Ranked causal effects         ρ(e)  horizontal bar chart
Level 5  Uncertainty-aware dashboard   aleatoric · epistemic · noise floor · coverage map

Usage:
    python -m pipeline.visualization --dataset etth1 --model lstm
    python -m pipeline.visualization --results results/etth1/lstm/karma_results.json
    python -m pipeline.visualization --results results/etth1/lstm/karma_results.json \\
        --target OT --format pdf --lam 0.025
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_FG = [
    "#185FA5",  # blue
    "#1D9E75",  # teal
    "#D85A30",  # coral
    "#BA7517",  # amber
    "#534AB7",  # purple
    "#444441",  # dark gray
    "#0F6E56",  # dark teal
    "#993C1D",  # dark coral
    "#854F0B",  # dark amber
    "#3C3489",  # dark purple
]
_FG_LIGHT = [
    "#B5D4F4", "#9FE1CB", "#F5C4B3", "#FAC775", "#AFA9EC",
    "#B4B2A9", "#5DCAA5", "#F0997B", "#EF9F27", "#7F77DD",
]
_ZERO_COL = "#D3D1C7"
_ZERO_ECOL = "#B4B2A9"
_RED = "#E24B4A"
_AMBER = "#EF9F27"
_GREEN = "#1D9E75"
_GREEN_LIGHT = "#9FE1CB"
_AMBER_LIGHT = "#FAC775"
_RED_LIGHT = "#F7C1C1"
_GRAY = "#888780"
_LGRAY = "#D3D1C7"
_BLUE_RAMP = ["#185FA5", "#378ADD", "#85B7EB", "#B5D4F4", "#E6F1FB"]

_ZERO_THRESH = 1e-10


def _var_colour(d: int, active: list[int]) -> str:
    if d in active:
        return _FG[active.index(d) % len(_FG)]
    return _ZERO_COL


def _var_colour_light(d: int, active: list[int]) -> str:
    if d in active:
        return _FG_LIGHT[active.index(d) % len(_FG_LIGHT)]
    return _ZERO_ECOL


# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────


def _set_theme() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "axes.edgecolor": _GRAY,
            "grid.color": _LGRAY,
            "grid.linewidth": 0.4,
            "grid.linestyle": "--",
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.color": _GRAY,
            "ytick.color": _GRAY,
            "legend.fontsize": 8,
            "legend.framealpha": 0.75,
            "legend.edgecolor": _LGRAY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and extraction
# ─────────────────────────────────────────────────────────────────────────────


def load_results(path: str | Path) -> dict:
    """Load karma_results.json produced by pipeline/karma_pipeline.py."""
    with open(path) as f:
        return json.load(f)


def _extract(results: dict, lam_override: float | None = None) -> dict:
    """Parse all arrays and scalars needed by the five level plots."""
    cfg = results["config"]
    p2 = results["pillar2"]
    p3 = results["pillar3"]
    vi = p3["variable_importance"]

    D = cfg["D"]
    N = cfg.get("N", 3)
    M = cfg.get("M", 32)
    K_star = p2["K_star"]
    lam = lam_override if lam_override is not None else cfg.get("lam", 0.025)
    delta_pred = p2["delta_pred"]

    Phi_n = np.array(vi["Phi_n"])
    phi_raw = np.array(vi["phi"])          # (D, K_star) or (D, K_max)
    phi_dk = phi_raw[:, :K_star]
    phi_dk = np.where(np.abs(phi_dk) < _ZERO_THRESH, 0.0, phi_dk)

    active = [d for d in range(D) if Phi_n[d] > _ZERO_THRESH]
    edges = p3.get("retained_edges", [])
    sample_kernels = p3.get("sample_kernels", [])

    # Variable names: prefer stored list from config, then infer from edges
    stored_names = cfg.get("var_names")
    if stored_names and len(stored_names) == D:
        var_names = list(stored_names)
    else:
        var_names = [f"$X^{{{d}}}$" for d in range(D)]
        for e in edges:
            if "src_name" in e:
                var_names[e["src"]] = e["src_name"]
            if "tgt_name" in e:
                var_names[e["tgt"]] = e["tgt_name"]

    return dict(
        D=D, N=N, M=M, K_star=K_star, lam=lam, delta_pred=delta_pred,
        Phi_n=Phi_n, phi_dk=phi_dk, active=active,
        var_names=var_names, edges=edges, sample_kernels=sample_kernels,
        dataset=cfg.get("dataset", ""), oracle=cfg.get("oracle", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — Variable importance
# ─────────────────────────────────────────────────────────────────────────────


def plot_level1(
    ex: dict,
    var_names: list[str] | None = None,
    figsize: tuple = (6.0, 3.8),
) -> plt.Figure:
    """
    Level 1: horizontal bar chart of normalised variable importance Φ̃ᵈ.

    Parameters
    ----------
    ex        : dict from _extract()
    var_names : override variable labels
    figsize   : figure size in inches

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    D, K_star, lam = ex["D"], ex["K_star"], ex["lam"]
    Phi_n, active = ex["Phi_n"], ex["active"]
    names = var_names or ex["var_names"]

    order = np.argsort(Phi_n)[::-1]
    phi_ord = Phi_n[order]
    labs = [names[i] for i in order]
    cols = [_var_colour(i, active) for i in order]
    ecols = [_var_colour_light(i, active) for i in order]
    is_zero = phi_ord < _ZERO_THRESH

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        range(D), phi_ord,
        color=cols, edgecolor=ecols, linewidth=0.6, height=0.58, zorder=3,
    )

    for bar, val, zero in zip(bars, phi_ord, is_zero):
        if zero:
            ax.text(
                lam * 0.12, bar.get_y() + bar.get_height() / 2,
                "certified zero", va="center", ha="left",
                fontsize=7.5, color=_LGRAY, style="italic",
            )
        else:
            ax.text(
                bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8, color=_GRAY,
            )

    ax.axvline(lam, color=_RED, linewidth=0.9, linestyle="--", zorder=4,
               label=f"$\\lambda = {lam}$")

    first_zero = int(np.searchsorted(-phi_ord, -_ZERO_THRESH))
    if 0 < first_zero < D:
        ax.axhline(first_zero - 0.5, color=_LGRAY, linewidth=0.6, linestyle=":", zorder=2)

    active_sorted = np.sort(Phi_n[Phi_n > _ZERO_THRESH])[::-1]
    cumul = np.cumsum(active_sorted)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(cumul)
    ax2.set_xticklabels([f"{c:.2f}" for c in cumul], fontsize=7.5, color=_GRAY)
    ax2.set_xlabel("Cumulative $\\sum\\tilde{\\Phi}^{d'}$", fontsize=8, color=_GRAY)
    ax2.spines["top"].set_visible(True)
    ax2.spines["top"].set_linewidth(0.4)
    ax2.spines["top"].set_color(_LGRAY)
    ax2.tick_params(axis="x", colors=_GRAY, width=0.4)

    ax.set_yticks(range(D))
    ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised importance $\\tilde{\\Phi}^{d'}$")
    ax.set_title(
        f"Level 1  $\\cdot$  Variable importance\n"
        f"$D={D}$,  $K^*={K_star}$,  $\\lambda={lam}$",
        loc="left", pad=4,
    )
    ax.set_xlim(0, max(phi_ord) * 1.38)
    ax.legend(loc="lower right", handlelength=1.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — Lag profiles
# ─────────────────────────────────────────────────────────────────────────────


def plot_level2(
    ex: dict,
    var_names: list[str] | None = None,
    ci_pct: float = 18.0,
    figsize: tuple = (6.5, 3.6),
) -> plt.Figure:
    """
    Level 2: line plot of lag-resolved influence φᵈₖ vs lag k.
    Only active (non-zero) variables are plotted.

    Parameters
    ----------
    ex      : dict from _extract()
    var_names : override variable labels
    ci_pct  : ± percentage band (pseudo-CI). Set to 0 to suppress.
    figsize : figure size

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    D, K_star, lam = ex["D"], ex["K_star"], ex["lam"]
    phi_dk, active = ex["phi_dk"], ex["active"]
    names = var_names or ex["var_names"]

    marks = ["o", "s", "^", "D", "v", "P", "X", "h", "8", "*"]
    lags = np.arange(1, K_star + 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, d in enumerate(active):
        vals = phi_dk[d]
        if vals.max() < _ZERO_THRESH:
            continue
        c = _FG[i % len(_FG)]
        ax.plot(
            lags, vals, color=c, linewidth=1.8,
            marker=marks[i % len(marks)], markersize=5.5,
            label=names[d], zorder=4,
        )
        if ci_pct > 0:
            frac = ci_pct / 100.0
            ax.fill_between(
                lags, vals * (1 - frac), vals * (1 + frac),
                color=c, alpha=0.09, zorder=2,
            )

    ax.axhline(lam, color=_RED, linewidth=0.9, linestyle="--", zorder=3,
               label=f"$\\lambda = {lam}$")

    ax.set_xticks(lags)
    ax.set_xticklabels([f"$k={k}$" for k in lags])
    ax.set_xlabel("Lag $k$")
    ax.set_ylabel("Lag-resolved influence $\\phi^{d'}_k$")
    ax.set_title(
        f"Level 2  $\\cdot$  Lag profiles  $\\phi^{{d'}}_k$\n"
        f"$K^*={K_star}$ lags  (certified-zero variables omitted)",
        loc="left", pad=4,
    )
    ax.legend(loc="upper right", title="source variable",
              title_fontsize=7.5, ncol=2, fontsize=8)
    ax.set_xlim(0.7, K_star + 0.3)
    ax.set_ylim(-0.02, phi_dk.max() * 1.28)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — Regime distance matrix
# ─────────────────────────────────────────────────────────────────────────────


def _build_regime_arrays(
    sample_kernels: list[dict],
    target_var: str,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Extract (H, N) kernel matrix and metadata for a single target variable
    from the sample_kernels list saved in karma_results.json.

    Returns
    -------
    kernels     : (H, N) probability arrays
    pi_approx   : (H,) stationary weights, approximated from pool sizes
    coverage    : (H,) coverage bar per history  (1 - floor / (lam/2)) clipped
    hist_labels : list[str] of length H
    """
    H = len(sample_kernels)
    kernels = np.zeros((H, N))
    pool_sizes = np.zeros(H)
    floors = np.zeros(H)
    labels = []

    for i, sk in enumerate(sample_kernels):
        probs = sk["probs"]
        if isinstance(probs, dict):
            # New format: dict keyed by var_name
            ker = np.array(probs.get(target_var, list(probs.values())[0]))
        else:
            # Legacy format: list of per-dim arrays; use index 0
            ker = np.array(probs[0])
        ker = ker / ker.sum()
        kernels[i] = ker[:N]
        pool_sizes[i] = sk.get("pool_size", 1)
        floors[i] = sk.get("floor", 0.0)

        h_arr = sk.get("h_arr", [])
        if h_arr:
            flat = [int(v) for row in h_arr for v in row]
            labels.append(f"h=({','.join(str(v) for v in flat[:6])}…)")
        else:
            labels.append(f"h{sk['h_idx']}")

    pi_approx = pool_sizes / pool_sizes.sum()
    # coverage: treat floor >= lam/2 as zero coverage, floor=0 as full
    coverage = floors   # raw floor values; interpreted in the plot

    return kernels, pi_approx, coverage, labels


def plot_level3(
    ex: dict,
    target_var: str | None = None,
    var_names: list[str] | None = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """
    Level 3: regime distance matrix Δ_regime(h, h′) = TV(T̂(·|h), T̂(·|h′)).

    Clustered heatmap of pairwise TV distances between transition kernels.
    Each cell is the TV distance between two sampled history profiles.
    Histories with a high noise floor are flagged with hatching.

    Parameters
    ----------
    ex         : dict from _extract()
    target_var : name of the target variable (default: first var_name)
    var_names  : override variable labels
    figsize    : figure size

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    N, lam = ex["N"], ex["lam"]
    names = var_names or ex["var_names"]
    sample_kernels = ex["sample_kernels"]
    K_star = ex["K_star"]

    if not sample_kernels:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No sample kernels available in results",
                ha="center", va="center", transform=ax.transAxes, color=_GRAY)
        ax.set_title("Level 3  $\\cdot$  Regime distance matrix", loc="left")
        return fig

    # Resolve target variable name
    if target_var is None:
        probs0 = sample_kernels[0]["probs"]
        target_var = next(iter(probs0)) if isinstance(probs0, dict) else names[0]

    kernels, pi_approx, floors, hist_labels = _build_regime_arrays(
        sample_kernels, target_var, N
    )
    H = kernels.shape[0]

    # TV distance matrix
    tv_mat = np.zeros((H, H))
    for i in range(H):
        for j in range(H):
            tv_mat[i, j] = 0.5 * np.sum(np.abs(kernels[i] - kernels[j]))

    # Hierarchical clustering (only if H > 2)
    if H > 2:
        condensed = tv_mat[np.triu_indices(H, k=1)]
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    else:
        order = np.arange(H)
        Z = None

    tv_ord = tv_mat[np.ix_(order, order)]
    labs_ord = [hist_labels[i] for i in order]
    pi_ord = pi_approx[order]
    floor_ord = floors[order]

    # Figure layout: dendrogram (left) + heatmap (right)
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[0.12, 1], height_ratios=[1, 0.12],
        hspace=0.04, wspace=0.04,
    )
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])
    ax_dend.axis("off")
    ax_bot.axis("off")

    if H > 2 and Z is not None:
        dendrogram(
            Z, orientation="left", ax=ax_dend, no_labels=True,
            link_color_func=lambda k: _GRAY,
            above_threshold_color=_GRAY,
        )
        ax_dend.set_xlim(ax_dend.get_xlim()[::-1])
        for sp in ax_dend.spines.values():
            sp.set_visible(False)
        ax_dend.tick_params(left=False, bottom=False)

    cmap = sns.color_palette("Blues", as_cmap=True)
    im = ax_heat.imshow(tv_ord, cmap=cmap, vmin=0, vmax=0.5, aspect="auto", zorder=2)

    for i in range(H):
        for j in range(H):
            val = tv_ord[i, j]
            tc = "white" if val > 0.28 else "#2C2C2A"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=8, color=tc)

    ax_heat.set_xticks(range(H))
    ax_heat.set_yticks(range(H))
    ax_heat.set_xticklabels(labs_ord, rotation=35, ha="right", fontsize=8)
    ax_heat.set_yticklabels(labs_ord, fontsize=8)
    ax_heat.set_title(
        f"Level 3  $\\cdot$  Regime distance matrix  target = {target_var}\n"
        f"$\\Delta_{{\\mathrm{{regime}}}}(h, h') = TV(\\hat{{T}}(\\cdot|h),\\, \\hat{{T}}(\\cdot|h'))$",
        loc="left", pad=6,
    )

    # π* weight bar on right
    ax_pi = ax_heat.inset_axes([1.02, 0, 0.04, 1])
    for i, pi_h in enumerate(pi_ord):
        ax_pi.barh(i, pi_h, height=0.8, color=_FG[0], alpha=0.75)
    ax_pi.set_xlim(0, max(pi_ord) * 1.5)
    ax_pi.set_ylim(-0.5, H - 0.5)
    ax_pi.set_yticks([])
    ax_pi.set_xlabel("$\\hat{\\pi}^*$", fontsize=7.5)
    ax_pi.invert_yaxis()
    for sp in ax_pi.spines.values():
        sp.set_visible(False)
    ax_pi.tick_params(labelsize=7)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.12)
    cbar.set_label("TV distance", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    fig.subplots_adjust(left=0.18, right=0.88, top=0.88, bottom=0.12)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Level 4 — Ranked causal effects
# ─────────────────────────────────────────────────────────────────────────────


def plot_level4(
    ex: dict,
    var_names: list[str] | None = None,
    top_n: int = 25,
    figsize: tuple = (6.5, 5.8),
) -> plt.Figure:
    """
    Level 4: horizontal bar chart of TV edge contributions ρ(e), ranked
    descending, coloured by source variable.

    Parameters
    ----------
    ex        : dict from _extract()
    var_names : override variable labels
    top_n     : maximum edges to display
    figsize   : figure size

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    K_star, lam, active = ex["K_star"], ex["lam"], ex["active"]
    names = var_names or ex["var_names"]
    edges = sorted(ex["edges"], key=lambda e: e["rho"], reverse=True)[:top_n]
    n_e = len(edges)

    if n_e == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No retained edges", ha="center", va="center",
                transform=ax.transAxes, color=_GRAY)
        ax.set_title("Level 4  $\\cdot$  Ranked causal effects", loc="left")
        return fig

    labels = [
        f"{names[e['src']]} → {names[e['tgt']]},  $k={e['lag']}$"
        for e in edges
    ]
    rho_vals = [e["rho"] for e in edges]
    retained = [e["rho"] > lam for e in edges]
    bar_cols = [_var_colour(e["src"], active) if r else _ZERO_COL
                for e, r in zip(edges, retained)]
    bar_ecs = [_var_colour_light(e["src"], active) if r else _ZERO_ECOL
               for e, r in zip(edges, retained)]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        range(n_e), rho_vals,
        color=bar_cols, edgecolor=bar_ecs, linewidth=0.6, height=0.58, zorder=3,
    )
    ax.axvline(lam, color=_RED, linewidth=1.0, linestyle="--", zorder=4,
               label=f"$\\lambda = {lam}$")

    for bar, e, ret in zip(bars, edges, retained):
        is_self = e["src"] == e["tgt"]
        suffix = "  (self)" if is_self else ""
        ax.text(
            bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{e['rho']:.4f}{suffix}", va="center", ha="left",
            fontsize=7.5, color=_GRAY if ret else "#B4B2A9",
        )

    ax.set_yticks(range(n_e))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"TV edge contribution  $\rho(e)$")
    ax.set_title(
        f"Level 4  $\\cdot$  Ranked causal effects\n"
        f"Top {n_e} edges,  $\\lambda = {lam}$,  $K^* = {K_star}$",
        loc="left", pad=4,
    )
    ax.set_xlim(0, max(rho_vals) * 1.48)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    src_vars = sorted({e["src"] for e in edges if e["rho"] > lam})
    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor=_var_colour(d, active),
                edgecolor=_var_colour_light(d, active),
                linewidth=0.6,
                label=names[d],
            )
            for d in src_vars
        ],
        loc="lower right", fontsize=8,
        title="source variable", title_fontsize=7.5, framealpha=0.85,
    )

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Level 5 — Uncertainty-aware dashboard
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _HistoryU:
    h_idx: int
    kernel: np.ndarray
    n_queries: int
    aleatoric: float
    epistemic: np.ndarray
    noise_floor: float
    coverage: float
    reliable: bool
    pi_star: float


def _aleatoric(kernel: np.ndarray) -> float:
    k = np.clip(kernel, 1e-12, 1.0)
    return float(-np.sum(k * np.log(k)))


def _epistemic_var(kernel: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.full_like(kernel, np.inf)
    return kernel * (1.0 - kernel) / n


def _build_level5_results(
    sample_kernels: list[dict],
    N: int,
) -> list[_HistoryU]:
    """Build _HistoryU records with aleatoric/epistemic averaged over all D target variables."""
    rows = []
    pool_sizes = [sk.get("pool_size", 1) for sk in sample_kernels]
    pi_total = sum(pool_sizes) or 1.0

    for sk in sample_kernels:
        probs = sk["probs"]
        if isinstance(probs, dict):
            all_kerns = np.stack([np.array(v, dtype=float) for v in probs.values()])
        else:
            all_kerns = np.array(probs, dtype=float)
        all_kerns = all_kerns / all_kerns.sum(axis=1, keepdims=True)  # (D, N)

        ps = sk.get("pool_size", 1)
        pi_h = ps / pi_total

        avg_aleatoric = float(np.mean([_aleatoric(k) for k in all_kerns]))
        avg_epistemic = np.mean(
            np.stack([_epistemic_var(k, ps) for k in all_kerns]), axis=0
        )  # (N,)

        rows.append(_HistoryU(
            h_idx=sk["h_idx"],
            kernel=all_kerns.mean(axis=0),
            n_queries=ps,
            aleatoric=avg_aleatoric,
            epistemic=avg_epistemic,
            noise_floor=sk.get("floor", 0.0),
            coverage=1.0,
            reliable=True,
            pi_star=pi_h,
        ))

    rows.sort(key=lambda r: r.aleatoric, reverse=True)
    return rows


def _tier_colour(r: _HistoryU, lam: float) -> str:
    if r.noise_floor >= lam / 2.0:
        return _RED
    elif r.noise_floor >= lam / 4.0:
        return _AMBER
    return _GREEN


def _tier_colour_light(r: _HistoryU, lam: float) -> str:
    if r.noise_floor >= lam / 2.0:
        return _RED_LIGHT
    elif r.noise_floor >= lam / 4.0:
        return _AMBER_LIGHT
    return _GREEN_LIGHT


def _reliability_legend(lam: float) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(facecolor=_GREEN, edgecolor=_GREEN_LIGHT, linewidth=0.6,
                       label=f"reliable  (floor < λ/4 = {lam/4:.3f})"),
        mpatches.Patch(facecolor=_AMBER, edgecolor=_AMBER_LIGHT, linewidth=0.6,
                       label=f"marginal  (λ/4 ≤ floor < λ/2 = {lam/2:.3f})"),
        mpatches.Patch(facecolor=_RED, edgecolor=_RED_LIGHT, linewidth=0.6,
                       label=f"unreliable  (floor ≥ λ/2 = {lam/2:.3f})"),
    ]


def _panel5_aleatoric(ax, rows, labels, N):
    n = len(rows)
    values = [r.aleatoric for r in rows]

    bars = ax.barh(
        range(n), values,
        color=_FG[0], edgecolor=_FG_LIGHT[0],
        linewidth=0.6, height=0.55, zorder=3, alpha=0.85,
    )
    for bar, val, r in zip(bars, values, rows):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=7.5, color=_GRAY)
        if bar.get_width() > 0.15:
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
                    f"n={r.n_queries}", va="center", ha="right", fontsize=6.5, color="white")

    max_H = np.log(N)
    ax.axvline(max_H, color=_GRAY, linewidth=0.8, linestyle="--", zorder=2,
               label=f"max H = ln({N}) = {max_H:.2f}")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Aleatoric entropy  $H^d_h$  (nats, avg over variables)")
    ax.set_title("A  ·  Aleatoric uncertainty per history", loc="left", pad=6)
    ax.set_xlim(0, max_H * 1.25)
    ax.legend(loc="lower right", handlelength=1.5)
    ax.invert_yaxis()


def _panel5_epistemic(ax, rows, labels, lam):
    n = len(rows)
    N = rows[0].kernel.shape[0]
    lefts = np.zeros(n)

    for s in range(N):
        values = np.array([
            float(np.sqrt(r.epistemic[s])) if not np.isinf(r.epistemic[s]) else 0.0
            for r in rows
        ])
        ax.barh(
            range(n), values, left=lefts,
            color=_BLUE_RAMP[s % len(_BLUE_RAMP)],
            edgecolor="white", linewidth=0.4, height=0.55,
            label=f"s′ = {s}", zorder=3,
        )
        lefts += values

    for i, r in enumerate(rows):
        total = float(np.sum(np.sqrt(r.epistemic[r.epistemic < np.inf]))) if r.n_queries > 0 else 0.0
        ax.text(lefts[i] + 0.001, i, f"{total:.3f}", va="center", ha="left",
                fontsize=7.5, color=_GRAY)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Epistemic std  $\\sqrt{\\mathrm{Var}(\\hat{T}(s'|h))}$  (avg over variables)")
    ax.set_title("B  ·  Epistemic uncertainty by bin", loc="left", pad=6)
    ax.legend(loc="lower right", title="next bin", title_fontsize=7)
    ax.invert_yaxis()


def _panel5_avg_kernel(ax, sample_kernels: list[dict], N: int, names: list[str]) -> None:
    """Panel C: π*-weighted average marginal distribution per target variable."""
    if not sample_kernels:
        ax.text(0.5, 0.5, "No sample kernels", ha="center", va="center",
                transform=ax.transAxes, color=_GRAY)
        return

    pool_sizes = np.array([sk.get("pool_size", 1) for sk in sample_kernels], dtype=float)
    pi_weights = pool_sizes / pool_sizes.sum()

    probs0 = sample_kernels[0]["probs"]
    var_name_list = list(probs0.keys()) if isinstance(probs0, dict) else [
        names[d] if d < len(names) else f"X^{d}" for d in range(len(probs0))
    ]
    D = len(var_name_list)

    p_avg = np.zeros((D, N))
    for i, sk in enumerate(sample_kernels):
        probs = sk["probs"]
        for j, vn in enumerate(var_name_list):
            k = np.array(probs[vn] if isinstance(probs, dict) and vn in probs
                         else np.ones(N), dtype=float)
            k /= k.sum()
            p_avg[j] += pi_weights[i] * k

    x = np.arange(D)
    bottoms = np.zeros(D)
    for s in range(N):
        ax.bar(x, p_avg[:, s], bottom=bottoms,
               color=_BLUE_RAMP[s % len(_BLUE_RAMP)],
               edgecolor="white", linewidth=0.4,
               label=f"bin {s}", zorder=3)
        bottoms += p_avg[:, s]

    display_labels = [names[i] if i < len(names) else vn
                      for i, vn in enumerate(var_name_list)]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel("$\\pi^*$-weighted probability")
    ax.set_title(
        "C  ·  $\\pi^*$-weighted marginal distribution per target variable",
        loc="left", pad=6,
    )
    ax.axhline(1.0 / N, color=_LGRAY, linewidth=0.7, linestyle="--", zorder=2,
               label=f"uniform 1/{N}")
    ax.legend(loc="upper right", title="bin", title_fontsize=7, fontsize=8, ncol=N)
    ax.set_ylim(0, 1.15)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)


def _panel5_coverage_map(ax, rows, labels, lam, N):
    aleat = np.array([r.aleatoric for r in rows])
    epist = np.array([
        float(np.sum(np.sqrt(r.epistemic[r.epistemic < np.inf]))) if r.n_queries > 0 else 0.0
        for r in rows
    ])
    sizes = np.array([max(60, r.pi_star * 2800) for r in rows])

    ax.scatter(aleat, epist, s=sizes,
               c=[_FG[0]] * len(rows), edgecolors=[_FG_LIGHT[0]] * len(rows),
               linewidths=1.0, alpha=0.88, zorder=4)

    for i, (r, lab) in enumerate(zip(rows, labels)):
        ax.annotate(lab, xy=(aleat[i], epist[i]), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5, color=_GRAY)

    max_H = np.log(N)
    ax.axvline(max_H * 0.75, color=_GRAY, linewidth=0.6, linestyle="--",
               zorder=2, alpha=0.6)
    if len(epist) > 2:
        ax.axhline(np.percentile(epist, 66), color=_GRAY, linewidth=0.6,
                   linestyle=":", zorder=2, alpha=0.6)

    for pi_val, lab in [(0.10, "π*=0.10"), (0.20, "π*=0.20")]:
        ax.scatter([], [], s=max(60, pi_val * 2800), c=[_LGRAY],
                   edgecolors=[_GRAY], linewidths=0.8, label=lab, alpha=0.7)

    ax.text(0.04, 0.04, "ideal\n(low both)", transform=ax.transAxes,
            fontsize=7.5, color=_GREEN, va="bottom")
    ax.text(0.62, 0.04, "high aleatoric\n(model uncertain)", transform=ax.transAxes,
            fontsize=7.5, color=_AMBER, va="bottom")
    ax.text(0.04, 0.72, "high epistemic\n(need more data)", transform=ax.transAxes,
            fontsize=7.5, color=_RED, va="bottom")

    ax.set_xlabel("Aleatoric entropy  $H^d_h$  (nats)")
    ax.set_ylabel("Total epistemic std  $\\Sigma\\sqrt{\\mathrm{Var}(\\hat{T}(s'|h))}$")
    ax.set_title("D  ·  Coverage map", loc="left", pad=6)
    ax.legend(loc="upper right", title="bubble size", title_fontsize=7)


def plot_level5(
    ex: dict,
    var_names: list[str] | None = None,
    max_rows: int = 20,
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """
    Level 5: four-panel uncertainty-aware dashboard (averaged over all target variables).

    Panel A  Aleatoric entropy per history — π*-weighted average over D variables
    Panel B  Epistemic std per bin — π*-weighted average over D variables
    Panel C  π*-weighted marginal distribution per target variable
    Panel D  Coverage map: aleatoric vs epistemic scatter

    Parameters
    ----------
    ex        : dict from _extract()
    var_names : override variable labels
    max_rows  : cap on number of histories shown in bar panels
    figsize   : figure size

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    N, lam, delta_pred, M = ex["N"], ex["lam"], ex["delta_pred"], ex["M"]
    names = var_names or ex["var_names"]
    D = ex["D"]
    sample_kernels = ex["sample_kernels"]

    if not sample_kernels:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No sample kernels available in results",
                ha="center", va="center", transform=ax.transAxes, color=_GRAY)
        ax.set_title("Level 5  ·  Uncertainty dashboard", loc="left")
        return fig

    rows = _build_level5_results(sample_kernels, N)
    rows = rows[:max_rows]
    n = len(rows)
    labels = []
    for r in rows:
        sk = next((s for s in sample_kernels if s["h_idx"] == r.h_idx), None)
        if sk and sk.get("h_arr"):
            flat = [int(v) for row in sk["h_arr"] for v in row]
            labels.append(f"h=({','.join(str(v) for v in flat[:6])}…)")
        else:
            labels.append(f"h{r.h_idx}")

    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.text(0.5, 0.985,
             f"Level 5  ·  Uncertainty-aware explanation  ·  {D} variables ($\\pi^*$-weighted)",
             ha="center", va="top", fontsize=12, color="#2C2C2A")
    fig.text(0.5, 0.962,
             f"$\\lambda = {lam}$   ·   $\\hat{{\\Delta}}_{{\\mathrm{{pred}}}} = {delta_pred}$"
             f"   ·   $M = {M}$   ·   {n} sampled histories",
             ha="center", va="top", fontsize=8.5, color=_GRAY)

    gs = fig.add_gridspec(
        2, 2, left=0.12, right=0.96, top=0.91, bottom=0.08,
        hspace=0.48, wspace=0.38,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    _panel5_aleatoric(ax_A, rows, labels, N)
    _panel5_epistemic(ax_B, rows, labels, lam)
    _panel5_avg_kernel(ax_C, sample_kernels, N, names)
    _panel5_coverage_map(ax_D, rows, labels, lam, N)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Combined overview (Levels 1 + 2 + 4 in one page)
# ─────────────────────────────────────────────────────────────────────────────


def plot_overview(
    ex: dict,
    var_names: list[str] | None = None,
    top_n: int = 20,
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """
    Three-panel overview: Level 1 | Level 2 | Level 4.
    Suitable for a single-page summary in a paper appendix.

    Parameters
    ----------
    ex        : dict from _extract()
    var_names : override variable labels
    top_n     : max edges to show in Level 4 panel
    figsize   : figure size

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    D, K_star, lam = ex["D"], ex["K_star"], ex["lam"]
    Phi_n, phi_dk, active = ex["Phi_n"], ex["phi_dk"], ex["active"]
    names = var_names or ex["var_names"]
    edges = sorted(ex["edges"], key=lambda e: e["rho"], reverse=True)[:top_n]
    dataset, oracle = ex.get("dataset", ""), ex.get("oracle", "")
    delta_pred = ex["delta_pred"]

    fig, axes = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1.1, 1.3]},
    )

    # ── Panel A: Level 1 ─────────────────────────────────────────────────────
    ax = axes[0]
    order = np.argsort(Phi_n)[::-1]
    phi_ord = Phi_n[order]
    is_zero = phi_ord < _ZERO_THRESH
    bars = ax.barh(
        range(D), phi_ord,
        color=[_var_colour(i, active) for i in order],
        edgecolor=[_var_colour_light(i, active) for i in order],
        linewidth=0.6, height=0.58, zorder=3,
    )
    for bar, val, zero in zip(bars, phi_ord, is_zero):
        if zero:
            ax.text(lam * 0.12, bar.get_y() + bar.get_height() / 2,
                    "zero", va="center", ha="left", fontsize=6.5, color=_LGRAY, style="italic")
        else:
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=7.5, color=_GRAY)
    ax.axvline(lam, color=_RED, linewidth=0.8, linestyle="--", zorder=4,
               label=f"$\\lambda={lam}$")
    fz = int(np.searchsorted(-phi_ord, -_ZERO_THRESH))
    if 0 < fz < D:
        ax.axhline(fz - 0.5, color=_LGRAY, linewidth=0.5, linestyle=":", zorder=2)
    ax.set_yticks(range(D))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("$\\tilde{\\Phi}^{d'}$", fontsize=9)
    ax.set_title("A  $\\cdot$  Variable importance", loc="left", pad=4, fontsize=9.5)
    ax.set_xlim(0, max(phi_ord) * 1.38)
    ax.legend(loc="lower right", fontsize=7.5, handlelength=1.2)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    # ── Panel B: Level 2 ─────────────────────────────────────────────────────
    ax = axes[1]
    marks = ["o", "s", "^", "D", "v", "P", "X", "h"]
    lags = np.arange(1, K_star + 1)
    for i, d in enumerate(active):
        vals = phi_dk[d]
        if vals.max() < _ZERO_THRESH:
            continue
        c = _FG[i % len(_FG)]
        ax.plot(lags, vals, color=c, linewidth=1.6,
                marker=marks[i % len(marks)], markersize=5, label=names[d], zorder=4)
        ax.fill_between(lags, vals * 0.82, vals * 1.18, color=c, alpha=0.09, zorder=2)
    ax.axhline(lam, color=_RED, linewidth=0.8, linestyle="--", zorder=3,
               label=f"$\\lambda={lam}$")
    ax.set_xticks(lags)
    ax.set_xticklabels([f"$k={k}$" for k in lags])
    ax.set_xlabel("Lag $k$", fontsize=9)
    ax.set_ylabel("$\\phi^{d'}_k$", fontsize=9)
    ax.set_title("B  $\\cdot$  Lag profiles", loc="left", pad=4, fontsize=9.5)
    ax.legend(loc="upper right", title="source", title_fontsize=7, ncol=2, fontsize=7.5)
    ax.set_xlim(0.7, K_star + 0.3)
    ax.set_ylim(-0.02, phi_dk.max() * 1.28)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    # ── Panel C: Level 4 ─────────────────────────────────────────────────────
    ax = axes[2]
    n_e = len(edges)
    if n_e > 0:
        elabels = [f"{names[e['src']]}→{names[e['tgt']]}, $k={e['lag']}$" for e in edges]
        rho_vals = [e["rho"] for e in edges]
        retained = [e["rho"] > lam for e in edges]
        bars = ax.barh(
            range(n_e), rho_vals,
            color=[_var_colour(e["src"], active) if r else _ZERO_COL for e, r in zip(edges, retained)],
            edgecolor=[_var_colour_light(e["src"], active) if r else _ZERO_ECOL for e, r in zip(edges, retained)],
            linewidth=0.5, height=0.58, zorder=3,
        )
        ax.axvline(lam, color=_RED, linewidth=0.8, linestyle="--", zorder=4,
                   label=f"$\\lambda={lam}$")
        for bar, e, ret in zip(bars, edges, retained):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{e['rho']:.3f}", va="center", ha="left", fontsize=7,
                    color=_GRAY if ret else "#B4B2A9")
        ax.set_yticks(range(n_e))
        ax.set_yticklabels(elabels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(rho_vals) * 1.45)
        src_vars = sorted({e["src"] for e in edges if e["rho"] > lam})
        ax.legend(
            handles=[mpatches.Patch(
                facecolor=_var_colour(d, active), edgecolor=_var_colour_light(d, active),
                linewidth=0.5, label=names[d],
            ) for d in src_vars],
            loc="lower right", fontsize=7, title="source", title_fontsize=7,
        )
    ax.set_xlabel(r"$\rho(e)$", fontsize=9)
    ax.set_title("C  $\\cdot$  Ranked causal effects", loc="left", pad=4, fontsize=9.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    fig.suptitle(
        f"KARMA explanation hierarchy  ·  {dataset} / {oracle}"
        f"  ·  $D={D}$,  $K^*={K_star}$,  $\\lambda={lam}$,"
        f"  $\\hat{{\\Delta}}_{{\\mathrm{{pred}}}}={delta_pred:.4f}$",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main entry: generate all figures from a results JSON
# ─────────────────────────────────────────────────────────────────────────────


def visualize(
    results_path: str | Path,
    figures_dir: str | Path,
    target_var: str | None = None,
    var_names: list[str] | None = None,
    lam: float | None = None,
    top_n: int = 25,
    fmt: str = "pdf",
    dpi: int = 220,
    verbose: bool = True,
) -> dict[str, plt.Figure]:
    """
    Generate all five KARMA explanation levels and save to figures_dir.

    Parameters
    ----------
    results_path : path to karma_results.json
    figures_dir  : output directory for saved figures
    target_var   : variable name used for Level 3 and Level 5 (default: first)
    var_names    : override all variable labels (length D list)
    lam          : override lambda from the JSON config
    top_n        : max edges in Level 4
    fmt          : 'pdf' | 'png' | 'both'
    dpi          : dots per inch (used for raster formats)
    verbose      : print progress

    Returns
    -------
    dict mapping figure name -> plt.Figure
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    ex = _extract(results, lam_override=lam)

    if verbose:
        print(f"\n{'='*60}")
        print(f"KARMA Visualization  {ex['dataset']} / {ex['oracle']}")
        print(f"  D={ex['D']}  K*={ex['K_star']}  λ={ex['lam']}")
        print(f"  figures → {figures_dir}/")
        print("=" * 60)

    figs: dict[str, plt.Figure] = {}

    figs["level1_importance"] = plot_level1(ex, var_names=var_names)
    figs["level2_lagprofiles"] = plot_level2(ex, var_names=var_names)
    figs["level3_regime"] = plot_level3(ex, target_var=target_var, var_names=var_names)
    figs["level4_causal"] = plot_level4(ex, var_names=var_names, top_n=top_n)
    figs["level5_uncertainty"] = plot_level5(ex, var_names=var_names)
    figs["overview"] = plot_overview(ex, var_names=var_names, top_n=top_n)

    exts = ["pdf", "png"] if fmt == "both" else [fmt]
    for name, fig in figs.items():
        for ext in exts:
            path = figures_dir / f"karma_{name}.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            if verbose:
                print(f"  saved → {path}")
        plt.close(fig)

    if verbose:
        print(f"\nAll figures saved to {figures_dir}/")

    return figs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    "etth1", "ettm2", "weather", "exchange_rate",
    "beijing", "electricity", "web_traffic",
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate KARMA Level 1–5 figures from pipeline output."
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--results", type=str,
        help="Path to karma_results.json (explicit)",
    )
    source.add_argument(
        "--dataset", type=str, choices=DATASETS,
        help="Dataset name (resolves results path from configs)",
    )

    p.add_argument(
        "--model", type=str, default="lstm", choices=["lstm", "tcn"],
        help="Model type (used with --dataset, default: lstm)",
    )
    p.add_argument(
        "--figures_dir", type=str, default=None,
        help="Output directory for figures (default: <results_dir>/figures)",
    )
    p.add_argument(
        "--target", type=str, default=None,
        help="Target variable name for Level 3 and Level 5",
    )
    p.add_argument(
        "--lam", type=float, default=None,
        help="Override lambda from the JSON config",
    )
    p.add_argument(
        "--top_n", type=int, default=25,
        help="Max edges shown in Level 4 (default: 25)",
    )
    p.add_argument(
        "--format", dest="fmt", default="pdf",
        choices=["pdf", "png", "both"],
        help="Output format (default: pdf)",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # Resolve results path
    if args.results:
        results_path = Path(args.results)
    else:
        dataset_cfg = yaml.safe_load(
            (CONFIGS_DIR / "datasets" / f"{args.dataset}.yaml").read_text()
        )
        results_dir = Path(dataset_cfg["results_dir"]) / args.model
        results_path = results_dir / "karma_results.json"

    if not results_path.exists():
        print(f"Error: results file not found: {results_path}")
        raise SystemExit(1)

    # Resolve figures directory
    if args.figures_dir:
        figures_dir = Path(args.figures_dir)
    else:
        figures_dir = results_path.parent / "figures"

    visualize(
        results_path=results_path,
        figures_dir=figures_dir,
        target_var=args.target,
        lam=args.lam,
        top_n=args.top_n,
        fmt=args.fmt,
        dpi=args.dpi,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
