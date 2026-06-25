"""
karma_level3.py
===============
Level 3 — Regime explanation via the marginal interdependence index.

For target variable d and history h in H+_{K*}, computes:

    Psi^d(h) = sum_{h' in H+_{K*}} pi*(h') * TV(T^{f,d}(.|h), T^{f,d}(.|h'))

the pi*-weighted average TV distance between the model's predictive
distribution at h and at a history drawn from the stationary distribution.
A high Psi^d(h) identifies h as a distinctive regime.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total variation distance between two probability vectors.

    TV(p, q) = 0.5 * sum_s |p(s) - q(s)|

    Parameters
    ----------
    p, q : np.ndarray of shape (N,)
        Probability vectors summing to 1.

    Returns
    -------
    float in [0, 1]
    """
    return 0.5 * float(np.sum(np.abs(p - q)))


def compute_psi(
    kernels: np.ndarray,
    pi_star: np.ndarray,
) -> np.ndarray:
    """
    Compute the marginal interdependence index Psi^d(h) for all histories h.

    Psi^d(h) = sum_{h'} pi*(h') * TV(T^{f,d}(.|h), T^{f,d}(.|h'))

    Parameters
    ----------
    kernels : np.ndarray of shape (H, N)
        Estimated marginal transition kernels for a single target variable d.
        kernels[i] = T^{f,d}(. | h_i), a probability vector over N bins.
        H = |H+_{K*}| is the number of observed histories.
    pi_star : np.ndarray of shape (H,)
        Stationary weights pi*(h) for each observed history.
        Must sum to 1 (or will be normalised internally).

    Returns
    -------
    psi : np.ndarray of shape (H,)
        Psi^d(h_i) for each observed history h_i.
    """
    H, N = kernels.shape
    assert pi_star.shape == (H,), f"pi_star shape {pi_star.shape} does not match H={H}"

    # Normalise pi* in case of floating-point drift
    pi_star = pi_star / pi_star.sum()

    # Pairwise TV matrix  shape (H, H)
    # TV(p, q) = 0.5 * ||p - q||_1
    # Vectorised: diff[i, j] = kernels[i] - kernels[j]  shape (H, H, N)
    # Avoid materialising (H, H, N) for large H — use loop over h
    # For H up to ~5000 the vectorised form is fine; for larger H use batching.

    if H <= 5000:
        # Fully vectorised  O(H^2 * N)
        diff = kernels[:, None, :] - kernels[None, :, :]  # (H, H, N)
        tv_matrix = 0.5 * np.abs(diff).sum(axis=2)  # (H, H)
        psi = tv_matrix @ pi_star  # (H,)
    else:
        # Batched to avoid memory issues for large H
        psi = _compute_psi_batched(kernels, pi_star, batch_size=1000)

    return psi


def _compute_psi_batched(
    kernels: np.ndarray,
    pi_star: np.ndarray,
    batch_size: int = 1000,
) -> np.ndarray:
    """Batched computation of Psi for large history spaces."""
    H = kernels.shape[0]
    psi = np.zeros(H)
    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        # diff shape (batch, H, N)
        diff = kernels[start:end, None, :] - kernels[None, :, :]
        tv_batch = 0.5 * np.abs(diff).sum(axis=2)  # (batch, H)
        psi[start:end] = tv_batch @ pi_star
    return psi


def top_k_histories(
    psi: np.ndarray,
    k: int,
    h_index: Optional[list] = None,
) -> dict:
    """
    Return the top-k most distinctive histories by Psi^d(h).

    Parameters
    ----------
    psi      : np.ndarray of shape (H,)   Psi^d values
    k        : int                         number of top histories to return
    h_index  : list of length H, optional  history identifiers (e.g. tuples)

    Returns
    -------
    dict with keys:
        "indices"  : np.ndarray of shape (k,)  — indices into psi array
        "psi"      : np.ndarray of shape (k,)  — Psi^d values, descending
        "histories": list of length k           — history identifiers if given
    """
    k = min(k, len(psi))
    top_idx = np.argsort(psi)[::-1][:k]
    result = {
        "indices": top_idx,
        "psi": psi[top_idx],
    }
    if h_index is not None:
        result["histories"] = [h_index[i] for i in top_idx]
    return result


def compute_level3(kernels: list[dict], k: int, pi_star: dict) -> dict:
    """
    Compute the full Level 3 output for all D target variables.

    Parameters
    ----------
    kernels_per_var : np.ndarray of shape (D, H, N)
        Estimated marginal transition kernels for all D target variables.
        kernels_per_var[d, i, :] = T^{f,d}(. | h_i)
    pi_star : np.ndarray of shape (H,)
        Stationary weights for each observed history.
    k : int
        Number of top-k distinctive histories to report per variable.
    h_index : list of length H, optional
        History identifiers for readable output.

    Returns
    -------
    dict with keys:
        "psi"       : np.ndarray of shape (D, H)  — Psi^d(h) for all d, h
        "top_k"     : list of D dicts             — top-k results per variable
                      each dict has "indices", "psi", optionally "histories"
        "psi_mean"  : np.ndarray of shape (H,)   — mean Psi across variables
    """
    level3_inputs = build_level3_inputs(kernels, pi_star)
    kernels_per_var = level3_inputs["kernels_per_var"]
    pi_star = level3_inputs["pi_star"]
    h_index = level3_inputs["h_index"]
    D, H, N = kernels_per_var.shape

    psi_all = np.zeros((D, H))
    for d in range(D):
        psi_all[d] = compute_psi(kernels_per_var[d], pi_star)

    top_k_per_var = [
        top_k_histories(psi_all[d], k=k, h_index=h_index) for d in range(D)
    ]

    # psi_topk[d, j] = Psi^d value for the j-th top-k history of variable d
    # shape (D, k)
    psi_topk = np.array([top["psi"] for top in top_k_per_var])

    # h_index_topk[d] = list of length k with the h_index labels for each top-k entry
    h_index_topk = [top["histories"] for top in top_k_per_var]

    return {
        "psi": psi_all,  # (D, H)  full psi matrix
        "psi_mean": psi_all.mean(axis=0),  # (H,)    mean across variables
        "top_k": top_k_per_var,  # list of D dicts with indices/psi/histories
        "psi_topk": psi_topk,  # (D, k)  psi values for top-k histories
        "h_index_topk": h_index_topk,  # list of D lists, each length k
    }


def plot_psi_topk_heatmap(
    level3_result: dict,
    var_names: Optional[list] = None,
    title: str = "Regime Distinctiveness (Ψ) — Top-k Histories",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a heatmap of psi_topk: rows = variables, columns = top-k histories.

    Parameters
    ----------
    level3_result : dict   output of compute_level3
    var_names     : list of D strings, e.g. ["X^0", ..., "X^5"]
    title         : figure title
    ax            : existing Axes to draw on; creates a new figure if None

    Returns
    -------
    fig : plt.Figure
    """
    psi_topk = level3_result["psi_topk"].T  # (k, D) for heatmap plotting
    h_index_topk = level3_result["h_index_topk"]  # list of D lists of length k

    k, D = psi_topk.shape

    if var_names is None:
        var_names = [f"X^{d}" for d in range(D)]

    # Build column labels: use h_index values from variable 0 (rank order shared)
    col_labels = [f"h={h}" for h in h_index_topk[0]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(3, D), max(4, k)))
    else:
        fig = ax.get_figure()

    im = ax.imshow(psi_topk, aspect="auto", cmap="YlOrRd", vmin=0)

    # Axes labels
    ax.set_xticks(range(D))
    ax.set_xticklabels(var_names, fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Top-k history (rank 1 = most distinctive)", fontsize=9)
    ax.set_ylabel("Target variable", fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)

    # Annotate cells with psi values
    for j in range(k):
        for d in range(D):
            val = psi_topk[j, d]
            text_color = "white" if val > 0.6 * psi_topk.max() else "black"
            ax.text(
                d,
                j,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    plt.colorbar(im, ax=ax, label="Ψ (regime distinctiveness)", fraction=0.03, pad=0.04)
    fig.tight_layout()
    return fig


def pairwise_regime_distance(
    kernels: np.ndarray,
    indices: np.ndarray,
    pi_star: np.ndarray,
) -> np.ndarray:
    """
    Compute the pairwise regime distance sub-matrix for a subset of histories.

    Delta^d_regime(h_i, h_j) = TV(T^{f,d}(.|h_i), T^{f,d}(.|h_j))

    Parameters
    ----------
    kernels : np.ndarray of shape (H, N)
        Marginal kernels for target variable d.
    indices : np.ndarray of shape (k,)
        Indices of the histories to include (e.g. top-k from top_k_histories).
    pi_star : np.ndarray of shape (H,)
        Stationary weights (not used in the distance itself, kept for API
        consistency; used externally for pi*-weighted clustering).

    Returns
    -------
    dist_matrix : np.ndarray of shape (k, k)
        Symmetric pairwise TV distance matrix.
    """
    sub = kernels[indices]  # (k, N)
    diff = sub[:, None, :] - sub[None, :, :]  # (k, k, N)
    dist_matrix = 0.5 * np.abs(diff).sum(axis=2)  # (k, k)
    return dist_matrix


def build_level3_inputs(sample_kernels: list, pi_star_lookup: dict) -> dict:
    """
    Convert sample_kernels list from the estimator into the arrays
    needed by compute_level3.

    Parameters
    ----------
    sample_kernels : list of dicts, each with keys:
        "h_idx"     : int         history index
        "h_arr"     : list        shape (K_star, D) decoded history
        "pool_size" : int
        "floor"     : float       noise floor rho^d_floor(h)
        "probs"     : list        shape (D, N) — marginal kernels for all D vars

    pi_star_lookup : dict  {h_idx: float}
        Stationary weight pi*(h) for each observed history.
        Pass your full pi* dict here, not just the sampled subset.

    Returns
    -------
    dict with:
        "kernels_per_var" : np.ndarray (D, H, N)
        "pi_star"         : np.ndarray (H,)
        "h_index"         : list of h_idx values, length H
        "floors"          : np.ndarray (H,)
    """
    if not sample_kernels:
        raise ValueError("sample_kernels is empty")

    H = len(sample_kernels)
    D = len(sample_kernels[0]["probs"])
    N = len(sample_kernels[0]["probs"][0])

    kernels_per_var = np.zeros((D, H, N))
    floors = np.zeros(H)
    h_index = []

    for i, sk in enumerate(sample_kernels):
        probs = np.array(sk["probs"])  # shape (D, N)
        # Normalise each row to guard against floating-point drift
        probs = probs / probs.sum(axis=1, keepdims=True)
        kernels_per_var[:, i, :] = probs
        floors[i] = sk["floor"]
        h_index.append(sk["h_idx"])

    # Build pi* vector aligned to h_index order
    pi_raw = np.array([pi_star_lookup.get(h, 0.0) for h in h_index])
    if pi_raw.sum() == 0:
        raise ValueError("All pi*(h) weights are zero for the sampled histories.")
    pi_star = pi_raw / pi_raw.sum()  # normalise over the sampled subset

    return {
        "kernels_per_var": kernels_per_var,
        "pi_star": pi_star,
        "h_index": h_index,
        "floors": floors,
    }
