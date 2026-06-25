"""
var_generator.py
----------------
VAR (Vector Autoregression) data generator for KARMA synthetic benchmark.

Generates multivariate time series from a sparse VAR(K) process with a
known ground-truth adjacency structure G*, then recovers G* using PCMCI
(tigramite) so you have the three-way comparison: G* vs PCMCI vs KARMA.

Design choices aligned with KARMA paper:
  - D=4 variables (tractable state space for N=4 bins, K*=3)
  - K_true=3 lags (matches expected K* in equity experiments)
  - 6-8 directed edges injected (sparse, interpretable DAG)
  - 1 spurious edge injected only in train window (for G_f != G* test)
  - Coefficients kept small enough for stationarity
  - Noise is Gaussian; can be scaled to simulate heteroskedasticity

Usage
-----
    python var_generator.py

Outputs (saved to ./data/)
---------
    G_star.npy          — true adjacency tensor  (D, D, K_true)  bool
    G_star_edges.csv    — human-readable edge list
    train.npy           — training series        (T_train, D)
    test.npy            — test series            (T_test,  D)
    pcmci_edges.csv     — edges recovered by PCMCI on train data
    metadata.json       — all generation parameters
"""

# import os
# import json
import numpy as np

# import pandas as pd
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── parameters ───────────────────────────────────────────────────────────────
D = 4  # number of variables
K_TRUE = 3  # true VAR order  (= target K* for KARMA)
T_TRAIN = 5_000  # training series length
T_TEST = 1_000  # test series length
NOISE_STD = 0.3  # innovation standard deviation
COEF_BASE = 0.25  # coefficient magnitude on true edges
SPURIOUS_COEF = 0.20  # coefficient of injected spurious edge (train only)

VAR_NAMES = ["SPX", "SX5E", "NKY", "FTSE"]  # equity analogue labels

# ── ground-truth edge list  (source_var, target_var, lag) ────────────────────
# Designed to mimic known equity stylised facts:
#   momentum:           SPX(t-1)  -> SPX(t)
#   transatlantic:      SX5E(t-2) -> SPX(t)
#   asian spillover:    NKY(t-1)  -> SX5E(t)
#   volatility cluster: FTSE(t-1) -> FTSE(t)
#   cross-lag:          SPX(t-3)  -> NKY(t)
#   cross-lag:          NKY(t-2)  -> FTSE(t)
#   mean reversion:     SX5E(t-3) -> SX5E(t)   (7th edge)
TRUE_EDGES = [
    (0, 0, 1),  # SPX  -> SPX  lag 1  (momentum)
    (1, 0, 2),  # SX5E -> SPX  lag 2  (transatlantic contagion)
    (2, 1, 1),  # NKY  -> SX5E lag 1  (asian spillover)
    (3, 3, 1),  # FTSE -> FTSE lag 1  (volatility clustering)
    (0, 2, 3),  # SPX  -> NKY  lag 3  (cross-lag)
    (2, 3, 2),  # NKY  -> FTSE lag 2  (cross-lag)
    (1, 1, 3),  # SX5E -> SX5E lag 3  (mean reversion)
]

# Spurious edge present ONLY in train window — used to test G_f != G* recovery
# Variable 3 (FTSE) at lag 1 -> Variable 2 (NKY): a correlation that disappears
# in the test period, so PCMCI on raw data should NOT find it, but a model
# trained on train data may learn it, and KARMA should faithfully report it.
SPURIOUS_EDGE = (3, 2, 1)  # FTSE(t-1) -> NKY(t)  [train only]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build coefficient tensor
# ─────────────────────────────────────────────────────────────────────────────


def build_coef_tensor(edges, D, K, coef, existing=None):
    """Return (D, D, K) coefficient tensor. existing allows additive updates."""
    A = np.zeros((D, D, K)) if existing is None else existing.copy()
    for src, tgt, lag in edges:
        assert 1 <= lag <= K, f"lag {lag} out of range [1,{K}]"
        A[src, tgt, lag - 1] = coef
    return A


def check_stationarity(A, D, K, tol=0.97):
    """
    Companion matrix spectral radius check.
    VAR(K) is stationary iff max |eigenvalue| of companion matrix < 1.
    """
    comp = np.zeros((D * K, D * K))
    for k in range(K):
        comp[:D, k * D : (k + 1) * D] = A[:, :, k].T
    if K > 1:
        comp[D:, : D * (K - 1)] = np.eye(D * (K - 1))
    radius = np.max(np.abs(np.linalg.eigvals(comp)))
    return radius, radius < tol


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Simulate VAR series
# ─────────────────────────────────────────────────────────────────────────────


def simulate_var(A, D, K, T, noise_std, rng, burn_in=500):
    """
    Simulate T observations from VAR(K) with coefficient tensor A (D,D,K).
    A[i,j,k] = effect of X_i(t-k-1) on X_j(t).
    Returns array of shape (T, D).
    """
    total = T + burn_in
    X = np.zeros((total, D))
    # initialise with small noise
    X[:K] = rng.normal(0, noise_std, (K, D))

    for t in range(K, total):
        innovations = rng.normal(0, noise_std, D)
        x_next = innovations.copy()
        for k in range(K):
            x_next += A[:, :, k].T @ X[t - k - 1]
        X[t] = x_next

    return X[burn_in:]  # discard burn-in


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PCMCI recovery
# ─────────────────────────────────────────────────────────────────────────────


def run_pcmci(data, K, var_names, alpha=0.05):
    """
    Run PCMCI with ParCorr on data array (T, D).
    Returns edges as list of (src, tgt, lag, p_value, coefficient).
    """
    dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=K, pc_alpha=alpha)

    edges = []
    p_matrix = results["p_matrix"]  # shape (D, D, tau_max+1)
    val_matrix = results["val_matrix"]

    for tgt in range(data.shape[1]):
        for src in range(data.shape[1]):
            for lag in range(1, K + 1):
                p = p_matrix[src, tgt, lag]
                val = val_matrix[src, tgt, lag]
                if p < alpha:
                    edges.append(
                        {
                            "src": src,
                            "tgt": tgt,
                            "lag": lag,
                            "src_name": var_names[src],
                            "tgt_name": var_names[tgt],
                            "p_value": round(float(p), 5),
                            "coef": round(float(val), 5),
                        }
                    )
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Precision / recall vs G*
# ─────────────────────────────────────────────────────────────────────────────


def edge_metrics(recovered_edges, true_edges, D, K):
    """Compute precision, recall, F1 of recovered edge set vs true edge set."""
    true_set = {(s, t, l) for (s, t, l) in true_edges}
    rec_set = {(e["src"], e["tgt"], e["lag"]) for e in recovered_edges}

    tp = len(true_set & rec_set)
    fp = len(rec_set - true_set)
    fn = len(true_set - rec_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# # ─────────────────────────────────────────────────────────────────────────────
# # 5.  Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     save_directory = "data/generated/var"
#     os.makedirs(save_directory, exist_ok=True)

#     # ── build coefficient tensors ────────────────────────────────────────────
#     A_true     = build_coef_tensor(TRUE_EDGES,    D, K_TRUE, COEF_BASE)
#     A_spurious = build_coef_tensor([SPURIOUS_EDGE], D, K_TRUE, SPURIOUS_COEF,
#                                    existing=A_true)

#     radius_true, stationary = check_stationarity(A_true, D, K_TRUE)
#     print(f"Spectral radius (true DAG):      {radius_true:.4f}  "
#           f"{'✓ stationary' if stationary else '✗ NOT stationary'}")

#     radius_spur, _ = check_stationarity(A_spurious, D, K_TRUE)
#     print(f"Spectral radius (+ spurious):    {radius_spur:.4f}")

#     if not stationary:
#         raise ValueError("VAR is not stationary — reduce COEF_BASE.")

#     # ── simulate ─────────────────────────────────────────────────────────────
#     # Train: uses A_spurious (spurious edge active)
#     # Test:  uses A_true     (spurious edge absent — simulates regime change)
#     train = simulate_var(A_spurious, D, K_TRUE, T_TRAIN, NOISE_STD, rng)
#     test  = simulate_var(A_true,     D, K_TRUE, T_TEST,  NOISE_STD, rng)

#     print(f"\nTrain shape: {train.shape}   Test shape: {test.shape}")
#     print(f"Train mean:  {train.mean(axis=0).round(3)}")
#     print(f"Train std:   {train.std(axis=0).round(3)}")

#     # ── save arrays ──────────────────────────────────────────────────────────
#     np.save(save_directory + "/train.npy", train)
#     np.save(save_directory + "/test.npy",  test)

#     # ── G* adjacency tensor and edge list ────────────────────────────────────
#     G_star = np.zeros((D, D, K_TRUE), dtype=bool)
#     for (s, t, l) in TRUE_EDGES:
#         G_star[s, t, l - 1] = True
#     G_star_path = save_directory + "/G_star.npy"
#     np.save(G_star_path, G_star)

#     edge_df = pd.DataFrame([
#         {"src": s, "tgt": t, "lag": l,
#          "src_name": VAR_NAMES[s], "tgt_name": VAR_NAMES[t],
#          "coef": COEF_BASE, "spurious": False}
#         for (s, t, l) in TRUE_EDGES
#     ] + [
#         {"src": SPURIOUS_EDGE[0], "tgt": SPURIOUS_EDGE[1],
#          "lag": SPURIOUS_EDGE[2],
#          "src_name": VAR_NAMES[SPURIOUS_EDGE[0]],
#          "tgt_name": VAR_NAMES[SPURIOUS_EDGE[1]],
#          "coef": SPURIOUS_COEF, "spurious": True}
#     ])
#     edge_df.to_csv(save_directory + "/G_star_edges.csv", index=False)
#     print(f"\nG* edges ({len(TRUE_EDGES)} true + 1 spurious):")
#     print(edge_df.to_string(index=False))

#     # ── PCMCI on train data ───────────────────────────────────────────────────
#     print("\nRunning PCMCI on train data ...")
#     pcmci_edges = run_pcmci(train, K_TRUE, VAR_NAMES, alpha=0.05)
#     pcmci_df    = pd.DataFrame(pcmci_edges)
#     pcmci_df.to_csv(save_directory + "/pcmci_edges.csv", index=False)
#     print(f"PCMCI recovered {len(pcmci_edges)} edges:")
#     if pcmci_edges:
#         print(pcmci_df.to_string(index=False))

#     # ── metrics vs G* (excluding spurious from true set for fair eval) ────────
#     metrics = edge_metrics(pcmci_edges, TRUE_EDGES, D, K_TRUE)
#     print(f"\nPCMCI vs G* (excl. spurious):  {metrics}")

#     # check whether PCMCI recovered the spurious edge
#     spur_recovered = any(
#         e["src"] == SPURIOUS_EDGE[0] and
#         e["tgt"] == SPURIOUS_EDGE[1] and
#         e["lag"] == SPURIOUS_EDGE[2]
#         for e in pcmci_edges
#     )
#     print(f"Spurious edge recovered by PCMCI: {spur_recovered}  "
#           f"(expected: True — it IS in train data)")

#     # ── metadata ─────────────────────────────────────────────────────────────
#     meta = {
#         "generator":      "VAR",
#         "D":              D,
#         "K_true":         K_TRUE,
#         "T_train":        T_TRAIN,
#         "T_test":         T_TEST,
#         "noise_std":      NOISE_STD,
#         "coef_base":      COEF_BASE,
#         "spurious_coef":  SPURIOUS_COEF,
#         "spurious_edge":  list(SPURIOUS_EDGE),
#         "true_edges":     [list(e) for e in TRUE_EDGES],
#         "var_names":      VAR_NAMES,
#         "seed":           SEED,
#         "spectral_radius": round(radius_true, 6),
#         "pcmci_metrics":  metrics,
#         "pcmci_spurious_recovered": spur_recovered,
#     }
#     with open(save_directory + "/metadata_var.json", "w") as f:
#         json.dump(meta, f, indent=2)

#     print("\nSaved to " + save_directory + ": train.npy, test.npy, G_star.npy, "
#           "G_star_edges.csv, pcmci_edges.csv, metadata_var.json")


# if __name__ == "__main__":
#     main()
