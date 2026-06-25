import numpy as np
from karma.utils.discretiser import Discretiser


def tmin_B(n_pool_min: int, n_observed_histories: int, W: int) -> int:
    """T_min = W + n_pool_min * |H+_{K*}|  (linear in observed histories)."""
    return W + n_pool_min * n_observed_histories


def tmin_A(N: int, D: int, K: int, lam: float, W: int) -> int:
    """T_min (empirical counting) = W + (2N/lambda^2) * N^{DK}."""
    return W + int(np.ceil(2 * N / lam**2)) * (N ** (D * K))


def tmin_C(W: int, n_pool_min: int, n_observed_histories: int) -> int:
    """T_min = W + n_pool_min * length_of_leaves (linear in leaf count)."""
    return W + n_pool_min * n_observed_histories


def check_design(
    disc: Discretiser,
    K: int,
    T_train: int,
    n_pool_min: int = 10,
    lam: float = 0.025,
    W: int = 30,
    kernel_type: str = "C",
) -> None:
    """
    Print design guideline check before running Pillar 3.

    Tells the practitioner whether T_train is sufficient for the chosen
    N, K combination, and what to do if not.
    """
    N = disc.N
    D = disc.D_
    # rough expected |H+| for stationary process
    n_hist_est = min(T_train - K, N ** (D * K))
    mean_pool = T_train / max(n_hist_est, 1)
    if kernel_type == "C":
        t_need = tmin_C(W, n_pool_min, n_hist_est)
    elif kernel_type == "B":
        t_need = tmin_B(n_pool_min, n_hist_est, W)
    elif kernel_type == "A":
        t_need = tmin_A(N, D, K, lam, W)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    print(f"\nDesign check (D={D}, N={N}, K={K}, T_train={T_train:,}):")
    print(f"  Expected |H+_{K}|:      ~{n_hist_est:,}")
    print(f"  Expected mean pool:    ~{mean_pool:.1f}")
    print(f"  T_min needed:          ~{t_need:,}")
    print(f"  T_train sufficient:    {'YES' if T_train >= t_need else 'NO'}")
    if T_train < t_need:
        print(
            f"  -> Need T_train >= {t_need:,}, or decrease N to "
            f"{max(2, N-1)} or decrease K to {max(1, K-1)}"
        )
    else:
        print(
            f"  -> Proceed with Pillar 3. "
            f"Use M > {int(2/lam**2):,} for MC coverage."
        )


def decode_history(h_idx: int, K: int, D: int, N: int) -> np.ndarray:
    """
    Decode flat history index back to (K, D) int array.
    Inverse of disc.history_to_index.
    """
    h_arr = np.zeros((K, D), dtype=int)
    idx = h_idx
    for k in reversed(range(K)):
        for d in reversed(range(D)):
            h_arr[k, d] = idx % N
            idx //= N
    return h_arr
