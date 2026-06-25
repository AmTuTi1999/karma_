import numpy as np
from collections import defaultdict
from typing import Callable, Optional


from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import FactoredKernelEstimator


def compute_pi_star(X: np.ndarray, disc: Discretiser, K: int) -> dict:
    """
    Empirical stationary weights pi_hat*(h) from raw MVTS X.
    Unchanged from joint version — weights are over history space H_K
    regardless of whether the kernel is joint or factored.

    Returns
    -------
    pi_star : dict  h_idx -> float
    """
    T = X.shape[0]
    X_disc = disc.transform(X)
    counts = defaultdict(int)
    for t in range(K, T):
        h_idx = disc.history_to_index(X_disc[t - K : t])
        counts[h_idx] += 1
    total = sum(counts.values())
    return {h: cnt / total for h, cnt in counts.items()}


def compute_delta_pred(
    f: Callable,
    tilde_hs: np.ndarray,
    K: int,
    baseline: np.ndarray,
    loss: str = "classification",
) -> float:
    """Delta^pred(K, b) — average prediction change when prefix -> b."""
    n, W, D = tilde_hs.shape
    prefix_len = W - K
    assert baseline.shape == (prefix_len, D)
    total = 0.0
    for i in range(n):
        y_full = f(tilde_hs[i])
        padded = np.vstack([baseline, tilde_hs[i][-K:]])
        y_surr = f(padded)
        if loss == "classification":
            total += float(
                not np.array_equal(np.atleast_1d(y_full), np.atleast_1d(y_surr))
            )
        elif loss == "regression":
            total += float(
                np.mean(np.abs(np.atleast_1d(y_full) - np.atleast_1d(y_surr)))
            )
        else:
            raise ValueError(f"Unknown loss '{loss}'")
    return total / n


def compute_delta_f(
    est_K: FactoredKernelEstimator,
    est_K1: FactoredKernelEstimator,
    pi_star: dict,
    disc: Discretiser,
) -> float:
    """Factored TV stopping criterion Delta^f(K), averaged over D."""
    D = disc.D_
    delta = 0.0
    for h_idx, pi_h in pi_star.items():
        if pi_h == 0.0:
            continue
        tv_h = (
            sum(
                0.5
                * np.sum(
                    np.abs(
                        est_K.predict_marginal(d, h_idx)
                        - est_K1.predict_marginal(d, h_idx)
                    )
                )
                for d in range(D)
            )
            / D
        )
        delta += pi_h * tv_h
    return delta


def baseline_zeros(W: int, K: int, D: int) -> np.ndarray:
    return np.zeros((W - K, D))


def baseline_marginal_mean(X_train: np.ndarray, W: int, K: int) -> np.ndarray:
    T, D = X_train.shape
    if T < W:
        return np.zeros((W - K, D))
    prefixes = np.array([X_train[t : t + W - K] for t in range(T - W + 1)])
    return prefixes.mean(axis=0)


def baseline_central_bin(disc: Discretiser, W: int, K: int) -> np.ndarray:
    D, N = disc.D_, disc.N
    mid = np.zeros(D)
    for d in range(D):
        bounds = disc.quantiles_[d]
        c = N // 2
        lo = bounds[c - 1] if c > 0 else bounds[0] - 1.0
        hi = bounds[c] if c < N - 1 else bounds[-1] + 1.0
        mid[d] = 0.5 * (lo + hi)
    return np.tile(mid, (W - K, 1))


def _cache_outputs(f: Callable, tilde_hs: np.ndarray) -> np.ndarray:
    n = tilde_hs.shape[0]
    out0 = np.atleast_1d(f(tilde_hs[0]))
    outs = np.zeros((n,) + out0.shape)
    outs[0] = out0
    for i in range(1, n):
        outs[i] = np.atleast_1d(f(tilde_hs[i]))
    return outs


def select_K_and_baseline(
    f: Callable,
    X_train: np.ndarray,
    X_val: np.ndarray,
    disc: Discretiser,
    W: int,
    eps: float = 0.05,
    K_max: Optional[int] = None,
    loss: str = "classification",
    alpha: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    K* and b* selection with delta_pred as the sole stopping certificate.

    Stopping criterion (Pillar 2 certificate)
    ------------------------------------------
    K* = min{ K >= 1 : Delta^pred(K, b*_K) < eps }

    Delta^pred is a direct oracle test — no kernel estimation involved.
    It certifies that replacing any prefix with b* changes f's prediction
    by at most eps on average. This is the compression theorem certificate
    and the foundation of certified-zero attributions for lags > K*.

    Role of Delta^f (advisory diagnostic only)
    -------------------------------------------
    Delta^f(K) is computed and reported at each K as a consistency
    diagnostic showing where the empirical kernel approximately converges.
    It is NOT part of the stopping condition.

    Rationale: Delta^f relies on empirical kernel estimates that may be
    dominated by the Dirichlet prior when counts are sparse. When most
    histories have few oracle visits, TV distances between successive-order
    kernels are small not because the model has converged but because both
    estimates are near-uniform. Using Delta^f as a stopping condition would
    therefore certify K* for the wrong reason. Delta^pred has no such
    problem — it is a direct oracle comparison requiring no kernel estimation.

    Parameters
    ----------
    f         : oracle Callable (W, D) -> (D,) prediction
    tilde_hs  : np.ndarray (n, W, D)  held-out oracle query windows
                n <= T_held - W  (all independent)
    X_train   : np.ndarray (T, D)  raw MVTS for pi_hat* weights
    disc      : Discretiser (fitted)
    W         : model input window size
    baselines : dict name -> callable(W, K) -> np.ndarray (W-K, D)
    eps       : tolerance for Delta^pred stopping condition
    K_max     : maximum K to search (default W)
    loss      : "classification" or "regression"
    alpha     : Dirichlet prior for empirical kernel (diagnostic only)
    verbose   : print per-K diagnostics

    Returns
    -------
    dict with keys:
        K_star       : int    certified Markov order
        b_star_name  : str    name of selected baseline
        b_star       : ndarray (W-K*, D)  model-certified baseline
        delta_pred   : float  certified Delta^pred(K*, b*) < eps
        delta_f      : float  Delta^f(K*) — diagnostic only, not certified
        pi_star      : dict   h_idx -> float  stationary weights at K*
        history      : list   per-K diagnostic rows
    """

    n_q = len(X_val) - W
    tilde_hs = X_val[:n_q]  # (n_q, W, D) held-out query windows for prefix->b tests
    if verbose:
        print(f"Held-out query windows: n={n_q} (all independent, n <= T_held-W)")

    n, W_data, D = tilde_hs.shape
    assert W_data == W
    K_max = K_max or W
    outs = _cache_outputs(f, tilde_hs)

    history_log = []
    best_b_name, best_b = None, None
    best_delta_pred = float("inf")
    pi_star_K = {}

    baselines = {
        "zeros": lambda W, K: baseline_zeros(W, K, D),
        "marginal_mean": lambda W, K: baseline_marginal_mean(X_train, W, K),
        "central_bin": lambda W, K: baseline_central_bin(disc, W, K),
    }

    for K in range(1, K_max + 1):
        pi_star_K = compute_pi_star(X_train, disc, K)

        # ── Delta^f: advisory diagnostic, not used in stopping condition ──
        # Computed from empirical kernel which may be prior-dominated when
        # counts are sparse. Reported for paper Table diagnostics only.
        est_K = FactoredKernelEstimator(disc, K=K, alpha=alpha)
        est_K.record_queries_batch(tilde_hs, outs)

        # ── Delta^pred: SOLE stopping certificate (direct oracle test) ────
        # No kernel estimation. Certifies oracle insensitivity to prefix.
        best_delta_pred = float("inf")
        best_b_name = None
        best_b = None
        prefix_len = W - K

        if prefix_len == 0:
            # K = W: prefix is empty, surrogate is trivially valid
            best_delta_pred = 0.0
            best_b_name = "none (K=W)"
            best_b = np.zeros((0, D))
        else:
            for b_name, b_spec in baselines.items():
                b = b_spec(W, K) if callable(b_spec) else b_spec
                if b.shape[0] != prefix_len:
                    continue
                dp = compute_delta_pred(f, tilde_hs, K, b, loss=loss)
                if dp < best_delta_pred:
                    best_delta_pred = dp
                    best_b_name = b_name
                    best_b = b

        history_log.append(
            {
                "K": K,
                "delta_pred": best_delta_pred,  # certificate
                "b_name": best_b_name,
                "memory_fraction": K / W,
                "n_observed_histories": len(pi_star_K),
            }
        )

        if verbose:
            print(
                f"  K={K:2d} | "
                f"Δ^pred={best_delta_pred:.4f} [CERT] | "
                f"b={best_b_name} | "
                f"K/W={K/W:.2f} | "
                f"|H+|={len(pi_star_K)}"
            )

        # ── Stopping condition: Delta^pred only ───────────────────────────
        if best_delta_pred < eps:
            if verbose:
                print(
                    f"\n  => K* = {K}, b* = {best_b_name}"
                    f"\n     Certificate: Δ^pred = {best_delta_pred:.4f} < {eps}"
                    f"\n     Compression: W/K* = {W}/{K} = {W/K:.1f}x"
                    f"\n     Certified zeros for lags k > {K}"
                )
            return {
                "K_star": K,
                "b_star_name": best_b_name,
                "b_star": best_b,
                "delta_pred": best_delta_pred,  # the certificate
                "pi_star": pi_star_K,
                "history": history_log,
            }

    if verbose:
        print(f"\n  => Warning: Δ^pred did not reach {eps} by K_max={K_max}.")
        print(f"     Best Δ^pred = {best_delta_pred:.4f} at K={K_max}.")
        print("     Consider increasing K_max or eps.")
    return {
        "K_star": K_max,
        "b_star_name": best_b_name,
        "b_star": best_b,
        "delta_pred": best_delta_pred,
        "pi_star": pi_star_K,
        "history": history_log,
    }
