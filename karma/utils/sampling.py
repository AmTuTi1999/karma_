import numpy as np
from collections import defaultdict
from typing import Optional
from .discretiser import Discretiser


class SuffixPool:
    """
    Index of training windows grouped by their discretised K-suffix.

    For each h in H+_{K*}, stores all training windows whose discretised
    K-suffix equals h. These provide authentic within-bin continuous
    variation for Monte Carlo kernel estimation.

    Design check
    ------------
    After building, call coverage_stats() to verify mean_pool_size >= n_pool_min.
    If not, either increase T_train or decrease N (see module docstring).

    Parameters
    ----------
    disc : Discretiser
    K    : int   K* from select_K_and_baseline
    W    : int   oracle input window size
    """

    def __init__(self, disc: Discretiser, K: int, W: int):
        self.disc = disc
        self.K = K
        self.W = W
        self._pool: dict = defaultdict(list)  # h_idx -> list of (W, D) arrays

    def build(self, X_train: np.ndarray) -> "SuffixPool":
        """
        Index all W-length training windows by their discretised K-suffix.

        Parameters
        ----------
        X_train : np.ndarray (T, D) or (n, W, D)
        """
        if X_train.ndim == 3:
            # Reconstruct time-series from windowed form (assumes overlapping windows)
            n, W_data, D_data = X_train.shape
            if W_data != self.W:
                raise ValueError(
                    f"Expected window length W={self.W} but got {W_data} in X_train"
                )
            X_train = np.vstack([X_train[0]] + [X_train[i, -1:] for i in range(1, n)])

        T, D = X_train.shape
        X_disc = self.disc.transform(X_train)

        for t in range(self.W, T):
            suffix_disc = X_disc[t - self.K : t]  # (K, D)
            h_idx = self.disc.history_to_index(suffix_disc)
            window_cont = X_train[t - self.W : t].copy()  # (W, D)
            self._pool[h_idx].append(window_cont)

        return self

    def sample(self, h_idx: int, rng: np.random.Generator) -> Optional[np.ndarray]:
        """
        Sample one continuous window from pool(h).
        Returns None if pool is empty.
        """
        pool = self._pool.get(h_idx, [])
        if not pool:
            return None
        return pool[rng.integers(0, len(pool))].copy()

    def pool_size(self, h_idx: int) -> int:
        return len(self._pool.get(h_idx, []))

    @property
    def observed_histories(self) -> list:
        return list(self._pool.keys())

    def coverage_stats(self, pi_star: dict, n_pool_min: int = 10) -> dict:
        """
        Report pool depth statistics.

        A pool_coverage close to 1.0 at n_pool_min=10 means Pillar 3
        kernel estimates are reliable. If pool_coverage is low, either
        increase T_train or decrease N.
        """
        total_pi = sum(pi_star.values())
        covered_pi = 0.0
        sizes = []

        for h_idx, pi_h in pi_star.items():
            sz = self.pool_size(h_idx)
            sizes.append(sz)
            if sz >= n_pool_min:
                covered_pi += pi_h

        return {
            "n_observed_histories": len(pi_star),
            "pool_coverage": round(covered_pi / total_pi, 4) if total_pi > 0 else 0.0,
            "mean_pool_size": round(float(np.mean(sizes)), 2) if sizes else 0.0,
            "min_pool_size": int(np.min(sizes)) if sizes else 0,
            "max_pool_size": int(np.max(sizes)) if sizes else 0,
        }
