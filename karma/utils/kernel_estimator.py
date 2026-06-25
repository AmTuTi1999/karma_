import math
import numpy as np
from collections import defaultdict
from typing import Optional, Callable, Protocol
from abc import ABC, abstractmethod
from tqdm import tqdm
from karma.utils.discretiser import Discretiser
from karma.utils.sampling import SuffixPool


class KernelEstimatorProtocol(Protocol):
    K: int

    def predict_marginal(self, d: int, h_idx: int, step: int = 0) -> np.ndarray:
        """Returns (N,) marginal distribution over bin of X^d at forecast step."""
        ...

    def noise_floor_mc(self, h_idx: int) -> float:
        """Monte Carlo noise floor: sqrt(1/2M)."""
        ...

    @property
    def estimated_histories(self) -> list: ...


def _normalize_output(output_raw: np.ndarray) -> np.ndarray:
    """
    Normalize oracle output to shape (L, D).

    Accepts:
        scalar or 0-d array  -> (1, 1)
        (D,) 1-d array       -> (1, D)  single-step
        (L, D) 2-d array     -> (L, D)  multi-step (returned as-is)
    """
    arr = np.array(output_raw, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[None, :]  # (1, D)
    return arr  # (L, D)


class KernelEstimator(ABC):
    """
    Estimates the model kernel T^f_K from oracle queries.

    Supports both single-step (L=1) and multi-step (L>1) forecasts.
    The horizon L is auto-detected from the first oracle output shape.

    For each oracle query i:
      - tilde_h_i  : full W-length continuous window fed to f
      - h_i^(K)    : K-length discretised suffix (conditioning history)
      - s'_i       : psi(f(tilde_h_i)), discretised model output — shape (L, D)

    Stores counts n^f(h, l, s') and n^f(h) and computes the
    Dirichlet posterior mean with Jeffreys prior (alpha = 0.5).

    Parameters
    ----------
    disc : Discretiser
        Fitted discretiser.
    K : int
        Markov order for this estimator.
    alpha : float
        Dirichlet prior concentration (default 0.5 = Jeffreys prior).
    """

    def __init__(self, disc: Discretiser, K: int, alpha: float = 0.5):
        self.disc = disc
        self.K = K
        self.alpha = alpha
        self._nf_h: dict = defaultdict(int)
        self.n_queries = 0
        self._horizon: Optional[int] = None

    @property
    def horizon(self) -> int:
        """Forecast horizon L (auto-detected from first oracle query)."""
        return self._horizon if self._horizon is not None else 1

    def _set_horizon(self, L: int) -> None:
        if self._horizon is None:
            self._horizon = L
        elif self._horizon != L:
            raise ValueError(
                f"Inconsistent forecast horizon: expected {self._horizon}, got {L}"
            )

    def _extract_suffix_and_output(
        self, tilde_h: np.ndarray, output_raw: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """
        Compute (h_idx, s_primes) from a raw oracle query.

        Parameters
        ----------
        tilde_h    : np.ndarray (W, D) — full continuous input window
        output_raw : np.ndarray        — f(tilde_h); shape (D,) for single-step
                                         or (L, D) for multi-step

        Returns
        -------
        h_idx    : int             — flat index of the K-length discretised suffix
        s_primes : np.ndarray (L, D) — discretised outputs per forecast step
        """
        W, D = tilde_h.shape
        assert W >= self.K, f"Window W={W} must be >= K={self.K}"

        suffix_disc = self.disc.transform(tilde_h[-self.K :])  # (K, D) int
        h_idx = self.disc.history_to_index(suffix_disc)

        output_norm = _normalize_output(output_raw)  # (L, D)
        s_primes = self.disc.transform(output_norm)  # (L, D) int

        return h_idx, s_primes

    def _record_shared(self, h_idx: int):
        """Increment the shared marginal count n^f(h) and total query counter."""
        self._nf_h[h_idx] += 1
        self.n_queries += 1

    @abstractmethod
    def record_query(self, tilde_h: np.ndarray, output_raw: np.ndarray):
        """Record one oracle query. Implemented by each subclass."""

    def record_queries_batch(self, tilde_hs: np.ndarray, outputs: np.ndarray):
        """
        Record a batch of oracle queries.

        Parameters
        ----------
        tilde_hs : np.ndarray (n, W, D)
        outputs  : np.ndarray
            (n,) for univariate single-step,
            (n, D) for multivariate single-step, or
            (n, L, D) for multivariate multi-step.
        """
        outputs = np.asarray(outputs, dtype=float)
        if outputs.ndim == 1:
            outputs = outputs[:, None]  # (n,) -> (n, 1) univariate single-step
        for i in range(tilde_hs.shape[0]):
            self.record_query(tilde_hs[i], outputs[i])

    def n_f_h(self, h_idx: int) -> int:
        """Return n^f(h): number of oracle queries with suffix h."""
        return self._nf_h.get(h_idx, 0)

    @property
    def visited_histories(self) -> list:
        """List of h_idx values with n^f(h) > 0."""
        return list(self._nf_h.keys())

    def noise_floor(self, h_idx: int) -> float:
        """
        Finite-sample noise floor rho_floor(h) = sqrt(N^D / (2 * n^f(h))).
        Returns inf if h was never visited.
        """
        nfh = self.n_f_h(h_idx)
        if nfh == 0:
            return float("inf")
        return np.sqrt(self.disc.n_states / (2.0 * nfh))

    @abstractmethod
    def predict(self, h_idx: int) -> np.ndarray:
        """
        Return a distribution (or per-step distributions) given history h_idx.

        Returns
        -------
        probs : np.ndarray
            (L, D, N) for factored estimators — per-step marginal distributions.
            (L, n_states) for joint estimators — per-step joint distributions.
        """


class JointKernelEstimator(KernelEstimator):
    """
    Concrete joint kernel estimator supporting multi-step forecasts.

    Stores a per-step joint count table n^f(h, l, s') over the full N^D
    state space and estimates T^f_K(s' | h, l) via Dirichlet posterior mean.

    For L-step forecasts the oracle returns (L, D) and counts are stored
    per step l in [L].  l=0 recovers single-step behaviour.
    """

    def __init__(self, disc: Discretiser, K: int, alpha: float = 0.5):
        super().__init__(disc, K, alpha)
        # n^f(h, l, s'): (h_idx, step, s_idx) -> int
        self._nf_joint: dict = defaultdict(int)

    def record_query(
        self, tilde_h: np.ndarray, output_raw, disc_output: Optional[np.ndarray] = None
    ):
        """
        Record one oracle query.

        Parameters
        ----------
        tilde_h    : np.ndarray (W, D)
        output_raw : array-like — shape (D,) for single-step or (L, D) for multi-step
        disc_output : np.ndarray, optional
            Pre-discretised output; shape (D,) or (L, D).
        """
        if disc_output is not None:
            W, D = tilde_h.shape
            assert W >= self.K, f"Window W={W} must be >= K={self.K}"
            suffix_disc = self.disc.transform(tilde_h[-self.K :])
            h_idx = self.disc.history_to_index(suffix_disc)
            s_primes = _normalize_output(np.array(disc_output, dtype=float))  # (L, D)
        else:
            output_raw = np.array(output_raw, dtype=float)
            if output_raw.ndim == 1:
                assert output_raw.shape[0] == self.disc.D_, (
                    f"Oracle output dim {output_raw.shape[0]} != D={self.disc.D_}. "
                    f"Ensure f returns a ({self.disc.D_},) or (L, {self.disc.D_}) array."
                )
            h_idx, s_primes = self._extract_suffix_and_output(tilde_h, output_raw)

        L = s_primes.shape[0]
        self._set_horizon(L)

        for step in range(L):
            s_idx = self.disc.state_to_index(s_primes[step])
            self._nf_joint[(h_idx, step, s_idx)] += 1
        self._record_shared(h_idx)

    def predict(self, h_idx: int, step: int = 0) -> np.ndarray:
        """
        Return Dirichlet posterior mean T^f_K(. | h) for the given forecast step.

        Returns
        -------
        probs : np.ndarray (n_states,)
        """
        S = self.disc.n_states
        counts = np.array(
            [self._nf_joint.get((h_idx, step, s), 0) for s in range(S)], dtype=float
        )
        return (counts + self.alpha) / (counts.sum() + S * self.alpha)

    def predict_all_steps(self, h_idx: int) -> np.ndarray:
        """
        Return (L, n_states) joint distributions for all forecast steps.
        """
        L = self.horizon
        return np.stack([self.predict(h_idx, step) for step in range(L)], axis=0)


class FactoredKernelEstimator(KernelEstimator):
    """
    Estimates D factored marginal kernels T^{f,d,l}_K(s'^d | h) from oracle
    queries, one per target variable d in [D] and forecast step l in [L].

    Joint kernel replaced by D*L independent marginal estimators.
    Each query contributes to all D*L marginals simultaneously.

    Count structure
    ---------------
    For each target variable d and forecast step l:
        n^{f,d,l}(h, s'^d) = queries with suffix h whose d-th output
                              at step l equals s'^d
        n^f(h)             = total queries with suffix h

    Marginal Dirichlet posterior mean:

        T^{f,d,l}_K(s'^d | h) = (n^{f,d,l}(h, s'^d) + alpha)
                                 / (n^f(h) + N * alpha)

    Parameters
    ----------
    disc  : Discretiser
    K     : int — Markov order
    alpha : float — Dirichlet prior (default 0.5 = Jeffreys)
    """

    def __init__(self, disc: Discretiser, K: int, alpha: float = 0.5):
        self.disc = disc
        self.K = K
        self.alpha = alpha
        D = disc.D_
        # _nf_marginal[d][(h_idx, step, s)] -> int
        self._nf_marginal: list = [defaultdict(int) for _ in range(D)]
        self._nf_h: dict = defaultdict(int)
        self._horizon: Optional[int] = None

    def _set_horizon(self, L: int) -> None:
        if self._horizon is None:
            self._horizon = L
        elif self._horizon != L:
            raise ValueError(
                f"Inconsistent forecast horizon: expected {self._horizon}, got {L}"
            )

    @property
    def horizon(self) -> int:
        return self._horizon if self._horizon is not None else 1

    def record_query(self, tilde_h: np.ndarray, output_raw: np.ndarray):
        W, D = tilde_h.shape
        suffix_disc = self.disc.transform(tilde_h[-self.K :])
        h_idx = self.disc.history_to_index(suffix_disc)

        output_norm = _normalize_output(np.array(output_raw, dtype=float))  # (L, D)

        # Pad or truncate to match D if needed (handles legacy edge cases)
        L_out, D_out = output_norm.shape
        if D_out < D:
            padded = np.zeros((L_out, D))
            padded[:, :D_out] = output_norm
            output_norm = padded
        elif D_out > D:
            output_norm = output_norm[:, :D]

        L = output_norm.shape[0]
        self._set_horizon(L)

        s_primes = self.disc.transform(output_norm)  # (L, D) int
        for step in range(L):
            for d in range(D):
                self._nf_marginal[d][(h_idx, step, int(s_primes[step, d]))] += 1
        self._nf_h[h_idx] += 1

    def record_queries_batch(self, tilde_hs: np.ndarray, outputs: np.ndarray):
        """
        Parameters
        ----------
        tilde_hs : (n, W, D)
        outputs  : (n,), (n, D), or (n, L, D)
        """
        outputs = np.asarray(outputs, dtype=float)
        if outputs.ndim == 1:
            outputs = outputs[:, None]  # (n,) -> (n, 1) univariate single-step
        for i in range(tilde_hs.shape[0]):
            self.record_query(tilde_hs[i], outputs[i])

    def predict_marginal(self, d: int, h_idx: int, step: int = 0) -> np.ndarray:
        """
        Return T^{f,d}_K(. | h) — marginal distribution for variable d at the given step.

        Returns (N,) array. Default step=0 preserves single-step behaviour.
        """
        N = self.disc.N
        counts = np.array(
            [self._nf_marginal[d].get((h_idx, step, s), 0) for s in range(N)],
            dtype=float,
        )
        return (counts + self.alpha) / (counts.sum() + N * self.alpha)

    def predict(self, h_idx: int) -> np.ndarray:
        """
        Return (L, D, N) array of per-step marginal distributions.

        predict(h_idx)[step, d, :] gives T^{f,d}_K(. | h) at forecast step.
        """
        L = self.horizon
        D = self.disc.D_
        return np.stack(
            [
                np.stack(
                    [self.predict_marginal(d, h_idx, step) for d in range(D)], axis=0
                )
                for step in range(L)
            ],
            axis=0,
        )  # (L, D, N)

    def n_f_h(self, h_idx: int) -> int:
        return self._nf_h.get(h_idx, 0)

    @property
    def visited_histories(self) -> list:
        return list(self._nf_h.keys())


class BStarKernelEstimator:
    """
    Estimates D factored marginal kernels T^{f,d,l}_{K*}(s'^d | h) via
    Monte Carlo using the model-certified baseline b* as the prefix.

    Supports multi-step forecasts: if f returns (L, D) per window, counts
    are stored as (L, D, N) arrays and predict returns (L, D, N).

    Construction of each oracle window tilde_h^{(m)}:
        1. Sample window from pool(h)  -- real observed training data
        2. Replace prefix with b*:
           tilde_h^{(m)} = b* ⊕ window[-K*:]
        3. Query oracle, record output bins per variable d and step l.

    Parameters
    ----------
    disc       : Discretiser
    pool       : SuffixPool  built from X_train at order K*
    K          : int         K*
    W          : int         oracle window size
    b_star     : np.ndarray  (W-K, D)  model-certified baseline
    delta_pred : float       certified prefix sensitivity from Pillar 2
    M          : int         Monte Carlo draws per history
    rng        : np.random.Generator
    """

    def __init__(
        self,
        disc: Discretiser,
        pool: SuffixPool,
        K: int,
        W: int,
        b_star: np.ndarray,
        delta_pred: float = 0.0,
        M: int = 3200,
        rng: Optional[np.random.Generator] = None,
    ):
        self.disc = disc
        self.pool = pool
        self.K = K
        self.W = W
        self.b_star = b_star
        self.delta_pred = delta_pred
        self.M = M
        self.rng = rng or np.random.default_rng(42)
        self._counts: dict = {}   # h_idx -> (L, D, N) int array
        self._n_draws: dict = {}  # h_idx -> int
        self._samples: dict = {}  # h_idx -> (M, L, D) int8 array of raw MC draws
        self._horizon: Optional[int] = None

    @property
    def horizon(self) -> int:
        return self._horizon if self._horizon is not None else 1

    def fit(
        self,
        f: Callable,
        pi_star: dict,
        verbose: bool = True,
        batched: bool = True,
        mega_batch: bool = False,
        mega_batch_size: int = 4096,
    ) -> None:
        """
        Estimate marginal kernel for all h in H+_{K*}.

        Parameters
        ----------
        f               : oracle Callable.
                          batched=True  -> expects (M, W, D), returns (M, D) or (M, L, D).
                          batched=False -> expects (W, D), returns (D,) or (L, D).
        pi_star         : dict h_idx -> float
        verbose         : print progress
        batched         : if True, all M windows are stacked into a single batch per
                          history (H oracle calls total).
        mega_batch      : if True, collect windows from ALL histories into one array
                          and issue at most ceil(H*M / mega_batch_size) oracle calls
                          total. Maximises GPU throughput at the cost of peak RAM
                          proportional to H*M*W*D.  Overrides `batched`.
        mega_batch_size : chunk size for the mega-batch oracle calls.
        """
        D = self.disc.D_
        N = self.disc.N
        prefix_len = self.W - self.K
        n_hist = len(pi_star)

        if verbose:
            print(f"  BStarKernelEstimator: |H+_K|={n_hist}, M={self.M}")
            if mega_batch:
                est_total = n_hist * self.M
                n_calls = math.ceil(est_total / mega_batch_size)
                print(
                    f"  Oracle calls: ~{n_calls:,}  "
                    f"(mega-batch, chunk={mega_batch_size}, total~{est_total:,})"
                )
            elif batched:
                print(f"  Oracle calls: ~{n_hist:,}  (batched, M={self.M} per call)")
            else:
                print(f"  Oracle calls: ~{n_hist * self.M:,}")
            print(
                f"  Prefix: b* ({prefix_len} steps, "
                f"delta_pred={self.delta_pred:.4f})"
            )

        # ── Mega-batch mode: one (chunked) oracle call across all histories ────
        if mega_batch:
            all_windows: list[np.ndarray] = []
            hist_slices: dict = {}  # h_idx -> (start, end) index into all_windows

            if verbose:
                print("  Phase 1/2 — collecting windows across all histories …")

            for h_idx, pi_h in pi_star.items():
                if pi_h == 0.0:
                    continue
                windows = [self.pool.sample(h_idx, self.rng) for _ in range(self.M)]
                windows = [w for w in windows if w is not None]
                if not windows:
                    continue
                start = len(all_windows)
                if prefix_len > 0:
                    all_windows.extend(
                        np.vstack([self.b_star, w[-self.K :]]) for w in windows
                    )
                else:
                    all_windows.extend(windows)
                hist_slices[h_idx] = (start, len(all_windows))

            if all_windows:
                mega = np.stack(all_windows)  # (total, W, D)
                total = len(mega)

                if verbose:
                    print(
                        f"  Phase 2/2 — oracle forward pass on {total:,} windows "
                        f"in chunks of {mega_batch_size} …"
                    )

                all_outputs: list[np.ndarray] = []
                for chunk_start in tqdm(
                    range(0, total, mega_batch_size),
                    desc="GPU chunks",
                    disable=not verbose,
                ):
                    chunk = mega[chunk_start : chunk_start + mega_batch_size]
                    out = np.array(f(chunk), dtype=float)
                    if out.ndim == 2:
                        out = out[:, None, :]  # (B, D) -> (B, 1, D)
                    all_outputs.append(out)

                all_outputs_np = np.concatenate(all_outputs, axis=0)  # (total, L, D)
                total_out, L, D_out = all_outputs_np.shape
                self._set_horizon(L)

                s_all = self.disc.transform(all_outputs_np.reshape(-1, D_out)).reshape(
                    total_out, L, D_out
                )

                for h_idx, (start, end) in hist_slices.items():
                    s_h = s_all[start:end]  # (m, L, D)
                    counts = np.zeros((L, D, N), dtype=int)
                    for step in range(L):
                        for d in range(D):
                            col = s_h[:, step, d].astype(int).flatten()
                            counts[step, d] += np.bincount(col, minlength=N)
                    self._counts[h_idx] = counts
                    self._n_draws[h_idx] = end - start
                    self._samples[h_idx] = s_h[:, :, :D].astype(np.int8)

            # Fill empty counts for histories with no pool samples
            L_fill = self._horizon or 1
            for h_idx, pi_h in pi_star.items():
                if h_idx not in self._counts:
                    self._counts[h_idx] = np.zeros((L_fill, D, N), dtype=int)
                    self._n_draws[h_idx] = 0
            return

        # ── Standard per-history loop ─────────────────────────────────────────
        for _, (h_idx, pi_h) in tqdm(
            enumerate(pi_star.items()), total=len(pi_star), desc="Estimating"
        ):
            if pi_h == 0.0:
                continue

            if batched:
                windows = [self.pool.sample(h_idx, self.rng) for _ in range(self.M)]
                windows = [w for w in windows if w is not None]

                if windows:
                    if prefix_len > 0:
                        batch = np.stack(
                            [np.vstack([self.b_star, w[-self.K :]]) for w in windows]
                        )
                    else:
                        batch = np.stack(windows)
                    outputs = f(batch)  # (M, D) or (M, L, D)

                    # Normalize to (M, L, D)
                    outputs = np.array(outputs, dtype=float)
                    if outputs.ndim == 2:
                        outputs = outputs[:, None, :]  # (M, D) -> (M, 1, D)
                    elif outputs.ndim == 3 and outputs.shape[2] == 1:
                        outputs = outputs[:, :, 0][:, :, None]
                    M_actual, L, D_out = outputs.shape

                    self._set_horizon(L)
                    counts = np.zeros((L, D, N), dtype=int)

                    s_primes = self.disc.transform(outputs.reshape(-1, D_out)).reshape(
                        M_actual, L, D_out
                    )

                    for step in range(L):
                        for d in range(D):
                            col = s_primes[:, step, d].astype(int).flatten()
                            counts[step, d] += np.bincount(col, minlength=N)
                    n_drawn = len(windows)
                    self._samples[h_idx] = s_primes[:, :, :D].astype(np.int8)
                else:
                    L = self._horizon or 1
                    counts = np.zeros((L, D, N), dtype=int)
                    n_drawn = 0

            else:
                n_drawn = 0
                L = self._horizon or 1
                counts = np.zeros((L, D, N), dtype=int)
                sample_list: list = []

                for _ in range(self.M):
                    window = self.pool.sample(h_idx, self.rng)
                    if window is None:
                        continue
                    if prefix_len > 0:
                        tilde_h = np.vstack([self.b_star, window[-self.K :]])
                    else:
                        tilde_h = window

                    output = np.array(f(tilde_h), dtype=float)
                    output_norm = _normalize_output(output)  # (L, D)
                    L_out = output_norm.shape[0]
                    self._set_horizon(L_out)

                    if L_out != L:
                        L = L_out
                        counts = np.zeros((L, D, N), dtype=int)

                    s_prime = self.disc.transform(output_norm)  # (L, D)
                    for step in range(L):
                        counts[step, np.arange(D), s_prime[step].astype(int)] += 1
                    sample_list.append(s_prime)
                    n_drawn += 1

                if sample_list:
                    self._samples[h_idx] = np.stack(sample_list).astype(np.int8)

            self._counts[h_idx] = counts
            self._n_draws[h_idx] = n_drawn

    def _set_horizon(self, L: int) -> None:
        if self._horizon is None:
            self._horizon = L
        elif self._horizon != L:
            raise ValueError(
                f"Inconsistent forecast horizon: expected {self._horizon}, got {L}"
            )

    def predict_marginal(self, d: int, h_idx: int, step: int = 0) -> np.ndarray:
        """
        T^{f,d,l}_{K*}(. | h) from MC counts with Jeffreys smoothing.
        Returns (N,) array. Falls back to uniform for unestimated histories.
        Default step=0 preserves single-step behaviour.
        """
        N = self.disc.N
        alpha = 0.5
        if h_idx not in self._counts:
            return np.full(N, 1.0 / N)
        counts = self._counts[h_idx][step, d].astype(float)
        return (counts + alpha) / (self._n_draws.get(h_idx, self.M) + N * alpha)

    def predict(self, h_idx: int) -> np.ndarray:
        """Returns (L, D, N) array of per-step marginal distributions."""
        L = self.horizon
        D = self.disc.D_
        N = self.disc.N
        alpha = 0.5
        if h_idx not in self._counts:
            return np.full((L, D, N), 1.0 / N)
        c = self._counts[h_idx].astype(np.float32)  # (L, D, N)
        n_draws = self._n_draws.get(h_idx, self.M)
        return (c + alpha) / (n_draws + N * alpha)

    def predict_batch(self, h_idxs: list) -> np.ndarray:
        """Returns (n, L, D, N) kernels for a list of h_idx values.
        Unvisited histories fall back to uniform."""
        L, D, N = self.horizon, self.disc.D_, self.disc.N
        alpha = 0.5
        n = len(h_idxs)
        out = np.full((n, L, D, N), 1.0 / N, dtype=np.float32)
        for i, h in enumerate(h_idxs):
            if h in self._counts:
                c = self._counts[h].astype(np.float32)
                n_draws = self._n_draws.get(h, self.M)
                out[i] = (c + alpha) / (n_draws + N * alpha)
        return out

    def predict_path_marginal(
        self, d: int, h_idx: int, tau: Optional[list] = None
    ) -> np.ndarray:
        """
        T^{f,M_d,tau}_{K*}(. | h) — true joint trajectory of variable d.

        Estimates the path kernel along M_d (the constant path that selects
        dimension d at every step) by counting co-occurrences of the full
        length-|tau| trajectory across the M MC draws.  This captures
        cross-step correlations in the oracle output that the factored
        product-of-marginals approximation misses.

        Returns (N^|tau|,) probability simplex with Jeffreys smoothing.
        Falls back to uniform for unestimated histories.
        """
        L = self.horizon
        N = self.disc.N
        if tau is None:
            tau = list(range(L))
        q = len(tau)
        Nq = N ** q
        alpha = 0.5

        if h_idx not in self._samples:
            return np.full(Nq, 1.0 / Nq)

        samp = self._samples[h_idx]        # (M, L, D) int8
        traj = samp[:, tau, d].astype(int) # (M, q)
        powers = N ** np.arange(q - 1, -1, -1)
        flat_idx = (traj * powers[None, :]).sum(axis=1)  # (M,)
        counts = np.bincount(flat_idx, minlength=Nq).astype(float)
        return (counts + alpha) / (len(samp) + Nq * alpha)

    def noise_floor_mc(self, h_idx: int) -> float:
        """sqrt(pi/2M) — MC variance component."""
        M = self._n_draws.get(h_idx, 0)
        return float("inf") if M == 0 else np.sqrt(math.pi / (2.0 * M))

    def noise_floor_within_bin(self, h_idx: int) -> float:
        """1/sqrt(pool_size) — within-bin variation component."""
        sz = self.pool.pool_size(h_idx)
        return float("inf") if sz == 0 else np.sqrt(math.pi / (2.0 * sz))

    def noise_floor(self, h_idx: int) -> float:
        """Sum of all three noise floor components."""
        return (
            self.noise_floor_mc(h_idx)
            + self.delta_pred
            + self.noise_floor_within_bin(h_idx)
        )

    def keri(self, pi_star: dict, lam: float, kappa: float = 2) -> float:
        """Fraction of pi*-weighted histories with total floor < lam/2."""
        keri = 0
        for h_idx, pi_h in pi_star.items():
            keri += pi_h * max(0, 1 - self.noise_floor(h_idx) / (kappa * (lam / 4)))
        return keri

    @property
    def estimated_histories(self) -> list:
        return list(self._counts.keys())

    def n_f_h(self, h_idx: int) -> int:
        return self._counts.get(h_idx, 0)


def _decode_history(h_idx: int, K: int, D: int, N: int) -> np.ndarray:
    """Flat history index -> (K, D) int array."""
    h_arr = np.zeros((K, D), dtype=int)
    idx = h_idx
    for k in reversed(range(K)):
        for d in reversed(range(D)):
            h_arr[k, d] = idx % N
            idx //= N
    return h_arr


class TreeNode:
    """
    A node in the regression tree over the history space H+_{K*}.

    Each internal node stores a split condition (variable d', lag k, bin n)
    and pointers to left/right children.
    Each leaf stores a pooled Dirichlet kernel estimate of shape (L, D, N),
    one probability simplex per (forecast step, target variable).

    Parameters
    ----------
    histories : list[int]
        h_idx values assigned to this node.
    depth : int
        Depth of this node in the tree (root = 0).
    """

    def __init__(self, histories: list, depth: int = 0):
        self.histories = histories
        self.depth = depth

        self.split_var: Optional[int] = None
        self.split_lag: Optional[int] = None
        self.split_bin: Optional[int] = None

        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None

        # (L, D, N) float array; set for leaves only
        self.leaf_kernel: Optional[np.ndarray] = None
        self.pooled_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class TreeKernelEstimator:
    """
    Estimates D factored marginal kernels T^{f,d,l}_{K*}(s'^d | h) via a
    regression tree that partitions H+_{K*} into leaves and pools counts.

    Supports multi-step forecasts: counts are (L, D, N) per history, leaf
    kernels are (L, D, N), and the split criterion averages KL cost over
    all L forecast steps and D target variables.

    Parameters
    ----------
    disc          : Discretiser
    K             : int — Markov order K*.
    n_pool        : int — Minimum pooled count required per leaf.
    max_depth     : int — Maximum tree depth.
    min_cost_reduction : float — Stop splitting if KL gain < this.
    alpha         : float — Dirichlet prior (default 0.5 = Jeffreys).
    """

    def __init__(
        self,
        disc: Discretiser,
        K: int,
        W: int,
        M: int,
        pool: SuffixPool,
        b_star: np.ndarray,
        n_pool: int = 10,
        max_depth: int = 10,
        min_cost_reduction: float = 1e-4,
        alpha: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        device: str = "auto",
    ):
        self.disc = disc
        self.K = K
        self.W = W
        self.M = M
        self.n_pool = n_pool
        self.max_depth = max_depth
        self.min_cost_reduction = min_cost_reduction
        self.alpha = alpha
        self.b_star = b_star
        self.pool = pool

        self._root: Optional[TreeNode] = None
        self.global_fallback: bool = False
        self.rng = rng if rng is not None else np.random.default_rng()

        self.n_leaves: int = 0
        self.split_variables: list = []
        self._counts: dict = {}
        self._samples: dict = {}  # h_idx -> (M, L, D) int8 array of raw MC draws
        self._horizon: Optional[int] = None

        # GPU device for the split-search KL tensor (auto-detects CUDA)
        if device == "auto":
            try:
                import torch

                self._torch_device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            except ImportError:
                self._torch_device = None
        elif device == "cpu":
            self._torch_device = None
        else:
            import torch

            self._torch_device = torch.device(device)

    @property
    def horizon(self) -> int:
        return self._horizon if self._horizon is not None else 1

    def _set_horizon(self, L: int) -> None:
        if self._horizon is None:
            self._horizon = L
        elif self._horizon != L:
            raise ValueError(
                f"Inconsistent forecast horizon: expected {self._horizon}, got {L}"
            )

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence D_KL(p || q) for two 1-D distributions p, q."""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum == 0 or q_sum == 0:
            return float("inf")
        p_norm = p / p_sum
        q_norm = q / q_sum
        mask = (p_norm > 0) & (q_norm > 0)
        return np.sum(p_norm[mask] * np.log(p_norm[mask] / q_norm[mask]))

    def _kl_batch(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Mean KL divergence D_KL(p || q) over a (L, D, N) pair of kernels.

        Operates on the full (L, D, N) arrays at once — no Python loop over
        (l, d) pairs — so it replaces the list-comprehension in _split_cost
        and _node_cost.
        """
        # p, q: (L, D, N)
        row_sums_p = p.sum(axis=-1, keepdims=True)  # (L, D, 1)
        row_sums_q = q.sum(axis=-1, keepdims=True)
        # Treat zero-sum rows as degenerate — contribute 0 to the mean
        valid = (row_sums_p[..., 0] > 0) & (row_sums_q[..., 0] > 0)  # (L, D)
        if not valid.any():
            return 0.0
        p_n = np.where(row_sums_p > 0, p / row_sums_p, 0.0)
        q_n = np.where(row_sums_q > 0, q / row_sums_q, 1.0 / q.shape[-1])
        mask = (p_n > 0) & (q_n > 0)
        kl_ld = np.where(mask, p_n * np.log(np.where(mask, p_n / q_n, 1.0)), 0.0)
        kl_per_ld = kl_ld.sum(axis=-1)  # (L, D)
        return float(kl_per_ld[valid].mean())

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        f: Callable,
        pi_star: dict,
        verbose: bool = True,
        batched: bool = True,
        mega_batch: bool = False,
        mega_batch_size: int = 4096,
    ) -> "TreeKernelEstimator":
        """
        Build the regression tree from per-history count arrays.

        Parameters
        ----------
        f               : oracle Callable — returns (D,) or (L, D) for single
                          window, (M, D) or (M, L, D) for batched input.
        pi_star         : dict h_idx -> float
        verbose         : print progress
        batched         : if True, batch M windows per history into one f call
                          (H oracle calls total).
        mega_batch      : if True, collect windows from ALL histories into one
                          array and issue at most ceil(H*M / mega_batch_size)
                          oracle calls total.  Maximises GPU throughput at the
                          cost of peak RAM proportional to H*M*W*D.
                          Overrides `batched`.
        mega_batch_size : chunk size for mega-batch oracle calls.
        """
        D = self.disc.D_
        N = self.disc.N
        prefix_len = self.W - self.K
        n_hist = len(pi_star)

        if verbose:
            print(f"  TreeKernelEstimator: |H+_K|={n_hist}, M={self.M}")
            if mega_batch:
                est_total = n_hist * self.M
                n_calls = math.ceil(est_total / mega_batch_size)
                print(
                    f"  Oracle calls: ~{n_calls:,}  "
                    f"(mega-batch, chunk={mega_batch_size}, total~{est_total:,})"
                )
            elif batched:
                print(f"  Oracle calls: ~{n_hist:,}  (batched, M={self.M} per call)")
            else:
                print(f"  Oracle calls: ~{n_hist * self.M:,}")

        # ── Mega-batch mode ────────────────────────────────────────────────────
        if mega_batch:
            all_windows: list[np.ndarray] = []
            hist_slices: dict = {}  # h_idx -> (start, end)

            if verbose:
                print("  Phase 1/2 — collecting windows across all histories …")

            for h_idx, pi_h in pi_star.items():
                if pi_h == 0.0:
                    continue
                windows = [self.pool.sample(h_idx, self.rng) for _ in range(self.M)]
                windows = [w for w in windows if w is not None]
                if not windows:
                    continue
                start = len(all_windows)
                if prefix_len > 0:
                    all_windows.extend(
                        np.vstack([self.b_star, w[-self.K :]]) for w in windows
                    )
                else:
                    all_windows.extend(windows)
                hist_slices[h_idx] = (start, len(all_windows))

            if all_windows:
                mega = np.stack(all_windows)  # (total, W, D)
                total = len(mega)

                if verbose:
                    print(
                        f"  Phase 2/2 — oracle forward pass on {total:,} windows "
                        f"in chunks of {mega_batch_size} …"
                    )

                all_outputs: list[np.ndarray] = []
                for chunk_start in tqdm(
                    range(0, total, mega_batch_size),
                    desc="GPU chunks",
                    disable=not verbose,
                ):
                    chunk = mega[chunk_start : chunk_start + mega_batch_size]
                    out = np.array(f(chunk), dtype=float)
                    if out.ndim == 2:
                        out = out[:, None, :]  # (B, D) -> (B, 1, D)
                    all_outputs.append(out)

                all_out = np.concatenate(all_outputs, axis=0)  # (total, L, D)
                total_out, L, D_out = all_out.shape
                self._set_horizon(L)

                s_all = self.disc.transform(all_out.reshape(-1, D_out)).reshape(
                    total_out, L, D_out
                )

                for h_idx, (start, end) in hist_slices.items():
                    s_h = s_all[start:end]  # (m, L, D)
                    counts = np.zeros((L, D, N), dtype=int)
                    for step in range(L):
                        for d in range(D):
                            col = s_h[:, step, d].astype(int).flatten()
                            counts[step, d] += np.bincount(col, minlength=N)
                    self._counts[h_idx] = counts
                    self._samples[h_idx] = s_h[:, :, :D].astype(np.int8)

            # Fill empty slots for histories with no pool samples
            L_fill = self._horizon or 1
            for h_idx, pi_h in pi_star.items():
                if h_idx not in self._counts:
                    self._counts[h_idx] = np.zeros((L_fill, D, N), dtype=int)

        else:
            # ── Standard per-history loop ──────────────────────────────────────
            for _, (h_idx, pi_h) in tqdm(
                enumerate(pi_star.items()), total=len(pi_star), desc="Estimating"
            ):
                if pi_h == 0.0:
                    continue

                if batched:
                    windows = [self.pool.sample(h_idx, self.rng) for _ in range(self.M)]
                    windows = [w for w in windows if w is not None]

                    if windows:
                        if prefix_len > 0:
                            batch = np.stack(
                                [
                                    np.vstack([self.b_star, w[-self.K :]])
                                    for w in windows
                                ]
                            )
                        else:
                            batch = np.stack(windows)

                        outputs = np.array(f(batch), dtype=float)
                        if outputs.ndim == 2:
                            outputs = outputs[:, None, :]  # (M, D) -> (M, 1, D)
                        elif outputs.ndim == 3 and outputs.shape[2] == 1:
                            outputs = outputs[:, :, 0][:, :, None]

                        M_actual, L, D_out = outputs.shape
                        self._set_horizon(L)
                        counts = np.zeros((L, D, N), dtype=int)

                        s_primes = self.disc.transform(
                            outputs.reshape(-1, D_out)
                        ).reshape(M_actual, L, D_out)

                        for step in range(L):
                            for d in range(D):
                                col = s_primes[:, step, d].astype(int).flatten()
                                counts[step, d] += np.bincount(col, minlength=N)
                        self._samples[h_idx] = s_primes[:, :, :D].astype(np.int8)
                    else:
                        L = self._horizon or 1
                        counts = np.zeros((L, D, N), dtype=int)

                else:
                    L = self._horizon or 1
                    counts = np.zeros((L, D, N), dtype=int)
                    sample_list: list = []

                    for _ in range(self.M):
                        window = self.pool.sample(h_idx, self.rng)
                        if window is None:
                            continue
                        if prefix_len > 0:
                            tilde_h = np.vstack([self.b_star, window[-self.K :]])
                        else:
                            tilde_h = window

                        output = np.array(f(tilde_h), dtype=float)
                        output_norm = _normalize_output(output)  # (L, D)
                        L_out = output_norm.shape[0]
                        self._set_horizon(L_out)
                        if L_out != L:
                            L = L_out
                            counts = np.zeros((L, D, N), dtype=int)

                        s_prime = self.disc.transform(output_norm)  # (L, D)
                        for step in range(L):
                            counts[step, np.arange(D), s_prime[step].astype(int)] += 1
                        sample_list.append(s_prime)

                    if sample_list:
                        self._samples[h_idx] = np.stack(sample_list).astype(np.int8)

                self._counts[h_idx] = counts

        # Finalise horizon
        L = self._horizon or 1
        self._L = L
        self._D = D
        self._N = N

        # Separate covered from sparse
        h_cov = [
            h
            for h in list(pi_star.keys())
            if self._pooled_count(h, self._counts) >= self.n_pool
        ]
        h_sparse = [h for h in list(pi_star.keys()) if h not in set(h_cov)]

        if len(h_cov) == 0:
            self.global_fallback = True
            global_counts = np.zeros((L, D, N), dtype=int)
            for h in list(pi_star.keys()):
                if h in self._counts:
                    global_counts += self._counts[h]
            root = TreeNode(histories=list(pi_star.keys()), depth=0)
            root.leaf_kernel = self._compute_leaf_kernel(global_counts)
            root.pooled_count = int(global_counts.sum()) // max(D * L, 1)
            self._root = root
            self.n_leaves = 1
            return self

        if verbose:
            print(
                f"  TreeKernelEstimator: |H^cov|={len(h_cov)}, "
                f"|H^sparse|={len(h_sparse)}"
            )
        self._pi_star = pi_star

        self._root = self._fit_node(histories=h_cov, depth=0)

        self.n_leaves = self._count_leaves(self._root)
        self.split_variables = list(
            {
                (node.split_var, node.split_lag)
                for node in self._iter_internal_nodes(self._root)
            }
        )

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_marginal(self, d: int, h_idx: int, step: int = 0) -> np.ndarray:
        """
        Return T^{f,d,l}_{K*}(. | h) for variable d at forecast step.

        Default step=0 preserves single-step behaviour.
        Returns (N,) probability simplex.
        """
        if self._root is None:
            raise RuntimeError("Call fit() before predict_marginal().")
        leaf = self._route(h_idx)
        return leaf.leaf_kernel[step, d]

    def predict(self, h_idx: int) -> np.ndarray:
        """
        Return (L, D, N) array of per-step marginal distributions.
        """
        if self._root is None:
            raise RuntimeError("Call fit() before predict().")
        leaf = self._route(h_idx)
        return leaf.leaf_kernel  # (L, D, N)

    def predict_batch(self, h_idxs: list) -> np.ndarray:
        """Returns (n, L, D, N) kernels by routing each h through the tree."""
        if self._root is None:
            raise RuntimeError("Call fit() before predict_batch().")
        L, D, N = self._L, self._D, self._N
        n = len(h_idxs)
        out = np.empty((n, L, D, N), dtype=np.float32)
        for i, h in enumerate(h_idxs):
            out[i] = self._route(h).leaf_kernel
        return out

    def predict_path_marginal(
        self, d: int, h_idx: int, tau: Optional[list] = None
    ) -> np.ndarray:
        """
        T^{f,M_d,tau}_{K*}(. | h) — true joint trajectory of variable d.

        Pools raw MC samples from all histories routed to the same leaf as h,
        then counts co-occurrences of the length-|tau| trajectory of variable d.
        This mirrors how TKE pools counts for the leaf kernel, capturing
        cross-step correlations that the product-of-marginals misses.

        Returns (N^|tau|,) probability simplex with Jeffreys smoothing.
        """
        if self._root is None:
            raise RuntimeError("Call fit() before predict_path_marginal().")
        N = self._N
        L = self._L
        if tau is None:
            tau = list(range(L))
        q = len(tau)
        Nq = N ** q
        alpha = self.alpha

        leaf = self._route(h_idx)
        parts = [self._samples[h] for h in leaf.histories if h in self._samples]
        if not parts:
            return np.full(Nq, 1.0 / Nq)

        samp = np.concatenate(parts, axis=0)   # (total, L, D) int8
        traj = samp[:, tau, d].astype(int)     # (total, q)
        powers = N ** np.arange(q - 1, -1, -1)
        flat_idx = (traj * powers[None, :]).sum(axis=1)
        counts = np.bincount(flat_idx, minlength=Nq).astype(float)
        return (counts + alpha) / (len(samp) + Nq * alpha)

    def noise_floor(self, h_idx: int) -> float:
        """1/sqrt(P_ell) + delta_ell."""
        if self._root is None:
            return float("inf")
        leaf = self._route(h_idx)
        if leaf.pooled_count == 0:
            return float("inf")
        return 1.0 / np.sqrt(leaf.pooled_count) + self._within_leaf_bias(leaf)

    def keri(self, pi_star: dict, lam: float, kappa: float = 2) -> float:
        """Fraction of pi*-weighted histories with total floor < lam/2."""
        keri = 0
        for h_idx, pi_h in pi_star.items():
            keri += pi_h * max(0, 1 - self.noise_floor(h_idx) / (kappa * (lam / 4)))
        return keri

    def n_f_h(self, h_idx: int) -> int:
        return self._counts.get(h_idx, 0)

    def _fit_node(self, histories: list, depth: int) -> TreeNode:
        node = TreeNode(histories=histories, depth=depth)

        node_counts = self._aggregate_counts(histories)  # (L, D, N)
        node.pooled_count = int(node_counts.sum()) // max(self._D * self._L, 1)

        if len(histories) == 1 or depth >= self.max_depth:
            node.leaf_kernel = self._compute_leaf_kernel(node_counts)
            return node

        node_kernel = self._compute_leaf_kernel(node_counts)
        current_cost = self._node_cost(histories, node_kernel)

        # ── Vectorised split search over all C = D×K×N candidates ────────────
        H = len(histories)
        K, D, N, L = self.K, self._D, self._N, self._L

        # (H, K, D) discrete feature matrix — lag × variable bin for each history
        F = np.stack([_decode_history(h, K, D, N) for h in histories])

        # (H, L, D, N) MC count arrays
        zeros_ldn = np.zeros((L, D, N), dtype=int)
        counts_arr = np.stack([self._counts.get(h, zeros_ldn) for h in histories])

        # per-history pooled sample count (H,) — used for n_pool filter
        per_h_pool = (counts_arr.sum(axis=(1, 2, 3)) // max(D * L, 1)).astype(np.int64)
        total_pool = int(per_h_pool.sum())

        # (H, L, D, N) individual Dirichlet kernels
        hc_f = counts_arr.astype(float)
        hc_sums = hc_f.sum(axis=-1, keepdims=True)  # (H, L, D, 1)
        h_kernels = (hc_f + self.alpha) / (hc_sums + N * self.alpha)

        # pi* weights (H,)
        weights = np.array([self._pi_star.get(h, 0.0) for h in histories])

        # Build (C, H) left-child boolean mask for all C = D*K*N candidates
        # masks_full[d, k, n, h] = (F[h, k, d] == n)
        F_dkh = F.transpose(2, 1, 0)  # (D, K, H)
        n_vals = np.arange(N)
        masks_full = F_dkh[:, :, None, :] == n_vals[None, None, :, None]  # (D,K,N,H)
        masks_flat = masks_full.reshape(-1, H)  # (C, H) bool

        # Filter: non-trivial splits with enough pooled samples on each side
        left_pool = masks_flat.astype(np.int64) @ per_h_pool  # (C,)
        right_pool = total_pool - left_pool
        valid = (
            masks_flat.any(axis=1)
            & (~masks_flat).any(axis=1)
            & (left_pool >= self.n_pool)
            & (right_pool >= self.n_pool)
        )
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            node.leaf_kernel = self._compute_leaf_kernel(node_counts)
            return node

        # Aggregate counts for each valid split's two children via matmul
        mv = masks_flat[valid_idx].astype(float)  # (C_v, H)
        counts_flat = counts_arr.reshape(H, -1).astype(float)  # (H, L*D*N)
        C_v = len(valid_idx)

        left_counts = (mv @ counts_flat).reshape(C_v, L, D, N)
        right_counts = ((1.0 - mv) @ counts_flat).reshape(C_v, L, D, N)

        # Dirichlet kernels for each candidate child
        def _batch_kernel(c_arr: np.ndarray) -> np.ndarray:
            sums = c_arr.sum(axis=-1, keepdims=True)
            return (c_arr + self.alpha) / (sums + N * self.alpha)

        left_k = _batch_kernel(left_counts)  # (C_v, L, D, N)
        right_k = _batch_kernel(right_counts)

        # KL costs: dispatch to GPU if available, else numpy with chunking.
        # CHUNK is sized so that the (ch, H, L, D, N) broadcast stays under
        # ~400 MB — critical for large-D datasets (e.g. web_traffic D=145).
        _elems_per_cand = H * L * D * N
        CHUNK = max(1, min(128, 100_000_000 // _elems_per_cand))
        split_costs = np.empty(C_v, dtype=float)

        if self._torch_device is not None and self._torch_device.type == "cuda":
            import torch

            dev = self._torch_device
            lk_t = torch.from_numpy(left_k).float().to(dev)  # (C_v, L, D, N)
            rk_t = torch.from_numpy(right_k).float().to(dev)
            hk_t = torch.from_numpy(h_kernels).float().to(dev)  # (H, L, D, N)
            w_t = torch.from_numpy(weights).float().to(dev)  # (H,)
            mv_t = torch.from_numpy(mv).float().to(dev)  # (C_v, H)
            eps = 1e-30

            for cs in range(0, C_v, CHUNK):
                ce = min(cs + CHUNK, C_v)
                mv_e = mv_t[cs:ce, :, None, None, None]  # (ch, H, 1, 1, 1)
                child_k = (
                    mv_e * lk_t[cs:ce, None] + (1.0 - mv_e) * rk_t[cs:ce, None]
                )  # (ch, H, L, D, N)
                hk_e = hk_t[None]  # (1, H, L, D, N)
                q = torch.clamp(child_k, min=eps)
                ratio = torch.clamp(hk_e / q, min=eps)
                kl_terms = torch.where(
                    hk_e > 0, hk_e * torch.log(ratio), torch.zeros_like(hk_e)
                )
                kl_hc = kl_terms.sum(dim=-1).mean(dim=(-2, -1))  # (ch, H)
                split_costs[cs:ce] = (kl_hc * w_t[None]).sum(dim=-1).cpu().numpy()
        else:
            for cs in range(0, C_v, CHUNK):
                ce = min(cs + CHUNK, C_v)
                mv_e = mv[cs:ce, :, None, None, None]  # (ch, H, 1, 1, 1)
                child_k = (
                    mv_e * left_k[cs:ce, None] + (1.0 - mv_e) * right_k[cs:ce, None]
                )  # (ch, H, L, D, N)
                hk_e = h_kernels[None]  # (1, H, L, D, N)
                safe_q = np.maximum(child_k, 1e-30)
                ratio = np.where(hk_e > 0, hk_e / safe_q, 1.0)
                kl_terms = np.where(hk_e > 0, hk_e * np.log(ratio), 0.0)
                kl_hc = kl_terms.sum(axis=-1).mean(axis=(-2, -1))  # (ch, H)
                split_costs[cs:ce] = (kl_hc * weights[None]).sum(axis=-1)

        best_vi = int(np.argmin(split_costs))
        best_cost = float(split_costs[best_vi])

        if (current_cost - best_cost) < self.min_cost_reduction:
            node.leaf_kernel = self._compute_leaf_kernel(node_counts)
            return node

        # Decode best split index → (d, k, n)
        c_best = valid_idx[best_vi]
        d_prime = int(c_best // (K * N))
        k_best = int((c_best % (K * N)) // N)
        n_best = int(c_best % N)

        left_mask = masks_flat[c_best]
        left_h = [histories[i] for i in range(H) if left_mask[i]]
        right_h = [histories[i] for i in range(H) if not left_mask[i]]

        node.split_var = d_prime
        node.split_lag = k_best
        node.split_bin = n_best
        node.left = self._fit_node(left_h, depth + 1)
        node.right = self._fit_node(right_h, depth + 1)

        return node

    def _route(self, h_idx: int) -> TreeNode:
        node = self._root
        h_disc = _decode_history(h_idx, self.K, self.disc.D_, self.disc.N)

        while not node.is_leaf:
            if h_disc[node.split_lag, node.split_var] == node.split_bin:
                node = node.left
            else:
                node = node.right

        return node

    def _split_cost(self, left_h: list, right_h: list) -> float:
        """
        pi*-weighted KL divergence of each history's kernel from its child's
        pooled kernel, averaged over L forecast steps and D target variables.
        """
        cost = 0.0
        for child_h in [left_h, right_h]:
            child_counts = self._aggregate_counts(child_h)
            child_kernel = self._compute_leaf_kernel(child_counts)  # (L, D, N)
            for h in child_h:
                w = self._pi_star.get(h, 0.0)
                if w == 0.0 or h not in self._counts:
                    continue
                h_kernel = self._individual_kernel(h)  # (L, D, N)
                cost += w * self._kl_batch(h_kernel, child_kernel)
        return cost

    def _node_cost(self, histories: list, pooled_kernel: np.ndarray) -> float:
        """KL cost of current node before splitting, averaged over steps and vars."""
        cost = 0.0
        for h in histories:
            w = self._pi_star.get(h, 0.0)
            if w == 0.0 or h not in self._counts:
                continue
            h_kernel = self._individual_kernel(h)
            cost += w * self._kl_batch(h_kernel, pooled_kernel)
        return cost

    def _apply_split(self, histories: list, d_prime: int, k: int, n: int) -> tuple:
        left, right = [], []
        for h in histories:
            h_disc = _decode_history(h, self.K, self.disc.D_, self.disc.N)
            if h_disc[k, d_prime] == n:
                left.append(h)
            else:
                right.append(h)
        return left, right

    def _aggregate_counts(self, histories: list) -> np.ndarray:
        """Sum (L, D, N) count arrays over a list of h_idx values."""
        L, D, N = self._L, self._D, self._N
        total = np.zeros((L, D, N), dtype=int)
        for h in histories:
            if h in self._counts:
                total += self._counts[h]
        return total

    def _compute_leaf_kernel(self, counts: np.ndarray) -> np.ndarray:
        """
        Dirichlet posterior mean from pooled (L, D, N) counts.
        Returns (L, D, N) float array.
        """
        c = counts.astype(float)
        N = c.shape[-1]
        row_sums = c.sum(axis=-1, keepdims=True)  # (L, D, 1)
        return (c + self.alpha) / (row_sums + N * self.alpha)

    def _individual_kernel(self, h_idx: int) -> np.ndarray:
        """Dirichlet posterior mean for a single history — (L, D, N)."""
        return self._compute_leaf_kernel(self._counts[h_idx])

    def _pooled_count(self, h_idx: int, counts: dict) -> int:
        if h_idx not in counts:
            return 0
        L = max(self._L if hasattr(self, "_L") else 1, 1)
        D = max(self._D if hasattr(self, "_D") else 1, 1)
        return int(counts[h_idx].sum()) // max(D * L, 1)

    def _pooled_count_list(self, histories: list) -> int:
        total = 0
        for h in histories:
            if h in self._counts:
                total += int(self._counts[h].sum()) // max(self._D * self._L, 1)
        return total

    def _within_leaf_bias(self, leaf: TreeNode) -> float:
        """
        delta_ell = max_{h in leaf} avg TV over L steps and D vars.
        """
        if leaf.leaf_kernel is None:
            return 0.0
        max_tv = 0.0
        for h in leaf.histories:
            if h not in self._counts:
                continue
            h_kernel = self._individual_kernel(h)  # (L, D, N)
            tv = float(0.5 * np.abs(h_kernel - leaf.leaf_kernel).sum(axis=-1).mean())
            max_tv = max(max_tv, tv)
        return max_tv

    def _count_leaves(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _iter_internal_nodes(self, node: TreeNode):
        if not node.is_leaf:
            yield node
            yield from self._iter_internal_nodes(node.left)
            yield from self._iter_internal_nodes(node.right)

    def print_tree(self, node: Optional[TreeNode] = None, indent: int = 0) -> None:
        if node is None:
            node = self._root
        if node is None:
            print("Tree not yet fit.")
            return

        prefix = "  " * indent
        if node.is_leaf:
            kernel_str = np.array2string(
                node.leaf_kernel[0, 0], precision=3, suppress_small=True
            )
            print(
                f"{prefix}LEAF  P_ell={node.pooled_count}  "
                f"kernel[l=0,d=0]={kernel_str}"
            )
        else:
            print(
                f"{prefix}SPLIT  X^{node.split_var}_{{t-{node.split_lag}}} "
                f"== {node.split_bin}"
            )
            print(f"{prefix}  LEFT  (== {node.split_bin}):")
            self.print_tree(node.left, indent + 2)
            print(f"{prefix}  RIGHT (!= {node.split_bin}):")
            self.print_tree(node.right, indent + 2)

    def leaf_summary(self) -> list:
        summaries = []
        for leaf in self._iter_leaves(self._root):
            delta = self._within_leaf_bias(leaf)
            pf = (
                1.0 / np.sqrt(leaf.pooled_count) + delta
                if leaf.pooled_count > 0
                else float("inf")
            )
            summaries.append(
                {
                    "n_histories": len(leaf.histories),
                    "pooled_count": leaf.pooled_count,
                    "delta_ell": round(delta, 4),
                    "noise_floor": round(pf, 4),
                    "kernel_l0_d0": (
                        leaf.leaf_kernel[0, 0].tolist()
                        if leaf.leaf_kernel is not None
                        else None
                    ),
                }
            )
        return summaries

    def _iter_leaves(self, node: TreeNode):
        if node.is_leaf:
            yield node
        else:
            yield from self._iter_leaves(node.left)
            yield from self._iter_leaves(node.right)

    @property
    def estimated_histories(self) -> list:
        return list(self._counts.keys())
