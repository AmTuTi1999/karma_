import numpy as np


class Discretiser:
    """
    Discretises a continuous D-dimensional MVTS into N bins per variable
    via equal-frequency quantisation on the training window.

    Parameters
    ----------
    N : int
        Number of bins per variable (recommended N in {3,4,5}).
    """

    def __init__(self, N: int = 4):
        self.N = N
        self.quantiles_ = None  # shape (D, N-1): bin boundaries q^d_1,...,q^d_{N-1}
        self.D_ = None

    def fit(self, X_train: np.ndarray) -> "Discretiser":
        """
        Estimate quantile boundaries from training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (T, D) or (n, W, D)
            Training MVTS. Each column is one variable X^d.
            If shape is 3D, windows are flattened along time axis.
        """
        if X_train.ndim == 3:
            # Convert windows to time-series (avoid redundant duplicates in overlapping windows
            # by taking the full first window plus one new frame per subsequent window)
            n, W, D = X_train.shape
            X_train = np.vstack([X_train[0]] + [X_train[i, -1:] for i in range(1, n)])
        elif X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D or 3D array, got ndim={X_train.ndim}")

        D = X_train.shape[-1]
        self.D_ = D
        # q^d_n = n/N quantile of X^d on training window
        self.quantiles_ = np.quantile(
            X_train,
            q=np.linspace(0, 1, self.N + 1)[1:-1],  # (N-1,) interior boundaries
            axis=0,  # (N-1, D)
        ).T  # -> (D, N-1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply discretisation map psi: R^D -> [N]^D.

        Parameters
        ----------
        X : np.ndarray, shape (T, D), (n, W, D), or (D,)

        Returns
        -------
        X_disc : np.ndarray of int, shape matching input
            Values in {0, 1, ..., N-1}.
        """
        scalar = X.ndim == 1
        if scalar:
            X = X[None, :]

        orig_shape = X.shape
        if X.ndim == 3:
            # Flatten windows to 2D for quantisation; restore original shape later.
            n, W, D = X.shape
            X_flat = X.reshape(-1, D)
            X_disc_flat = np.zeros_like(X_flat, dtype=int)
            for d in range(self.D_):
                X_disc_flat[:, d] = np.searchsorted(
                    self.quantiles_[d], X_flat[:, d], side="right"
                )
            X_disc = X_disc_flat.reshape(orig_shape)
            return X_disc

        X_disc = np.zeros(X.shape, dtype=int)
        for d in range(self.D_):
            X_disc[:, d] = np.searchsorted(self.quantiles_[d], X[:, d], side="right")

        if scalar:
            return X_disc[0]
        return X_disc

    def state_to_index(self, state: np.ndarray) -> int:
        """Convert joint state (D,) int array to flat integer index in [N^D]."""
        idx = 0
        for d in range(self.D_):
            idx = idx * self.N + int(state[d])
        return idx

    def index_to_state(self, idx: int) -> np.ndarray:
        """Convert flat index to joint state (D,) int array."""
        state = np.zeros(self.D_, dtype=int)
        for d in reversed(range(self.D_)):
            state[d] = idx % self.N
            idx //= self.N
        return state

    @property
    def n_states(self) -> int:
        return self.N**self.D_

    def history_to_index(self, history: np.ndarray) -> int:
        """
        Convert K-length history (K, D) int array to flat index in [N^{DK}].
        history[0] = oldest state, history[-1] = most recent state.
        """
        K, D = history.shape
        idx = 0
        for k in range(K):
            for d in range(D):
                idx = idx * self.N + int(history[k, d])
        return idx
