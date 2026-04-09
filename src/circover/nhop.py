"""
NHOP: Normalised Histogram Overlap Percentage
=============================================
Exact implementation of Definition 4.1 and Algorithm 1 from the thesis.

NHOP(X, Xs) = (1/k) * sum_j NHOP_j(X, Xs)
NHOP_j      = sum_b min(p_b^j, q_b^j)   (joint-normalised histograms)

Identity:  NHOP_j = 1 - TV(P^j, Q^j)    (Theorem 4.5)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


class NHOP(BaseEstimator):
    """
    Normalised Histogram Overlap Percentage.

    Parameters
    ----------
    n_bins : int, default=30
        Number of histogram bins B.

    Examples
    --------
    >>> import numpy as np
    >>> from circover import NHOP
    >>> rng = np.random.default_rng(42)
    >>> X  = rng.normal(0, 1, (200, 4))
    >>> Xs = rng.normal(0.3, 1, (150, 4))
    >>> nhop = NHOP(n_bins=30)
    >>> nhop.score(X, Xs)          # scalar in [0, 1]
    >>> nhop.score_per_feature(X, Xs)   # array of length 4
    """

    def __init__(self, n_bins: int = 30):
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, Xs: np.ndarray) -> float:
        """
        Overall NHOP score averaged across all features.

        Parameters
        ----------
        X  : array of shape (n, k) — original (reference) set
        Xs : array of shape (m, k) — synthetic (comparison) set

        Returns
        -------
        float in [0, 1]
        """
        return float(np.mean(self.score_per_feature(X, Xs)))

    def score_per_feature(self, X: np.ndarray, Xs: np.ndarray) -> np.ndarray:
        """
        Per-feature NHOP scores.

        Returns
        -------
        ndarray of shape (k,)  — NHOP_j for each dimension j
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        if X.ndim == 1:
            X = X[:, None]
        if Xs.ndim == 1:
            Xs = Xs[:, None]
        k = X.shape[1]
        return np.array([self._nhop_1d(X[:, j], Xs[:, j]) for j in range(k)])

    def tv_per_feature(self, X: np.ndarray, Xs: np.ndarray) -> np.ndarray:
        """
        Per-feature Total Variation distance.
        Uses the exact identity TV_j = 1 - NHOP_j (Theorem 4.5).
        """
        return 1.0 - self.score_per_feature(X, Xs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _nhop_1d(self, x: np.ndarray, xs: np.ndarray) -> float:
        """NHOP for a single dimension (Algorithm 1, inner loop)."""
        lo = min(x.min(), xs.min())
        hi = max(x.max(), xs.max())

        # Degenerate case: all values identical across both sets
        if lo == hi:
            return 1.0

        # Joint normalisation -> [0, 1]
        x_norm = (x - lo) / (hi - lo)
        xs_norm = (xs - lo) / (hi - lo)

        # Equal-width bins on [0, 1] (last bin closed on right)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        p = np.histogram(x_norm,  bins=edges)[0] / len(x)
        q = np.histogram(xs_norm, bins=edges)[0] / len(xs)

        return float(np.sum(np.minimum(p, q)))
