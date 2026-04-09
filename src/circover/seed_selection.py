"""
Geometry-Preserving Seed Selection
====================================
Implementation of Algorithm 2 (Chapter 6) of the thesis.

Composite score = NHOP + AGTP - w_jsd * JSD_bar - w_z * Z
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from .nhop import NHOP


class GeometricSeedSelector:
    """
    Geometry-preserving seed selection for oversampling.

    Selects a subset of minority instances whose marginal distributions,
    geometric structure, and local spacing best represent the full minority
    class (Section 6.4 composite criterion).

    Parameters
    ----------
    n_seeds : int
        Number of seeds to select.
    n_candidates : int, default=100
        Number of random candidate seed sets evaluated.
    n_clusters : int, default=5
        K-Means clusters for stratified sampling.
    n_pca : int or None, default=None
        PCA components for scoring. None = no PCA (native dimension).
    k_topo : int, default=5
        k for topological similarity (k-NN distances).
    n_bins : int, default=30
        Histogram bins for NHOP, JSD, T_sim.
    w_jsd : float, default=0.3
        Penalty weight for JSD term.
    w_z : float, default=0.5
        Penalty weight for smoothness Z term.
    random_state : int or None, default=None
    """

    def __init__(
        self,
        n_seeds: int,
        n_candidates: int = 100,
        n_clusters: int = 5,
        n_pca: int | None = None,
        k_topo: int = 5,
        n_bins: int = 30,
        w_jsd: float = 0.3,
        w_z: float = 0.5,
        random_state=None,
    ):
        self.n_seeds = n_seeds
        self.n_candidates = n_candidates
        self.n_clusters = n_clusters
        self.n_pca = n_pca
        self.k_topo = k_topo
        self.n_bins = n_bins
        self.w_jsd = w_jsd
        self.w_z = w_z
        self.random_state = random_state

    # ------------------------------------------------------------------
    def select(self, X_min: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Run seed selection on minority class points.

        Parameters
        ----------
        X_min : array of shape (n, d)

        Returns
        -------
        indices : ndarray of shape (n_seeds,)
            Row indices of selected seeds in X_min.
        best_score : float
        """
        rng = check_random_state(self.random_state)
        n = len(X_min)

        # --- PCA projection for scoring ---
        X_score = self._project(X_min)

        # --- K-Means clustering ---
        k = min(self.n_clusters, n)
        km = KMeans(n_clusters=k, random_state=rng.randint(0, 2**31), n_init=10)
        labels = km.fit_predict(X_score)

        # Cluster-proportional allocation
        counts = np.bincount(labels, minlength=k)
        alloc = self._proportional_alloc(counts, self.n_seeds)

        # --- Candidate search ---
        nhop_scorer = NHOP(n_bins=self.n_bins)
        best_score = -np.inf
        best_idx = None

        for _ in range(self.n_candidates):
            idx = self._stratified_sample(labels, alloc, k, rng)
            Xs = X_score[idx]
            score = self._composite_score(X_score, Xs, nhop_scorer)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx, best_score

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------
    def _composite_score(
        self, X: np.ndarray, Xs: np.ndarray, nhop: NHOP
    ) -> float:
        """Score = NHOP + AGTP - w_jsd*JSD - w_z*Z  (Eq. 6.4)"""
        nhop_val = nhop.score(X, Xs)
        agtp_val = self._agtp(X, Xs)
        jsd_val = self._jsd_mean(X, Xs)
        z_val = self._smoothness_z(Xs)
        return nhop_val + agtp_val - self.w_jsd * jsd_val - self.w_z * z_val

    def _agtp(self, X: np.ndarray, Xs: np.ndarray) -> float:
        """AGTP = 0.5*(G_sim + T_sim)  (Eqs. 6.2-6.4)"""
        return 0.5 * (self._geom_sim(X, Xs) + self._topo_sim(X, Xs))

    def _geom_sim(self, X: np.ndarray, Xs: np.ndarray) -> float:
        """Geometric similarity via mean and covariance  (Eq. 6.2)"""
        eps = 1e-9
        mu_x = X.mean(axis=0)
        mu_s = Xs.mean(axis=0)
        spread = np.mean(np.linalg.norm(X - mu_x, axis=1))

        mean_sim = max(0.0, 1.0 - np.linalg.norm(mu_x - mu_s) / (spread + eps))

        C_x = np.cov(X, rowvar=False) if X.shape[0] > 1 else np.eye(X.shape[1])
        C_s = np.cov(Xs, rowvar=False) if Xs.shape[0] > 1 else np.eye(Xs.shape[1])
        cov_norm = np.linalg.norm(C_x, "fro")
        cov_sim = max(0.0, 1.0 - np.linalg.norm(C_x - C_s, "fro") / (cov_norm + eps))

        return 0.5 * (mean_sim + cov_sim)

    def _topo_sim(self, X: np.ndarray, Xs: np.ndarray) -> float:
        """Topological similarity via k-NN distance histograms  (Eq. 6.3)"""
        k = min(self.k_topo, len(X) - 1, len(Xs) - 1)
        if k < 1:
            return 1.0

        def knn_dists(A):
            nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
            d, _ = nn.kneighbors(A)
            return d[:, 1:].mean(axis=1)  # exclude self

        dx = knn_dists(X)
        ds = knn_dists(Xs)

        lo, hi = min(dx.min(), ds.min()), max(dx.max(), ds.max())
        if lo == hi:
            return 1.0
        edges = np.linspace(lo, hi, self.n_bins + 1)
        p = np.histogram(dx, bins=edges)[0] / len(dx)
        q = np.histogram(ds, bins=edges)[0] / len(ds)
        return float(np.sum(np.minimum(p, q)))

    def _jsd_mean(self, X: np.ndarray, Xs: np.ndarray) -> float:
        """Mean JSD across features  (Eq. 6.1)"""
        k = X.shape[1]
        return float(np.mean([self._jsd_1d(X[:, j], Xs[:, j]) for j in range(k)]))

    def _jsd_1d(self, x: np.ndarray, xs: np.ndarray) -> float:
        lo = min(x.min(), xs.min())
        hi = max(x.max(), xs.max())
        if lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, self.n_bins + 1)
        p = np.histogram(x,  bins=edges)[0] / len(x)  + 1e-10
        q = np.histogram(xs, bins=edges)[0] / len(xs) + 1e-10
        p /= p.sum(); q /= q.sum()
        m = 0.5 * (p + q)
        return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

    def _smoothness_z(self, Xs: np.ndarray) -> float:
        """Spacing regulariser Z = std(nn_dists) / mean(nn_dists)  (Eq. 6.5)"""
        if len(Xs) < 2:
            return 0.0
        nn = NearestNeighbors(n_neighbors=2).fit(Xs)
        d, _ = nn.kneighbors(Xs)
        nn_dists = d[:, 1]
        mu = nn_dists.mean()
        return float(nn_dists.std() / (mu + 1e-9))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _project(self, X: np.ndarray) -> np.ndarray:
        if self.n_pca is None or self.n_pca >= X.shape[1]:
            return X
        n_comp = min(self.n_pca, X.shape[0] - 1, X.shape[1])
        return PCA(n_components=n_comp).fit_transform(X)

    @staticmethod
    def _proportional_alloc(counts: np.ndarray, total: int) -> np.ndarray:
        n = counts.sum()
        alloc = np.floor(counts / n * total).astype(int)
        remainder = total - alloc.sum()
        fracs = (counts / n * total) - alloc
        for idx in np.argsort(-fracs)[:remainder]:
            alloc[idx] += 1
        return alloc

    @staticmethod
    def _stratified_sample(
        labels: np.ndarray, alloc: np.ndarray, k: int, rng
    ) -> np.ndarray:
        idx = []
        for c in range(k):
            pool = np.where(labels == c)[0]
            take = min(alloc[c], len(pool))
            if take > 0:
                idx.extend(rng.choice(pool, size=take, replace=False))
        return np.array(idx)
