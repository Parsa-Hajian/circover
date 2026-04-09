"""
Shared gravity scoring framework used by GVM-CO and LS-CO (Section 7.2).
"""

from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_gravity(
    X: np.ndarray,
    labels: np.ndarray,
    k_nn: int = 5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gravity scores and gravity centers for each cluster.

    Parameters
    ----------
    X      : (n, 2) projected minority points
    labels : (n,)  cluster assignments
    k_nn   : neighbours for sparsity/gravity-center weights

    Returns
    -------
    g_norm   : (K,) normalised gravity scores in [0,1]   (Eq. 7.5-7.6)
    g_centers: (K, 2) gravity centers                    (Eq. 7.7)
    centroids: (K, 2) plain K-means centroids
    """
    K = int(labels.max()) + 1
    d = X.shape[1]
    g_raw = np.zeros(K)
    g_centers = np.zeros((K, d))
    centroids = np.zeros((K, d))

    for c in range(K):
        mask = labels == c
        Xc = X[mask]
        nc = len(Xc)
        if nc == 0:
            continue

        mu_c = Xc.mean(axis=0)
        centroids[c] = mu_c

        # Spread (Eq. 7.1)
        spread = np.mean(np.linalg.norm(Xc - mu_c, axis=1)) + eps

        # Area (Eq. 7.2)
        area = np.pi * spread ** 2 + eps

        # Density (Eq. 7.3)
        density = nc / area

        # Sparsity: mean intra-cluster k-NN distance (Eq. 7.4)
        k = min(k_nn, nc - 1)
        if k > 0:
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
            d, _ = nn.kneighbors(Xc)
            sparsity = d[:, 1:].mean()
        else:
            sparsity = spread

        # Gravity score (Eq. 7.5)
        g_raw[c] = density / (spread * sparsity + eps)

        # Gravity center: density-weighted centroid (Eq. 7.7)
        if k > 0:
            nn2 = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
            d2, _ = nn2.kneighbors(Xc)
            sigma_local = d2[:, 1:].mean(axis=1) + eps
        else:
            sigma_local = np.full(nc, spread)

        weights = 1.0 / sigma_local
        g_centers[c] = (weights[:, None] * Xc).sum(axis=0) / weights.sum()

    # Normalise gravity scores to [0,1] (Eq. 7.6)
    g_min, g_max = g_raw.min(), g_raw.max()
    g_norm = (g_raw - g_min) / (g_max - g_min + eps)

    return g_norm, g_centers, centroids


def kappa_from_score(
    s_combined: np.ndarray,
    kappa_max: float = 4.0,
    gamma: float = 1.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Map combined score -> Von Mises concentration (Eq. 7.10-7.11)."""
    return np.maximum(kappa_max * (s_combined ** gamma), eps)
