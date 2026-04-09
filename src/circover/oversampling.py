"""
Circular Oversampling Algorithms
=================================
Three algorithms from Chapter 7 of the thesis, all inheriting from
imbalanced-learn's BaseOverSampler so they work inside sklearn pipelines.

    from circover import GVMCO, LRECO, LSCO

    pipe = Pipeline([
        ("over", GVMCO(random_state=42)),
        ("clf",  RandomForestClassifier()),
    ])
    pipe.fit(X_train, y_train)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import vonmises
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from imblearn.over_sampling.base import BaseOverSampler

from ._gravity import compute_gravity, kappa_from_score


# ============================================================================
#  Shared helpers
# ============================================================================

def _pca_project(X: np.ndarray, n_components: int = 2):
    """Fit PCA and return (X_2d, pca_object)."""
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X), pca


def _pca_inverse(X_2d: np.ndarray, pca: PCA) -> np.ndarray:
    return pca.inverse_transform(X_2d)


def _circle(xi: np.ndarray, xj: np.ndarray):
    """Circle center and radius from two points (Eqs. 7.1-7.2)."""
    center = 0.5 * (xi + xj)
    radius = 0.5 * np.linalg.norm(xi - xj)
    return center, radius


def _uniform_disk_point(center, radius, rng):
    """Area-uniform sample inside a 2-D disk."""
    theta = rng.uniform(0, 2 * np.pi)
    r = radius * np.sqrt(rng.uniform(0, 1))
    return center + r * np.array([np.cos(theta), np.sin(theta)])


def _uniform_ball_point(center, radius, d, rng):
    """Volume-uniform sample inside a d-dimensional ball (Eq. 7.3 native mode)."""
    z = rng.standard_normal(d)
    z /= np.linalg.norm(z) + 1e-15
    r = radius * (rng.uniform(0, 1) ** (1.0 / d))
    return center + r * z


def _build_knn(X: np.ndarray, k: int):
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]  # exclude self


# ============================================================================
#  Algorithm 1 — GVM-CO
# ============================================================================

class GVMCO(BaseOverSampler):
    """
    Gravity-biased Von Mises Circular Oversampling (GVM-CO).

    Chapter 7, Algorithm 1 of the thesis.

    Parameters
    ----------
    sampling_strategy : float or "auto", default="auto"
        Ratio of minority to majority after resampling, or "auto"
        (balance classes to 1:1).
    n_clusters : int, default=5
        K-Means clusters on projected minority class.
    k_neighbors : int, default=5
        Neighbours per seed in the k-NN graph.
    kappa_max : float, default=4.0
        Maximum Von Mises concentration parameter.
    gamma : float, default=1.5
        Non-linearity exponent for kappa mapping.
    alpha : float, default=0.5
        Additive-multiplicative blend (Eq. 7.9).
    use_pca : bool, default=True
        If False, operate directly in original feature space (native-dim mode).
    random_state : int or None, default=None
    """

    def __init__(
        self,
        sampling_strategy="auto",
        n_clusters: int = 5,
        k_neighbors: int = 5,
        kappa_max: float = 4.0,
        gamma: float = 1.5,
        alpha: float = 0.5,
        use_pca: bool = True,
        random_state=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.kappa_max = kappa_max
        self.gamma = gamma
        self.alpha = alpha
        self.use_pca = use_pca
        self.random_state = random_state

    def _fit_resample(self, X, y):
        rng = check_random_state(self.random_state)
        X_res, y_res = X.copy(), y.copy()

        for cls, n_synth in self.sampling_strategy_.items():
            X_min = X[y == cls]
            synth = self._generate(X_min, n_synth, rng)
            X_res = np.vstack([X_res, synth])
            y_res = np.hstack([y_res, np.full(n_synth, cls)])

        return X_res, y_res

    def _generate(self, X_min: np.ndarray, n_synth: int, rng) -> np.ndarray:
        n, d = X_min.shape

        if self.use_pca and d > 2:
            X_proj, pca = _pca_project(X_min, 2)
        else:
            X_proj, pca = X_min.copy(), None

        # Clustering
        k = min(self.n_clusters, n)
        km = KMeans(n_clusters=k, random_state=rng.randint(0, 2**31), n_init=10)
        labels = km.fit_predict(X_proj)

        g_norm, g_centers, _ = compute_gravity(X_proj, labels, self.k_neighbors)
        knn_idx = _build_knn(X_proj, self.k_neighbors)

        synth_proj = []
        for _ in range(n_synth):
            # Pick random seed
            i = rng.randint(0, n)
            c = labels[i]
            j = knn_idx[i, rng.randint(0, knn_idx.shape[1])]

            center, radius = _circle(X_proj[i], X_proj[j])
            if radius < 1e-12:
                synth_proj.append(center.copy())
                continue

            # Von Mises direction toward gravity center (Eq. 7.6)
            gc = g_centers[c]
            direction = gc - center
            mu = np.arctan2(direction[1], direction[0]) if X_proj.shape[1] == 2 else 0.0

            # Concentration score (Eqs. 7.7-7.10)
            d_all = np.linalg.norm(X_proj - center, axis=1)
            d_min, d_max = d_all.min(), d_all.max()
            dist = np.linalg.norm(gc - center)
            s_d = (d_max - dist) / (d_max - d_min + 1e-9)
            s_g = g_norm[c]
            s_combined = (
                self.alpha * (s_d + s_g) / 2.0
                + (1.0 - self.alpha) * s_d * s_g
            )
            kappa = kappa_from_score(
                np.array([s_combined]), self.kappa_max, self.gamma
            )[0]

            if pca is None or not self.use_pca:
                # Native-dimension mode: vMF approximation (Eq. 7.13)
                z = rng.standard_normal(d)
                mu_vec = direction / (np.linalg.norm(direction) + 1e-15)
                z_biased = z + kappa * mu_vec
                z_biased /= np.linalg.norm(z_biased) + 1e-15
                r = radius * (rng.uniform(0, 1) ** (1.0 / d))
                pt = center + r * z_biased
            else:
                theta = vonmises.rvs(kappa, loc=mu, random_state=rng)
                r = radius * np.sqrt(rng.uniform(0, 1))
                pt = center + r * np.array([np.cos(theta), np.sin(theta)])

            synth_proj.append(pt)

        synth_proj = np.array(synth_proj)

        if pca is not None and self.use_pca:
            return _pca_inverse(synth_proj, pca)
        return synth_proj


# ============================================================================
#  Algorithm 2 — LRE-CO
# ============================================================================

class LRECO(BaseOverSampler):
    """
    Local Region Estimation Circular Oversampling (LRE-CO).

    Chapter 7, Algorithm 2 of the thesis.
    Restricts generation to disk ∩ Voronoi cell with a certainty threshold.

    Parameters
    ----------
    sampling_strategy : float or "auto", default="auto"
    n_clusters : int, default=5
    k_neighbors : int, default=5
    certainty_threshold : float, default=0.80
        Minimum minority fraction inside disk to accept it (Eq. 7.14).
    max_rejections : int, default=50
        Max rejection-sampling iterations per sample.
    use_pca : bool, default=True
    random_state : int or None, default=None
    """

    def __init__(
        self,
        sampling_strategy="auto",
        n_clusters: int = 5,
        k_neighbors: int = 5,
        certainty_threshold: float = 0.80,
        max_rejections: int = 50,
        use_pca: bool = True,
        random_state=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.certainty_threshold = certainty_threshold
        self.max_rejections = max_rejections
        self.use_pca = use_pca
        self.random_state = random_state

    def _fit_resample(self, X, y):
        rng = check_random_state(self.random_state)
        X_res, y_res = X.copy(), y.copy()

        for cls, n_synth in self.sampling_strategy_.items():
            X_min = X[y == cls]
            X_maj = X[y != cls]
            synth = self._generate(X_min, X_maj, n_synth, rng)
            X_res = np.vstack([X_res, synth])
            y_res = np.hstack([y_res, np.full(n_synth, cls)])

        return X_res, y_res

    def _generate(self, X_min, X_maj, n_synth, rng):
        n, d = X_min.shape

        if self.use_pca and d > 2:
            X_proj, pca = _pca_project(X_min, 2)
            if len(X_maj) > 0:
                X_maj_proj = pca.transform(X_maj)
            else:
                X_maj_proj = np.empty((0, 2))
        else:
            X_proj, pca = X_min.copy(), None
            X_maj_proj = X_maj.copy()

        k = min(self.n_clusters, n)
        km = KMeans(n_clusters=k, random_state=rng.randint(0, 2**31), n_init=10)
        labels = km.fit_predict(X_proj)
        centroids = np.array([X_proj[labels == c].mean(axis=0) for c in range(k)])

        g_norm, _, _ = compute_gravity(X_proj, labels, self.k_neighbors)
        knn_idx = _build_knn(X_proj, self.k_neighbors)

        # Gravity-weighted allocation per cluster (Eq. 7.15)
        counts = np.bincount(labels, minlength=k).astype(float)
        weights = counts * g_norm
        weights /= weights.sum() + 1e-9
        alloc = np.floor(weights * n_synth).astype(int)
        alloc[np.argmax(weights)] += n_synth - alloc.sum()

        all_proj = np.vstack([X_proj, X_maj_proj]) if len(X_maj_proj) else X_proj
        all_labels_min = np.ones(len(X_proj))
        all_labels_maj = np.zeros(len(X_maj_proj))
        is_min = np.hstack([all_labels_min, all_labels_maj]).astype(bool)

        synth_proj = []
        for c in range(k):
            cluster_pts = X_proj[labels == c]
            cluster_knn = knn_idx[labels == c]

            for _ in range(alloc[c]):
                for _attempt in range(20):  # retry if no valid circle
                    ii = rng.randint(0, len(cluster_pts))
                    xi = cluster_pts[ii]
                    jj = rng.randint(0, cluster_knn.shape[1])
                    xj = X_proj[cluster_knn[ii, jj]]

                    center, radius = _circle(xi, xj)
                    if radius < 1e-12:
                        continue

                    # Certainty check (Eq. 7.14)
                    dists = np.linalg.norm(all_proj - center, axis=1)
                    inside = dists <= radius
                    n_min_in = is_min[inside].sum()
                    n_total_in = inside.sum()
                    certainty = n_min_in / (n_total_in + 1e-9)
                    if certainty < self.certainty_threshold:
                        continue

                    # Rejection sampling: disk ∩ Voronoi_c
                    for _ in range(self.max_rejections):
                        cand = _uniform_disk_point(center, radius, rng)
                        # Check Voronoi: nearest centroid must be c
                        nearest = np.argmin(np.linalg.norm(centroids - cand, axis=1))
                        if nearest == c:
                            synth_proj.append(cand)
                            break
                    else:
                        synth_proj.append(_uniform_disk_point(center, radius, rng))
                    break
                else:
                    # Fallback: random seed pair in cluster
                    ii = rng.randint(0, len(cluster_pts))
                    xi = cluster_pts[ii]
                    synth_proj.append(xi + rng.standard_normal(xi.shape) * 0.01)

        synth_proj = np.array(synth_proj)
        if pca is not None and self.use_pca:
            return _pca_inverse(synth_proj, pca)
        return synth_proj


# ============================================================================
#  Algorithm 3 — LS-CO
# ============================================================================

class LSCO(BaseOverSampler):
    """
    Layered Segmental Circular Oversampling (LS-CO).

    Chapter 7, Algorithm 3 of the thesis.
    Decomposes each circle into L annular layers × M angular segments
    with differentiated Von Mises sampling per cell.

    Parameters
    ----------
    sampling_strategy : float or "auto", default="auto"
    n_clusters : int, default=5
    k_neighbors : int, default=5
    n_layers : int, default=4
        Number of concentric annular layers (L).
    n_segments : int, default=8
        Number of angular segments per layer (M).
    kappa_max : float, default=4.0
    beta : float, default=1.0
        Layer weight decay rate (Eq. 7.16).
    sigma_theta : float, default=1.0
        Angular spread for segment concentration (Eq. 7.18).
    use_pca : bool, default=True
    random_state : int or None, default=None
    """

    def __init__(
        self,
        sampling_strategy="auto",
        n_clusters: int = 5,
        k_neighbors: int = 5,
        n_layers: int = 4,
        n_segments: int = 8,
        kappa_max: float = 4.0,
        beta: float = 1.0,
        sigma_theta: float = 1.0,
        use_pca: bool = True,
        random_state=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.n_layers = n_layers
        self.n_segments = n_segments
        self.kappa_max = kappa_max
        self.beta = beta
        self.sigma_theta = sigma_theta
        self.use_pca = use_pca
        self.random_state = random_state

    def _fit_resample(self, X, y):
        rng = check_random_state(self.random_state)
        X_res, y_res = X.copy(), y.copy()

        for cls, n_synth in self.sampling_strategy_.items():
            X_min = X[y == cls]
            synth = self._generate(X_min, n_synth, rng)
            X_res = np.vstack([X_res, synth])
            y_res = np.hstack([y_res, np.full(n_synth, cls)])

        return X_res, y_res

    def _generate(self, X_min, n_synth, rng):
        n, d = X_min.shape

        if self.use_pca and d > 2:
            X_proj, pca = _pca_project(X_min, 2)
        else:
            X_proj, pca = X_min.copy(), None

        k = min(self.n_clusters, n)
        km = KMeans(n_clusters=k, random_state=rng.randint(0, 2**31), n_init=10)
        labels = km.fit_predict(X_proj)

        g_norm, g_centers, _ = compute_gravity(X_proj, labels, self.k_neighbors)
        knn_idx = _build_knn(X_proj, self.k_neighbors)

        L = self.n_layers
        M = self.n_segments
        seg_edges = np.linspace(0, 2 * np.pi, M + 1)
        seg_mids = 0.5 * (seg_edges[:-1] + seg_edges[1:])

        synth_proj = []
        for _ in range(n_synth):
            i = rng.randint(0, n)
            c = labels[i]
            j = knn_idx[i, rng.randint(0, knn_idx.shape[1])]

            center, radius = _circle(X_proj[i], X_proj[j])
            if radius < 1e-12:
                synth_proj.append(center.copy())
                continue

            gc = g_centers[c]
            direction = gc - center
            mu = np.arctan2(direction[1], direction[0])
            r_gc = np.linalg.norm(direction)  # dist to gravity center

            # Layer weights (Eq. 7.16)
            r_mids = np.array([(l - 0.5) / L * radius for l in range(1, L + 1)])
            w_layers = np.exp(-self.beta * np.abs(r_mids - r_gc) / (radius + 1e-9))
            w_layers /= w_layers.sum()

            # Choose layer proportionally
            ell = rng.choice(L, p=w_layers)  # 0-indexed
            r_inner = ell / L * radius
            r_outer = (ell + 1) / L * radius

            # Segment concentration (Eq. 7.18)
            kappa_segs = (
                self.kappa_max
                * ((L - ell) / L)
                * np.exp(-(seg_mids - mu) ** 2 / (2 * self.sigma_theta ** 2))
            )
            # Segment probabilities proportional to kappa
            seg_probs = kappa_segs / (kappa_segs.sum() + 1e-9)
            seg_probs = seg_probs / seg_probs.sum()  # ensure exact sum=1
            m = rng.choice(M, p=seg_probs)

            # Sample angle from Von Mises within segment [theta_m, theta_{m+1})
            for _ in range(20):
                theta = vonmises.rvs(max(kappa_segs[m], 0.01), loc=mu, random_state=rng)
                theta = theta % (2 * np.pi)
                if seg_edges[m] <= theta < seg_edges[m + 1]:
                    break
            # area-uniform radial sample within annulus (Eq. 7.19)
            r = np.sqrt(r_inner ** 2 + rng.uniform(0, 1) * (r_outer ** 2 - r_inner ** 2))
            if X_proj.shape[1] == 2:
                pt = center + r * np.array([np.cos(theta), np.sin(theta)])
            else:
                # Native-dim: vMF direction biased by gravity, radially clamped to annulus
                z = rng.standard_normal(X_proj.shape[1])
                mu_vec = direction / (np.linalg.norm(direction) + 1e-15)
                z_biased = z + kappa_segs[m] * mu_vec
                z_biased /= np.linalg.norm(z_biased) + 1e-15
                pt = center + r * z_biased
            synth_proj.append(pt)

        synth_proj = np.array(synth_proj)
        if pca is not None and self.use_pca:
            return _pca_inverse(synth_proj, pca)
        return synth_proj
