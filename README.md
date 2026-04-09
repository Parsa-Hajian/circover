# circover

**NHOP metric, geometry-preserving seed selection, and circular oversampling for imbalanced classification.**

From the thesis: *"From Distributional Similarity to Causal Imbalance: NHOP, Circular Oversampling, and a Controlled Degradation Study"* — Parsa Hajiannejad, Università degli Studi di Milano, 2025.

## Install

```bash
pip install circover
```

## Quick start

```python
import circover as cc

# NHOP: measure how faithfully synthetic data reproduces the original distribution
nhop = cc.NHOP(n_bins=30)
nhop.score(X_original, X_synthetic)           # scalar in [0, 1]
nhop.score_per_feature(X_original, X_synth)   # per-feature array
nhop.tv_per_feature(X_original, X_synth)      # TV distance = 1 - NHOP

# Geometry-preserving seed selection
selector = cc.GeometricSeedSelector(n_seeds=20, random_state=42)
seed_indices, score = selector.select(X_minority)

# Circular oversamplers — drop-in replacements for SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ("over", cc.GVMCO(random_state=42)),   # or LRECO, LSCO
    ("clf",  RandomForestClassifier()),
])
pipe.fit(X_train, y_train)
```

## Algorithms

| Class | Algorithm | Description |
|---|---|---|
| `NHOP` | — | Normalised Histogram Overlap Percentage metric |
| `GeometricSeedSelector` | Alg. 2 | Geometry-preserving seed selection (NHOP + AGTP + JSD + Z) |
| `GVMCO` | Alg. 1 | Gravity-biased Von Mises Circular Oversampling |
| `LRECO` | Alg. 2 | Local Region Estimation Circular Oversampling (Voronoi-constrained) |
| `LSCO` | Alg. 3 | Layered Segmental Circular Oversampling |

All oversamplers are compatible with `imbalanced-learn` pipelines and `sklearn` cross-validation.

## Key parameters

```python
cc.GVMCO(
    n_clusters=5,       # K-Means clusters on minority class
    k_neighbors=5,      # k-NN graph for circle formation
    kappa_max=4.0,      # max Von Mises concentration
    use_pca=True,       # False = native-dimension mode
    random_state=42,
)

cc.NHOP(n_bins=30)      # histogram bins B (default 30, stable range: 20-50)
```
