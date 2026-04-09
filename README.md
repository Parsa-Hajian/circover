# circover

**Geometry-preserving seed selection, circular oversampling, and controlled degradation benchmark for imbalanced classification.**

From the thesis: *"From Distributional Similarity to Causal Imbalance: Circular Oversampling, Seed Selection, and a Controlled Degradation Study"* — Parsa Hajiannejad, Università degli Studi di Milano, 2025.

## Install

```bash
pip install circover
```

## Modules

| Class | Description |
|---|---|
| `BhattacharyyaCoefficient` / `BC` | Bhattacharyya Coefficient for marginal distributional similarity |
| `GeometricSeedSelector` | Geometry-preserving seed selection (BC + AGTP + JSD + Z) |
| `GVMCO` | Gravity-biased Von Mises Circular Oversampling |
| `LRECO` | Local Region Estimation Circular Oversampling (Voronoi-constrained) |
| `LSCO` | Layered Segmental Circular Oversampling |
| `DegradationBench` | Controlled degradation-and-recovery benchmark |

## Quick start

```python
import circover as cc

# Bhattacharyya Coefficient: measure distributional similarity
bc = cc.BhattacharyyaCoefficient(n_bins=30)
bc.score(X_original, X_synthetic)              # scalar in [0, 1]
bc.score_per_feature(X_original, X_synthetic)  # per-feature array

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

## Degradation-and-Recovery Benchmark

Run any oversampler (or pipeline) through a controlled degradation protocol to measure its recovery power:

```python
bench = cc.DegradationBench(steps=10, metric="f1", cv=5, random_state=42)
results = bench.run(pipe, X, y)          # DataFrame: degradation, score, n_minority
bench.plot(results)                      # recovery curve with ARI annotation
ari = cc.DegradationBench.area_recovery_index(results)   # scalar summary
```

The `DegradationBench`:
1. Removes minority samples in `steps` equal increments (0% -> 100%)
2. Evaluates the estimator via cross-validation at each level
3. Computes the **Area Recovery Index (ARI)** = integral of score(delta) — higher = better recovery

Compare multiple methods by their ARI:

```python
results_smote = bench.run(smote_pipe, X, y)
results_gvm   = bench.run(gvm_pipe,   X, y)

ari_smote = cc.DegradationBench.area_recovery_index(results_smote)
ari_gvm   = cc.DegradationBench.area_recovery_index(results_gvm)
print(f"SMOTE ARI: {ari_smote:.3f}   GVM-CO ARI: {ari_gvm:.3f}")
```

## Key parameters

```python
cc.GVMCO(
    n_clusters=5,       # K-Means clusters on minority class
    k_neighbors=5,      # k-NN graph for circle formation
    kappa_max=4.0,      # max Von Mises concentration
    use_pca=True,       # False = native-dimension mode
    random_state=42,
)

cc.BhattacharyyaCoefficient(n_bins=30)  # histogram bins B (default 30)

cc.DegradationBench(
    steps=10,           # number of degradation levels
    metric="f1",        # sklearn scoring string
    cv=5,               # cross-validation folds
    random_state=42,
)
```

All oversamplers are compatible with `imbalanced-learn` pipelines and `sklearn` cross-validation.
