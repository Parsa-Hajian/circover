"""
circover — Geometry-preserving seed selection, circular oversampling
           methods (GVM-CO, LRE-CO, LS-CO), and degradation-recovery benchmark.

Quick start
-----------
    import circover as cc

    # Bhattacharyya Coefficient
    score = cc.BhattacharyyaCoefficient(n_bins=30).score(X_original, X_synthetic)

    # Seed selection
    seeds, s = cc.GeometricSeedSelector(n_seeds=20).select(X_minority)

    # Oversamplers (drop-in for SMOTE)
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    pipe = Pipeline([
        ("over", cc.GVMCO(random_state=42)),
        ("clf",  RandomForestClassifier()),
    ])
    pipe.fit(X_train, y_train)
"""

from .bhattacharyya import BhattacharyyaCoefficient, BC
from .seed_selection import GeometricSeedSelector
from .oversampling import GVMCO, LRECO, LSCO
from .degradation import DegradationBench

__all__ = [
    "BhattacharyyaCoefficient",
    "BC",
    "GeometricSeedSelector",
    "GVMCO",
    "LRECO",
    "LSCO",
    "DegradationBench",
]
__version__ = "0.3.0"
