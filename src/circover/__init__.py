"""
circover — NHOP metric, geometry-preserving seed selection,
           and circular oversampling methods (GVM-CO, LRE-CO, LS-CO).

Quick start
-----------
    import circover as cc

    # NHOP metric
    score = cc.NHOP(n_bins=30).score(X_original, X_synthetic)

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

from .nhop import NHOP
from .seed_selection import GeometricSeedSelector
from .oversampling import GVMCO, LRECO, LSCO

__all__ = ["NHOP", "GeometricSeedSelector", "GVMCO", "LRECO", "LSCO"]
__version__ = "0.1.0"
