import numpy as np
import pytest
from sklearn.datasets import make_classification
from circover import GVMCO, LRECO, LSCO


X, y = make_classification(
    n_samples=200, n_features=6, n_informative=4,
    weights=[0.8, 0.2], random_state=42
)


@pytest.mark.parametrize("Cls", [GVMCO, LRECO, LSCO])
def test_resampled_shape(Cls):
    over = Cls(random_state=42)
    X_r, y_r = over.fit_resample(X, y)
    assert X_r.shape[1] == X.shape[1]
    assert len(X_r) > len(X)


@pytest.mark.parametrize("Cls", [GVMCO, LRECO, LSCO])
def test_class_balance(Cls):
    over = Cls(random_state=42)
    _, y_r = over.fit_resample(X, y)
    counts = np.bincount(y_r)
    assert counts[0] == counts[1]


@pytest.mark.parametrize("Cls", [GVMCO, LSCO])
def test_no_pca(Cls):
    over = Cls(use_pca=False, random_state=42)
    X_r, y_r = over.fit_resample(X, y)
    assert X_r.shape[1] == X.shape[1]


def test_gvmco_in_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipe = ImbPipeline([
        ("over", GVMCO(random_state=42)),
        ("clf",  RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)
