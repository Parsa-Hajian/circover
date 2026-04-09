import numpy as np
import pytest
from circover import BhattacharyyaCoefficient, BC


rng = np.random.default_rng(42)
X  = rng.normal(0, 1, (300, 4))
Xs = rng.normal(0.5, 1, (200, 4))


def test_alias():
    assert BC is BhattacharyyaCoefficient


def test_identical_returns_one():
    assert BC().score(X, X) == pytest.approx(1.0, abs=1e-6)


def test_bounded():
    s = BC().score(X, Xs)
    assert 0.0 <= s <= 1.0


def test_symmetric():
    assert BC().score(X, Xs) == pytest.approx(BC().score(Xs, X), abs=1e-6)


def test_per_feature_shape():
    scores = BC().score_per_feature(X, Xs)
    assert scores.shape == (4,)
    assert all(0 <= s <= 1 for s in scores)


def test_degenerate_constant_feature():
    X_const = np.ones((100, 1))
    Xs_const = np.ones((80, 1))
    assert BC().score(X_const, Xs_const) == pytest.approx(1.0)


def test_disjoint_near_zero():
    X_left  = np.random.default_rng(0).uniform(0, 1, (200, 1))
    X_right = np.random.default_rng(1).uniform(5, 6, (200, 1))
    assert BC(n_bins=30).score(X_left, X_right) < 0.05
