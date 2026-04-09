import numpy as np
import pytest
from circover import NHOP


rng = np.random.default_rng(42)
X  = rng.normal(0, 1, (300, 4))
Xs = rng.normal(0.5, 1, (200, 4))


def test_identical_returns_one():
    assert NHOP().score(X, X) == pytest.approx(1.0, abs=1e-6)


def test_bounded():
    s = NHOP().score(X, Xs)
    assert 0.0 <= s <= 1.0


def test_symmetric():
    assert NHOP().score(X, Xs) == pytest.approx(NHOP().score(Xs, X), abs=1e-6)


def test_per_feature_shape():
    scores = NHOP().score_per_feature(X, Xs)
    assert scores.shape == (4,)
    assert all(0 <= s <= 1 for s in scores)


def test_tv_identity():
    nhop = NHOP(n_bins=30)
    nhop_scores = nhop.score_per_feature(X, Xs)
    tv_scores = nhop.tv_per_feature(X, Xs)
    np.testing.assert_allclose(nhop_scores + tv_scores, 1.0, atol=1e-10)


def test_degenerate_constant_feature():
    X_const = np.ones((100, 1))
    Xs_const = np.ones((80, 1))
    assert NHOP().score(X_const, Xs_const) == pytest.approx(1.0)


def test_disjoint_near_zero():
    X_left  = np.random.default_rng(0).uniform(0, 1, (200, 1))
    X_right = np.random.default_rng(1).uniform(5, 6, (200, 1))
    assert NHOP(n_bins=30).score(X_left, X_right) < 0.05
