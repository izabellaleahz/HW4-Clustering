import numpy as np
import pytest
from sklearn.metrics import silhouette_score

from cluster import Silhouette, make_clusters


def test_bad_inputs():
    """score() should reject non-arrays, wrong dims, and mismatched lengths."""
    s = Silhouette()
    with pytest.raises(TypeError):
        s.score([[1, 2]], np.array([0]))
    with pytest.raises(ValueError):
        s.score(np.array([1, 2, 3]), np.array([0, 0, 1]))  # X not 2D
    with pytest.raises(ValueError):
        s.score(np.random.rand(5, 2), np.random.rand(3))   # length mismatch


def test_single_cluster_raises():
    """Silhouette requires at least 2 clusters."""
    mat = np.random.rand(10, 2)
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        Silhouette().score(mat, labels)


def test_matches_sklearn():
    """Mean of per-point scores should match sklearn's silhouette_score."""
    mat, labels = make_clusters(n=120, m=2, k=3, scale=0.3, seed=3)
    scores = Silhouette().score(mat, labels)
    expected = silhouette_score(mat, labels)

    assert scores.shape == (mat.shape[0],)
    assert np.isfinite(scores).all()
    assert abs(np.mean(scores) - expected) < 1e-6


def test_scores_in_range():
    """Every silhouette score should be in [-1, 1]."""
    mat, labels = make_clusters(n=100, m=2, k=4, scale=0.5, seed=5)
    scores = Silhouette().score(mat, labels)
    assert (scores >= -1 - 1e-9).all() and (scores <= 1 + 1e-9).all()


def test_well_separated_high_scores():
    """Tight, far-apart clusters should have silhouette scores near 1."""
    rng = np.random.RandomState(0)
    mat = np.vstack([
        rng.normal(loc=[0, 0], scale=0.01, size=(40, 2)),
        rng.normal(loc=[10, 10], scale=0.01, size=(40, 2)),
    ])
    labels = np.array([0]*40 + [1]*40)
    scores = Silhouette().score(mat, labels)
    assert np.mean(scores) > 0.95
