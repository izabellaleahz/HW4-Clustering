import numpy as np
import pytest

from cluster import KMeans, make_clusters


def test_invalid_k():
    """k must be a positive integer."""
    with pytest.raises(TypeError):
        KMeans(k=2.0)
    with pytest.raises(ValueError):
        KMeans(k=0)
    with pytest.raises(ValueError):
        KMeans(k=-3)


def test_fit_bad_input():
    """fit() should reject non-arrays, wrong dims, and n < k."""
    km = KMeans(k=3)
    with pytest.raises(TypeError):
        km.fit([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        km.fit(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        km.fit(np.random.rand(2, 3))  # only 2 points for k=3


def test_predict_before_fit():
    """predict / get_error / get_centroids should fail before fit."""
    km = KMeans(k=2)
    with pytest.raises(ValueError):
        km.predict(np.random.rand(5, 2))
    with pytest.raises(ValueError):
        km.get_error()
    with pytest.raises(ValueError):
        km.get_centroids()


def test_predict_feature_mismatch():
    """predict() should reject data with wrong number of features."""
    mat, _ = make_clusters(n=30, m=2, k=2, seed=0)
    km = KMeans(k=2)
    km.fit(mat)
    with pytest.raises(ValueError):
        km.predict(np.random.rand(5, 4))


def test_fit_predict_shapes():
    """Labels, centroids, and error should have correct shapes and ranges."""
    mat, _ = make_clusters(n=200, m=2, k=3, scale=0.3, seed=1)
    km = KMeans(k=3, tol=1e-5, max_iter=100)
    km.fit(mat)
    labels = km.predict(mat)

    assert labels.shape == (200,)
    assert labels.min() >= 0 and labels.max() < 3
    assert km.get_centroids().shape == (3, 2)
    assert km.get_error() >= 0


def test_recovers_separated_clusters():
    """Well-separated clusters should be recovered perfectly (up to label permutation)."""
    rng = np.random.RandomState(42)
    c1 = rng.normal(loc=[0, 0], scale=0.05, size=(50, 2))
    c2 = rng.normal(loc=[10, 0], scale=0.05, size=(50, 2))
    c3 = rng.normal(loc=[0, 10], scale=0.05, size=(50, 2))
    mat = np.vstack([c1, c2, c3])
    true = np.array([0]*50 + [1]*50 + [2]*50)

    km = KMeans(k=3, tol=1e-8, max_iter=300)
    km.fit(mat)
    pred = km.predict(mat)

    # each true group should map to exactly one predicted label
    for g in range(3):
        assert len(np.unique(pred[true == g])) == 1


def test_k1_single_cluster():
    """k=1 should assign every point to the same label."""
    mat, _ = make_clusters(n=50, m=2, k=3, seed=8)
    km = KMeans(k=1)
    km.fit(mat)
    assert np.unique(km.predict(mat)).size == 1
