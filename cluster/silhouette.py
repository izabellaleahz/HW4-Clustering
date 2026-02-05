import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        self._fitted = True

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError('X and y must be numpy arrays')
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have same number of observations')

        labels = np.unique(y)
        if labels.size < 2:
            raise ValueError('silhouette score requires at least 2 clusters')

        dist_mat = cdist(X, X)
        scores = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            same_mask = y == y[i]
            same_count = np.sum(same_mask)
            if same_count <= 1:
                scores[i] = 0.0
                continue

            a = np.mean(dist_mat[i, same_mask & (np.arange(X.shape[0]) != i)])

            b = None
            for label in labels:
                if label == y[i]:
                    continue
                other_mask = y == label
                mean_dist = np.mean(dist_mat[i, other_mask])
                if b is None or mean_dist < b:
                    b = mean_dist

            denom = max(a, b) if b is not None else a
            scores[i] = 0.0 if denom == 0 else (b - a) / denom

        return scores