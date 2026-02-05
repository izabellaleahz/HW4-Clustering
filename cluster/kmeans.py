import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int):
            raise TypeError('k must be an int')
        if k <= 0:
            raise ValueError('k must be > 0')
        if tol <= 0:
            raise ValueError('tol must be > 0')
        if max_iter <= 0:
            raise ValueError('max_iter must be > 0')

        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self._centroids = None
        self._error = None
        self._fitted = False
        self._n_features = None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if not isinstance(mat, np.ndarray):
            raise TypeError('mat must be a numpy array')
        if mat.ndim != 2:
            raise ValueError('mat must be a 2D array')
        n, m = mat.shape
        if n < self.k:
            raise ValueError('number of observations must be >= k')

        self._n_features = m
        init_idx = np.random.choice(n, size=self.k, replace=False)
        centroids = mat[init_idx].copy()
        prev_error = None

        for _ in range(self.max_iter):
            distances = cdist(mat, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for idx in range(self.k):
                members = mat[labels == idx]
                if members.size == 0:
                    new_centroids[idx] = mat[np.random.randint(0, n)]
                else:
                    new_centroids[idx] = members.mean(axis=0)

            distances = cdist(mat, new_centroids)
            min_dist = distances[np.arange(n), np.argmin(distances, axis=1)]
            error = np.mean(min_dist ** 2)

            if prev_error is not None and abs(prev_error - error) < self.tol:
                centroids = new_centroids
                prev_error = error
                break

            centroids = new_centroids
            prev_error = error

        self._centroids = centroids
        self._error = prev_error if prev_error is not None else 0.0
        self._fitted = True

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if not self._fitted:
            raise ValueError('model must be fit before calling predict')
        if not isinstance(mat, np.ndarray):
            raise TypeError('mat must be a numpy array')
        if mat.ndim != 2:
            raise ValueError('mat must be a 2D array')
        if mat.shape[1] != self._n_features:
            raise ValueError('input data must have same number of features as fit data')

        distances = cdist(mat, self._centroids)
        return np.argmin(distances, axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if not self._fitted:
            raise ValueError('model must be fit before calling get_error')
        return float(self._error)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if not self._fitted:
            raise ValueError('model must be fit before calling get_centroids')
        return self._centroids.copy()