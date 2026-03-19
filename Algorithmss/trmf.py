import numpy as np

class TRMFImputer:
    def __init__(self, rank=5, lag_idx=[1, 2, 3], lambdas=(1.0, 1.0, 1.0), max_iter=50):
        """
        Initialize the TRMF imputer with hyperparameters.
        :param rank: Latent dimension size (k)
        :param lag_idx: List of lag indices for temporal dependency
        :param lambdas: Tuple (lambdaF, lambdaX, lambdaW) for regularization
        :param max_iter: Number of iterations for optimization
        """
        self.rank = rank
        self.lag_idx = lag_idx
        self.lambdaF, self.lambdaX, self.lambdaW = lambdas
        self.max_iter = max_iter

        self.F = None  # latent features for variables
        self.X = None  # temporal latent features
        self.lag_val = None  # autoregressive weights

    def _initialize(self, Y):
        n, T = Y.shape
        np.random.seed(0)
        self.F = np.random.rand(n, self.rank)
        self.X = np.random.rand(self.rank, T)
        self.lag_val = np.random.randn(self.rank, len(self.lag_idx))

    def _reconstruct(self):
        return self.F @ self.X

    def fit(self, Y, observed_mask):
        """
        Fit the TRMF model to observed data.
        :param Y: Input matrix (n x T) with missing values
        :param observed_mask: Boolean mask (same shape as Y) indicating observed entries
        """
        n, T = Y.shape
        self._initialize(Y)

        for it in range(self.max_iter):
            # Update X using regularized least squares with autoregressive constraint
            for t in range(max(self.lag_idx), T):
                A = self.F.T @ self.F + self.lambdaX * np.eye(self.rank)
                b = self.F.T @ Y[:, t]

                # Add autoregressive terms
                for l_idx, lag in enumerate(self.lag_idx):
                    b -= self.lambdaW * self.lag_val[:, l_idx] * self.X[:, t - lag]

                self.X[:, t] = np.linalg.solve(A, b)

            # Update F
            for i in range(n):
                obs_t = observed_mask[i]
                if not np.any(obs_t):
                    continue
                A = self.X[:, obs_t] @ self.X[:, obs_t].T + self.lambdaF * np.eye(self.rank)
                b = Y[i, obs_t] @ self.X[:, obs_t].T
                self.F[i] = np.linalg.solve(A, b)

            # Update lag_val
            for k in range(self.rank):
                for l_idx, lag in enumerate(self.lag_idx):
                    if T - lag <= 0:
                        continue
                    num = np.dot(self.X[k, lag:], self.X[k, :-lag])
                    den = np.dot(self.X[k, :-lag], self.X[k, :-lag]) + 1e-6
                    self.lag_val[k, l_idx] = num / den

    def transform(self):
        """
        Return the reconstructed matrix.
        """
        return self._reconstruct()

    def fit_transform(self, Y):
        """
        Shortcut to fit and return imputed matrix.
        :param Y: Input matrix with missing values
        :return: Imputed matrix
        """
        observed_mask = ~np.isnan(Y)
        Y_filled = np.nan_to_num(Y, nan=0.0)
        self.fit(Y_filled, observed_mask)
        return self.transform()
