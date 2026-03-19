import numpy as np

class DynaMMo:
    def __init__(self, latent_dim=5, max_iter=50, alpha=0.1, beta=0.1):
        self.k = latent_dim      # latent dimension
        self.max_iter = max_iter
        self.alpha = alpha       # regularization term for W
        self.beta = beta         # regularization term for A (autoregressive)

    def fit_transform(self, X):
        """
        Fit DynaMMo on data X and impute missing values.
        X: 2D numpy array with NaNs as missing values
        """
        X = np.array(X, dtype=np.float64)
        n, m = X.shape
        mask = ~np.isnan(X)

        # Initialize missing values to column mean
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X_filled))
        X_filled[inds] = np.take(col_means, inds[1])

        # Initialize latent variables Z, autoregression matrix A, projection W
        Z = np.random.randn(n, self.k)
        A = np.eye(self.k) * 0.9  # simple auto-regressive init
        W = np.random.randn(m, self.k)

        for it in range(self.max_iter):
            # Step 1: Update Z (latent factors)
            for t in range(n):
                # Prediction from A*Z[t-1] if not first timepoint
                prior = A @ Z[t - 1] if t > 0 else np.zeros(self.k)
                W_obs = W[mask[t], :]  # observed features
                X_obs = X_filled[t, mask[t]]

                # Solve linear system to update Z[t]
                M = W_obs.T @ W_obs + self.beta * np.eye(self.k)
                b = W_obs.T @ X_obs + self.beta * prior
                Z[t] = np.linalg.solve(M, b)

            # Step 2: Update W (projection matrix)
            for j in range(m):
                Z_j = Z[mask[:, j], :]
                X_j = X_filled[mask[:, j], j]
                M = Z_j.T @ Z_j + self.alpha * np.eye(self.k)
                b = Z_j.T @ X_j
                W[j] = np.linalg.solve(M, b)

            # Step 3: Update A (autoregression)
            Z_prev = Z[:-1]
            Z_next = Z[1:]
            A = np.linalg.solve(Z_prev.T @ Z_prev + self.beta * np.eye(self.k), Z_prev.T @ Z_next)

            # Step 4: Reconstruct and fill missing values
            X_hat = Z @ W.T
            X_filled[~mask] = X_hat[~mask]

        return X_filled
