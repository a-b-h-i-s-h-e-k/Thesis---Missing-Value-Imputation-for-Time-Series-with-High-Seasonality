import numpy as np

class CDRec:
    def __init__(self, rank=5, lambda_reg=0.1, iterations=50):
        self.rank = rank                # Low-rank factorization rank
        self.lambda_reg = lambda_reg    # Regularization parameter
        self.iterations = iterations    # Number of update steps

    def impute(self, X):
        """
        Impute missing values in X using low-rank matrix factorization (CDRec).
        Missing values should be marked with np.nan.
        """
        X = np.array(X, dtype=np.float64)
        missing_mask = np.isnan(X)
        X_filled = X.copy()

        # Initialize missing values to column means
        col_means = np.nanmean(X_filled, axis=0)
        for i in range(X_filled.shape[0]):
            for j in range(X_filled.shape[1]):
                if missing_mask[i, j]:
                    X_filled[i, j] = col_means[j]

        n, m = X.shape
        U = np.random.rand(n, self.rank)
        V = np.random.rand(m, self.rank)

        # Alternating Minimization
        for it in range(self.iterations):
            # Update U
            for i in range(n):
                V_j = V[~missing_mask[i], :]
                X_i = X_filled[i, ~missing_mask[i]]
                if V_j.size > 0:
                    A = V_j.T @ V_j + self.lambda_reg * np.eye(self.rank)
                    b = V_j.T @ X_i
                    U[i] = np.linalg.solve(A, b)

            # Update V
            for j in range(m):
                U_i = U[~missing_mask[:, j], :]
                X_j = X_filled[~missing_mask[:, j], j]
                if U_i.size > 0:
                    A = U_i.T @ U_i + self.lambda_reg * np.eye(self.rank)
                    b = U_i.T @ X_j
                    V[j] = np.linalg.solve(A, b)

            # Reconstruct matrix
            X_hat = U @ V.T
            X_filled[missing_mask] = X_hat[missing_mask]

        return X_filled
