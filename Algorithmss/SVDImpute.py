'''
IterativeSVD is equivalent to SVDImpute or SoftImpute with iterative reconstruction.
It fills missing values, computes a low-rank approximation using truncated SVD, 
and re-imputes missing entries until convergence.
'''


import numpy as np
from numpy.linalg import svd

class IterativeSVD:
    def __init__(self, rank=5, max_iters=100, tol=1e-5, gradual_rank_increase=True):
        self.rank = rank
        self.max_iters = max_iters
        self.tol = tol
        self.gradual_rank_increase = gradual_rank_increase

    def fit_transform(self, X):
        """
        X: 2D numpy array with NaNs
        Returns: imputed matrix
        """
        X = np.array(X, dtype=np.float64)
        X_filled = X.copy()
        nan_mask = np.isnan(X)

        # Fill missing values with 0 initially
        X_filled[nan_mask] = 0

        prev = X_filled.copy()

        for it in range(self.max_iters):
            # Gradual rank increase: 1, 2, 4, 8, ...
            if self.gradual_rank_increase:
                curr_rank = min(2 ** it, self.rank)
                if it >= 20:  # Prevent overflow
                    self.gradual_rank_increase = False
            else:
                curr_rank = self.rank

            # SVD decomposition
            U, S, Vt = svd(X_filled, full_matrices=False)
            S_truncated = np.diag(S[:curr_rank])
            U_trunc = U[:, :curr_rank]
            Vt_trunc = Vt[:curr_rank, :]

            # Reconstruct matrix
            X_reconstructed = U_trunc @ S_truncated @ Vt_trunc

            # Convergence check (only on imputed values)
            delta = np.linalg.norm(prev[nan_mask] - X_reconstructed[nan_mask])
            norm = np.linalg.norm(prev[nan_mask])
            if norm > 0 and delta / norm < self.tol:
                break

            # Replace only missing values with reconstructed ones
            X_filled[nan_mask] = X_reconstructed[nan_mask]
            prev = X_filled.copy()

        return X_filled
