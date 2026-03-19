import numpy as np
import scipy.sparse.linalg as sla

class SoftImpute:
    def __init__(self, max_rank=5, shrinkage_value=None, max_iters=100, threshold=1e-5):
        self.max_rank = max_rank
        self.shrinkage_value = shrinkage_value  # Optional, will be estimated if not given
        self.max_iters = max_iters
        self.threshold = threshold

    def _converged(self, X_old, X_new, mask):
        # Compute the relative error over known entries
        delta = np.nansum((X_old[mask] - X_new[mask]) ** 2)
        old_norm = np.nansum(X_old[mask] ** 2)
        return old_norm > np.finfo(float).eps and (delta / old_norm) < self.threshold

    def _svd_step(self, X):
        # Perform truncated SVD
        U, s, Vt = sla.svds(X, k=self.max_rank)
        s = np.maximum(s - self.shrinkage_value, 0)  # Soft-thresholding
        rank = np.sum(s > 0)
        S = np.diag(s[:rank])
        return (U[:, :rank] @ S @ Vt[:rank, :]), rank

    def fit_transform(self, X):
        # X: input matrix with np.nan in missing values
        X_filled = np.where(np.isnan(X), 0, X).copy()
        mask = ~np.isnan(X)

        if self.shrinkage_value is None:
            # Estimate shrinkage value using largest singular value / 50
            u, s, vt = np.linalg.svd(X_filled, full_matrices=False)
            self.shrinkage_value = s[0] / 50.0

        for iter in range(self.max_iters):
            X_reconstructed, rank = self._svd_step(X_filled)

            if X_reconstructed is None:
                print("SVD failed.")
                break

            if self._converged(X_filled, X_reconstructed, mask):
                break

            X_filled[~mask] = X_reconstructed[~mask]

        return X_filled
