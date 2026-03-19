import numpy as np

class ST_MVL:
    def __init__(self, window_sizes=[3, 5, 7], max_neighbors=5):
        self.window_sizes = window_sizes  # Window sizes to search over
        self.max_neighbors = max_neighbors  # Max number of neighbors to use for imputation

    def _euclidean_distance(self, a, b):
        # Compute Euclidean distance, ignoring NaNs
        mask = ~np.isnan(a) & ~np.isnan(b)
        if not np.any(mask):
            return np.inf
        return np.linalg.norm(a[mask] - b[mask])

    def _get_neighbors(self, X, target_idx, window_size):
        half_window = window_size // 2
        start = max(0, target_idx - half_window)
        end = min(X.shape[0], target_idx + half_window + 1)

        neighbors = []
        for i in range(X.shape[0]):
            if i == target_idx or np.isnan(X[i, :]).any():
                continue

            i_start = max(0, i - half_window)
            i_end = min(X.shape[0], i + half_window + 1)

            if (end - start) != (i_end - i_start):
                continue

            dist = self._euclidean_distance(X[start:end], X[i_start:i_end])
            if dist != np.inf:
                neighbors.append((dist, i))

        neighbors.sort(key=lambda x: x[0])
        return neighbors[:self.max_neighbors]

    def impute(self, X):
        # X: time series matrix (2D), with NaNs as missing values
        X_filled = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.isnan(X[i, j]):
                    best_estimate = None
                    for w in self.window_sizes:
                        neighbors = self._get_neighbors(X_filled, i, w)
                        values = []
                        for _, idx in neighbors:
                            if not np.isnan(X_filled[idx, j]):
                                values.append(X_filled[idx, j])
                        if values:
                            best_estimate = np.mean(values)
                            break  # Use the smallest window with enough neighbors

                    if best_estimate is not None:
                        X_filled[i, j] = best_estimate

        return X_filled
