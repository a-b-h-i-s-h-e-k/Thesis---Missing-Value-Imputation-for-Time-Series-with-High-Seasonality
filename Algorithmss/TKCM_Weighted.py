import numpy as np

def mod(x, y):
    return ((x % y) + y) % y

class Opts:
    def __init__(self, k, l, d, L, offset, ts, ref_ts):
        self.k = k
        self.l = l
        self.d = d
        self.L = L
        self.offset = offset
        self.ts = ts
        self.ref_ts = ref_ts

def weighted_tkcm(opts):
    k, l, d, L, offset = opts.k, opts.l, opts.d, opts.L, opts.offset
    ts = opts.ts
    ref_ts = opts.ref_ts

    nr_patterns = L - 2 * l + 1
    M = np.full((k + 1, nr_patterns + 1), np.inf)
    D = np.zeros(nr_patterns + 1)
    A = np.zeros(k, dtype=int)

    # Step 1: compute pattern dissimilarities
    for j in range(1, nr_patterns + 1):
        dist = 0
        for i in range(d):
            for x in range(l):
                pos1 = offset + l + j - 1 - x
                pos2 = offset - x
                x1 = ref_ts[i][mod(pos1, L)]
                x2 = ref_ts[i][mod(pos2, L)]
                dist += (x1 - x2) ** 2
        D[j] = np.sqrt(dist)

    # Step 2.1: dynamic programming
    M[0, :] = 0
    for i in range(1, k + 1):
        for j in range(nr_patterns + 1):
            if i > j:
                M[i, j] = np.inf
            else:
                pred = max(j - l, 0)
                M[i, j] = min(M[i, j - 1] if j > 0 else np.inf, D[j] + M[i - 1, pred])

    # Step 2.2: backtracking to select top-k matches
    i = k
    j = nr_patterns
    while i > 0 and j >= 0:
        if j > 0 and M[i, j] == M[i, j - 1]:
            j -= 1
        else:
            A[i - 1] = j
            i -= 1
            j = max(j - l, 0)

    # Step 3: impute using weighted average
    distances = []
    values = []

    for i in range(k):
        pos = offset + l + A[i] - 1
        val = ts[mod(pos, L)]
        dist = D[A[i]] + 1e-8  # small epsilon to avoid division by zero
        values.append(val)
        distances.append(dist)

    distances = np.array(distances)
    values = np.array(values)
    weights = 1 / distances
    weighted_avg = np.dot(weights, values) / np.sum(weights)

    ts[offset] = weighted_avg
    return weighted_avg


# ✅ External-friendly wrapper class for direct use
class TKCM_Weighted:
    def __init__(self, k=5, l=4):
        self.k = k
        self.l = l

    def impute(self, data, offset=None, ref_ts=None):
        """
        Parameters:
        - data: 1D np.array with a single np.nan
        - ref_ts: 2D np.array (d x L), the reference series
        - offset: index of the missing value (optional, auto-detected if None)
        
        Returns:
        - A full series with imputed value (1D np.array)
        """
        if offset is None:
            missing = np.where(np.isnan(data))[0]
            if len(missing) == 0:
                raise ValueError("No missing value (NaN) found in the series.")
            offset = missing[0]

        if ref_ts is None:
            raise ValueError("You must provide ref_ts (reference series).")

        d, L = ref_ts.shape
        ts_filled = data.copy()
        ts_filled[np.isnan(ts_filled)] = 0

        opts = Opts(
            k=self.k,
            l=self.l,
            d=d,
            L=L,
            offset=offset,
            ts=ts_filled,
            ref_ts=ref_ts
        )

        _ = weighted_tkcm(opts)
        return ts_filled







