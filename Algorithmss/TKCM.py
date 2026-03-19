import numpy as np

def mod(x, y):
    return ((x % y) + y) % y

class Opts:
    def __init__(self, k, l, d, L, offset, ts, ref_ts):
        self.k = k            # number of patterns to select
        self.l = l            # pattern length
        self.d = d            # number of dimensions
        self.L = L            # total length of time series
        self.offset = offset  # target missing point index
        self.ts = ts          # 1D target time series
        self.ref_ts = ref_ts  # 2D reference time series (d x L)

def tkcm(opts):
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

    # Step 2.2: backtracking
    i = k
    j = nr_patterns
    while i > 0 and j >= 0:
        if j > 0 and M[i, j] == M[i, j - 1]:
            j -= 1
        else:
            A[i - 1] = j
            i -= 1
            j = max(j - l, 0)

    # Step 3: impute missing value
    sum_vals = 0
    for i in range(k):
        pos = offset + l + A[i] - 1
        sum_vals += ts[mod(pos, L)]

    imputed_value = sum_vals / k
    ts[offset] = imputed_value
    return imputed_value

# ✅ TKCM Wrapper class (can be used externally)
class TKCM:
    def __init__(self, k=5, l=4):
        self.k = k
        self.l = l

    def impute(self, data, offset=None, ref_ts=None):
        """
        Parameters:
        - data: 1D np.array with one NaN value
        - ref_ts: 2D np.array (d x L), the reference matrix
        - offset: (optional) index of the missing value (auto-detected if None)

        Returns:
        - 1D np.array with imputed value filled in
        """
        if offset is None:
            missing = np.where(np.isnan(data))[0]
            if len(missing) == 0:
                raise ValueError("No NaN found in target series.")
            offset = missing[0]

        if ref_ts is None:
            raise ValueError("ref_ts must be provided.")

        d, L = ref_ts.shape
        ts_filled = data.copy()
        ts_filled[np.isnan(ts_filled)] = 0  # temporarily fill

        opts = Opts(
            k=self.k,
            l=self.l,
            d=d,
            L=L,
            offset=offset,
            ts=ts_filled,
            ref_ts=ref_ts
        )

        _ = tkcm(opts)
        return ts_filled
