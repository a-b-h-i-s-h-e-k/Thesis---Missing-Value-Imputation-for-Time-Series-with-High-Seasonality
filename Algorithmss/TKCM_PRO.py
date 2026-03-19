import numpy as np

def mod(x, y):
    return ((x % y) + y) % y

class OptsPro:
    def __init__(self, k, l, d, L, offset, ts, ref_ts):
        self.k = k
        self.l = l
        self.d = d
        self.L = L
        self.offset = offset
        self.ts = ts
        self.ref_ts = ref_ts

def tkcm_pro(opts):
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

    # Step 3: impute by summing ref_ts[i][pos] across all i, then average over k
    sum_vals = 0
    for i in range(k):
        pos = offset + l + A[i] - 1
        pos_mod = mod(pos, L)
        summed_value = np.sum(ref_ts[:, pos_mod])
        sum_vals += summed_value

    imputed_value = sum_vals / k
    ts[offset] = imputed_value
    return imputed_value

# ✅ TKCM_PRO Class (External API)
class TKCM_PRO:
    def __init__(self, k=5, l=4):
        self.k = k
        self.l = l

    def impute(self, data, offset=None, ref_ts=None):
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

        opts = OptsPro(
            k=self.k,
            l=self.l,
            d=d,
            L=L,
            offset=offset,
            ts=ts_filled,
            ref_ts=ref_ts
        )

        _ = tkcm_pro(opts)
        return ts_filled
