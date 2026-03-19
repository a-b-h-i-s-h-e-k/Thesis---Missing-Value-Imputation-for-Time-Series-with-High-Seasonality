import numpy as np

def mod(x, y):
    return ((x % y) + y) % y

class OptsPlus:
    def __init__(self, k, l, d, L, offset, ts, ref_ts, alpha=0.5, beta=0.5):
        self.k = k
        self.l = l
        self.d = d
        self.L = L
        self.offset = offset
        self.ts = ts              # 1D target time series (with missing value)
        self.ref_ts = ref_ts      # 2D reference time series (d x L)
        self.alpha = alpha        # Weight for reference similarity
        self.beta = beta          # Weight for target similarity

def tkcm_plus(opts):
    k, l, d, L, offset = opts.k, opts.l, opts.d, opts.L, opts.offset
    ts = opts.ts
    ref_ts = opts.ref_ts
    alpha, beta = opts.alpha, opts.beta

    nr_patterns = L - 2 * l + 1
    M = np.full((k + 1, nr_patterns + 1), np.inf)
    D = np.zeros(nr_patterns + 1)
    A = np.zeros(k, dtype=int)

    # Step 1: Compute hybrid dissimilarities (reference + target)
    for j in range(1, nr_patterns + 1):
        dist_ref = 0
        for i in range(d):
            for x in range(l):
                pos1 = offset + l + j - 1 - x
                pos2 = offset - x
                x1 = ref_ts[i][mod(pos1, L)]
                x2 = ref_ts[i][mod(pos2, L)]
                dist_ref += (x1 - x2) ** 2
        dist_ref = np.sqrt(dist_ref)

        # NEW: distance to same target series (using past values only)
        dist_target = 0
        for x in range(l):
            pos1 = offset + l + j - 1 - x
            pos2 = offset - x
            x1 = ts[mod(pos1, L)]
            x2 = ts[mod(pos2, L)]
            dist_target += (x1 - x2) ** 2
        dist_target = np.sqrt(dist_target)

        # Combine with weights
        D[j] = alpha * dist_ref + beta * dist_target

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

    # Step 3: Impute missing value from selected matches
    sum_vals = 0
    for i in range(k):
        pos = offset + l + A[i] - 1
        sum_vals += ts[mod(pos, L)]

    imputed_value = sum_vals / k
    ts[offset] = imputed_value
    return imputed_value

# ✅ TKCM_PLUS Class (External API)
class TKCM_PLUS:
    def __init__(self, k=5, l=4, alpha=0.5, beta=0.5):
        self.k = k
        self.l = l
        self.alpha = alpha
        self.beta = beta

    def impute(self, data, offset=None, ref_ts=None):
        """
        Impute a missing value using enhanced TKCM_PLUS logic:
        - Combines similarity with both reference series and target's past pattern.

        Parameters:
        - data: 1D np.array with one NaN value
        - ref_ts: 2D np.array (d x L)
        - offset: (optional) index of the missing value

        Returns:
        - 1D np.array with imputed value filled
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
        ts_filled[np.isnan(ts_filled)] = 0  # temporary fill

        opts = OptsPlus(
            k=self.k,
            l=self.l,
            d=d,
            L=L,
            offset=offset,
            ts=ts_filled,
            ref_ts=ref_ts,
            alpha=self.alpha,
            beta=self.beta
        )

        _ = tkcm_plus(opts)
        return ts_filled
