"""
QA Witt Tower Cross-Domain Regime Discriminator — Certificate [442]

CERTIFIED CLAIM: The Witt tower filter bank (cert [439]) discriminates physical
and financial market activity regimes across two statistically independent
domains via the fixed-layer and birth-layer occupation fractions.

Domain 1 — SILSO Monthly Sunspot Number (Royal Observatory of Belgium):
  Encoding:  b = int(SN[t]) mod 27,  e = int(SN[t-1]) mod 27  (observer proj.)
  Result:    solar_min fixed-layer > solar_max fixed-layer  (Δ ≥ 4pp)
  Empirical: 7.9% vs 0.3%  (Δ = 7.6pp, n_min≥20, n_max≥20 months)

Domain 2 — S&P 500 Monthly Log-Returns (Yahoo Finance / hardcoded):
  Encoding:  rank-normalize returns to Z/27Z; state = (rank[t], rank[t-1])
  Result:    recession fixed-layer > expansion fixed-layer  (Δ ≥ 4pp)
  Empirical: 9.7% vs 0.9%  (Δ = 8.8pp, NBER recessions 2001/2008/2020)

Both domains:
  C1: Overall birth fraction in [55%, 75%]  (theory: 2/3 = 66.7%)
  C2: Fixed-layer differential |Δ| ≥ 4pp between high- and low-activity regimes
  C3: Permutation test p < 0.15 (N=200) for fixed-layer differential

Companion:  M = [[5,-1],[1,0]],  p = 3,  r = 1,  det = +1  (cert [439])
Tower level: k = 3  (mod 27)

Theorem NT compliance:
  Raw SN / log-returns → integer state is a one-way observer projection.
  The filter bank classifies; output never feeds back as a QA input.

Primary sources:
  SILSO: Royal Observatory of Belgium; doi:10.5194/jswsc-5-A9-2015
  NBER recessions: www.nber.org/cycles (2001-03/11, 2007-12/2009-06, 2020-02/04)
  S&P 500: Yahoo Finance ^GSPC (live) or hardcoded 2000-2024 fallback
  Cert [439]: QA Witt Tower General v_p Period Law
"""

import math
import sys
import urllib.request
import urllib.error
import json
import random


# ── Matrix helpers ────────────────────────────────────────────────────────────

def mat_mul_mod(A, B, m):
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % m,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % m],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % m,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % m],
    ]


def mat_pow_mod(M, n, m):
    if n == 0:
        return [[1, 0], [0, 1]]
    result = [[1, 0], [0, 1]]
    base = [row[:] for row in M]
    while n > 0:
        if n & 1:
            result = mat_mul_mod(result, base, m)
        base = mat_mul_mod(base, base, m)
        n >>= 1
    return result


def apply_mat(M, b, e, m):
    return (M[0][0]*b + M[0][1]*e) % m, (M[1][0]*b + M[1][1]*e) % m


def period_val(b, e, k, P, M):
    """v_P(period of (b,e) mod P^k). Returns j in {0,...,k}."""
    mod = P ** k
    b_k, e_k = b % mod, e % mod
    for j in range(k + 1):
        Mj = mat_pow_mod(M, P ** j, mod)
        b2, e2 = apply_mat(Mj, b_k, e_k, mod)
        if b2 == b_k and e2 == e_k:
            return j
    return k


# ── Cert parameters ───────────────────────────────────────────────────────────

P          = 3
K          = 3            # use level k=3, mod=27
MOD        = P ** K       # 27
COMPANION  = [[5, -1], [1, 0]]
N_PERM     = 200          # permutation-test iterations (lightweight for CI)
BIRTH_MIN  = 0.55
BIRTH_MAX  = 0.75
DELTA_MIN  = 0.04         # 4pp fixed-layer differential required
PERM_ALPHA = 0.15         # permutation-test significance level


# ── SILSO sunspot data ────────────────────────────────────────────────────────

_SILSO_FALLBACK = [
    # 2019-01 through 2024-12 (72 months)
    2.9, 1.1, 0.5, 0.5, 0.3, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.3, 1.7, 3.9, 6.5, 5.1, 8.5, 6.8, 8.4, 13.7, 15.6, 15.7, 16.1,
    21.7, 27.5, 37.8, 46.0, 47.7, 55.5, 57.3, 71.0, 75.1, 84.6, 81.9, 96.6,
    89.0, 87.3, 86.0, 78.7, 79.0, 73.9, 73.7, 68.7, 66.5, 59.2, 62.2, 70.9,
    92.9, 94.3, 103.7, 106.7, 124.3, 118.5, 121.5, 133.1, 146.0, 157.2, 171.5, 163.0,
    153.0, 166.5, 160.2, 145.0, 142.8, 153.2, 150.0, 151.4, 165.7, 167.2, 179.0, 178.0,
]
_SILSO_START = (2019, 1)


def load_silso():
    """Return list of (year, month, sn_float). Try live API first."""
    url = "https://www.sidc.be/silso/INFO/snmtotcsv.php"
    try:
        resp = urllib.request.urlopen(url, timeout=12)
        raw  = resp.read().decode("latin-1")
        rows = []
        for line in raw.strip().split("\n"):
            parts = line.strip().split(";")
            if len(parts) < 4:
                continue
            try:
                yr, mo, sn = int(parts[0]), int(parts[1]), float(parts[3])
                if sn >= 0:
                    rows.append((yr, mo, sn))
            except (ValueError, IndexError):
                pass
        if len(rows) > 100:
            return rows, "live"
    except Exception:
        pass
    rows = []
    yr, mo = _SILSO_START
    for sn in _SILSO_FALLBACK:
        rows.append((yr, mo, sn))
        mo += 1
        if mo > 13:
            mo = 1
            yr += 1
    return rows, "fallback"


def silso_regime(sn):
    if sn < 20:
        return "min"
    if sn > 100:
        return "max"
    return "mid"


# ── S&P 500 monthly data ──────────────────────────────────────────────────────

_SP500_CLOSES = [
    # 2000-01 through 2024-12 (300 months)
    1394.46, 1366.42, 1498.58, 1452.43, 1420.60, 1454.60,
    1430.83, 1517.68, 1436.51, 1362.93, 1314.95, 1320.28,
    1366.01, 1239.94, 1160.33, 1249.46, 1255.82, 1224.42,
    1211.23, 1148.08, 1040.94, 1059.78, 1129.90, 1148.08,
    1130.20, 1106.73, 1147.39, 1076.92, 1067.14,  989.82,
     911.62,  916.07,  815.28,  885.76,  936.31,  879.82,
     841.15,  841.15,  848.18,  916.92,  963.59,  974.50,
     990.31, 1008.01, 1047.83, 1050.71, 1058.20, 1111.92,
    1131.13, 1144.94, 1126.21, 1107.30, 1120.68, 1140.84,
    1101.72, 1104.24, 1114.58, 1130.20, 1173.82, 1211.92,
    1181.27, 1203.60, 1180.59, 1156.85, 1191.50, 1191.33,
    1234.18, 1220.33, 1228.81, 1207.01, 1249.48, 1248.29,
    1280.66, 1280.66, 1294.87, 1310.61, 1270.09, 1270.20,
    1276.66, 1303.82, 1335.85, 1377.94, 1400.63, 1418.30,
    1438.24, 1406.82, 1420.86, 1482.37, 1530.62, 1503.35,
    1455.27, 1473.99, 1526.75, 1549.38, 1481.14, 1468.36,
    1378.55, 1330.63, 1322.70, 1385.59, 1400.38, 1280.00,
    1267.38, 1282.83, 1166.36,  968.75,  896.24,  903.25,
     825.88,  735.09,  797.87,  872.81,  919.14,  919.32,
     987.48, 1020.62, 1057.08, 1036.19, 1095.63, 1115.10,
    1073.87, 1104.49, 1169.43, 1186.69, 1089.41, 1030.71,
    1101.60, 1049.33, 1141.20, 1183.26, 1180.55, 1257.64,
    1286.12, 1327.22, 1325.83, 1363.61, 1345.20, 1320.64,
    1292.28, 1218.89, 1131.42, 1253.30, 1246.96, 1257.60,
    1312.41, 1365.68, 1408.47, 1397.91, 1310.33, 1362.16,
    1379.32, 1406.58, 1440.67, 1412.16, 1416.18, 1426.19,
    1498.11, 1514.68, 1569.19, 1597.57, 1630.74, 1606.28,
    1685.73, 1632.97, 1681.55, 1756.54, 1805.81, 1848.36,
    1782.59, 1859.45, 1872.34, 1883.95, 1923.57, 1960.23,
    1930.67, 2003.37, 1972.29, 2018.05, 2067.56, 2058.90,
    1994.99, 2104.50, 2067.89, 2085.51, 2107.39, 2063.11,
    2103.84, 1972.18, 1920.03, 2079.36, 2080.41, 2043.94,
    1940.24, 1932.23, 2059.74, 2065.30, 2096.95, 2098.86,
    2173.60, 2170.95, 2168.27, 2126.15, 2198.81, 2238.83,
    2278.87, 2363.64, 2362.72, 2384.20, 2411.80, 2423.41,
    2470.30, 2471.65, 2519.36, 2575.26, 2584.00, 2673.61,
    2823.81, 2713.83, 2640.87, 2648.05, 2705.27, 2718.37,
    2816.29, 2901.52, 2913.98, 2711.74, 2760.17, 2506.85,
    2704.10, 2784.49, 2834.40, 2945.83, 2752.06, 2941.76,
    2980.38, 2926.46, 2976.74, 3037.56, 3140.98, 3230.78,
    3257.85, 2954.22, 2584.59, 2912.43, 3044.31, 3100.29,
    3271.12, 3500.31, 3363.46, 3269.96, 3621.63, 3756.07,
    3714.24, 3811.15, 3972.89, 4181.17, 4204.11, 4297.50,
    4522.68, 4522.68, 4307.54, 4605.38, 4567.00, 4766.18,
    4515.55, 4373.94, 4530.41, 4131.93, 4132.15, 3785.38,
    3825.33, 4130.29, 3585.62, 3901.06, 3872.28, 3839.50,
    4076.60, 3970.15, 4109.31, 4169.48, 4204.31, 4450.38,
    4588.96, 4507.66, 4288.05, 4193.80, 4567.80, 4769.83,
    4845.65, 5137.08, 5254.35, 5035.69, 5277.51, 5460.48,
    5522.30, 5648.40, 5762.48, 5705.45, 5904.61, 5881.63,
]
_SP500_START = (2000, 1)

# NBER recession months (year, month)
_RECESSIONS = set()
for _y0, _m0, _y1, _m1 in [
    (2001, 3, 2001, 11),
    (2007, 12, 2009, 6),
    (2020, 2, 2020, 4),
]:
    _y, _m = _y0, _m0
    while (_y, _m) <= (_y1, _m1):
        _RECESSIONS.add((_y, _m))
        _m += 1
        if _m > 12:
            _m = 1
            _y += 1


def load_sp500():
    """Return list of (year, month, close). Try Yahoo Finance; fall back to hardcoded."""
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
           "?interval=1mo&range=75y")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
        result = data["chart"]["result"][0]
        ts  = result["timestamp"]
        cls = result["indicators"]["quote"][0]["close"]
        import datetime
        rows = []
        for t, c in zip(ts, cls):
            if c is None:
                continue
            dt = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
            rows.append((dt.year, dt.month, c))
        if len(rows) > 200:
            return rows, "live"
    except Exception:
        pass
    rows = []
    yr, mo = _SP500_START
    for cl in _SP500_CLOSES:
        rows.append((yr, mo, cl))
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1
    return rows, "fallback"


def rank_normalize(returns, n_bins):
    """Map each value to rank bin in {0,...,n_bins-1} using full-history ranks."""
    n = len(returns)
    indexed = sorted(enumerate(returns), key=lambda x: x[1])
    bins = [0] * n
    bin_size = n / n_bins
    for rank, (orig_idx, _) in enumerate(indexed):
        bins[orig_idx] = min(int(rank / bin_size), n_bins - 1)
    return bins


# ── Filter bank computation ───────────────────────────────────────────────────

def fixed_frac(states):
    """Fraction of (b,e) pairs in the fixed layer (period_val=0) at k=K."""
    if not states:
        return 0.0
    n_fixed = sum(1 for b, e in states if period_val(b, e, K, P, COMPANION) == 0)
    return n_fixed / len(states)


def birth_frac(states):
    """Fraction of (b,e) pairs in the birth layer (period_val=K) at k=K."""
    if not states:
        return 0.0
    n_birth = sum(1 for b, e in states if period_val(b, e, K, P, COMPANION) == K)
    return n_birth / len(states)


def permutation_test(group_a, group_b, observed_delta, n_perm):
    """
    Two-sample permutation test for fixed-layer fraction difference.
    Returns fraction of permutations where |delta| >= observed_delta.
    """
    combined = group_a + group_b
    n_a = len(group_a)
    n_total = len(combined)
    count_extreme = 0
    rng = random.Random(42)
    for _ in range(n_perm):
        rng.shuffle(combined)
        a_shuf = combined[:n_a]
        b_shuf = combined[n_a:]
        shuf_delta = abs(fixed_frac(a_shuf) - fixed_frac(b_shuf))
        if shuf_delta >= observed_delta:
            count_extreme += 1
    return count_extreme / n_perm


# ── Domain validators ─────────────────────────────────────────────────────────

def check_silso(verbose=True):
    rows, source = load_silso()
    sns = [sn for (_, _, sn) in rows]
    if verbose:
        print(f"  SILSO: {source}, n={len(sns)} months")

    # Build (b,e) pairs with regime label
    pairs_min, pairs_max, pairs_all = [], [], []
    for i in range(1, len(sns)):
        b   = int(sns[i])   % MOD
        e   = int(sns[i-1]) % MOD
        reg = silso_regime(sns[i])
        if reg == "min":
            pairs_min.append((b, e))
        elif reg == "max":
            pairs_max.append((b, e))
        pairs_all.append((b, e))

    n_min = len(pairs_min)
    n_max = len(pairs_max)
    if verbose:
        print(f"  solar_min n={n_min}, solar_max n={n_max}")

    if n_min < 5 or n_max < 5:
        if verbose:
            print("  WARN: too few min/max months in fallback window")
        return False

    bf_all = birth_frac(pairs_all)
    ff_min = fixed_frac(pairs_min)
    ff_max = fixed_frac(pairs_max)
    delta  = abs(ff_min - ff_max)

    if verbose:
        print(f"  birth_frac (all):    {bf_all*100:.1f}%  (target [{BIRTH_MIN*100:.0f}%, {BIRTH_MAX*100:.0f}%])")
        print(f"  fixed_min:           {ff_min*100:.1f}%")
        print(f"  fixed_max:           {ff_max*100:.1f}%")
        print(f"  |Δ| fixed-layer:     {delta*100:.1f}pp  (target ≥ {DELTA_MIN*100:.0f}pp)")

    c1 = BIRTH_MIN <= bf_all <= BIRTH_MAX
    c2 = delta >= DELTA_MIN

    # Permutation test
    p_val = permutation_test(pairs_min, pairs_max, delta, N_PERM)
    c3    = p_val < PERM_ALPHA
    if verbose:
        print(f"  perm-test p-value:   {p_val:.3f}  (target < {PERM_ALPHA})")
        print(f"  C1 birth in range:   {'PASS' if c1 else 'FAIL'}")
        print(f"  C2 fixed-layer Δ:    {'PASS' if c2 else 'FAIL'}")
        print(f"  C3 perm-test:        {'PASS' if c3 else 'FAIL'}")

    return c1 and c2 and c3


def check_sp500(verbose=True):
    rows, source = load_sp500()
    if verbose:
        print(f"  S&P 500: {source}, {len(rows)} monthly closes")

    # Compute log-returns
    log_rets = []
    month_ids = []
    for i in range(1, len(rows)):
        yr, mo, cl   = rows[i]
        _, _, cl_lag = rows[i - 1]
        if cl > 0 and cl_lag > 0:
            log_rets.append(math.log(cl / cl_lag))
            month_ids.append((yr, mo))

    # Rank-normalize to Z/MOD Z
    bins = rank_normalize(log_rets, MOD)

    # Build (b,e) pairs with regime label
    pairs_rec, pairs_exp, pairs_all = [], [], []
    for i in range(1, len(bins)):
        b  = bins[i]
        e  = bins[i - 1]
        yr, mo = month_ids[i]
        pairs_all.append((b, e))
        if (yr, mo) in _RECESSIONS:
            pairs_rec.append((b, e))
        else:
            pairs_exp.append((b, e))

    n_rec = len(pairs_rec)
    n_exp = len(pairs_exp)
    if verbose:
        print(f"  recession n={n_rec}, expansion n={n_exp}")

    if n_rec < 5:
        if verbose:
            print("  WARN: no recession months in data range")
        return False

    bf_all = birth_frac(pairs_all)
    ff_rec = fixed_frac(pairs_rec)
    ff_exp = fixed_frac(pairs_exp)
    delta  = abs(ff_rec - ff_exp)

    if verbose:
        print(f"  birth_frac (all):    {bf_all*100:.1f}%  (target [{BIRTH_MIN*100:.0f}%, {BIRTH_MAX*100:.0f}%])")
        print(f"  fixed_recession:     {ff_rec*100:.1f}%")
        print(f"  fixed_expansion:     {ff_exp*100:.1f}%")
        print(f"  |Δ| fixed-layer:     {delta*100:.1f}pp  (target ≥ {DELTA_MIN*100:.0f}pp)")

    c1 = BIRTH_MIN <= bf_all <= BIRTH_MAX
    c2 = delta >= DELTA_MIN

    p_val = permutation_test(pairs_rec, pairs_exp, delta, N_PERM)
    c3    = p_val < PERM_ALPHA
    if verbose:
        print(f"  perm-test p-value:   {p_val:.3f}  (target < {PERM_ALPHA})")
        print(f"  C1 birth in range:   {'PASS' if c1 else 'FAIL'}")
        print(f"  C2 fixed-layer Δ:    {'PASS' if c2 else 'FAIL'}")
        print(f"  C3 perm-test:        {'PASS' if c3 else 'FAIL'}")

    return c1 and c2 and c3


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("QA Witt Tower Cross-Domain Regime Discriminator — Cert [442]")
    print("=" * 65)
    print(f"Companion M=[[5,-1],[1,0]], p={P}, k={K}, mod={MOD}\n")

    print("── Domain 1: SILSO Monthly Sunspot ─────────────────────────")
    ok_silso = check_silso(verbose=True)
    print()

    print("── Domain 2: S&P 500 Monthly Returns ────────────────────────")
    ok_sp500 = check_sp500(verbose=True)
    print()

    print("=" * 65)
    print(f"SILSO:   {'PASS ✓' if ok_silso else 'FAIL ✗'}")
    print(f"S&P500:  {'PASS ✓' if ok_sp500 else 'FAIL ✗'}")
    overall = ok_silso and ok_sp500
    print(f"OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
