"""
QA Witt Tower Safe-Haven Null — Certificate [443]

CERTIFIED CLAIM: The Witt tower filter bank (cert [439]) produces a certified null
for safe-haven assets (Gold): recession fixed-layer elevation is absent (≤ 2%,
permutation p > 0.15) because safe-haven crisis returns do not cluster at the
fixed-point states of the companion matrix.

Geometric ground truth:
  M = [[5,-1],[1,0]] mod 27 has exactly 3 fixed-point states: {(0,0),(9,9),(18,18)}.
  These require b ≡ 0 mod 9 AND e ≡ 0 mod 9.
  Crisis-sell-off assets (GSPC) land at rank bin 0-3 on consecutive months → near (0,0).
  Safe-haven assets (Gold) either rise (upper bins) or oscillate (mid-range), so
  consecutive return pairs systematically avoid the three fixed-point bins.

Domain — Gold (GC=F) Monthly Log-Returns (~2000–2026):
  Encoding:  rank-normalize log-returns to Z/27Z; state = (rank[t], rank[t-1])
  C1: Overall birth fraction ∈ [55%, 75%]  (theory: 2/3 = 66.7%)
  C2: Recession fixed-layer ≤ 2%           (null: no elevation)
  C3: Permutation test p > 0.15            (null NOT rejected)
  C4: Fixed-point locus = {(0,0),(9,9),(18,18)} — exhaustive algebraic check
  C5: GFC (2007-12/2009-06) mean Gold rank bin > 12  (above mid, away from fixed pts)

Companion:  M = [[5,-1],[1,0]],  p = 3,  r = 1,  det = +1  (cert [439])
Tower level: k = 3  (mod 27)

Theorem NT compliance:
  Log-returns → integer rank bins is a one-way observer projection.
  The filter bank classifies; output never feeds back as a QA input.

Primary sources:
  NBER: www.nber.org/cycles (2001-03/11, 2007-12/2009-06, 2020-02/04)
  Gold: Yahoo Finance GC=F (live) or hardcoded 2000-2024 fallback
  Cert [439]: QA Witt Tower General v_p Period Law
  Cert [442]: QA Witt Tower Cross-Domain Regime Discriminator (positive control)
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
K          = 3            # level k=3, mod=27
MOD        = P ** K       # 27
COMPANION  = [[5, -1], [1, 0]]
N_PERM     = 200
BIRTH_MIN  = 0.55
BIRTH_MAX  = 0.75
FIXED_MAX  = 0.02         # recession fixed-layer ≤ 2% (null)
PERM_NULL  = 0.15         # permutation p must NOT be below this for null
GFC_START  = (2007, 12)
GFC_END    = (2009, 6)


# ── Gold monthly closes (2000-01 to 2024-12, hardcoded fallback) ──────────────

_GOLD_CLOSES = [
    # 2000
    285.6, 295.4, 285.0, 275.8, 272.5, 285.7, 282.4, 274.5, 276.3, 270.2, 266.3, 273.6,
    # 2001
    265.5, 261.9, 263.0, 260.6, 270.4, 270.3, 267.2, 272.4, 293.8, 283.0, 276.5, 276.4,
    # 2002
    281.5, 296.8, 296.0, 302.9, 314.5, 321.2, 313.3, 310.5, 323.8, 317.2, 319.0, 345.0,
    # 2003
    367.4, 359.3, 340.5, 329.0, 364.5, 356.0, 353.6, 376.5, 389.2, 379.3, 391.0, 415.3,
    # 2004
    414.8, 402.0, 407.3, 403.5, 384.5, 393.3, 391.2, 402.5, 406.5, 421.3, 442.0, 442.0,
    # 2005
    424.2, 422.7, 427.5, 436.5, 422.5, 441.0, 425.5, 437.5, 473.5, 469.5, 476.5, 513.0,
    # 2006
    549.0, 555.5, 582.5, 625.5, 676.5, 613.5, 633.5, 631.5, 599.5, 604.5, 627.5, 636.5,
    # 2007
    632.0, 665.5, 654.0, 680.5, 657.5, 655.5, 665.5, 665.5, 730.5, 748.5, 806.5, 838.0,
    # 2008
    920.3, 922.3, 1011.3, 871.0, 888.7, 930.3, 939.7, 844.5, 891.0, 730.0, 760.3, 870.0,
    # 2009
    919.5, 942.8, 924.3, 871.3, 975.5, 945.5, 934.3, 953.3, 994.5, 1044.3, 1127.0, 1087.5,
    # 2010
    1118.0, 1106.5, 1113.5, 1180.3, 1207.5, 1244.5, 1168.5, 1215.5, 1308.5, 1340.5, 1364.0, 1421.5,
    # 2011
    1320.0, 1408.5, 1426.5, 1530.0, 1537.5, 1502.5, 1630.5, 1827.5, 1899.0, 1620.5, 1751.5, 1563.0,
    # 2012
    1740.5, 1714.0, 1663.5, 1652.5, 1563.5, 1599.0, 1622.5, 1623.5, 1776.5, 1722.5, 1729.5, 1657.0,
    # 2013
    1663.5, 1579.0, 1595.5, 1453.0, 1393.5, 1233.0, 1325.5, 1412.5, 1325.5, 1323.5, 1253.5, 1202.0,
    # 2014
    1244.0, 1325.0, 1293.5, 1294.0, 1244.0, 1325.5, 1307.0, 1286.5, 1220.5, 1230.5, 1178.5, 1183.0,
    # 2015
    1284.5, 1213.0, 1183.5, 1178.5, 1192.5, 1173.5, 1099.5, 1136.5, 1115.5, 1142.5, 1062.0, 1062.0,
    # 2016
    1098.5, 1233.5, 1238.5, 1289.5, 1217.5, 1323.5, 1349.5, 1312.5, 1317.5, 1275.5, 1177.5, 1152.0,
    # 2017
    1210.5, 1255.5, 1248.5, 1268.5, 1266.5, 1242.5, 1258.5, 1323.5, 1318.5, 1273.5, 1276.5, 1302.5,
    # 2018
    1303.0, 1318.5, 1324.5, 1317.5, 1304.5, 1252.5, 1226.5, 1205.5, 1202.5, 1215.5, 1222.5, 1278.5,
    # 2019
    1321.5, 1317.5, 1294.5, 1284.5, 1285.5, 1409.5, 1425.5, 1530.5, 1485.5, 1491.5, 1462.5, 1523.5,
    # 2020
    1590.5, 1584.5, 1577.5, 1688.5, 1728.5, 1800.5, 1975.5, 1969.5, 1880.5, 1879.5, 1776.5, 1898.5,
    # 2021
    1847.5, 1776.5, 1707.5, 1779.5, 1905.5, 1770.5, 1817.5, 1815.5, 1755.5, 1795.5, 1785.5, 1829.5,
    # 2022
    1796.5, 1909.5, 1937.5, 1897.5, 1848.5, 1808.5, 1753.5, 1745.5, 1660.5, 1634.5, 1750.5, 1802.5,
    # 2023
    1949.5, 1827.5, 1968.5, 2000.5, 1983.5, 1912.5, 1963.5, 1966.5, 1870.5, 1998.5, 2035.5, 2063.5,
    # 2024
    2040.5, 2045.5, 2230.5, 2286.5, 2346.5, 2330.5, 2448.5, 2503.5, 2635.5, 2745.5, 2673.5, 2637.5,
]
_GOLD_START = (2000, 1)

# NBER recession months
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


def load_gold():
    """Return list of (year, month, close). Try Yahoo Finance; fall back to hardcoded."""
    import urllib.parse
    ticker = "GC=F"
    enc = urllib.parse.quote(ticker, safe="")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
           "?interval=1mo&range=25y")
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
        if len(rows) > 100:
            return rows, "live"
    except Exception:
        pass
    rows = []
    yr, mo = _GOLD_START
    for cl in _GOLD_CLOSES:
        rows.append((yr, mo, cl))
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1
    return rows, "fallback"


# ── Filter bank helpers ───────────────────────────────────────────────────────

def rank_normalize(returns, n_bins):
    n = len(returns)
    indexed = sorted(enumerate(returns), key=lambda x: x[1])
    bins = [0] * n
    bin_size = n / n_bins
    for rank, (orig_idx, _) in enumerate(indexed):
        bins[orig_idx] = min(int(rank / bin_size), n_bins - 1)
    return bins


def fixed_frac(states):
    if not states:
        return 0.0
    n_fixed = sum(1 for b, e in states if period_val(b, e, K, P, COMPANION) == 0)
    return n_fixed / len(states)


def birth_frac(states):
    if not states:
        return 0.0
    n_birth = sum(1 for b, e in states if period_val(b, e, K, P, COMPANION) == K)
    return n_birth / len(states)


def permutation_test(group_a, group_b, observed_delta, n_perm):
    combined = group_a + group_b
    n_a = len(group_a)
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


# ── C4: Geometric invariant — fixed-point locus ───────────────────────────────

def compute_fixed_points(companion, mod):
    """Exhaustive: find all (b,e) in Z/mod Z with M(b,e) = (b,e)."""
    fps = []
    for b in range(mod):
        for e in range(mod):
            b2, e2 = apply_mat(companion, b, e, mod)
            if b2 == b and e2 == e:
                fps.append((b, e))
    return fps


# ── Main validator ────────────────────────────────────────────────────────────

def check_gold(verbose=True):
    rows, source = load_gold()
    if verbose:
        print(f"  Gold (GC=F): {source}, {len(rows)} monthly closes")

    log_rets = []
    month_ids = []
    for i in range(1, len(rows)):
        yr, mo, cl   = rows[i]
        _, _, cl_lag = rows[i - 1]
        if cl > 0 and cl_lag > 0:
            log_rets.append(math.log(cl / cl_lag))
            month_ids.append((yr, mo))

    bins = rank_normalize(log_rets, MOD)

    pairs_rec, pairs_exp, pairs_all = [], [], []
    pairs_gfc = []
    bins_gfc  = []
    for i in range(1, len(bins)):
        b  = bins[i]
        e  = bins[i - 1]
        yr, mo = month_ids[i]
        pairs_all.append((b, e))
        if (yr, mo) in _RECESSIONS:
            pairs_rec.append((b, e))
        else:
            pairs_exp.append((b, e))
        if GFC_START <= (yr, mo) <= GFC_END:
            pairs_gfc.append((b, e))
            bins_gfc.append(b)

    n_rec = len(pairs_rec)
    n_gfc = len(pairs_gfc)
    if verbose:
        print(f"  recession n={n_rec}, expansion n={len(pairs_exp)}")
        print(f"  GFC window ({GFC_START[0]}-{GFC_START[1]:02d} to "
              f"{GFC_END[0]}-{GFC_END[1]:02d}) n={n_gfc}")

    if n_rec < 5:
        if verbose:
            print("  WARN: too few recession months in data range")
        return False

    bf_all = birth_frac(pairs_all)
    ff_rec = fixed_frac(pairs_rec)
    ff_exp = fixed_frac(pairs_exp)
    delta  = abs(ff_rec - ff_exp)

    # C5: GFC mean rank bin
    gfc_mean_bin = sum(bins_gfc) / len(bins_gfc) if bins_gfc else 0.0

    if verbose:
        print(f"  birth_frac (all):    {bf_all*100:.1f}%  "
              f"(target [{BIRTH_MIN*100:.0f}%, {BIRTH_MAX*100:.0f}%])")
        print(f"  fixed_recession:     {ff_rec*100:.1f}%  (target ≤ {FIXED_MAX*100:.0f}%)")
        print(f"  fixed_expansion:     {ff_exp*100:.1f}%")
        print(f"  |Δ| fixed-layer:     {delta*100:.1f}pp")

    c1 = BIRTH_MIN <= bf_all <= BIRTH_MAX
    c2 = ff_rec <= FIXED_MAX

    p_val = permutation_test(pairs_rec, pairs_exp, delta, N_PERM)
    c3    = p_val >= PERM_NULL   # null: p is NOT below threshold

    if verbose:
        print(f"  perm-test p-value:   {p_val:.3f}  (target ≥ {PERM_NULL} for null)")
        print(f"  GFC mean rank bin:   {gfc_mean_bin:.1f}  (target > 12, mid of Z/27Z)")

    c5 = gfc_mean_bin > 12.0

    if verbose:
        print(f"  C1 birth in range:   {'PASS' if c1 else 'FAIL'}")
        print(f"  C2 rec fixed ≤ 2%:   {'PASS' if c2 else 'FAIL'}")
        print(f"  C3 perm null (≥0.15):{'PASS' if c3 else 'FAIL'}")
        print(f"  C5 GFC bin > 12:     {'PASS' if c5 else 'FAIL'}")

    return c1 and c2 and c3 and c5


def check_geometric(verbose=True):
    """C4: verify fixed-point locus of M mod 27 is exactly {(0,0),(9,9),(18,18)}."""
    fps = compute_fixed_points(COMPANION, MOD)
    expected = sorted([(0, 0), (9, 9), (18, 18)])
    ok = sorted(fps) == expected

    if verbose:
        print(f"  Fixed-point states: {fps}")
        print(f"  Expected:           {expected}")
        print(f"  C4 locus exact:     {'PASS' if ok else 'FAIL'}")
        if ok:
            print(f"  Algebraic reason: M·(b,e)=(b,e) mod 27 ⟹ 4b≡0 mod 27 ⟹ b≡0 mod 9;")
            print(f"  and b≡e mod 27 ⟹ e≡0 mod 9. Only 3 solutions in Z/27Z.")

    return ok


def main():
    print("QA Witt Tower Safe-Haven Null — Cert [443]")
    print("=" * 65)
    print(f"Companion M=[[5,-1],[1,0]], p={P}, k={K}, mod={MOD}\n")

    print("── C4: Geometric invariant — fixed-point locus ──────────────")
    ok_geo = check_geometric(verbose=True)
    print()

    print("── Gold (GC=F) Monthly Returns ──────────────────────────────")
    ok_gold = check_gold(verbose=True)
    print()

    print("=" * 65)
    print(f"Geometric:  {'PASS ✓' if ok_geo  else 'FAIL ✗'}")
    print(f"Gold null:  {'PASS ✓' if ok_gold else 'FAIL ✗'}")
    overall = ok_geo and ok_gold
    print(f"OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
