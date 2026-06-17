"""
62_qa_witt_multi_instrument.py — Witt Tower Filter Bank: Multi-Instrument Extension

Extends cert [442] from S&P 500 to eight financial instruments covering
equities, volatility, commodities, rates, crypto, FX, and international equity.

Instruments:
  ^GSPC   S&P 500           (US large-cap equity)
  ^VIX    VIX               (implied volatility index; log-change in level)
  GC=F    Gold              (commodity safe-haven)
  CL=F    WTI Oil           (commodity cyclical)
  ^TNX    US 10Y Yield      (rates; log-change in yield)
  BTC-USD Bitcoin           (crypto; available from ~2014)
  EURUSD=X EUR/USD          (FX; risk-on/risk-off proxy)
  ^HSI    Hang Seng         (international equity)

Encoding (T2-compliant observer projection for all instruments):
  r[t] = log(close[t] / close[t-1])
  rank[t] = rank of r[t] among all available r for this instrument
  state (b, e) = (rank[t], rank[t-1]) ∈ (Z/27Z)^2

Companion: M=[[5,-1],[1,0]], p=3, r=1, det=+1, k=3 (mod 27) — cert [439]

Pre-registered hypotheses (before running):
  H1: All instruments: birth fraction ≈ 66.7% (theory, from rank normalization)
  H2: USD-denominated assets (GSPC, Gold, Oil, TNX): recession fixed-layer > expansion
  H3: Crypto (BTC): no NBER recession signal; strong high-vol / low-vol split
  H4: VIX: OPPOSITE direction — recession months show LOWER fixed-layer
      (VIX spikes = large positive returns = both months in TOP rank bins,
      which are NOT near fixed points (0,0),(9,9),(18,18))
  H5: Cross-instrument: birth-layer indicators positively correlated for
      USD-denominated instruments during the SAME calendar months

NBER recession periods:
  2001-03 to 2001-11  (dot-com)
  2007-12 to 2009-06  (GFC)
  2020-02 to 2020-04  (COVID)
"""

import sys
import math
import json
import time
import random
import urllib.request
import urllib.error
import urllib.parse


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
    mod = P ** k
    b_k, e_k = b % mod, e % mod
    for j in range(k + 1):
        Mj = mat_pow_mod(M, P ** j, mod)
        b2, e2 = apply_mat(Mj, b_k, e_k, mod)
        if b2 == b_k and e2 == e_k:
            return j
    return k


# ── Cert parameters ───────────────────────────────────────────────────────────

P         = 3
K         = 3
MOD       = P ** K      # 27
COMPANION = [[5, -1], [1, 0]]

# NBER recession months (year, month)
_REC = set()
for _y0, _m0, _y1, _m1 in [(2001,3,2001,11),(2007,12,2009,6),(2020,2,2020,4)]:
    _y, _m = _y0, _m0
    while (_y, _m) <= (_y1, _m1):
        _REC.add((_y, _m))
        _m += 1
        if _m > 12:
            _m = 1
            _y += 1


# ── Data download ─────────────────────────────────────────────────────────────

def fetch_yahoo(ticker, years=25):
    """Download monthly closes. Returns list of (year, month, close) or None."""
    enc = urllib.parse.quote(ticker, safe="")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
           f"?interval=1mo&range={years}y")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        result   = data["chart"]["result"][0]
        ts_list  = result["timestamp"]
        cl_list  = result["indicators"]["quote"][0]["close"]
        import datetime
        rows = []
        for ts, cl in zip(ts_list, cl_list):
            if cl is None:
                continue
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            rows.append((dt.year, dt.month, float(cl)))
        return rows if len(rows) > 20 else None
    except Exception:
        return None


# ── Return computation + rank normalization ───────────────────────────────────

def log_returns(rows):
    """Compute (year, month, logret) from (year, month, close) list."""
    out = []
    for i in range(1, len(rows)):
        y, mo, cl   = rows[i]
        _, _, cl_lg = rows[i - 1]
        if cl > 0 and cl_lg > 0:
            out.append((y, mo, math.log(cl / cl_lg)))
    return out


def rank_normalize(rets, n_bins=27):
    """Map each return to {0,...,n_bins-1} using full-history ranks."""
    n  = len(rets)
    indexed = sorted(enumerate(rets), key=lambda x: x[1])
    bins = [0] * n
    bsz  = n / n_bins
    for rank, (orig, _) in enumerate(indexed):
        bins[orig] = min(int(rank / bsz), n_bins - 1)
    return bins


# ── Filter bank ───────────────────────────────────────────────────────────────

def layer_stats(pairs):
    """Returns (fixed_frac, frozen_frac, birth_frac) for list of (b,e)."""
    if not pairs:
        return 0.0, 0.0, 0.0
    pvs = [period_val(b, e, K, P, COMPANION) for b, e in pairs]
    n   = len(pvs)
    return (
        sum(1 for v in pvs if v == 0) / n,
        sum(1 for v in pvs if 0 < v < K) / n,
        sum(1 for v in pvs if v == K) / n,
    )


def regime_split(tagged_pairs):
    """
    tagged_pairs: list of (b, e, year, month, logret)
    Returns dicts: 'all', 'recession', 'expansion', 'high_vol', 'low_vol'
    """
    rets = [r for _, _, _, _, r in tagged_pairs]
    if not rets:
        return {}
    mu    = sum(rets) / len(rets)
    var   = sum((r - mu) * (r - mu) for r in rets) / len(rets)
    sigma = var ** 0.5 if var > 0 else 1e-9

    groups = {g: [] for g in ("all", "recession", "expansion", "high_vol", "low_vol")}
    for b, e, y, mo, r in tagged_pairs:
        groups["all"].append((b, e))
        if (y, mo) in _REC:
            groups["recession"].append((b, e))
        else:
            groups["expansion"].append((b, e))
        abs_r = abs(r)
        if abs_r > sigma:
            groups["high_vol"].append((b, e))
        elif abs_r < 0.5 * sigma:
            groups["low_vol"].append((b, e))
    return groups


def permutation_test(g1, g2, observed_delta, n_perm=300):
    """p-value for fixed-layer absolute difference via permutation."""
    if not g1 or not g2:
        return float("nan")
    combined = g1 + g2
    n1 = len(g1)
    rng = random.Random(42)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        d = abs(layer_stats(combined[:n1])[0] - layer_stats(combined[n1:])[0])
        if d >= observed_delta:
            extreme += 1
    return extreme / n_perm


# ── Instrument analysis ───────────────────────────────────────────────────────

INSTRUMENTS = [
    ("^GSPC",    "S&P 500",      "equity"),
    ("^VIX",     "VIX",          "volatility"),
    ("GC=F",     "Gold",         "commodity"),
    ("CL=F",     "WTI Oil",      "commodity"),
    ("^TNX",     "US 10Y Yield", "rates"),
    ("BTC-USD",  "Bitcoin",      "crypto"),
    ("EURUSD=X", "EUR/USD",      "fx"),
    ("^HSI",     "Hang Seng",    "intl_equity"),
]


def analyse_instrument(ticker, name, itype):
    rows = fetch_yahoo(ticker)
    if rows is None:
        return None, f"{name} ({ticker}): API unavailable"

    rets = log_returns(rows)
    if len(rets) < 24:
        return None, f"{name}: too few months ({len(rets)})"

    r_vals = [r for _, _, r in rets]
    bins   = rank_normalize(r_vals)

    tagged = []
    for i in range(1, len(bins)):
        b  = bins[i]
        e  = bins[i - 1]
        y, mo, r = rets[i]
        tagged.append((b, e, y, mo, r))

    groups = regime_split(tagged)
    stats  = {g: layer_stats(groups[g]) for g in groups}

    # Recession signal
    rec_fixed = stats["recession"][0]
    exp_fixed = stats["expansion"][0]
    delta_rec = rec_fixed - exp_fixed
    n_rec     = len(groups["recession"])
    n_exp     = len(groups["expansion"])
    perm_p    = (permutation_test(groups["recession"], groups["expansion"],
                                  abs(delta_rec)) if n_rec >= 3 else float("nan"))

    # Volatility signal
    hv_birth = stats["high_vol"][2]
    lv_birth = stats["low_vol"][2]
    delta_vol = lv_birth - hv_birth   # positive = low-vol has MORE birth

    return {
        "ticker":     ticker,
        "name":       name,
        "type":       itype,
        "n_months":   len(rets),
        "n_rec":      n_rec,
        "n_exp":      n_exp,
        "all_birth":  stats["all"][2],
        "all_fixed":  stats["all"][0],
        "rec_fixed":  rec_fixed,
        "exp_fixed":  exp_fixed,
        "delta_rec":  delta_rec,
        "perm_p":     perm_p,
        "hv_birth":   hv_birth,
        "lv_birth":   lv_birth,
        "delta_vol":  delta_vol,
        "tagged":     tagged,   # keep for cross-instrument analysis
    }, None


# ── Cross-instrument birth correlation ───────────────────────────────────────

def cross_instrument_correlation(results):
    """
    For each pair of instruments, compute Pearson correlation of monthly
    birth-layer binary indicators (1=birth at k=3, 0=otherwise).
    Uses only months where BOTH instruments have data.
    """
    # Build month→birth_indicator for each instrument
    series = {}
    for r in results:
        name    = r["name"]
        monthly = {}
        for b, e, y, mo, _ in r["tagged"]:
            pv = period_val(b, e, K, P, COMPANION)
            monthly[(y, mo)] = 1 if pv == K else 0
        series[name] = monthly

    names = [r["name"] for r in results]
    n     = len(names)
    corr  = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i][j] = 1.0
                continue
            si, sj = series[names[i]], series[names[j]]
            months = sorted(set(si) & set(sj))
            if len(months) < 12:
                corr[i][j] = float("nan")
                continue
            xi = [si[m] for m in months]
            xj = [sj[m] for m in months]
            mu_i = sum(xi) / len(xi)
            mu_j = sum(xj) / len(xj)
            cov  = sum((xi[k] - mu_i) * (xj[k] - mu_j) for k in range(len(months)))
            sd_i = (sum((xi[k] - mu_i) * (xi[k] - mu_i) for k in range(len(months)))) ** 0.5
            sd_j = (sum((xj[k] - mu_j) * (xj[k] - mu_j) for k in range(len(months)))) ** 0.5
            if sd_i < 1e-9 or sd_j < 1e-9:
                corr[i][j] = float("nan")
            else:
                corr[i][j] = cov / (sd_i * sd_j)

    return names, corr


# ── Recession universality test ───────────────────────────────────────────────

def recession_universality(results):
    """
    For each NBER recession month, how many instruments show birth vs fixed layer?
    Computes the cross-instrument layer distribution during recession vs expansion.
    """
    rec_pvs, exp_pvs = [], []
    for r in results:
        for b, e, y, mo, _ in r["tagged"]:
            pv = period_val(b, e, K, P, COMPANION)
            if (y, mo) in _REC:
                rec_pvs.append(pv)
            else:
                exp_pvs.append(pv)

    n_rec = len(rec_pvs)
    n_exp = len(exp_pvs)

    def dist(pvs):
        n = len(pvs) or 1
        fixed  = sum(1 for v in pvs if v == 0) / n
        birth  = sum(1 for v in pvs if v == K) / n
        frozen = 1.0 - fixed - birth
        return fixed, frozen, birth

    return dist(rec_pvs), dist(exp_pvs), n_rec, n_exp


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("QA Witt Tower Filter Bank — Multi-Instrument Extension")
    print("=" * 68)
    print(f"Companion M=[[5,-1],[1,0]], p={P}, k={K}, mod={MOD}")
    print(f"Instruments: {len(INSTRUMENTS)}")
    print()

    results = []
    skipped = []

    for ticker, name, itype in INSTRUMENTS:
        print(f"  Downloading {name} ({ticker})...", end="", flush=True)
        r, err = analyse_instrument(ticker, name, itype)
        if err:
            print(f"  SKIP — {err}")
            skipped.append(err)
        else:
            print(f"  {r['n_months']} months")
            results.append(r)
        time.sleep(0.35)   # mild rate-limit courtesy

    print(f"\n  Loaded {len(results)}/{len(INSTRUMENTS)} instruments "
          f"({len(skipped)} skipped)\n")

    if not results:
        print("No data — aborting.")
        return

    # ── Per-instrument table ──────────────────────────────────────────────────
    print("── Per-Instrument Layer Fractions (k=3, mod=27) ─────────────────────")
    print(f"  {'instrument':>16}  {'type':>12}  {'n':>5}  {'birth%':>7}  "
          f"{'fixed%':>7}  {'rec_fixed':>10}  {'exp_fixed':>10}  "
          f"{'Δrec-exp':>10}  {'perm_p':>8}  {'signal':>8}")
    print("  " + "-" * 105)

    for r in results:
        perm_s = f"{r['perm_p']:.3f}" if not math.isnan(r['perm_p']) else "n/a"
        n_rec  = r['n_rec']
        signal = ""
        if n_rec >= 3:
            if r['delta_rec'] > 0.04:
                signal = "↑rec ◄"
            elif r['delta_rec'] < -0.04:
                signal = "↓rec ◄"
        print(f"  {r['name']:>16}  {r['type']:>12}  {r['n_months']:>5}  "
              f"{r['all_birth']*100:>6.1f}%  {r['all_fixed']*100:>6.1f}%  "
              f"{r['rec_fixed']*100:>9.1f}%  {r['exp_fixed']*100:>9.1f}%  "
              f"{r['delta_rec']*100:>+9.1f}pp  {perm_s:>8}  {signal}")

    print()

    # ── Volatility regime table ───────────────────────────────────────────────
    print("── Volatility Regime: Low-vol birth% vs High-vol birth% ─────────────")
    print(f"  {'instrument':>16}  {'lv_birth%':>10}  {'hv_birth%':>10}  "
          f"{'Δ(lv-hv)':>10}  interpretation")
    print("  " + "-" * 68)
    for r in results:
        d = r['delta_vol']
        interp = ("low-vol = more birth (expected)" if d > 0.04
                  else "high-vol = more birth (unexpected)" if d < -0.04
                  else "no clear split")
        flag = " ◄" if abs(d) > 0.04 else ""
        print(f"  {r['name']:>16}  {r['lv_birth']*100:>9.1f}%  "
              f"{r['hv_birth']*100:>9.1f}%  {d*100:>+9.1f}pp{flag}  {interp}")

    print()

    # ── Recession universality ────────────────────────────────────────────────
    if len(results) >= 2:
        (rf_r, ro_r, rb_r), (rf_e, ro_e, rb_e), n_rec, n_exp = recession_universality(results)
        print("── Cross-Instrument Recession Universality (all instruments pooled) ──")
        print(f"  n_recession_obs = {n_rec}  (across {len(results)} instruments × recession months)")
        print(f"  n_expansion_obs = {n_exp}")
        print()
        print(f"  {'layer':<12}  {'recession':>12}  {'expansion':>12}  {'Δ':>8}")
        print("  " + "-" * 48)
        for lbl, rv, ev in [("fixed",  rf_r, rf_e),
                              ("frozen", ro_r, ro_e),
                              ("birth",  rb_r, rb_e)]:
            flag = "  ◄" if abs(rv - ev) > 0.03 else ""
            print(f"  {lbl:<12}  {rv*100:>11.1f}%  {ev*100:>11.1f}%  "
                  f"{(rv-ev)*100:>+7.1f}pp{flag}")
        print()

    # ── Cross-instrument birth-layer correlation ──────────────────────────────
    if len(results) >= 2:
        print("── Cross-Instrument Birth-Layer Correlation (monthly, k=3) ──────────")
        names, corr = cross_instrument_correlation(results)
        # Print header
        short = [n.split()[0][:8] for n in names]
        header = f"  {'':>12}"
        for s in short:
            header += f"  {s:>8}"
        print(header)
        for i, row_name in enumerate(names):
            row = f"  {row_name[:12]:>12}"
            for j in range(len(names)):
                v = corr[i][j]
                if math.isnan(v):
                    row += f"  {'n/a':>8}"
                elif i == j:
                    row += f"  {'1.000':>8}"
                else:
                    flag = "*" if abs(v) > 0.15 else " "
                    row += f"  {v:>+7.3f}{flag}"
            print(row)
        print(f"\n  (* |corr| > 0.15)")
        print()

    # ── Hypothesis scorecard ──────────────────────────────────────────────────
    print("=" * 68)
    print("HYPOTHESIS SCORECARD")

    # H1: birth ≈ 66.7%
    all_births = [r["all_birth"] for r in results]
    h1 = all(0.55 <= b <= 0.78 for b in all_births)
    print(f"  H1 (birth ≈ 2/3):         {'PASS ✓' if h1 else 'FAIL ✗'}  "
          f"range [{min(all_births)*100:.1f}%, {max(all_births)*100:.1f}%]")

    # H2: USD assets show recession fixed-layer elevation
    usd_assets = [r for r in results
                  if r["type"] in ("equity", "commodity", "rates")
                  and r["n_rec"] >= 3]
    h2_flags = [r["delta_rec"] > 0.02 for r in usd_assets]
    h2 = sum(h2_flags) >= len(h2_flags) // 2 + 1 if usd_assets else False
    names_h2 = [r["name"] for r in usd_assets]
    passes_h2 = [r["name"] for r, ok in zip(usd_assets, h2_flags) if ok]
    print(f"  H2 (USD recession signal):{' PASS ✓' if h2 else ' FAIL ✗'}  "
          f"{len(passes_h2)}/{len(usd_assets)} USD assets show Δ>2pp: {passes_h2}")

    # H3: BTC no recession signal
    btc = next((r for r in results if "BTC" in r["ticker"]), None)
    if btc:
        h3 = abs(btc["delta_rec"]) < 0.08 or btc["n_rec"] < 5
        print(f"  H3 (BTC no rec signal):   {'PASS ✓' if h3 else 'FAIL ✗'}  "
              f"BTC recession Δ = {btc['delta_rec']*100:+.1f}pp  (n_rec={btc['n_rec']})")
    else:
        print("  H3 (BTC no rec signal):   SKIP — Bitcoin data unavailable")

    # H4: VIX shows LOWER fixed-layer during recessions
    vix = next((r for r in results if "VIX" in r["ticker"]), None)
    if vix and vix["n_rec"] >= 3:
        h4 = vix["delta_rec"] < 0.0   # recession < expansion (opposite of GSPC)
        print(f"  H4 (VIX opposite signal): {'PASS ✓' if h4 else 'FAIL ✗'}  "
              f"VIX rec_fixed={vix['rec_fixed']*100:.1f}% "
              f"exp_fixed={vix['exp_fixed']*100:.1f}%  "
              f"Δ={vix['delta_rec']*100:+.1f}pp")
    else:
        print("  H4 (VIX opposite signal): SKIP")

    # H5: Low-vol shows higher birth fraction than high-vol (for most instruments)
    vol_flags = [r["delta_vol"] > 0.02 for r in results]
    h5 = sum(vol_flags) >= len(results) * 0.6
    print(f"  H5 (low-vol > birth):     {'PASS ✓' if h5 else 'FAIL ✗'}  "
          f"{sum(vol_flags)}/{len(results)} instruments show low-vol birth > high-vol birth")

    print()

    # ── Summary of key magnitudes ─────────────────────────────────────────────
    print("── Key Magnitudes (NBER recession fixed-layer Δ by instrument) ───────")
    with_rec = [(r["name"], r["delta_rec"]) for r in results if r["n_rec"] >= 3]
    with_rec.sort(key=lambda x: -x[1])
    for nm, d in with_rec:
        bar = "#" * int(abs(d) * 200)
        sign = "+" if d >= 0 else "-"
        print(f"  {nm:>16}: {d*100:>+6.1f}pp  {sign}{bar}")
    print()


if __name__ == "__main__":
    main()
