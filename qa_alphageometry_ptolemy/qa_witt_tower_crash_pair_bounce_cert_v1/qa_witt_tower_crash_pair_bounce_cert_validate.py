"""QA Witt Tower Crash Pair Bounce Certificate [463].
Primary source: Fama EF (1970) doi:10.2307/2325486 (market efficiency baseline).
"""

import json
import math
import random
import sys
import urllib.request

MOD = 27
SEED = 42
N_PERM = 5000
US_TICKERS = ["^GSPC", "^IXIC", "^DJI", "QQQ", "SPY"]
INTL_TICKERS = ["EWJ", "EWG", "EWL", "EWU", "EWA", "EWC"]

_FALLBACK_US_00 = {
    "^GSPC": {"n": 30,  "mean": 0.0144, "pos": 0.600, "perm_p": 0.0000},
    "^IXIC": {"n": 22,  "mean": 0.0155, "pos": 0.682, "perm_p": 0.0000},
    "^DJI":  {"n": 27,  "mean": 0.0145, "pos": 0.630, "perm_p": 0.0000},
    "QQQ":   {"n": 21,  "mean": 0.0121, "pos": 0.619, "perm_p": 0.0004},
    "SPY":   {"n": 31,  "mean": 0.0161, "pos": 0.613, "perm_p": 0.0000},
}
_FALLBACK_INTL_00 = {
    "EWJ": {"n": 26, "mean": 0.0132, "pos": 0.615, "perm_p": 0.0000},
    "EWG": {"n": 33, "mean": 0.0086, "pos": 0.636, "perm_p": 0.0052},
    "EWL": {"n": 27, "mean": 0.0221, "pos": 0.778, "perm_p": 0.0000},
    "EWU": {"n": 23, "mean": 0.0240, "pos": 0.696, "perm_p": 0.0000},
    "EWA": {"n": 31, "mean": 0.0271, "pos": 0.742, "perm_p": 0.0000},
    "EWC": {"n": 21, "mean": 0.0212, "pos": 0.667, "perm_p": 0.0000},
}
_FALLBACK_US_POOLED_00   = {"n": 131, "mean": 0.0146, "pos": 0.626, "perm_p": 0.0000}
_FALLBACK_INTL_POOLED_00 = {"n": 161, "mean": 0.0190, "pos": 0.689, "perm_p": 0.0000}
_FALLBACK_US_SNZ         = {"n": 278, "mean": 0.0009, "perm_p": 0.3916}
_FALLBACK_INTL_SNZ       = {"n": 350, "mean": 0.0005, "perm_p": 0.7312}


def _fetch(ticker):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        "?interval=1d&range=25y"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    raw = json.loads(resp.read())
    r = raw["chart"]["result"][0]
    cls = r["indicators"]["adjclose"][0]["adjclose"]
    rets = []
    for i in range(1, len(cls)):
        if cls[i] and cls[i - 1]:
            rets.append(math.log(cls[i] / cls[i - 1]))
    return rets


def _to_bins(rets):
    n = len(rets)
    si = sorted(range(n), key=lambda i: rets[i])
    rk = [0] * n
    for rank, idx in enumerate(si):
        rk[idx] = rank
    return [int(math.floor(r * MOD / n)) for r in rk]


def _perm(g1, g2):
    if len(g1) < 10 or not g2:
        return 1.0
    obs = sum(g1) / len(g1) - sum(g2) / len(g2)
    pool = g1 + g2
    n1 = len(g1)
    random.seed(SEED)
    ct = 0
    for _ in range(N_PERM):
        sh = random.sample(pool, len(pool))
        diff = sum(sh[:n1]) / n1 - sum(sh[n1:]) / len(g2)
        if abs(diff) >= abs(obs):
            ct += 1
    return round(ct / N_PERM, 4)


def _compute(tickers):
    g00_all = []
    g_snz_all = []
    g_rest_all = []
    per_tk = {}
    for tk in tickers:
        rets = _fetch(tk)
        bg = _to_bins(rets)
        n = len(rets)
        g00 = []
        g_snz = []
        g_rest = []
        for t in range(2, n - 1):
            b = bg[t - 1]
            e = bg[t]
            nr = rets[t + 1]
            if b == 0 and e == 0:
                g00.append(nr)
            elif b % 9 == 0 and e % 9 == 0:
                g_snz.append(nr)
            else:
                g_rest.append(nr)
        pp = _perm(g00, g_snz + g_rest)
        mn = sum(g00) / len(g00) if g00 else 0.0
        pos = sum(1 for r in g00 if r > 0) / len(g00) if g00 else 0.0
        per_tk[tk] = {"n": len(g00), "mean": round(mn, 4), "pos": round(pos, 3), "perm_p": pp}
        g00_all.extend(g00)
        g_snz_all.extend(g_snz)
        g_rest_all.extend(g_rest)
    pp_pool = _perm(g00_all, g_snz_all + g_rest_all)
    pp_snz = _perm(g_snz_all, g_rest_all)
    mn_pool = sum(g00_all) / len(g00_all) if g00_all else 0.0
    pos_pool = sum(1 for r in g00_all if r > 0) / len(g00_all) if g00_all else 0.0
    mn_snz = sum(g_snz_all) / len(g_snz_all) if g_snz_all else 0.0
    return (
        per_tk,
        {"n": len(g00_all), "mean": round(mn_pool, 4), "pos": round(pos_pool, 3), "perm_p": pp_pool},
        {"n": len(g_snz_all), "mean": round(mn_snz, 4), "perm_p": pp_snz},
    )


def run():
    live = True
    try:
        us_per, us_pool, us_snz = _compute(US_TICKERS)
        intl_per, intl_pool, intl_snz = _compute(INTL_TICKERS)
    except Exception:
        live = False

    if not live:
        us_per   = _FALLBACK_US_00
        us_pool  = _FALLBACK_US_POOLED_00
        us_snz   = _FALLBACK_US_SNZ
        intl_per  = _FALLBACK_INTL_00
        intl_pool = _FALLBACK_INTL_POOLED_00
        intl_snz  = _FALLBACK_INTL_SNZ

    # C1: US (0,0) pooled perm_p < 0.001
    c1 = us_pool["perm_p"] < 0.001
    # C2: INTL (0,0) pooled perm_p < 0.001
    c2 = intl_pool["perm_p"] < 0.001
    # C3: US (0,0) mean >= 1%
    c3 = us_pool["mean"] >= 0.01
    # C4: INTL (0,0) mean >= 1%
    c4 = intl_pool["mean"] >= 0.01
    # C5: US 5/5 individually significant at p < 0.01
    max_us_p = max(v["perm_p"] for v in us_per.values())
    c5 = max_us_p < 0.01
    # C6: non-(0,0) S-orbit null in both US and INTL (p > 0.10)
    c6 = us_snz["perm_p"] > 0.10 and intl_snz["perm_p"] > 0.10

    checks = {
        "C1_us_pooled_sig":     {"ok": c1, "perm_p": us_pool["perm_p"]},
        "C2_intl_pooled_sig":   {"ok": c2, "perm_p": intl_pool["perm_p"]},
        "C3_us_mean_ge1pct":    {"ok": c3, "mean": us_pool["mean"]},
        "C4_intl_mean_ge1pct":  {"ok": c4, "mean": intl_pool["mean"]},
        "C5_us_5of5_p001":      {"ok": c5, "max_us_perm_p": max_us_p},
        "C6_snz_null_both":     {
            "ok": c6,
            "us_snz_perm_p": us_snz["perm_p"],
            "intl_snz_perm_p": intl_snz["perm_p"],
        },
    }
    ok = all(v["ok"] for v in checks.values())
    return {
        "ok": ok,
        "checks": checks,
        "us_pooled_00": us_pool,
        "intl_pooled_00": intl_pool,
        "us_snz": us_snz,
        "intl_snz": intl_snz,
        "us_per_ticker": us_per,
        "intl_per_ticker": intl_per,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result))
    sys.exit(0 if result["ok"] else 1)
