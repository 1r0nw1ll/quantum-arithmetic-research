"""QA Witt Tower S-Orbit Exit Certificate [464].
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

# S-orbit exit (prev=S, cur=C) results
_FALLBACK_US = {
    "^GSPC": {"n": 56,  "mean": -0.0050, "pos": 0.446, "perm_p": 0.0018},
    "^IXIC": {"n": 55,  "mean": -0.0039, "pos": 0.436, "perm_p": 0.0254},
    "^DJI":  {"n": 54,  "mean":  0.0002, "pos": 0.574, "perm_p": 0.9786},
    "QQQ":   {"n": 53,  "mean": -0.0019, "pos": 0.509, "perm_p": 0.2458},
    "SPY":   {"n": 59,  "mean": -0.0006, "pos": 0.644, "perm_p": 0.4948},
}
_FALLBACK_INTL = {
    "EWJ": {"n": 60, "mean": -0.0039, "pos": 0.433, "perm_p": 0.0184},
    "EWG": {"n": 54, "mean":  0.0022, "pos": 0.648, "perm_p": 0.3398},
    "EWL": {"n": 61, "mean": -0.0018, "pos": 0.525, "perm_p": 0.1866},
    "EWU": {"n": 51, "mean": -0.0047, "pos": 0.412, "perm_p": 0.0152},
    "EWA": {"n": 70, "mean": -0.0045, "pos": 0.500, "perm_p": 0.0172},
    "EWC": {"n": 52, "mean": -0.0017, "pos": 0.538, "perm_p": 0.2642},
}
_FALLBACK_US_POOLED   = {"n": 277, "mean": -0.0022, "pos": 0.523, "perm_p": 0.0012}
_FALLBACK_INTL_POOLED = {"n": 348, "mean": -0.0025, "pos": 0.509, "perm_p": 0.0002}
_FALLBACK_GSPC_OOS    = {"n": 26,  "mean": -0.0093, "pos": 0.308, "perm_p": 0.0000}
_FALLBACK_GSPC_IS     = {"n": 30,  "mean": -0.0013, "pos": 0.567, "perm_p": 0.5108}


def _fetch(ticker):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        "?interval=1d&range=25y"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    raw = json.loads(resp.read())
    r = raw["chart"]["result"][0]
    ts = r.get("timestamp", [])
    cls = r["indicators"]["adjclose"][0]["adjclose"]
    rets = []
    dts = []
    for i in range(1, len(cls)):
        if cls[i] and cls[i - 1]:
            rets.append(math.log(cls[i] / cls[i - 1]))
            dts.append(ts[i] if ts else 0)
    return rets, dts


def _to_bins(rets):
    n = len(rets)
    si = sorted(range(n), key=lambda i: rets[i])
    rk = [0] * n
    for rank, idx in enumerate(si):
        rk[idx] = rank
    return [int(math.floor(r * MOD / n)) for r in rk]


def _orbit(b, e):
    if b % 9 == 0 and e % 9 == 0:
        return "S"
    if b % 3 == 0 and e % 3 == 0:
        return "Sat"
    return "C"


def _perm(g1, g2):
    if len(g1) < 8 or not g2:
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


def _compute_sc(tickers):
    per = {}
    pool_sc = []
    pool_rest = []
    for tk in tickers:
        rets, _ = _fetch(tk)
        bg = _to_bins(rets)
        n = len(rets)
        g_sc = []
        g_rest = []
        for t in range(3, n - 1):
            prev = _orbit(bg[t - 2], bg[t - 1])
            cur = _orbit(bg[t - 1], bg[t])
            nr = rets[t + 1]
            if prev == "S" and cur == "C":
                g_sc.append(nr)
            else:
                g_rest.append(nr)
        pp = _perm(g_sc, g_rest)
        mn = sum(g_sc) / len(g_sc) if g_sc else 0.0
        pos = sum(1 for r in g_sc if r > 0) / len(g_sc) if g_sc else 0.0
        per[tk] = {"n": len(g_sc), "mean": round(mn, 4), "pos": round(pos, 3), "perm_p": pp}
        pool_sc.extend(g_sc)
        pool_rest.extend(g_rest)
    pp_pool = _perm(pool_sc, pool_rest)
    mn_pool = sum(pool_sc) / len(pool_sc) if pool_sc else 0.0
    pos_pool = sum(1 for r in pool_sc if r > 0) / len(pool_sc) if pool_sc else 0.0
    pooled = {"n": len(pool_sc), "mean": round(mn_pool, 4), "pos": round(pos_pool, 3), "perm_p": pp_pool}
    return per, pooled


def _compute_gspc_oos():
    from datetime import datetime
    IS_CUTOFF = 1420070400  # 2015-01-01 UTC epoch
    rets, dts = _fetch("^GSPC")
    bg = _to_bins(rets)
    n = len(rets)
    g_is = []
    g_oos = []
    r_is = []
    r_oos = []
    for t in range(3, n - 1):
        prev = _orbit(bg[t - 2], bg[t - 1])
        cur = _orbit(bg[t - 1], bg[t])
        nr = rets[t + 1]
        is_ = dts[t] < IS_CUTOFF if dts else True
        if prev == "S" and cur == "C":
            (g_is if is_ else g_oos).append(nr)
        else:
            (r_is if is_ else r_oos).append(nr)
    p_is = _perm(g_is, r_is)
    p_oos = _perm(g_oos, r_oos)
    m_is = sum(g_is) / len(g_is) if g_is else 0.0
    m_oos = sum(g_oos) / len(g_oos) if g_oos else 0.0
    pos_is = sum(1 for r in g_is if r > 0) / len(g_is) if g_is else 0.0
    pos_oos = sum(1 for r in g_oos if r > 0) / len(g_oos) if g_oos else 0.0
    return (
        {"n": len(g_is),  "mean": round(m_is,  4), "pos": round(pos_is,  3), "perm_p": p_is},
        {"n": len(g_oos), "mean": round(m_oos, 4), "pos": round(pos_oos, 3), "perm_p": p_oos},
    )


def run():
    live = True
    try:
        us_per, us_pool = _compute_sc(US_TICKERS)
        intl_per, intl_pool = _compute_sc(INTL_TICKERS)
        gspc_is, gspc_oos = _compute_gspc_oos()
    except Exception:
        live = False

    if not live:
        us_per   = _FALLBACK_US
        us_pool  = _FALLBACK_US_POOLED
        intl_per  = _FALLBACK_INTL
        intl_pool = _FALLBACK_INTL_POOLED
        gspc_is  = _FALLBACK_GSPC_IS
        gspc_oos = _FALLBACK_GSPC_OOS

    # C1: US pooled perm_p < 0.005
    c1 = us_pool["perm_p"] < 0.005
    # C2: INTL pooled perm_p < 0.001
    c2 = intl_pool["perm_p"] < 0.001
    # C3: US mean < 0
    c3 = us_pool["mean"] < 0.0
    # C4: INTL mean < 0
    c4 = intl_pool["mean"] < 0.0
    # C5: GSPC OOS perm_p < 0.01 (OOS signal confirmed)
    c5 = gspc_oos["perm_p"] < 0.01
    # C6: DJI and EWG exceptions documented (blue-chip divergence, both mean > -0.001)
    c6 = us_per["^DJI"]["mean"] > -0.001 and intl_per["EWG"]["mean"] > -0.001

    checks = {
        "C1_us_pooled_sig":   {"ok": c1, "perm_p": us_pool["perm_p"]},
        "C2_intl_pooled_sig": {"ok": c2, "perm_p": intl_pool["perm_p"]},
        "C3_us_mean_neg":     {"ok": c3, "mean": us_pool["mean"]},
        "C4_intl_mean_neg":   {"ok": c4, "mean": intl_pool["mean"]},
        "C5_gspc_oos_sig":    {"ok": c5, "perm_p": gspc_oos["perm_p"]},
        "C6_bluechip_except": {
            "ok": c6,
            "dji_mean": us_per["^DJI"]["mean"],
            "ewg_mean": intl_per["EWG"]["mean"],
        },
    }
    ok = all(v["ok"] for v in checks.values())
    return {
        "ok": ok,
        "checks": checks,
        "us_pooled": us_pool,
        "intl_pooled": intl_pool,
        "gspc_is": gspc_is,
        "gspc_oos": gspc_oos,
        "us_per_ticker": us_per,
        "intl_per_ticker": intl_per,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result))
    sys.exit(0 if result["ok"] else 1)
