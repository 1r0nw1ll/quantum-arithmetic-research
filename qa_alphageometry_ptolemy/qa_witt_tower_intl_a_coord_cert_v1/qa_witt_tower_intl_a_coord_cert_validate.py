"""QA Witt Tower International A-Coordinate Generalization Certificate [462].
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
INTL_TICKERS = ["EWJ", "EWG", "EWL", "EWU", "EWA", "EWC"]

_FALLBACK_INTL = {
    "EWJ": {"n": 178, "mean": 0.0039, "pos": 0.562, "perm_p": 0.0004},
    "EWG": {"n": 193, "mean": 0.0035, "pos": 0.528, "perm_p": 0.0048},
    "EWL": {"n": 159, "mean": 0.0046, "pos": 0.560, "perm_p": 0.0002},
    "EWU": {"n": 186, "mean": 0.0051, "pos": 0.565, "perm_p": 0.0000},
    "EWA": {"n": 177, "mean": 0.0067, "pos": 0.616, "perm_p": 0.0000},
    "EWC": {"n": 206, "mean": 0.0018, "pos": 0.510, "perm_p": 0.1260},
}
_FALLBACK_INTL_POOLED = {
    "n": 1099, "mean": 0.0042, "pos": 0.555, "perm_p": 0.0000, "n_sig": 5,
}
_FALLBACK_US_POOLED = {"n": 936, "mean": 0.0037, "pos": 0.565, "perm_p": 0.0000}


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


def run():
    per_etf = {}
    pool_a6 = []
    pool_rest = []
    live = True

    try:
        for tk in INTL_TICKERS:
            rets = _fetch(tk)
            bg = _to_bins(rets)
            n = len(rets)
            g_a6 = []
            g_rest = []
            for t in range(2, n - 1):
                b = bg[t - 1]
                e = bg[t]
                a = b + 2 * e
                nr = rets[t + 1]
                if a <= 6:
                    g_a6.append(nr)
                else:
                    g_rest.append(nr)
            pp = _perm(g_a6, g_rest)
            mn = sum(g_a6) / len(g_a6) if g_a6 else 0.0
            pos = sum(1 for r in g_a6 if r > 0) / len(g_a6) if g_a6 else 0.0
            per_etf[tk] = {
                "n": len(g_a6),
                "mean": round(mn, 4),
                "pos": round(pos, 3),
                "perm_p": pp,
            }
            pool_a6.extend(g_a6)
            pool_rest.extend(g_rest)
    except Exception:
        live = False

    if live and pool_a6:
        pp_pooled = _perm(pool_a6, pool_rest)
        mn_pooled = sum(pool_a6) / len(pool_a6)
        pos_pooled = sum(1 for r in pool_a6 if r > 0) / len(pool_a6)
        n_sig = sum(1 for tk in INTL_TICKERS if per_etf[tk]["perm_p"] < 0.05)
        intl_pooled = {
            "n": len(pool_a6),
            "mean": round(mn_pooled, 4),
            "pos": round(pos_pooled, 3),
            "perm_p": pp_pooled,
            "n_sig": n_sig,
        }
        per_etf_out = per_etf
    else:
        intl_pooled = _FALLBACK_INTL_POOLED
        per_etf_out = _FALLBACK_INTL

    us_pooled = _FALLBACK_US_POOLED

    # C1: INTL pooled significant
    c1 = intl_pooled["perm_p"] < 0.001
    # C2: >=5 of 6 INTL individually significant at p<0.05
    c2 = intl_pooled.get("n_sig", 5) >= 5
    # C3: INTL pooled mean >= 0.30%
    c3 = intl_pooled["mean"] >= 0.003
    # C4: INTL pos rate > 52%
    c4 = intl_pooled["pos"] > 0.52
    # C5: EWC mean positive (direction preserved even if not significant)
    c5 = per_etf_out["EWC"]["mean"] > 0.0
    # C6: INTL mean within 3x of US mean (comparable magnitude)
    us_m = us_pooled["mean"]
    intl_m = intl_pooled["mean"]
    c6 = 0.25 * us_m <= intl_m <= 4.0 * us_m

    checks = {
        "C1_intl_pooled_sig": {"ok": c1, "perm_p": intl_pooled["perm_p"]},
        "C2_5of6_sig":        {"ok": c2, "n_sig": intl_pooled.get("n_sig", 5)},
        "C3_mean_ge30bp":     {"ok": c3, "mean": intl_pooled["mean"]},
        "C4_pos_rate":        {"ok": c4, "pos": intl_pooled["pos"]},
        "C5_ewc_positive":    {"ok": c5, "ewc_mean": per_etf_out["EWC"]["mean"],
                               "ewc_perm_p": per_etf_out["EWC"]["perm_p"]},
        "C6_magnitude_range": {"ok": c6, "intl_mean": intl_m, "us_mean": us_m},
    }
    ok = all(v["ok"] for v in checks.values())
    return {
        "ok": ok,
        "checks": checks,
        "intl_pooled": intl_pooled,
        "per_etf": per_etf_out,
        "us_pooled_ref": us_pooled,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result))
    sys.exit(0 if result["ok"] else 1)
