#!/usr/bin/env python3
"""Compute cross-asset transfer numbers for cert [474].

Asset classes:
  Bonds:       TLT (20y Treasury), IEF (7-10y Treasury)
  Gold:        GLD
  Commodities: USO (crude oil), DBA (agricultural)
  REIT:        VNQ

Signals: a<=6 (b+2e<=6 raw A2) and crash pair (b=0 AND e=0).
Target: next-day return.
"""
import json, math, random, urllib.request

MOD = 27
SEED = 42
N_PERM = 5000

CROSS_TICKERS = {
    "TLT":  "bonds_lt",
    "IEF":  "bonds_mt",
    "GLD":  "gold",
    "USO":  "crude",
    "DBA":  "agri",
    "VNQ":  "reit",
}


def _fetch(ticker):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&range=25y")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    raw = json.loads(resp.read())
    r = raw["chart"]["result"][0]
    cls = r["indicators"]["adjclose"][0]["adjclose"]
    rets = []
    for i in range(1, len(cls)):
        if cls[i] and cls[i-1]:
            rets.append(math.log(cls[i] / cls[i-1]))
    return rets


def _to_bins(rets):
    n = len(rets)
    si = sorted(range(n), key=lambda i: rets[i])
    rk = [0]*n
    for rank, idx in enumerate(si): rk[idx] = rank
    return [int(math.floor(r * MOD / n)) for r in rk]


def _mean(xs): return sum(xs)/len(xs) if xs else 0.0
def _std(xs):
    if len(xs) < 2: return 1e-9
    mu = _mean(xs)
    return math.sqrt(_mean([(x-mu)**2 for x in xs]))


def _perm(g1, g2):
    if len(g1) < 5 or len(g2) < 5: return 1.0
    obs = _mean(g1) - _mean(g2)
    pool = g1 + g2; n1 = len(g1)
    random.seed(SEED); ct = 0
    for _ in range(N_PERM):
        sh = pool[:]; random.shuffle(sh)
        if abs(_mean(sh[:n1]) - _mean(sh[n1:])) >= abs(obs): ct += 1
    return round(ct / N_PERM, 4)


def _compute_ticker(tk):
    try:
        rets = _fetch(tk)
    except Exception as e:
        return None, str(e)
    bins = _to_bins(rets)
    n = len(rets)
    a6_sig, a6_ctrl = [], []
    cp_sig,  cp_ctrl  = [], []
    for t in range(1, n-1):
        b, e = bins[t-1], bins[t]
        a = b + 2*e
        nxt = rets[t+1] if t+1 < n else None
        if nxt is None: continue
        if a <= 6:
            a6_sig.append(nxt)
        else:
            a6_ctrl.append(nxt)
        if b == 0 and e == 0:
            cp_sig.append(nxt)
        else:
            cp_ctrl.append(nxt)
    return {
        "n": n,
        "a6": {
            "n": len(a6_sig),
            "mean": round(_mean(a6_sig), 5),
            "perm_p": _perm(a6_sig, a6_ctrl),
        },
        "cp": {
            "n": len(cp_sig),
            "mean": round(_mean(cp_sig), 5),
            "perm_p": _perm(cp_sig, cp_ctrl),
        },
    }, None


if __name__ == "__main__":
    results = {}
    for tk, label in CROSS_TICKERS.items():
        print(f"  {tk} ({label})...", flush=True)
        r, err = _compute_ticker(tk)
        if err:
            print(f"    ERROR: {err}")
        else:
            results[tk] = r
            print(f"    a6: n={r['a6']['n']} mean={r['a6']['mean']*100:.3f}% p={r['a6']['perm_p']}")
            print(f"    cp: n={r['cp']['n']} mean={r['cp']['mean']*100:.3f}% p={r['cp']['perm_p']}")
    print()
    print(json.dumps(results, indent=2))
