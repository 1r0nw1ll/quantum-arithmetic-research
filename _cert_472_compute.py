#!/usr/bin/env python3
"""Compute exit strategy metrics for cert [472].
Strategy A: exit at day+1 close (per-trade return = d1)
Strategy B: hold to day+3 close (per-trade return = d1+d2+d3)
Also tests: re-enter at day+2 close, exit day+3 (capture day+3 recovery only).
"""
import json, math, random, urllib.request
from datetime import datetime, timezone

MOD = 27
SEED = 42
N_PERM = 5000
US_TICKERS  = ["^GSPC", "^IXIC", "^DJI", "QQQ", "SPY"]
INTL_TICKERS = ["EWJ", "EWG", "EWL", "EWU", "EWA", "EWC"]

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
def _sharpe(xs): return _mean(xs) / _std(xs) if _std(xs) > 1e-9 else 0.0

def _perm(g1, g2):
    if len(g1) < 5 or len(g2) < 5: return 1.0
    obs = _mean(g1) - _mean(g2)
    pool = g1 + g2; n1 = len(g1)
    random.seed(SEED); ct = 0
    for _ in range(N_PERM):
        sh = pool[:]; random.shuffle(sh)
        if abs(_mean(sh[:n1]) - _mean(sh[n1:])) >= abs(obs): ct += 1
    return round(ct/N_PERM, 4)

def _compute_group(tickers):
    A_all, B_all, C_all, ctrl_all = [], [], [], []  # A=d1, B=cum3, C=d3only (re-entry)
    per_tk = {}
    for tk in tickers:
        try: rets = _fetch(tk)
        except Exception as e: print(f"  SKIP {tk}: {e}"); continue
        bins = _to_bins(rets); n = len(rets)
        A, B, C, ctrl = [], [], [], []
        for t in range(1, n-3):
            b, e = bins[t-1], bins[t]
            d1, d2, d3 = rets[t+1], rets[t+2], rets[t+3]
            if b == 0 and e == 0:
                A.append(d1)
                B.append(d1+d2+d3)
                C.append(d3)       # re-enter after day+2, hold day+3
            else:
                ctrl.append(d1)
        # per-trade Sharpe (no annualization — comparing strategies on same signal set)
        per_tk[tk] = {
            "n": len(A),
            "A": {"mean": round(_mean(A),5), "std": round(_std(A),5),
                  "sharpe": round(_sharpe(A),4), "perm_p": _perm(A, ctrl)},
            "B": {"mean": round(_mean(B),5), "std": round(_std(B),5),
                  "sharpe": round(_sharpe(B),4), "perm_p": _perm(B, ctrl)},
            "C": {"mean": round(_mean(C),5), "std": round(_std(C),5),
                  "sharpe": round(_sharpe(C),4), "perm_p": _perm(C, ctrl)},
        }
        print(f"  {tk}: n={len(A)}, "
              f"A={_mean(A)*100:.2f}%(sh={_sharpe(A):.3f}), "
              f"B={_mean(B)*100:.2f}%(sh={_sharpe(B):.3f}), "
              f"C={_mean(C)*100:.2f}%(sh={_sharpe(C):.3f})")
        A_all+=A; B_all+=B; C_all+=C; ctrl_all+=ctrl

    pooled = {
        "n": len(A_all),
        "A": {"mean": round(_mean(A_all),5), "std": round(_std(A_all),5),
              "sharpe": round(_sharpe(A_all),4), "perm_p": _perm(A_all, ctrl_all)},
        "B": {"mean": round(_mean(B_all),5), "std": round(_std(B_all),5),
              "sharpe": round(_sharpe(B_all),4), "perm_p": _perm(B_all, ctrl_all)},
        "C": {"mean": round(_mean(C_all),5), "std": round(_std(C_all),5),
              "sharpe": round(_sharpe(C_all),4), "perm_p": _perm(C_all, ctrl_all)},
        "A_vs_B_perm_p": _perm(A_all, B_all),
        "A_vs_C_perm_p": _perm(A_all, C_all),
    }
    print(f"  POOLED: n={pooled['n']}, "
          f"A={pooled['A']['mean']*100:.3f}%(sh={pooled['A']['sharpe']:.4f},p={pooled['A']['perm_p']}), "
          f"B={pooled['B']['mean']*100:.3f}%(sh={pooled['B']['sharpe']:.4f},p={pooled['B']['perm_p']}), "
          f"C={pooled['C']['mean']*100:.3f}%(sh={pooled['C']['sharpe']:.4f},p={pooled['C']['perm_p']}), "
          f"A_vs_B_p={pooled['A_vs_B_perm_p']}")
    return {"per_tk": per_tk, "pooled": pooled}

if __name__ == "__main__":
    print("=== US ===")
    us = _compute_group(US_TICKERS)
    print("=== INTL ===")
    intl = _compute_group(INTL_TICKERS)
    print("=== FULL JSON ===")
    print(json.dumps({"us": us, "intl": intl}, indent=2))
