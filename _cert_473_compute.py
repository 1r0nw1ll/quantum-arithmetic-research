#!/usr/bin/env python3
"""Compute OOS holdout numbers for cert [473].

OOS split: 2016-01-01 and later.
Signals: a<=6 daily (b+2e<=6) and (0,0) crash pair.
"""
import json, math, random, datetime, urllib.request

MOD = 27
SEED = 42
N_PERM = 5000
OOS_START = "2016-01-01"
US_TICKERS   = ["^GSPC", "^IXIC", "^DJI", "QQQ", "SPY"]
INTL_TICKERS = ["EWJ", "EWG", "EWL", "EWU", "EWA", "EWC"]


def _fetch(ticker):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&range=25y")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    raw = json.loads(resp.read())
    r = raw["chart"]["result"][0]
    ts   = r["timestamp"]
    cls  = r["indicators"]["adjclose"][0]["adjclose"]
    dates, rets = [], []
    for i in range(1, len(cls)):
        if cls[i] and cls[i-1]:
            dates.append(datetime.date.fromtimestamp(ts[i]).isoformat())
            rets.append(math.log(cls[i] / cls[i-1]))
    return dates, rets


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
    return round(ct/N_PERM, 4)


def _compute_group(tickers):
    # a<=6 signal: b+2e <= 6 (raw A2)
    a6_oos_sig, a6_oos_ctrl = [], []
    a6_is_sig,  a6_is_ctrl  = [], []
    # crash pair (0,0): b==0 AND e==0
    cp_oos_sig, cp_oos_ctrl = [], []
    cp_is_sig,  cp_is_ctrl  = [], []

    for tk in tickers:
        print(f"  fetching {tk}...", flush=True)
        try: dates, rets = _fetch(tk)
        except Exception as exc:
            print(f"  ERROR {tk}: {exc}"); continue
        bins = _to_bins(rets)
        n = len(rets)
        for t in range(1, n-1):
            b, e = bins[t-1], bins[t]
            a = b + 2*e            # raw A2
            cp_flag = (b == 0 and e == 0)
            a6_flag = (a <= 6)
            nxt = rets[t+1] if t+1 < n else None
            if nxt is None: continue
            oos = dates[t] >= OOS_START
            if oos:
                if a6_flag: a6_oos_sig.append(nxt)
                else:        a6_oos_ctrl.append(nxt)
                if cp_flag: cp_oos_sig.append(nxt)
                else:        cp_oos_ctrl.append(nxt)
            else:
                if a6_flag: a6_is_sig.append(nxt)
                else:        a6_is_ctrl.append(nxt)
                if cp_flag: cp_is_sig.append(nxt)
                else:        cp_is_ctrl.append(nxt)

    return {
        "a6": {
            "oos": {"n": len(a6_oos_sig), "mean": round(_mean(a6_oos_sig),5),
                    "perm_p": _perm(a6_oos_sig, a6_oos_ctrl)},
            "is":  {"n": len(a6_is_sig),  "mean": round(_mean(a6_is_sig),5),
                    "perm_p": _perm(a6_is_sig, a6_is_ctrl)},
        },
        "cp": {
            "oos": {"n": len(cp_oos_sig), "mean": round(_mean(cp_oos_sig),5),
                    "perm_p": _perm(cp_oos_sig, cp_oos_ctrl)},
            "is":  {"n": len(cp_is_sig),  "mean": round(_mean(cp_is_sig),5),
                    "perm_p": _perm(cp_is_sig, cp_is_ctrl)},
        },
    }


if __name__ == "__main__":
    print("=== US ===")
    us = _compute_group(US_TICKERS)
    print(json.dumps(us, indent=2))
    print("=== INTL ===")
    intl = _compute_group(INTL_TICKERS)
    print(json.dumps(intl, indent=2))
    print("=== DONE ===")
