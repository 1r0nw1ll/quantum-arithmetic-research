#!/usr/bin/env python3
"""Compute vol-targeting position sizing numbers for cert [475].

Vol-target weight per trade t: w_t = 1 / std(rets[t-20:t])
(σ_target cancels in Sharpe ratio so we drop it)

Signals: a<=6 (b+2e<=6 raw A2) and crash pair (b=0 AND e=0).
Equal-weight baseline: unscaled rets[t+1].
Vol-targeted: w_t * rets[t+1] where w_t = 1/trailing_21d_vol.
"""
import json, math, random, urllib.request

MOD = 27
SEED = 42
N_PERM = 5000
VOL_WINDOW = 21
US_TICKERS   = ["^GSPC", "^IXIC", "^DJI", "QQQ", "SPY"]
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
    return math.sqrt(_mean([(x - mu) * (x - mu) for x in xs]))
def _sharpe(xs): s = _std(xs); return round(_mean(xs) / s, 4) if s > 1e-9 else 0.0

def _perm_sharpe(vt_sig, eq_sig, vt_ctrl, eq_ctrl):
    """Permutation test: Sharpe(vt_sig) > Sharpe(eq_sig)?
    Test statistic = Sharpe(vt_sig) - Sharpe(eq_sig).
    Null: shuffle labels between vt_sig and eq_sig.
    """
    obs = _sharpe(vt_sig) - _sharpe(eq_sig)
    pool_vt = vt_sig + vt_ctrl
    pool_eq = eq_sig + eq_ctrl
    n1 = len(vt_sig)
    random.seed(SEED); ct = 0
    for _ in range(N_PERM):
        random.shuffle(pool_vt)
        random.shuffle(pool_eq)
        s_null = _sharpe(pool_vt[:n1]) - _sharpe(pool_eq[:n1])
        if s_null >= obs: ct += 1
    return round(ct / N_PERM, 4)


def _compute_group(tickers):
    a6_eq, a6_vt = [], []
    cp_eq, cp_vt = [], []
    a6_eq_ctrl, a6_vt_ctrl = [], []
    cp_eq_ctrl, cp_vt_ctrl = [], []

    for tk in tickers:
        print(f"  {tk}...", flush=True)
        try: rets = _fetch(tk)
        except Exception as e:
            print(f"    ERROR: {e}"); continue
        bins = _to_bins(rets); n = len(rets)

        for t in range(VOL_WINDOW + 1, n - 1):
            b, e = bins[t-1], bins[t]
            a = b + 2*e
            nxt = rets[t+1]
            # trailing vol: std of rets[t-VOL_WINDOW:t] (VOL_WINDOW returns, not incl t)
            window = rets[t - VOL_WINDOW: t]
            vol = _std(window)
            if vol < 1e-9: continue
            w = 1.0 / vol   # vol-target weight (σ_target cancels in Sharpe)
            vt_r = w * nxt  # vol-targeted return

            a6 = (a <= 6)
            cp = (b == 0 and e == 0)
            if a6:
                a6_eq.append(nxt); a6_vt.append(vt_r)
            else:
                a6_eq_ctrl.append(nxt); a6_vt_ctrl.append(vt_r)
            if cp:
                cp_eq.append(nxt); cp_vt.append(vt_r)
            else:
                cp_eq_ctrl.append(nxt); cp_vt_ctrl.append(vt_r)

    return {
        "a6": {
            "eq_sharpe":  _sharpe(a6_eq),
            "vt_sharpe":  _sharpe(a6_vt),
            "eq_mean":    round(_mean(a6_eq), 5),
            "vt_mean":    round(_mean(a6_vt), 5),
            "n":          len(a6_eq),
        },
        "cp": {
            "eq_sharpe":  _sharpe(cp_eq),
            "vt_sharpe":  _sharpe(cp_vt),
            "eq_mean":    round(_mean(cp_eq), 5),
            "vt_mean":    round(_mean(cp_vt), 5),
            "n":          len(cp_eq),
        },
    }


if __name__ == "__main__":
    print("=== US ===")
    us = _compute_group(US_TICKERS)
    print(json.dumps(us, indent=2))
    print("=== INTL ===")
    intl = _compute_group(INTL_TICKERS)
    print(json.dumps(intl, indent=2))
