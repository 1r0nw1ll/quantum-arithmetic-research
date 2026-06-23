#!/usr/bin/env python3
"""Compute empirical numbers for certs [469], [470], [471].
Run once to get fallback values; results printed as JSON.
"""
import json
import math
import random
import urllib.request
from datetime import datetime, timezone, timedelta

MOD = 27
SEED = 42
N_PERM = 5000
US_TICKERS = ["^GSPC", "^IXIC", "^DJI", "QQQ", "SPY"]
IS_CUTOFF = "2015-01-01"


def _fetch_daily(ticker, range_="25y"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&range={range_}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    raw = json.loads(resp.read())
    r = raw["chart"]["result"][0]
    ts = r["timestamp"]
    cls = r["indicators"]["adjclose"][0]["adjclose"]
    dates = [datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d") for t in ts]
    rets, dts = [], []
    for i in range(1, len(cls)):
        if cls[i] and cls[i-1]:
            rets.append(math.log(cls[i] / cls[i-1]))
            dts.append(dates[i])
    return rets, dts


def _fetch_weekly(ticker, range_="27y"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1wk&range={range_}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    raw = json.loads(resp.read())
    r = raw["chart"]["result"][0]
    ts = r["timestamp"]
    cls = r["indicators"]["adjclose"][0]["adjclose"]
    dates = [datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d") for t in ts]
    rets, dts = [], []
    for i in range(1, len(cls)):
        if cls[i] and cls[i-1]:
            rets.append(math.log(cls[i] / cls[i-1]))
            dts.append(dates[i])
    return rets, dts


def _to_bins(rets):
    n = len(rets)
    si = sorted(range(n), key=lambda i: rets[i])
    rk = [0] * n
    for rank, idx in enumerate(si):
        rk[idx] = rank
    return [int(math.floor(r * MOD / n)) for r in rk]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _perm(g1, g2, n_perm=N_PERM, seed=SEED):
    if len(g1) < 5 or len(g2) < 5:
        return 1.0
    obs = _mean(g1) - _mean(g2)
    pool = g1 + g2
    n1 = len(g1)
    random.seed(seed)
    ct = sum(1 for _ in range(n_perm)
             if abs(_mean(random.sample(pool, n1)) - _mean(pool[n1:] if False else
                [x for i,x in enumerate(pool) if i not in set(random.sample(range(len(pool)), n1))]))
                >= abs(obs))
    # Simpler perm
    random.seed(seed)
    ct = 0
    for _ in range(n_perm):
        sh = pool[:]
        random.shuffle(sh)
        diff = _mean(sh[:n1]) - _mean(sh[n1:])
        if abs(diff) >= abs(obs):
            ct += 1
    return round(ct / n_perm, 4)


# ============================================================
# [469] Vol-normalized returns
# ============================================================
def compute_469():
    print("=== [469] Vol-normalized ===")
    VOL_WINDOW = 21
    pooled_sig_raw, pooled_sig_vol, pooled_ctrl_raw, pooled_ctrl_vol = [], [], [], []
    per_idx = {}

    for tk in US_TICKERS:
        rets, dts = _fetch_daily(tk)
        bins = _to_bins(rets)
        n = len(rets)
        sig_raw, sig_vol, ctrl_raw, ctrl_vol = [], [], [], []
        vol_sig_days, vol_ctrl_days = [], []

        for t in range(VOL_WINDOW, n - 1):
            b = bins[t-1]
            e = bins[t]
            a = b + 2*e
            nr = rets[t+1]
            window = rets[t-VOL_WINDOW:t]
            mu = _mean(window)
            vol = math.sqrt(_mean([(x-mu)**2 for x in window]))
            vol = max(vol, 1e-6)
            nr_vol = nr / vol

            if a <= 6:
                sig_raw.append(nr)
                sig_vol.append(nr_vol)
                vol_sig_days.append(vol)
            else:
                ctrl_raw.append(nr)
                ctrl_vol.append(nr_vol)
                vol_ctrl_days.append(vol)

        pp_raw = _perm(sig_raw, ctrl_raw)
        pp_vol = _perm(sig_vol, ctrl_vol)
        mean_vol_sig = _mean(vol_sig_days)
        mean_vol_ctrl = _mean(vol_ctrl_days)

        per_idx[tk] = {
            "n_sig": len(sig_raw),
            "raw_mean": round(_mean(sig_raw), 5),
            "raw_perm_p": pp_raw,
            "vol_mean": round(_mean(sig_vol), 4),
            "vol_perm_p": pp_vol,
            "mean_vol_sig": round(mean_vol_sig, 5),
            "mean_vol_ctrl": round(mean_vol_ctrl, 5),
            "vol_ratio": round(mean_vol_sig / mean_vol_ctrl, 3),
        }
        print(f"  {tk}: n={len(sig_raw)}, raw_mean={_mean(sig_raw)*100:.2f}%, "
              f"raw_p={pp_raw}, vol_mean={_mean(sig_vol):.3f}, vol_p={pp_vol}, "
              f"vol_ratio={mean_vol_sig/mean_vol_ctrl:.3f}")

        pooled_sig_raw += sig_raw
        pooled_sig_vol += sig_vol
        pooled_ctrl_raw += ctrl_raw
        pooled_ctrl_vol += ctrl_vol

    pooled_pp_raw = _perm(pooled_sig_raw, pooled_ctrl_raw)
    pooled_pp_vol = _perm(pooled_sig_vol, pooled_ctrl_vol)
    print(f"  POOLED: n={len(pooled_sig_raw)}, raw_mean={_mean(pooled_sig_raw)*100:.3f}%, "
          f"raw_p={pooled_pp_raw}, vol_mean={_mean(pooled_sig_vol):.4f}, vol_p={pooled_pp_vol}")
    print(f"  Global vol_ratio (sig/ctrl): {_mean([p['vol_ratio'] for p in per_idx.values()]):.3f}")

    result = {
        "per_idx": per_idx,
        "pooled": {
            "n": len(pooled_sig_raw),
            "raw_mean": round(_mean(pooled_sig_raw), 5),
            "raw_perm_p": pooled_pp_raw,
            "vol_mean": round(_mean(pooled_sig_vol), 4),
            "vol_perm_p": pooled_pp_vol,
        }
    }
    return result


# ============================================================
# [470] Crash pair persistence (day+2, day+3)
# ============================================================
def compute_470():
    print("=== [470] Crash pair persistence ===")
    pooled_d1, pooled_d2, pooled_d3, pooled_ctrl = [], [], [], []
    per_idx = {}

    for tk in US_TICKERS:
        rets, _ = _fetch_daily(tk)
        bins = _to_bins(rets)
        n = len(rets)
        d1, d2, d3, ctrl = [], [], [], []

        for t in range(1, n - 3):
            b = bins[t-1]
            e = bins[t]
            nr1 = rets[t+1]
            nr2 = rets[t+2]
            nr3 = rets[t+3]
            if b == 0 and e == 0:
                d1.append(nr1)
                d2.append(nr2)
                d3.append(nr3)
            else:
                ctrl.append(nr1)

        c3 = [d1[i]+d2[i]+d3[i] for i in range(len(d1))]
        pp1 = _perm(d1, ctrl)
        pp2 = _perm(d2, ctrl)
        pp3 = _perm(d3, ctrl)

        # For cum3, compare against random 3-day ctrl sums
        random.seed(SEED)
        ctrl_c3 = []
        for _ in range(len(d1) * 3):
            s = random.sample(ctrl, 3)
            ctrl_c3.append(sum(s))
        ppc3 = _perm(c3, ctrl_c3)

        per_idx[tk] = {
            "n": len(d1),
            "d1": {"mean": round(_mean(d1), 5), "perm_p": pp1},
            "d2": {"mean": round(_mean(d2), 5), "perm_p": pp2},
            "d3": {"mean": round(_mean(d3), 5), "perm_p": pp3},
            "c3": {"mean": round(_mean(c3), 5), "perm_p": ppc3},
        }
        print(f"  {tk}: n={len(d1)}, "
              f"d1={_mean(d1)*100:.2f}%(p={pp1}), "
              f"d2={_mean(d2)*100:.2f}%(p={pp2}), "
              f"d3={_mean(d3)*100:.2f}%(p={pp3}), "
              f"cum3={_mean(c3)*100:.2f}%(p={ppc3})")

        pooled_d1 += d1
        pooled_d2 += d2
        pooled_d3 += d3
        pooled_ctrl += ctrl

    c3_pool = [pooled_d1[i]+pooled_d2[i]+pooled_d3[i] for i in range(len(pooled_d1))]
    random.seed(SEED)
    ctrl_c3_pool = [sum(random.sample(pooled_ctrl, 3)) for _ in range(len(pooled_d1)*3)]
    pp1p = _perm(pooled_d1, pooled_ctrl)
    pp2p = _perm(pooled_d2, pooled_ctrl)
    pp3p = _perm(pooled_d3, pooled_ctrl)
    ppc3p = _perm(c3_pool, ctrl_c3_pool)

    print(f"  POOLED: n={len(pooled_d1)}, "
          f"d1={_mean(pooled_d1)*100:.3f}%(p={pp1p}), "
          f"d2={_mean(pooled_d2)*100:.3f}%(p={pp2p}), "
          f"d3={_mean(pooled_d3)*100:.3f}%(p={pp3p}), "
          f"cum3={_mean(c3_pool)*100:.3f}%(p={ppc3p})")

    return {
        "per_idx": per_idx,
        "pooled": {
            "n": len(pooled_d1),
            "d1": {"mean": round(_mean(pooled_d1), 5), "perm_p": pp1p},
            "d2": {"mean": round(_mean(pooled_d2), 5), "perm_p": pp2p},
            "d3": {"mean": round(_mean(pooled_d3), 5), "perm_p": pp3p},
            "c3": {"mean": round(_mean(c3_pool), 5), "perm_p": ppc3p},
        }
    }


# ============================================================
# [471] Multi-scale alignment
# ============================================================
def _week_start(date_str):
    """Return Monday date of the week containing date_str."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    monday = d - timedelta(days=d.weekday())
    return monday.strftime("%Y-%m-%d")


def compute_471():
    print("=== [471] Multi-scale alignment ===")
    pooled_both, pooled_daily_only, pooled_ctrl = [], [], []
    per_idx = {}

    for tk in US_TICKERS:
        # --- Weekly S-orbit weeks ---
        w_rets, w_dts = _fetch_weekly(tk)
        w_bins = _to_bins(w_rets)
        nw = len(w_rets)
        # S-orbit condition on pair (bins[t-1], bins[t]) predicts week at dts[t+1]
        sorbit_weeks = set()
        for t in range(1, nw - 1):
            bw = w_bins[t-1]
            ew = w_bins[t]
            if bw % 9 == 0 and ew % 9 == 0:
                # Flag the PREDICTED week: the week starting at dts[t+1]
                sorbit_weeks.add(_week_start(w_dts[t+1]))

        # --- Daily a<=6 ---
        d_rets, d_dts = _fetch_daily(tk)
        d_bins = _to_bins(d_rets)
        nd = len(d_rets)

        both, daily_only, ctrl = [], [], []
        for t in range(1, nd - 1):
            b = d_bins[t-1]
            e = d_bins[t]
            a = b + 2*e
            nr = d_rets[t+1]
            day_a6 = (a <= 6)
            day_ws = _week_start(d_dts[t]) in sorbit_weeks

            if day_a6 and day_ws:
                both.append(nr)
            elif day_a6:
                daily_only.append(nr)
            else:
                ctrl.append(nr)

        pp_both = _perm(both, ctrl)
        pp_d6 = _perm(daily_only, ctrl)
        pp_both_vs_d6 = _perm(both, daily_only)

        per_idx[tk] = {
            "n_both": len(both),
            "n_d6": len(daily_only),
            "both_mean": round(_mean(both), 5),
            "d6_mean": round(_mean(daily_only), 5),
            "both_perm_p": pp_both,
            "d6_perm_p": pp_d6,
            "both_vs_d6_perm_p": pp_both_vs_d6,
            "amplification": round((_mean(both) - _mean(daily_only)) * 100, 3) if daily_only else None,
        }
        print(f"  {tk}: n_both={len(both)}, n_d6={len(daily_only)}, "
              f"both={_mean(both)*100:.3f}%(p={pp_both}), "
              f"d6only={_mean(daily_only)*100:.3f}%(p={pp_d6}), "
              f"both_vs_d6_p={pp_both_vs_d6}, "
              f"amp={(_mean(both)-_mean(daily_only))*100:+.3f}%")

        pooled_both += both
        pooled_daily_only += daily_only
        pooled_ctrl += ctrl

    pp_both_p = _perm(pooled_both, pooled_ctrl)
    pp_d6_p = _perm(pooled_daily_only, pooled_ctrl)
    pp_both_vs_d6_p = _perm(pooled_both, pooled_daily_only)

    print(f"  POOLED: n_both={len(pooled_both)}, n_d6={len(pooled_daily_only)}, "
          f"both={_mean(pooled_both)*100:.3f}%(p={pp_both_p}), "
          f"d6only={_mean(pooled_daily_only)*100:.3f}%(p={pp_d6_p}), "
          f"both_vs_d6_p={pp_both_vs_d6_p}")

    return {
        "per_idx": per_idx,
        "pooled": {
            "n_both": len(pooled_both),
            "n_d6": len(pooled_daily_only),
            "both_mean": round(_mean(pooled_both), 5),
            "d6_mean": round(_mean(pooled_daily_only), 5),
            "both_perm_p": pp_both_p,
            "d6_perm_p": pp_d6_p,
            "both_vs_d6_perm_p": pp_both_vs_d6_p,
        }
    }


if __name__ == "__main__":
    results = {}
    results["469"] = compute_469()
    print()
    results["470"] = compute_470()
    print()
    results["471"] = compute_471()
    print()
    print("=== FULL JSON ===")
    print(json.dumps(results, indent=2))
