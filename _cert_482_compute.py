#!/usr/bin/env python3
"""Compute: BTC-USD a<=6 Daily Direction [482]
Replicates cert [461] a=b+2e<=6 operator on Bitcoin and Ethereum.
Rank-bin approach: floor(rank*27/N); consecutive (b,e) pairs; a=b+2e.
"""
import random
import numpy as np
import yfinance as yf

MOD    = 27
N_PERM = 5000
SEED   = 42
OOS_DATE = "2020-01-01"

TICKERS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}


def _rank_bins(closes):
    """floor(rank * 27 / N) where rank is 0-indexed."""
    n = len(closes)
    order = sorted(range(n), key=lambda i: closes[i])
    ranks = [0] * n
    for rk, idx in enumerate(order):
        ranks[idx] = rk
    return [r * MOD // n for r in ranks]  # integer arithmetic only


def analyze(ticker, name):
    print(f"\n=== {name} ({ticker}) ===")
    df = yf.download(ticker, start="2015-01-01", end="2026-06-18", progress=False)
    closes = [float(c) for c in df[("Close", ticker)].dropna().tolist()]
    dates  = list(df[("Close", ticker)].dropna().index)
    n = len(closes)
    print(f"  N={n} daily closes ({dates[0].date()} to {dates[-1].date()})")

    bins = _rank_bins(closes)

    results = []
    for t in range(n - 2):
        b = bins[t]
        e = bins[t + 1]
        a = b + 2 * e          # A2: always derived; raw (not mod-reduced)
        # Next-day return (percent)
        ret = (closes[t + 2] / closes[t + 1] - 1) * 100
        is_oos = dates[t + 1].strftime("%Y-%m-%d") >= OOS_DATE
        results.append({"b": b, "e": e, "a": a, "ret": ret, "oos": is_oos})

    signal  = [r for r in results if r["a"] <= 6]
    nonsig  = [r for r in results if r["a"] > 6]
    sig_oos = [r for r in signal if r["oos"]]
    sig_is  = [r for r in signal if not r["oos"]]

    n_sig = len(signal)
    mean_sig  = sum(r["ret"] for r in signal) / n_sig if signal else 0
    mean_base = sum(r["ret"] for r in results) / len(results)
    mean_ns   = sum(r["ret"] for r in nonsig) / len(nonsig) if nonsig else 0
    mean_oos  = sum(r["ret"] for r in sig_oos) / len(sig_oos) if sig_oos else 0
    mean_is   = sum(r["ret"] for r in sig_is) / len(sig_is) if sig_is else 0

    print(f"  Signal (a<=6): n={n_sig} ({n_sig/len(results)*100:.1f}%)")
    print(f"  Base mean: {mean_base:+.4f}%  Signal mean: {mean_sig:+.4f}%  NonSig: {mean_ns:+.4f}%")
    print(f"  IS ({OOS_DATE[:4]}-): n={len(sig_is)} mean={mean_is:+.4f}%")
    print(f"  OOS ({OOS_DATE[:4]}+): n={len(sig_oos)} mean={mean_oos:+.4f}%")

    # Permutation test
    all_rets = [r["ret"] for r in results]
    random.seed(SEED)
    perm_means = []
    for _ in range(N_PERM):
        random.shuffle(all_rets)
        perm_means.append(sum(all_rets[:n_sig]) / n_sig)
    perm_p = sum(1 for m in perm_means if m >= mean_sig) / N_PERM
    print(f"  Permutation p (one-sided): {perm_p:.4f}")

    # Directional accuracy
    n_up = sum(1 for r in signal if r["ret"] > 0)
    pct_up = n_up / n_sig * 100 if signal else 0
    print(f"  Directional: {n_up}/{n_sig} = {pct_up:.1f}% positive next day")

    return {
        "ticker": ticker, "name": name, "n_total": n,
        "n_signal": n_sig, "signal_pct": round(n_sig/len(results)*100, 2),
        "mean_base": round(mean_base, 4), "mean_signal": round(mean_sig, 4),
        "mean_nonsig": round(mean_ns, 4),
        "mean_is": round(mean_is, 4), "n_is": len(sig_is),
        "mean_oos": round(mean_oos, 4), "n_oos": len(sig_oos),
        "perm_p": round(perm_p, 4),
        "pct_up": round(pct_up, 2), "n_up": n_up,
        "signal_excess": round(mean_sig - mean_base, 4),
    }


if __name__ == "__main__":
    print("=== Crypto a<=6 Operator Compute [482] ===")
    results = {}
    for ticker, name in TICKERS.items():
        results[ticker] = analyze(ticker, name)

    print("\n=== Summary ===")
    for ticker, r in results.items():
        print(f"  {ticker}: signal={r['n_signal']}d mean={r['mean_signal']:+.4f}% "
              f"base={r['mean_base']:+.4f}% excess={r['signal_excess']:+.4f}% "
              f"p={r['perm_p']:.4f} OOS={r['mean_oos']:+.4f}%")

    print("\n=== Fallback values ===")
    for ticker, r in results.items():
        print(f"  {ticker}: {r}")
