#!/usr/bin/env python3
"""
finance_generator_inference.py — [209] Generator Inference on Finance Returns

Applies QA Signal Generator Inference to multi-asset return series.
Replaces QUINTILE_TO_STATE hardcoded lookup with canonical generator inference.

For each asset's daily return series:
  1. Quantize to {1,...,9} using expanding-window percentile bins
  2. Infer generators: e_t = ((b_{t+1} - b_t - 1) % 9) + 1
  3. Generator distribution = QA fingerprint of the asset's dynamics
  4. Cross-asset synchrony = coupling metric per [207]

Hypothesis: generator synchrony predicts future volatility (the QCI claim,
now grounded in [209] instead of hardcoded CMAP).

Uses yfinance for real data, falls back to synthetic if unavailable.
"""

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "observer": "signal_generator_inference_cert_209",
    "b_meaning": "quantized_daily_return",
    "e_meaning": "inferred_generator_from_return_evolution",
    "axioms_checked": ["A1", "A2", "S1", "S2", "T1", "T2"],
}

import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import pearsonr

np.random.seed(42)

MOD = 9
WINDOW = 63       # rolling window for synchrony (~ 1 quarter)
FH = 21           # forecast horizon for vol prediction (~ 1 month)
MIN_HISTORY = 20  # minimum days for percentile estimation
TICKERS = ["SPY", "TLT", "GLD", "USO", "UUP", "VIX"]  # noqa: T2-D-5 — observer data labels


# ── Core [209] Functions ─────────────────────────────────────────────────────

def infer_generator(b_t: int, b_next: int, m: int = MOD) -> int:
    """Unique A1-compliant generator: QA step inversion."""
    return ((b_next - b_t - 1) % m) + 1


def quantize_expanding_window(returns: np.ndarray, m: int = MOD) -> list[int]:
    """
    Quantize returns using expanding-window percentiles (no look-ahead).
    Observer layer: continuous -> {1,...,m}. Boundary crossed here.
    """
    n = len(returns)
    quantized = []
    edges_pct = [100 * k / m for k in range(1, m)]  # m-1 edges for m bins

    for i in range(n):
        if i < MIN_HISTORY:
            quantized.append((m + 1) // 2)  # middle bin
            continue
        history = returns[:i]
        edges = [float(np.percentile(history, p)) for p in edges_pct]
        val = float(returns[i])
        b = 1
        for edge in edges:
            if val > edge:
                b += 1
        quantized.append(b)

    return quantized


def generator_stats(quantized: list[int], m: int = MOD) -> dict:
    """Generator distribution stats for one series."""
    n = len(quantized)
    if n < 2:
        return {"entropy": 0.0, "mean_e": 0.0, "identity_frac": 0.0}

    generators = [infer_generator(quantized[t], quantized[t + 1], m)
                  for t in range(n - 1)]

    counts = [0] * m
    for g in generators:
        counts[g - 1] += 1
    total = len(generators)
    dist = [c / total for c in counts]

    entropy = -sum(p * np.log2(p) for p in dist if p > 0)
    identity_frac = counts[m - 1] / total

    return {
        "entropy": entropy,
        "mean_e": float(np.mean(generators)),
        "identity_frac": identity_frac,
        "generators": generators,
    }


def rolling_synchrony(all_quantized: list[list[int]], window: int = WINDOW,
                      m: int = MOD) -> list[float]:
    """Rolling cross-asset generator synchrony."""
    n_assets = len(all_quantized)
    min_len = min(len(q) for q in all_quantized) - 1
    if min_len < window:
        return []

    synch_series = []
    for t in range(window, min_len):
        scores = []
        for step in range(t - window, t):
            gens = [infer_generator(all_quantized[i][step],
                                    all_quantized[i][step + 1], m)
                    for i in range(n_assets)]
            modal = Counter(gens).most_common(1)[0][1] / n_assets
            scores.append(modal)
        synch_series.append(float(np.mean(scores)))

    return synch_series


def rolling_realized_vol(returns_dict: dict, window: int = WINDOW) -> list[float]:
    """Rolling realized volatility (cross-asset mean abs return)."""
    arrays = list(returns_dict.values())
    min_len = min(len(a) for a in arrays)
    vol_series = []
    for t in range(window, min_len):
        vals = [float(np.std(a[t - window:t])) for a in arrays]
        vol_series.append(float(np.mean(vals)))
    return vol_series


# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_data():
    """Try yfinance, fall back to synthetic."""
    try:
        import yfinance as yf  # noqa: T2-D-5 — observer data fetch
        import pandas as pd
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            closes = {}
            for t in TICKERS:
                df = yf.download(t, start="2010-01-01", end="2025-01-01",
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    closes[t] = df["Close"].squeeze()
            if len(closes) >= 2:
                frame = pd.DataFrame(closes).dropna()
                log_ret = np.log(frame / frame.shift(1)).dropna()
                return {t: log_ret[t].values for t in log_ret.columns}, "real"
    except Exception:
        pass

    # Synthetic fallback
    print("  yfinance unavailable — using synthetic returns")
    rng = np.random.default_rng(42)
    n = 3000
    cov = np.array([
        [1.0, -0.3, 0.1, 0.2, -0.1, 0.4],
        [-0.3, 1.0, 0.2, -0.1, 0.3, -0.2],
        [0.1, 0.2, 1.0, 0.3, -0.2, 0.1],
        [0.2, -0.1, 0.3, 1.0, 0.1, 0.3],
        [-0.1, 0.3, -0.2, 0.1, 1.0, -0.1],
        [0.4, -0.2, 0.1, 0.3, -0.1, 1.0],
    ]) * 0.0001
    mean = np.array([0.0003, -0.0001, 0.0002, 0.0001, 0.0, 0.0])
    samples = rng.multivariate_normal(mean, cov, size=n)
    return {t: samples[:, i] for i, t in enumerate(TICKERS)}, "synthetic"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("[209] FINANCE GENERATOR INFERENCE")
    print("e_t = ((b_{t+1} - b_t - 1) % 9) + 1")
    print("=" * 72)

    returns, data_source = fetch_data()
    tickers = sorted(returns.keys())
    n_days = min(len(returns[t]) for t in tickers)
    print(f"\n  Data: {data_source}, {len(tickers)} assets, {n_days} days")
    print(f"  Assets: {tickers}")

    # Quantize all assets with expanding-window percentiles
    print("\n  Quantizing returns...")
    all_quantized = [quantize_expanding_window(returns[t][:n_days]) for t in tickers]

    # Per-asset generator stats
    print(f"\n  {'Asset':<8} {'Entropy':>8} {'IdentFrac':>10} {'Mean_e':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for i, t in enumerate(tickers):
        stats = generator_stats(all_quantized[i])
        print(f"  {t:<8} {stats['entropy']:>8.4f} {stats['identity_frac']:>10.4f} {stats['mean_e']:>8.4f}")

    # Cross-asset synchrony
    global_synch = []
    min_len = min(len(q) for q in all_quantized) - 1
    for t in range(min_len):
        gens = [infer_generator(all_quantized[i][t], all_quantized[i][t + 1], MOD)
                for i in range(len(tickers))]
        modal = Counter(gens).most_common(1)[0][1] / len(tickers)
        global_synch.append(modal)
    mean_synch = float(np.mean(global_synch))
    print(f"\n  Global generator synchrony: {mean_synch:.4f}")
    print(f"  (baseline for {len(tickers)} independent at m=9: {1/MOD:.4f})")

    # Rolling synchrony vs future volatility
    print(f"\n  Computing rolling synchrony (window={WINDOW}) vs future vol (h={FH})...")
    synch_roll = rolling_synchrony(all_quantized, WINDOW)
    vol_roll = rolling_realized_vol(returns, WINDOW)

    # Align: synchrony at time t vs vol at time t+FH
    n_align = min(len(synch_roll), len(vol_roll) - FH)
    if n_align > 50:
        synch_arr = np.array(synch_roll[:n_align])
        future_vol = np.array(vol_roll[FH:FH + n_align])

        r, p = pearsonr(synch_arr, future_vol)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n  Synchrony vs future vol (h={FH}d):")
        print(f"    raw r = {r:+.4f}, p = {p:.6f} {sig}")
        print(f"    n = {n_align}")

        # Partial correlation controlling for current vol
        current_vol = np.array(vol_roll[:n_align])
        # Residualize
        from numpy.polynomial.polynomial import polyfit, polyval
        c_sv = polyfit(current_vol, synch_arr, 1)
        c_fv = polyfit(current_vol, future_vol, 1)
        resid_s = synch_arr - polyval(current_vol, c_sv)
        resid_f = future_vol - polyval(current_vol, c_fv)
        r_partial, p_partial = pearsonr(resid_s, resid_f)
        sig_p = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else "ns"
        print(f"    partial r (beyond current vol) = {r_partial:+.4f}, p = {p_partial:.6f} {sig_p}")
    else:
        r, p, r_partial, p_partial = 0, 1, 0, 1
        print(f"  Insufficient aligned data (n={n_align})")

    # Save
    results = {
        "domain": "finance",
        "data_source": data_source,
        "n_assets": len(tickers),
        "n_days": n_days,
        "tickers": tickers,
        "global_synchrony": mean_synch,
        "synch_vs_future_vol": {"r": float(r), "p": float(p)},
        "partial_r_beyond_vol": {"r": float(r_partial), "p": float(p_partial)},
    }
    out_path = Path("finance_generator_inference_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
