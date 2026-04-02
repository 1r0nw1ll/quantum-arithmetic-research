#!/usr/bin/env python3
"""
QCI v2 — Process-level null test using EXACT script 35 pipeline.

Zero changes to how QCI is computed. Just adds surrogate comparison
by generating surrogate return series and running the identical pipeline.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=qci_v2_surrogate_test"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")

ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
PERIOD = "10y"
MODULUS = 24
N_CLUSTERS = 6
CLUSTER_MAP = {0: 8, 1: 16, 2: 24, 3: 5, 4: 3, 5: 11}
QCI_WINDOW = 63
N_SURROGATES = 200


def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


def run_qci_pipeline(prices_df):
    """Exact script 35 pipeline. Returns (qci_oos, rv_fut_oos, rv_curr_oos, common)."""
    log_ret_5d = np.log(prices_df / prices_df.shift(5)).dropna()
    std = (log_ret_5d - log_ret_5d.rolling(63).mean()) / (log_ret_5d.rolling(63).std() + 1e-10)
    ret_vec = std.dropna()

    n = len(ret_vec)
    half = n // 2

    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    kmeans.fit(ret_vec.iloc[:half].values)
    all_labels = kmeans.predict(ret_vec.values)

    dates = ret_vec.index
    t_match = []
    for t in range(len(all_labels) - 2):
        b_t = CLUSTER_MAP[all_labels[t]]
        e_t = CLUSTER_MAP[all_labels[t + 1]]
        actual_e_next = CLUSTER_MAP[all_labels[t + 2]]
        pred_e_next = qa_mod(b_t + e_t, MODULUS)
        t_match.append(1 if pred_e_next == actual_e_next else 0)

    t_match = pd.Series(t_match, index=dates[:-2], name="t_match")
    qci = t_match.rolling(QCI_WINDOW, min_periods=QCI_WINDOW // 2).mean()
    qci.name = "QCI"

    spy = prices_df.iloc[:, 0]
    daily_ret = np.log(spy / spy.shift(1))
    rv_current = daily_ret.rolling(21).std() * np.sqrt(252)
    rv_future = daily_ret.shift(-21).rolling(21).std() * np.sqrt(252)

    oos_start = dates[half]
    common = qci.dropna().index.intersection(rv_current.dropna().index).intersection(rv_future.dropna().index)
    common = common[common >= oos_start]

    return qci.loc[common], rv_future.loc[common], rv_current.loc[common], common


def generate_surrogate_prices(prices_df, method):
    """Generate surrogate price series from real prices."""
    # Work with log returns to preserve structure
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    values = log_ret.values.copy()
    T, D = values.shape

    if method == 'phase_randomized':
        out = np.zeros_like(values)
        n_freq = T // 2 + 1
        rand_phases = np.random.uniform(0, 2 * np.pi, n_freq)
        rand_phases[0] = 0
        if T % 2 == 0:
            rand_phases[-1] = 0
        for d in range(D):
            ft = np.fft.rfft(values[:, d])
            ft_surr = np.abs(ft) * np.exp(1j * (np.angle(ft) + rand_phases[:len(ft)]))
            out[:, d] = np.fft.irfft(ft_surr, n=T)
        surr_ret = pd.DataFrame(out, index=log_ret.index, columns=log_ret.columns)

    elif method == 'ar1':
        out = np.zeros_like(values)
        for d in range(D):
            col = values[:, d]
            phi = np.corrcoef(col[:-1], col[1:])[0, 1]
            sigma = np.std(col) * np.sqrt(max(1 - phi * phi, 0.01))
            out[0, d] = col[0]
            for t in range(1, T):
                out[t, d] = phi * out[t-1, d] + np.random.randn() * sigma
        surr_ret = pd.DataFrame(out, index=log_ret.index, columns=log_ret.columns)

    elif method == 'block_shuffled':
        block_size = 50
        n_blocks = T // block_size
        perm = np.random.permutation(n_blocks)
        out = np.zeros_like(values)
        for i, idx in enumerate(perm):
            src = idx * block_size
            dst = i * block_size
            length = min(block_size, T - dst, T - src)
            out[dst:dst+length] = values[src:src+length]
        rem = T - n_blocks * block_size
        if rem > 0:
            out[-rem:] = values[-rem:]
        surr_ret = pd.DataFrame(out, index=log_ret.index, columns=log_ret.columns)

    elif method == 'row_permuted':
        perm = np.random.permutation(T)
        out = values[perm]
        surr_ret = pd.DataFrame(out, index=log_ret.index, columns=log_ret.columns)

    # Reconstruct prices from surrogate returns (vectorized)
    surr_cum = surr_ret.cumsum()
    surr_prices = prices_df.iloc[0:1].values * np.exp(surr_cum.values)
    surr_prices = pd.DataFrame(surr_prices, index=surr_ret.index, columns=surr_ret.columns)
    # Prepend the first real price row
    first_row = prices_df.iloc[0:1].copy()
    surr_prices = pd.concat([first_row, surr_prices])

    return surr_prices


def main():
    print("=" * 70)
    print("QCI v2 — EXACT SCRIPT 35 PIPELINE + PROCESS-LEVEL NULLS")
    print("=" * 70)

    print("\nFetching data...")
    prices = pd.DataFrame()
    for t in ASSETS:
        try:
            df = yf.download(t, period=PERIOD, progress=False)
            if "Close" in df.columns:
                prices[t] = df["Close"]
            elif ("Close", t) in df.columns:
                prices[t] = df[("Close", t)]
        except:
            pass
    prices = prices.dropna()
    print(f"  {len(prices)} days, {len(prices.columns)} assets")
    print(f"  {prices.index[0].date()} to {prices.index[-1].date()}")

    # ── REAL QCI (exact script 35) ──
    print("\nRunning exact script 35 pipeline on real data...")
    qci_oos, rv_fut_oos, rv_curr_oos, common = run_qci_pipeline(prices)

    real_r, real_p = stats.pearsonr(qci_oos, rv_fut_oos)
    print(f"\n  REAL RESULT:")
    print(f"    QCI vs future vol: r = {real_r:+.4f}, p = {real_p:.8f}")
    print(f"    OOS days: {len(common)}")
    print(f"    QCI mean={qci_oos.mean():.4f}, std={qci_oos.std():.4f}")

    # Partial r beyond lagged RV
    from numpy.linalg import lstsq
    mask = ~(qci_oos.isna() | rv_fut_oos.isna() | rv_curr_oos.isna())
    X = np.column_stack([rv_curr_oos[mask].values, np.ones(mask.sum())])
    qci_resid = qci_oos[mask].values - X @ lstsq(X, qci_oos[mask].values, rcond=None)[0]
    fut_resid = rv_fut_oos[mask].values - X @ lstsq(X, rv_fut_oos[mask].values, rcond=None)[0]
    partial_r, partial_p = stats.pearsonr(qci_resid, fut_resid)
    print(f"    Partial r (beyond RV): {partial_r:+.4f}, p = {partial_p:.8f}")

    # ── SURROGATE COMPARISON ──
    methods = ['phase_randomized', 'ar1', 'block_shuffled', 'row_permuted']

    for method in methods:
        print(f"\n  Surrogates: {method} ({N_SURROGATES} runs)...")
        surr_rs = []
        surr_partial_rs = []

        for i in range(N_SURROGATES):
            np.random.seed(42 + i * 13 + hash(method) % 9999)
            try:
                surr_prices = generate_surrogate_prices(prices, method)
                surr_qci, surr_fut, surr_curr, surr_common = run_qci_pipeline(surr_prices)

                if len(surr_common) < 50:
                    continue

                sr, _ = stats.pearsonr(surr_qci, surr_fut)
                surr_rs.append(sr)

                # Partial r
                sm = ~(surr_qci.isna() | surr_fut.isna() | surr_curr.isna())
                if sm.sum() < 30:
                    continue
                Xs = np.column_stack([surr_curr[sm].values, np.ones(sm.sum())])
                qr = surr_qci[sm].values - Xs @ lstsq(Xs, surr_qci[sm].values, rcond=None)[0]
                fr = surr_fut[sm].values - Xs @ lstsq(Xs, surr_fut[sm].values, rcond=None)[0]
                spr, _ = stats.pearsonr(qr, fr)
                surr_partial_rs.append(spr)

            except Exception as e:
                continue

            if (i + 1) % 50 == 0:
                print(f"    ...{i+1}/{N_SURROGATES} done")

        if not surr_rs:
            print(f"    No valid surrogates for {method}")
            continue

        surr_rs = np.array(surr_rs)
        mean_r = surr_rs.mean()
        std_r = surr_rs.std()
        z = (real_r - mean_r) / (std_r + 1e-10)
        rank_p = np.mean(np.abs(surr_rs) >= np.abs(real_r))

        print(f"    n={len(surr_rs)}, E[r_null]={mean_r:+.4f} ± {std_r:.4f}")
        print(f"    Real r={real_r:+.4f}, z={z:+.2f}, rank_p={rank_p:.4f}")
        print(f"    Beats null: {'YES ✓' if rank_p < 0.05 else 'NO ✗'}")

        if surr_partial_rs:
            spr_arr = np.array(surr_partial_rs)
            z_p = (partial_r - spr_arr.mean()) / (spr_arr.std() + 1e-10)
            rp_p = np.mean(np.abs(spr_arr) >= np.abs(partial_r))
            print(f"    Partial: real={partial_r:+.4f}, null_mean={spr_arr.mean():+.4f}, "
                  f"z={z_p:+.2f}, rank_p={rp_p:.4f}")
            print(f"    Partial beats null: {'YES ✓' if rp_p < 0.05 else 'NO ✗'}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
