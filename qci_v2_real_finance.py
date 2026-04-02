#!/usr/bin/env python3
"""
QCI v2 on REAL Finance Data — Process-Level Null Validation

Uses the same pipeline as the validated script 35, but adds
surrogate comparison with CORRECTLY DESIGNED null models.

Null models:
  A. Phase-randomized surrogates (preserves spectrum, destroys phase coupling)
  B. AR(1)-fitted surrogates (preserves autocorrelation per asset)
  C. Block-shuffled (preserves local structure, destroys long-range)
  D. Row-permuted (shuffles time, destroys all temporal structure)

Stress is ALWAYS defined the same way: future realized vol > 75th pctile.
This is NOT circular — future vol is a forward-looking, independent target.

Pipeline (identical to script 35):
  - 6 assets, 10y daily, 5d log returns, rolling z-score
  - K-means K=6, mod=24, QCI window=63
  - OOS: second half only
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
import yfinance as yf

# FIXED PARAMETERS (same as script 35)
ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
PERIOD = "10y"
MODULUS = 24
N_CLUSTERS = 6
CLUSTER_MAP = {0: 8, 1: 16, 2: 24, 3: 5, 4: 3, 5: 11}  # domain-tuned (as in script 35)
QCI_WINDOW = 63
N_SURROGATES = 200


def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


def compute_qci_pipeline(ret_matrix, seed=42):
    """Full pipeline: z-score → cluster → T-match → rolling QCI.
    Returns (qci_oos, dates_oos) for the OOS period."""
    n = len(ret_matrix)
    half = n // 2

    # Standardize
    std = (ret_matrix - ret_matrix.rolling(QCI_WINDOW).mean()) / (ret_matrix.rolling(QCI_WINDOW).std() + 1e-10)
    std = std.dropna()

    # K-means
    np.random.seed(seed)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)
    km.fit(std.iloc[:half].values)
    labels = km.predict(std.values)

    # T-match
    t_match = []
    for t in range(len(labels) - 2):
        b_t = CLUSTER_MAP[labels[t]]
        e_t = CLUSTER_MAP[labels[t + 1]]
        pred = qa_mod(b_t + e_t, MODULUS)
        actual = CLUSTER_MAP[labels[t + 2]]
        t_match.append(1 if pred == actual else 0)

    t_match = pd.Series(t_match, index=std.index[:-2])

    # Rolling QCI
    qci = t_match.rolling(QCI_WINDOW, min_periods=QCI_WINDOW // 2).mean()

    # OOS
    oos_start = std.index[half]
    qci_oos = qci[qci.index >= oos_start].dropna()

    return qci_oos


def generate_surrogate(ret_df, method):
    """Generate surrogate return matrix preserving different properties."""
    values = ret_df.values.copy()
    T, D = values.shape

    if method == 'phase_randomized':
        out = np.zeros_like(values)
        # Use same random phases across assets to preserve cross-correlation structure
        n_freq = T // 2 + 1
        rand_phases = np.random.uniform(0, 2 * np.pi, n_freq)
        rand_phases[0] = 0
        if T % 2 == 0:
            rand_phases[-1] = 0
        for d in range(D):
            ft = np.fft.rfft(values[:, d])
            ft_surr = np.abs(ft) * np.exp(1j * (np.angle(ft) + rand_phases[:len(ft)]))
            out[:, d] = np.fft.irfft(ft_surr, n=T)
        return pd.DataFrame(out, index=ret_df.index, columns=ret_df.columns)

    elif method == 'ar1':
        out = np.zeros_like(values)
        for d in range(D):
            col = values[:, d]
            phi = np.corrcoef(col[:-1], col[1:])[0, 1]
            sigma = np.std(col) * np.sqrt(max(1 - phi * phi, 0.01))
            out[0, d] = col[0]
            for t in range(1, T):
                out[t, d] = phi * out[t-1, d] + np.random.randn() * sigma
        return pd.DataFrame(out, index=ret_df.index, columns=ret_df.columns)

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
        return pd.DataFrame(out, index=ret_df.index, columns=ret_df.columns)

    elif method == 'row_permuted':
        perm = np.random.permutation(T)
        out = values[perm]
        return pd.DataFrame(out, index=ret_df.index, columns=ret_df.columns)


def main():
    print('=' * 70)
    print('QCI v2 — REAL FINANCE DATA + PROCESS-LEVEL NULLS')
    print('=' * 70)
    print()

    # Fetch real data
    print('Fetching real market data...')
    prices = pd.DataFrame()
    for t in ASSETS:
        try:
            df = yf.download(t, period=PERIOD, progress=False)
            if "Close" in df.columns:
                prices[t] = df["Close"]
            elif ("Close", t) in df.columns:
                prices[t] = df[("Close", t)]
        except Exception as e:
            print(f'  Warning: {t} failed: {e}')
    prices = prices.dropna()
    print(f'  Got {len(prices)} days, {len(prices.columns)} assets')
    print(f'  Period: {prices.index[0].date()} to {prices.index[-1].date()}')

    # Returns
    log_ret_5d = np.log(prices / prices.shift(5)).dropna()
    ret_vec = (log_ret_5d - log_ret_5d.rolling(QCI_WINDOW).mean()) / (log_ret_5d.rolling(QCI_WINDOW).std() + 1e-10)
    ret_vec = ret_vec.dropna()

    # Future vol target (NOT circular — forward looking)
    spy = prices[prices.columns[0]]
    daily_ret = np.log(spy / spy.shift(1))
    rv_future = daily_ret.shift(-21).rolling(21).std() * np.sqrt(252)

    # Real QCI
    print('\nComputing real QCI...')
    real_qci = compute_qci_pipeline(log_ret_5d)

    # Align
    common = real_qci.index.intersection(rv_future.dropna().index)
    n_oos = len(common)
    real_qci_al = real_qci.loc[common]
    rv_fut_al = rv_future.loc[common]

    real_r, real_p = stats.pearsonr(real_qci_al, rv_fut_al)
    print(f'\n  REAL QCI vs future vol (OOS, n={n_oos}):')
    print(f'    r = {real_r:+.4f}, p = {real_p:.6f}')

    # Partial correlation (control for lagged RV)
    rv_current = daily_ret.rolling(21).std() * np.sqrt(252)
    rv_curr_al = rv_current.loc[common]

    # Partial r: residualize both QCI and future vol on current RV
    from numpy.linalg import lstsq
    X = rv_curr_al.values.reshape(-1, 1)
    mask = ~(np.isnan(X.ravel()) | np.isnan(real_qci_al.values) | np.isnan(rv_fut_al.values))
    X_clean = np.column_stack([X[mask], np.ones(mask.sum())])
    qci_resid = real_qci_al.values[mask] - X_clean @ lstsq(X_clean, real_qci_al.values[mask], rcond=None)[0]
    fut_resid = rv_fut_al.values[mask] - X_clean @ lstsq(X_clean, rv_fut_al.values[mask], rcond=None)[0]
    partial_r, partial_p = stats.pearsonr(qci_resid, fut_resid)
    print(f'    Partial r (beyond lagged RV) = {partial_r:+.4f}, p = {partial_p:.6f}')

    # ── SURROGATE COMPARISON ──
    print(f'\n  Running {N_SURROGATES} surrogates per type...')
    surrogate_methods = ['phase_randomized', 'ar1', 'block_shuffled', 'row_permuted']

    for method in surrogate_methods:
        surr_rs = []
        surr_partial_rs = []
        for i in range(N_SURROGATES):
            np.random.seed(42 + i * 7 + hash(method) % 10000)
            surr_ret = generate_surrogate(log_ret_5d, method)
            try:
                surr_qci = compute_qci_pipeline(surr_ret, seed=42 + i)
                surr_common = surr_qci.index.intersection(rv_future.dropna().index)
                if len(surr_common) < 50:
                    continue
                sr, _ = stats.pearsonr(surr_qci.loc[surr_common], rv_fut_al.loc[surr_common])
                surr_rs.append(sr)

                # Partial r for surrogate
                X_s = rv_curr_al.loc[surr_common].values.reshape(-1, 1)
                mask_s = ~(np.isnan(X_s.ravel()) | np.isnan(surr_qci.loc[surr_common].values) | np.isnan(rv_fut_al.loc[surr_common].values))
                if mask_s.sum() < 30:
                    continue
                X_sc = np.column_stack([X_s[mask_s], np.ones(mask_s.sum())])
                q_res = surr_qci.loc[surr_common].values[mask_s] - X_sc @ lstsq(X_sc, surr_qci.loc[surr_common].values[mask_s], rcond=None)[0]
                f_res = rv_fut_al.loc[surr_common].values[mask_s] - X_sc @ lstsq(X_sc, rv_fut_al.loc[surr_common].values[mask_s], rcond=None)[0]
                spr, _ = stats.pearsonr(q_res, f_res)
                surr_partial_rs.append(spr)
            except Exception:
                continue

        if surr_rs:
            surr_rs = np.array(surr_rs)
            mean_r = surr_rs.mean()
            std_r = surr_rs.std()
            z = (real_r - mean_r) / (std_r + 1e-10)
            rank_p = np.mean(np.abs(surr_rs) >= np.abs(real_r))

            print(f'\n  {method} ({len(surr_rs)} valid):')
            print(f'    E[r_null] = {mean_r:+.4f} ± {std_r:.4f}')
            print(f'    Real r = {real_r:+.4f}, z = {z:+.2f}, rank_p = {rank_p:.4f}')
            beaten = rank_p < 0.05
            print(f'    Beats null? {"YES ✓" if beaten else "NO ✗"}')

            if surr_partial_rs:
                spr_arr = np.array(surr_partial_rs)
                z_partial = (partial_r - spr_arr.mean()) / (spr_arr.std() + 1e-10)
                rank_p_partial = np.mean(np.abs(spr_arr) >= np.abs(partial_r))
                print(f'    Partial r: real={partial_r:+.4f}, null_mean={spr_arr.mean():+.4f}, z={z_partial:+.2f}, rank_p={rank_p_partial:.4f}')
                print(f'    Partial beats null? {"YES ✓" if rank_p_partial < 0.05 else "NO ✗"}')

    print(f'\n{"="*70}')
    print('DONE. Check if real QCI beats ALL surrogate types.')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
