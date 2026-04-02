#!/usr/bin/env python3
"""
49_era5_multilayer.py — Multi-layer observer on ERA5 reanalysis
================================================================

Tests the local/global QCI separation (validated in finance) on ERA5 data.

Local observer: per-gridpoint variability (the 15 channels = 3 vars × 5 locations)
Global observer: cross-location structure (spatial correlation, PCA concentration,
                 cross-location dispersion)

Hypothesis: QCI_gap = QCI_local - QCI_global outperforms QCI_local alone,
mirroring the finance result where gap partial r = -0.42 vs local partial r = -0.22.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=era5_multilayer, state_alphabet=atmospheric_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from numpy.linalg import lstsq

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_CSV = os.path.join(HERE, ".era5_extracted.csv")

MODULUS = 24
N_CLUSTERS = 6
QCI_WINDOW = 63
FH = 21
CMAP = {0: 8, 1: 16, 2: 24, 3: 5, 4: 3, 5: 11}

np.random.seed(42)


def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


def compute_qci(labels, cmap, m, window):
    t_match = []
    for t in range(len(labels) - 2):
        b = cmap.get(labels[t], 5)
        e = cmap.get(labels[t + 1], 5)
        actual = cmap.get(labels[t + 2], 5)
        pred = qa_mod(b + e, m)
        t_match.append(1 if pred == actual else 0)
    return pd.Series(t_match).rolling(window, min_periods=window // 2).mean()


def compute_global_features(df, channels, window=63):
    """Compute global structure features from multi-channel data."""
    n = len(df)
    # Group channels by variable (3 vars × 5 locations)
    var_prefixes = list(set(c.split('_')[0] for c in channels))

    # 1. Cross-location correlation (rolling mean pairwise correlation)
    roll_corr = df[channels].rolling(window).corr()
    # Mean off-diagonal correlation per timestep
    mean_corr = []
    for t in range(n):
        try:
            corr_mat = df[channels].iloc[max(0, t - window):t + 1].corr().values
            np.fill_diagonal(corr_mat, np.nan)
            mean_corr.append(np.nanmean(corr_mat))
        except:
            mean_corr.append(np.nan)

    # 2. PCA concentration (fraction of variance in first PC)
    pca_conc = []
    for t in range(window, n):
        chunk = df[channels].iloc[t - window:t].values
        chunk_clean = chunk[~np.isnan(chunk).any(axis=1)]
        if len(chunk_clean) > 5:
            try:
                pca = PCA(n_components=min(3, len(channels)))
                pca.fit(chunk_clean)
                pca_conc.append(pca.explained_variance_ratio_[0])
            except:
                pca_conc.append(np.nan)
        else:
            pca_conc.append(np.nan)
    pca_conc = [np.nan] * window + pca_conc

    # 3. Cross-location dispersion (mean std across locations for each variable)
    disp = []
    for t in range(n):
        stds = []
        for prefix in var_prefixes:
            var_cols = [c for c in channels if c.startswith(prefix)]
            if len(var_cols) > 1:
                stds.append(df[var_cols].iloc[t].std())
        disp.append(np.mean(stds) if stds else np.nan)

    global_df = pd.DataFrame({
        "spatial_corr": mean_corr,
        "pca_concentration": pca_conc,
        "spatial_dispersion": disp,
    }, index=df.index)

    return global_df


def compute_partial_r(x, y, z):
    """Partial r(x, y | z)."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 100:
        return np.nan, np.nan
    X = np.column_stack([z[valid], np.ones(valid.sum())])
    x_r = x[valid] - X @ lstsq(X, x[valid], rcond=None)[0]
    y_r = y[valid] - X @ lstsq(X, y[valid], rcond=None)[0]
    return stats.pearsonr(x_r, y_r)


def main():
    if not os.path.exists(EXTRACTED_CSV):
        print(f"ERROR: {EXTRACTED_CSV} not found")
        return

    df_raw = pd.read_csv(EXTRACTED_CSV, index_col=0, parse_dates=True)
    channels = list(df_raw.columns)
    print(f"Loaded: {len(df_raw)} days × {len(channels)} channels")

    # ====================================================================
    # LOCAL OBSERVER — same as single-layer (all 15 channels)
    # ====================================================================
    print("\n" + "=" * 70)
    print("LOCAL OBSERVER (15 channels → k-means → QCI)")
    print("=" * 70)

    roll_mean = df_raw[channels].rolling(365, min_periods=180).mean()
    roll_std = df_raw[channels].rolling(365, min_periods=180).std() + 1e-10
    std_df = (df_raw[channels] - roll_mean) / roll_std
    std_df = std_df.dropna()

    n = len(std_df)
    half = n // 2
    dates = std_df.index

    km_local = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km_local.fit(std_df.iloc[:half].values)
    local_labels = km_local.predict(std_df.values)

    qci_local = compute_qci(local_labels, CMAP, MODULUS, QCI_WINDOW)
    qci_local.index = dates[:-2]

    # ====================================================================
    # GLOBAL OBSERVER — spatial structure features
    # ====================================================================
    print("\n" + "=" * 70)
    print("GLOBAL OBSERVER (spatial correlation, PCA, dispersion)")
    print("=" * 70)

    global_features = compute_global_features(df_raw, channels, window=QCI_WINDOW)
    global_channels = list(global_features.columns)

    # Standardize global features
    gf_std = (global_features - global_features.rolling(365, min_periods=180).mean()) / \
             (global_features.rolling(365, min_periods=180).std() + 1e-10)
    gf_std = gf_std.reindex(std_df.index).dropna()

    # Use same time range
    common_idx = std_df.index.intersection(gf_std.index)
    gf_aligned = gf_std.loc[common_idx]

    # Cluster global features
    gf_half = len(gf_aligned) // 2
    km_global = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km_global.fit(gf_aligned.iloc[:gf_half].values)
    global_labels = km_global.predict(gf_aligned.values)

    qci_global = compute_qci(global_labels, CMAP, MODULUS, QCI_WINDOW)
    qci_global.index = common_idx[:-2]

    # ====================================================================
    # GAP SIGNAL
    # ====================================================================
    print("\n" + "=" * 70)
    print("GAP SIGNAL (QCI_local - QCI_global)")
    print("=" * 70)

    common_qci = qci_local.dropna().index.intersection(qci_global.dropna().index)

    qci_gap = qci_local.loc[common_qci] - qci_global.loc[common_qci]

    # Target: future atmospheric variability
    future_var = df_raw[channels].reindex(std_df.index).std(axis=1).shift(-FH).rolling(FH).mean()
    lagged_var = df_raw[channels].reindex(std_df.index).std(axis=1).shift(1)

    # OOS alignment
    common = common_qci.intersection(future_var.dropna().index)
    common = common[common >= dates[half]]
    print(f"OOS days: {len(common)}")

    fvar = future_var.loc[common].values
    lvar = lagged_var.loc[common].values

    # ====================================================================
    # RESULTS
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    for name, qci_series in [("LOCAL", qci_local), ("GLOBAL", qci_global), ("GAP", qci_gap)]:
        qci_oos = qci_series.loc[common].values

        r_raw, p_raw = stats.pearsonr(qci_oos, fvar)
        r_partial, p_partial = compute_partial_r(qci_oos, fvar, lvar)

        sig_raw = "***" if p_raw < 0.001 else "**" if p_raw < 0.01 else "*" if p_raw < 0.05 else "ns"
        sig_par = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else "ns"

        print(f"\n  {name}:")
        print(f"    raw r = {r_raw:+.4f} (p={p_raw:.2e}) {sig_raw}")
        print(f"    partial r = {r_partial:+.4f} (p={p_partial:.2e}) {sig_par}")

    # Check sign relationship
    local_oos = qci_local.loc[common].values
    global_oos = qci_global.loc[common].values
    gap_oos = qci_gap.loc[common].values

    r_lg, _ = stats.pearsonr(local_oos, global_oos)
    print(f"\n  Local-Global correlation: r = {r_lg:+.4f}")

    r_local_raw, _ = stats.pearsonr(local_oos, fvar)
    r_global_raw, _ = stats.pearsonr(global_oos, fvar)
    r_gap_raw, _ = stats.pearsonr(gap_oos, fvar)

    print(f"\n  Sign check:")
    print(f"    Local vs future_var:  {'+' if r_local_raw > 0 else '-'}")
    print(f"    Global vs future_var: {'+' if r_global_raw > 0 else '-'}")
    if (r_local_raw > 0) != (r_global_raw > 0):
        print(f"    OPPOSITE SIGNS — mirrors finance pattern!")
    else:
        print(f"    Same sign — does NOT mirror finance pattern")

    # ====================================================================
    # VERDICT
    # ====================================================================
    r_local_partial, _ = compute_partial_r(local_oos, fvar, lvar)
    r_gap_partial, _ = compute_partial_r(gap_oos, fvar, lvar)

    print(f"\n{'=' * 70}")
    improvement = abs(r_gap_partial) - abs(r_local_partial)
    if improvement > 0.05:
        print(f"VERDICT: Multi-layer IMPROVES over local — gap partial |r| "
              f"= {abs(r_gap_partial):.3f} vs local {abs(r_local_partial):.3f} "
              f"(Δ = +{improvement:.3f})")
    elif improvement > 0:
        print(f"VERDICT: Multi-layer shows MARGINAL improvement — gap partial |r| "
              f"= {abs(r_gap_partial):.3f} vs local {abs(r_local_partial):.3f}")
    else:
        print(f"VERDICT: Multi-layer does NOT improve — gap partial |r| "
              f"= {abs(r_gap_partial):.3f} vs local {abs(r_local_partial):.3f}")
    print(f"{'=' * 70}")

    output = {
        "domain": "era5_multilayer",
        "n_oos": len(common),
        "local": {"raw_r": float(r_local_raw),
                   "partial_r": float(r_local_partial)},
        "global": {"raw_r": float(r_global_raw)},
        "gap": {"raw_r": float(r_gap_raw),
                "partial_r": float(r_gap_partial)},
        "local_global_corr": float(r_lg),
        "improvement": float(improvement),
    }
    with open(os.path.join(HERE, "49_era5_multilayer_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to 49_era5_multilayer_results.json")


if __name__ == "__main__":
    main()
