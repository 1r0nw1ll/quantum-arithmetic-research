#!/usr/bin/env python3
"""
49_forecast_coherence_surrogates_v2.py — CORRECTED surrogate validation for script 49
======================================================================================

v1 had two issues:
1. Surrogate generated its own targets (circular)
2. Sign test was one-sided only

CORRECTED DESIGN:
- Keep REAL future atmospheric variability as target
- Keep REAL variability regimes for orbit test
- Only QCI/orbits come from surrogates
- Two-tailed test for r/partial_r

Uses pre-extracted CSV from v1 run (.era5_extracted.csv).
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=forecast_coherence, state_alphabet=atmospheric_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

# EXACT PARAMETERS FROM SCRIPT 49
MODULUS = 24
N_CLUSTERS = 6
QCI_WINDOW = 63
FH = 21
CMAP = {0: 8, 1: 16, 2: 24, 3: 5, 4: 3, 5: 11}

N_SURROGATES = 200
BLOCK_SIZE = 21

np.random.seed(42)

EXTRACTED_CSV = os.path.join(HERE, ".era5_extracted.csv")


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


# ============================================================================
# SURROGATE GENERATORS
# ============================================================================

def make_phase_randomized(df, rng):
    n = len(df)
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    freqs = np.fft.rfftfreq(n)
    random_phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    random_phases[0] = 0
    if n % 2 == 0:
        random_phases[-1] = 0
    for col in df.columns:
        fft_vals = np.fft.rfft(df[col].values)
        fft_shifted = np.abs(fft_vals) * np.exp(1j * (np.angle(fft_vals) + random_phases))
        result[col] = np.fft.irfft(fft_shifted, n=n)
    return result


def make_ar1(df, rng):
    n = len(df)
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        vals = df[col].values
        phi = np.corrcoef(vals[:-1], vals[1:])[0, 1]
        sigma = np.std(vals) * np.sqrt(max(1 - phi * phi, 0.001))
        synth = np.zeros(n)
        synth[0] = rng.normal(0, np.std(vals))
        for t in range(1, n):
            synth[t] = phi * synth[t - 1] + rng.normal(0, sigma)
        result[col] = synth
    return result


def make_block_shuffled(df, rng, block_size=BLOCK_SIZE):
    n = len(df)
    n_blocks = n // block_size
    indices = np.arange(n_blocks)
    rng.shuffle(indices)
    new_order = []
    for idx in indices:
        new_order.extend(range(idx * block_size, (idx + 1) * block_size))
    remainder = n - n_blocks * block_size
    if remainder > 0:
        new_order.extend(range(n_blocks * block_size, n))
    result = df.iloc[new_order].copy()
    result.index = df.index
    return result


def compute_partial_r(qci_oos, fvar, lagged_var):
    valid = np.isfinite(lagged_var) & np.isfinite(qci_oos) & np.isfinite(fvar)
    if valid.sum() < 100:
        return np.nan
    from numpy.linalg import lstsq
    X = np.column_stack([lagged_var[valid], np.ones(valid.sum())])
    qci_r = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
    var_r = fvar[valid] - X @ lstsq(X, fvar[valid], rcond=None)[0]
    r, _ = stats.pearsonr(qci_r, var_r)
    return r


def main():
    if not os.path.exists(EXTRACTED_CSV):
        print(f"ERROR: {EXTRACTED_CSV} not found. Run 49_forecast_coherence_surrogates.py first to download data.")
        return

    df_raw = pd.read_csv(EXTRACTED_CSV, index_col=0, parse_dates=True)
    channels = list(df_raw.columns)
    print(f"Loaded: {len(df_raw)} days × {len(channels)} channels")

    # ====================================================================
    # REAL PIPELINE
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Compute REAL targets and REAL QCI")
    print("=" * 70)

    roll_mean = df_raw[channels].rolling(365, min_periods=180).mean()
    roll_std = df_raw[channels].rolling(365, min_periods=180).std() + 1e-10
    std_df = (df_raw[channels] - roll_mean) / roll_std
    std_df = std_df.dropna()

    n = len(std_df)
    half = n // 2
    dates = std_df.index
    print(f"  Standardized: {n} days, train={half}, test={n-half}")

    # REAL targets
    real_future_var = df_raw[channels].reindex(std_df.index).std(axis=1).shift(-FH).rolling(FH).mean()
    real_lagged_var = df_raw[channels].reindex(std_df.index).std(axis=1).shift(1)

    # Variability regimes (for orbit test)
    common_full = real_future_var.dropna().index
    common_full = common_full[common_full >= dates[half]]

    # REAL labels
    km_real = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km_real.fit(std_df.iloc[:half].values)
    real_labels = km_real.predict(std_df.values)

    # REAL QCI
    real_qci = compute_qci(real_labels, CMAP, MODULUS, QCI_WINDOW)
    real_qci.index = dates[:-2]

    common = real_qci.dropna().index.intersection(common_full)
    print(f"  OOS days: {len(common)}")

    real_qci_oos = real_qci.loc[common].values
    real_fvar = real_future_var.loc[common].values
    real_lagged_oos = real_lagged_var.loc[common].values

    real_r, real_p = stats.pearsonr(real_qci_oos, real_fvar)
    real_partial_r = compute_partial_r(real_qci_oos, real_fvar, real_lagged_oos)

    # Orbit by regime
    var_tercile = np.percentile(real_fvar, [33, 67])
    real_regime = np.where(real_fvar < var_tercile[0], "low",
                  np.where(real_fvar > var_tercile[1], "high", "mid"))

    orbit_by_regime = {"low": [], "mid": [], "high": []}
    for t_idx, t in enumerate(range(half, min(len(real_labels) - 1, half + len(common)))):
        b = CMAP.get(real_labels[t], 5)
        e = CMAP.get(real_labels[t + 1], 5)
        if b == e:
            orb = "singularity"
        elif t % 8 == 0:
            orb = "satellite"
        else:
            orb = "cosmos"
        if t_idx < len(real_regime):
            orbit_by_regime.setdefault(real_regime[t_idx], []).append(orb)

    contingency = np.zeros((3, 3), dtype=int)
    for i, reg in enumerate(["low", "mid", "high"]):
        for j, orb in enumerate(["singularity", "satellite", "cosmos"]):
            contingency[i, j] = orbit_by_regime.get(reg, []).count(orb)
    try:
        real_chi2, _, _, _ = stats.chi2_contingency(contingency + 1)
    except:
        real_chi2 = np.nan

    print(f"\n  REAL r(QCI, future_var) = {real_r:+.4f} (p={real_p:.8f})")
    print(f"  REAL partial r = {real_partial_r:+.4f}")
    print(f"  REAL chi2 = {real_chi2:.2f}")

    # ====================================================================
    # SURROGATES — real targets, surrogate QCI only
    # ====================================================================
    surrogate_types = ["phase_randomized", "ar1", "block_shuffled", "row_permuted"]
    surrogate_results = {st: {"r_var": [], "partial_r": [], "chi2": []} for st in surrogate_types}

    for st in surrogate_types:
        print(f"\n{'=' * 70}")
        print(f"SURROGATE: {st} ({N_SURROGATES} iterations)")
        print(f"  Using REAL future variability as target")
        print("=" * 70)

        for i in range(N_SURROGATES):
            rng = np.random.RandomState(2000 + i)

            if st == "row_permuted":
                surr_labels = real_labels.copy()
                rng.shuffle(surr_labels)
            else:
                if st == "phase_randomized":
                    surr_data = make_phase_randomized(df_raw[channels], rng)
                elif st == "ar1":
                    surr_data = make_ar1(df_raw[channels], rng)
                elif st == "block_shuffled":
                    surr_data = make_block_shuffled(df_raw[channels], rng)

                surr_data.columns = channels
                surr_rm = surr_data.rolling(365, min_periods=180).mean()
                surr_rs = surr_data.rolling(365, min_periods=180).std() + 1e-10
                surr_std = (surr_data - surr_rm) / surr_rs
                surr_std = surr_std.reindex(std_df.index).dropna()

                if len(surr_std) < 500:
                    for k in ["r_var", "partial_r", "chi2"]:
                        surrogate_results[st][k].append(np.nan)
                    continue

                surr_half = min(half, len(surr_std) // 2)
                km_s = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
                km_s.fit(surr_std.iloc[:surr_half].values)
                surr_labels = km_s.predict(surr_std.values)

                if len(surr_labels) < len(real_labels):
                    surr_labels = np.pad(surr_labels, (0, len(real_labels) - len(surr_labels)), mode='edge')
                else:
                    surr_labels = surr_labels[:len(real_labels)]

            # Surrogate QCI
            surr_qci = compute_qci(surr_labels, CMAP, MODULUS, QCI_WINDOW)
            surr_qci.index = dates[:-2]

            try:
                surr_qci_oos = surr_qci.loc[common].values
                valid_mask = np.isfinite(surr_qci_oos) & np.isfinite(real_fvar)
                if valid_mask.sum() < 100:
                    raise ValueError
                r_s, _ = stats.pearsonr(surr_qci_oos[valid_mask], real_fvar[valid_mask])
            except:
                r_s = np.nan

            try:
                pr_s = compute_partial_r(surr_qci_oos, real_fvar, real_lagged_oos)
            except:
                pr_s = np.nan

            # Chi² with surrogate orbits vs REAL regimes
            s_orb_by_reg = {"low": [], "mid": [], "high": []}
            for t_idx, t in enumerate(range(half, min(len(surr_labels) - 1, half + len(common)))):
                b = CMAP.get(surr_labels[t], 5)
                e = CMAP.get(surr_labels[t + 1], 5)
                if b == e:
                    orb = "singularity"
                elif t % 8 == 0:
                    orb = "satellite"
                else:
                    orb = "cosmos"
                if t_idx < len(real_regime):
                    s_orb_by_reg.setdefault(real_regime[t_idx], []).append(orb)

            s_cont = np.zeros((3, 3), dtype=int)
            for ii, reg in enumerate(["low", "mid", "high"]):
                for jj, orb in enumerate(["singularity", "satellite", "cosmos"]):
                    s_cont[ii, jj] = s_orb_by_reg.get(reg, []).count(orb)
            try:
                chi2_s, _, _, _ = stats.chi2_contingency(s_cont + 1)
            except:
                chi2_s = np.nan

            surrogate_results[st]["r_var"].append(r_s)
            surrogate_results[st]["partial_r"].append(pr_s)
            surrogate_results[st]["chi2"].append(chi2_s)

            if (i + 1) % 50 == 0:
                sys.stdout.write(f"\r  {i + 1}/{N_SURROGATES}")
                sys.stdout.flush()
        print()

    # ====================================================================
    # COMPARISON
    # ====================================================================
    print("\n" + "=" * 70)
    print("SURROGATE COMPARISON (CORRECTED DESIGN)")
    print("=" * 70)

    summary = {}
    for st in surrogate_types:
        print(f"\n--- {st} ---")
        for metric in ["r_var", "partial_r", "chi2"]:
            vals = np.array(surrogate_results[st][metric])
            vals = vals[np.isfinite(vals)]
            real_val = {"r_var": real_r, "partial_r": real_partial_r, "chi2": real_chi2}[metric]

            if len(vals) == 0:
                continue

            mean_s = np.mean(vals)
            std_s = np.std(vals)

            if metric == "chi2":
                rank_p = np.mean(vals >= real_val)
                z = (real_val - mean_s) / std_s if std_s > 0 else 0
            else:
                rank_p = np.mean(np.abs(vals) >= np.abs(real_val))
                z = (np.abs(real_val) - np.mean(np.abs(vals))) / np.std(np.abs(vals)) if np.std(np.abs(vals)) > 0 else 0

            beats = "BEATS" if rank_p < 0.05 else "FAILS"
            sig = "***" if rank_p < 0.001 else "**" if rank_p < 0.01 else "*" if rank_p < 0.05 else "ns"

            print(f"  {metric}: real={real_val:+.4f}, surr_mean={mean_s:+.4f}±{std_s:.4f}, "
                  f"z={z:+.2f}, rank_p={rank_p:.4f} → {beats} {sig}")

            summary[f"{st}_{metric}"] = {
                "real": float(real_val), "surr_mean": float(mean_s), "surr_std": float(std_s),
                "z": float(z), "rank_p": float(rank_p), "beats": beats == "BEATS",
                "n_valid": int(len(vals)),
            }

    print(f"\n{'=' * 70}")
    print("SCORECARD")
    for metric in ["r_var", "partial_r", "chi2"]:
        n_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_{metric}", {}).get("beats", False))
        label = {"r_var": "raw r", "partial_r": "partial r", "chi2": "chi² orbit-regime"}[metric]
        print(f"  {label:>20}: {n_pass}/4 surrogate types beaten")

    r_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_r_var", {}).get("beats", False))
    pr_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_partial_r", {}).get("beats", False))
    chi2_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_chi2", {}).get("beats", False))

    if r_pass >= 3 or pr_pass >= 3:
        print(f"\nVERDICT: ERA5 Tier 3 CONFIRMED — predictive signal survives surrogates")
    elif chi2_pass >= 3:
        print(f"\nVERDICT: ERA5 Tier 2+ — structural discrimination survives but prediction weak")
    else:
        print(f"\nVERDICT: ERA5 does not survive corrected surrogates")
    print("=" * 70)

    output = {
        "domain": "era5_reanalysis_surrogates_v2",
        "design": "CORRECTED: real targets, surrogate QCI only",
        "real_results": {"r_var": float(real_r), "partial_r": float(real_partial_r),
                         "chi2": float(real_chi2), "r_p": float(real_p)},
        "n_surrogates": N_SURROGATES, "surrogate_comparison": summary,
        "r_pass": r_pass, "partial_r_pass": pr_pass, "chi2_pass": chi2_pass,
        "params": {"K": N_CLUSTERS, "QCI_WINDOW": QCI_WINDOW, "FH": FH, "MODULUS": MODULUS},
    }
    with open(os.path.join(HERE, "49_forecast_coherence_surrogate_v2_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved")


if __name__ == "__main__":
    main()
