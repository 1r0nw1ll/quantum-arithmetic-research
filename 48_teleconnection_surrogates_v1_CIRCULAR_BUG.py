#!/usr/bin/env python3
"""
48_teleconnection_surrogates.py — Process-level surrogate validation for script 48
===================================================================================

Bolts 4 surrogate types onto the EXACT pipeline from 48_teleconnection_topographic_observer.py.
Does NOT re-implement — reuses identical functions and parameters.

Surrogate types (matching finance validation protocol):
1. Phase-randomized: FFT each index, randomize phases, IFFT (preserves spectrum)
2. AR(1): fit AR(1) to each index, generate from fitted model
3. Block-shuffled: shuffle 12-month blocks (preserves seasonal autocorrelation)
4. Row-permuted: shuffle QCI labels (most conservative null)

For each surrogate, runs the full pipeline and computes:
- Test 2: chi² for orbit distribution by ENSO phase
- Test 3: partial r(QCI, future_disp | lagged_ONI)

Real result must beat surrogate distribution to confirm Tier 3.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=teleconnection_topographic, state_alphabet=climate_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
from datetime import datetime
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# EXACT PARAMETERS FROM SCRIPT 48 — DO NOT MODIFY
# ============================================================================
MODULUS = 24
N_CLUSTERS = 4
QCI_WINDOW = 24
FH = 12
CMAP = {0: 8, 1: 16, 2: 24, 3: 5}

N_SURROGATES = 200
BLOCK_SIZE = 12  # months — one year blocks for block shuffle

np.random.seed(42)


# ============================================================================
# EXACT PIPELINE FUNCTIONS FROM SCRIPT 48
# ============================================================================

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


def classify_orbit(b, e, m):
    d = qa_mod(b + e, m)
    a = qa_mod(b + 2 * e, m)
    if b == e == d == a:
        return "singularity"
    state = (b, e)
    b_iter, e_iter = b, e
    for step in range(8):
        b_next = e_iter
        e_next = qa_mod(b_iter + e_iter, m)
        if (b_next, e_next) == state:
            return "satellite"
        b_iter, e_iter = b_next, e_next
    return "cosmos"


# ============================================================================
# DATA FETCHING — identical to script 48
# ============================================================================

def fetch_oni():
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("latin-1").strip().split("\n")
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                year = int(parts[1])
                anom = float(parts[3])
                season_map = {
                    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
                    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
                    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
                }
                month = season_map.get(parts[0], None)
                if month and year >= 1950:
                    rows.append({"date": f"{year}-{month:02d}-01", "ONI": anom})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows).set_index("date")


def fetch_monthly_index(url, name, skip_header=1, year_col=0, month_col=1, value_col=2):
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("latin-1").strip().split("\n")
    rows = []
    for line in lines[skip_header:]:
        parts = line.split()
        if len(parts) > max(year_col, month_col, value_col):
            try:
                year = int(parts[year_col])
                month = int(parts[month_col])
                val = float(parts[value_col])
                if 1900 <= year <= 2030 and 1 <= month <= 12 and val > -99:
                    rows.append({"date": f"{year}-{month:02d}-01", name: val})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows).set_index("date")


def fetch_pdo():
    url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("latin-1").strip().split("\n")
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 13:
            try:
                year = int(parts[0])
                for month in range(1, 13):
                    val = float(parts[month])
                    if val > -99 and 1900 <= year <= 2030:
                        rows.append({"date": f"{year}-{month:02d}-01", "PDO": val})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows).set_index("date")


def fetch_amo():
    url = "https://www.psl.noaa.gov/data/correlation/amon.us.data"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("latin-1").strip().split("\n")
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 13:
            try:
                year = int(parts[0])
                for month in range(1, 13):
                    val = float(parts[month])
                    if val > -99 and 1900 <= year <= 2030:
                        rows.append({"date": f"{year}-{month:02d}-01", "AMO": val})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows).set_index("date")


def fetch_all_indices():
    print("Fetching NOAA climate indices...")
    oni = fetch_oni()
    nao = fetch_monthly_index(
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii", "NAO")
    ao = fetch_monthly_index(
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii", "AO")
    pdo = fetch_pdo()
    amo = fetch_amo()
    df = oni.join(nao, how="inner").join(ao, how="inner").join(pdo, how="inner").join(amo, how="inner")
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    print(f"  {len(df)} common months ({df.index.min():%Y-%m} to {df.index.max():%Y-%m})")
    return df


def classify_enso(oni_values):
    phases = []
    for v in oni_values:
        if v >= 0.5:
            phases.append("el_nino")
        elif v <= -0.5:
            phases.append("la_nina")
        else:
            phases.append("neutral")
    return phases


# ============================================================================
# SURROGATE GENERATORS
# ============================================================================

def make_phase_randomized(df, rng):
    """FFT each column, randomize phases (shared across columns), IFFT.
    Preserves power spectrum AND cross-correlation structure."""
    n = len(df)
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    # Shared random phases for cross-correlation preservation
    freqs = np.fft.rfftfreq(n)
    random_phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    random_phases[0] = 0  # DC component
    if n % 2 == 0:
        random_phases[-1] = 0  # Nyquist
    for col in df.columns:
        fft_vals = np.fft.rfft(df[col].values)
        fft_shifted = np.abs(fft_vals) * np.exp(1j * (np.angle(fft_vals) + random_phases))
        result[col] = np.fft.irfft(fft_shifted, n=n)
    return result


def make_ar1(df, rng):
    """Fit AR(1) to each column, generate synthetic series."""
    n = len(df)
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        vals = df[col].values
        phi = np.corrcoef(vals[:-1], vals[1:])[0, 1]
        sigma = np.std(vals) * np.sqrt(1 - phi * phi)
        synth = np.zeros(n)
        synth[0] = rng.normal(0, np.std(vals))
        for t in range(1, n):
            synth[t] = phi * synth[t - 1] + rng.normal(0, sigma)
        result[col] = synth
    return result


def make_block_shuffled(df, rng, block_size=BLOCK_SIZE):
    """Shuffle non-overlapping blocks of block_size months."""
    n = len(df)
    n_blocks = n // block_size
    remainder = n % block_size
    indices = np.arange(n_blocks)
    rng.shuffle(indices)
    new_order = []
    for idx in indices:
        new_order.extend(range(idx * block_size, (idx + 1) * block_size))
    if remainder > 0:
        new_order.extend(range(n_blocks * block_size, n))
    result = df.iloc[new_order].copy()
    result.index = df.index
    return result


# ============================================================================
# CORE PIPELINE — runs one dataset through the full script 48 analysis
# ============================================================================

def run_pipeline(df_raw, channels, return_labels=False):
    """Run the exact script 48 pipeline on a dataset.
    Returns dict with chi2, partial_r, and optionally QCI labels."""
    # Standardize (rolling 120-month z-score)
    roll_mean = df_raw[channels].rolling(120, min_periods=60).mean()
    roll_std = df_raw[channels].rolling(120, min_periods=60).std() + 1e-10
    std_df = (df_raw[channels] - roll_mean) / roll_std
    std_df = std_df.dropna()

    n = len(std_df)
    if n < 200:
        return {"chi2": np.nan, "partial_r": np.nan, "r_disp": np.nan}

    half = n // 2
    dates = std_df.index

    # K-means on train half
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_df.iloc[:half].values)
    all_labels = km.predict(std_df.values)

    # QCI
    qci = compute_qci(all_labels, CMAP, MODULUS, QCI_WINDOW)
    qci.index = dates[:-2]

    # Future cross-index dispersion
    future_disp = df_raw[channels].reindex(std_df.index).shift(-FH).rolling(FH).std().mean(axis=1)

    common = qci.dropna().index.intersection(future_disp.dropna().index)
    common = common[common >= dates[half]]

    if len(common) < 30:
        return {"chi2": np.nan, "partial_r": np.nan, "r_disp": np.nan}

    qci_oos = qci.loc[common].values
    fdisp = future_disp.loc[common].values

    # Test 1: raw r
    r_disp, _ = stats.pearsonr(qci_oos, fdisp)

    # Test 2: chi² orbit by ENSO phase
    enso_phases = classify_enso(df_raw["ONI"].reindex(std_df.index).values)
    oos_start = half
    orbit_by_phase = {"el_nino": [], "la_nina": [], "neutral": []}
    for t in range(oos_start, len(all_labels) - 1):
        b = CMAP.get(all_labels[t], 5)
        e = CMAP.get(all_labels[t + 1], 5)
        orb = classify_orbit(b, e, MODULUS)
        phase = enso_phases[t] if t < len(enso_phases) else "neutral"
        orbit_by_phase.setdefault(phase, []).append(orb)

    orbit_names = ["singularity", "satellite", "cosmos"]
    phase_names = ["el_nino", "la_nina", "neutral"]
    contingency = np.zeros((3, 3), dtype=int)
    for i, phase in enumerate(phase_names):
        for j, orb in enumerate(orbit_names):
            contingency[i, j] = orbit_by_phase.get(phase, []).count(orb)

    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency + 1)
    except Exception:
        chi2 = np.nan

    # Test 3: partial r(QCI, future_disp | lagged_ONI)
    lagged_oni = df_raw["ONI"].reindex(std_df.index).shift(1).loc[common].values
    valid = np.isfinite(lagged_oni) & np.isfinite(qci_oos) & np.isfinite(fdisp)
    if valid.sum() >= 30:
        from numpy.linalg import lstsq
        X = np.column_stack([lagged_oni[valid], np.ones(valid.sum())])
        qci_resid = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
        disp_resid = fdisp[valid] - X @ lstsq(X, fdisp[valid], rcond=None)[0]
        r_partial, _ = stats.pearsonr(qci_resid, disp_resid)
    else:
        r_partial = np.nan

    result = {"chi2": chi2, "partial_r": r_partial, "r_disp": r_disp}
    if return_labels:
        result["labels"] = all_labels
    return result


# ============================================================================
# ROW-PERMUTED SURROGATE — shuffles QCI labels, not the data
# ============================================================================

def run_pipeline_permuted(df_raw, channels, rng):
    """Run pipeline but permute QCI labels (most conservative null)."""
    roll_mean = df_raw[channels].rolling(120, min_periods=60).mean()
    roll_std = df_raw[channels].rolling(120, min_periods=60).std() + 1e-10
    std_df = (df_raw[channels] - roll_mean) / roll_std
    std_df = std_df.dropna()

    n = len(std_df)
    if n < 200:
        return {"chi2": np.nan, "partial_r": np.nan, "r_disp": np.nan}

    half = n // 2
    dates = std_df.index

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_df.iloc[:half].values)
    all_labels = km.predict(std_df.values)

    # PERMUTE: shuffle the label sequence
    perm_labels = all_labels.copy()
    rng.shuffle(perm_labels)

    qci = compute_qci(perm_labels, CMAP, MODULUS, QCI_WINDOW)
    qci.index = dates[:-2]

    future_disp = df_raw[channels].reindex(std_df.index).shift(-FH).rolling(FH).std().mean(axis=1)
    common = qci.dropna().index.intersection(future_disp.dropna().index)
    common = common[common >= dates[half]]

    if len(common) < 30:
        return {"chi2": np.nan, "partial_r": np.nan, "r_disp": np.nan}

    qci_oos = qci.loc[common].values
    fdisp = future_disp.loc[common].values
    r_disp, _ = stats.pearsonr(qci_oos, fdisp)

    # chi² with permuted labels
    enso_phases = classify_enso(df_raw["ONI"].reindex(std_df.index).values)
    orbit_by_phase = {"el_nino": [], "la_nina": [], "neutral": []}
    for t in range(half, len(perm_labels) - 1):
        b = CMAP.get(perm_labels[t], 5)
        e = CMAP.get(perm_labels[t + 1], 5)
        orb = classify_orbit(b, e, MODULUS)
        phase = enso_phases[t] if t < len(enso_phases) else "neutral"
        orbit_by_phase.setdefault(phase, []).append(orb)

    orbit_names = ["singularity", "satellite", "cosmos"]
    phase_names = ["el_nino", "la_nina", "neutral"]
    contingency = np.zeros((3, 3), dtype=int)
    for i, phase in enumerate(phase_names):
        for j, orb in enumerate(orbit_names):
            contingency[i, j] = orbit_by_phase.get(phase, []).count(orb)

    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency + 1)
    except Exception:
        chi2 = np.nan

    lagged_oni = df_raw["ONI"].reindex(std_df.index).shift(1).loc[common].values
    valid = np.isfinite(lagged_oni) & np.isfinite(qci_oos) & np.isfinite(fdisp)
    if valid.sum() >= 30:
        from numpy.linalg import lstsq
        X = np.column_stack([lagged_oni[valid], np.ones(valid.sum())])
        qci_resid = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
        disp_resid = fdisp[valid] - X @ lstsq(X, fdisp[valid], rcond=None)[0]
        r_partial, _ = stats.pearsonr(qci_resid, disp_resid)
    else:
        r_partial = np.nan

    return {"chi2": chi2, "partial_r": r_partial, "r_disp": r_disp}


# ============================================================================
# MAIN
# ============================================================================

def main():
    df_raw = fetch_all_indices()
    channels = ["ONI", "NAO", "AO", "PDO", "AMO"]

    # Step 1: Run real pipeline
    print("\n" + "=" * 70)
    print("REAL DATA — running exact script 48 pipeline")
    print("=" * 70)
    real = run_pipeline(df_raw, channels)
    print(f"  chi2 = {real['chi2']:.2f}")
    print(f"  partial r = {real['partial_r']:+.4f}")
    print(f"  raw r(QCI, disp) = {real['r_disp']:+.4f}")

    # Step 2: Run surrogates
    surrogate_types = ["phase_randomized", "ar1", "block_shuffled", "row_permuted"]
    surrogate_results = {st: {"chi2": [], "partial_r": [], "r_disp": []} for st in surrogate_types}

    for st in surrogate_types:
        print(f"\n{'=' * 70}")
        print(f"SURROGATE: {st} ({N_SURROGATES} iterations)")
        print("=" * 70)

        for i in range(N_SURROGATES):
            rng = np.random.RandomState(1000 + i)

            if st == "row_permuted":
                result = run_pipeline_permuted(df_raw, channels, rng)
            else:
                # Generate surrogate data
                if st == "phase_randomized":
                    surr_df = make_phase_randomized(df_raw[channels], rng)
                elif st == "ar1":
                    surr_df = make_ar1(df_raw[channels], rng)
                elif st == "block_shuffled":
                    surr_df = make_block_shuffled(df_raw[channels], rng)

                # Rebuild full df with ONI for ENSO classification
                surr_full = surr_df.copy()
                surr_full.columns = channels
                result = run_pipeline(surr_full, channels)

            for k in ["chi2", "partial_r", "r_disp"]:
                surrogate_results[st][k].append(result[k])

            if (i + 1) % 50 == 0:
                sys.stdout.write(f"\r  {i + 1}/{N_SURROGATES}")
                sys.stdout.flush()
        print()

    # Step 3: Compare
    print("\n" + "=" * 70)
    print("SURROGATE COMPARISON")
    print("=" * 70)

    summary = {}
    for st in surrogate_types:
        print(f"\n--- {st} ---")
        for metric in ["chi2", "partial_r"]:
            vals = np.array(surrogate_results[st][metric])
            vals = vals[np.isfinite(vals)]
            real_val = real[metric]

            if len(vals) == 0:
                print(f"  {metric}: no valid surrogates")
                continue

            mean_s = np.mean(vals)
            std_s = np.std(vals)

            if metric == "chi2":
                # Higher chi² = stronger signal (one-tailed: real > surrogates)
                rank_p = np.mean(vals >= real_val)
                beats = "BEATS" if rank_p < 0.05 else "FAILS"
                z = (real_val - mean_s) / std_s if std_s > 0 else 0
            else:
                # For partial r: more negative = stronger (one-tailed: real < surrogates)
                rank_p = np.mean(vals <= real_val)
                beats = "BEATS" if rank_p < 0.05 else "FAILS"
                z = (real_val - mean_s) / std_s if std_s > 0 else 0

            print(f"  {metric}: real={real_val:+.4f}, surr_mean={mean_s:+.4f}±{std_s:.4f}, "
                  f"z={z:+.2f}, rank_p={rank_p:.4f} → {beats}")

            summary[f"{st}_{metric}"] = {
                "real": float(real_val),
                "surr_mean": float(mean_s),
                "surr_std": float(std_s),
                "z": float(z),
                "rank_p": float(rank_p),
                "beats": beats == "BEATS",
                "n_valid": int(len(vals)),
            }

    # Step 4: Overall verdict
    print(f"\n{'=' * 70}")
    chi2_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_chi2", {}).get("beats", False))
    partial_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_partial_r", {}).get("beats", False))
    print(f"chi² orbit test:     {chi2_pass}/4 surrogate types beaten")
    print(f"partial r test:      {partial_pass}/4 surrogate types beaten")

    if chi2_pass >= 3 or partial_pass >= 3:
        print("\nVERDICT: Climate teleconnection Tier 3 CONFIRMED — signal survives process-level nulls")
    elif chi2_pass >= 2 or partial_pass >= 2:
        print("\nVERDICT: Climate teleconnection TRENDING — partial surrogate survival")
    else:
        print("\nVERDICT: Climate teleconnection FAILS process-level nulls")
    print("=" * 70)

    # Save
    output = {
        "domain": "climate_teleconnection_surrogates",
        "real_results": {
            "chi2": float(real["chi2"]),
            "partial_r": float(real["partial_r"]),
            "r_disp": float(real["r_disp"]),
        },
        "n_surrogates": N_SURROGATES,
        "surrogate_comparison": summary,
        "chi2_pass": chi2_pass,
        "partial_pass": partial_pass,
        "params": {"K": N_CLUSTERS, "QCI_WINDOW": QCI_WINDOW, "FH": FH, "MODULUS": MODULUS},
    }
    outpath = os.path.join(HERE, "48_teleconnection_surrogate_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
