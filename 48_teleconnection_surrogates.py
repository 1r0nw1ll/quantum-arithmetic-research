#!/usr/bin/env python3
"""
48_teleconnection_surrogates_v2.py — CORRECTED surrogate validation for script 48
==================================================================================

v1 had a circular null design: randomized data generated its own ENSO phases,
so smooth surrogates trivially produced high chi². Same bug as the earlier
null control failure.

CORRECTED DESIGN:
- For chi² (ENSO orbit test): Keep REAL ENSO phases. Only orbits change.
  Asks: "do orbits from REAL data discriminate ENSO better than orbits from RANDOM data?"
- For partial r: Keep REAL future_disp target. Only QCI changes.
  Asks: "does QCI from REAL data predict real dispersion better than QCI from RANDOM data?"

Surrogate types:
1. Phase-randomized: FFT indices, shared random phases, IFFT → cluster → QCI/orbits
2. AR(1): fit per-channel, generate → cluster → QCI/orbits
3. Block-shuffled: 12-month blocks → cluster → QCI/orbits
4. Row-permuted: shuffle cluster labels (keeps real data, randomizes encoding)

In ALL cases, ENSO phases and future_disp come from REAL data.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=teleconnection_topographic, state_alphabet=climate_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# EXACT PARAMETERS FROM SCRIPT 48
MODULUS = 24
N_CLUSTERS = 4
QCI_WINDOW = 24
FH = 12
CMAP = {0: 8, 1: 16, 2: 24, 3: 5}

N_SURROGATES = 200
BLOCK_SIZE = 12

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


# ============================================================================
# CORRECTED PIPELINE — real targets, surrogate QCI/orbits only
# ============================================================================

def standardize_and_split(df, channels):
    """Standardize with rolling 120-month z-score, split in half."""
    roll_mean = df[channels].rolling(120, min_periods=60).mean()
    roll_std = df[channels].rolling(120, min_periods=60).std() + 1e-10
    std_df = (df[channels] - roll_mean) / roll_std
    std_df = std_df.dropna()
    return std_df


def get_labels(std_df, half):
    """K-means cluster, train on first half, predict all."""
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_df.iloc[:half].values)
    return km.predict(std_df.values)


def compute_orbit_by_phase(all_labels, enso_phases, oos_start):
    """Compute orbit distribution by ENSO phase for OOS period."""
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
    return chi2, contingency


def compute_partial_r(qci_oos, fdisp, lagged_oni):
    """Partial r(QCI, future_disp | lagged_ONI)."""
    valid = np.isfinite(lagged_oni) & np.isfinite(qci_oos) & np.isfinite(fdisp)
    if valid.sum() < 30:
        return np.nan
    from numpy.linalg import lstsq
    X = np.column_stack([lagged_oni[valid], np.ones(valid.sum())])
    qci_resid = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
    disp_resid = fdisp[valid] - X @ lstsq(X, fdisp[valid], rcond=None)[0]
    r_partial, _ = stats.pearsonr(qci_resid, disp_resid)
    return r_partial


def main():
    df_raw = fetch_all_indices()
    channels = ["ONI", "NAO", "AO", "PDO", "AMO"]

    # ====================================================================
    # REAL PIPELINE — compute real targets and real QCI/orbits
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Compute REAL targets (ENSO phases + future dispersion)")
    print("=" * 70)

    std_df_real = standardize_and_split(df_raw, channels)
    n = len(std_df_real)
    half = n // 2
    dates = std_df_real.index

    # REAL ENSO phases — from REAL ONI, used for ALL surrogate comparisons
    real_enso_phases = classify_enso(df_raw["ONI"].reindex(std_df_real.index).values)

    # REAL future dispersion — from REAL data, used for ALL surrogate comparisons
    real_future_disp = df_raw[channels].reindex(std_df_real.index).shift(-FH).rolling(FH).std().mean(axis=1)

    # REAL lagged ONI — for partial correlation control
    real_lagged_oni = df_raw["ONI"].reindex(std_df_real.index).shift(1)

    # REAL QCI and orbits
    real_labels = get_labels(std_df_real, half)
    real_qci = compute_qci(real_labels, CMAP, MODULUS, QCI_WINDOW)
    real_qci.index = dates[:-2]

    # OOS alignment
    common = real_qci.dropna().index.intersection(real_future_disp.dropna().index)
    common = common[common >= dates[half]]
    print(f"  OOS months: {len(common)}")

    # REAL test statistics
    real_qci_oos = real_qci.loc[common].values
    real_fdisp = real_future_disp.loc[common].values
    real_r, real_p = stats.pearsonr(real_qci_oos, real_fdisp)

    real_chi2, real_cont = compute_orbit_by_phase(real_labels, real_enso_phases, half)

    real_lagged_oos = real_lagged_oni.loc[common].values
    real_partial_r = compute_partial_r(real_qci_oos, real_fdisp, real_lagged_oos)

    print(f"\n  REAL chi2 = {real_chi2:.2f}")
    print(f"  REAL r(QCI, disp) = {real_r:+.4f} (p={real_p:.6f})")
    print(f"  REAL partial r = {real_partial_r:+.4f}")

    # Print orbit table
    orbit_names = ["singularity", "satellite", "cosmos"]
    phase_names = ["el_nino", "la_nina", "neutral"]
    print(f"\n  {'':>12} {'sing':>10} {'sat':>10} {'cos':>10} {'total':>8}")
    for i, phase in enumerate(phase_names):
        total = real_cont[i].sum()
        if total > 0:
            pcts = real_cont[i] / total * 100
            print(f"  {phase:>12} {pcts[0]:>9.1f}% {pcts[1]:>9.1f}% {pcts[2]:>9.1f}% {total:>7d}")

    # ====================================================================
    # SURROGATES — CORRECTED: real targets, surrogate QCI/orbits only
    # ====================================================================
    surrogate_types = ["phase_randomized", "ar1", "block_shuffled", "row_permuted"]
    surrogate_results = {st: {"chi2": [], "partial_r": [], "r_disp": []} for st in surrogate_types}

    for st in surrogate_types:
        print(f"\n{'=' * 70}")
        print(f"SURROGATE: {st} ({N_SURROGATES} iterations)")
        print(f"  Using REAL ENSO phases and REAL future dispersion as targets")
        print("=" * 70)

        for i in range(N_SURROGATES):
            rng = np.random.RandomState(1000 + i)

            if st == "row_permuted":
                # Shuffle cluster labels from REAL data
                surr_labels = real_labels.copy()
                rng.shuffle(surr_labels)
            else:
                # Generate surrogate data, cluster it
                if st == "phase_randomized":
                    surr_data = make_phase_randomized(df_raw[channels], rng)
                elif st == "ar1":
                    surr_data = make_ar1(df_raw[channels], rng)
                elif st == "block_shuffled":
                    surr_data = make_block_shuffled(df_raw[channels], rng)

                surr_data.columns = channels
                # Standardize surrogate data the same way
                surr_roll_mean = surr_data.rolling(120, min_periods=60).mean()
                surr_roll_std = surr_data.rolling(120, min_periods=60).std() + 1e-10
                surr_std = (surr_data - surr_roll_mean) / surr_roll_std

                # Align to same index as real
                surr_std = surr_std.reindex(std_df_real.index).dropna()
                if len(surr_std) < 200:
                    surrogate_results[st]["chi2"].append(np.nan)
                    surrogate_results[st]["partial_r"].append(np.nan)
                    surrogate_results[st]["r_disp"].append(np.nan)
                    continue

                # Cluster with same train/test split position
                surr_half = min(half, len(surr_std) // 2)
                surr_labels = get_labels(surr_std, surr_half)

                # Pad/truncate to match real label length
                if len(surr_labels) < len(real_labels):
                    surr_labels = np.pad(surr_labels, (0, len(real_labels) - len(surr_labels)),
                                         mode='edge')
                else:
                    surr_labels = surr_labels[:len(real_labels)]

            # Compute surrogate QCI using surrogate labels
            surr_qci = compute_qci(surr_labels, CMAP, MODULUS, QCI_WINDOW)
            surr_qci.index = dates[:-2]

            # TEST 1: surrogate QCI vs REAL future dispersion
            try:
                surr_qci_oos = surr_qci.loc[common].values
                if np.isfinite(surr_qci_oos).sum() < 30:
                    raise ValueError("too few valid")
                r_s, _ = stats.pearsonr(surr_qci_oos[np.isfinite(surr_qci_oos)],
                                        real_fdisp[np.isfinite(surr_qci_oos)])
            except Exception:
                r_s = np.nan

            # TEST 2: surrogate orbits vs REAL ENSO phases
            chi2_s, _ = compute_orbit_by_phase(surr_labels, real_enso_phases, half)

            # TEST 3: surrogate QCI partial r vs REAL targets
            try:
                surr_qci_oos_all = surr_qci.loc[common].values
                pr_s = compute_partial_r(surr_qci_oos_all, real_fdisp, real_lagged_oos)
            except Exception:
                pr_s = np.nan

            surrogate_results[st]["chi2"].append(chi2_s)
            surrogate_results[st]["partial_r"].append(pr_s)
            surrogate_results[st]["r_disp"].append(r_s)

            if (i + 1) % 50 == 0:
                sys.stdout.write(f"\r  {i + 1}/{N_SURROGATES}")
                sys.stdout.flush()
        print()

    # ====================================================================
    # COMPARISON — two-tailed for r, one-tailed (real > surr) for chi²
    # ====================================================================
    print("\n" + "=" * 70)
    print("SURROGATE COMPARISON (CORRECTED DESIGN)")
    print("  chi²: one-tailed (real > surrogate)")
    print("  r/partial_r: two-tailed (|real| > |surrogate|)")
    print("=" * 70)

    summary = {}
    for st in surrogate_types:
        print(f"\n--- {st} ---")
        for metric in ["chi2", "partial_r", "r_disp"]:
            vals = np.array(surrogate_results[st][metric])
            vals = vals[np.isfinite(vals)]
            real_val = {"chi2": real_chi2, "partial_r": real_partial_r, "r_disp": real_r}[metric]

            if len(vals) == 0:
                print(f"  {metric}: no valid surrogates")
                continue

            mean_s = np.mean(vals)
            std_s = np.std(vals)

            if metric == "chi2":
                # One-tailed: real > surrogates = stronger association
                rank_p = np.mean(vals >= real_val)
                z = (real_val - mean_s) / std_s if std_s > 0 else 0
            else:
                # Two-tailed: |real| > |surrogates|
                rank_p = np.mean(np.abs(vals) >= np.abs(real_val))
                z = (np.abs(real_val) - np.mean(np.abs(vals))) / np.std(np.abs(vals)) if np.std(np.abs(vals)) > 0 else 0

            beats = "BEATS" if rank_p < 0.05 else "FAILS"
            sig = "***" if rank_p < 0.001 else "**" if rank_p < 0.01 else "*" if rank_p < 0.05 else "ns"

            print(f"  {metric}: real={real_val:+.4f}, surr_mean={mean_s:+.4f}±{std_s:.4f}, "
                  f"z={z:+.2f}, rank_p={rank_p:.4f} → {beats} {sig}")

            summary[f"{st}_{metric}"] = {
                "real": float(real_val),
                "surr_mean": float(mean_s),
                "surr_std": float(std_s),
                "z": float(z),
                "rank_p": float(rank_p),
                "beats": beats == "BEATS",
                "n_valid": int(len(vals)),
            }

    # ====================================================================
    # VERDICT
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("SCORECARD")
    print("=" * 70)
    for metric in ["chi2", "partial_r", "r_disp"]:
        n_pass = sum(1 for st in surrogate_types
                     if summary.get(f"{st}_{metric}", {}).get("beats", False))
        label = {"chi2": "chi² orbit-ENSO", "partial_r": "partial r", "r_disp": "raw r"}[metric]
        print(f"  {label:>20}: {n_pass}/4 surrogate types beaten")

    chi2_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_chi2", {}).get("beats", False))
    pr_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_partial_r", {}).get("beats", False))

    if chi2_pass >= 3 or pr_pass >= 3:
        print("\nVERDICT: Climate teleconnection Tier 3 CONFIRMED")
    elif chi2_pass >= 2 or pr_pass >= 2:
        print("\nVERDICT: Climate teleconnection TRENDING — partial survival")
    else:
        print("\nVERDICT: Climate teleconnection does not survive surrogates")
    print("=" * 70)

    output = {
        "domain": "climate_teleconnection_surrogates_v2",
        "design": "CORRECTED: real ENSO phases + real future_disp as targets; only QCI/orbits from surrogates",
        "real_results": {
            "chi2": float(real_chi2),
            "partial_r": float(real_partial_r),
            "r_disp": float(real_r),
            "r_p": float(real_p),
        },
        "n_surrogates": N_SURROGATES,
        "surrogate_comparison": summary,
        "chi2_pass": chi2_pass,
        "partial_r_pass": pr_pass,
        "params": {"K": N_CLUSTERS, "QCI_WINDOW": QCI_WINDOW, "FH": FH, "MODULUS": MODULUS},
    }
    outpath = os.path.join(HERE, "48_teleconnection_surrogate_v2_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
