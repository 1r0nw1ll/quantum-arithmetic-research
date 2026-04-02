#!/usr/bin/env python3
"""
49_forecast_coherence_surrogates.py — Process-level surrogate validation for script 49
========================================================================================

Bolts 4 surrogate types onto the EXACT pipeline from 49_forecast_coherence_observer.py.
Disk-efficient: extracts only 5 grid points per chunk (no full-array caching).

If a pre-extracted CSV exists (.era5_extracted.csv), uses that directly.
Otherwise downloads from WeatherBench2 ERA5 (GCS, no API key).

Surrogate types (matching finance/climate validation protocol):
1. Phase-randomized: FFT each channel, shared random phases, IFFT
2. AR(1): fit AR(1) per channel, generate synthetic
3. Block-shuffled: shuffle 21-day blocks (preserves synoptic autocorrelation)
4. Row-permuted: shuffle QCI labels
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=forecast_coherence, state_alphabet=atmospheric_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# EXACT PARAMETERS FROM SCRIPT 49 — DO NOT MODIFY
# ============================================================================
MODULUS = 24
N_CLUSTERS = 6
QCI_WINDOW = 63
FH = 21
CMAP = {0: 8, 1: 16, 2: 24, 3: 5, 4: 3, 5: 11}

N_SURROGATES = 200
BLOCK_SIZE = 21  # days — synoptic-scale blocks

np.random.seed(42)

WB2_BASE = "https://storage.googleapis.com/weatherbench2/datasets/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"

GRID_POINTS = {
    "North_Atlantic":   (30, 220),
    "Tropical_Pacific": (60, 160),
    "North_Pacific":    (30, 140),
    "Europe":           (27, 7),
    "Southern_Ocean":   (100, 120),
}

# Only 3 variables (same as original run — disk ran out before v_wind and humidity)
VARIABLES = ["geopotential", "temperature", "u_component_of_wind"]

EXTRACTED_CSV = os.path.join(HERE, ".era5_extracted.csv")


# ============================================================================
# EXACT PIPELINE FUNCTIONS FROM SCRIPT 49
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


# ============================================================================
# DISK-EFFICIENT ERA5 FETCHING — extracts grid points only, no chunk caching
# ============================================================================

def fetch_zarr_metadata():
    """Fetch .zmetadata for coordinate info."""
    url = f"{WB2_BASE}/.zmetadata"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_coord_array_nocache(name, meta):
    """Fetch coordinate array without caching to disk."""
    url = f"{WB2_BASE}/{name}/0"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()

    zarray = meta["metadata"][f"{name}/.zarray"]
    dtype = zarray["dtype"]
    shape = tuple(zarray["shape"])

    try:
        import blosc2
        decompressed = blosc2.decompress(raw)
        return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
    except Exception:
        return np.frombuffer(raw, dtype=dtype).reshape(shape)


def fetch_chunk_extract_points(var_name, chunk_idx, level_idx):
    """Fetch one zarr chunk, extract 5 grid points at target level, discard rest.
    Returns dict of {location_name: array of 48 values} or None on failure."""
    url = f"{WB2_BASE}/{var_name}/{chunk_idx}.0.0.0"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        import blosc2
        try:
            decompressed = blosc2.decompress(raw)
            arr = np.frombuffer(decompressed, dtype="<f4").reshape(48, 13, 240, 121)
        except Exception:
            arr = np.frombuffer(raw, dtype="<f4").reshape(48, 13, 240, 121)

        result = {}
        for loc_name, (lat_i, lon_i) in GRID_POINTS.items():
            col_name = f"{var_name[:4]}_{loc_name}"
            result[col_name] = arr[:, level_idx, lon_i, lat_i].copy()
        del arr  # free memory immediately
        return result
    except Exception as e:
        print(f"  Failed chunk {chunk_idx}: {e}")
        return None


def fetch_era5_data(start_year=2014, end_year=2023):
    """Fetch ERA5 data — disk-efficient, only grid point values kept."""
    # Check for pre-extracted CSV
    if os.path.exists(EXTRACTED_CSV):
        print(f"Loading pre-extracted data from {EXTRACTED_CSV}")
        df = pd.read_csv(EXTRACTED_CSV, index_col=0, parse_dates=True)
        print(f"  {len(df)} days × {len(df.columns)} channels")
        return df

    print("Downloading ERA5 from WeatherBench2 (no disk caching)...")
    meta = fetch_zarr_metadata()

    time_arr = fetch_coord_array_nocache("time", meta)
    level_arr = fetch_coord_array_nocache("level", meta)

    base_time = np.datetime64("1959-01-01")
    dates = base_time + time_arr.astype("timedelta64[D]")
    print(f"  Time: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"  Levels: {level_arr}")

    level_500_idx = int(np.argmin(np.abs(level_arr - 500)))
    print(f"  500hPa index: {level_500_idx} (actual: {level_arr[level_500_idx]})")

    start_date = np.datetime64(f"{start_year}-01-01")
    end_date = np.datetime64(f"{end_year}-12-31")
    time_mask = (dates >= start_date) & (dates <= end_date)
    time_indices = np.where(time_mask)[0]
    chunk_start = time_indices[0] // 48
    chunk_end = time_indices[-1] // 48
    n_chunks = chunk_end - chunk_start + 1
    print(f"  Chunks: {chunk_start}–{chunk_end} ({n_chunks} chunks × {len(VARIABLES)} vars)")

    all_data = {}
    for var in VARIABLES:
        print(f"  Fetching {var}...")
        for ci in range(chunk_start, chunk_end + 1):
            extracted = fetch_chunk_extract_points(var, ci, level_500_idx)
            if extracted is None:
                print(f"    ABORT {var} at chunk {ci}")
                break
            for col_name, vals in extracted.items():
                all_data.setdefault(col_name, []).extend(vals.tolist())
            sys.stdout.write(f"\r    Chunk {ci - chunk_start + 1}/{n_chunks}")
            sys.stdout.flush()
        print()

    min_len = min(len(v) for v in all_data.values())
    # Also cap at available dates
    avail_dates = len(dates) - chunk_start * 48
    min_len = min(min_len, avail_dates)
    for k in all_data:
        all_data[k] = all_data[k][:min_len]

    all_dates = dates[chunk_start * 48: chunk_start * 48 + min_len]
    df = pd.DataFrame(all_data, index=pd.DatetimeIndex(all_dates))
    df = df.loc[start_date:end_date]

    # Save extracted data for reuse (~500KB)
    df.to_csv(EXTRACTED_CSV)
    print(f"  Saved extracted data to {EXTRACTED_CSV} ({os.path.getsize(EXTRACTED_CSV) // 1024} KB)")
    print(f"  Final: {len(df)} days × {len(df.columns)} channels")
    return df


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
# CORE PIPELINE — exact script 49 analysis
# ============================================================================

def run_pipeline(df, channels):
    """Run exact script 49 pipeline. Returns dict with test statistics."""
    roll_mean = df[channels].rolling(365, min_periods=180).mean()
    roll_std = df[channels].rolling(365, min_periods=180).std() + 1e-10
    std_df = (df[channels] - roll_mean) / roll_std
    std_df = std_df.dropna()

    n = len(std_df)
    if n < 500:
        return {"r_var": np.nan, "partial_r": np.nan, "chi2": np.nan}

    half = n // 2
    dates = std_df.index

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_df.iloc[:half].values)
    all_labels = km.predict(std_df.values)

    qci = compute_qci(all_labels, CMAP, MODULUS, QCI_WINDOW)
    qci.index = dates[:-2]

    future_var = df[channels].reindex(std_df.index).std(axis=1).shift(-FH).rolling(FH).mean()

    common = qci.dropna().index.intersection(future_var.dropna().index)
    common = common[common >= dates[half]]

    if len(common) < 100:
        return {"r_var": np.nan, "partial_r": np.nan, "chi2": np.nan}

    qci_oos = qci.loc[common].values
    fvar = future_var.loc[common].values

    r_var, _ = stats.pearsonr(qci_oos, fvar)

    # Orbit by variability regime
    var_tercile = np.percentile(fvar, [33, 67])
    regime = np.where(fvar < var_tercile[0], "low",
             np.where(fvar > var_tercile[1], "high", "mid"))

    orbit_by_regime = {"low": [], "mid": [], "high": []}
    for t_idx, t in enumerate(range(half, min(len(all_labels) - 1, half + len(common)))):
        b = CMAP.get(all_labels[t], 5)
        e = CMAP.get(all_labels[t + 1], 5)
        d = qa_mod(b + e, MODULUS)
        a = qa_mod(b + 2 * e, MODULUS)
        if b == e:
            orb = "singularity"
        elif t % 8 == 0:
            orb = "satellite"
        else:
            orb = "cosmos"
        if t_idx < len(regime):
            orbit_by_regime.setdefault(regime[t_idx], []).append(orb)

    orbit_names = ["singularity", "satellite", "cosmos"]
    regime_names = ["low", "mid", "high"]
    contingency = np.zeros((3, 3), dtype=int)
    for i, reg in enumerate(regime_names):
        for j, orb in enumerate(orbit_names):
            contingency[i, j] = orbit_by_regime.get(reg, []).count(orb)

    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency + 1)
    except Exception:
        chi2 = np.nan

    # Partial r(QCI, future_var | lagged_var)
    from numpy.linalg import lstsq
    lagged_var = df[channels].reindex(std_df.index).std(axis=1).shift(1).loc[common].values
    valid = np.isfinite(lagged_var) & np.isfinite(qci_oos) & np.isfinite(fvar)
    if valid.sum() >= 100:
        X = np.column_stack([lagged_var[valid], np.ones(valid.sum())])
        qci_r = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
        var_r = fvar[valid] - X @ lstsq(X, fvar[valid], rcond=None)[0]
        r_partial, _ = stats.pearsonr(qci_r, var_r)
    else:
        r_partial = np.nan

    return {"r_var": r_var, "partial_r": r_partial, "chi2": chi2}


def run_pipeline_permuted(df, channels, rng):
    """Run pipeline with permuted QCI labels."""
    roll_mean = df[channels].rolling(365, min_periods=180).mean()
    roll_std = df[channels].rolling(365, min_periods=180).std() + 1e-10
    std_df = (df[channels] - roll_mean) / roll_std
    std_df = std_df.dropna()

    n = len(std_df)
    if n < 500:
        return {"r_var": np.nan, "partial_r": np.nan, "chi2": np.nan}

    half = n // 2
    dates = std_df.index

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_df.iloc[:half].values)
    all_labels = km.predict(std_df.values)

    perm_labels = all_labels.copy()
    rng.shuffle(perm_labels)

    qci = compute_qci(perm_labels, CMAP, MODULUS, QCI_WINDOW)
    qci.index = dates[:-2]

    future_var = df[channels].reindex(std_df.index).std(axis=1).shift(-FH).rolling(FH).mean()
    common = qci.dropna().index.intersection(future_var.dropna().index)
    common = common[common >= dates[half]]

    if len(common) < 100:
        return {"r_var": np.nan, "partial_r": np.nan, "chi2": np.nan}

    qci_oos = qci.loc[common].values
    fvar = future_var.loc[common].values
    r_var, _ = stats.pearsonr(qci_oos, fvar)

    # chi² with permuted labels
    var_tercile = np.percentile(fvar, [33, 67])
    regime = np.where(fvar < var_tercile[0], "low",
             np.where(fvar > var_tercile[1], "high", "mid"))
    orbit_by_regime = {"low": [], "mid": [], "high": []}
    for t_idx, t in enumerate(range(half, min(len(perm_labels) - 1, half + len(common)))):
        b = CMAP.get(perm_labels[t], 5)
        e = CMAP.get(perm_labels[t + 1], 5)
        if b == e:
            orb = "singularity"
        elif t % 8 == 0:
            orb = "satellite"
        else:
            orb = "cosmos"
        if t_idx < len(regime):
            orbit_by_regime.setdefault(regime[t_idx], []).append(orb)

    contingency = np.zeros((3, 3), dtype=int)
    for i, reg in enumerate(["low", "mid", "high"]):
        for j, orb in enumerate(["singularity", "satellite", "cosmos"]):
            contingency[i, j] = orbit_by_regime.get(reg, []).count(orb)
    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency + 1)
    except Exception:
        chi2 = np.nan

    from numpy.linalg import lstsq
    lagged_var = df[channels].reindex(std_df.index).std(axis=1).shift(1).loc[common].values
    valid = np.isfinite(lagged_var) & np.isfinite(qci_oos) & np.isfinite(fvar)
    if valid.sum() >= 100:
        X = np.column_stack([lagged_var[valid], np.ones(valid.sum())])
        qci_r = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
        var_r = fvar[valid] - X @ lstsq(X, fvar[valid], rcond=None)[0]
        r_partial, _ = stats.pearsonr(qci_r, var_r)
    else:
        r_partial = np.nan

    return {"r_var": r_var, "partial_r": r_partial, "chi2": chi2}


# ============================================================================
# MAIN
# ============================================================================

def main():
    df_raw = fetch_era5_data()
    channels = list(df_raw.columns)

    # Step 1: Real pipeline
    print("\n" + "=" * 70)
    print("REAL DATA — running exact script 49 pipeline")
    print("=" * 70)
    real = run_pipeline(df_raw, channels)
    print(f"  r(QCI, future_var) = {real['r_var']:+.4f}")
    print(f"  partial r = {real['partial_r']:+.4f}")
    print(f"  chi2 = {real['chi2']:.2f}")

    # Step 2: Surrogates
    surrogate_types = ["phase_randomized", "ar1", "block_shuffled", "row_permuted"]
    surrogate_results = {st: {"r_var": [], "partial_r": [], "chi2": []} for st in surrogate_types}

    for st in surrogate_types:
        print(f"\n{'=' * 70}")
        print(f"SURROGATE: {st} ({N_SURROGATES} iterations)")
        print("=" * 70)

        for i in range(N_SURROGATES):
            rng = np.random.RandomState(2000 + i)

            if st == "row_permuted":
                result = run_pipeline_permuted(df_raw, channels, rng)
            else:
                if st == "phase_randomized":
                    surr_df = make_phase_randomized(df_raw[channels], rng)
                elif st == "ar1":
                    surr_df = make_ar1(df_raw[channels], rng)
                elif st == "block_shuffled":
                    surr_df = make_block_shuffled(df_raw[channels], rng)

                surr_df.columns = channels
                result = run_pipeline(surr_df, channels)

            for k in ["r_var", "partial_r", "chi2"]:
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
        for metric in ["r_var", "partial_r"]:
            vals = np.array(surrogate_results[st][metric])
            vals = vals[np.isfinite(vals)]
            real_val = real[metric]

            if len(vals) == 0:
                print(f"  {metric}: no valid surrogates")
                continue

            mean_s = np.mean(vals)
            std_s = np.std(vals)

            # More negative = stronger signal for both metrics
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

    # Step 4: Verdict
    print(f"\n{'=' * 70}")
    r_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_r_var", {}).get("beats", False))
    partial_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_partial_r", {}).get("beats", False))
    print(f"raw r test:          {r_pass}/4 surrogate types beaten")
    print(f"partial r test:      {partial_pass}/4 surrogate types beaten")

    if r_pass >= 3 or partial_pass >= 3:
        print("\nVERDICT: ERA5 reanalysis Tier 3 CONFIRMED — signal survives process-level nulls")
    elif r_pass >= 2 or partial_pass >= 2:
        print("\nVERDICT: ERA5 reanalysis TRENDING — partial surrogate survival")
    else:
        print("\nVERDICT: ERA5 reanalysis FAILS process-level nulls")
    print("=" * 70)

    output = {
        "domain": "era5_reanalysis_surrogates",
        "real_results": {k: float(v) for k, v in real.items()},
        "n_surrogates": N_SURROGATES,
        "surrogate_comparison": summary,
        "r_pass": r_pass,
        "partial_pass": partial_pass,
        "params": {"K": N_CLUSTERS, "QCI_WINDOW": QCI_WINDOW, "FH": FH, "MODULUS": MODULUS},
    }
    outpath = os.path.join(HERE, "49_forecast_coherence_surrogate_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
