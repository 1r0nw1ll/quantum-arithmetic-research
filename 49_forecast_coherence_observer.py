#!/usr/bin/env python3
"""
49_forecast_coherence_observer.py — QA observer on ERA5 reanalysis multi-variable data
========================================================================================

Climate domain (Direction A). Multi-channel = atmospheric state variables at key locations.
Uses WeatherBench2 ERA5 daily reanalysis from Google Cloud Storage (no API key).

Downloads data via direct HTTP (bypasses gcsfs/aiohttp IPv6 issues).
Selects 5 variables at 500hPa for 5 grid points spanning major climate zones.

Channels: geopotential(Z500), temperature(T500), u_wind(U500), v_wind(V500),
          specific_humidity(Q500) — at 5 strategic grid points.

This gives 25 channels (5 vars × 5 locations), similar to the 6-asset finance observer
but with richer spatial structure.

Tests:
1. Does QCI predict future atmospheric variability?
2. Does orbit distribution differ between high/low variability periods?
3. Does QCI add beyond lagged variability (partial correlation)?
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=forecast_coherence, state_alphabet=atmospheric_microstate"

import os, sys, json, struct, io
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
from datetime import datetime, timedelta
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODULUS = 24
N_CLUSTERS = 6
QCI_WINDOW = 63
FH = 21
CMAP = {0: 8, 1: 16, 2: 24, 3: 5, 4: 3, 5: 11}
CACHE_DIR = os.path.join(HERE, ".era5_cache")

np.random.seed(42)

# WeatherBench2 ERA5 daily dataset
WB2_BASE = "https://storage.googleapis.com/weatherbench2/datasets/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"

# 5 strategic grid points (lat_idx, lon_idx) on the 240x121 grid (1.5° resolution)
# Grid: lat from 90 to -90 (121 pts), lon from 0 to 358.5 (240 pts)
GRID_POINTS = {
    "North_Atlantic":  (30, 220),   # ~45N, 330E = 30W
    "Tropical_Pacific": (60, 160),  # ~0N, 240E = 120W (Nino 3.4 region)
    "North_Pacific":    (30, 140),  # ~45N, 210E = 150W
    "Europe":           (27, 7),    # ~50N, 10E
    "Southern_Ocean":   (100, 120), # ~45S, 180E
}

# Variables at 500 hPa (index in the 13-level array)
# Need to discover which index = 500 hPa
VARIABLES = ["geopotential", "temperature", "u_component_of_wind",
             "v_component_of_wind", "specific_humidity"]


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


def fetch_zarr_chunk(var_name, chunk_idx):
    """Fetch a single zarr chunk via direct HTTP."""
    # Chunks are [48, 13, 240, 121] — chunk_idx is the time chunk
    url = f"{WB2_BASE}/{var_name}/{chunk_idx}.0.0.0"
    cache_path = os.path.join(CACHE_DIR, f"{var_name}_{chunk_idx}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            # Zarr stores float32 arrays, possibly compressed
            # Check if it's blosc-compressed
            import blosc2
            try:
                decompressed = blosc2.decompress(raw)
                arr = np.frombuffer(decompressed, dtype="<f4").reshape(48, 13, 240, 121)
            except Exception:
                arr = np.frombuffer(raw, dtype="<f4").reshape(48, 13, 240, 121)
            np.save(cache_path, arr)
            return arr
    except Exception as e:
        print(f"  Failed to fetch {var_name} chunk {chunk_idx}: {e}")
        return None


def fetch_coord_array(name):
    """Fetch a 1D coordinate array (time, level, lat, lon)."""
    cache_path = os.path.join(CACHE_DIR, f"{name}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path, allow_pickle=True)

    url = f"{WB2_BASE}/{name}/0"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()

    # Check metadata for dtype
    meta_url = f"{WB2_BASE}/.zmetadata"
    meta = json.loads(urllib.request.urlopen(
        urllib.request.Request(meta_url, headers={"User-Agent": "QA"}), timeout=15).read())
    zarray = meta["metadata"][f"{name}/.zarray"]
    dtype = zarray["dtype"]
    shape = tuple(zarray["shape"])

    try:
        import blosc2
        decompressed = blosc2.decompress(raw)
        arr = np.frombuffer(decompressed, dtype=dtype).reshape(shape)
    except Exception:
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)

    np.save(cache_path, arr)
    return arr


def fetch_era5_subset(start_year=2014, end_year=2023):
    """Fetch ERA5 data for selected variables, levels, and grid points."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Fetching ERA5 coordinate arrays...")
    time_arr = fetch_coord_array("time")
    level_arr = fetch_coord_array("level")
    lat_arr = fetch_coord_array("latitude")
    lon_arr = fetch_coord_array("longitude")

    # Time is in days since 1959-01-01 (per .zattrs)
    base_time = np.datetime64("1959-01-01")
    dates = base_time + time_arr.astype("timedelta64[D]")

    print(f"  Time: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"  Levels (hPa): {level_arr}")
    print(f"  Grid: {len(lat_arr)} x {len(lon_arr)}")

    # Find 500 hPa index
    level_500_idx = np.argmin(np.abs(level_arr - 500))
    print(f"  500 hPa index: {level_500_idx} (actual: {level_arr[level_500_idx]} hPa)")

    # Find time range
    start_date = np.datetime64(f"{start_year}-01-01")
    end_date = np.datetime64(f"{end_year}-12-31")
    time_mask = (dates >= start_date) & (dates <= end_date)
    time_indices = np.where(time_mask)[0]
    print(f"  Selected time range: {dates[time_indices[0]]} to {dates[time_indices[-1]]} ({len(time_indices)} days)")

    # Determine which chunks we need (48 days per chunk)
    chunk_start = time_indices[0] // 48
    chunk_end = time_indices[-1] // 48
    n_chunks = chunk_end - chunk_start + 1
    print(f"  Chunks needed: {chunk_start} to {chunk_end} ({n_chunks} chunks)")

    # Fetch data for each variable
    all_data = {}
    for var in VARIABLES:
        print(f"\n  Fetching {var}...")
        var_data = []
        for ci in range(chunk_start, chunk_end + 1):
            chunk = fetch_zarr_chunk(var, ci)
            if chunk is None:
                print(f"    Chunk {ci} failed — aborting {var}")
                break
            # Extract 500hPa data at our grid points
            for loc_name, (lat_i, lon_i) in GRID_POINTS.items():
                col_name = f"{var[:4]}_{loc_name}"
                # chunk shape: [48, 13, 240, 121]
                vals = chunk[:, level_500_idx, lon_i, lat_i]
                if ci == chunk_start:
                    all_data.setdefault(col_name, [])
                all_data[col_name].extend(vals.tolist())
            sys.stdout.write(f"\r    Chunk {ci-chunk_start+1}/{n_chunks}")
            sys.stdout.flush()
        print()

    # Build DataFrame
    min_len = min(len(v) for v in all_data.values())
    for k in all_data:
        all_data[k] = all_data[k][:min_len]

    # Align with dates
    all_dates = dates[chunk_start * 48 : chunk_start * 48 + min_len]
    df = pd.DataFrame(all_data, index=pd.DatetimeIndex(all_dates))

    # Filter to requested range
    df = df.loc[start_date:end_date]
    print(f"\nFinal dataset: {len(df)} days × {len(df.columns)} channels")
    return df


def main():
    df = fetch_era5_subset(start_year=2014, end_year=2023)
    if len(df) < 1000:
        print("Insufficient data")
        return

    channels = list(df.columns)

    # Standardize (rolling 365-day z-score)
    roll_mean = df.rolling(365, min_periods=180).mean()
    roll_std = df.rolling(365, min_periods=180).std() + 1e-10
    std_df = (df - roll_mean) / roll_std
    std_df = std_df.dropna()

    n = len(std_df)
    half = n // 2
    dates = std_df.index

    print(f"\nStandardized anomalies: {n} days, train={half}, test={n-half}")

    # K-means
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_df.iloc[:half].values)
    all_labels = km.predict(std_df.values)

    # QCI
    qci = compute_qci(all_labels, CMAP, MODULUS, QCI_WINDOW)
    qci.index = dates[:-2]

    # Future atmospheric variability (rolling std of mean across all channels)
    future_var = df.reindex(std_df.index).std(axis=1).shift(-FH).rolling(FH).mean()

    # Align OOS
    common = qci.dropna().index.intersection(future_var.dropna().index)
    common = common[common >= dates[half]]

    print(f"OOS: {len(common)} days")
    print(f"\n{'='*60}")

    # Test 1: QCI vs future variability
    print("TEST 1: QCI vs future atmospheric variability")
    print(f"{'='*60}")

    if len(common) >= 100:
        qci_oos = qci.loc[common].values
        fvar = future_var.loc[common].values
        r_var, p_var = stats.pearsonr(qci_oos, fvar)
        sig = "***" if p_var < 0.001 else "**" if p_var < 0.01 else "*" if p_var < 0.05 else "ns"
        print(f"  r = {r_var:+.4f}, p = {p_var:.8f} {sig}")
    else:
        r_var, p_var = float("nan"), float("nan")
        print("  Insufficient OOS data")

    # Test 2: Orbit distribution by high/low variability periods
    print(f"\n{'='*60}")
    print("TEST 2: Orbit distribution by variability regime")
    print(f"{'='*60}")

    if len(common) >= 100:
        var_tercile = np.percentile(fvar, [33, 67])
        regime = np.where(fvar < var_tercile[0], "low",
                 np.where(fvar > var_tercile[1], "high", "mid"))

        orbit_by_regime = {"low": [], "mid": [], "high": []}
        for t_idx, t in enumerate(range(half, min(len(all_labels) - 1, half + len(common)))):
            b = CMAP.get(all_labels[t], 5)
            e = CMAP.get(all_labels[t + 1], 5)
            d = qa_mod(b + e, MODULUS)
            a = qa_mod(b + 2 * e, MODULUS)
            # Simple orbit classification
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

        print(f"\n  {'':>8} {'sing':>8} {'sat':>8} {'cos':>8} {'total':>8}")
        for i, reg in enumerate(regime_names):
            total = contingency[i].sum()
            if total > 0:
                pcts = contingency[i] / total * 100
                print(f"  {reg:>8} {pcts[0]:>7.1f}% {pcts[1]:>7.1f}% {pcts[2]:>7.1f}% {total:>7d}")

        chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency + 1)
        sig2 = "***" if chi2_p < 0.001 else "**" if chi2_p < 0.01 else "*" if chi2_p < 0.05 else "ns"
        print(f"\n  chi2 = {chi2:.2f}, p = {chi2_p:.6f} {sig2}")
    else:
        chi2, chi2_p = float("nan"), float("nan")

    # Test 3: Partial correlation
    print(f"\n{'='*60}")
    print("TEST 3: Partial r(QCI, future_var | lagged_var)")
    print(f"{'='*60}")

    if len(common) >= 100:
        from numpy.linalg import lstsq
        lagged_var = df.reindex(std_df.index).std(axis=1).shift(1).loc[common].values
        valid = np.isfinite(lagged_var) & np.isfinite(qci_oos) & np.isfinite(fvar)
        if valid.sum() >= 100:
            X = np.column_stack([lagged_var[valid], np.ones(valid.sum())])
            qci_r = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
            var_r = fvar[valid] - X @ lstsq(X, fvar[valid], rcond=None)[0]
            r_partial, p_partial = stats.pearsonr(qci_r, var_r)
            sig3 = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else "ns"
            print(f"  partial r = {r_partial:+.4f}, p = {p_partial:.8f} {sig3}")
        else:
            r_partial, p_partial = float("nan"), float("nan")
            print("  Insufficient valid data")
    else:
        r_partial, p_partial = float("nan"), float("nan")
        print("  Insufficient OOS data")

    # Verdict
    print(f"\n{'='*60}")
    any_sig = any(p < 0.05 for p in [p_var, chi2_p, p_partial] if np.isfinite(p))
    if any_sig:
        print("VERDICT: QA observer detects structure in ERA5 reanalysis data.")
    else:
        print("VERDICT: QCI does not significantly predict atmospheric variability.")
    print(f"{'='*60}")

    results = {
        "domain": "era5_reanalysis",
        "channels": channels,
        "n_days_total": n,
        "n_oos": int(len(common)),
        "test1_qci_vs_variability": {"r": float(r_var), "p": float(p_var)},
        "test2_orbit_by_regime": {"chi2": float(chi2), "p": float(chi2_p)},
        "test3_partial_r": {"r": float(r_partial), "p": float(p_partial)},
        "params": {"K": N_CLUSTERS, "QCI_WINDOW": QCI_WINDOW, "FH": FH, "MODULUS": MODULUS},
    }
    with open(os.path.join(HERE, "49_forecast_coherence_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to 49_forecast_coherence_results.json")


if __name__ == "__main__":
    main()
