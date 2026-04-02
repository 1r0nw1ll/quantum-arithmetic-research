#!/usr/bin/env python3
"""
46_seismic_surrogates.py — Process-level surrogate validation for script 46
============================================================================

CORRECTED DESIGN (matching climate v2):
- Keep REAL future seismic activity (future_count) and REAL activity regimes as targets
- Only QCI/orbits come from surrogates
- Two-tailed test for r, one-tailed for chi²

Surrogate types:
1. Phase-randomized: FFT each daily channel, shared random phases, IFFT → cluster → QCI
2. AR(1): fit per-channel, generate → cluster → QCI
3. Block-shuffled: 21-day blocks → cluster → QCI
4. Row-permuted: shuffle cluster labels (keeps real data, randomizes temporal order)
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=seismic_topographic, state_alphabet=earthquake_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
from numpy.linalg import lstsq
from datetime import datetime, timedelta
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# EXACT PARAMETERS FROM SCRIPT 46
MODULUS = 24
N_CLUSTERS = 4
QCI_WINDOW = 63
FH = 21
CMAP = {0: 8, 1: 16, 2: 24, 3: 5}

N_SURROGATES = 200
BLOCK_SIZE = 21

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


from qa_orbit_rules import orbit_family


# ============================================================================
# DATA FETCHING — identical to script 46
# ============================================================================

def fetch_usgs_earthquakes():
    end = datetime.now()
    start = end - timedelta(days=3650)
    print(f"Fetching USGS earthquakes (M4+, {start.date()} to {end.date()})...")
    features = []
    year_start = start
    while year_start < end:
        year_end = min(year_start + timedelta(days=365), end)
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
            f"format=geojson&starttime={year_start.strftime('%Y-%m-%d')}"
            f"&endtime={year_end.strftime('%Y-%m-%d')}"
            f"&minmagnitude=4&orderby=time&limit=20000"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            batch = data.get("features", [])
            features.extend(batch)
            print(f"  {year_start.year}: {len(batch)} events")
        except Exception as e:
            print(f"  {year_start.year}: FAILED ({e})")
        year_start = year_end
    print(f"  Total: {len(features)} earthquakes")

    rows = []
    for f in features:
        props = f.get("properties", {})
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [0, 0, 0])
        t = props.get("time")
        if t is None:
            continue
        rows.append({
            "time": pd.Timestamp(t, unit="ms"),
            "mag": props.get("mag", 0),
            "depth": coords[2] if len(coords) > 2 else 0,
        })

    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


def daily_aggregate(quakes):
    daily = quakes.resample("D").agg({"mag": ["count", "mean", "max"], "depth": "mean"})
    daily.columns = ["count", "mean_mag", "max_mag", "mean_depth"]
    daily = daily.fillna(0)
    daily.loc[daily["count"] == 0, ["mean_mag", "max_mag"]] = 4.0
    daily.loc[daily["count"] == 0, "mean_depth"] = 30.0
    return daily


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


def compute_partial_r(qci_oos, fcount, lagged_count):
    valid = np.isfinite(lagged_count) & np.isfinite(qci_oos) & np.isfinite(fcount)
    if valid.sum() < 100:
        return np.nan
    X = np.column_stack([lagged_count[valid], np.ones(valid.sum())])
    qci_r = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
    fc_r = fcount[valid] - X @ lstsq(X, fcount[valid], rcond=None)[0]
    r, _ = stats.pearsonr(qci_r, fc_r)
    return r


# ============================================================================
# MAIN
# ============================================================================

def main():
    quakes = fetch_usgs_earthquakes()
    if len(quakes) < 1000:
        print("Insufficient data")
        return

    daily = daily_aggregate(quakes)
    channels = ["count", "mean_mag", "max_mag", "mean_depth"]

    # Standardize
    std_daily = (daily - daily.rolling(252).mean()) / (daily.rolling(252).std() + 1e-10)
    std_daily = std_daily.dropna()

    n = len(std_daily)
    half = n // 2
    dates = std_daily.index
    print(f"Standardized: {n} days, train={half}, test={n-half}")

    # REAL targets
    real_future_count = daily["count"].shift(-FH).rolling(FH).sum()
    real_lagged_count = daily["count"].shift(1).rolling(FH).sum()

    # REAL labels
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_daily.iloc[:half].values)
    real_labels = km.predict(std_daily.values)

    # REAL QCI
    real_qci = compute_qci(real_labels, CMAP, MODULUS, QCI_WINDOW)
    real_qci.index = dates[:-2]

    common = real_qci.dropna().index.intersection(real_future_count.dropna().index)
    common = common[common >= dates[half]]
    print(f"OOS: {len(common)} days")

    real_qci_oos = real_qci.loc[common].values
    real_fcount = real_future_count.loc[common].values
    real_lagged_oos = real_lagged_count.reindex(common).values

    real_r, real_p = stats.pearsonr(real_qci_oos, real_fcount)
    real_partial_r = compute_partial_r(real_qci_oos, real_fcount, real_lagged_oos)

    # REAL orbit-regime chi²
    activity_rank = real_future_count.loc[common].rank(pct=True)
    real_regime = np.where(activity_rank >= 0.75, "active",
                  np.where(activity_rank <= 0.25, "quiet", "normal"))

    date_to_idx = {d: i for i, d in enumerate(dates)}
    real_orbits = []
    for d in common:
        idx = date_to_idx.get(d)
        if idx is None or idx + 1 >= len(real_labels):
            real_orbits.append("cosmos")
            continue
        b = CMAP.get(real_labels[idx], 5)
        e = CMAP.get(real_labels[idx + 1], 5)
        real_orbits.append(orbit_family(int(b), int(e), MODULUS))

    orbit_names = ["singularity", "satellite", "cosmos"]
    regime_names = ["active", "normal", "quiet"]
    contingency = np.zeros((3, 3), dtype=int)
    for i, reg in enumerate(regime_names):
        for j, orb in enumerate(orbit_names):
            contingency[i, j] = sum(1 for k in range(len(common))
                                    if real_regime[k] == reg and real_orbits[k] == orb)
    try:
        real_chi2, _, _, _ = stats.chi2_contingency(contingency + 1)
    except:
        real_chi2 = np.nan

    print(f"\n  REAL r(QCI, future_count) = {real_r:+.4f} (p={real_p:.8f})")
    print(f"  REAL partial r = {real_partial_r:+.4f}")
    print(f"  REAL chi2 = {real_chi2:.2f}")

    # SURROGATES
    surrogate_types = ["phase_randomized", "ar1", "block_shuffled", "row_permuted"]
    surrogate_results = {st: {"r": [], "partial_r": [], "chi2": []} for st in surrogate_types}

    for st in surrogate_types:
        print(f"\n{'=' * 70}")
        print(f"SURROGATE: {st} ({N_SURROGATES} iterations)")
        print("=" * 70)

        for i in range(N_SURROGATES):
            rng = np.random.RandomState(3000 + i)

            if st == "row_permuted":
                surr_labels = real_labels.copy()
                rng.shuffle(surr_labels)
            else:
                if st == "phase_randomized":
                    surr_data = make_phase_randomized(daily[channels], rng)
                elif st == "ar1":
                    surr_data = make_ar1(daily[channels], rng)
                elif st == "block_shuffled":
                    surr_data = make_block_shuffled(daily[channels], rng)

                surr_data.columns = channels
                surr_std = (surr_data - surr_data.rolling(252).mean()) / (surr_data.rolling(252).std() + 1e-10)
                surr_std = surr_std.reindex(std_daily.index).dropna()

                if len(surr_std) < n // 2:
                    for k in ["r", "partial_r", "chi2"]:
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

            surr_qci = compute_qci(surr_labels, CMAP, MODULUS, QCI_WINDOW)
            surr_qci.index = dates[:-2]

            try:
                surr_qci_oos = surr_qci.loc[common].values
                valid = np.isfinite(surr_qci_oos) & np.isfinite(real_fcount)
                r_s, _ = stats.pearsonr(surr_qci_oos[valid], real_fcount[valid])
            except:
                r_s = np.nan

            try:
                pr_s = compute_partial_r(surr_qci_oos, real_fcount, real_lagged_oos)
            except:
                pr_s = np.nan

            # chi² with surrogate orbits vs REAL regimes
            surr_orbits = []
            for d in common:
                idx = date_to_idx.get(d)
                if idx is None or idx + 1 >= len(surr_labels):
                    surr_orbits.append("cosmos")
                    continue
                b = CMAP.get(surr_labels[idx], 5)
                e = CMAP.get(surr_labels[idx + 1], 5)
                surr_orbits.append(orbit_family(int(b), int(e), MODULUS))

            s_cont = np.zeros((3, 3), dtype=int)
            for ii, reg in enumerate(regime_names):
                for jj, orb in enumerate(orbit_names):
                    s_cont[ii, jj] = sum(1 for k in range(len(common))
                                         if real_regime[k] == reg and surr_orbits[k] == orb)
            try:
                chi2_s, _, _, _ = stats.chi2_contingency(s_cont + 1)
            except:
                chi2_s = np.nan

            surrogate_results[st]["r"].append(r_s)
            surrogate_results[st]["partial_r"].append(pr_s)
            surrogate_results[st]["chi2"].append(chi2_s)

            if (i + 1) % 50 == 0:
                sys.stdout.write(f"\r  {i + 1}/{N_SURROGATES}")
                sys.stdout.flush()
        print()

    # COMPARISON
    print("\n" + "=" * 70)
    print("SURROGATE COMPARISON (CORRECTED DESIGN)")
    print("=" * 70)

    summary = {}
    for st in surrogate_types:
        print(f"\n--- {st} ---")
        for metric in ["r", "partial_r", "chi2"]:
            vals = np.array(surrogate_results[st][metric])
            vals = vals[np.isfinite(vals)]
            real_val = {"r": real_r, "partial_r": real_partial_r, "chi2": real_chi2}[metric]

            if len(vals) == 0:
                continue

            mean_s, std_s = np.mean(vals), np.std(vals)

            if metric == "chi2":
                rank_p = np.mean(vals >= real_val)
                z = (real_val - mean_s) / std_s if std_s > 0 else 0
            else:
                rank_p = np.mean(np.abs(vals) >= np.abs(real_val))
                z = (np.abs(real_val) - np.mean(np.abs(vals))) / np.std(np.abs(vals)) if np.std(np.abs(vals)) > 0 else 0

            beats = "BEATS" if rank_p < 0.05 else "FAILS"
            sig = "***" if rank_p < 0.001 else "**" if rank_p < 0.01 else "*" if rank_p < 0.05 else "ns"
            print(f"  {metric}: real={real_val:+.4f}, surr={mean_s:+.4f}±{std_s:.4f}, z={z:+.2f}, rank_p={rank_p:.4f} → {beats} {sig}")

            summary[f"{st}_{metric}"] = {
                "real": float(real_val), "surr_mean": float(mean_s), "surr_std": float(std_s),
                "z": float(z), "rank_p": float(rank_p), "beats": beats == "BEATS",
            }

    print(f"\n{'=' * 70}")
    print("SCORECARD")
    for metric in ["r", "partial_r", "chi2"]:
        n_pass = sum(1 for st in surrogate_types if summary.get(f"{st}_{metric}", {}).get("beats", False))
        print(f"  {metric:>12}: {n_pass}/4")

    output = {
        "domain": "seismology_surrogates",
        "design": "CORRECTED: real targets, surrogate QCI only",
        "real": {"r": float(real_r), "partial_r": float(real_partial_r), "chi2": float(real_chi2)},
        "n_surrogates": N_SURROGATES,
        "summary": summary,
    }
    with open(os.path.join(HERE, "46_seismic_surrogate_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to 46_seismic_surrogate_results.json")


if __name__ == "__main__":
    main()
