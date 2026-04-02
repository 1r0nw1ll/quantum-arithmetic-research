#!/usr/bin/env python3
"""
48_teleconnection_topographic_observer.py — QA observer on climate oscillation indices
=======================================================================================

Climate domain (Direction B). Multi-channel = major teleconnection indices.
Uses NOAA freely available monthly climate indices (no API key).

Channels: ONI (ENSO), NAO, AO, PDO, AMO — five major climate oscillation modes.
Each captures a different ocean basin / atmospheric mechanism.
Together they describe the global climate state monthly.

Tests:
1. Does QCI predict future cross-index dispersion?
2. Does orbit distribution differ between El Nino / La Nina / Neutral?
3. Does QCI add beyond lagged index values (partial correlation)?

HONEST RISK: ~540 monthly observations is thin (seismic had 1679 OOS days).
We may lack statistical power. This is acknowledged upfront.
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
MODULUS = 24
N_CLUSTERS = 4
QCI_WINDOW = 24   # 24 months (2 years) — longer than finance/seismic due to monthly resolution
FH = 12           # 12-month forecast horizon
CMAP = {0: 8, 1: 16, 2: 24, 3: 5}

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


def classify_orbit(b, e, m):
    """Classify QA state into orbit: singularity, satellite, cosmos."""
    d = qa_mod(b + e, m)
    a = qa_mod(b + 2 * e, m)
    # Check singularity: fixed point
    if b == e == d == a:
        return "singularity"
    # Check satellite: 8-cycle (period divides 8)
    state = (b, e)
    for step in range(8):
        b_next = e
        e_next = qa_mod(b + e, m)
        if (b_next, e_next) == state:
            return "satellite"
        b, e = b_next, e_next
    return "cosmos"


# ============================================================================
# DATA FETCHING — all NOAA, no API key
# ============================================================================

def fetch_oni():
    """Oceanic Nino Index (ENSO) — monthly from 1950."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("latin-1").strip().split("\n")

    rows = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 4:
            try:
                year = int(parts[1])
                anom = float(parts[3])
                # Season string like "DJF" — map to month (middle month)
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
    """Generic fetcher for NOAA monthly index files."""
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
    """Pacific Decadal Oscillation — monthly grid format."""
    url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("latin-1").strip().split("\n")

    rows = []
    for line in lines[2:]:  # skip header lines
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
    """Atlantic Multidecadal Oscillation — monthly grid format."""
    url = "https://www.psl.noaa.gov/data/correlation/amon.us.data"
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("latin-1").strip().split("\n")

    # First line has year range
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
    """Fetch all 5 teleconnection indices and merge."""
    print("Fetching NOAA climate indices (no API key needed)...")

    oni = fetch_oni()
    print(f"  ONI (ENSO): {len(oni)} months")

    nao = fetch_monthly_index(
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        "NAO")
    print(f"  NAO: {len(nao)} months")

    ao = fetch_monthly_index(
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
        "AO")
    print(f"  AO: {len(ao)} months")

    pdo = fetch_pdo()
    print(f"  PDO: {len(pdo)} months")

    amo = fetch_amo()
    print(f"  AMO: {len(amo)} months")

    # Merge on date
    df = oni.join(nao, how="inner").join(ao, how="inner").join(pdo, how="inner").join(amo, how="inner")
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    print(f"  Common months: {len(df)} ({df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')})")
    return df


# ============================================================================
# ENSO PHASE CLASSIFICATION (ground truth for orbit distribution test)
# ============================================================================

def classify_enso(oni_values):
    """Classify ENSO phase: El Nino (ONI >= 0.5), La Nina (ONI <= -0.5), Neutral."""
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
# MAIN
# ============================================================================

def main():
    df = fetch_all_indices()
    if len(df) < 200:
        print("Insufficient data")
        return

    channels = ["ONI", "NAO", "AO", "PDO", "AMO"]

    # Standardize (rolling 120-month z-score)
    roll_mean = df[channels].rolling(120, min_periods=60).mean()
    roll_std = df[channels].rolling(120, min_periods=60).std() + 1e-10
    std_df = (df[channels] - roll_mean) / roll_std
    std_df = std_df.dropna()

    n = len(std_df)
    half = n // 2
    dates = std_df.index

    print(f"\nStandardized indices: {n} months, train={half}, test={n-half}")

    # K-means on train half
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_df.iloc[:half].values)
    all_labels = km.predict(std_df.values)

    # QCI
    qci = compute_qci(all_labels, CMAP, MODULUS, QCI_WINDOW)
    qci.index = dates[:-2]

    # ====================================================================
    # TEST 1: QCI vs future cross-index dispersion
    # ====================================================================
    future_disp = df[channels].reindex(std_df.index).shift(-FH).rolling(FH).std().mean(axis=1)

    common = qci.dropna().index.intersection(future_disp.dropna().index)
    common = common[common >= dates[half]]

    print(f"OOS: {len(common)} months")
    print(f"\n{'='*60}")
    print("TEST 1: QCI vs future cross-index dispersion")
    print(f"{'='*60}")

    if len(common) >= 30:
        qci_oos = qci.loc[common].values
        fdisp = future_disp.loc[common].values
        r_disp, p_disp = stats.pearsonr(qci_oos, fdisp)
        sig = "***" if p_disp < 0.001 else "**" if p_disp < 0.01 else "*" if p_disp < 0.05 else "ns"
        print(f"  r = {r_disp:+.4f}, p = {p_disp:.6f} {sig}")
    else:
        r_disp, p_disp = float("nan"), float("nan")
        print("  Insufficient OOS data")

    # ====================================================================
    # TEST 2: Orbit distribution by ENSO phase
    # ====================================================================
    print(f"\n{'='*60}")
    print("TEST 2: Orbit distribution by ENSO phase")
    print(f"{'='*60}")

    # Get orbits for OOS period
    oos_mask = np.zeros(len(all_labels), dtype=bool)
    oos_start_idx = half
    oos_mask[oos_start_idx:] = True

    orbit_labels = []
    enso_phases = classify_enso(df["ONI"].reindex(std_df.index).values)

    orbit_by_phase = {"el_nino": [], "la_nina": [], "neutral": []}
    for t in range(len(all_labels) - 1):
        if not oos_mask[t]:
            continue
        b = CMAP.get(all_labels[t], 5)
        e = CMAP.get(all_labels[t + 1], 5)
        orb = classify_orbit(b, e, MODULUS)
        phase = enso_phases[t] if t < len(enso_phases) else "neutral"
        orbit_by_phase.setdefault(phase, []).append(orb)

    # Build contingency table
    orbit_names = ["singularity", "satellite", "cosmos"]
    phase_names = ["el_nino", "la_nina", "neutral"]
    contingency = np.zeros((len(phase_names), len(orbit_names)), dtype=int)
    for i, phase in enumerate(phase_names):
        for j, orb in enumerate(orbit_names):
            contingency[i, j] = orbit_by_phase.get(phase, []).count(orb)

    print(f"\n  {'':>12} {'singularity':>12} {'satellite':>12} {'cosmos':>12} {'total':>8}")
    for i, phase in enumerate(phase_names):
        total = contingency[i].sum()
        pcts = contingency[i] / max(total, 1) * 100
        print(f"  {phase:>12} {pcts[0]:>11.1f}% {pcts[1]:>11.1f}% {pcts[2]:>11.1f}% {total:>7d}")

    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency + 1)  # +1 to avoid zeros
    sig2 = "***" if chi2_p < 0.001 else "**" if chi2_p < 0.01 else "*" if chi2_p < 0.05 else "ns"
    print(f"\n  chi2 = {chi2:.2f}, p = {chi2_p:.6f} {sig2} (dof={dof})")

    # ====================================================================
    # TEST 3: Partial correlation (QCI beyond lagged ONI)
    # ====================================================================
    print(f"\n{'='*60}")
    print("TEST 3: Partial r(QCI, future_disp | lagged_ONI)")
    print(f"{'='*60}")

    if len(common) >= 30:
        lagged_oni = df["ONI"].reindex(std_df.index).shift(1).loc[common].values

        # Residualize both QCI and future_disp on lagged_oni
        valid = np.isfinite(lagged_oni) & np.isfinite(qci_oos) & np.isfinite(fdisp)
        if valid.sum() >= 30:
            from numpy.linalg import lstsq
            X = np.column_stack([lagged_oni[valid], np.ones(valid.sum())])
            qci_resid = qci_oos[valid] - X @ lstsq(X, qci_oos[valid], rcond=None)[0]
            disp_resid = fdisp[valid] - X @ lstsq(X, fdisp[valid], rcond=None)[0]
            r_partial, p_partial = stats.pearsonr(qci_resid, disp_resid)
            sig3 = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else "ns"
            print(f"  partial r = {r_partial:+.4f}, p = {p_partial:.6f} {sig3}")
        else:
            r_partial, p_partial = float("nan"), float("nan")
            print("  Insufficient valid data after alignment")
    else:
        r_partial, p_partial = float("nan"), float("nan")
        print("  Insufficient OOS data")

    # ====================================================================
    # VERDICT
    # ====================================================================
    print(f"\n{'='*60}")
    any_sig = (p_disp < 0.05 if np.isfinite(p_disp) else False) or \
              (chi2_p < 0.05) or \
              (p_partial < 0.05 if np.isfinite(p_partial) else False)
    if any_sig:
        print("VERDICT: QA topographic observer detects climate teleconnection structure.")
    else:
        print("VERDICT: QCI does not significantly predict climate dynamics.")
        print("  Monthly resolution may lack power. Consider Direction A (forecast coherence).")
    print(f"{'='*60}")

    # Save results
    results = {
        "domain": "climate_teleconnection",
        "channels": channels,
        "n_months_total": n,
        "n_oos": int(len(common)),
        "test1_qci_vs_dispersion": {"r": float(r_disp), "p": float(p_disp)},
        "test2_orbit_by_enso": {"chi2": float(chi2), "p": float(chi2_p), "dof": int(dof)},
        "test3_partial_r": {"r": float(r_partial), "p": float(p_partial)},
        "params": {"K": N_CLUSTERS, "QCI_WINDOW": QCI_WINDOW, "FH": FH, "MODULUS": MODULUS},
    }
    outpath = os.path.join(HERE, "48_teleconnection_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
