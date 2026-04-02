#!/usr/bin/env python3
"""
46_seismic_topographic_observer.py — QA observer on USGS earthquake catalog
============================================================================

New domain: seismology. Multi-channel signal from USGS earthquake catalog.
Channels: magnitude, depth, latitude, longitude (4 channels per event).
Daily aggregation: count, mean magnitude, mean depth, max magnitude.

Tests: does QCI predict future seismic activity level? Does orbit
distribution differ between active and quiet periods?
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

HERE = os.path.dirname(os.path.abspath(__file__))
MODULUS = 24
N_CLUSTERS = 4
QCI_WINDOW = 63
FH = 21
CMAP = {0: 8, 1: 16, 2: 24, 3: 5}


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


def fetch_usgs_earthquakes():
    """Fetch 10 years of M4+ earthquakes from USGS API."""
    import urllib.request
    import json as _json

    end = datetime.now()
    start = end - timedelta(days=3650)

    print(f"Fetching USGS earthquakes (M4+, {start.date()} to {end.date()})...")
    features = []
    # Fetch year by year (USGS limits query size)
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
                data = _json.loads(resp.read())
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
            "lon": coords[0],
            "lat": coords[1],
        })

    df = pd.DataFrame(rows)
    df = df.set_index("time").sort_index()
    return df


def daily_aggregate(quakes):
    """Aggregate earthquake catalog to daily features (4 channels)."""
    daily = quakes.resample("D").agg({
        "mag": ["count", "mean", "max"],
        "depth": "mean",
    })
    daily.columns = ["count", "mean_mag", "max_mag", "mean_depth"]
    daily = daily.fillna(0)
    # Replace 0 mean_mag/max_mag/mean_depth on quiet days
    daily.loc[daily["count"] == 0, ["mean_mag", "max_mag"]] = 4.0  # baseline M4
    daily.loc[daily["count"] == 0, "mean_depth"] = 30.0  # baseline depth
    return daily


def main():
    quakes = fetch_usgs_earthquakes()
    if quakes is None or len(quakes) < 1000:
        print("Insufficient data")
        return

    daily = daily_aggregate(quakes)
    print(f"Daily aggregated: {len(daily)} days, {daily['count'].sum():.0f} total events")

    # Standardize
    std_daily = (daily - daily.rolling(252).mean()) / (daily.rolling(252).std() + 1e-10)
    std_daily = std_daily.dropna()

    n = len(std_daily)
    half = n // 2
    dates = std_daily.index

    print(f"Standardized: {n} days, train={half}, test={n-half}")

    # K-means clustering
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_daily.iloc[:half].values)
    all_labels = km.predict(std_daily.values)

    unique, counts = np.unique(all_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Microstate {u}: {c} days ({100*c/n:.1f}%)")

    # QCI
    qci = compute_qci(all_labels, CMAP, MODULUS, QCI_WINDOW)
    qci.index = dates[:-2]

    # Future seismic activity: rolling count of M4+ events in next FH days
    future_count = daily["count"].shift(-FH).rolling(FH).sum()

    # Align OOS
    common = qci.dropna().index.intersection(future_count.dropna().index)
    common = common[common >= dates[half]]

    if len(common) < 100:
        print("Insufficient OOS data")
        return

    qci_oos = qci.loc[common].values
    fcount = future_count.loc[common].values

    print(f"\nOOS: {len(common)} days")

    # Test 1: QCI vs future activity
    r, p = stats.pearsonr(qci_oos, fcount)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"\n{'='*60}")
    print("QCI vs FUTURE seismic activity (next {FH}d M4+ count)")
    print(f"{'='*60}")
    print(f"  r = {r:+.4f}, p = {p:.8f} {sig}")

    # Test 2: Orbit distribution by activity level
    activity_rank = future_count.loc[common].rank(pct=True)
    regime = pd.Series("normal", index=common)
    regime[activity_rank >= 0.75] = "active"
    regime[activity_rank <= 0.25] = "quiet"

    from qa_orbit_rules import orbit_family
    print(f"\nOrbit distribution by seismic regime (OOS):")

    # Build orbit series aligned to common index
    # Map dates to label indices
    date_to_idx = {d: i for i, d in enumerate(dates)}
    orbit_classes = []
    orbit_dates = []
    for d in common:
        idx = date_to_idx.get(d)
        if idx is None or idx + 1 >= len(all_labels):
            continue
        b = CMAP.get(all_labels[idx], 5)
        e = CMAP.get(all_labels[idx + 1], 5)
        orbit_classes.append(orbit_family(int(b), int(e), MODULUS))
        orbit_dates.append(d)

    orbit_series = pd.Series(orbit_classes, index=orbit_dates)
    regime_aligned = regime.loc[orbit_series.index]

    for reg in ["quiet", "normal", "active"]:
        mask = regime_aligned == reg
        if mask.sum() < 10:
            continue
        subset = orbit_series.loc[mask]
        dist = subset.value_counts(normalize=True)
        print(f"  {reg:>7} ({mask.sum()} days):", {k: f"{v:.1%}" for k, v in dist.items()})

    # Chi-squared test
    active_orbits = orbit_series.loc[regime_aligned == "active"]
    quiet_orbits = orbit_series.loc[regime_aligned == "quiet"]
    if len(active_orbits) > 10 and len(quiet_orbits) > 10:
        cats = sorted(set(active_orbits) | set(quiet_orbits))
        contingency = np.array([
            [active_orbits.value_counts().get(c, 0) for c in cats],
            [quiet_orbits.value_counts().get(c, 0) for c in cats],
        ])
        if contingency.shape[1] >= 2:
            chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
            sig_chi = "***" if p_chi < 0.001 else "**" if p_chi < 0.01 else "*" if p_chi < 0.05 else "ns"
            print(f"\n  Active vs Quiet χ²={chi2:.2f}, p={p_chi:.6f} {sig_chi}")

    print(f"\n{'='*60}")
    if p < 0.05:
        print("VERDICT: QA topographic observer detects seismic structure.")
        print(f"  QCI correlates with future seismic activity (r={r:+.4f})")
    else:
        print("VERDICT: QCI does not predict future seismic activity.")
        print("  May need different channels or observer design for seismology.")
    print(f"{'='*60}")

    with open(os.path.join(HERE, "46_seismic_results.json"), "w") as f:
        json.dump({"r": float(r), "p": float(p), "n_oos": len(common)}, f, indent=2)


if __name__ == "__main__":
    main()
