#!/usr/bin/env python3
"""
47_climate_topographic_observer.py — QA observer on multi-station temperature data
====================================================================================

Climate domain. Multi-channel = multiple weather stations.
Uses NOAA Global Historical Climatology Network (GHCN) daily temperature data
accessed via meteostat library (free, no API key needed).

Channels: daily temperature from 6 major US cities.
Tests: does QCI predict future temperature variability?
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=climate_topographic, state_alphabet=temperature_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
from numpy.linalg import lstsq
from datetime import datetime

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


def fetch_temperature_data():
    """Fetch daily temperature from multiple US cities via meteostat."""
    try:
        from meteostat import Point, TimeSeries, Granularity
    except ImportError:
        print("meteostat not installed. Trying pip install...")
        os.system("pip install meteostat --break-system-packages -q")
        from meteostat import Point, TimeSeries, Granularity

    # 6 US cities spread across climate zones
    cities = {
        "NYC": Point(40.7128, -74.0060),
        "LA": Point(34.0522, -118.2437),
        "Chicago": Point(41.8781, -87.6298),
        "Houston": Point(29.7604, -95.3698),
        "Phoenix": Point(33.4484, -112.0740),
        "Seattle": Point(47.6062, -122.3321),
    }

    start = datetime(2014, 1, 1)
    end = datetime(2024, 12, 31)

    print(f"Fetching temperature data for {len(cities)} cities ({start.year}-{end.year})...")
    temps = pd.DataFrame()
    for name, point in cities.items():
        try:
            from meteostat import daily as meteo_daily
            data = meteo_daily(point, start, end)
            if "tavg" in data.columns:
                temps[name] = data["tavg"]
                print(f"  {name}: {data['tavg'].notna().sum()} days")
            else:
                print(f"  {name}: no tavg column")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    temps = temps.dropna()
    print(f"  Common days: {len(temps)}")
    return temps


def main():
    temps = fetch_temperature_data()
    if len(temps) < 1000:
        print("Insufficient data")
        return

    # Compute daily temperature anomalies (remove seasonal cycle)
    anomalies = temps - temps.rolling(365, center=True, min_periods=180).mean()
    anomalies = anomalies.dropna()

    # Standardize
    std_anom = (anomalies - anomalies.rolling(365).mean()) / (anomalies.rolling(365).std() + 1e-10)
    std_anom = std_anom.dropna()

    n = len(std_anom)
    half = n // 2
    dates = std_anom.index

    print(f"\nStandardized anomalies: {n} days, train={half}, test={n-half}")

    # K-means
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(std_anom.iloc[:half].values)
    all_labels = km.predict(std_anom.values)

    # QCI
    qci = compute_qci(all_labels, CMAP, MODULUS, QCI_WINDOW)
    qci.index = dates[:-2]

    # Future temperature variability (rolling std of mean temp across cities)
    mean_temp = temps.reindex(std_anom.index).mean(axis=1)
    future_var = mean_temp.shift(-FH).rolling(FH).std()

    # Also: future cross-city dispersion (how different cities diverge)
    future_disp = temps.reindex(std_anom.index).shift(-FH).rolling(FH).apply(lambda x: x.std()).mean(axis=1)

    # Align OOS
    common = qci.dropna().index.intersection(future_var.dropna().index) \
        .intersection(future_disp.dropna().index)
    common = common[common >= dates[half]]

    if len(common) < 100:
        print("Insufficient OOS data")
        return

    qci_oos = qci.loc[common].values

    print(f"OOS: {len(common)} days")
    print(f"\n{'='*60}")
    print("QCI vs FUTURE climate metrics")
    print(f"{'='*60}")

    # Test 1: QCI vs future temperature variability
    fvar = future_var.loc[common].values
    r_var, p_var = stats.pearsonr(qci_oos, fvar)
    sig_var = "***" if p_var < 0.001 else "**" if p_var < 0.01 else "*" if p_var < 0.05 else "ns"
    print(f"  QCI vs future temp variability: r={r_var:+.4f}, p={p_var:.8f} {sig_var}")

    # Test 2: QCI vs future cross-city dispersion
    fdisp = future_disp.loc[common].values
    r_disp, p_disp = stats.pearsonr(qci_oos, fdisp)
    sig_disp = "***" if p_disp < 0.001 else "**" if p_disp < 0.01 else "*" if p_disp < 0.05 else "ns"
    print(f"  QCI vs future cross-city disp:  r={r_disp:+.4f}, p={p_disp:.8f} {sig_disp}")

    print(f"\n{'='*60}")
    if p_var < 0.05 or p_disp < 0.05:
        print("VERDICT: QA topographic observer detects climate structure.")
    else:
        print("VERDICT: QCI does not predict future climate variability.")
        print("  Climate may need longer windows or different observer design.")
    print(f"{'='*60}")

    with open(os.path.join(HERE, "47_climate_results.json"), "w") as f:
        json.dump({
            "r_variability": float(r_var), "p_variability": float(p_var),
            "r_dispersion": float(r_disp), "p_dispersion": float(p_disp),
            "n_oos": len(common),
        }, f, indent=2)


if __name__ == "__main__":
    main()
