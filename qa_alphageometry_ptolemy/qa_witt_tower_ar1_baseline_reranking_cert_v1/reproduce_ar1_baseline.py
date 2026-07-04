#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=reproduction script for cert [516]; sources cited in mapping_protocol_ref.json -->
"""
Reproduction script for cert [516]'s empirical record.

Pure Python, no external dependencies. Regenerates the AR(1)-predicted
ratio for each Witt Tower discrimination-ladder domain at its own
reported (or, for rivers, live-fetched) lag-1 autocorrelation, using the
IDENTICAL rank-bin operator (b,e in {0..26}, a=b+2e<=6) that certs
[490]-[495] use on the real data.

Run with --fetch-rivers to re-fetch live USGS data for the rivers rho
(requires network + curl); otherwise uses the recorded 2026-07-04 value.
"""
import argparse
import math
import random
import subprocess
import json

MOD = 27
SIGNAL_THRESHOLD = 6
INDEPENDENCE_FRAC = 16.0 / 729.0

# rho, observed pooled ratio (taken directly from each cert's own fallback/
# artifact data), source cert id
DOMAINS = {
    "SST [493]":           (0.942, 4.432, 493),
    "Temperature [492]":   (0.728, 3.400, 492),
    "Rivers [490]":        (0.310, 2.689, 490),   # rho: see --fetch-rivers
    "Precipitation [494]": (0.318, 3.048, 494),
    "EEG interictal [491]": (-0.262, 0.725, 491),
    "FX 1-min [495]":      (-0.127, 1.009, 495),
}

RIVER_GAUGES = {"Potomac": "01646500", "Hudson": "01372500",
                "Missouri": "06018500", "Eel": "11477000"}


def rank_bins(vals):
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return [int(r * MOD / n) for r in ranks]


def simulate_ar1_ratio(rho, n=8000, trials=40, seed=0):
    rng = random.Random(seed)
    ratios = []
    for _ in range(trials):
        x = [rng.gauss(0, 1)]
        innov = (1 - rho * rho) ** 0.5
        for _ in range(n - 1):
            x.append(rho * x[-1] + innov * rng.gauss(0, 1))
        bins = rank_bins(x)
        n_triplets = n - 2
        n_sig = sum(1 for t in range(n_triplets) if bins[t] + 2 * bins[t + 1] <= SIGNAL_THRESHOLD)
        n_expected = n_triplets * INDEPENDENCE_FRAC
        ratios.append(n_sig / n_expected)
    mean = sum(ratios) / len(ratios)
    var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    return mean, var ** 0.5


def fetch_river_rho():
    """Live-fetch USGS discharge data and compute pooled log-return lag-1
    autocorrelation across the 4 gauges cert [490] uses."""
    rhos = []
    for name, site in RIVER_GAUGES.items():
        url = (f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site}"
               f"&parameterCd=00060&startDT=2000-01-01&endDT=2026-01-01")
        r = subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True, timeout=70)
        d = json.loads(r.stdout)
        ts = d.get("value", {}).get("timeSeries", [])
        if not ts:
            print(f"  {name}: fetch failed, skipping")
            continue
        raw = ts[0]["values"][0]["value"]
        vals = []
        for e in raw:
            try:
                v = float(e["value"])
                vals.append(v if v > 0 else None)
            except Exception:
                vals.append(None)
        log_rets = [math.log(vals[i + 1] / vals[i]) * 100 for i in range(len(vals) - 1)
                    if vals[i] and vals[i + 1] and vals[i] > 0 and vals[i + 1] > 0]
        n = len(log_rets)
        mu = sum(log_rets) / n
        var = sum((x - mu) ** 2 for x in log_rets) / n
        cov1 = sum((log_rets[i] - mu) * (log_rets[i + 1] - mu) for i in range(n - 1)) / (n - 1)
        rho = cov1 / var if var > 0 else 0.0
        rhos.append(rho)
        print(f"  {name}: n={n} rho={rho:.4f}")
    return sum(rhos) / len(rhos) if rhos else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-rivers", action="store_true",
                         help="re-fetch live USGS data for the rivers rho (requires network)")
    args = parser.parse_args()

    domains = dict(DOMAINS)
    if args.fetch_rivers:
        print("Fetching live USGS river discharge data...")
        rho = fetch_river_rho()
        if rho is not None:
            _, observed, cert_id = domains["Rivers [490]"]
            domains["Rivers [490]"] = (rho, observed, cert_id)
        print()

    print(f"{'domain':24s} {'rho':>8s} {'AR1_predicted':>15s} {'observed':>10s} {'excess_ratio':>13s}")
    rows = []
    for name, (rho, observed, cert_id) in domains.items():
        mean, sd = simulate_ar1_ratio(rho)
        excess = observed / mean
        rows.append((name, rho, mean, sd, observed, excess))
        print(f"{name:24s} {rho:+8.3f} {mean:9.2f}+-{sd:.2f} {observed:10.3f} {excess:13.2f}")

    print()
    print("Reranked by genuine excess beyond plain AR(1) correlation (descending):")
    for name, rho, mean, sd, observed, excess in sorted(rows, key=lambda r: -r[5]):
        print(f"  {name:24s} excess_ratio={excess:.2f}x  (raw={observed:.2f}x, AR1_predicted={mean:.2f}x)")
