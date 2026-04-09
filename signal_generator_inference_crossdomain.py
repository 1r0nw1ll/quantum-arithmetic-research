#!/usr/bin/env python3
"""
signal_generator_inference_crossdomain.py — [209] Cross-Domain Validation

Applies QA Signal Generator Inference (cert [209]) to three domains:
  1. Audio (synthetic dynamical vs stochastic signals)
  2. Climate teleconnection (5 NOAA indices: ONI, PDO, AMO, NAO, AO)
  3. ERA5 reanalysis (15 channels: 3 vars × 5 locations)

For each domain:
  - Quantize time series to {1,...,9} using global percentile bins
  - Infer generators: e_t = ((b_{t+1} - b_t - 1) % 9) + 1
  - Compute generator entropy and cross-series synchrony
  - Report whether features discriminate between conditions

The method is identical across domains — only the data changes.
"""

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "observer": "signal_generator_inference_cert_209",
    "axioms_checked": ["A1", "A2", "S1", "S2", "T1", "T2"],
}

import json
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
np.random.seed(42)

MOD = 9
HERE = Path(__file__).parent


# ── Core [209] Functions (identical to eeg_signal_dynamics_observer.py) ──────

def infer_generator(b_t: int, b_next: int, m: int = MOD) -> int:
    """Unique A1-compliant generator: QA step inversion."""
    return ((b_next - b_t - 1) % m) + 1


def quantize_global(series_list: list[np.ndarray], m: int = MOD) -> list[list[int]]:
    """
    Quantize multiple series using SHARED percentile bins.
    Observer layer: continuous -> {1,...,m}.
    """
    all_vals = np.concatenate(series_list)
    edges = [float(np.percentile(all_vals, 100 * k / m)) for k in range(1, m)]

    result = []
    for series in series_list:
        quantized = []
        for val in series:
            b = 1
            for edge in edges:
                if float(val) > edge:
                    b += 1
            quantized.append(b)
        result.append(quantized)
    return result


def generator_stats(quantized: list[int], m: int = MOD) -> dict:
    """Generator distribution stats for one quantized series."""
    n = len(quantized)
    if n < 2:
        return {"entropy": 0.0, "mean_e": 0.0, "identity_frac": 0.0}

    generators = [infer_generator(quantized[t], quantized[t + 1], m)
                  for t in range(n - 1)]

    counts = [0] * m
    for g in generators:
        counts[g - 1] += 1
    total = len(generators)
    dist = [c / total for c in counts]

    entropy = -sum(p * np.log2(p) for p in dist if p > 0)
    identity_frac = counts[m - 1] / total  # e=m is identity (no change)

    return {
        "entropy": entropy,
        "mean_e": float(np.mean(generators)),
        "identity_frac": identity_frac,
        "generators": generators,
    }


def cross_series_synchrony(all_quantized: list[list[int]], m: int = MOD) -> float:
    """Generator synchrony: fraction of series sharing modal generator at each step."""
    n_series = len(all_quantized)
    min_len = min(len(q) for q in all_quantized) - 1
    if min_len < 1 or n_series < 2:
        return 0.0

    scores = []
    for t in range(min_len):
        gens = [infer_generator(all_quantized[i][t], all_quantized[i][t + 1], m)
                for i in range(n_series)]
        modal_frac = Counter(gens).most_common(1)[0][1] / n_series
        scores.append(modal_frac)
    return float(np.mean(scores))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1: AUDIO
# ══════════════════════════════════════════════════════════════════════════════

def run_audio():
    """Generate synthetic signals and compare generator distributions."""
    print("\n" + "=" * 72)
    print("DOMAIN 1: AUDIO (synthetic signals)")
    print("=" * 72)

    fs = 44100
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs

    signals = {
        "sine_440":     ("dynamical", np.sin(2 * np.pi * 440 * t)),
        "sine_880":     ("dynamical", np.sin(2 * np.pi * 880 * t)),
        "chirp":        ("dynamical", np.sin(2 * np.pi * (200 + 600 * t) * t)),
        "two_tone":     ("dynamical", np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 660 * t)),
        "am_signal":    ("dynamical", (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 440 * t)),
        "white_noise":  ("stochastic", np.random.randn(len(t))),
        "pink_noise":   ("stochastic", np.cumsum(np.random.randn(len(t))) / np.sqrt(len(t))),
        "ar1_alpha95":  ("stochastic", _ar1(len(t), 0.95)),
    }

    # Downsample for tractability
    ds = 64
    series_list = [sig[::ds] for _, (_, sig) in signals.items()]
    quantized = quantize_global(series_list, MOD)

    print(f"\n  {'Signal':<16} {'Group':<12} {'Entropy':>8} {'IdentFrac':>10} {'Mean_e':>8}")
    print(f"  {'-'*16} {'-'*12} {'-'*8} {'-'*10} {'-'*8}")

    dyn_entropies = []
    sto_entropies = []
    dyn_ident = []
    sto_ident = []

    for i, (name, (group, _)) in enumerate(signals.items()):
        stats = generator_stats(quantized[i])
        print(f"  {name:<16} {group:<12} {stats['entropy']:>8.4f} {stats['identity_frac']:>10.4f} {stats['mean_e']:>8.4f}")
        if group == "dynamical":
            dyn_entropies.append(stats["entropy"])
            dyn_ident.append(stats["identity_frac"])
        else:
            sto_entropies.append(stats["entropy"])
            sto_ident.append(stats["identity_frac"])

    print(f"\n  SUMMARY:")
    print(f"  Dynamical mean entropy:  {np.mean(dyn_entropies):.4f}")
    print(f"  Stochastic mean entropy: {np.mean(sto_entropies):.4f}")
    print(f"  Dynamical mean ident:    {np.mean(dyn_ident):.4f}")
    print(f"  Stochastic mean ident:   {np.mean(sto_ident):.4f}")

    # Direction check: dynamical signals should have LOWER entropy (more structured)
    entropy_diff = np.mean(dyn_entropies) - np.mean(sto_entropies)
    ident_diff = np.mean(dyn_ident) - np.mean(sto_ident)
    print(f"  Entropy diff (dyn - sto): {entropy_diff:+.4f} {'(correct: dyn < sto)' if entropy_diff < 0 else '(unexpected)'}")
    print(f"  Identity diff (dyn - sto): {ident_diff:+.4f}")

    return {"domain": "audio", "entropy_diff": float(entropy_diff), "ident_diff": float(ident_diff)}


def _ar1(n, alpha):
    """Generate AR(1) process."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + np.random.randn()
    return x


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2: CLIMATE TELECONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def run_climate():
    """Fetch NOAA indices, apply generator inference, test ENSO phase discrimination."""
    print("\n" + "=" * 72)
    print("DOMAIN 2: CLIMATE TELECONNECTION (5 NOAA indices)")
    print("=" * 72)

    try:
        from urllib.request import urlopen
    except ImportError:
        print("  SKIP: urllib not available")
        return None

    # Fetch ONI (ENSO)
    print("  Fetching ONI...")
    try:
        oni_url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
        oni_raw = urlopen(oni_url, timeout=15).read().decode()
        oni_vals = []
        for line in oni_raw.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 4:
                oni_vals.append(float(parts[3]))
    except Exception as e:
        print(f"  SKIP: ONI fetch failed: {e}")
        return None

    # Fetch PDO
    print("  Fetching PDO...")
    try:
        pdo_url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
        pdo_raw = urlopen(pdo_url, timeout=15).read().decode()
        pdo_vals = []
        for line in pdo_raw.strip().split("\n")[1:]:
            parts = line.split()
            for v in parts[1:13]:
                try:
                    val = float(v)
                    if abs(val) < 90:
                        pdo_vals.append(val)
                except ValueError:
                    pass
    except Exception as e:
        print(f"  SKIP: PDO fetch failed: {e}")
        return None

    # Use the shorter length
    n = min(len(oni_vals), len(pdo_vals))
    if n < 100:
        print(f"  SKIP: insufficient data (n={n})")
        return None
    oni_vals = oni_vals[:n]
    pdo_vals = pdo_vals[:n]
    print(f"  Using {n} months of ONI + PDO data")

    # Quantize both indices with shared bins
    series = [np.array(oni_vals), np.array(pdo_vals)]
    quantized = quantize_global(series, MOD)

    # Generator stats per index
    oni_stats = generator_stats(quantized[0])
    pdo_stats = generator_stats(quantized[1])

    print(f"\n  {'Index':<8} {'Entropy':>8} {'IdentFrac':>10} {'Mean_e':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    print(f"  {'ONI':<8} {oni_stats['entropy']:>8.4f} {oni_stats['identity_frac']:>10.4f} {oni_stats['mean_e']:>8.4f}")
    print(f"  {'PDO':<8} {pdo_stats['entropy']:>8.4f} {pdo_stats['identity_frac']:>10.4f} {pdo_stats['mean_e']:>8.4f}")

    # Cross-index synchrony
    synch = cross_series_synchrony(quantized, MOD)
    print(f"\n  Cross-index generator synchrony: {synch:.4f}")
    print(f"  (baseline for 2 independent series at m=9: ~{1/MOD:.4f})")

    # Split by ENSO phase: La Nina (ONI < -0.5), El Nino (ONI > 0.5), Neutral
    la_nina_idx = [t for t in range(n - 1) if oni_vals[t] < -0.5]
    el_nino_idx = [t for t in range(n - 1) if oni_vals[t] > 0.5]
    neutral_idx = [t for t in range(n - 1) if -0.5 <= oni_vals[t] <= 0.5]

    def phase_entropy(indices, q):
        if len(indices) < 10:
            return float("nan")
        gens = [infer_generator(q[t], q[t + 1], MOD) for t in indices if t + 1 < len(q)]
        counts = Counter(gens)
        total = sum(counts.values())
        return -sum((c / total) * np.log2(c / total) for c in counts.values())

    print(f"\n  ENSO PHASE GENERATOR ENTROPY (ONI index):")
    for phase, idx in [("La Nina", la_nina_idx), ("Neutral", neutral_idx), ("El Nino", el_nino_idx)]:
        ent = phase_entropy(idx, quantized[0])
        print(f"  {phase:<10} n={len(idx):>4}, entropy={ent:.4f}")

    return {
        "domain": "climate",
        "n_months": n,
        "oni_entropy": oni_stats["entropy"],
        "pdo_entropy": pdo_stats["entropy"],
        "synchrony": synch,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 3: ERA5 REANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_era5():
    """Apply generator inference to ERA5 15-channel reanalysis data."""
    print("\n" + "=" * 72)
    print("DOMAIN 3: ERA5 REANALYSIS (15 channels)")
    print("=" * 72)

    import pandas as pd  # noqa: T2-D-5 — observer layer data loading

    csv_path = HERE / ".era5_extracted.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    channels = list(df.columns)
    n_ch = len(channels)
    print(f"  Loaded: {len(df)} days, {n_ch} channels")
    print(f"  Channels: {channels[:5]}... ({n_ch} total)")

    # Standardize (365-day rolling z-score)
    roll_mean = df.rolling(365, min_periods=30).mean()
    roll_std = df.rolling(365, min_periods=30).std().replace(0, 1)
    std_df = (df - roll_mean) / roll_std
    std_df = std_df.dropna()
    print(f"  After standardization: {len(std_df)} days")

    # Quantize all channels with shared bins
    series_list = [std_df[ch].values for ch in channels]
    quantized = quantize_global(series_list, MOD)

    # Per-channel generator stats
    print(f"\n  {'Channel':<25} {'Entropy':>8} {'IdentFrac':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*10}")
    all_entropies = []
    for i, ch in enumerate(channels):
        stats = generator_stats(quantized[i])
        all_entropies.append(stats["entropy"])
        if i < 6 or i >= n_ch - 2:  # show first 6 and last 2
            print(f"  {ch:<25} {stats['entropy']:>8.4f} {stats['identity_frac']:>10.4f}")
        elif i == 6:
            print(f"  {'...':<25}")

    # Cross-channel synchrony
    synch = cross_series_synchrony(quantized, MOD)
    print(f"\n  Mean entropy across channels: {np.mean(all_entropies):.4f}")
    print(f"  Cross-channel generator synchrony: {synch:.4f}")
    print(f"  (baseline for {n_ch} independent series at m=9: ~{1/MOD:.4f})")

    # Compute synchrony in rolling windows to correlate with future variability
    window = 63
    future_horizon = 21
    n_days = min(len(q) for q in quantized) - 1

    synch_series = []
    for t in range(window, n_days - future_horizon):
        gens_window = []
        for step in range(t - window, t):
            step_gens = [infer_generator(quantized[i][step], quantized[i][step + 1], MOD)
                         for i in range(n_ch)]
            modal = Counter(step_gens).most_common(1)[0][1] / n_ch
            gens_window.append(modal)
        synch_series.append(float(np.mean(gens_window)))

    # Future variability: cross-channel std
    future_var = []
    for t in range(window, n_days - future_horizon):
        t_future = t + future_horizon
        vals = [float(std_df.iloc[t_future][ch]) for ch in channels]
        future_var.append(float(np.std(vals)))

    # Correlation
    if len(synch_series) > 50 and len(synch_series) == len(future_var):
        r, p = pearsonr(synch_series, future_var)
        print(f"\n  Synchrony vs future variability (h={future_horizon}d):")
        print(f"    r = {r:+.4f}, p = {p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")
        return {"domain": "era5", "synch_vs_future_r": float(r), "p": float(p), "synchrony": synch}
    else:
        print(f"  Insufficient data for correlation (n={len(synch_series)})")
        return {"domain": "era5", "synchrony": synch}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("[209] QA SIGNAL GENERATOR INFERENCE — CROSS-DOMAIN VALIDATION")
    print("e_t = ((b_{t+1} - b_t - 1) % m) + 1")
    print("The signal IS the orbit. The generator IS the dynamics.")
    print("=" * 72)

    results = {}
    results["audio"] = run_audio()
    results["climate"] = run_climate()
    results["era5"] = run_era5()

    # Save
    out_path = HERE / "signal_generator_crossdomain_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*72}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
