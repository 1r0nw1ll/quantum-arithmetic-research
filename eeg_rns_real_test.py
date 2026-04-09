#!/usr/bin/env python3
"""
eeg_rns_real_test.py — Run the RNS observer on real CHB-MIT data (chb01).

Bridges eeg_rns_observer.py (the observer) with eeg_chbmit_scale.py (the data loader).
"""

QA_COMPLIANCE = "bridge script — loads real EEG, applies RNS observer, reports ΔR²"

import sys
import json
import numpy as np  # noqa: T2-D-5 — observer projection layer
from pathlib import Path

# Add wt-papers to path for the data loader
sys.path.insert(0, str(Path("/home/player2/wt-papers")))
sys.path.insert(0, str(Path("/home/player2/signal_experiments")))

from eeg_chbmit_scale import load_patient_dataset  # data loader
from eeg_rns_observer import extract_segment_features, nested_model_rns  # RNS observer

np.random.seed(42)

PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")


def main():
    print("=" * 72)
    print("RNS OBSERVER — REAL CHB-MIT DATA (chb01)")
    print("=" * 72)

    # Load real data
    print("\nLoading chb01 data...")
    dataset = load_patient_dataset(PATIENT_DIR)
    if not dataset:
        print("ERROR: No data loaded. Check path and summary file.")
        sys.exit(1)

    seizure_windows = [d for d in dataset if d["type"] == "seizure"]
    baseline_windows = [d for d in dataset if d["type"] == "baseline"]
    print(f"  Loaded: {len(seizure_windows)} seizure windows, {len(baseline_windows)} baseline windows")

    if len(seizure_windows) < 2 or len(baseline_windows) < 2:
        print("ERROR: Need at least 2 windows of each type.")
        sys.exit(1)

    # Extract RNS features for each window
    print("\nExtracting RNS features...")
    all_features = []
    all_labels = []

    for win in dataset:
        multi_ch = win["multi_ch"]  # shape: (n_channels, n_samples)
        features = extract_segment_features(multi_ch, fs=win["fs"])
        all_features.append(features)
        all_labels.append(1 if win["type"] == "seizure" else 0)

    # Build feature matrix
    feature_names = sorted(all_features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in all_features])
    y = np.array(all_labels)

    print(f"  Feature matrix: {X.shape[0]} windows × {X.shape[1]} features")
    print(f"  Seizure: {sum(y)}, Baseline: {len(y) - sum(y)}")

    # Print orbit distributions
    print("\n" + "-" * 72)
    print("ORBIT DISTRIBUTION (mod-9)")
    print(f"  {'Type':<12} {'Cosmos':>10} {'Satellite':>10} {'Singularity':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

    for label_name, label_val in [("seizure", 1), ("baseline", 0)]:
        mask = y == label_val
        feats = [all_features[i] for i in range(len(y)) if mask[i]]
        if feats:
            cosmos = np.mean([f["cosmos_frac_9"] for f in feats])
            satellite = np.mean([f["satellite_frac_9"] for f in feats])
            singularity = np.mean([f["singularity_frac_9"] for f in feats])
            print(f"  {label_name:<12} {cosmos:>10.4f} {satellite:>10.4f} {singularity:>12.4f}")

    print("\nORBIT DISTRIBUTION (mod-24)")
    print(f"  {'Type':<12} {'Cosmos':>10} {'Satellite':>10} {'Singularity':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

    for label_name, label_val in [("seizure", 1), ("baseline", 0)]:
        mask = y == label_val
        feats = [all_features[i] for i in range(len(y)) if mask[i]]
        if feats:
            cosmos = np.mean([f["cosmos_frac_24"] for f in feats])
            satellite = np.mean([f["satellite_frac_24"] for f in feats])
            singularity = np.mean([f["singularity_frac_24"] for f in feats])
            print(f"  {label_name:<12} {cosmos:>10.4f} {satellite:>10.4f} {singularity:>12.4f}")

    # CRT consistency
    print("\nCRT CONSISTENCY")
    print(f"  {'Type':<12} {'CRT Rate':>10} {'Mean f_9':>10} {'Mean f_24':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for label_name, label_val in [("seizure", 1), ("baseline", 0)]:
        mask = y == label_val
        feats = [all_features[i] for i in range(len(y)) if mask[i]]
        if feats:
            crt = np.mean([f["crt_consistency"] for f in feats])
            mf9 = np.mean([f["mean_f_9"] for f in feats])
            mf24 = np.mean([f["mean_f_24"] for f in feats])
            print(f"  {label_name:<12} {crt:>10.4f} {mf9:>10.2f} {mf24:>10.2f}")

    # Nested logistic regression
    print("\n" + "-" * 72)
    print("NESTED LOGISTIC REGRESSION")

    # Compute delta power ratio as baseline feature (observer projection)
    # Delta band = 0.5-4 Hz; total = 0.5-40 Hz
    delta_ratios = []
    for win in dataset:
        ch0 = win["multi_ch"][0]  # first channel
        fs = win["fs"]
        from numpy.fft import rfft, rfftfreq  # noqa: T2-D-5 — observer projection
        freqs = rfftfreq(len(ch0), 1.0 / fs)
        psd = np.abs(rfft(ch0)) * np.abs(rfft(ch0))
        delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
        total_mask = (freqs >= 0.5) & (freqs <= 40.0)
        delta_power = np.sum(psd[delta_mask])
        total_power = np.sum(psd[total_mask])
        delta_ratios.append(delta_power / max(total_power, 1e-12))
    delta_arr = np.array(delta_ratios)
    sing9_arr = np.array([f["singularity_frac_9"] for f in all_features])
    cosm9_arr = np.array([f["cosmos_frac_9"] for f in all_features])
    crt_arr = np.array([f["crt_consistency"] for f in all_features])
    mf9_arr = np.array([f["mean_f_9"] for f in all_features])

    result = nested_model_rns(y, delta_arr, sing9_arr, cosm9_arr, crt_arr, mf9_arr)

    print(f"  R² (delta only):          {result['r2_delta']:.4f}")
    print(f"  R² (delta + RNS):         {result['r2_full']:.4f}")
    print(f"  Delta R² (RNS beyond delta): +{result['delta_r2']:.4f}")
    print(f"  LR stat:                   {result['lr_stat']:.3f}")
    print(f"  p(RNS adds):               {result['p_rns_add']:.6f}")
    print(f"  Significance:              {'***' if result['p_rns_add'] < 0.001 else '**' if result['p_rns_add'] < 0.01 else '*' if result['p_rns_add'] < 0.05 else 'ns'}")

    print("\n" + "-" * 72)
    print("COMPARISON REFERENCE")
    print("  Topographic k-means (Observer 3): mean ΔR² = +0.210 (10 patients)")
    print(f"  RNS observer (this run):          ΔR² = +{result['delta_r2']:.4f} (chb01 real data)")

    # Save results
    results = {
        "patient": "chb01",
        "n_seizure": int(sum(y)),
        "n_baseline": int(len(y) - sum(y)),
        "orbit_distributions": {},
        "crt_consistency": {},
    }

    for label_name, label_val in [("seizure", 1), ("baseline", 0)]:
        mask = y == label_val
        feats = [all_features[i] for i in range(len(y)) if mask[i]]
        if feats:
            results["orbit_distributions"][label_name] = {
                "cosmos_9": float(np.mean([f["cosmos_frac_9"] for f in feats])),
                "satellite_9": float(np.mean([f["satellite_frac_9"] for f in feats])),
                "singularity_9": float(np.mean([f["singularity_frac_9"] for f in feats])),
            }
            results["crt_consistency"][label_name] = float(np.mean([f["crt_consistency"] for f in feats]))

    results["nested_regression"] = result

    out_path = Path("/home/player2/signal_experiments/eeg_rns_real_results_chb01.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
