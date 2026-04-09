#!/usr/bin/env python3
"""
eeg_combined_observer_test.py — Combined Observer: Topographic k-means + RNS Eigenspectrum

Tests whether the RNS dual-modulus architecture (eigenspectrum coherence projection)
adds discriminative power BEYOND the topographic k-means Observer 3.

Nested model hierarchy:
  Model 0: intercept only
  Model 1: delta power ratio (classical EEG baseline)
  Model 2: delta + Observer 3 (singularity_frac, cosmos_frac)           [+2 df]
  Model 3: delta + Observer 3 + RNS (sing9, cos9, crt, mean_f9)        [+4 df]

If Model 3 beats Model 2, then the RNS eigenspectrum captures information
that topographic k-means CANNOT — specifically the CRT cross-modulus consistency
and f-value orbit norm that encode mode structure differently from microstates.

Standalone script. Runs on chb01 real CHB-MIT data.
"""

QA_COMPLIANCE = "combined observer test — Obs3 + RNS eigenspectrum on real EEG"

import sys
import json
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, str(Path("/home/player2/wt-papers")))
sys.path.insert(0, str(Path("/home/player2/signal_experiments")))

from eeg_chbmit_scale import (
    load_patient_dataset, fit_patient_kmeans,
    delta_power_ratio,
)
from eeg_orbit_observer_comparison import classify_segment_topographic
from eeg_orbit_classifier import compute_orbit_sequence, orbit_statistics
from eeg_rns_observer import extract_segment_features

np.random.seed(42)

PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")


# ── Logistic regression utilities (standalone, same as other scripts) ─────────

def _fit_logistic(X, y, lr=0.1, n_iter=3000, l2=1e-4):
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        logits = np.clip(X @ beta, -30, 30)
        probs = expit(logits)
        beta -= lr * (X.T @ (probs - y) / len(y) + l2 * beta)
    return beta


def _ll(X, y, beta):
    logits = np.clip(X @ beta, -30, 30)
    probs = np.clip(expit(logits), 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def _std(x):
    sd = x.std()
    return (x - x.mean()) / (sd + 1e-9)


def main():
    print("=" * 72)
    print("COMBINED OBSERVER TEST — Topographic k-means + RNS Eigenspectrum")
    print("Patient: chb01 (real CHB-MIT data)")
    print("=" * 72)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading chb01 data...")
    dataset = load_patient_dataset(PATIENT_DIR)
    if not dataset:
        print("ERROR: No data loaded.")
        sys.exit(1)

    n_sei = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    print(f"  Loaded: {n_sei} seizure + {n_base} baseline windows")

    # Filter to modal channel count (same as eeg_chbmit_scale.py)
    from collections import Counter
    ch_counts = Counter(d["multi_ch"].shape[0] for d in dataset)
    modal_ch = ch_counts.most_common(1)[0][0]
    dataset = [d for d in dataset if d["multi_ch"].shape[0] == modal_ch]
    n_sei = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    print(f"  After montage filter ({modal_ch} ch): {n_sei} seizure + {n_base} baseline")

    if n_sei < 2 or n_base < 2:
        print("ERROR: Need at least 2 windows of each type.")
        sys.exit(1)

    fs = dataset[0]["fs"]
    y = np.array([1.0 if d["type"] == "seizure" else 0.0 for d in dataset])

    # ── Feature extraction ───────────────────────────────────────────────────

    # 1. Delta power ratio (classical baseline)
    print("\nExtracting features...")
    print("  [1/3] Delta power ratio...")
    delta = np.array([
        delta_power_ratio(d["waveform"].astype(np.float64), fs)
        for d in dataset
    ])

    # 2. Observer 3: topographic k-means → orbit fractions
    print("  [2/3] Observer 3 (topographic k-means)...")
    km, c2s = fit_patient_kmeans(dataset)
    obs3_sing = []
    obs3_cos = []
    for d in dataset:
        ms = classify_segment_topographic(d["multi_ch"], km, c2s, fs)
        orb = orbit_statistics(compute_orbit_sequence(ms))
        obs3_sing.append(orb["singularity_frac"])
        obs3_cos.append(orb["cosmos_frac"])
    obs3_sing = np.array(obs3_sing)
    obs3_cos = np.array(obs3_cos)

    # 3. RNS eigenspectrum features
    print("  [3/3] RNS eigenspectrum observer...")
    rns_features = []
    for d in dataset:
        feats = extract_segment_features(d["multi_ch"], fs=d["fs"])
        rns_features.append(feats)
    rns_sing9 = np.array([f["singularity_frac_9"] for f in rns_features])
    rns_cos9 = np.array([f["cosmos_frac_9"] for f in rns_features])
    rns_crt = np.array([f["crt_consistency"] for f in rns_features])
    rns_mf9 = np.array([f["mean_f_9"] for f in rns_features])

    n = len(y)
    print(f"\n  Feature matrix: {n} windows")

    # ── Nested model hierarchy ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("NESTED MODEL HIERARCHY")
    print("=" * 72)

    # Model 0: intercept only
    X0 = np.ones((n, 1))
    ll0 = _ll(X0, y, _fit_logistic(X0, y))

    # Model 1: delta power ratio
    X1 = np.c_[np.ones(n), _std(delta)]
    ll1 = _ll(X1, y, _fit_logistic(X1, y))

    # Model 2: delta + Observer 3 (sing, cos)
    X2 = np.c_[np.ones(n), _std(delta), _std(obs3_sing), _std(obs3_cos)]
    ll2 = _ll(X2, y, _fit_logistic(X2, y))

    # Model 3: delta + Observer 3 + RNS (sing9, cos9, crt, mean_f9)
    X3 = np.c_[np.ones(n), _std(delta), _std(obs3_sing), _std(obs3_cos),
               _std(rns_sing9), _std(rns_cos9), _std(rns_crt), _std(rns_mf9)]
    ll3 = _ll(X3, y, _fit_logistic(X3, y))

    # Also test: Model 2b: delta + RNS only (without Observer 3)
    X2b = np.c_[np.ones(n), _std(delta), _std(rns_sing9), _std(rns_cos9),
                _std(rns_crt), _std(rns_mf9)]
    ll2b = _ll(X2b, y, _fit_logistic(X2b, y))

    # R² values (McFadden pseudo-R²)
    r2_0 = 0.0
    r2_1 = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_2 = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    r2_3 = 1.0 - ll3 / ll0 if ll0 != 0 else 0.0
    r2_2b = 1.0 - ll2b / ll0 if ll0 != 0 else 0.0

    # LR tests
    # Obs3 beyond delta: Model 2 vs Model 1 (df=2)
    lr_obs3 = 2.0 * (ll2 - ll1)
    p_obs3 = float(chi2.sf(max(0, lr_obs3), df=2))

    # RNS beyond delta: Model 2b vs Model 1 (df=4)
    lr_rns = 2.0 * (ll2b - ll1)
    p_rns = float(chi2.sf(max(0, lr_rns), df=4))

    # RNS beyond Obs3: Model 3 vs Model 2 (df=4)
    lr_rns_beyond_obs3 = 2.0 * (ll3 - ll2)
    p_rns_beyond_obs3 = float(chi2.sf(max(0, lr_rns_beyond_obs3), df=4))

    # Obs3 beyond RNS: Model 3 vs Model 2b (df=2)
    lr_obs3_beyond_rns = 2.0 * (ll3 - ll2b)
    p_obs3_beyond_rns = float(chi2.sf(max(0, lr_obs3_beyond_rns), df=2))

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    print(f"\n  {'Model':<45} {'R²':>8} {'ΔR²':>8} {'LR':>8} {'p':>10} {'Sig':>5}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*5}")
    print(f"  {'M0: intercept only':<45} {r2_0:>8.4f}")
    print(f"  {'M1: delta':<45} {r2_1:>8.4f} {r2_1-r2_0:>+8.4f}")
    print(f"  {'M2: delta + Obs3':<45} {r2_2:>8.4f} {r2_2-r2_1:>+8.4f} {lr_obs3:>8.3f} {p_obs3:>10.6f} {sig(p_obs3):>5}")
    print(f"  {'M2b: delta + RNS':<45} {r2_2b:>8.4f} {r2_2b-r2_1:>+8.4f} {lr_rns:>8.3f} {p_rns:>10.6f} {sig(p_rns):>5}")
    print(f"  {'M3: delta + Obs3 + RNS':<45} {r2_3:>8.4f} {r2_3-r2_2:>+8.4f} {lr_rns_beyond_obs3:>8.3f} {p_rns_beyond_obs3:>10.6f} {sig(p_rns_beyond_obs3):>5}")

    print(f"\n  KEY COMPARISONS:")
    print(f"  Obs3 beyond delta:    ΔR² = {r2_2-r2_1:+.4f}, p = {p_obs3:.6f} {sig(p_obs3)}")
    print(f"  RNS beyond delta:     ΔR² = {r2_2b-r2_1:+.4f}, p = {p_rns:.6f} {sig(p_rns)}")
    print(f"  RNS beyond Obs3:      ΔR² = {r2_3-r2_2:+.4f}, p = {p_rns_beyond_obs3:.6f} {sig(p_rns_beyond_obs3)}")
    print(f"  Obs3 beyond RNS:      ΔR² = {r2_3-r2_2b:+.4f}, p = {p_obs3_beyond_rns:.6f} {sig(p_obs3_beyond_rns)}")
    print(f"  Combined:             R² = {r2_3:.4f} (vs delta-only {r2_1:.4f})")

    # ── Feature distributions ────────────────────────────────────────────────
    print(f"\n" + "-" * 72)
    print("FEATURE DISTRIBUTIONS")
    print(f"  {'Feature':<25} {'Seizure':>10} {'Baseline':>10} {'Diff':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    features_table = [
        ("delta_ratio", delta),
        ("obs3_singularity", obs3_sing),
        ("obs3_cosmos", obs3_cos),
        ("rns_singularity_9", rns_sing9),
        ("rns_cosmos_9", rns_cos9),
        ("rns_crt_consistency", rns_crt),
        ("rns_mean_f_9", rns_mf9),
    ]
    for name, arr in features_table:
        sei_mean = arr[y == 1].mean()
        base_mean = arr[y == 0].mean()
        print(f"  {name:<25} {sei_mean:>10.4f} {base_mean:>10.4f} {sei_mean-base_mean:>+10.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "patient": "chb01",
        "n_seizure": int(sum(y)),
        "n_baseline": int(len(y) - sum(y)),
        "models": {
            "M0_intercept": {"r2": r2_0},
            "M1_delta": {"r2": float(r2_1)},
            "M2_delta_obs3": {"r2": float(r2_2), "delta_r2_vs_M1": float(r2_2 - r2_1), "p_vs_M1": p_obs3},
            "M2b_delta_rns": {"r2": float(r2_2b), "delta_r2_vs_M1": float(r2_2b - r2_1), "p_vs_M1": p_rns},
            "M3_delta_obs3_rns": {
                "r2": float(r2_3),
                "delta_r2_vs_M2": float(r2_3 - r2_2),
                "p_rns_beyond_obs3": p_rns_beyond_obs3,
                "delta_r2_vs_M2b": float(r2_3 - r2_2b),
                "p_obs3_beyond_rns": p_obs3_beyond_rns,
            },
        },
    }

    out_path = Path("/home/player2/signal_experiments/eeg_combined_observer_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
