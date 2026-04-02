#!/usr/bin/env python3
"""
eeg_surrogates.py — Process-level surrogate validation for EEG topographic QA
==============================================================================

CORRECTED DESIGN: Keep REAL seizure labels and REAL delta power.
Replace QA orbit features with surrogates.

Surrogate types:
1. Permuted-microstates: shuffle the topographic microstate sequence per segment
   before computing orbit statistics. Preserves microstate distribution, destroys
   temporal order that QA orbit depends on.
2. Random-orbit-fracs: replace orbit fractions with random values (uniform).
   Tests whether ANY orbit-fraction-shaped predictor adds beyond delta.
3. Permuted-segments: shuffle orbit features across segments (keeps real orbit
   distributions but breaks association with seizure labels).

For each surrogate + each patient: run the same nested LR (seizure ~ delta + surr_sing + surr_cos),
compute ΔR², then Fisher's combined p across patients.

Real result must beat surrogate distribution.
"""

QA_COMPLIANCE = "empirical_observer — EEG signal is observer input; QA discrete orbit is the classifier state"

import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import expit
from scipy.stats import chi2 as chi2_dist
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import sys, json

from eeg_orbit_observer_comparison import (
    _read_edf_all_channels,
    build_topographic_features,
    classify_segment_topographic,
    fit_topographic_kmeans as _fit_km_single,
    _topographic_feature_vector,
    WINDOW_SECONDS,
)
from eeg_orbit_classifier import compute_orbit_sequence, orbit_statistics
from eeg_autocorrelation_baseline import delta_power_ratio
from eeg_chbmit_scale import (
    parse_summary, load_patient_dataset, fit_patient_kmeans,
    nested_model, fishers_combined_p, CHBMIT_ROOT, WINDOW_SEC
)

N_SURROGATES = 200


def analyse_patient_real(patient_id, patient_dir):
    """Run real pipeline, return (y, delta, sing, cos_, result) or None."""
    dataset = load_patient_dataset(patient_dir)
    if not dataset:
        return None

    n_sei = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    if n_sei < 2 or n_base < 2:
        return None

    fs = dataset[0]["fs"]
    km, c2s = fit_patient_kmeans(dataset)

    ch_counts = Counter(d["multi_ch"].shape[0] for d in dataset)
    modal_ch = ch_counts.most_common(1)[0][0]
    dataset_v = [d for d in dataset if d["multi_ch"].shape[0] == modal_ch]

    if sum(1 for d in dataset_v if d["type"] == "seizure") < 2 or \
       sum(1 for d in dataset_v if d["type"] == "baseline") < 2:
        return None

    y = np.array([1 if d["type"] == "seizure" else 0 for d in dataset_v], dtype=float)
    delta = np.array([delta_power_ratio(d["waveform"].astype(np.float64), fs) for d in dataset_v])

    # Compute real microstate sequences and orbit stats per segment
    all_ms_seqs = []
    sing, cos_ = [], []
    for d in dataset_v:
        ms = classify_segment_topographic(d["multi_ch"], km, c2s, fs)
        all_ms_seqs.append(ms)
        orb = orbit_statistics(compute_orbit_sequence(ms))
        sing.append(orb["singularity_frac"])
        cos_.append(orb["cosmos_frac"])

    sing, cos_ = np.array(sing), np.array(cos_)
    result = nested_model(y, delta, sing, cos_)

    return {
        "y": y, "delta": delta, "sing": sing, "cos_": cos_,
        "all_ms_seqs": all_ms_seqs, "result": result,
        "patient_id": patient_id, "km": km, "c2s": c2s,
    }


def surrogate_orbit_stats(all_ms_seqs, surr_type, rng):
    """Generate surrogate orbit statistics from real microstate sequences."""
    surr_sing, surr_cos = [], []

    for ms_seq in all_ms_seqs:
        if surr_type == "permuted_microstates":
            # Shuffle microstate sequence, then compute orbits
            perm_ms = list(ms_seq)
            rng.shuffle(perm_ms)
            orb = orbit_statistics(compute_orbit_sequence(perm_ms))
            surr_sing.append(orb["singularity_frac"])
            surr_cos.append(orb["cosmos_frac"])

        elif surr_type == "random_fracs":
            # Random fractions that sum to ~1
            s = rng.uniform(0, 0.5)
            c = rng.uniform(0, 0.5)
            surr_sing.append(s)
            surr_cos.append(c)

    return np.array(surr_sing), np.array(surr_cos)


def main():
    print("=" * 72)
    print("EEG Surrogate Validation — topographic QA orbit features")
    print("Real seizure labels + real delta. Surrogate orbit features only.")
    print("=" * 72)

    # Discover patients
    patient_dirs = sorted(CHBMIT_ROOT.glob("chb*/"))
    if not patient_dirs:
        print(f"No patients found at {CHBMIT_ROOT}")
        return

    # Step 1: Run real pipeline for all patients
    print(f"\nFound {len(patient_dirs)} patient directories")
    patient_data = {}

    for pdir in patient_dirs:
        pid = pdir.name
        print(f"\n  [{pid}] Processing...")
        try:
            result = analyse_patient_real(pid, pdir)
            if result is not None:
                patient_data[pid] = result
                r = result["result"]
                print(f"  [{pid}] ΔR²={r['delta_r2']:+.4f}, p={r['p_qa_add']:.4f}, "
                      f"n={r['n_sei']}+{r['n_base']}")
            else:
                print(f"  [{pid}] Skipped (insufficient data)")
        except Exception as e:
            print(f"  [{pid}] Error: {e}")

    if len(patient_data) < 3:
        print(f"\nOnly {len(patient_data)} patients — need at least 3")
        return

    # Real aggregate
    real_delta_r2s = [patient_data[pid]["result"]["delta_r2"] for pid in patient_data]
    real_p_values = [patient_data[pid]["result"]["p_qa_add"] for pid in patient_data]
    real_mean_dr2 = np.mean(real_delta_r2s)
    real_fisher_chi2, real_fisher_p = fishers_combined_p(real_p_values)

    print(f"\n{'=' * 72}")
    print(f"REAL AGGREGATE ({len(patient_data)} patients):")
    print(f"  Mean ΔR² = {real_mean_dr2:+.4f}")
    print(f"  Fisher χ² = {real_fisher_chi2:.1f}, combined p = {real_fisher_p:.2e}")
    print(f"{'=' * 72}")

    # Step 2: Surrogates
    surr_types = ["permuted_microstates", "random_fracs", "permuted_segments"]
    surr_results = {st: {"mean_dr2": [], "fisher_chi2": []} for st in surr_types}

    for st in surr_types:
        print(f"\nSURROGATE: {st} ({N_SURROGATES} iterations)")

        for i in range(N_SURROGATES):
            rng = np.random.RandomState(5000 + i)
            surr_dr2s = []
            surr_pvals = []

            for pid, pdata in patient_data.items():
                y = pdata["y"]
                delta = pdata["delta"]

                if st == "permuted_segments":
                    # Shuffle orbit features across segments
                    idx = rng.permutation(len(y))
                    surr_sing = pdata["sing"][idx]
                    surr_cos = pdata["cos_"][idx]
                else:
                    surr_sing, surr_cos = surrogate_orbit_stats(
                        pdata["all_ms_seqs"], st, rng)

                try:
                    r = nested_model(y, delta, surr_sing, surr_cos)
                    surr_dr2s.append(r["delta_r2"])
                    surr_pvals.append(r["p_qa_add"])
                except:
                    surr_dr2s.append(0.0)
                    surr_pvals.append(1.0)

            surr_results[st]["mean_dr2"].append(np.mean(surr_dr2s))
            fc, _ = fishers_combined_p(surr_pvals)
            surr_results[st]["fisher_chi2"].append(fc)

            if (i + 1) % 50 == 0:
                sys.stdout.write(f"\r  {i + 1}/{N_SURROGATES}")
                sys.stdout.flush()
        print()

    # Step 3: Compare
    print(f"\n{'=' * 72}")
    print("SURROGATE COMPARISON")
    print("=" * 72)

    summary = {}
    for st in surr_types:
        print(f"\n--- {st} ---")
        for metric in ["mean_dr2", "fisher_chi2"]:
            vals = np.array(surr_results[st][metric])
            vals = vals[np.isfinite(vals)]
            real_val = real_mean_dr2 if metric == "mean_dr2" else real_fisher_chi2
            mean_s, std_s = np.mean(vals), np.std(vals)

            # One-tailed: real > surrogate
            rank_p = np.mean(vals >= real_val)
            z = (real_val - mean_s) / std_s if std_s > 0 else 0
            beats = "BEATS" if rank_p < 0.05 else "FAILS"
            sig = "***" if rank_p < 0.001 else "**" if rank_p < 0.01 else "*" if rank_p < 0.05 else "ns"

            print(f"  {metric}: real={real_val:+.4f}, surr={mean_s:+.4f}±{std_s:.4f}, "
                  f"z={z:+.2f}, rank_p={rank_p:.4f} → {beats} {sig}")

            summary[f"{st}_{metric}"] = {
                "real": float(real_val), "surr_mean": float(mean_s), "surr_std": float(std_s),
                "z": float(z), "rank_p": float(rank_p), "beats": beats == "BEATS",
            }

    print(f"\n{'=' * 72}")
    print("SCORECARD")
    for metric in ["mean_dr2", "fisher_chi2"]:
        n_pass = sum(1 for st in surr_types if summary.get(f"{st}_{metric}", {}).get("beats", False))
        print(f"  {metric:>15}: {n_pass}/3 surrogate types beaten")

    dr2_pass = sum(1 for st in surr_types if summary.get(f"{st}_mean_dr2", {}).get("beats", False))
    fc_pass = sum(1 for st in surr_types if summary.get(f"{st}_fisher_chi2", {}).get("beats", False))

    if dr2_pass >= 2 or fc_pass >= 2:
        print(f"\nVERDICT: EEG Tier 3 CONFIRMED")
    else:
        print(f"\nVERDICT: EEG does not survive surrogates")
    print("=" * 72)

    output = {
        "domain": "eeg_surrogates",
        "design": "CORRECTED: real seizure labels + real delta, surrogate orbit features only",
        "n_patients": len(patient_data),
        "real": {"mean_dr2": float(real_mean_dr2), "fisher_chi2": float(real_fisher_chi2),
                 "fisher_p": float(real_fisher_p)},
        "n_surrogates": N_SURROGATES,
        "summary": summary,
        "dr2_pass": dr2_pass, "fc_pass": fc_pass,
    }
    with open("eeg_surrogate_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to eeg_surrogate_results.json")


if __name__ == "__main__":
    main()
