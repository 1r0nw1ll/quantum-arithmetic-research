#!/usr/bin/env python3
"""
eeg_214_norm_family.py — Pythagorean family classification for EEG channels.

Extends the [209] signal generator inference with [214] Eisenstein norm-flip
analysis. For each EEG channel at each time step, the (b_t, e_t) pair is
classified into one of four Pythagorean families by its Eisenstein norm mod 9:

    Fibonacci:   f mod 9 ∈ {1, 8}
    Lucas:       f mod 9 ∈ {4, 5}
    Phibonacci:  f mod 9 ∈ {2, 7}
    Null:        f mod 9 = 0

Three new features per channel-window:

    1. FAMILY DISTRIBUTION: fraction of time steps in each of the 4 families.
       Seizure channels might concentrate in fewer families (reduced dynamical
       diversity) compared to baseline.

    2. NORM-SIGN COHERENCE: fraction of adjacent time steps where the norm sign
       ALTERNATES (as [214] predicts for T-orbit traversal). High coherence
       means the channel is following a clean T-orbit; low coherence means
       disrupted dynamics.

    3. DOMINANT FAMILY: the most frequent family per window. A channel that
       shifts from one dominant family during baseline to another during
       seizure would indicate a structural transition.

Runs on CHB-MIT chb01, reports per-feature ΔR² vs seizure/baseline label.
"""

QA_COMPLIANCE = "observer=eeg_norm_family_analyzer, state_alphabet=eeg_quantized_mod9, tier=pythagorean_family_classifier"

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path("/home/player2/signal_experiments")))
sys.path.insert(0, str(Path("/home/player2/wt-papers")))
sys.path.insert(0, str(Path("/home/player2/signal_experiments/qa_lab")))

np.random.seed(42)

PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")
MOD = 9
DOWNSAMPLE = 4

# Pythagorean family classification from [214]
FAMILY_MAP = {
    1: "Fibonacci", 8: "Fibonacci",
    4: "Lucas",     5: "Lucas",
    2: "Phibonacci", 7: "Phibonacci",
    0: "null",
    3: "other_3", 6: "other_6",
}
FAMILY_NAMES = ["Fibonacci", "Lucas", "Phibonacci", "null"]


# ── QA primitives (S1 compliant: b*b not b**2) ──────────────────────────────

def eisenstein_norm(b, e):
    """f(b,e) = b*b + b*e - e*e. Integer, not mod-reduced."""
    return b * b + b * e - e * e


def infer_generator(b_t, b_next, m):
    """[209] generator inference: e = ((b_next - b_t - 1) % m) + 1."""
    return ((b_next - b_t - 1) % m) + 1


def family_of_norm(f_val, m=9):
    """Classify Eisenstein norm mod m into Pythagorean family."""
    return FAMILY_MAP.get(f_val % m, f"other_{f_val % m}")


# ── Quantization (from eeg_209) ─────────────────────────────────────────────

def quantize_multichannel(multi_ch, m, downsample=DOWNSAMPLE):
    """Global percentile bins across all channels. Returns list of list[int]."""
    n_ch = multi_ch.shape[0]
    ds = multi_ch[:, ::downsample]
    n_samp = ds.shape[1]
    if n_samp < 2:
        return [[1] for _ in range(n_ch)]
    all_vals = ds.ravel()
    edges = [float(np.percentile(all_vals, 100 * k / m)) for k in range(1, m)]
    result = []
    for ch in range(n_ch):
        q = []
        for t in range(n_samp):
            val = float(ds[ch, t])
            b = 1
            for edge in edges:
                if val > edge:
                    b += 1
            q.append(b)
        result.append(q)
    return result


# ── New [214] features ───────────────────────────────────────────────────────

def compute_norm_family_features(quantized_channel, m=MOD):
    """Compute [214]-based features for a single channel's quantized time series.

    Returns dict of features:
        family_frac_*: fraction of time steps in each family
        sign_coherence: fraction of adjacent steps with alternating norm sign
        dominant_family: most common family label
        norm_entropy: entropy of the family distribution
    """
    n = len(quantized_channel)
    if n < 3:
        return {f"family_frac_{f}": 0.0 for f in FAMILY_NAMES} | {
            "sign_coherence": 0.0, "dominant_family": "null", "norm_entropy": 0.0
        }

    # Compute (b, e) pairs and classify
    family_counts = Counter()
    norm_signs = []

    for t in range(n - 1):
        b = quantized_channel[t]
        e = infer_generator(quantized_channel[t], quantized_channel[t + 1], m)
        f_val = eisenstein_norm(b, e)
        fam = family_of_norm(f_val, m)
        family_counts[fam] += 1

        sign = 1 if f_val > 0 else (-1 if f_val < 0 else 0)
        norm_signs.append(sign)

    total = sum(family_counts.values())

    # Family distribution fractions
    features = {}
    for f in FAMILY_NAMES:
        features[f"family_frac_{f}"] = family_counts.get(f, 0) / total if total > 0 else 0.0

    # Norm-sign coherence: fraction of adjacent steps with alternating sign
    # [214] predicts f(T(s)) = -f(s), so sign should alternate
    n_alt = 0
    n_pairs = 0
    for i in range(len(norm_signs) - 1):
        if norm_signs[i] != 0 and norm_signs[i + 1] != 0:
            n_pairs += 1
            if norm_signs[i] != norm_signs[i + 1]:
                n_alt += 1
    features["sign_coherence"] = n_alt / n_pairs if n_pairs > 0 else 0.0

    # Dominant family
    if family_counts:
        features["dominant_family"] = family_counts.most_common(1)[0][0]
    else:
        features["dominant_family"] = "null"

    # Family entropy
    if total > 0:
        probs = [c / total for c in family_counts.values() if c > 0]
        features["norm_entropy"] = -sum(p * np.log2(p) for p in probs)
    else:
        features["norm_entropy"] = 0.0

    return features


def compute_all_channel_features(all_quantized, m=MOD):
    """Compute [214] features for all channels. Returns list of feature dicts."""
    return [compute_norm_family_features(ch, m) for ch in all_quantized]


# ── Data loading ────────────────────────────────────────────────────────────

def load_chb01_segments(patient_dir, window_sec=4.0, sr=256):
    """Load CHB-MIT chb01 EEG segments with seizure/baseline labels.

    Returns list of (multi_ch_array, label, file_name) tuples.
    label: 1 = seizure, 0 = baseline.
    """
    try:
        from eeg_chbmit_scale import load_patient_dataset
        return load_patient_dataset(patient_dir, window_sec=window_sec)
    except Exception as ex:
        print(f"WARNING: could not load CHB-MIT data: {ex}")
        print("Generating synthetic demo data instead.")
        return _generate_demo_data(sr, window_sec)


def _generate_demo_data(sr, window_sec):
    """Fallback: 20 synthetic segments (10 baseline, 10 seizure) for testing."""
    n_samp = int(sr * window_sec)
    n_ch = 18
    segments = []
    for i in range(20):
        label = 1 if i >= 10 else 0
        if label == 0:
            data = np.random.randn(n_ch, n_samp) * 50  # noqa: T2-D-5
        else:
            data = np.random.randn(n_ch, n_samp) * 50 + 30 * np.sin(  # noqa: T2-D-5
                2 * np.pi * 3 * np.arange(n_samp) / sr
            )
        segments.append((data, label, f"synthetic_{i:02d}"))
    return segments


# ── Main experiment ──────────────────────────────────────────────────────────

def main():
    print("=== EEG [214] Norm-Family Classification ===")
    print(f"Modulus: {MOD}, Downsample: {DOWNSAMPLE}")
    print()

    segments = load_chb01_segments(PATIENT_DIR)
    if not segments:
        print("No data loaded.")
        return

    print(f"Loaded {len(segments)} segments")

    all_features = []
    all_labels = []

    for seg in segments:
        if isinstance(seg, dict):
            data = np.array(seg.get("multi_ch", []))
            label = 1 if seg.get("type") == "seizure" else 0
        elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
            data, label = seg[0], seg[1]
            if isinstance(label, str):
                label = 1 if label == "seizure" else 0
        else:
            continue
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            continue
        quantized = quantize_multichannel(data, MOD)
        ch_features = compute_all_channel_features(quantized)

        # Average features across channels for this segment
        avg_feat = {}
        numeric_keys = [k for k in ch_features[0] if k != "dominant_family"]
        for k in numeric_keys:
            avg_feat[k] = float(np.mean([f[k] for f in ch_features]))

        # Dominant family = mode across channels
        dom_families = [f["dominant_family"] for f in ch_features]
        avg_feat["dominant_family"] = Counter(dom_families).most_common(1)[0][0]

        all_features.append(avg_feat)
        all_labels.append(label)

    if not all_features:
        print("No features computed.")
        return

    labels = np.array(all_labels)
    n_seizure = int(labels.sum())
    n_baseline = len(labels) - n_seizure
    print(f"Segments: {n_baseline} baseline, {n_seizure} seizure")
    print()

    # Per-feature analysis: ΔR² (simple r² between feature and label)
    numeric_keys = [k for k in all_features[0] if k != "dominant_family"]

    print("=== Per-Feature ΔR² (feature vs seizure label) ===")
    results = {}
    for k in sorted(numeric_keys):
        vals = np.array([f[k] for f in all_features])
        if vals.std() < 1e-10:
            r2 = 0.0
        else:
            corr = np.corrcoef(vals, labels.astype(float))[0, 1]
            r2 = corr * corr  # observer-layer measurement
        direction = "seizure>" if np.mean(vals[labels == 1]) > np.mean(vals[labels == 0]) else "baseline>"
        baseline_mean = float(np.mean(vals[labels == 0])) if n_baseline > 0 else 0.0
        seizure_mean = float(np.mean(vals[labels == 1])) if n_seizure > 0 else 0.0
        print(f"  {k:25s}: R²={r2:.4f}  baseline={baseline_mean:.4f}  seizure={seizure_mean:.4f}  {direction}")
        results[k] = {
            "R2": round(r2, 6),
            "baseline_mean": round(baseline_mean, 6),
            "seizure_mean": round(seizure_mean, 6),
            "direction": direction,
        }

    # Dominant family shift analysis
    print()
    print("=== Dominant Family Distribution ===")
    for lab_name, lab_val in [("baseline", 0), ("seizure", 1)]:
        dom = [f["dominant_family"] for f, l in zip(all_features, all_labels) if l == lab_val]
        dist = Counter(dom)
        total = sum(dist.values())
        print(f"  {lab_name}: {dict(dist)} (n={total})")

    # Save results
    out = {
        "experiment": "eeg_214_norm_family",
        "modulus": MOD,
        "n_segments": len(all_features),
        "n_baseline": n_baseline,
        "n_seizure": n_seizure,
        "per_feature_r2": results,
    }
    out_path = Path("eeg_214_norm_family_results.json")
    with open(out_path, "w") as f_out:
        json.dump(out, f_out, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
