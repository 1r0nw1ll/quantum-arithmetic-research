#!/usr/bin/env python3
"""
50_cardiac_preregistered.py — Pre-registered Φ(D) test on cardiac ECG data
===========================================================================

PRE-REGISTRATION (written BEFORE seeing any results):

  Domain: Cardiac (MIT-BIH Arrhythmia Database, PhysioNet)
  Classification: DISORDER-STRESS (arrhythmia disrupts normal sinus rhythm)
  Φ(D) = -1
  Prediction: QA orbit features distinguish arrhythmia from normal beats.
  Architecture: Same topographic observer as other domains (k-means → QA → orbit stats)

  This is the 7th domain tested, 5th with real data.
  If QA orbit features add ΔR² beyond R-R interval baseline for discriminating
  arrhythmia, AND survive corrected surrogates, this is a Tier 3 confirmation
  AND a successful Φ(D) pre-registration (path toward Tier 4).

Data: MIT-BIH Arrhythmia Database (48 records, 30 min each, 2-lead ECG, 360 Hz)
      Annotations: N=normal, V=PVC, A=APB, etc.
      Downloaded via wfdb library from PhysioNet.

CORRECTED SURROGATE DESIGN: Real labels + real R-R intervals held fixed,
only QA orbit features are surrogated.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=cardiac_topographic, state_alphabet=ecg_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
from scipy.special import expit
from scipy.stats import chi2 as chi2_dist
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
MODULUS = 9  # mod-9 for cardiac (smaller state space, shorter segments)
N_CLUSTERS = 4
N_SURROGATES = 200

np.random.seed(42)


def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


# MIT-BIH record numbers (standard 48 records)
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# Arrhythmia beat types (non-normal)
ARRHYTHMIA_TYPES = {"V", "A", "F", "f", "j", "a", "J", "S", "E", "/", "!"}


def extract_beat_features(record_name, max_beats=5000):
    """Extract multi-channel beat-level features from one MIT-BIH record.
    Returns list of dicts: {type: 'normal'/'arrhythmia', features: array}"""
    try:
        record = wfdb.rdrecord(record_name, pn_dir="mitdb")
        annotation = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
    except Exception as e:
        print(f"  [{record_name}] Download failed: {e}")
        return []

    signals = record.p_signal  # (n_samples, n_channels)
    fs = record.fs
    beat_samples = annotation.sample
    beat_symbols = annotation.symbol

    beats = []
    # Extract features around each beat
    window_samples = int(0.4 * fs)  # 400ms window around each beat
    half_win = window_samples // 2

    for i in range(1, len(beat_samples) - 1):
        sym = beat_symbols[i]
        if sym == "+" or sym == "~" or sym == "|":
            continue  # rhythm change markers, not beats

        center = beat_samples[i]
        if center - half_win < 0 or center + half_win >= len(signals):
            continue

        # Beat type
        if sym == "N" or sym == ".":
            btype = "normal"
        elif sym in ARRHYTHMIA_TYPES:
            btype = "arrhythmia"
        else:
            continue  # skip unknown types

        # Features: R-R interval + morphological
        rr_prev = (beat_samples[i] - beat_samples[i - 1]) / fs
        rr_next = (beat_samples[i + 1] - beat_samples[i]) / fs

        segment = signals[center - half_win: center + half_win, :]

        # Multi-channel features (4 per channel + 2 R-R = 10 features for 2-lead)
        feats = []
        for ch in range(segment.shape[1]):
            ch_data = segment[:, ch]
            feats.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(ch_data) - np.min(ch_data),  # amplitude
                np.sum(np.abs(np.diff(ch_data))),  # total variation
            ])
        feats.extend([rr_prev, rr_next])

        beats.append({
            "type": btype,
            "features": np.array(feats),
            "rr_prev": rr_prev,
        })

        if len(beats) >= max_beats:
            break

    return beats


def compute_orbit_features(features_array, km, n_clusters, m=MODULUS):
    """Compute QA orbit statistics from feature array via k-means labels."""
    from qa_orbit_rules import orbit_family

    labels = km.predict(features_array.reshape(1, -1) if features_array.ndim == 1
                        else features_array)
    # For single-beat orbit: use consecutive label pairs from the cluster assignment
    # Since we have one beat, use the cluster as b and a default as e
    # Better: compute orbits from sequences of beats
    return labels


def nested_model(y, baseline, feat1, feat2):
    """Nested logistic regression: y ~ baseline vs y ~ baseline + feat1 + feat2."""
    def _std(x):
        sd = x.std()
        return (x - x.mean()) / (sd + 1e-9)

    def _fit(X, y, lr=0.1, n_iter=3000, l2=1e-4):
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

    X0 = np.ones((len(y), 1))
    ll0 = _ll(X0, y, _fit(X0, y))

    X1 = np.c_[np.ones(len(y)), _std(baseline)]
    ll1 = _ll(X1, y, _fit(X1, y))

    X2 = np.c_[np.ones(len(y)), _std(baseline), _std(feat1), _std(feat2)]
    ll2 = _ll(X2, y, _fit(X2, y))

    r2_base = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_full = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    lr_stat = 2.0 * (ll2 - ll1)
    p_val = float(chi2_dist.sf(max(0.0, lr_stat), df=2))

    return {"r2_base": r2_base, "r2_full": r2_full,
            "delta_r2": r2_full - r2_base, "p_qa_add": p_val}


def main():
    print("=" * 72)
    print("PRE-REGISTERED Φ(D) TEST: Cardiac Domain")
    print("Classification: DISORDER-STRESS (Φ = -1)")
    print("Prediction: QA orbit features discriminate arrhythmia from normal")
    print("Data: MIT-BIH Arrhythmia Database (PhysioNet)")
    print("=" * 72)

    # Download and extract features
    all_beats = []
    records_used = 0
    for rec in MITBIH_RECORDS:
        print(f"  Downloading {rec}...", end=" ")
        beats = extract_beat_features(rec)
        n_norm = sum(1 for b in beats if b["type"] == "normal")
        n_arr = sum(1 for b in beats if b["type"] == "arrhythmia")
        print(f"{n_norm} normal, {n_arr} arrhythmia")
        if beats:
            all_beats.extend(beats)
            records_used += 1

    n_normal = sum(1 for b in all_beats if b["type"] == "normal")
    n_arrhythmia = sum(1 for b in all_beats if b["type"] == "arrhythmia")
    print(f"\nTotal: {len(all_beats)} beats ({n_normal} normal, {n_arrhythmia} arrhythmia)")
    print(f"Records used: {records_used}")

    if n_arrhythmia < 50 or n_normal < 50:
        print("Insufficient data")
        return

    # Build arrays
    features = np.array([b["features"] for b in all_beats])
    y = np.array([1.0 if b["type"] == "arrhythmia" else 0.0 for b in all_beats])
    rr = np.array([b["rr_prev"] for b in all_beats])

    # k-means on features (unsupervised — no labels used)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(features)
    labels = km.predict(features)

    # Compute orbit statistics from consecutive beat pairs
    from qa_orbit_rules import orbit_family
    CMAP = {0: 3, 1: 6, 2: 9, 3: 1}  # mod-9 mapping

    sing_fracs, cos_fracs = [], []
    window = 20  # sliding window of 20 beats for orbit stats

    for i in range(len(labels) - window):
        seg_labels = labels[i:i + window]
        orbits = []
        for j in range(len(seg_labels) - 1):
            b = CMAP.get(seg_labels[j], 1)
            e = CMAP.get(seg_labels[j + 1], 1)
            orbits.append(orbit_family(int(b), int(e), MODULUS))

        n_orb = len(orbits)
        sing_fracs.append(sum(1 for o in orbits if o == "singularity") / n_orb)
        cos_fracs.append(sum(1 for o in orbits if o == "cosmos") / n_orb)

    sing_fracs = np.array(sing_fracs)
    cos_fracs = np.array(cos_fracs)

    # Align: use the LAST beat in each window for the label
    y_aligned = y[window:][:len(sing_fracs)]
    rr_aligned = rr[window:][:len(sing_fracs)]

    print(f"\nWindowed: {len(y_aligned)} samples, "
          f"{int(y_aligned.sum())} arrhythmia, {int((1-y_aligned).sum())} normal")

    # ====================================================================
    # REAL NESTED MODEL
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("NESTED MODEL: arrhythmia ~ R-R interval vs ~ R-R + QA_sing + QA_cos")
    print("=" * 70)

    result = nested_model(y_aligned, rr_aligned, sing_fracs, cos_fracs)
    print(f"  R² (R-R only): {result['r2_base']:.4f}")
    print(f"  R² (R-R + QA): {result['r2_full']:.4f}")
    print(f"  ΔR²: {result['delta_r2']:+.4f}")
    print(f"  p(QA adds): {result['p_qa_add']:.6f}")

    sig = "***" if result["p_qa_add"] < 0.001 else "**" if result["p_qa_add"] < 0.01 else "*" if result["p_qa_add"] < 0.05 else "ns"
    print(f"  Significance: {sig}")

    # ====================================================================
    # CORRECTED SURROGATES
    # ====================================================================
    print(f"\n{'=' * 70}")
    print(f"SURROGATE VALIDATION ({N_SURROGATES} iterations)")
    print("Real labels + real R-R held fixed. Only orbit features surrogated.")
    print("=" * 70)

    surr_types = ["permuted_segments", "random_fracs"]
    surr_results = {st: {"delta_r2": []} for st in surr_types}

    for st in surr_types:
        print(f"\n  {st}:")
        for i in range(N_SURROGATES):
            rng = np.random.RandomState(7000 + i)

            if st == "permuted_segments":
                idx = rng.permutation(len(y_aligned))
                s_sing = sing_fracs[idx]
                s_cos = cos_fracs[idx]
            elif st == "random_fracs":
                s_sing = rng.uniform(0, 0.5, len(y_aligned))
                s_cos = rng.uniform(0, 0.5, len(y_aligned))

            try:
                r = nested_model(y_aligned, rr_aligned, s_sing, s_cos)
                surr_results[st]["delta_r2"].append(r["delta_r2"])
            except:
                surr_results[st]["delta_r2"].append(0.0)

            if (i + 1) % 50 == 0:
                sys.stdout.write(f"\r    {i + 1}/{N_SURROGATES}")
                sys.stdout.flush()
        print()

    # Compare
    print(f"\n{'=' * 70}")
    print("SURROGATE COMPARISON")
    print("=" * 70)

    summary = {}
    for st in surr_types:
        vals = np.array(surr_results[st]["delta_r2"])
        vals = vals[np.isfinite(vals)]
        mean_s, std_s = np.mean(vals), np.std(vals)
        rank_p = float(np.mean(vals >= result["delta_r2"]))
        z = (result["delta_r2"] - mean_s) / std_s if std_s > 0 else 0
        beats = "BEATS" if rank_p < 0.05 else "FAILS"
        sig_s = "***" if rank_p < 0.001 else "**" if rank_p < 0.01 else "*" if rank_p < 0.05 else "ns"

        print(f"  {st}: surr_mean={mean_s:+.4f}±{std_s:.4f}, rank_p={rank_p:.4f} → {beats} {sig_s}")
        summary[st] = {"real": float(result["delta_r2"]), "surr_mean": float(mean_s),
                        "surr_std": float(std_s), "rank_p": float(rank_p), "beats": beats == "BEATS"}

    n_pass = sum(1 for v in summary.values() if v["beats"])

    # ====================================================================
    # Φ(D) VERDICT
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("Φ(D) PRE-REGISTRATION VERDICT")
    print("=" * 70)

    checks = {
        "QA adds beyond R-R baseline": result["delta_r2"] > 0 and result["p_qa_add"] < 0.05,
        "Survives surrogates (≥1/2)": n_pass >= 1,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if all(checks.values()):
        print(f"\n  Φ(D) = -1 CONFIRMED for cardiac domain")
        print(f"  QA orbit features discriminate arrhythmia beyond R-R interval")
        print(f"  7th domain tested, 5th Tier 3 if confirmed")
    else:
        print(f"\n  Φ(D) test INCONCLUSIVE or FAILED")

    output = {
        "domain": "cardiac_preregistered",
        "phi_preregistered": -1,
        "classification": "disorder-stress",
        "data": "MIT-BIH Arrhythmia Database",
        "n_beats": len(all_beats), "n_normal": n_normal, "n_arrhythmia": n_arrhythmia,
        "records_used": records_used,
        "real_result": result,
        "surrogate_summary": summary,
        "n_surr_pass": n_pass,
        "phi_confirmed": all(checks.values()),
    }
    with open(os.path.join(HERE, "50_cardiac_preregistered_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to 50_cardiac_preregistered_results.json")


if __name__ == "__main__":
    main()
