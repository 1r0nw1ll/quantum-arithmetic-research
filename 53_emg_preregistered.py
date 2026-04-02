#!/usr/bin/env python3
"""
53_emg_preregistered.py — Pre-registered Φ(D) test on EMG data
================================================================

PRE-REGISTRATION (written BEFORE seeing any results):

  Domain: Electromyography (PhysioNet EMG Database)
  Classification: DISORDER-STRESS (Φ = -1)
  Rationale: Myopathy and neuropathy disrupt normal motor unit recruitment.
    Normal EMG = organized motor unit firing patterns.
    Pathological = disrupted recruitment, altered waveform morphology.
    Directly analogous to EEG (seizure disrupts brain rhythm) and
    cardiac (arrhythmia disrupts sinus rhythm).
  Prediction: QA orbit features discriminate pathological from normal
    beyond RMS amplitude baseline.

  Architecture: Same topographic observer (k-means → QA → orbit stats).
  This is the 3rd Φ(D) attempt (1st cardiac PASS, bearing ceiling, network mismatch).
  EMG is a TEMPORAL MULTI-CHANNEL signal — meets all QA requirements.

Data: PhysioNet EMG Database (emgdb/1.0.0)
      3 records: emg_healthy, emg_myopathy, emg_neuropathy
      Each has MUAP (motor unit action potential) waveforms at ~4kHz
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=emg_topographic, state_alphabet=emg_microstate"

import os, sys, json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import chi2 as chi2_dist
from scipy.signal import welch
import wfdb

from qa_orbit_rules import orbit_family

HERE = os.path.dirname(os.path.abspath(__file__))
MODULUS = 9
N_CLUSTERS = 4
N_SURROGATES = 200
WINDOW_SAMPLES = 512
HOP = 256

np.random.seed(42)


def extract_emg_features(record_name, pn_dir="emgdb/1.0.0"):
    """Download and extract windowed features from one EMG record."""
    try:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    except Exception as e:
        print(f"  Failed: {e}")
        return None, None

    signal = record.p_signal  # (n_samples, n_channels)
    fs = record.fs
    n_channels = signal.shape[1]
    n_samples = signal.shape[0]

    print(f"    {n_samples} samples, {n_channels} channels, {fs}Hz")

    features = []
    for start in range(0, n_samples - WINDOW_SAMPLES, HOP):
        window = signal[start:start + WINDOW_SAMPLES, :]

        feats = []
        for ch in range(n_channels):
            ch_data = window[:, ch]
            rms = np.sqrt(np.mean(ch_data * ch_data))
            peak = np.max(np.abs(ch_data))
            zcr = np.sum(np.abs(np.diff(np.sign(ch_data)))) / (2 * len(ch_data))
            # Frequency features
            f, psd = welch(ch_data, fs=fs, nperseg=min(128, len(ch_data)))
            total_p = np.sum(psd) + 1e-10
            mean_freq = np.sum(f * psd) / total_p
            median_freq_idx = np.searchsorted(np.cumsum(psd) / total_p, 0.5)
            median_freq = f[min(median_freq_idx, len(f) - 1)]

            feats.extend([rms, peak, zcr, mean_freq, median_freq])

        features.append(feats)

    return np.array(features), fs


def nested_model(y, baseline, feat1, feat2):
    scaler = StandardScaler()

    p1 = np.clip(y.mean(), 1e-10, 1 - 1e-10)
    ll0 = float(np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1)))

    X1 = scaler.fit_transform(baseline.reshape(-1, 1))
    lr1 = LogisticRegression(C=1e4, max_iter=1000, solver='lbfgs')
    lr1.fit(X1, y)
    p1_pred = np.clip(lr1.predict_proba(X1)[:, 1], 1e-10, 1 - 1e-10)
    ll1 = float(np.sum(y * np.log(p1_pred) + (1 - y) * np.log(1 - p1_pred)))

    X2 = scaler.fit_transform(np.c_[baseline, feat1, feat2])
    lr2 = LogisticRegression(C=1e4, max_iter=1000, solver='lbfgs')
    lr2.fit(X2, y)
    p2_pred = np.clip(lr2.predict_proba(X2)[:, 1], 1e-10, 1 - 1e-10)
    ll2 = float(np.sum(y * np.log(p2_pred) + (1 - y) * np.log(1 - p2_pred)))

    r2_base = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_full = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    lr_stat = 2.0 * (ll2 - ll1)
    p_val = float(chi2_dist.sf(max(0.0, lr_stat), df=2))

    return {"r2_base": r2_base, "r2_full": r2_full,
            "delta_r2": r2_full - r2_base, "p_qa_add": p_val}


def main():
    print("=" * 72)
    print("PRE-REGISTERED Φ(D) TEST: EMG Domain")
    print("Classification: DISORDER-STRESS (Φ = -1)")
    print("Prediction: QA orbit features discriminate pathological from normal")
    print("Data: PhysioNet EMG Database")
    print("=" * 72)

    records = {
        "emg_healthy": "normal",
        "emg_myopathy": "pathological",
        "emg_neuropathy": "pathological",
    }

    all_features = []
    all_labels = []
    all_rms = []

    for rec_name, label in records.items():
        print(f"\n  {rec_name} ({label}):")
        feats, fs = extract_emg_features(rec_name)
        if feats is None:
            continue
        print(f"    {len(feats)} windows extracted")

        for i in range(len(feats)):
            all_features.append(feats[i])
            all_labels.append(1.0 if label == "pathological" else 0.0)
            all_rms.append(feats[i, 0])  # First feature is RMS ch0

    features = np.array(all_features)
    y = np.array(all_labels)
    rms = np.array(all_rms)

    n_normal = int((y == 0).sum())
    n_path = int((y == 1).sum())
    print(f"\nTotal: {len(y)} windows ({n_normal} normal, {n_path} pathological)")

    if n_path < 30 or n_normal < 30:
        print("Insufficient data")
        return

    # k-means on features (unsupervised)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(features_scaled)
    labels = km.predict(features_scaled)

    # QA orbit statistics
    CMAP = {0: 3, 1: 6, 2: 9, 3: 1}
    window = 20

    sing_fracs, cos_fracs = [], []
    for i in range(len(labels) - window):
        seg = labels[i:i + window]
        orbits = []
        for j in range(len(seg) - 1):
            b = CMAP.get(seg[j], 1)
            e = CMAP.get(seg[j + 1], 1)
            orbits.append(orbit_family(int(b), int(e), MODULUS))
        n_orb = len(orbits)
        sing_fracs.append(sum(1 for o in orbits if o == "singularity") / n_orb)
        cos_fracs.append(sum(1 for o in orbits if o == "cosmos") / n_orb)

    sing_fracs = np.array(sing_fracs)
    cos_fracs = np.array(cos_fracs)

    y_aligned = y[window:][:len(sing_fracs)]
    rms_aligned = rms[window:][:len(sing_fracs)]

    print(f"Windowed: {len(y_aligned)} samples, "
          f"{int(y_aligned.sum())} pathological, {int((1 - y_aligned).sum())} normal")

    # ====================================================================
    # REAL NESTED MODEL
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("NESTED MODEL: pathological ~ RMS vs ~ RMS + QA_sing + QA_cos")
    print("=" * 70)

    result = nested_model(y_aligned, rms_aligned, sing_fracs, cos_fracs)
    print(f"  R² (RMS only): {result['r2_base']:.4f}")
    print(f"  R² (RMS + QA): {result['r2_full']:.4f}")
    print(f"  ΔR²: {result['delta_r2']:+.4f}")
    print(f"  p(QA adds): {result['p_qa_add']:.6f}")

    sig = "***" if result["p_qa_add"] < 0.001 else "**" if result["p_qa_add"] < 0.01 else "*" if result["p_qa_add"] < 0.05 else "ns"
    print(f"  Significance: {sig}")

    # ====================================================================
    # CORRECTED SURROGATES
    # ====================================================================
    print(f"\n{'=' * 70}")
    print(f"SURROGATE VALIDATION ({N_SURROGATES} iterations)")
    print("=" * 70)

    surr_types = ["permuted_segments", "random_fracs"]
    surr_results = {st: {"delta_r2": []} for st in surr_types}

    for st in surr_types:
        print(f"\n  {st}:")
        for i in range(N_SURROGATES):
            rng = np.random.RandomState(10000 + i)
            if st == "permuted_segments":
                idx = rng.permutation(len(y_aligned))
                s_sing = sing_fracs[idx]
                s_cos = cos_fracs[idx]
            elif st == "random_fracs":
                s_sing = rng.uniform(0, 0.5, len(y_aligned))
                s_cos = rng.uniform(0, 0.5, len(y_aligned))

            try:
                r = nested_model(y_aligned, rms_aligned, s_sing, s_cos)
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

        print(f"  {st}: real={result['delta_r2']:+.4f}, surr={mean_s:+.4f}±{std_s:.4f}, "
              f"rank_p={rank_p:.4f} → {beats} {sig_s}")
        summary[st] = {"real": float(result["delta_r2"]), "surr_mean": float(mean_s),
                        "surr_std": float(std_s), "rank_p": float(rank_p), "beats": beats == "BEATS"}

    n_pass = sum(1 for v in summary.values() if v["beats"])

    print(f"\n{'=' * 70}")
    print("Φ(D) PRE-REGISTRATION VERDICT")
    print("=" * 70)

    checks = {
        "QA adds beyond RMS baseline": result["delta_r2"] > 0 and result["p_qa_add"] < 0.05,
        "Survives surrogates (≥1/2)": n_pass >= 1,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if all(checks.values()):
        print(f"\n  Φ(D) = -1 CONFIRMED for EMG domain")
        print(f"  2nd successful Φ(D) pre-registration (after cardiac)")
    else:
        print(f"\n  Φ(D) test FAILED for EMG domain")

    output = {
        "domain": "emg_preregistered", "phi_preregistered": -1,
        "classification": "disorder-stress", "data": "PhysioNet EMG Database",
        "n_windows": len(y_aligned), "n_normal": n_normal, "n_pathological": n_path,
        "real_result": result, "surrogate_summary": summary,
        "n_surr_pass": n_pass, "phi_confirmed": all(checks.values()),
    }
    with open(os.path.join(HERE, "53_emg_preregistered_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to 53_emg_preregistered_results.json")


if __name__ == "__main__":
    main()
