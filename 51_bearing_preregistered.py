#!/usr/bin/env python3
"""
51_bearing_preregistered.py — Pre-registered Φ(D) test on bearing vibration data
==================================================================================

PRE-REGISTRATION (written BEFORE seeing any results):

  Domain: Industrial vibration (CWRU Bearing Fault Dataset)
  Classification: DISORDER-STRESS (Φ = -1)
  Rationale: Bearing fault disrupts the regular rotation pattern.
    Normal bearing = periodic, structured vibration.
    Fault = disruption of that structure (impact events, broadband noise).
    Analogous to cardiac: arrhythmia disrupts sinus rhythm.
  Prediction: QA orbit features discriminate fault from normal beyond
    RMS vibration level baseline.

  Architecture: Same as all other domains (k-means → QA → orbit stats)
  This is the 8th domain tested, 2nd pre-registered Φ(D) attempt.

Data: CWRU Bearing Data Center (Case Western Reserve University)
      12kHz drive-end accelerometer, 4 load conditions (0-3 HP)
      Normal + Inner Race + Ball + Outer Race faults at 7mil diameter
      Downloaded as .mat files via direct URL.

CORRECTED SURROGATE DESIGN: Real labels + real RMS held fixed,
only QA orbit features are surrogated.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=bearing_topographic, state_alphabet=vibration_microstate"

import os, sys, json, io
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats, io as sio
from scipy.stats import chi2 as chi2_dist
from scipy.signal import welch
import urllib.request

from qa_orbit_rules import orbit_family

HERE = os.path.dirname(os.path.abspath(__file__))
MODULUS = 9
N_CLUSTERS = 4
N_SURROGATES = 200
WINDOW_SAMPLES = 2048  # ~170ms at 12kHz
HOP = 1024  # 50% overlap

np.random.seed(42)

# CWRU Bearing Data Center — drive end, 12kHz, 7mil fault diameter
# Format: {file_number: (label, load_hp)}
CWRU_FILES = {
    # Normal baseline
    97:  ("normal", 0), 98:  ("normal", 1), 99:  ("normal", 2), 100: ("normal", 3),
    # Inner race fault 7mil
    105: ("fault", 0), 106: ("fault", 1), 107: ("fault", 2), 108: ("fault", 3),
    # Ball fault 7mil
    118: ("fault", 0), 119: ("fault", 1), 120: ("fault", 2), 121: ("fault", 3),
    # Outer race fault 7mil (centered)
    130: ("fault", 0), 131: ("fault", 1), 132: ("fault", 2), 133: ("fault", 3),
}

CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files"


def download_cwru_mat(file_num):
    """Download a CWRU .mat file and return the drive-end signal."""
    cache_dir = os.path.join(HERE, ".cwru_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{file_num}.mat")

    if not os.path.exists(cache_path):
        url = f"{CWRU_BASE_URL}/{file_num}.mat"
        req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(cache_path, "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"  Failed to download {file_num}.mat: {e}")
            return None

    try:
        mat = sio.loadmat(cache_path)
        # Find the drive-end (DE) signal key
        de_keys = [k for k in mat.keys() if "DE" in k and "time" in k]
        if not de_keys:
            # Try alternative naming
            de_keys = [k for k in mat.keys() if not k.startswith("_")]
            de_keys = [k for k in de_keys if "DE" in k or "de" in k.lower()]
        if not de_keys:
            # Last resort: take first non-metadata key
            de_keys = [k for k in mat.keys() if not k.startswith("_")]
        if de_keys:
            signal = mat[de_keys[0]].flatten()
            return signal
        return None
    except Exception as e:
        print(f"  Failed to load {file_num}.mat: {e}")
        return None


def extract_window_features(signal, window_size=WINDOW_SAMPLES, hop=HOP):
    """Extract multi-channel features from vibration signal windows.
    Returns array of shape (n_windows, n_features)."""
    n = len(signal)
    features = []

    for start in range(0, n - window_size, hop):
        window = signal[start:start + window_size]

        # Time-domain features
        rms = np.sqrt(np.mean(window * window))
        peak = np.max(np.abs(window))
        crest = peak / (rms + 1e-10)
        kurtosis = stats.kurtosis(window)
        skewness = stats.skew(window)

        # Frequency-domain features
        freqs, psd = welch(window, fs=12000, nperseg=min(256, len(window)))
        total_power = np.sum(psd)
        # Band powers (0-1kHz, 1-3kHz, 3-6kHz)
        band1 = np.sum(psd[freqs <= 1000]) / (total_power + 1e-10)
        band2 = np.sum(psd[(freqs > 1000) & (freqs <= 3000)]) / (total_power + 1e-10)
        band3 = np.sum(psd[freqs > 3000]) / (total_power + 1e-10)
        spectral_centroid = np.sum(freqs * psd) / (total_power + 1e-10)

        features.append([rms, peak, crest, kurtosis, skewness,
                         band1, band2, band3, spectral_centroid, total_power])

    return np.array(features)


def nested_model(y, baseline, feat1, feat2):
    """Nested LR using sklearn: y ~ baseline vs y ~ baseline + feat1 + feat2."""
    scaler = StandardScaler()

    n = len(y)
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
    print("PRE-REGISTERED Φ(D) TEST: Bearing Fault Domain")
    print("Classification: DISORDER-STRESS (Φ = -1)")
    print("Prediction: QA orbit features discriminate fault from normal")
    print("Data: CWRU Bearing Data Center")
    print("=" * 72)

    # Download and extract features
    all_features = []
    all_labels = []
    all_rms = []

    for file_num, (label, load) in CWRU_FILES.items():
        print(f"  {file_num}.mat ({label}, {load}HP)...", end=" ")
        signal = download_cwru_mat(file_num)
        if signal is None:
            print("SKIP")
            continue

        feats = extract_window_features(signal)
        print(f"{len(feats)} windows")

        for i in range(len(feats)):
            all_features.append(feats[i])
            all_labels.append(1.0 if label == "fault" else 0.0)
            all_rms.append(feats[i, 0])  # RMS is the baseline feature

    features = np.array(all_features)
    y = np.array(all_labels)
    rms = np.array(all_rms)

    n_normal = int((y == 0).sum())
    n_fault = int((y == 1).sum())
    print(f"\nTotal: {len(y)} windows ({n_normal} normal, {n_fault} fault)")

    if n_fault < 50 or n_normal < 50:
        print("Insufficient data")
        return

    # k-means on features (unsupervised)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(features)
    labels = km.predict(features)

    # Compute orbit statistics from consecutive windows
    CMAP = {0: 3, 1: 6, 2: 9, 3: 1}
    window_size = 20

    sing_fracs, cos_fracs = [], []
    for i in range(len(labels) - window_size):
        seg = labels[i:i + window_size]
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

    y_aligned = y[window_size:][:len(sing_fracs)]
    rms_aligned = rms[window_size:][:len(sing_fracs)]

    print(f"Windowed: {len(y_aligned)} samples, "
          f"{int(y_aligned.sum())} fault, {int((1 - y_aligned).sum())} normal")

    # ====================================================================
    # REAL NESTED MODEL
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("NESTED MODEL: fault ~ RMS vs ~ RMS + QA_sing + QA_cos")
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
    print("Real labels + real RMS held fixed. Only orbit features surrogated.")
    print("=" * 70)

    surr_types = ["permuted_segments", "random_fracs"]
    surr_results = {st: {"delta_r2": []} for st in surr_types}

    for st in surr_types:
        print(f"\n  {st}:")
        for i in range(N_SURROGATES):
            rng = np.random.RandomState(8000 + i)

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

    # ====================================================================
    # Φ(D) VERDICT
    # ====================================================================
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
        print(f"\n  Φ(D) = -1 CONFIRMED for bearing fault domain")
        print(f"  QA orbit features discriminate fault beyond RMS vibration level")
        print(f"  8th domain tested, 6th Tier 3 confirmed, 2nd Φ(D) pre-reg pass")
    else:
        print(f"\n  Φ(D) test INCONCLUSIVE or FAILED for bearing fault domain")

    output = {
        "domain": "bearing_fault_preregistered",
        "phi_preregistered": -1,
        "classification": "disorder-stress",
        "data": "CWRU Bearing Data Center",
        "n_windows": len(y_aligned), "n_normal": n_normal, "n_fault": n_fault,
        "real_result": result,
        "surrogate_summary": summary,
        "n_surr_pass": n_pass,
        "phi_confirmed": all(checks.values()),
    }
    with open(os.path.join(HERE, "51_bearing_preregistered_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to 51_bearing_preregistered_results.json")


if __name__ == "__main__":
    main()
