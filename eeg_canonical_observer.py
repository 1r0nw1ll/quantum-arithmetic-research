#!/usr/bin/env python3
"""
eeg_canonical_observer.py — Canonical (b,e) Observer for EEG Seizure Detection

Uses domain-forced (b,e) assignment:
  b = amplitude state at a spatial location (channel RMS, quantized)
  e = propagation speed to next location (phase lag along spatial chain)

Both components have physical meaning:
  b = WHERE you are (spatial amplitude state)
  e = HOW FAST you're moving (propagation velocity along scalp)

The orbit classification then reflects real propagation dynamics:
  Cosmos:      actively propagating signal (high e, cycling through states)
  Satellite:   oscillating between nearby states (moderate e)
  Singularity: static spatial pattern (zero propagation)

Architecture (Theorem NT compliant):

  [OBSERVER]   EEG window (23 ch x T samples)
                -> per-channel RMS amplitude (continuous -> b after quantisation)
                -> cross-correlation phase lag between adjacent chain channels
                   (integer samples -> e after A1 mapping)
                -> boundary crossed once for each component
                |
  [QA LAYER]   (b, e) pairs in {1,...,9}^2 — one per spatial chain hop
                -> f = b*b + b*e - e*e  (S1 compliant)
                -> orbit classification via divisibility
                |
  [PROJECTION] orbit fractions, f-values -> nested logistic regression

Axiom compliance:
  A1: all states in {1,...,m}. Uses ((x-1) % m) + 1.
  A2: d = b+e, a = b+2*e — derived, never assigned.
  S1: b*b throughout.
  S2: b, e are Python int inside QA layer.
  T1: QA time = window index.
  T2: EEG signal enters ONLY at observer boundary.

Standalone script.
"""

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "observer": "canonical_amplitude_propagation",
    "axioms_checked": ["A1", "A2", "T1", "T2", "S1", "S2"],
    "b_meaning": "spatial_amplitude_state",
    "e_meaning": "propagation_velocity",
    "state_alphabet": ["cosmos", "satellite", "singularity"],
    "qa_layer_types": "int",
}

import sys
import json
import numpy as np
from pathlib import Path
from numpy.fft import rfft, irfft
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, str(Path("/home/player2/wt-papers")))
sys.path.insert(0, str(Path("/home/player2/signal_experiments")))

from eeg_chbmit_scale import load_patient_dataset

np.random.seed(42)

# ── Parameters ────────────────────────────────────────────────────────────────

MOD = 9
WINDOW_SEC = 10.0
FS_DEFAULT = 256.0
SEARCH_MS = 50       # max phase lag search window (ms)
SCALE_B = 9.0        # scale for amplitude -> b
PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")

# Spatial chains: indices into 23-channel CHB-MIT bipolar montage
# Each chain follows anterior -> posterior along the scalp
CHAINS = {
    "L_temporal":     [0, 1, 2, 3],     # FP1-F7 -> F7-T7 -> T7-P7 -> P7-O1
    "L_parasagittal": [4, 5, 6, 7],     # FP1-F3 -> F3-C3 -> C3-P3 -> P3-O1
    "R_parasagittal": [8, 9, 10, 11],   # FP2-F4 -> F4-C4 -> C4-P4 -> P4-O2
    "R_temporal":     [12, 13, 14, 15],  # FP2-F8 -> F8-T8 -> T8-P8 -> P8-O2
}
# 4 chains x 3 hops each = 12 (b,e) pairs per window


# ── QA Core Arithmetic (canonical imports) ───────────────────────────────────

from qa_orbit_rules import norm_f, orbit_family  # canonical orbit classification


def qa_residue(value: float, m: int) -> int:
    """Observer projection: continuous -> {1,...,m}. A1 compliant."""
    x = int(round(value))
    return ((x - 1) % m) + 1


# ── Observer Projection: Canonical (b,e) ────────────────────────────────────

def channel_rms(data: np.ndarray) -> float:
    """RMS amplitude. Observer layer (continuous)."""
    return float(np.sqrt(np.mean(data * data)))


def phase_lag_samples(ch_a: np.ndarray, ch_b: np.ndarray, fs: float,
                      search_ms: float = SEARCH_MS) -> int:
    """
    Phase lag between two channels in samples (integer).
    Positive = ch_b leads ch_a (signal propagates a -> b).
    Uses cross-correlation peak within search window.

    Returns integer — naturally discrete, not a quantised continuous value.
    """
    sig_a = ch_a - ch_a.mean()
    sig_b = ch_b - ch_b.mean()
    na = np.sqrt(np.sum(sig_a * sig_a))
    nb = np.sqrt(np.sum(sig_b * sig_b))
    if na < 1e-10 or nb < 1e-10:
        return 0

    n = len(sig_a)
    search = int(search_ms * fs / 1000)
    cc = irfft(rfft(sig_a) * np.conj(rfft(sig_b)), n=n)
    cc /= (na * nb)

    # Search within +/- search samples around zero lag
    center = np.concatenate([cc[-search:], cc[:search + 1]])
    peak_idx = int(np.argmax(np.abs(center))) - search
    return peak_idx


def extract_canonical_pairs(multi_ch: np.ndarray, fs: float) -> list[dict]:
    """
    Extract canonical (b,e) pairs from one EEG window.

    For each hop along each spatial chain:
      b = source channel RMS amplitude (quantised to {1,...,9})
      e = |phase lag| to next channel (A1-mapped to {1,...,9})

    Returns list of dicts with b, e, orbit, f_value per pair.
    """
    pairs = []

    # Compute all channel RMS values, normalise to [0,1] for b
    n_ch = multi_ch.shape[0]
    rms_all = np.array([channel_rms(multi_ch[i, :]) for i in range(n_ch)])
    rms_min, rms_max = rms_all.min(), rms_all.max()
    rms_range = rms_max - rms_min
    if rms_range < 1e-12:
        rms_norm = np.full_like(rms_all, 0.5)
    else:
        rms_norm = (rms_all - rms_min) / rms_range

    for chain_name, chain_idx in CHAINS.items():
        for k in range(len(chain_idx) - 1):
            src = chain_idx[k]
            dst = chain_idx[k + 1]

            # b = amplitude state of source channel (observer -> quantise)
            b = qa_residue(rms_norm[src] * SCALE_B, MOD)

            # e = propagation speed (|phase lag| in samples)
            lag = phase_lag_samples(multi_ch[src, :], multi_ch[dst, :], fs)
            # A1 mapping: |lag| -> {1,...,9}
            # lag=0 -> e=1 (singularity-like, no propagation)
            # Higher |lag| -> higher e (more propagation)
            e = qa_residue(abs(lag), MOD)

            # QA layer: all int (S2)
            b = int(b)
            e = int(e)

            f_val = norm_f(b, e)
            orb = orbit_family(b, e, MOD)

            pairs.append({
                "b": b, "e": e,
                "f": f_val,
                "orbit": orb,
                "chain": chain_name,
                "src_ch": src, "dst_ch": dst,
                "lag_samples": lag,
            })

    return pairs


def extract_window_features(multi_ch: np.ndarray, fs: float) -> dict:
    """Extract canonical observer features from one EEG window."""
    pairs = extract_canonical_pairs(multi_ch, fs)
    n = len(pairs)
    if n == 0:
        return {
            "cosmos_frac": 0.0, "satellite_frac": 0.0, "singularity_frac": 0.0,
            "mean_f": 0.0, "mean_abs_lag": 0.0, "mean_b": 0.0, "mean_e": 0.0,
            "n_pairs": 0,
        }

    cosmos = sum(1 for p in pairs if p["orbit"] == "cosmos") / n
    satellite = sum(1 for p in pairs if p["orbit"] == "satellite") / n
    singularity = sum(1 for p in pairs if p["orbit"] == "singularity") / n
    mean_f = float(np.mean([p["f"] for p in pairs]))
    mean_lag = float(np.mean([abs(p["lag_samples"]) for p in pairs]))
    mean_b = float(np.mean([p["b"] for p in pairs]))
    mean_e = float(np.mean([p["e"] for p in pairs]))

    return {
        "cosmos_frac": cosmos,
        "satellite_frac": satellite,
        "singularity_frac": singularity,
        "mean_f": mean_f,
        "mean_abs_lag": mean_lag,
        "mean_b": mean_b,
        "mean_e": mean_e,
        "n_pairs": n,
    }


# ── Logistic regression ─────────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("CANONICAL (b,e) OBSERVER — Amplitude x Propagation")
    print("b = spatial amplitude state, e = propagation velocity")
    print("Patient: chb01 (real CHB-MIT data)")
    print("=" * 72)

    # Load data
    print("\nLoading chb01 data...")
    dataset = load_patient_dataset(PATIENT_DIR)
    if not dataset:
        print("ERROR: No data loaded.")
        sys.exit(1)

    n_sei = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    print(f"  Loaded: {n_sei} seizure + {n_base} baseline windows")

    # Extract features
    print("\nExtracting canonical (b,e) features...")
    all_features = []
    all_labels = []
    for d in dataset:
        feats = extract_window_features(d["multi_ch"], d["fs"])
        feats["type"] = d["type"]
        all_features.append(feats)
        all_labels.append(1 if d["type"] == "seizure" else 0)

    y = np.array(all_labels, dtype=float)

    # Orbit distributions
    print(f"\n{'':2s}{'Type':<12} {'Cosmos':>8} {'Satellite':>10} {'Singularity':>12} {'mean_e':>8} {'mean_f':>8} {'|lag|':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for label in ["seizure", "baseline"]:
        feats = [f for f in all_features if f["type"] == label]
        cos_ = np.mean([f["cosmos_frac"] for f in feats])
        sat_ = np.mean([f["satellite_frac"] for f in feats])
        sin_ = np.mean([f["singularity_frac"] for f in feats])
        me = np.mean([f["mean_e"] for f in feats])
        mf = np.mean([f["mean_f"] for f in feats])
        ml = np.mean([f["mean_abs_lag"] for f in feats])
        print(f"  {label:<12} {cos_:>8.4f} {sat_:>10.4f} {sin_:>12.4f} {me:>8.2f} {mf:>8.2f} {ml:>8.2f}")

    # Delta power ratio (baseline)
    from eeg_rns_observer import delta_power_ratio  # noqa: T2-D-5
    delta = np.array([
        delta_power_ratio(d["waveform"].astype(np.float64), d["fs"])
        for d in dataset
    ])

    # Canonical observer features
    sing = np.array([f["singularity_frac"] for f in all_features])
    cos_ = np.array([f["cosmos_frac"] for f in all_features])
    mean_f = np.array([f["mean_f"] for f in all_features])
    mean_e = np.array([f["mean_e"] for f in all_features])

    # Nested model
    print(f"\n{'='*72}")
    print("NESTED LOGISTIC REGRESSION")

    n = len(y)
    X0 = np.ones((n, 1))
    ll0 = _ll(X0, y, _fit_logistic(X0, y))

    X1 = np.c_[np.ones(n), _std(delta)]
    ll1 = _ll(X1, y, _fit_logistic(X1, y))

    # Model 2: delta + canonical QA (sing, cos, mean_f, mean_e)
    X2 = np.c_[np.ones(n), _std(delta), _std(sing), _std(cos_),
               _std(mean_f), _std(mean_e)]
    ll2 = _ll(X2, y, _fit_logistic(X2, y))

    r2_delta = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_full = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    delta_r2 = r2_full - r2_delta
    lr_stat = 2.0 * (ll2 - ll1)
    p_val = float(chi2.sf(max(0, lr_stat), df=4))

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    print(f"\n  R² (delta only):                    {r2_delta:.4f}")
    print(f"  R² (delta + canonical QA):           {r2_full:.4f}")
    print(f"  ΔR² (canonical QA beyond delta):     {delta_r2:+.4f}")
    print(f"  LR stat:                             {lr_stat:.3f}")
    print(f"  p(QA adds):                          {p_val:.6f} {sig(p_val)}")

    print(f"\n  COMPARISON:")
    print(f"  Observer 3 (topographic k-means):     ΔR² = +0.252  (chb01)")
    print(f"  RNS eigenspectrum:                    ΔR² = +0.138  (chb01)")
    print(f"  Canonical (b=amplitude, e=propagation): ΔR² = {delta_r2:+.4f}  (chb01)")

    # Save
    results = {
        "patient": "chb01",
        "observer": "canonical_amplitude_propagation",
        "b_meaning": "spatial_amplitude_state",
        "e_meaning": "propagation_velocity_phase_lag",
        "n_seizure": int(sum(y)),
        "n_baseline": int(len(y) - sum(y)),
        "n_pairs_per_window": 12,
        "chains": list(CHAINS.keys()),
        "r2_delta": float(r2_delta),
        "r2_full": float(r2_full),
        "delta_r2": float(delta_r2),
        "lr_stat": float(lr_stat),
        "p_qa_add": float(p_val),
    }
    out_path = Path("eeg_canonical_observer_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
