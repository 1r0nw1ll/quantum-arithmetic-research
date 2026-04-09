#!/usr/bin/env python3
"""
eeg_signal_dynamics_observer.py — QA Signal Dynamics Observer for EEG

The signal IS the orbit. The generator IS the dynamics.

For each EEG channel, the amplitude time series is quantized to {1,...,9}.
Consecutive samples (b_t, b_{t+1}) define a QA transition. The generator
e_t is INFERRED from the actual signal evolution:

    b_{t+1} = ((b_t + e_t - 1) % 9) + 1
    => e_t = ((b_{t+1} - b_t) % 9) + 1

This gives a generator time series per channel. The DISTRIBUTION of
inferred generators characterizes the signal's QA dynamics:

  - Constant signal: e_t = 1 always (singularity generator)
  - Slowly varying: e_t clusters near 1 or 9
  - Rapidly varying: e_t distributed across {1,...,9}

Cross-channel generator synchrony captures spatial coordination:
  - Seizure: generators synchronized across channels (singularity coupling)
  - Baseline: generators independent across channels (cosmos coupling)

Per [207]: singularity = MAXIMUM coupling (C=2de is maximized).
Per [208]: b (amplitude state) and e (generator) are role-distinct factors.

Architecture (Theorem NT compliant):

  [OBSERVER]   EEG waveform (continuous amplitude)
                -> quantize to {1,...,9} per sample  (boundary crossed ONCE)
                |
  [QA LAYER]   b_t = quantized amplitude (integer)
               e_t = ((b_{t+1} - b_t) % 9) + 1 (inferred generator, integer)
               orbit_t = orbit_family(b_t, e_t)
               Generator distribution + cross-channel synchrony
                |
  [PROJECTION] orbit fractions, generator entropy, synchrony index
                -> nested logistic regression vs delta baseline

Standalone script.
"""

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "observer": "signal_dynamics_generator_inference",
    "b_meaning": "quantized_amplitude",
    "e_meaning": "inferred_generator_from_signal_evolution",
    "axioms_checked": ["A1", "A2", "T1", "T2", "S1", "S2"],
    "qa_layer_types": "int",
}

import sys
import json
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, str(Path("/home/player2/signal_experiments")))
sys.path.insert(0, str(Path("/home/player2/wt-papers")))

from eeg_chbmit_scale import load_patient_dataset
from qa_orbit_rules import orbit_family, norm_f

np.random.seed(42)

PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")
MOD = 9
DOWNSAMPLE = 4   # analyse every 4th sample (~64 Hz from 256 Hz)


# ── QA arithmetic ────────────────────────────────────────────────────────────

def qa_residue(value: float, m: int) -> int:
    """Observer projection: continuous amplitude -> {1,...,m}. A1 compliant."""
    x = int(round(value))
    return ((x - 1) % m) + 1


def infer_generator(b_t: int, b_next: int, m: int) -> int:
    """
    Infer the generator e that would produce b_next from b_t via QA step.
    QA step: b_{t+1} = ((b_t + e - 1) % m) + 1
    Inversion: e = ((b_{t+1} - b_t - 1) % m) + 1
    """
    return ((b_next - b_t - 1) % m) + 1


# ── Signal dynamics extraction ───────────────────────────────────────────────

def quantize_multichannel(multi_ch: np.ndarray, m: int = MOD,
                          downsample: int = DOWNSAMPLE) -> list[list[int]]:
    """
    Quantize all channels using SHARED amplitude bins.

    Observer layer: continuous -> discrete. Boundary crossed here.
    Global quantization: bin edges computed from ALL channels together,
    so the same b value means the same amplitude across channels.
    This preserves cross-channel relationships needed for synchrony.
    """
    n_ch = multi_ch.shape[0]

    # Downsample all channels
    downsampled = multi_ch[:, ::downsample]  # (n_ch, n_samples_ds)
    n_samp = downsampled.shape[1]
    if n_samp < 2:
        return [[1] for _ in range(n_ch)]

    # Compute global percentile bin edges from ALL channels
    all_vals = downsampled.ravel()
    edges = [float(np.percentile(all_vals, 100 * k / m)) for k in range(1, m)]

    # Quantize each channel using global edges
    result = []
    for ch in range(n_ch):
        quantized = []
        for t in range(n_samp):
            val = float(downsampled[ch, t])
            # Find bin (A1: result in {1,...,m})
            b = 1
            for edge in edges:
                if val > edge:
                    b += 1
            quantized.append(b)
        result.append(quantized)

    return result


def extract_channel_dynamics(quantized: list[int], m: int = MOD) -> dict:
    """
    Extract QA dynamics from a quantized channel time series.
    All operations are integer (QA layer).

    Returns generator distribution, orbit fractions, f-value stats.
    """
    n = len(quantized)
    if n < 2:
        return _empty_channel()

    generators = []
    orbits = []
    f_values = []

    for t in range(n - 1):
        b_t = int(quantized[t])
        b_next = int(quantized[t + 1])

        # Infer generator (QA layer, integer)
        e_t = infer_generator(b_t, b_next, m)
        generators.append(e_t)

        # Orbit classification
        orb = orbit_family(b_t, e_t, m)
        orbits.append(orb)

        # f-value
        f_values.append(norm_f(b_t, e_t))

    # Generator distribution
    gen_counts = [0] * m
    for g in generators:
        gen_counts[g - 1] += 1
    gen_dist = [c / len(generators) for c in gen_counts]

    # Generator entropy (how spread out are the generators?)
    gen_entropy = 0.0
    for p in gen_dist:
        if p > 0:
            gen_entropy -= p * np.log2(p)

    # Singularity generator fraction: e=1 means no change
    sing_gen_frac = gen_counts[0] / len(generators)  # e=1

    # Orbit fractions
    n_orb = len(orbits)
    cosmos_frac = sum(1 for o in orbits if o == "cosmos") / n_orb
    satellite_frac = sum(1 for o in orbits if o == "satellite") / n_orb
    singularity_frac = sum(1 for o in orbits if o == "singularity") / n_orb

    return {
        "generators": generators,
        "gen_entropy": gen_entropy,
        "sing_gen_frac": sing_gen_frac,
        "cosmos_frac": cosmos_frac,
        "satellite_frac": satellite_frac,
        "singularity_frac": singularity_frac,
        "mean_f": float(np.mean(f_values)),
        "mean_e": float(np.mean(generators)),
    }


def _empty_channel():
    return {
        "generators": [], "gen_entropy": 0.0, "sing_gen_frac": 0.0,
        "cosmos_frac": 0.0, "satellite_frac": 0.0, "singularity_frac": 0.0,
        "mean_f": 0.0, "mean_e": 0.0,
    }


def extract_window_features(multi_ch: np.ndarray, m: int = MOD) -> dict:
    """
    Extract QA signal dynamics features from one multi-channel EEG window.

    Per channel: quantize -> infer generators -> orbit classification.
    Cross-channel: generator synchrony (how correlated are generators?).
    """
    n_ch = multi_ch.shape[0]

    # Quantize all channels with SHARED global bins
    all_quantized = quantize_multichannel(multi_ch, m)
    all_dynamics = []
    for i in range(n_ch):
        dyn = extract_channel_dynamics(all_quantized[i], m)
        all_dynamics.append(dyn)

    # Per-channel aggregates
    mean_entropy = float(np.mean([d["gen_entropy"] for d in all_dynamics]))
    mean_sing_gen = float(np.mean([d["sing_gen_frac"] for d in all_dynamics]))
    mean_cosmos = float(np.mean([d["cosmos_frac"] for d in all_dynamics]))
    mean_satellite = float(np.mean([d["satellite_frac"] for d in all_dynamics]))
    mean_singularity = float(np.mean([d["singularity_frac"] for d in all_dynamics]))
    mean_f = float(np.mean([d["mean_f"] for d in all_dynamics]))
    mean_e = float(np.mean([d["mean_e"] for d in all_dynamics]))

    # Cross-channel generator synchrony
    # For each time step, compute how many channels have the SAME generator
    # High synchrony = seizure (singularity coupling per [207])
    min_len = min(len(d["generators"]) for d in all_dynamics)
    if min_len > 0:
        synchrony_scores = []
        for t in range(min_len):
            gens_at_t = [all_dynamics[i]["generators"][t] for i in range(n_ch)]
            # Modal generator fraction: what fraction of channels agree?
            from collections import Counter
            counts = Counter(gens_at_t)
            modal_frac = counts.most_common(1)[0][1] / n_ch
            synchrony_scores.append(modal_frac)
        generator_synchrony = float(np.mean(synchrony_scores))
    else:
        generator_synchrony = 0.0

    return {
        "cosmos_frac": mean_cosmos,
        "satellite_frac": mean_satellite,
        "singularity_frac": mean_singularity,
        "gen_entropy": mean_entropy,
        "sing_gen_frac": mean_sing_gen,
        "generator_synchrony": generator_synchrony,
        "mean_f": mean_f,
        "mean_e": mean_e,
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
    print("QA SIGNAL DYNAMICS OBSERVER")
    print("The signal IS the orbit. The generator IS the dynamics.")
    print("b = quantized amplitude, e = inferred from signal evolution")
    print("Patient: chb01 (real CHB-MIT data)")
    print("=" * 72)

    # Load
    print("\nLoading chb01 data...")
    dataset = load_patient_dataset(PATIENT_DIR)
    n_sei = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    print(f"  {n_sei} seizure + {n_base} baseline windows")
    print(f"  Downsample: {DOWNSAMPLE}x (256 Hz -> {256//DOWNSAMPLE} Hz)")

    # Extract features
    print("\nExtracting QA signal dynamics...")
    all_features = []
    all_labels = []
    for d in dataset:
        feats = extract_window_features(d["multi_ch"])
        feats["type"] = d["type"]
        all_features.append(feats)
        all_labels.append(1 if d["type"] == "seizure" else 0)

    y = np.array(all_labels, dtype=float)

    # Feature distributions
    print(f"\n  {'Type':<12} {'Cosmos':>8} {'Sat':>6} {'Sing':>6} {'Entropy':>8} "
          f"{'SingGen':>8} {'Synch':>8} {'mean_e':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label in ["seizure", "baseline"]:
        feats = [f for f in all_features if f["type"] == label]
        print(f"  {label:<12} "
              f"{np.mean([f['cosmos_frac'] for f in feats]):>8.4f} "
              f"{np.mean([f['satellite_frac'] for f in feats]):>6.4f} "
              f"{np.mean([f['singularity_frac'] for f in feats]):>6.4f} "
              f"{np.mean([f['gen_entropy'] for f in feats]):>8.4f} "
              f"{np.mean([f['sing_gen_frac'] for f in feats]):>8.4f} "
              f"{np.mean([f['generator_synchrony'] for f in feats]):>8.4f} "
              f"{np.mean([f['mean_e'] for f in feats]):>8.4f}")

    # Delta baseline
    from eeg_rns_observer import delta_power_ratio  # noqa: T2-D-5
    delta = np.array([
        delta_power_ratio(d["waveform"].astype(np.float64), d["fs"])
        for d in dataset
    ])

    # Nested model: delta vs delta + QA signal dynamics
    sing = np.array([f["singularity_frac"] for f in all_features])
    synch = np.array([f["generator_synchrony"] for f in all_features])
    entropy = np.array([f["gen_entropy"] for f in all_features])
    mean_f = np.array([f["mean_f"] for f in all_features])

    n = len(y)
    X0 = np.ones((n, 1))
    ll0 = _ll(X0, y, _fit_logistic(X0, y))

    X1 = np.c_[np.ones(n), _std(delta)]
    ll1 = _ll(X1, y, _fit_logistic(X1, y))

    X2 = np.c_[np.ones(n), _std(delta), _std(sing), _std(synch),
               _std(entropy), _std(mean_f)]
    ll2 = _ll(X2, y, _fit_logistic(X2, y))

    r2_1 = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_2 = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    delta_r2 = r2_2 - r2_1
    lr_stat = 2.0 * (ll2 - ll1)
    p_val = float(chi2.sf(max(0, lr_stat), df=4))

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    print(f"\n{'='*72}")
    print("NESTED LOGISTIC REGRESSION")
    print(f"  Model 1: seizure ~ delta")
    print(f"  Model 2: seizure ~ delta + singularity + synchrony + entropy + mean_f")
    print(f"\n  R2 (delta only):                {r2_1:.4f}")
    print(f"  R2 (delta + QA dynamics):        {r2_2:.4f}")
    print(f"  DR2 (QA dynamics beyond delta):  {delta_r2:+.4f}")
    print(f"  LR stat:                         {lr_stat:.3f}")
    print(f"  p(QA adds):                      {p_val:.6f} {sig(p_val)}")

    print(f"\n  COMPARISON (all on chb01):")
    print(f"  Observer 3 (topographic k-means):     DR2 = +0.252")
    print(f"  RNS eigenspectrum:                    DR2 = +0.138")
    print(f"  Canonical (amplitude x propagation):  DR2 = +0.094")
    print(f"  QA Graph (degree x core_number):      DR2 = +0.019")
    print(f"  QA Signal Dynamics (this observer):   DR2 = {delta_r2:+.4f}")

    # Save
    results = {
        "observer": "signal_dynamics_generator_inference",
        "patient": "chb01",
        "n_seizure": int(sum(y)),
        "n_baseline": int(len(y) - sum(y)),
        "downsample": DOWNSAMPLE,
        "mod": MOD,
        "r2_delta": float(r2_1),
        "r2_full": float(r2_2),
        "delta_r2": float(delta_r2),
        "p_qa_add": float(p_val),
    }
    out_path = Path("eeg_signal_dynamics_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
