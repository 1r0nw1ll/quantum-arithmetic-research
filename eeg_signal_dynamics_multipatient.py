#!/usr/bin/env python3
"""
eeg_signal_dynamics_multipatient.py — [209] Multi-Patient EEG Validation

Runs the QA signal generator inference observer across all available
CHB-MIT patients. Reports per-patient ΔR² and Fisher's combined p-value.

Same method as eeg_signal_dynamics_observer.py:
  b = quantized amplitude (global bins across channels)
  e = inferred generator: ((b_{t+1} - b_t - 1) % 9) + 1
  Features: singularity_frac, generator_synchrony, entropy, mean_f
"""

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "observer": "signal_generator_inference_cert_209",
    "axioms_checked": ["A1", "A2", "S1", "S2", "T1", "T2"],
}

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, str(Path("/home/player2/signal_experiments")))
sys.path.insert(0, str(Path("/home/player2/wt-papers")))

from eeg_chbmit_scale import load_patient_dataset
from qa_orbit_rules import orbit_family, norm_f

np.random.seed(42)

CHBMIT_ROOT = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit")
MOD = 9
DOWNSAMPLE = 4


# ── Core [209] functions ─────────────────────────────────────────────────────

def infer_generator(b_t: int, b_next: int, m: int = MOD) -> int:
    return ((b_next - b_t - 1) % m) + 1


def quantize_multichannel(multi_ch: np.ndarray, m: int = MOD,
                          downsample: int = DOWNSAMPLE) -> list[list[int]]:
    n_ch = multi_ch.shape[0]
    downsampled = multi_ch[:, ::downsample]
    n_samp = downsampled.shape[1]
    if n_samp < 2:
        return [[1] for _ in range(n_ch)]
    all_vals = downsampled.ravel()
    edges = [float(np.percentile(all_vals, 100 * k / m)) for k in range(1, m)]
    result = []
    for ch in range(n_ch):
        quantized = []
        for t in range(n_samp):
            val = float(downsampled[ch, t])
            b = 1
            for edge in edges:
                if val > edge:
                    b += 1
            quantized.append(b)
        result.append(quantized)
    return result


def extract_window_features(multi_ch: np.ndarray, m: int = MOD) -> dict:
    n_ch = multi_ch.shape[0]
    all_quantized = quantize_multichannel(multi_ch, m)

    # Per-channel stats
    all_entropy = []
    all_sing = []
    all_f = []
    all_generators = []

    for i in range(n_ch):
        q = all_quantized[i]
        n = len(q)
        if n < 2:
            continue
        generators = [infer_generator(q[t], q[t + 1], m) for t in range(n - 1)]
        all_generators.append(generators)

        counts = [0] * m
        for g in generators:
            counts[g - 1] += 1
        total = len(generators)
        dist = [c / total for c in counts]
        entropy = -sum(p * np.log2(p) for p in dist if p > 0)
        all_entropy.append(entropy)

        # Orbit classification
        orbits = [orbit_family(((q[t] - 1) % m) + 1,
                               ((generators[t] - 1) % m) + 1, m)
                  for t in range(len(generators))]
        sing_frac = sum(1 for o in orbits if o == "singularity") / len(orbits)
        all_sing.append(sing_frac)

        f_vals = [norm_f(((q[t] - 1) % m) + 1, ((generators[t] - 1) % m) + 1)
                  for t in range(len(generators))]
        all_f.append(float(np.mean(f_vals)))

    # Cross-channel synchrony
    min_len = min(len(g) for g in all_generators) if all_generators else 0
    if min_len > 0:
        scores = []
        for t in range(min_len):
            gens_at_t = [all_generators[i][t] for i in range(len(all_generators))]
            modal_frac = Counter(gens_at_t).most_common(1)[0][1] / len(all_generators)
            scores.append(modal_frac)
        synchrony = float(np.mean(scores))
    else:
        synchrony = 0.0

    return {
        "singularity_frac": float(np.mean(all_sing)) if all_sing else 0.0,
        "gen_entropy": float(np.mean(all_entropy)) if all_entropy else 0.0,
        "generator_synchrony": synchrony,
        "mean_f": float(np.mean(all_f)) if all_f else 0.0,
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


def delta_power_ratio(waveform, fs):
    from numpy.fft import rfft, rfftfreq  # noqa: T2-D-5
    n = len(waveform)
    freqs = rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(rfft(waveform))
    total = float(np.sum(spectrum * spectrum))
    if total < 1e-12:
        return 0.0
    delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
    return float(np.sum(spectrum[delta_mask] * spectrum[delta_mask])) / total


# ── Per-patient analysis ─────────────────────────────────────────────────────

def analyse_patient(patient_id: str, patient_dir: Path) -> dict:
    dataset = load_patient_dataset(patient_dir)
    if not dataset:
        return None

    # Filter to modal channel count
    from collections import Counter as Ctr
    ch_counts = Ctr(d["multi_ch"].shape[0] for d in dataset)
    modal_ch = ch_counts.most_common(1)[0][0]
    dataset = [d for d in dataset if d["multi_ch"].shape[0] == modal_ch]

    n_sei = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    if n_sei < 2 or n_base < 2:
        return None

    y = np.array([1.0 if d["type"] == "seizure" else 0.0 for d in dataset])
    fs = dataset[0]["fs"]

    # Delta baseline
    delta = np.array([delta_power_ratio(d["waveform"].astype(np.float64), fs)
                      for d in dataset])

    # Signal dynamics features
    sd = [extract_window_features(d["multi_ch"]) for d in dataset]
    sing = np.array([f["singularity_frac"] for f in sd])
    synch = np.array([f["generator_synchrony"] for f in sd])
    entropy = np.array([f["gen_entropy"] for f in sd])
    mean_f = np.array([f["mean_f"] for f in sd])

    n = len(y)
    X0 = np.ones((n, 1))
    ll0 = _ll(X0, y, _fit_logistic(X0, y))

    X1 = np.c_[np.ones(n), _std(delta)]
    ll1 = _ll(X1, y, _fit_logistic(X1, y))

    X2 = np.c_[np.ones(n), _std(delta), _std(sing), _std(synch),
               _std(entropy), _std(mean_f)]
    ll2 = _ll(X2, y, _fit_logistic(X2, y))

    r2_delta = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_full = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    delta_r2 = r2_full - r2_delta
    lr_stat = 2.0 * (ll2 - ll1)
    p_val = float(chi2.sf(max(0, lr_stat), df=4))

    # Synchrony means
    synch_sei = float(synch[y == 1].mean()) if n_sei > 0 else 0.0
    synch_base = float(synch[y == 0].mean()) if n_base > 0 else 0.0

    return {
        "patient": patient_id,
        "n_sei": n_sei,
        "n_base": n_base,
        "r2_delta": float(r2_delta),
        "r2_full": float(r2_full),
        "delta_r2": float(delta_r2),
        "p_qa_add": float(p_val),
        "synch_sei": synch_sei,
        "synch_base": synch_base,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("[209] MULTI-PATIENT EEG VALIDATION — Signal Generator Inference")
    print("=" * 72)

    patient_dirs = sorted(CHBMIT_ROOT.glob("chb*/"))
    print(f"\nPatients found: {[p.name for p in patient_dirs]}")

    results = []
    for pdir in patient_dirs:
        pid = pdir.name
        print(f"\n  [{pid}] Processing...")
        try:
            res = analyse_patient(pid, pdir)
        except Exception as e:
            print(f"  [{pid}] ERROR: {e}")
            continue
        if res is None:
            print(f"  [{pid}] Skipped (insufficient data)")
            continue
        results.append(res)
        sig = "***" if res["p_qa_add"] < 0.001 else "**" if res["p_qa_add"] < 0.01 else "*" if res["p_qa_add"] < 0.05 else "ns"
        print(f"  [{pid}] n={res['n_sei']}+{res['n_base']}, "
              f"ΔR²={res['delta_r2']:+.4f}, p={res['p_qa_add']:.4f} {sig}, "
              f"synch: sei={res['synch_sei']:.3f} base={res['synch_base']:.3f}")

    if not results:
        print("\nNo usable patients.")
        return

    # Per-patient table
    print(f"\n{'='*72}")
    print("PER-PATIENT RESULTS")
    print(f"  {'Patient':<8} {'N_sei':>5} {'N_base':>6} {'R²(δ)':>7} {'ΔR²':>8} {'p(QA+)':>8} {'Synch_S':>8} {'Synch_B':>8}")
    print(f"  {'-'*8} {'-'*5} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        sig = "***" if r["p_qa_add"] < 0.001 else "**" if r["p_qa_add"] < 0.01 else "*" if r["p_qa_add"] < 0.05 else "ns"
        print(f"  {r['patient']:<8} {r['n_sei']:>5} {r['n_base']:>6} "
              f"{r['r2_delta']:>7.4f} {r['delta_r2']:>+8.4f} {r['p_qa_add']:>6.4f}{sig:>2} "
              f"{r['synch_sei']:>8.4f} {r['synch_base']:>8.4f}")

    # Aggregates
    print(f"\n{'='*72}")
    print("AGGREGATE ANALYSIS")

    n_pts = len(results)
    mean_dr2 = np.mean([r["delta_r2"] for r in results])
    mean_synch_sei = np.mean([r["synch_sei"] for r in results])
    mean_synch_base = np.mean([r["synch_base"] for r in results])
    n_positive = sum(1 for r in results if r["delta_r2"] > 0)
    n_synch_correct = sum(1 for r in results if r["synch_sei"] > r["synch_base"])

    # Fisher's combined p-value
    p_values = [max(r["p_qa_add"], 1e-15) for r in results]
    fisher_chi2 = -2.0 * sum(np.log(p) for p in p_values)
    fisher_df = 2 * n_pts
    fisher_p = float(chi2.sf(fisher_chi2, fisher_df))

    print(f"  Patients analyzed: {n_pts}")
    print(f"  Mean ΔR²: {mean_dr2:+.4f}")
    print(f"  ΔR² positive: {n_positive}/{n_pts}")
    print(f"  Synch sei > base: {n_synch_correct}/{n_pts}")
    print(f"  Mean synch: seizure={mean_synch_sei:.4f}, baseline={mean_synch_base:.4f}")
    print(f"\n  Fisher's combined p-value:")
    print(f"    χ² = {fisher_chi2:.2f}, df = {fisher_df}, p = {fisher_p:.6f} "
          f"{'***' if fisher_p < 0.001 else '**' if fisher_p < 0.01 else '*' if fisher_p < 0.05 else 'ns'}")

    # Save
    output = {
        "method": "signal_generator_inference_cert_209",
        "mod": MOD,
        "downsample": DOWNSAMPLE,
        "n_patients": n_pts,
        "mean_delta_r2": float(mean_dr2),
        "fisher_chi2": float(fisher_chi2),
        "fisher_p": float(fisher_p),
        "per_patient": results,
    }
    out_path = Path("eeg_signal_dynamics_multipatient_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
