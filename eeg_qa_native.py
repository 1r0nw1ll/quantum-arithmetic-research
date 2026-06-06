#!/usr/bin/env python3
"""QA-native EEG seizure analysis — no KMeans, no hardcoded state labels.

Theorem NT-compliant:
  Input gate (once):  raw EEG amplitude → quantile bins → {1,...,9}
  QA layer:           consecutive pairs (b_t, e_t) → orbit_fam → fractions
  Observer output:    fractions → z-score → logistic regression

Hypothesis: seizure dynamics shift the consecutive-sample (b,e) distribution
toward Singularity (9,9) or Satellite (3|b AND 3|e) relative to interictal.

Feature derivation (no free parameters after quantizer is fit):
  1. Fit 8 quantile boundaries from training-interictal amplitude values.
  2. For each window: quantize per-channel amplitudes → per-channel (b,e)
     pairs from consecutive samples → orbit fractions averaged across channels.
  3. Z-score using training-interictal mean/std (same as combined-2).
"""
import json, sys, warnings
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from eeg_feat_shared import (
    ready_patients, extract_windows, delta_power_scalar,
    fishers, eval_delta_r2, SEED, TEST_FRAC,
)

M = 9

# ── Orbit table (precomputed, {1,...,9}²) ──────────────────────────────────────

def _orbit_period(b0, e0):
    b, e = b0, e0
    for k in range(1, 300):
        b, e = e, (b + e - 1) % M + 1
        if b == b0 and e == e0:
            return k
    return -1

_ORBIT_FAM = {}
for _b in range(1, M + 1):
    for _e in range(1, M + 1):
        _p = _orbit_period(_b, _e)
        _ORBIT_FAM[(_b, _e)] = "Singularity" if _p == 1 else ("Satellite" if _p == 8 else "Cosmos")

# Vectorized: Satellite ↔ 3|b AND 3|e AND NOT (b==9 AND e==9)
# Singularity ↔ b==9 AND e==9
# (precomputed table matches this for m=9)


# ── Quantizer ──────────────────────────────────────────────────────────────────

def fit_quantizer(windows):
    """Fit M-1 quantile boundaries from all interictal training samples."""
    vals = np.concatenate([w["sig"].ravel() for w in windows]).astype(np.float64)
    boundaries = np.percentile(vals, np.linspace(0, 100, M + 1)[1:-1])
    return boundaries


def quantize(sig, boundaries):
    """Map (n_ch, n_t) signal to integer array in {1,...,M}."""
    q = np.searchsorted(boundaries, sig.astype(np.float64), side="right") + 1
    return np.clip(q, 1, M).astype(np.int32)


# ── QA orbit fractions ─────────────────────────────────────────────────────────

def orbit_fracs(sig, boundaries):
    """
    Per-channel consecutive-pair orbit fractions, averaged across channels.

    For each channel c:
      b_t = quantized(sig[c, t])
      e_t = quantized(sig[c, t+1])
      orbit = _ORBIT_FAM[(b_t, e_t)]

    Returns (sing_frac, sat_frac, cos_frac) averaged over all channels × pairs.
    """
    q = quantize(sig, boundaries)           # (n_ch, n_t)
    b = q[:, :-1]                           # (n_ch, n_t-1)
    e = q[:, 1:]                            # (n_ch, n_t-1)

    # Vectorized orbit classification
    b_div3 = (b % 3 == 0)
    e_div3 = (e % 3 == 0)
    is_sing = (b == M) & (e == M)
    is_sat  = b_div3 & e_div3 & ~is_sing
    is_cos  = ~(b_div3 & e_div3)

    n = b.size
    return float(is_sing.sum()) / n, float(is_sat.sum()) / n, float(is_cos.sum()) / n


# ── Per-patient analysis ───────────────────────────────────────────────────────

MIN_ICTAL = 12

def analyse(pdir):
    pid = pdir.name
    windows = extract_windows(pdir)
    if not windows or sum(w["label"] == 1 for w in windows) < MIN_ICTAL:
        return None

    labels  = np.array([w["label"] for w in windows])
    idx     = np.arange(len(labels))
    tr_idx, _ = train_test_split(idx, test_size=TEST_FRAC,
                                 stratify=labels, random_state=SEED)
    inter_train = [windows[i] for i in tr_idx if labels[i] == 0]

    # Fit quantizer from training-interictal only (input gate)
    boundaries = fit_quantizer(inter_train)

    # Extract orbit fractions for all windows
    sing = np.array([orbit_fracs(w["sig"], boundaries)[0] for w in windows])
    sat  = np.array([orbit_fracs(w["sig"], boundaries)[1] for w in windows])

    # Z-score from training-interictal baseline
    def zscore(arr):
        mu  = arr[tr_idx][labels[tr_idx] == 0].mean()
        std = arr[tr_idx][labels[tr_idx] == 0].std()
        return (arr - mu) / std if std > 1e-9 else np.zeros_like(arr)

    sing_z = zscore(sing)
    sat_z  = zscore(sat)

    delta = np.array([delta_power_scalar(w["sig"], w["fs"]) for w in windows])
    feats = np.column_stack([delta, sing_z, sat_z])

    dr2, p = eval_delta_r2(feats, labels)

    ictal = labels == 1
    print(f"  [{pid}]  ΔR²={dr2:+.3f}  p={p:.4f}  "
          f"sing_z={sing_z[ictal].mean():+.2f}  sat_z={sat_z[ictal].mean():+.2f}")
    return {"patient": pid, "dr2": round(dr2, 5), "p": round(p, 6),
            "sing_ictal_z": round(float(sing_z[ictal].mean()), 3),
            "sat_ictal_z":  round(float(sat_z[ictal].mean()), 3)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("QA-native EEG: consecutive-sample quantization → orbit fractions")
    print("="*60)
    results = [r for r in (analyse(pd) for pd in ready_patients()) if r]
    if not results:
        print("No patients passed inclusion.")
        return

    n    = len(results)
    dr2s = [r["dr2"] for r in results]
    ps   = [r["p"]   for r in results]
    chi2_stat, fisher_p = fishers(ps)
    n_pos = sum(d > 0 for d in dr2s)
    mean_dr2 = np.mean(dr2s)
    se_dr2   = np.std(dr2s, ddof=1) / n**0.5

    print(f"\nPatients: {n}   Positive ΔR²: {n_pos}/{n}")
    print(f"Mean ΔR²: {mean_dr2:+.4f}  SE: {se_dr2:.4f}")
    print(f"Fisher χ²: {chi2_stat:.1f}  p = {fisher_p:.3e}  (df={2*n})")

    out = Path(__file__).parent / "eeg_qa_native_results.json"
    out.write_text(json.dumps({
        "n": n, "mean_dr2": round(float(mean_dr2), 5),
        "se_dr2": round(float(se_dr2), 5),
        "fisher_chi2": round(float(chi2_stat), 2),
        "fisher_p": float(fisher_p),
        "n_positive": n_pos,
        "patients": results,
    }, indent=2))
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
