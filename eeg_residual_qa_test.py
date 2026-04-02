#!/usr/bin/env python3
QA_COMPLIANCE = "observer=eeg_residual, state_alphabet=mod24_microstate"
"""
eeg_residual_qa_test.py — Nested Model Test: Does QA add to Delta?

Research question: After controlling for delta power ratio, does QA orbit structure
explain additional variance in seizure vs baseline classification?

Model 1: seizure ~ delta_ratio
Model 2: seizure ~ delta_ratio + QA_singularity + QA_cosmos

Test: McFadden R², likelihood ratio test (chi-squared, df=2)

This is the fork-in-the-road test:
  - If QA adds nothing → QA is a structural interpretation layer (orbit collapse = seizure state)
  - If QA adds signal  → QA is a genuine empirical discriminant over classical features

All computation is in projection layer — floats permitted.
No QA integer layer here (this is a meta-analysis of already-computed QA projections).
"""

import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import expit  # sigmoid

# Reuse loaders and QA pipeline
from eeg_orbit_classifier import (
    load_chbmit_dataset, DEFAULT_DATA_DIR, classify_eeg_segment,
    compute_orbit_sequence, orbit_statistics, _read_edf_channel
)
from eeg_autocorrelation_baseline import delta_power_ratio


# ── Minimal logistic regression (no sklearn dependency) ───────────────────────

def _logistic_ll(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Log-likelihood of logistic model."""
    logits = X @ beta
    # clip for numerical stability
    logits = np.clip(logits, -30, 30)
    probs = expit(logits)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def _fit_logistic(X: np.ndarray, y: np.ndarray,
                  lr: float = 0.1, n_iter: int = 2000,
                  l2: float = 1e-4) -> np.ndarray:
    """Fit logistic regression by gradient descent. Returns beta."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(n_iter):
        logits = np.clip(X @ beta, -30, 30)
        probs = expit(logits)
        grad = X.T @ (probs - y) / n + l2 * beta
        beta -= lr * grad
    return beta


def mcfadden_r2(ll_model: float, ll_null: float) -> float:
    """McFadden's pseudo R²: 1 - LL_model / LL_null."""
    if ll_null == 0:
        return 0.0
    return float(1.0 - ll_model / ll_null)


def likelihood_ratio_test(ll_restricted: float, ll_full: float, df: int) -> tuple[float, float]:
    """LR statistic = 2*(LL_full - LL_restricted), chi-squared df."""
    lr_stat = 2.0 * (ll_full - ll_restricted)
    p_val = float(stats.chi2.sf(lr_stat, df))
    return float(lr_stat), p_val


# ── Per-segment feature extraction ────────────────────────────────────────────

def extract_per_segment(dataset: list[dict], fs: int) -> list[dict]:
    """Extract both classical and QA features for each segment."""
    results = []
    for seg in dataset:
        sig = seg["waveform"].astype(np.float64)

        # Classical features
        delta = delta_power_ratio(sig, fs)

        # QA features (observer → QA layer → projection)
        # classify_eeg_segment expects 1D array
        waveform_1d = seg["waveform"].ravel()
        microstates = classify_eeg_segment(waveform_1d, fs)
        orbit_seq = compute_orbit_sequence(microstates)
        orbit_stats = orbit_statistics(orbit_seq)

        results.append({
            "type":      seg["type"],
            "source":    seg.get("source", ""),
            "delta":     delta,
            "sing_frac": orbit_stats["singularity_frac"],
            "sat_frac":  orbit_stats["satellite_frac"],
            "cos_frac":  orbit_stats["cosmos_frac"],
            "label":     1 if seg["type"] == "seizure" else 0,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EEG Residual QA Test — Nested Logistic Regression")
    print("Does QA orbit add to delta ratio?")
    print("=" * 70)
    print()

    print("Loading CHB-MIT chb01 dataset...")
    dataset = load_chbmit_dataset(DEFAULT_DATA_DIR)
    if not dataset:
        print("ERROR: no data loaded")
        return

    n_sei  = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    print(f"  Total: {len(dataset)} segments ({n_sei} seizure, {n_base} baseline)")

    _, fs = _read_edf_channel(DEFAULT_DATA_DIR / "chb01_03.edf", 0)
    print(f"  Sample rate: {fs} Hz")
    print()

    print("Extracting per-segment features...")
    records = extract_per_segment(dataset, fs)
    print(f"  Done. {len(records)} records extracted.")
    print()

    # Build arrays
    y     = np.array([r["label"]     for r in records], dtype=float)
    delta = np.array([r["delta"]     for r in records], dtype=float)
    sing  = np.array([r["sing_frac"] for r in records], dtype=float)
    cos   = np.array([r["cos_frac"]  for r in records], dtype=float)
    sat   = np.array([r["sat_frac"]  for r in records], dtype=float)

    # Standardize features
    def _std(x):
        mu, sd = x.mean(), x.std()
        return (x - mu) / (sd + 1e-9)

    delta_s = _std(delta)
    sing_s  = _std(sing)
    cos_s   = _std(cos)
    sat_s   = _std(sat)

    # ── Model 0: null (intercept only) ────────────────────────────────────────
    X0 = np.ones((len(y), 1))
    beta0 = _fit_logistic(X0, y)
    ll0 = _logistic_ll(X0, y, beta0)
    print(f"  Null model LL:           {ll0:.4f}")

    # ── Model 1: delta only ────────────────────────────────────────────────────
    X1 = np.column_stack([np.ones(len(y)), delta_s])
    beta1 = _fit_logistic(X1, y)
    ll1 = _logistic_ll(X1, y, beta1)
    r2_1 = mcfadden_r2(ll1, ll0)
    print(f"  Model 1 (delta) LL:      {ll1:.4f}  McFadden R²={r2_1:.4f}")

    # ── Model 2: delta + QA (singularity + cosmos) ────────────────────────────
    # Note: singularity + satellite + cosmos = 1, so only 2 QA features are independent.
    # We use singularity + cosmos (satellite is the least significant from earlier t-tests).
    X2 = np.column_stack([np.ones(len(y)), delta_s, sing_s, cos_s])
    beta2 = _fit_logistic(X2, y)
    ll2 = _logistic_ll(X2, y, beta2)
    r2_2 = mcfadden_r2(ll2, ll0)
    print(f"  Model 2 (delta+QA) LL:   {ll2:.4f}  McFadden R²={r2_2:.4f}")

    # ── Model 3: QA only (no delta) ───────────────────────────────────────────
    X3 = np.column_stack([np.ones(len(y)), sing_s, cos_s])
    beta3 = _fit_logistic(X3, y)
    ll3 = _logistic_ll(X3, y, beta3)
    r2_3 = mcfadden_r2(ll3, ll0)
    print(f"  Model 3 (QA only) LL:    {ll3:.4f}  McFadden R²={r2_3:.4f}")
    print()

    # ── LR tests ──────────────────────────────────────────────────────────────
    print("Likelihood Ratio Tests:")

    # H0: QA adds nothing to delta (Model 2 vs Model 1, df=2)
    lr12, p12 = likelihood_ratio_test(ll1, ll2, df=2)
    sig12 = "***" if p12 < 0.001 else ("**" if p12 < 0.01 else ("*" if p12 < 0.05 else "ns"))
    print(f"  Model 2 vs Model 1 (QA adds to delta):")
    print(f"    LR={lr12:.3f}  df=2  p={p12:.4f}  {sig12}")

    # H0: delta adds nothing to QA (Model 2 vs Model 3, df=1)
    lr32, p32 = likelihood_ratio_test(ll3, ll2, df=1)
    sig32 = "***" if p32 < 0.001 else ("**" if p32 < 0.01 else ("*" if p32 < 0.05 else "ns"))
    print(f"  Model 2 vs Model 3 (delta adds to QA):")
    print(f"    LR={lr32:.3f}  df=1  p={p32:.4f}  {sig32}")

    # H0: delta adds nothing to null (Model 1 vs null, df=1)
    lr01, p01 = likelihood_ratio_test(ll0, ll1, df=1)
    sig01 = "***" if p01 < 0.001 else ("**" if p01 < 0.01 else ("*" if p01 < 0.05 else "ns"))
    print(f"  Model 1 vs Null (delta alone):")
    print(f"    LR={lr01:.3f}  df=1  p={p01:.4f}  {sig01}")

    # H0: QA adds nothing to null (Model 3 vs null, df=2)
    lr03, p03 = likelihood_ratio_test(ll0, ll3, df=2)
    sig03 = "***" if p03 < 0.001 else ("**" if p03 < 0.01 else ("*" if p03 < 0.05 else "ns"))
    print(f"  Model 3 vs Null (QA alone):")
    print(f"    LR={lr03:.3f}  df=2  p={p03:.4f}  {sig03}")

    print()

    # ── Coefficient inspection ─────────────────────────────────────────────────
    print("Model 2 coefficients (standardized):")
    print(f"  intercept:      {beta2[0]:+.4f}")
    print(f"  delta_ratio:    {beta2[1]:+.4f}")
    print(f"  QA_singularity: {beta2[2]:+.4f}")
    print(f"  QA_cosmos:      {beta2[3]:+.4f}")
    print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("VERDICT")
    print()

    incremental_r2 = r2_2 - r2_1
    print(f"  Delta alone:         McFadden R² = {r2_1:.4f}")
    print(f"  Delta + QA:          McFadden R² = {r2_2:.4f}")
    print(f"  QA increment:        ΔR²         = {incremental_r2:+.4f}")
    print(f"  QA-alone:            McFadden R² = {r2_3:.4f}")
    print()

    if p12 < 0.05:
        print("  QA ADDS to delta (p<0.05) — QA is a GENUINE EMPIRICAL DISCRIMINANT.")
        print("  Incremental variance explained beyond delta supports Track D empirical claim.")
    else:
        print("  QA does NOT add to delta (ns) — QA is a STRUCTURAL INTERPRETATION LAYER.")
        print("  The orbit collapse maps onto seizure state but does not improve prediction")
        print("  over delta power ratio alone.")
        print()
        print("  This is still a valid result:")
        print("  QA gives a formal reachability interpretation: seizure = orbit fixed point")
        print("  (not: QA outperforms classical EEG features)")

    print()
    print("  Appropriate paper claim:")
    if p12 < 0.05:
        print("  'QA orbit singularity contributes independent discriminative information")
        print("   beyond spectral delta power (LR test p<0.05, ΔR²={:.4f})'".format(incremental_r2))
    else:
        print("  'QA orbit structure provides a formal interpretation of seizure as orbit")
        print("   collapse (singularity state): the same information encoded by delta power")
        print("   ratio is represented geometrically as convergence to a QA fixed point.'")


if __name__ == "__main__":
    main()
