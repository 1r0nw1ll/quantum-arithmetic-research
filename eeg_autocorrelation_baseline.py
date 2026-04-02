#!/usr/bin/env python3
QA_COMPLIANCE = "observer=eeg_spectral, state_alphabet=mod24_microstate"
"""
eeg_autocorrelation_baseline.py — QA Baseline Comparison for EEG Orbit Classifier

Research question: Is the QA orbit separation (singularity p<0.0001) specific to the
QA algebraic structure, or is it a proxy for classical spectral/temporal features?

We test three classical discriminants on the SAME 80-segment CHB-MIT chb01 dataset:

  1. Lag-1 autocorrelation (AC1): measures short-term temporal persistence
  2. Delta-band power ratio: delta/(total) — the dominant ictal feature
  3. Spectral entropy: Shannon entropy of normalized power spectrum

If classical discriminants achieve similar t-statistics to QA orbit singularity (t=-4.77),
the QA result is a proxy. If QA is significantly stronger, the QA structure adds information.

Note: This is a PROJECTION layer script — all computations are float, acting on the
same observer-level features that the EEG classifier uses. No QA layer here.
"""

import numpy as np
from pathlib import Path
from scipy import stats

# Reuse the EDF reader and dataset loader from the orbit classifier
from eeg_orbit_classifier import load_chbmit_dataset, _read_edf_channel, DEFAULT_DATA_DIR


# ── Classical feature extraction ──────────────────────────────────────────────

def lag1_autocorrelation(signal: np.ndarray) -> float:
    """Pearson correlation between signal[:-1] and signal[1:]."""
    x = signal[:-1] - signal[:-1].mean()
    y = signal[1:]  - signal[1:].mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def delta_power_ratio(signal: np.ndarray, fs: int) -> float:
    """Delta (0.5–4 Hz) / total power ratio."""
    f = np.fft.rfftfreq(len(signal), 1.0 / fs)
    psd = np.abs(np.fft.rfft(signal)) ** 2
    delta_mask = (f >= 0.5) & (f <= 4.0)
    total = psd.sum()
    if total == 0:
        return 0.0
    return float(psd[delta_mask].sum() / total)


def spectral_entropy(signal: np.ndarray) -> float:
    """Normalized Shannon entropy of power spectrum."""
    psd = np.abs(np.fft.rfft(signal)) ** 2
    total = psd.sum()
    if total == 0:
        return 0.0
    p = psd / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    max_entropy = np.log(len(p))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


def gamma_power_ratio(signal: np.ndarray, fs: int) -> float:
    """Gamma (30–100 Hz) / total power ratio — primary QA observer discriminant."""
    f = np.fft.rfftfreq(len(signal), 1.0 / fs)
    psd = np.abs(np.fft.rfft(signal)) ** 2
    gamma_mask = (f >= 30.0) & (f <= 100.0)
    total = psd.sum()
    if total == 0:
        return 0.0
    return float(psd[gamma_mask].sum() / total)


# ── Main comparison ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EEG Autocorrelation Baseline — QA Specificity Test")
    print("Comparing classical features vs QA orbit singularity (t=-4.77)")
    print("=" * 70)
    print()

    # Load dataset (same as orbit classifier)
    print("Loading CHB-MIT chb01 dataset...")
    dataset = load_chbmit_dataset(DEFAULT_DATA_DIR)
    if not dataset:
        print("ERROR: no data loaded")
        return

    print(f"  Total: {len(dataset)} segments "
          f"({sum(1 for d in dataset if d['type']=='seizure')} seizure, "
          f"{sum(1 for d in dataset if d['type']=='baseline')} baseline)")
    print()

    # Need sample rate — read from first file
    first_edf = DEFAULT_DATA_DIR / "chb01_03.edf"
    _, fs = _read_edf_channel(first_edf, 0)
    print(f"  Sample rate: {fs} Hz")
    print()

    # Extract features for each segment
    sei_ac1, sei_delta, sei_entropy, sei_gamma = [], [], [], []
    base_ac1, base_delta, base_entropy, base_gamma = [], [], [], []

    for seg in dataset:
        sig = seg["waveform"].astype(np.float64)
        ac1     = lag1_autocorrelation(sig)
        delta   = delta_power_ratio(sig, fs)
        entropy = spectral_entropy(sig)
        gamma   = gamma_power_ratio(sig, fs)

        if seg["type"] == "seizure":
            sei_ac1.append(ac1);     sei_delta.append(delta)
            sei_entropy.append(entropy); sei_gamma.append(gamma)
        else:
            base_ac1.append(ac1);    base_delta.append(delta)
            base_entropy.append(entropy); base_gamma.append(gamma)

    # Report means
    print("Feature means (seizure vs baseline):")
    print(f"  {'Feature':<22}  {'Seizure':>10}  {'Baseline':>10}  Direction")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*12}")

    def _row(name, sei_vals, base_vals, higher_label):
        sm, bm = np.mean(sei_vals), np.mean(base_vals)
        direction = f"SEI>{higher_label}" if sm > bm else f"BASE>{higher_label}"
        print(f"  {name:<22}  {sm:>10.4f}  {bm:>10.4f}  {direction}")

    _row("Lag-1 autocorr (AC1)", sei_ac1, base_ac1, "BASE")
    _row("Delta ratio",          sei_delta, base_delta, "BASE")
    _row("Spectral entropy",     sei_entropy, base_entropy, "BASE")
    _row("Gamma ratio (QA obs)", sei_gamma, base_gamma, "BASE")

    # t-tests
    print()
    print("Welch t-tests (same test as QA orbit separation):")
    print(f"  {'Feature':<22}  {'t-stat':>8}  {'p-value':>10}  {'Sig?':>6}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*10}  {'-'*6}")

    # QA orbit reference values
    qa_singularity_t = -4.766
    qa_singularity_p = 1.2e-5  # approximate from output

    def _ttest_row(name, sei_vals, base_vals):
        t, p = stats.ttest_ind(sei_vals, base_vals, equal_var=False)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {name:<22}  {t:>8.3f}  {p:>10.4f}  {sig:>6}")
        return t, p

    t_ac1,     p_ac1     = _ttest_row("Lag-1 autocorr (AC1)", sei_ac1, base_ac1)
    t_delta,   p_delta   = _ttest_row("Delta ratio",          sei_delta, base_delta)
    t_entropy, p_entropy = _ttest_row("Spectral entropy",     sei_entropy, base_entropy)
    t_gamma,   p_gamma   = _ttest_row("Gamma ratio (QA obs)", sei_gamma, base_gamma)

    print()
    print(f"  QA singularity (reference): t={qa_singularity_t:.3f}  p<0.0001  ***")

    # Summary interpretation
    print()
    print("=" * 70)
    print("SPECIFICITY ANALYSIS")
    print()

    all_t = [
        ("Lag-1 AC1",    abs(t_ac1),     p_ac1),
        ("Delta ratio",  abs(t_delta),   p_delta),
        ("Spec entropy", abs(t_entropy), p_entropy),
        ("Gamma ratio",  abs(t_gamma),   p_gamma),
    ]

    qa_abs_t = abs(qa_singularity_t)
    print(f"  QA singularity |t| = {qa_abs_t:.3f}")
    print()

    for name, abs_t, p in sorted(all_t, key=lambda x: -x[1]):
        comparison = "STRONGER" if abs_t > qa_abs_t else "WEAKER"
        print(f"  {name:<18}  |t|={abs_t:.3f}  {comparison} than QA orbit")

    stronger = [n for n, abs_t, _ in all_t if abs_t > qa_abs_t]
    weaker   = [n for n, abs_t, _ in all_t if abs_t <= qa_abs_t]

    print()
    if not stronger:
        print("  VERDICT: QA orbit singularity is STRONGER than all classical features.")
        print("  The orbit separation is NOT merely a proxy for spectral/temporal statistics.")
        print("  The QA algebraic structure adds discriminative information.")
    elif len(stronger) == len(all_t):
        print("  VERDICT: All classical features are stronger than QA orbit singularity.")
        print("  The QA result may be a weaker proxy for spectral/temporal statistics.")
    else:
        print(f"  VERDICT: Mixed. Classical STRONGER: {stronger}")
        print(f"           Classical WEAKER:   {weaker}")
        print("  QA adds partial but not dominant information over classical features.")

    print()
    print("Note: t-statistic magnitude comparison is informal (different scales).")
    print("A proper specificity test requires multivariate regression with QA + classical.")


if __name__ == "__main__":
    main()
