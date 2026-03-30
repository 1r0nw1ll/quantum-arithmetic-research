#!/usr/bin/env python3
"""
qa_autocorrelation_baseline.py
================================
Empirical question: is orbit_follow_rate (OFR) measuring QA-specific structure,
or is it just a proxy for lag-1 autocorrelation (AC)?

The OFR check: for consecutive quantized audio states (b_t, e_t) = (x_t, x_{t+1}),
does x_{t+2} == (x_t + x_{t+1}) % m?

This is a mod-m Fibonacci/linear recurrence test. The baseline null hypothesis:
  OFR is fully explained by lag-1 AC — any smooth signal produces elevated OFR.

Method:
  1. Sweep sinusoids across 20 frequencies → vary lag-1 AC systematically
  2. Sweep AR(1) processes across 10 lag-1 correlation coefficients (rho)
  3. For each signal: compute lag1_AC and OFR under both quantization schemes
  4. Fit OFR ~ linear(lag1_AC) and compute R², residuals
  5. Test: do specific frequencies show OFR anomalies beyond AC prediction?
  6. Compare sine vs AR(1) at matched AC levels — same AC, different waveform structure

If R² ≈ 1.0: OFR is just lag-1 AC renamed.
If R² << 1.0 or residuals are structured: QA detects something AC does not.

Key result from prior experiment:
  sine_880Hz (8000Hz SR) → OFR = 0.1715,  chance = 0.1111
  white_noise            → OFR = 0.1063 ≈ chance
"""

import numpy as np
from collections import Counter

MODULUS  = 9
SR       = 8000
DURATION = 2.0

# ── QA arithmetic ─────────────────────────────────────────────────────────────

def quantize_raw(samples, m=MODULUS):
    clipped = np.clip(samples, -1.0, 1.0)
    states  = (((clipped + 1.0) / 2.0) * m).astype(int)
    return np.clip(states, 0, m - 1)

def quantize_eq(samples, m=MODULUS):
    n      = len(samples)
    ranks  = np.argsort(np.argsort(samples))
    states = (ranks * m // n).astype(int)
    return np.clip(states, 0, m - 1)

def compute_ofr(states, m=MODULUS):
    """
    OFR = fraction of triples (x_t, x_{t+1}, x_{t+2}) where
    x_{t+2} == (x_t + x_{t+1}) % m.
    (The first condition x_{t+1} == e_t is tautologically true.)
    """
    b = states[:-2]
    e = states[1:-1]
    n = states[2:]
    hits = np.sum(n == (b + e) % m)
    return hits / len(b)

def lag1_ac(samples):
    """Lag-1 Pearson autocorrelation of the raw float signal."""
    x  = samples[:-1]
    y  = samples[1:]
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = np.sqrt(((x - xm)**2).sum() * ((y - ym)**2).sum())
    return num / den if den > 0 else 0.0

# ── Signal generators ──────────────────────────────────────────────────────────

def gen_sine(freq, sr=SR, duration=DURATION):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def gen_ar1(rho, n=None, seed=None):
    """AR(1) process x_t = rho * x_{t-1} + eps, eps ~ N(0,1)."""
    if n is None:
        n = int(SR * DURATION)
    rng = np.random.default_rng(seed or 0)
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = rho * x[t-1] + np.sqrt(1 - rho**2) * rng.standard_normal()
    # Normalise to [-1,1]
    x -= x.mean()
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x

# ── Analysis ───────────────────────────────────────────────────────────────────

def analyse(label, samples, m=MODULUS):
    ac       = lag1_ac(samples)
    raw_s    = quantize_raw(samples, m)
    eq_s     = quantize_eq(samples, m)
    ofr_raw  = compute_ofr(raw_s, m)
    ofr_eq   = compute_ofr(eq_s, m)
    return dict(label=label, ac=round(ac, 4),
                ofr_raw=round(ofr_raw, 4), ofr_eq=round(ofr_eq, 4))

# ── Experiment 1: sine sweep ──────────────────────────────────────────────────

SINE_FREQS = [55, 110, 220, 440, 660, 880, 1000, 1100, 1320, 1760,
              2000, 2200, 2640, 3000, 3520, 4000, 4400, 5000, 6000, 7000]

def exp1_sine_sweep():
    print("=" * 68)
    print("EXPERIMENT 1: Sine sweep — OFR vs lag-1 AC across frequencies")
    print("=" * 68)
    print(f"  Chance OFR = 1/m = {1/MODULUS:.4f}   SR={SR}Hz  m={MODULUS}")
    print()
    print(f"  {'Label':22s}  {'AC':>7}  {'OFR_raw':>8}  {'OFR_eq':>8}  {'OFR_eq>chance':>14}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*14}")

    results = []
    chance  = 1 / MODULUS
    for freq in SINE_FREQS:
        sig = gen_sine(freq)
        r   = analyse(f"sine_{freq}Hz", sig)
        exc = f"+{r['ofr_eq']-chance:+.4f}"
        marker = " <<" if r['ofr_eq'] > chance * 1.3 else ""
        print(f"  {r['label']:22s}  {r['ac']:>7.4f}  {r['ofr_raw']:>8.4f}  {r['ofr_eq']:>8.4f}  {exc:>14}{marker}")
        results.append(r)

    return results

# ── Experiment 2: AR(1) sweep — matched AC ────────────────────────────────────

AR1_RHOS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]

def exp2_ar1_sweep():
    print()
    print("=" * 68)
    print("EXPERIMENT 2: AR(1) sweep — same AC as sines, different waveform")
    print("=" * 68)
    print(f"  {'Label':20s}  {'rho':>6}  {'AC':>7}  {'OFR_raw':>8}  {'OFR_eq':>8}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}")

    results = []
    for rho in AR1_RHOS:
        sig = gen_ar1(rho)
        r   = analyse(f"ar1_rho={rho:.2f}", sig)
        print(f"  {r['label']:20s}  {rho:>6.2f}  {r['ac']:>7.4f}  {r['ofr_raw']:>8.4f}  {r['ofr_eq']:>8.4f}")
        results.append(r)

    return results

# ── Experiment 3: AC-matched comparison ───────────────────────────────────────

def exp3_matched_comparison(sine_results, ar1_results):
    """
    At matched AC levels: compare sine OFR vs AR(1) OFR.
    If OFR just tracks AC: sine_OFR ≈ ar1_OFR at same AC.
    If OFR is waveform-specific: they diverge.
    """
    print()
    print("=" * 68)
    print("EXPERIMENT 3: AC-matched sine vs AR(1) — does waveform shape matter?")
    print("=" * 68)
    print()
    print(f"  {'Signal':26s}  {'AC':>7}  {'OFR_eq':>8}  {'Source':>10}")
    print(f"  {'-'*26}  {'-'*7}  {'-'*8}  {'-'*10}")

    # Find sine results with AC in various ranges and closest AR(1)
    for sr in sine_results:
        # Find closest AR(1) by AC
        closest_ar1 = min(ar1_results, key=lambda x: abs(x['ac'] - sr['ac']))
        diff = sr['ofr_eq'] - closest_ar1['ofr_eq']
        marker = f" Δ={diff:+.4f}"
        print(f"  {sr['label']:26s}  {sr['ac']:>7.4f}  {sr['ofr_eq']:>8.4f}  {'sine':>10}")
        print(f"  {closest_ar1['label']:26s}  {closest_ar1['ac']:>7.4f}  {closest_ar1['ofr_eq']:>8.4f}  {'AR(1)':>10}{marker}")
        print()

# ── Experiment 4: Regression ──────────────────────────────────────────────────

def exp4_regression(sine_results, ar1_results):
    """Fit OFR_eq ~ a * AC + b for sines and AR(1) separately."""
    print("=" * 68)
    print("EXPERIMENT 4: Linear regression OFR_eq ~ AC")
    print("=" * 68)

    chance = 1 / MODULUS

    for label, results in [("Sines", sine_results), ("AR(1)", ar1_results)]:
        ac_vals  = np.array([r['ac']     for r in results])
        ofr_vals = np.array([r['ofr_eq'] for r in results])
        # Fit linear regression
        A    = np.column_stack([ac_vals, np.ones_like(ac_vals)])
        coef, residuals, _, _ = np.linalg.lstsq(A, ofr_vals, rcond=None)
        pred = A @ coef
        ss_res = np.sum((ofr_vals - pred) ** 2)
        ss_tot = np.sum((ofr_vals - ofr_vals.mean()) ** 2)
        r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(ss_res / len(ofr_vals))

        print(f"\n  {label}:")
        print(f"    OFR_eq = {coef[0]:+.4f} * AC + {coef[1]:+.4f}")
        print(f"    R² = {r2:.4f}   RMSE = {rmse:.4f}")
        print(f"    OFR range: [{ofr_vals.min():.4f}, {ofr_vals.max():.4f}]")
        print(f"    AC range:  [{ac_vals.min():.4f},  {ac_vals.max():.4f}]")

        # Residuals for sines
        if label == "Sines":
            print(f"\n    Residuals (OFR_eq - predicted) by frequency:")
            for r, pred_v in zip(sine_results, pred):
                res = r['ofr_eq'] - pred_v
                flag = " <<" if abs(res) > 2 * rmse else ""
                print(f"      {r['label']:22s}  AC={r['ac']:.3f}  OFR={r['ofr_eq']:.4f}  pred={pred_v:.4f}  res={res:+.4f}{flag}")

# ── Experiment 5: Specific 880Hz deep dive ────────────────────────────────────

def exp5_880hz_deep(m=MODULUS):
    """
    880Hz at 8000Hz SR: lag-1 AC = cos(2π*880/8000).
    Prior result: OFR = 0.1715.
    Analytical check: what OFR does a pure sine with this AC predict
    under the AC-only model?
    """
    print()
    print("=" * 68)
    print("EXPERIMENT 5: 880Hz deep dive — analytical vs empirical OFR")
    print("=" * 68)

    freq = 880
    theoretical_ac = np.cos(2 * np.pi * freq / SR)
    print(f"\n  880Hz at {SR}Hz: theoretical lag-1 AC = cos(2π*{freq}/{SR}) = {theoretical_ac:.4f}")

    sig = gen_sine(880)
    r   = analyse("sine_880Hz", sig)
    print(f"  Measured lag-1 AC = {r['ac']:.4f}  (should match {theoretical_ac:.4f})")
    print(f"  OFR_raw = {r['ofr_raw']:.4f}")
    print(f"  OFR_eq  = {r['ofr_eq']:.4f}")
    print(f"  Chance  = {1/m:.4f}")
    print(f"  Excess above chance: {r['ofr_eq'] - 1/m:+.4f}")

    # Test: AR(1) with same rho as lag-1 AC of this sine
    rho = r['ac']
    ar1_matched = gen_ar1(rho, seed=42)
    r_ar1 = analyse(f"ar1_rho={rho:.3f}", ar1_matched)
    print(f"\n  AR(1) matched at rho={rho:.4f}:")
    print(f"    Measured AC = {r_ar1['ac']:.4f}")
    print(f"    OFR_eq      = {r_ar1['ofr_eq']:.4f}")
    print(f"    Δ OFR (sine - AR1) = {r['ofr_eq'] - r_ar1['ofr_eq']:+.4f}")

    if abs(r['ofr_eq'] - r_ar1['ofr_eq']) < 0.005:
        verdict = "INDISTINGUISHABLE — OFR_eq for 880Hz sine is explained by lag-1 AC alone"
    elif r['ofr_eq'] > r_ar1['ofr_eq'] + 0.01:
        verdict = "SINE ELEVATED — OFR_eq for 880Hz sine exceeds AC-matched AR(1) → waveform structure matters"
    else:
        verdict = "SINE SUPPRESSED — unusual; check quantization"

    print(f"\n  >> VERDICT: {verdict}")

    # Frequency scan near 880Hz: does OFR spike at specific frequencies?
    print(f"\n  Frequency scan around 880Hz (checking for resonance):")
    print(f"  {'Freq':>7}  {'AC':>7}  {'OFR_eq':>8}  {'Δchance':>8}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")
    for f in [770, 800, 820, 840, 860, 880, 900, 920, 940, 960, 1000, 1100]:
        s  = gen_sine(f)
        rv = analyse(f"{f}Hz", s)
        excess = rv['ofr_eq'] - 1/m
        flag = " <<" if excess > 0.05 else ""
        print(f"  {f:>7}  {rv['ac']:>7.4f}  {rv['ofr_eq']:>8.4f}  {excess:>+8.4f}{flag}")

    return r, r_ar1

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print()
    print("QA AUTOCORRELATION BASELINE EXPERIMENT")
    print("=" * 68)
    print("Question: is orbit_follow_rate (OFR) measuring QA-specific structure")
    print("or just lag-1 autocorrelation (AC)?")
    print()
    print("OFR check: x_{t+2} == (x_t + x_{t+1}) % m  (mod-9 Fibonacci recurrence)")
    print(f"Chance level: 1/{MODULUS} = {1/MODULUS:.4f}")
    print()

    sine_results = exp1_sine_sweep()
    ar1_results  = exp2_ar1_sweep()
    exp4_regression(sine_results, ar1_results)
    r_880, r_ar1 = exp5_880hz_deep()

    # Final verdict
    print()
    print("=" * 68)
    print("FINAL ASSESSMENT")
    print("=" * 68)
    sine_ofrs = [r['ofr_eq'] for r in sine_results]
    ar1_ofrs  = [r['ofr_eq'] for r in ar1_results]
    chance    = 1 / MODULUS
    sines_elevated = sum(1 for v in sine_ofrs if v > chance * 1.2)
    print(f"  Sines above 1.2× chance:  {sines_elevated}/{len(sine_ofrs)}")
    print(f"  AR(1) above 1.2× chance:  {sum(1 for v in ar1_ofrs if v > chance*1.2)}/{len(ar1_ofrs)}")
    print(f"  OFR range (sines):        [{min(sine_ofrs):.4f}, {max(sine_ofrs):.4f}]")
    print(f"  OFR range (AR1):          [{min(ar1_ofrs):.4f}, {max(ar1_ofrs):.4f}]")
    print()
    print("  If sines show much wider OFR range than AC-matched AR(1):")
    print("  → waveform structure (not just AC) influences OFR → QA detects something real")
    print("  If sines and AR(1) have similar OFR at matched AC:")
    print("  → OFR is an AC proxy → audio claim needs revision")

if __name__ == "__main__":
    main()
