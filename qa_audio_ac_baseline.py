#!/usr/bin/env python3
"""
qa_audio_ac_baseline.py
========================
Baseline comparison: is orbit_follow_rate QA-specific or just lag-1 autocorrelation?

Scientific question
-------------------
qa_audio_orbit_test.py found:
  dynamical signals  EQ orbit_follow_rate = 0.1432
  stochastic signals EQ orbit_follow_rate = 0.1288
  EQ gap = +0.0145

Is this gap explained by classical time-series statistics (lag-1 AC, AR(1) predictability)
or does it survive as a genuinely QA-specific structural signal?

Controls applied to each signal
---------------------------------
1. lag1_ac_bseq    — lag-1 autocorrelation of the quantized b-sequence
2. ar1_pred_rate   — AR(1) fit on b-sequence; fraction of steps where AR(1) predicts
                     correct next b (same threshold: exact match mod m)
3. shuffle_ofr     — orbit_follow_rate on shuffled b-sequence (destroys all structure)
4. phase_rand_ofr  — orbit_follow_rate on phase-randomized signal (preserves power
                     spectrum but randomises phase — destroys temporal structure)
5. orbit_follow_rate (raw and EQ) — the actual QA metric

Verdict logic
-------------
VERDICT: QA-SPECIFIC
  EQ gap > 0.005 AND gap is NOT explained by lag-1 AC
  (i.e., Pearson r between orbit_follow_rate and lag1_ac across signals < 0.8)

VERDICT: CONFOUNDED
  EQ gap > 0.005 BUT strongly correlated with lag-1 AC (r > 0.8)

VERDICT: NULL
  EQ gap <= 0.005 — no reliable effect to explain

Stdlib + numpy + scipy only (no PyTorch).
"""

import numpy as np
from collections import Counter
from scipy import stats
import sys

MODULUS = 9
SR      = 8000
DURATION = 2.0
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Signal generators (replicated from qa_audio_orbit_test.py)
# ---------------------------------------------------------------------------

def gen_sine(freq=440.0):
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def gen_two_tone(f1=440.0, f2=660.0):
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    return 0.5 * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))

def gen_am_sine(carrier=440.0, mod=5.0, depth=0.8):
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    return (1.0 + depth * np.sin(2 * np.pi * mod * t)) * np.sin(2 * np.pi * carrier * t)

def gen_chirp(f0=200.0, f1=2000.0):
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * DURATION) * t * t)
    return np.sin(phase)

def gen_white_noise(seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * DURATION))

def gen_pink_noise(seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION)
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    power = 1.0 / np.sqrt(freqs)
    power[0] = 0.0
    pink = np.fft.irfft(np.fft.rfft(white) * power, n=n)
    return pink / (np.std(pink) + 1e-8)

def gen_ar1(alpha=0.95, seed=RNG_SEED):
    """AR(1) process x_t = alpha*x_{t-1} + eps — purely statistical persistence."""
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION)
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = alpha * x[i-1] + rng.standard_normal() * np.sqrt(1 - alpha * alpha)
    return x

SIGNALS = [
    # (name, signal, group)
    ("sine_440Hz",    gen_sine(440),     "dynamical"),
    ("sine_880Hz",    gen_sine(880),     "dynamical"),
    ("two_tone",      gen_two_tone(),    "dynamical"),
    ("am_modulated",  gen_am_sine(),     "dynamical"),
    ("chirp",         gen_chirp(),       "dynamical"),
    ("ar1_alpha95",   gen_ar1(0.95),     "stochastic"),  # high-AC but purely statistical
    ("ar1_alpha50",   gen_ar1(0.50),     "stochastic"),  # low-AC statistical
    ("pink_noise",    gen_pink_noise(),  "stochastic"),
    ("white_noise",   gen_white_noise(), "stochastic"),
]

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize(samples, m=MODULUS):
    clipped = np.clip(samples, -1.0, 1.0)
    states = (((clipped + 1.0) / 2.0) * m).astype(int)
    return np.clip(states, 0, m - 1)

def equalize_quantize(samples, m=MODULUS):
    n = len(samples)
    ranks = np.argsort(np.argsort(samples))
    states = (ranks * m // n).astype(int)
    return np.clip(states, 0, m - 1)

# ---------------------------------------------------------------------------
# QA metrics
# ---------------------------------------------------------------------------

def orbit_follow_rate(states, m=MODULUS):
    """Fraction of consecutive state pairs (b,e)→(b',e') where (b',e') == T(b,e)."""
    b_seq = states[:-1]
    e_seq = states[1:]
    follow = 0
    for i in range(len(b_seq) - 1):
        b, e = b_seq[i], e_seq[i]
        tb, te = e, (b + e) % m          # T(b,e)
        if b_seq[i+1] == tb and e_seq[i+1] == te:
            follow += 1
    return follow / max(1, len(b_seq) - 1)

# ---------------------------------------------------------------------------
# Baseline controls
# ---------------------------------------------------------------------------

def lag1_ac(states):
    """Lag-1 autocorrelation of the integer state sequence."""
    x = states.astype(float)
    x -= x.mean()
    if x.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])

def ar1_prediction_rate(states, m=MODULUS):
    """
    Fit AR(1) to b-sequence. For each step, predict next b as round(alpha*b + c) % m.
    Return fraction of exact matches — direct analogue of orbit_follow_rate.
    """
    b_seq = states[:-1].astype(float)
    b_next = states[1:].astype(float)
    if len(b_seq) < 10:
        return 0.0
    # OLS: b_next = alpha * b_seq + c
    slope, intercept, *_ = stats.linregress(b_seq, b_next)
    predicted = (np.round(slope * b_seq + intercept).astype(int)) % m
    actual = states[1:].astype(int) % m
    return float(np.mean(predicted == actual))

def shuffle_ofr(states, m=MODULUS, seed=RNG_SEED):
    """orbit_follow_rate on randomly shuffled state sequence (null baseline)."""
    rng = np.random.default_rng(seed)
    shuffled = states.copy()
    rng.shuffle(shuffled)
    return orbit_follow_rate(shuffled, m)

def phase_randomize_ofr(signal, m=MODULUS, seed=RNG_SEED):
    """
    Phase-randomize the signal in Fourier space (preserves power spectrum,
    destroys temporal ordering), then recompute orbit_follow_rate.
    This isolates amplitude distribution and spectral shape as confounds.
    """
    rng = np.random.default_rng(seed)
    n = len(signal)
    fft = np.fft.rfft(signal)
    random_phases = rng.uniform(0, 2 * np.pi, len(fft))
    fft_randomized = np.abs(fft) * np.exp(1j * random_phases)
    sig_rand = np.fft.irfft(fft_randomized, n=n)
    # Normalize to same range
    sig_rand = sig_rand / (np.std(sig_rand) + 1e-8) * np.std(signal)
    states_rand = equalize_quantize(sig_rand, m)
    return orbit_follow_rate(states_rand, m)

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_all():
    chance = 1.0 / MODULUS
    results = []

    print(f"Chance level (1/m): {chance:.4f}   m={MODULUS}")
    print()

    hdr = f"{'Signal':<20}  {'OFR_eq':>8}  {'OFR_shuf':>9}  {'OFR_phase':>10}  {'lag1_AC':>8}  {'AR1_pred':>9}  {'group'}"
    print(hdr)
    print("-" * len(hdr))

    for name, signal, group in SIGNALS:
        eq_states  = equalize_quantize(signal, MODULUS)

        ofr_eq     = orbit_follow_rate(eq_states, MODULUS)
        ofr_shuf   = shuffle_ofr(eq_states, MODULUS)
        ofr_phase  = phase_randomize_ofr(signal, MODULUS)
        ac1        = lag1_ac(eq_states)
        ar1_pred   = ar1_prediction_rate(eq_states, MODULUS)

        results.append({
            "name":      name,
            "group":     group,
            "ofr_eq":    ofr_eq,
            "ofr_shuf":  ofr_shuf,
            "ofr_phase": ofr_phase,
            "lag1_ac":   ac1,
            "ar1_pred":  ar1_pred,
        })

        print(f"{name:<20}  {ofr_eq:>8.4f}  {ofr_shuf:>9.4f}  {ofr_phase:>10.4f}  "
              f"{ac1:>8.4f}  {ar1_pred:>9.4f}  {group}")

    print()
    return results

def summarize(results):
    chance = 1.0 / MODULUS

    dyn  = [r for r in results if r["group"] == "dynamical"]
    sto  = [r for r in results if r["group"] == "stochastic"]

    mean = lambda key, grp: float(np.mean([r[key] for r in grp]))

    # Group means
    dyn_ofr   = mean("ofr_eq",    dyn)
    sto_ofr   = mean("ofr_eq",    sto)
    dyn_ac    = mean("lag1_ac",   dyn)
    sto_ac    = mean("lag1_ac",   sto)
    dyn_ar1   = mean("ar1_pred",  dyn)
    sto_ar1   = mean("ar1_pred",  sto)
    dyn_phase = mean("ofr_phase", dyn)
    sto_phase = mean("ofr_phase", sto)
    shuf_mean = mean("ofr_shuf",  results)

    eq_gap       = dyn_ofr  - sto_ofr
    ac_gap       = dyn_ac   - sto_ac
    ar1_gap      = dyn_ar1  - sto_ar1
    phase_gap    = dyn_phase - sto_phase

    print("=== Group Means ===")
    print(f"{'Metric':<22}  {'dynamical':>10}  {'stochastic':>10}  {'gap':>8}")
    print("-" * 58)
    print(f"{'OFR (EQ)':22}  {dyn_ofr:>10.4f}  {sto_ofr:>10.4f}  {eq_gap:>+8.4f}")
    print(f"{'OFR (phase-rand)':22}  {dyn_phase:>10.4f}  {sto_phase:>10.4f}  {phase_gap:>+8.4f}")
    print(f"{'Lag-1 AC':22}  {dyn_ac:>10.4f}  {sto_ac:>10.4f}  {ac_gap:>+8.4f}")
    print(f"{'AR(1) pred rate':22}  {dyn_ar1:>10.4f}  {sto_ar1:>10.4f}  {ar1_gap:>+8.4f}")
    print(f"{'Shuffle OFR (null)':22}  {'—':>10}  {'—':>10}  {shuf_mean:>8.4f}  ← null baseline")
    print()

    # Correlation: does lag-1 AC predict OFR across all signals?
    all_ac  = [r["lag1_ac"]  for r in results]
    all_ofr = [r["ofr_eq"]   for r in results]
    r_ac_ofr, p_ac_ofr = stats.pearsonr(all_ac, all_ofr)

    all_ar1 = [r["ar1_pred"] for r in results]
    r_ar1_ofr, p_ar1_ofr = stats.pearsonr(all_ar1, all_ofr)

    all_phase = [r["ofr_phase"] for r in results]
    r_phase_ofr, p_phase_ofr = stats.pearsonr(all_phase, all_ofr)

    print("=== Correlations (across all signals) ===")
    print(f"  OFR ~ lag1_AC:        r={r_ac_ofr:+.3f}  p={p_ac_ofr:.4f}")
    print(f"  OFR ~ AR(1) pred:     r={r_ar1_ofr:+.3f}  p={p_ar1_ofr:.4f}")
    print(f"  OFR ~ phase-rand OFR: r={r_phase_ofr:+.3f}  p={p_phase_ofr:.4f}")
    print()

    # Phase-rand gap: does the OFR gap survive phase randomization?
    # If yes, the effect is NOT purely due to amplitude distribution / power spectrum
    phase_rand_gap_survives = phase_gap > 0.003

    # AC confound check: is OFR strongly predicted by lag-1 AC?
    ac_confounded = abs(r_ac_ofr) > 0.8 and p_ac_ofr < 0.05

    print("=== Baseline Assessment ===")
    print(f"  EQ OFR gap (dyn - sto):          {eq_gap:+.4f}  (chance={chance:.4f})")
    print(f"  Phase-rand gap survives:          {phase_rand_gap_survives}  (gap={phase_gap:+.4f})")
    print(f"  Lag-1 AC confound (r>0.8):        {ac_confounded}  (r={r_ac_ofr:+.3f})")
    print(f"  Shuffle null (expected ~{chance:.3f}):    {shuf_mean:.4f}")
    print()

    # Verdict
    if abs(eq_gap) <= 0.003:
        verdict = "NULL — EQ gap too small to interpret (<= 0.003)"
        qa_specific = False
    elif ac_confounded:
        verdict = (f"CONFOUNDED — OFR gap explained by lag-1 AC "
                   f"(r={r_ac_ofr:+.3f}, p={p_ac_ofr:.4f}). "
                   f"QA metric is capturing temporal autocorrelation, not structural orbit adherence.")
        qa_specific = False
    elif phase_rand_gap_survives:
        verdict = (f"QA-SPECIFIC — EQ gap={eq_gap:+.4f} survives phase randomization "
                   f"(phase gap={phase_gap:+.4f}) and is NOT explained by lag-1 AC "
                   f"(r={r_ac_ofr:+.3f}). Temporal orbit structure beyond spectral content confirmed.")
        qa_specific = True
    else:
        verdict = (f"PARTIAL — EQ gap exists ({eq_gap:+.4f}) but does not clearly survive "
                   f"phase randomization (phase gap={phase_gap:+.4f}). "
                   f"Effect may be amplitude-distribution driven.")
        qa_specific = None

    print(f"VERDICT: {verdict}")
    print()

    return {
        "eq_gap": round(eq_gap, 4),
        "phase_gap": round(phase_gap, 4),
        "ac_gap": round(ac_gap, 4),
        "r_ac_ofr": round(r_ac_ofr, 3),
        "p_ac_ofr": round(p_ac_ofr, 4),
        "r_ar1_ofr": round(r_ar1_ofr, 3),
        "shuf_null": round(shuf_mean, 4),
        "ac_confounded": ac_confounded,
        "phase_rand_gap_survives": phase_rand_gap_survives,
        "qa_specific": qa_specific,
        "verdict": verdict,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("QA Audio Orbit Autocorrelation Baseline")
    print("=" * 72)
    print()

    results = analyze_all()
    summary = summarize(results)

    sys.exit(0)
