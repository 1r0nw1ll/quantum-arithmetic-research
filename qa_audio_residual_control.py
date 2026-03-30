#!/usr/bin/env python3
"""
qa_audio_residual_control.py
==============================
Closes the audio file: is there any QA orbit signal BEYOND lag-1 autocorrelation?

Method: matched-AC comparison + partial correlation.

1. Match each dynamical signal to the stochastic signal with the nearest lag-1 AC.
   Compare OFR at the same AC level → residual at matched AC.

2. Compute partial correlation: corr(OFR, group | lag1_AC) using linear residuals.
   If partial correlation is significant → residual QA signal exists.
   If not → audio effect fully explained by AC.

3. AR(1) matched control: ar1_alpha95 has near-identical AC to sine_440Hz.
   Their OFR difference is the purest residual signal.

VERDICT: RESIDUAL_PRESENT if partial_r > 0.5 AND matched OFR gap > 0.005
         RESIDUAL_ABSENT  otherwise (audio file closed, AC confound confirmed)
"""

import numpy as np
from scipy import stats
import sys

MODULUS = 9
SR      = 8000
DURATION = 2.0

# ---------------------------------------------------------------------------
# Signal generators (same as ac_baseline)
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

def gen_ar1(alpha=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION)
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = alpha * x[i-1] + rng.standard_normal() * np.sqrt(1 - alpha*alpha)
    return x

def gen_pink_noise(seed=42):
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION)
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n); freqs[0] = 1.0
    power = 1.0 / np.sqrt(freqs); power[0] = 0.0
    pink = np.fft.irfft(np.fft.rfft(white) * power, n=n)
    return pink / (np.std(pink) + 1e-8)

def gen_white_noise(seed=42):
    return np.random.default_rng(seed).standard_normal(int(SR * DURATION))

SIGNALS = [
    ("sine_440Hz",   gen_sine(440),  "dynamical"),
    ("sine_880Hz",   gen_sine(880),  "dynamical"),
    ("two_tone",     gen_two_tone(), "dynamical"),
    ("am_modulated", gen_am_sine(),  "dynamical"),
    ("chirp",        gen_chirp(),    "dynamical"),
    ("ar1_alpha95",  gen_ar1(0.95),  "stochastic"),
    ("ar1_alpha50",  gen_ar1(0.50),  "stochastic"),
    ("pink_noise",   gen_pink_noise(), "stochastic"),
    ("white_noise",  gen_white_noise(), "stochastic"),
]

# ---------------------------------------------------------------------------
# QA + baseline metrics
# ---------------------------------------------------------------------------

def equalize_quantize(samples, m=MODULUS):
    n = len(samples)
    ranks = np.argsort(np.argsort(samples))
    states = (ranks * m // n).astype(int)
    return np.clip(states, 0, m - 1)

def orbit_follow_rate(states, m=MODULUS):
    b, e = states[:-1], states[1:]
    follow = sum(1 for i in range(len(b)-1)
                 if b[i+1] == e[i] and e[i+1] == (b[i]+e[i])%m)
    return follow / max(1, len(b)-1)

def lag1_ac(states):
    x = states.astype(float); x -= x.mean()
    return 0.0 if x.std() < 1e-10 else float(np.corrcoef(x[:-1], x[1:])[0,1])

def partial_corr_ofr_group_given_ac(data):
    """
    Partial correlation of OFR with binary group (1=dynamical, 0=stochastic)
    after removing linear effect of lag-1 AC.
    """
    ofr  = np.array([d["ofr"] for d in data])
    grp  = np.array([1.0 if d["group"]=="dynamical" else 0.0 for d in data])
    ac   = np.array([d["ac"]  for d in data])

    # Residualise both OFR and group against AC
    def residualize(y, x):
        slope, intercept, *_ = stats.linregress(x, y)
        return y - (slope * x + intercept)

    ofr_res = residualize(ofr, ac)
    grp_res = residualize(grp, ac)

    if np.std(ofr_res) < 1e-10 or np.std(grp_res) < 1e-10:
        return 0.0, 1.0
    return stats.pearsonr(ofr_res, grp_res)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 65)
    print("QA Audio Residual Control — partial AC analysis")
    print("=" * 65)
    print()

    data = []
    for name, sig, group in SIGNALS:
        states = equalize_quantize(sig)
        data.append({"name": name, "group": group,
                     "ofr": orbit_follow_rate(states), "ac": lag1_ac(states)})

    # Table
    print(f"{'Signal':<20}  {'OFR':>7}  {'lag1_AC':>8}  group")
    print("-" * 52)
    for d in data:
        print(f"{d['name']:<20}  {d['ofr']:>7.4f}  {d['ac']:>8.4f}  {d['group']}")
    print()

    # Partial correlation: OFR ~ group | AC
    partial_r, partial_p = partial_corr_ofr_group_given_ac(data)
    print(f"Partial corr(OFR, dynamical | lag1_AC): r={partial_r:+.3f}  p={partial_p:.4f}")
    print()

    # Matched-AC comparison: sine_440 (AC=0.933) vs ar1_alpha95 (AC=0.936)
    sine = next(d for d in data if d["name"] == "sine_440Hz")
    ar1  = next(d for d in data if d["name"] == "ar1_alpha95")
    matched_gap = sine["ofr"] - ar1["ofr"]
    print(f"Matched-AC comparison (nearest AC pair):")
    print(f"  sine_440Hz  OFR={sine['ofr']:.4f}  AC={sine['ac']:.4f}")
    print(f"  ar1_alpha95 OFR={ar1['ofr']:.4f}  AC={ar1['ac']:.4f}")
    print(f"  Residual OFR gap at matched AC: {matched_gap:+.4f}")
    print()

    # Chirp vs ar1_alpha50: both lower AC, different structure
    chirp = next(d for d in data if d["name"] == "chirp")
    ar1_50 = next(d for d in data if d["name"] == "ar1_alpha50")
    chirp_gap = chirp["ofr"] - ar1_50["ofr"]
    print(f"  chirp       OFR={chirp['ofr']:.4f}  AC={chirp['ac']:.4f}")
    print(f"  ar1_alpha50 OFR={ar1_50['ofr']:.4f}  AC={ar1_50['ac']:.4f}")
    print(f"  Residual OFR gap at lower AC:   {chirp_gap:+.4f}")
    print()

    # Verdict
    residual_present = (abs(partial_r) > 0.5 and partial_p < 0.10
                        and abs(matched_gap) > 0.005)

    if residual_present:
        verdict = (f"RESIDUAL_PRESENT — partial r={partial_r:+.3f} (p={partial_p:.3f}) "
                   f"and matched-AC gap={matched_gap:+.4f}. "
                   f"QA orbit structure exists beyond lag-1 AC, but effect is modest.")
    else:
        verdict = (f"RESIDUAL_ABSENT — partial r={partial_r:+.3f} (p={partial_p:.3f}). "
                   f"Audio OFR effect fully explained by lag-1 autocorrelation. "
                   f"Audio file CLOSED: reformulate metric or redirect to other domains.")

    print(f"VERDICT: {verdict}")
    print()

    return {"partial_r": round(partial_r, 3), "partial_p": round(partial_p, 4),
            "matched_gap": round(matched_gap, 4), "residual_present": residual_present,
            "verdict": verdict}

if __name__ == "__main__":
    run()
    sys.exit(0)
