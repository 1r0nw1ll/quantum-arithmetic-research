#!/usr/bin/env python3
"""
qa_audio_surrogates.py — Process-level surrogate validation for audio OFR
==========================================================================

CORRECTED DESIGN: Keep REAL signals, REAL lag-1 AC, REAL group labels.
Replace QA mod-9 rule with surrogate encoding rules.

Question: "Does the QA T-operator rule (next = (b+e) % m) detect dynamical
structure better than arbitrary prediction rules?"

Surrogate types:
1. Random-table: replace (b+e)%m with a random lookup table b×e→{0,...,m-1}
2. Shifted-mod: use (b+e+k)%m for random k (breaks QA alignment)
3. Random-mod: use random modulus m' from {5,7,11,13} instead of m=9
4. Permuted-states: permute the quantized state assignment (breaks QA ordering)

For each surrogate, compute OFR with the surrogate rule, then test
partial_corr(OFR, group | AC). Real QA rule must beat surrogates.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=audio_residual, state_alphabet=quantized_waveform"

import numpy as np
from scipy import stats
import sys

MODULUS = 9
SR = 8000
DURATION = 2.0
N_SURROGATES = 1000  # fast — only 9 signals per iteration


# ============================================================================
# SIGNAL GENERATORS — identical to qa_audio_residual_control.py
# ============================================================================

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
        x[i] = alpha * x[i - 1] + rng.standard_normal() * np.sqrt(1 - alpha * alpha)
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


# ============================================================================
# METRICS — identical to qa_audio_residual_control.py
# ============================================================================

def equalize_quantize(samples, m=MODULUS):
    n = len(samples)
    ranks = np.argsort(np.argsort(samples))
    states = (ranks * m // n).astype(int)
    return np.clip(states, 0, m - 1)


def lag1_ac(states):
    x = states.astype(float)
    x -= x.mean()
    return 0.0 if x.std() < 1e-10 else float(np.corrcoef(x[:-1], x[1:])[0, 1])


def orbit_follow_rate_qa(states, m=MODULUS):
    """REAL QA rule: next = (b + e) % m"""
    b, e = states[:-1], states[1:]
    follow = sum(1 for i in range(len(b) - 1)
                 if b[i + 1] == e[i] and e[i + 1] == (b[i] + e[i]) % m)
    return follow / max(1, len(b) - 1)


def orbit_follow_rate_table(states, table, m=MODULUS):
    """SURROGATE: use random lookup table instead of (b+e)%m"""
    b, e = states[:-1], states[1:]
    follow = sum(1 for i in range(len(b) - 1)
                 if b[i + 1] == e[i] and e[i + 1] == table[b[i], e[i]])
    return follow / max(1, len(b) - 1)


def orbit_follow_rate_shifted(states, k, m=MODULUS):
    """SURROGATE: use (b+e+k)%m instead of (b+e)%m"""
    b, e = states[:-1], states[1:]
    follow = sum(1 for i in range(len(b) - 1)
                 if b[i + 1] == e[i] and e[i + 1] == (b[i] + e[i] + k) % m)
    return follow / max(1, len(b) - 1)


def orbit_follow_rate_altmod(states, m_alt):
    """SURROGATE: use different modulus"""
    n = len(states)
    ranks = np.argsort(np.argsort(states[:len(states)]))
    states_alt = (ranks * m_alt // n).astype(int)
    states_alt = np.clip(states_alt, 0, m_alt - 1)
    b, e = states_alt[:-1], states_alt[1:]
    follow = sum(1 for i in range(len(b) - 1)
                 if b[i + 1] == e[i] and e[i + 1] == (b[i] + e[i]) % m_alt)
    return follow / max(1, len(b) - 1)


def partial_corr(ofr_vals, groups, ac_vals):
    """Partial correlation of OFR with group (1=dynamical) after removing AC."""
    ofr = np.array(ofr_vals)
    grp = np.array([1.0 if g == "dynamical" else 0.0 for g in groups])
    ac = np.array(ac_vals)

    def residualize(y, x):
        slope, intercept, *_ = stats.linregress(x, y)
        return y - (slope * x + intercept)

    ofr_res = residualize(ofr, ac)
    grp_res = residualize(grp, ac)

    if np.std(ofr_res) < 1e-10 or np.std(grp_res) < 1e-10:
        return 0.0, 1.0
    return stats.pearsonr(ofr_res, grp_res)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("QA Audio Surrogate Validation — encoding surrogates")
    print("Real signals, real AC, real groups. Surrogate: encoding rule only.")
    print("=" * 70)

    # Precompute REAL metrics
    all_states = []
    groups = []
    ac_vals = []
    real_ofr = []

    for name, sig, group in SIGNALS:
        states = equalize_quantize(sig)
        all_states.append(states)
        groups.append(group)
        ac_vals.append(lag1_ac(states))
        real_ofr.append(orbit_follow_rate_qa(states))

    real_partial_r, real_partial_p = partial_corr(real_ofr, groups, ac_vals)

    print(f"\nREAL QA (mod-{MODULUS}):")
    print(f"  partial r(OFR, dynamical | AC) = {real_partial_r:+.3f} (p={real_partial_p:.4f})")

    # Run surrogates
    surr_types = ["random_table", "shifted_mod", "random_mod", "permuted_states"]
    surr_partial_r = {st: [] for st in surr_types}

    for i in range(N_SURROGATES):
        rng = np.random.RandomState(4000 + i)

        # 1. Random table: random m×m lookup
        table = rng.randint(0, MODULUS, size=(MODULUS, MODULUS))
        ofr_rt = [orbit_follow_rate_table(s, table) for s in all_states]
        pr_rt, _ = partial_corr(ofr_rt, groups, ac_vals)
        surr_partial_r["random_table"].append(pr_rt)

        # 2. Shifted mod: (b+e+k)%m for random k in {1,...,m-1}
        k = rng.randint(1, MODULUS)
        ofr_sm = [orbit_follow_rate_shifted(s, k) for s in all_states]
        pr_sm, _ = partial_corr(ofr_sm, groups, ac_vals)
        surr_partial_r["shifted_mod"].append(pr_sm)

        # 3. Random modulus: use m' from {5,7,11,13}
        m_alt = rng.choice([5, 7, 11, 13])
        # Re-quantize with alt modulus from the original raw signals
        ofr_rm = []
        for name, sig, group in SIGNALS:
            n = len(sig)
            ranks = np.argsort(np.argsort(sig))
            states_alt = (ranks * m_alt // n).astype(int)
            states_alt = np.clip(states_alt, 0, m_alt - 1)
            b, e = states_alt[:-1], states_alt[1:]
            follow = sum(1 for j in range(len(b) - 1)
                         if b[j + 1] == e[j] and e[j + 1] == (b[j] + e[j]) % m_alt)
            ofr_rm.append(follow / max(1, len(b) - 1))
        pr_rm, _ = partial_corr(ofr_rm, groups, ac_vals)
        surr_partial_r["random_mod"].append(pr_rm)

        # 4. Permuted states: permute the state assignment
        perm = rng.permutation(MODULUS)
        ofr_ps = []
        for s in all_states:
            perm_states = perm[s]
            ofr_ps.append(orbit_follow_rate_qa(perm_states))
        pr_ps, _ = partial_corr(ofr_ps, groups, ac_vals)
        surr_partial_r["permuted_states"].append(pr_ps)

    # Compare
    print(f"\n{'=' * 70}")
    print(f"SURROGATE COMPARISON ({N_SURROGATES} iterations each)")
    print("=" * 70)

    summary = {}
    for st in surr_types:
        vals = np.array(surr_partial_r[st])
        vals = vals[np.isfinite(vals)]
        mean_s, std_s = np.mean(vals), np.std(vals)
        rank_p = np.mean(np.abs(vals) >= np.abs(real_partial_r))
        z = (np.abs(real_partial_r) - np.mean(np.abs(vals))) / np.std(np.abs(vals)) if np.std(np.abs(vals)) > 0 else 0
        beats = "BEATS" if rank_p < 0.05 else "FAILS"
        sig = "***" if rank_p < 0.001 else "**" if rank_p < 0.01 else "*" if rank_p < 0.05 else "ns"

        print(f"  {st:>20}: surr={mean_s:+.3f}±{std_s:.3f}, rank_p={rank_p:.4f} → {beats} {sig}")
        summary[st] = {
            "real": float(real_partial_r), "surr_mean": float(mean_s), "surr_std": float(std_s),
            "z": float(z), "rank_p": float(rank_p), "beats": beats == "BEATS",
        }

    n_pass = sum(1 for st in surr_types if summary[st]["beats"])
    print(f"\n  SCORECARD: {n_pass}/4 surrogate types beaten")

    if n_pass >= 3:
        print(f"\n  VERDICT: Audio Tier 3 CONFIRMED — QA mod-9 OFR beats encoding surrogates")
    elif n_pass >= 2:
        print(f"\n  VERDICT: Audio TRENDING")
    else:
        print(f"\n  VERDICT: Audio does not survive encoding surrogates")

    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    output = {
        "domain": "audio_surrogates",
        "design": "CORRECTED: real signals/AC/groups, surrogate encoding rules only",
        "real_partial_r": float(real_partial_r),
        "real_partial_p": float(real_partial_p),
        "n_surrogates": N_SURROGATES,
        "summary": summary,
        "n_pass": n_pass,
    }
    with open(os.path.join(here, "qa_audio_surrogate_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to qa_audio_surrogate_results.json")


if __name__ == "__main__":
    main()
