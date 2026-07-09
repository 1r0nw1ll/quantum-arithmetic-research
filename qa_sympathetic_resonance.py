#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=tuning_pattern_generator, state_alphabet=mod24_A1_compliant"
"""
QA Phase-Conjugate Sympathetic Resonance — falsifiable demo for the SVP capstone.

Keely's Sympathetic Vibratory Physics: two systems tuned to the same "chord"
couple and transfer energy by SYMPATHETIC vibration, regardless of their
instantaneous phase; systems of different chords do NOT couple (sympathetic
SELECTIVITY). This maps onto the phase-conjugate resonance we have certified four
ways (same-medium specificity: [518], [520], [521], channel equalizer).

Thesis (falsifiable): sympathetic coupling = phase-conjugate lock. Two resonators
of the SAME tuning pattern at DIFFERENT global phase (p_j = qa_add(p_i, phi))
couple strongly because the phase-conjugate lock (scan the compensation phase)
recognises them as the same chord; a DIFFERENT tuning does not couple even with
the scan. Naive (no lock) coupling instead sees same-tuning-different-phase as
uncoupled -- so it CANNOT be sympathetic vibration.

  - sympathetic coupling C_symp(i,j) = max_psi overlap(qa_add(p_i,psi), p_j) / N
  - naive coupling      C_naive(i,j) = overlap(p_i, p_j) / N
  Note: because C_symp maxes over m phase shifts, its false-coupling FLOOR is the
  max of m random overlaps (materially above 1/m); the different-chord baseline is
  that floor. A QA formalisation of Keely's 'neutral center' is left OPEN (the
  additive-identity pattern does not serve -- it degenerates under a global scan).

A1/S2/Theorem-NT: tuning phase state integer in {1,...,m}; pattern synthesis and
phase offsets are observer-layer; coupling is integer overlap.
"""
from __future__ import annotations
import numpy as np

M = 24
RNG = np.random.default_rng(42)


def qa_mod(x):
    return ((np.asarray(x, np.int64) - 1) % M) + 1


def qa_add(a, b):
    return qa_mod(np.asarray(a, np.int64) + np.asarray(b, np.int64))


def overlap(a, b):
    return int(np.sum(a == b))


def c_naive(pi, pj):
    return overlap(pi, pj) / len(pi)


def c_symp(pi, pj):
    """Sympathetic (phase-conjugate) coupling: best match over all global phase
    compensations -- the sympathetic lock."""
    return max(overlap(qa_add(pi, psi), pj) for psi in range(1, M + 1)) / len(pi)


def run():
    N = 64          # resonator pattern length (a 'chord')
    K = 5           # distinct tunings (chords)
    print(f"QA PHASE-CONJUGATE SYMPATHETIC RESONANCE  (m={M}, N={N}, {K} distinct tunings)\n")

    tunings = [RNG.integers(1, M + 1, N) for _ in range(K)]

    # [1] Sympathetic selectivity + phase invariance: same tuning at random phase
    #     offsets vs different tuning. Sympathetic (locked) should couple same
    #     chords regardless of phase; naive should not.
    print("[1] Coupling: same chord (random phase offset) vs different chord")
    print(f"{'pair':>18s} {'C_naive':>9s} {'C_symp':>8s}")
    trials = 200
    same_naive = same_symp = diff_naive = diff_symp = 0.0
    for _ in range(trials):
        t = RNG.integers(K)
        phi = int(RNG.integers(1, M + 1))
        pi = tunings[t]
        pj_same = qa_add(tunings[t], phi)             # same chord, shifted phase
        t2 = (t + 1 + RNG.integers(K - 1)) % K
        pj_diff = qa_add(tunings[t2], phi)            # different chord
        same_naive += c_naive(pi, pj_same); same_symp += c_symp(pi, pj_same)
        diff_naive += c_naive(pi, pj_diff); diff_symp += c_symp(pi, pj_diff)
    print(f"{'same chord':>18s} {same_naive/trials:9.3f} {same_symp/trials:8.3f}")
    print(f"{'different chord':>18s} {diff_naive/trials:9.3f} {diff_symp/trials:8.3f}")
    # NOTE: c_symp maxes over m phase shifts, so its false-coupling FLOOR is the
    # max of m random overlaps -- materially above 1/m. The different-chord row
    # IS that floor (~{:.2f}); it is the correct null, not 1/m.
    print(f"  c_symp false-coupling floor (different chord, max over {M} shifts) "
          f"= {diff_symp/trials:.3f}  [naive 1/m={1/M:.3f} does NOT apply to c_symp]")

    # [2] Selectivity sharpness: sympathetic coupling should be BIMODAL
    #     (same-chord near 1, different-chord near chance) -> a clean threshold.
    print("\n[2] Selectivity sharpness (sympathetic coupling distribution):")
    same_vals, diff_vals = [], []
    for _ in range(300):
        t = RNG.integers(K); phi = int(RNG.integers(1, M + 1))
        same_vals.append(c_symp(tunings[t], qa_add(tunings[t], phi)))
        t2 = (t + 1 + RNG.integers(K - 1)) % K
        diff_vals.append(c_symp(tunings[t], qa_add(tunings[t2], phi)))
    same_vals, diff_vals = np.array(same_vals), np.array(diff_vals)
    print(f"  same chord:      mean {same_vals.mean():.3f}  min {same_vals.min():.3f}")
    print(f"  different chord: mean {diff_vals.mean():.3f}  max {diff_vals.max():.3f}")
    gap = same_vals.min() - diff_vals.max()
    print(f"  selectivity gap (same.min - diff.max) = {gap:+.3f}  "
          f"({'SEPARABLE' if gap > 0 else 'overlapping'})")

    # [3] False-coupling floor characterised explicitly: c_symp of many INDEPENDENT
    #     random pattern pairs (the true null). Same-chord (1.000) must sit far
    #     above this floor for the selectivity claim to hold.
    print("\n[3] c_symp false-coupling floor (independent random pairs, the null):")
    null_vals = np.array([c_symp(RNG.integers(1, M + 1, N), RNG.integers(1, M + 1, N))
                          for _ in range(400)])
    print(f"  null: mean {null_vals.mean():.3f}  95th pct {np.percentile(null_vals,95):.3f}  "
          f"max {null_vals.max():.3f}")
    print(f"  same-chord coupling (1.000) exceeds the null 95th pct by "
          f"{1.0 - np.percentile(null_vals,95):.3f}")

    print("\nOPEN: a QA formalisation of Keely's 'neutral center' is NOT settled here"
          " -- the additive-identity pattern does NOT serve as one (under a global"
          " phase scan it degenerates to a constant, measuring a chord's modal phase,"
          " not coupling). Deferred, not claimed.")

    print("\nSympathetic selectivity via the phase-conjugate lock = same-medium "
          "specificity ([518]/[520]/[521]/equalizer). Naive coupling cannot see "
          "same-chord-different-phase, so it is NOT sympathetic vibration.")


if __name__ == "__main__":
    run()
