#!/usr/bin/env python3
"""
qa_eeg_microstate_alphabet_search.py — EEG Microstate Alphabet Design Tool

QA_COMPLIANCE = "design_tool_not_empirical_script"

Run BEFORE writing eeg_orbit_classifier.py to find alphabets that:
  1. Achieve nonzero satellite coverage
  2. Keep singularity load manageable
  3. Have interpretable EEG microstate → (b,e) mappings

EEG microstates A/B/C/D (Lehmann 1998) represent topographic voltage maps
that dominate EEG for ~60–120ms windows. Each is assigned a fixed (b,e) pair
declared a priori — NOT derived from any signal value.

Usage:
    python qa_eeg_microstate_alphabet_search.py           # audit candidates
    python qa_eeg_microstate_alphabet_search.py --suggest # show balanced options
"""

import sys
import itertools
from typing import NamedTuple

from qa_orbit_rules import norm_f, v3, orbit_family, qa_step
from qa_observer_alphabet_audit import audit_alphabet, print_report

MODULUS = 24  # canonical EEG modulus (same as seismic and finance)


# ── Candidate alphabets ────────────────────────────────────────────────────────
#
# EEG microstate literature uses 4 canonical states (A, B, C, D).
# Some researchers use 6 states. We test both.
#
# Constraints:
#   A1: all (b,e) in {1,...,24}
#   Satellite activation: at least one pair (b_i, e_j) in cross-product
#                         must have 8|b_i AND 8|e_j
#
# Design strategy: include at least one microstate with b or e = 8, 16, or 24
# so that transitions TO or FROM it activate the satellite channel.
#
# Physiological rationale for assignments:
#   A/B: frontal and occipital poles — high-energy states → larger values
#   C:   right-dominant pattern — intermediate
#   D:   anterior-central — quiet/baseline
#
# We try several candidate assignments and audit each.

CANDIDATES_4STATE: dict[str, dict[str, tuple[int, int]]] = {

    # Candidate 1: minimal satellite — one multiples-of-8 pair
    "4state_v1_minimal_sat": {
        "A_frontal":   ( 8, 16),  # 8%8=0, 16%8=0 → direct satellite pair!
        "B_occipital": ( 3,  7),  # cosmos
        "C_right":     (11, 19),  # cosmos
        "D_baseline":  (17, 23),  # cosmos
    },

    # Candidate 2: two satellite-capable values for richer cross-product
    "4state_v2_two_sat_vals": {
        "A_frontal":   ( 8,  3),  # b=8 enables sat as source
        "B_occipital": ( 5, 16),  # e=16 enables sat as target
        "C_right":     (11,  7),  # cosmos
        "D_baseline":  (17, 23),  # cosmos
    },

    # Candidate 3: singularity state for baseline (b=e=24 is the A1-compliant fixed point)
    "4state_v3_with_singularity": {
        "A_frontal":   ( 8,  3),  # b=8 enables sat as source
        "B_occipital": ( 5, 16),  # e=16 enables sat as target
        "C_right":     (11, 19),  # cosmos
        "D_baseline":  (24, 24),  # singularity — resting fixed point
    },

    # Candidate 4: symmetric satellite-capable — all four have satellite-capable coords
    "4state_v4_symmetric_sat": {
        "A_frontal":   ( 8,  5),  # b=8 (sat source)
        "B_occipital": ( 5,  8),  # e=8 (sat target)
        "C_right":     (16, 11),  # b=16 (sat source)
        "D_baseline":  (11, 16),  # e=16 (sat target)
    },

    # Candidate 5: match seismic design — direct satellite pair in alphabet
    "4state_v5_direct_satellite": {
        "A_frontal":   ( 8,  8),  # direct satellite pair! (not just via transition)
        "B_occipital": (16, 16),  # direct satellite pair!
        "C_right":     ( 3, 11),  # cosmos
        "D_baseline":  (17,  5),  # cosmos
    },
}

CANDIDATES_6STATE: dict[str, dict[str, tuple[int, int]]] = {

    # 6-state alphabet for higher temporal resolution
    "6state_v1_balanced": {
        "A_frontal":    ( 8,  3),  # b=8 sat-source
        "B_occipital":  ( 5, 16),  # e=16 sat-target
        "C_right":      (11,  7),  # cosmos
        "D_left":       (17, 23),  # cosmos
        "E_central":    (13,  1),  # cosmos
        "F_baseline":   (24, 24),  # singularity
    },

    # 6-state with two direct satellite pairs
    "6state_v2_rich_sat": {
        "A_frontal":    ( 8,  8),  # direct satellite
        "B_occipital":  (16, 16),  # direct satellite
        "C_right":      ( 8, 16),  # direct satellite
        "D_left":       ( 3, 11),  # cosmos
        "E_central":    (19,  7),  # cosmos
        "F_baseline":   (24, 24),  # singularity
    },

    # 6-state matching seismic design philosophy
    "6state_v3_seismic_analogue": {
        "quiet_eeg":    ( 9,  9),  # cosmos (mimics seismic quiet; uses transition encoding)
        "alpha_burst":  ( 8,  1),  # cosmos; b=8 enables satellite via transition
        "alpha_down":   ( 1,  8),  # cosmos; e=8 enables satellite via transition
        "delta_wave":   ( 3, 16),  # cosmos; e=16 enables satellite
        "theta_wave":   (16,  3),  # cosmos; b=16 enables satellite
        "artifact":     ( 7, 11),  # cosmos
    },
}


# ── Transition satellite analysis ─────────────────────────────────────────────

def satellite_transition_pairs(alphabet: dict[str, tuple[int, int]],
                               modulus: int = MODULUS) -> list[tuple[str, str, int, int]]:
    """Return all (label_a, label_b, b, e) cross-product pairs that give satellite orbit."""
    results = []
    for la, (ba, _) in alphabet.items():
        for lb, (_, eb) in alphabet.items():
            b = int(ba)
            e = int(eb)
            if orbit_family(b, e, m=modulus) == "satellite":
                results.append((la, lb, b, e))
    return results


def direct_satellite_pairs(alphabet: dict[str, tuple[int, int]],
                           modulus: int = MODULUS) -> list[tuple[str, int, int]]:
    """Return labels whose direct (b,e) pair is satellite."""
    return [(la, int(b), int(e))
            for la, (b, e) in alphabet.items()
            if orbit_family(int(b), int(e), m=modulus) == "satellite"]


def print_satellite_analysis(name: str, alphabet: dict[str, tuple[int, int]]) -> None:
    """Print satellite channel analysis for an alphabet."""
    direct = direct_satellite_pairs(alphabet)
    trans = satellite_transition_pairs(alphabet)

    print(f"\n  Direct satellite states:")
    if direct:
        for la, b, e in direct:
            print(f"    {la}: (b={b}, e={e})")
    else:
        print("    None — all direct pairs are cosmos or singularity")

    print(f"  Satellite transitions (b from source, e from target):")
    if trans:
        for la, lb, b, e in trans[:8]:
            print(f"    {la} → {lb}: b={b}, e={e}")
        if len(trans) > 8:
            print(f"    ... {len(trans)-8} more")
    else:
        print("    None — DEAD SATELLITE: alphabet cannot activate satellite channel")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    all_candidates = {**CANDIDATES_4STATE, **CANDIDATES_6STATE}

    print("=" * 70)
    print("QA EEG Microstate Alphabet Search")
    print("Design tool — run before writing eeg_orbit_classifier.py")
    print("=" * 70)
    print()
    print("Axiom A1: all (b,e) in {1,...,24}  [VERIFIED BELOW]")
    print("Satellite condition: 8|b AND 8|e for mod-24")
    print("Singularity: b=24 AND e=24 (unique fixed point)")
    print()

    # Audit each candidate
    results = []
    for name, alphabet in all_candidates.items():
        n_labels = len(alphabet)
        report = audit_alphabet(name, alphabet, modulus=MODULUS)
        results.append((name, alphabet, report))

    # Print ranked summary table
    print("=" * 70)
    print("CANDIDATE RANKING SUMMARY")
    print(f"{'Name':<36}  {'Labels':>6}  {'Sat%':>6}  {'Sing%':>6}  {'Verdict'}")
    print("-" * 70)
    for name, alphabet, r in sorted(results, key=lambda x: -x[2]["orbit_fractions"]["satellite"]):
        sat_pct = r["orbit_fractions"]["satellite"] * 100
        sing_pct = r["orbit_fractions"]["singularity"] * 100
        icon = "✓" if r["severity"] == "OK" else "⚠"
        print(f"  {name:<34}  {r['n_labels']:>6}  {sat_pct:>5.1f}%  {sing_pct:>5.1f}%  {icon} {r['verdict'][:30]}")

    # Print full report for BALANCED candidates only
    print()
    print("=" * 70)
    print("FULL REPORTS — BALANCED candidates")
    for name, alphabet, report in results:
        if report["severity"] == "OK":
            print_report(report)
            print_satellite_analysis(name, alphabet)

    # Print any WARN candidates with their satellite analysis
    print()
    print("=" * 70)
    print("SATELLITE ANALYSIS — all candidates")
    for name, alphabet, report in results:
        n = len(satellite_transition_pairs(alphabet))
        direct_n = len(direct_satellite_pairs(alphabet))
        print(f"  {name}: {n} satellite transitions, {direct_n} direct satellite pairs")

    # Recommendation
    print()
    print("=" * 70)
    print("DESIGN RECOMMENDATION")
    balanced = [(n, a, r) for n, a, r in results if r["severity"] == "OK"]
    if balanced:
        # Pick the one with most satellite coverage
        best_name, best_alpha, best_r = max(balanced,
            key=lambda x: x[2]["orbit_fractions"]["satellite"])
        print(f"  Best balanced candidate: {best_name}")
        print(f"  Satellite coverage: {best_r['orbit_fractions']['satellite']*100:.1f}%")
        print(f"  Singularity load:   {best_r['orbit_fractions']['singularity']*100:.1f}%")
        print()
        print("  Recommended STATE_ALPHABET for eeg_orbit_classifier.py:")
        print("  STATE_ALPHABET = {")
        for la, (b, e) in best_alpha.items():
            orb = orbit_family(int(b), int(e), m=MODULUS)
            print(f"      \"{la}\": ({b:2d}, {e:2d}),  # {orb}")
        print("  }")
        print()
        print("  Use TRANSITION encoding (b from class_t, e from class_{t+1})")
        print("  to activate satellite channel across microstate transitions.")
    else:
        print("  No BALANCED candidate found. All candidates have satellite issues.")
        print("  Run with --suggest to see the auto-suggested balanced alphabet.")

    if "--suggest" in args:
        print()
        print("=" * 70)
        print("AUTO-SUGGESTED ALPHABETS")
        for n_labels in (4, 6):
            from qa_observer_alphabet_audit import suggest_balanced_alphabet
            suggested = suggest_balanced_alphabet(MODULUS, n_labels)
            print(f"\n  {n_labels}-label suggestion:")
            for lbl, (b, e) in suggested.items():
                orb = orbit_family(b, e, m=MODULUS)
                print(f"    {lbl}: (b={b:2d}, e={e:2d})  orbit={orb}")


if __name__ == "__main__":
    main()
