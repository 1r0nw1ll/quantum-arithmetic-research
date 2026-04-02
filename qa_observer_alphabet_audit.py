#!/usr/bin/env python3
"""
qa_observer_alphabet_audit.py — Observer Alphabet Coverage Audit

QA_COMPLIANCE = "design_tool_not_empirical_script"


Usage:
    python qa_observer_alphabet_audit.py                    # audit built-in alphabets
    python qa_observer_alphabet_audit.py --suggest 24 8     # suggest an 8-label mod-24 alphabet

For every declared state alphabet, reports:
  - All (b,e) pairs and their orbit family
  - Orbit family coverage: singularity / satellite / cosmos reachable?
  - Diagonal singularity load (b=e pairs)
  - Algebraic singularity load (b≠e but v₃(f)≥2)
  - Whether satellite is reachable at all
  - Design recommendation

This is a design tool, not a compliance gate. Run it before finalising a
new observer alphabet to catch dead orbit channels early.
"""

import sys
import itertools
import json
from typing import NamedTuple

from qa_orbit_rules import norm_f, v3, orbit_family, qa_step


# ── Alphabet audit ─────────────────────────────────────────────────────────────

class PairResult(NamedTuple):
    label_a: str
    label_b: str
    b: int
    e: int
    f: int
    v3_val: int
    orbit: str
    diagonal: bool


def audit_alphabet(name: str,
                   alphabet: dict[str, tuple[int, int]],
                   modulus: int) -> dict:
    """
    Audit a state alphabet for orbit coverage.

    alphabet: dict mapping label → (b, e) integer pair
    Returns a report dict.
    """
    labels = list(alphabet.keys())
    pairs: list[PairResult] = []

    # All ordered pairs (label_a, label_b) — these are the QA states reachable
    # when classifying (signal_t → label_a, signal_{t+1} → label_b) or
    # when a single-element alphabet assigns b from one label, e from another.
    for la, lb in itertools.product(labels, repeat=2):
        b = alphabet[la][0]
        e = alphabet[lb][1] if len(alphabet[lb]) > 1 else alphabet[lb][0]
        f = norm_f(b, e)
        v = v3(f)
        orb = orbit_family(b, e, m=modulus)
        diag = (b == e)
        pairs.append(PairResult(la, lb, b, e, f, v, orb, diag))

    # Also audit the declared single pairs (la, la) — the direct assignment
    direct_pairs: list[PairResult] = []
    for la in labels:
        b, e = alphabet[la]
        f = norm_f(b, e)
        v = v3(f)
        orb = orbit_family(b, e, m=modulus)
        diag = (b == e)
        direct_pairs.append(PairResult(la, la, b, e, f, v, orb, diag))

    # Orbit coverage in cross-product
    orbits_seen = {p.orbit for p in pairs}
    direct_orbits = {p.orbit for p in direct_pairs}

    n_total = len(pairs)
    orbit_counts = {
        "singularity": sum(1 for p in pairs if p.orbit == "singularity"),
        "satellite":   sum(1 for p in pairs if p.orbit == "satellite"),
        "cosmos":      sum(1 for p in pairs if p.orbit == "cosmos"),
    }

    diagonal_load = sum(1 for p in pairs if p.diagonal) / n_total
    algebraic_sing = sum(1 for p in pairs if p.orbit == "singularity" and not p.diagonal) / n_total

    satellite_pairs = [p for p in pairs if p.orbit == "satellite"]

    # Design verdict
    if "satellite" not in orbits_seen:
        verdict = "DEAD_SATELLITE — alphabet cannot produce satellite orbit"
        severity = "WARN"
    elif orbit_counts["satellite"] / n_total < 0.05:
        verdict = "SPARSE_SATELLITE — satellite reachable but <5% of pairs"
        severity = "WARN"
    elif orbit_counts["singularity"] / n_total > 0.5:
        verdict = "SINGULARITY_HEAVY — >50% of pairs are singularity"
        severity = "WARN"
    else:
        verdict = "BALANCED — all three orbit families represented"
        severity = "OK"

    return {
        "name": name,
        "modulus": modulus,
        "n_labels": len(labels),
        "n_pairs": n_total,
        "orbit_counts": orbit_counts,
        "orbit_fractions": {k: v / n_total for k, v in orbit_counts.items()},
        "orbits_reachable": sorted(orbits_seen),
        "direct_orbits": sorted(direct_orbits),
        "diagonal_load": diagonal_load,
        "algebraic_singularity_load": algebraic_sing,
        "satellite_pairs": [(p.label_a, p.label_b, p.b, p.e) for p in satellite_pairs[:10]],
        "verdict": verdict,
        "severity": severity,
    }


def print_report(report: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Alphabet: {report['name']}  (mod {report['modulus']}, {report['n_labels']} labels)")
    print(f"{'='*60}")
    print(f"Cross-product pairs: {report['n_pairs']}")
    print(f"Orbits reachable (cross-product): {report['orbits_reachable']}")
    print(f"Orbits reachable (direct):         {report['direct_orbits']}")
    print()
    print("Orbit distribution:")
    for orb in ("singularity", "satellite", "cosmos"):
        n = report["orbit_counts"][orb]
        frac = report["orbit_fractions"][orb]
        bar_width = round(frac * 30)
        bar = "█" * bar_width
        print(f"  {orb:15s}  {n:4d} pairs  ({frac*100:5.1f}%)  {bar}")
    print()
    print(f"Diagonal singularity load:    {report['diagonal_load']*100:.1f}%  (b=e pairs)")
    print(f"Algebraic singularity load:   {report['algebraic_singularity_load']*100:.1f}%  (b≠e but v₃≥2)")
    print()

    if report["satellite_pairs"]:
        print(f"Satellite pairs (up to 10):")
        for la, lb, b, e in report["satellite_pairs"]:
            print(f"  ({la}, {lb})  b={b} e={e}  f={norm_f(b,e)}  v₃={v3(norm_f(b,e))}")
    else:
        print("Satellite pairs: NONE")
    print()

    icon = "✓" if report["severity"] == "OK" else "⚠"
    print(f"{icon} {report['verdict']}")


# ── Alphabet suggestion ────────────────────────────────────────────────────────

def suggest_balanced_alphabet(modulus: int, n_labels: int) -> dict[str, tuple[int, int]]:
    """
    Suggest an n_labels state alphabet for the given modulus that achieves
    satellite coverage by scanning the state space for satellite-producing pairs.

    Returns a dict: label → (b, e) such that the alphabet includes at least
    one satellite pair when cross-producted.
    """
    # Find values in {1,...,modulus} such that at least some cross-products give satellite.
    # Satellite condition: (m//3)|b AND (m//3)|e AND not singularity.
    # Note: v3(norm_f(b,e))==1 is algebraically impossible — see qa_orbit_rules.py.
    satellite_values = set()
    for b in range(1, modulus + 1):
        for e in range(1, modulus + 1):
            if orbit_family(b, e, m=modulus) == "satellite":
                satellite_values.add(b)
                satellite_values.add(e)

    if not satellite_values:
        return {}

    # Pick n_labels values spread across {1,...,modulus} that include satellite values
    import math
    # Start with evenly spaced
    step = modulus / n_labels
    candidates = [max(1, min(modulus, round(step * i + step / 2))) for i in range(n_labels)]

    # Replace one candidate with a satellite-producing value if none included
    sat_vals = sorted(satellite_values)
    included = set(candidates)
    if not (included & set(sat_vals)):
        # Replace the middle candidate with the first satellite value
        mid = n_labels // 2
        candidates[mid] = sat_vals[0]

    # Deduplicate (keep unique)
    seen: set[int] = set()
    unique: list[int] = []
    for v in candidates:
        while v in seen and v <= modulus:
            v += 1
        if v <= modulus:
            seen.add(v)
            unique.append(v)

    labels = [f"state_{i+1}" for i in range(len(unique))]
    return {lbl: (v, v) for lbl, v in zip(labels, unique)}


# ── Built-in alphabets ────────────────────────────────────────────────────────

SEISMIC_ALPHABET = {
    "quiet":        (9,  9),
    "p_wave":       (1,  8),
    "s_wave":       (8,  1),
    "surface_wave": (3, 16),
    "coda":         (16, 3),
    "disordered":   (7, 11),
}

FINANCE_QUINTILE_ALPHABET = {
    # Each label represents (b, e) when used as spy_q AND tlt_q independently
    "Q1": (3,  3),
    "Q2": (7,  7),
    "Q3": (12, 12),
    "Q4": (18, 18),
    "Q5": (22, 22),
}

# Finance cross-product alphabet (all 25 joint regimes)
def _build_finance_cross_alphabet():
    QMAP = {1: 3, 2: 7, 3: 12, 4: 18, 5: 22}
    result = {}
    for sq in range(1, 6):
        for tq in range(1, 6):
            result[f"SPY{sq}_TLT{tq}"] = (QMAP[sq], QMAP[tq])
    return result

FINANCE_CROSS_ALPHABET = _build_finance_cross_alphabet()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--suggest" in args:
        idx = args.index("--suggest")
        modulus = int(args[idx + 1]) if idx + 1 < len(args) else 24
        n_labels = int(args[idx + 2]) if idx + 2 < len(args) else 6
        print(f"\nSuggested balanced alphabet (mod={modulus}, {n_labels} labels):")
        suggested = suggest_balanced_alphabet(modulus, n_labels)
        for lbl, (b, e) in suggested.items():
            print(f"  {lbl}: (b={b}, e={e})  orbit={orbit_family(b, e)}")

        # Show what satellite pairs it enables
        print("\nSatellite pairs enabled by this alphabet:")
        found = 0
        for la, (ba, _) in suggested.items():
            for lb, (_, eb) in suggested.items():
                if orbit_family(int(ba), int(eb), m=modulus) == "satellite":
                    print(f"  ({la}, {lb})  b={ba} e={eb}  orbit=satellite")
                    found += 1
        if not found:
            print("  None — try different modulus or n_labels")
        return

    # Default: audit all built-in alphabets
    print("QA Observer Alphabet Coverage Audit")
    print("Authority: QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1")

    # Seismic direct pairs
    report = audit_alphabet("seismic_wave_classes (direct pairs)", SEISMIC_ALPHABET, modulus=24)
    print_report(report)

    # Seismic cross-product (treating each label as both b-source and e-source)
    seismic_cross = {k: (v[0], v[0]) for k, v in SEISMIC_ALPHABET.items()}
    report2 = audit_alphabet("seismic_wave_classes (b=e diagonal only)", seismic_cross, modulus=24)
    # Actually just show the cross product of the b values and e values
    seismic_b_e = {k: (SEISMIC_ALPHABET[k][0], SEISMIC_ALPHABET[k][1])
                   for k in SEISMIC_ALPHABET}
    report3 = audit_alphabet("seismic_wave_classes (declared b,e pairs)", seismic_b_e, modulus=24)
    print_report(report3)

    # Finance quintile cross-product
    report4 = audit_alphabet("finance_quintile_cross (25 joint regimes)",
                              FINANCE_CROSS_ALPHABET, modulus=24)
    print_report(report4)

    # Suggestion: what values would give satellite coverage?
    print(f"\n{'='*60}")
    print("Satellite-producing (b,e) pairs in mod-24:")
    count = 0
    for b in range(1, 25):
        for e in range(1, 25):
            if orbit_family(b, e, m=24) == "satellite":
                if count < 15:
                    f_val = norm_f(b, e)
                    print(f"  b={b:2d} e={e:2d}  f={f_val:5d}  orbit=satellite")
                count += 1
    print(f"  ... {count} total satellite pairs in mod-24 state space")

    print(f"\n{'='*60}")
    print("Design recommendation:")
    print("  To activate satellite channel, the observer alphabet must include")
    print("  at least one pair (b,e) where (m//3)|b AND (m//3)|e  (i.e. 8|b AND 8|e for mod-24).")
    print("  Run with --suggest 24 6 to see a balanced alphabet suggestion.")


if __name__ == "__main__":
    main()
