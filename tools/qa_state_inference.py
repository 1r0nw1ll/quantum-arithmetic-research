#!/usr/bin/env python3
"""QA State Inference — derive a system's internal (b, e) from physical measurements.

Given two or more independent observables that map to QA elements (W, F, C, D,
K, etc.), scan all integer (b, e) candidates in a range and select the one
whose QA-predicted values converge closest to the measurements.

Method:
    1. For each (b, e) in {1..b_max} × {1..e_max}, compute the QA element table.
    2. For each observable, compute the relative error |predicted − measured| / measured.
    3. Rank candidates by the maximum relative error across all observables.
    4. Report the top-k candidates with their per-observable errors.

The method is general: any QA element computable from (b, e, d, a) can serve
as an observable. The user supplies a list of (element_name, measured_value)
pairs. The tool does the rest.

Worked example (built-in):
    The Sixto Ramos machine has:
        - Outer W-radius measured at 273.3 → QA element W = d·(e + a)
        - Peak amplitude measured at 153.5 → QA element F = a·b
        - Dip amplitude measured at 152.5 → QA element F (symmetric)
    Inference result: (b, e) = (9, 4), Phibonacci family, all errors < 0.4%.

QA axiom compliance:
    A1: state space {1..m}; all QA ops use ((x-1) mod m) + 1
    A2: d = b+e, a = b+2e (derived, never assigned independently)
    T2: float comparison is observer-layer measurement; never fed into QA state
    S1: b*b not b**2 throughout
    S2: (b, e) are integers; float errors are observer projections
    T1: not applicable (no temporal dynamics in inference)

Will Dale + Claude, 2026-04-11.
"""

QA_COMPLIANCE = "observer=state_inference_tool, state_alphabet=qa_integer_scan, tier=convergent_multi_observable_inference"

import json
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# QA element table (integer-only, S1/A2-compliant)
# -----------------------------------------------------------------------------

def qa_mod(x: int, m: int) -> int:
    """A1: result in {1..m}."""
    return ((int(x) - 1) % m) + 1


def eisenstein_norm(b: int, e: int) -> int:
    """f(b,e) = b*b + b*e - e*e. S1: no ** operator."""
    return b * b + b * e - e * e


def qa_element_table(b: int, e: int) -> Dict[str, int]:
    """Compute all canonical QA elements from (b, e). Integer-only.

    A2: d and a are always derived as d = b+e, a = b+2e.
    S1: all squaring uses x*x, never x**2.
    """
    d = b + e       # A2: d always derived
    a = b + 2 * e   # A2: a always derived

    D = d * d
    X = e * d
    J = b * d
    K = d * a
    W = d * (e + a)  # = X + K
    P = 2 * W

    F = a * b
    C = 2 * e * d
    G = D + e * e
    H = C + F
    I_elem = abs(C - F)
    L = (C * F) // 12 if (C * F) % 12 == 0 else C * F / 12
    Y = a * a - D
    Z = e * e + a * d

    return {
        "b": b, "e": e, "d": d, "a": a,
        "D": D, "X": X, "J": J, "K": K,
        "W": W, "P": P, "F": F, "C": C,
        "G": G, "H": H, "I": I_elem, "L": L,
        "Y": Y, "Z": Z,
        "F_over_C": F / C if C != 0 else 0,
        "eisenstein_norm": eisenstein_norm(b, e),
        "eisenstein_norm_mod9": eisenstein_norm(b, e) % 9,
    }


# -----------------------------------------------------------------------------
# Orbit family classification
# -----------------------------------------------------------------------------

NORM_PAIR_TO_FAMILY = {
    frozenset({1, 8}): "Fibonacci",
    frozenset({4, 5}): "Lucas",
    frozenset({2, 7}): "Phibonacci",
    frozenset({0}):    "Tribonacci/Ninbonacci (null)",
}


def classify_family(b: int, e: int, m: int = 9) -> str:
    """Classify (b, e) into its Pythagorean Family on S_m by Eisenstein norm mod m."""
    norm_mod = eisenstein_norm(b, e) % m
    for pair, name in NORM_PAIR_TO_FAMILY.items():
        if norm_mod in pair:
            return name
    return f"unknown (norm mod {m} = {norm_mod})"


def orbit_length(b: int, e: int, m: int = 9) -> int:
    """Compute T-orbit length of (b, e) on S_m."""
    seen = set()
    cur = (b, e)
    while cur not in seen:
        seen.add(cur)
        cur = (cur[1], qa_mod(cur[0] + cur[1], m))
    return len(seen)


# -----------------------------------------------------------------------------
# Core inference engine
# -----------------------------------------------------------------------------

def infer_qa_state(
    observables: List[Tuple[str, float]],
    b_max: int = 14,
    e_max: int = 14,
    top_k: int = 5,
    m: int = 9,
) -> Dict[str, Any]:
    """Infer the best-matching (b, e) from physical measurements.

    Parameters
    ----------
    observables : list of (element_name, measured_value)
        Each pair maps a QA element name (e.g., "W", "F", "C") to a measured
        float value. The element name must match a key in qa_element_table().
    b_max, e_max : int
        Search range for b and e (inclusive).
    top_k : int
        Number of top candidates to report.
    m : int
        Modulus for orbit classification (default 9).

    Returns
    -------
    dict with keys: candidates (list), best (dict), observables (list),
    search_range (dict).
    """
    if not observables:
        raise ValueError("at least one observable required")

    for name, val in observables:
        if val == 0:
            raise ValueError(f"observable {name!r} has value 0 — cannot compute relative error")

    candidates = []

    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            table = qa_element_table(b, e)

            errors = {}
            max_error = 0.0
            valid = True

            for name, measured in observables:
                if name not in table:
                    valid = False
                    break
                predicted = float(table[name])
                if measured == 0:
                    valid = False
                    break
                rel_error = abs(predicted - measured) / abs(measured)
                errors[name] = {
                    "predicted": predicted,
                    "measured": measured,
                    "abs_error": round(abs(predicted - measured), 4),
                    "rel_error_pct": round(rel_error * 100, 4),
                }
                if rel_error > max_error:
                    max_error = rel_error

            if not valid:
                continue

            candidates.append({
                "b": b,
                "e": e,
                "d": b + e,
                "a": b + 2 * e,
                "max_rel_error_pct": round(max_error * 100, 4),
                "errors": errors,
                "family": classify_family(b, e, m),
                "orbit_length": orbit_length(b, e, m),
                "eisenstein_norm": eisenstein_norm(b, e),
                "norm_mod_m": eisenstein_norm(b, e) % m,
            })

    candidates.sort(key=lambda c: c["max_rel_error_pct"])
    top = candidates[:top_k]

    return {
        "observables": [{"element": n, "measured": v} for n, v in observables],
        "search_range": {"b_max": b_max, "e_max": e_max, "modulus": m},
        "num_candidates_scanned": b_max * e_max,
        "top_k": top_k,
        "candidates": top,
        "best": top[0] if top else None,
        "convergent": (
            top[0]["max_rel_error_pct"] < 1.0 if top else False
        ),
    }


# -----------------------------------------------------------------------------
# Built-in Sixto worked example
# -----------------------------------------------------------------------------

def sixto_demo() -> Dict[str, Any]:
    """Run the Sixto Ramos inference as a built-in demo.

    Observables:
        W (outer radius) = 273.3 (measured mean of 274.08, 269.76, 275.4, 273.96)
        F (peak amplitude) = 153.5 (measured peak of non-anomalous curves)

    Expected result: (b, e) = (9, 4), Phibonacci, max error < 0.4%.
    """
    return infer_qa_state(
        observables=[
            ("W", 273.3),
            ("F", 153.5),
        ],
        b_max=14,
        e_max=14,
        top_k=5,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="QA State Inference — derive (b, e) from physical measurements"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the built-in Sixto Ramos demo"
    )
    parser.add_argument(
        "--obs", nargs=2, action="append", metavar=("ELEMENT", "VALUE"),
        help="Observable: --obs W 273.3 --obs F 153.5"
    )
    parser.add_argument("--b-max", type=int, default=14)
    parser.add_argument("--e-max", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.demo:
        result = sixto_demo()
    elif args.obs:
        observables = [(name, float(val)) for name, val in args.obs]
        result = infer_qa_state(observables, args.b_max, args.e_max, args.top_k)
    else:
        parser.print_help()
        return

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Scanned {result['num_candidates_scanned']} candidates")
        print(f"Convergent: {result['convergent']}")
        print()
        best = result["best"]
        if best:
            print(f"BEST: (b, e) = ({best['b']}, {best['e']})")
            print(f"  d = {best['d']}, a = {best['a']}")
            print(f"  Family: {best['family']}")
            print(f"  Orbit length: {best['orbit_length']}")
            print(f"  Eisenstein norm: {best['eisenstein_norm']} (mod 9 = {best['norm_mod_m']})")
            print(f"  Max relative error: {best['max_rel_error_pct']:.4f}%")
            print()
            for elem, info in best["errors"].items():
                print(f"  {elem}: predicted={info['predicted']}, measured={info['measured']}, "
                      f"error={info['abs_error']} ({info['rel_error_pct']:.4f}%)")
            print()

            # Print full element table for best
            table = qa_element_table(best["b"], best["e"])
            print("  Full element table:")
            for k in ("b", "e", "d", "a", "D", "X", "J", "K", "W", "P",
                      "F", "C", "G", "H", "I", "Y", "Z"):
                print(f"    {k} = {table[k]}")

            # Pythagorean triple check
            c_val, f_val, g_val = table["C"], table["F"], table["G"]
            pyth_check = c_val * c_val + f_val * f_val == g_val * g_val
            print(f"    Pythagorean: C² + F² = {c_val*c_val} + {f_val*f_val} "
                  f"= {c_val*c_val + f_val*f_val}, G² = {g_val*g_val} "
                  f"{'✓' if pyth_check else '✗'}")
        else:
            print("No candidates found.")

        if len(result["candidates"]) > 1:
            print(f"\nOther top candidates:")
            for c in result["candidates"][1:]:
                print(f"  (b={c['b']}, e={c['e']}): max_error={c['max_rel_error_pct']:.4f}%, "
                      f"family={c['family']}")


if __name__ == "__main__":
    main()
