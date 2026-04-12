#!/usr/bin/env python3
"""
qa_fibonacci_hypergraph_cert_validate.py

Validator for QA_FIBONACCI_HYPERGRAPH_CERT.v1  [family 212]

Certifies: Every QA state (b,e) defines a length-4 Fibonacci window hyperedge
(b, e, d, a) with d = b+e, a = b+2e (both A1-adjusted mod m). The resulting
state-residue incidence hypergraph H(m) on vertex set {1,...,m} satisfies
three structural theorems on S_9.

Theorem 1 (Sliding Window):
    For every s in S_m, H(T(s)) = (e_s, d_s, a_s, (d_s + a_s) mod m).
    Equivalently, T acts on hyperedges as a 1-step Fibonacci window shift:
    drop the first element, append F_{k+4} = F_{k+2} + F_{k+3}.
    On S_9: 81/81 states match.

Theorem 2 (Uniform Vertex Degree):
    Every residue v in {1,...,m} appears in exactly 4m hyperedges.
    Proof: for each of the 4 positions (b, e, d, a), the set of states that
    place v in that position has size m (specific b or e values; for d = b+e
    and a = b+2e, one equation per b or e choice). Summing gives 4m, uniform.
    On S_9: every v has degree 36, total degree 324 = 4 * 9^2.

Theorem 3 (Orbit-Multiset Collapse):
    On S_9, the five T-orbits (sizes 24, 24, 24, 8, 1) produce distinct
    multiset hyperedges in counts (22, 22, 22, 4, 1) respectively.
    Cosmos orbits hit 22 of 24 possible multisets (2 collisions each).
    The satellite orbit hits 4 of 8 (2-to-1 collapse from period symmetry).
    The singularity is (9,9,9,9), a single multiset.

Source grounding:
    - Fibonacci (Leonardo of Pisa), "Liber Abaci" (1202) — original recurrence
    - Edouard Lucas, "Theorie des Fonctions Numeriques Simplement Periodiques"
      (American Journal of Mathematics, 1878) — periods of F_n mod m
    - D. D. Wall, "Fibonacci Series Modulo m" (American Mathematical Monthly,
      1960) — the Pisano period function pi(m)
    - Claude Berge, "Hypergraphs: Combinatorics of Finite Sets" (1989) —
      hypergraph theory foundations
    - Prerequisite cert: [191] qa_bateson_learning_levels_cert
    - Related cert: [192] qa_dual_extremality_24_cert (orbit length = Pisano period)
    - Related cert: [211] qa_cayley_bateson_filtration_cert (same T, graph view)

Checks:
    HGR_1     — schema_version matches
    HGR_SLIDE — sliding window theorem 81/81 on S_9 (independently recomputed)
    HGR_DEG   — vertex degree uniform at 4m = 36, total 324
    HGR_ORB   — orbit-multiset distribution matches (22, 22, 22, 4, 1)
    HGR_FIB   — declared Fibonacci recurrence formula present
    HGR_191   — cross-reference to family 191 present
    HGR_SRC   — source attribution to Fibonacci and Berge present
    HGR_WIT   — witnesses declared (cosmos, satellite, singularity)
    HGR_F     — fail_ledger well-formed

QA axiom compliance: integer state alphabet {1..m}, A1-compliant mod,
A2-derived d and a, no floats, integer path time.
"""

QA_COMPLIANCE = "cert_validator — validates finite S_9 Fibonacci hypergraph structure; integer state space; no observer, no floats, no continuous dynamics"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_FIBONACCI_HYPERGRAPH_CERT.v1"

# Theorem constants on S_9 — recomputed independently in validator, not trusted.
EXPECTED_VERTEX_DEGREE_S9 = 36   # = 4 * 9
EXPECTED_TOTAL_DEGREE_S9 = 324   # = 4 * 81
EXPECTED_ORBIT_MULTISETS_S9 = [
    {"length": 24, "distinct_multisets": 22},
    {"length": 24, "distinct_multisets": 22},
    {"length": 24, "distinct_multisets": 22},
    {"length": 8,  "distinct_multisets": 4},
    {"length": 1,  "distinct_multisets": 1},
]


# -----------------------------------------------------------------------------
# QA primitives (integer-only, A1-compliant) — stdlib only
# -----------------------------------------------------------------------------

def qa_mod(x, m):
    """A1: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m):
    """T: (b,e) -> (e, (b+e) mod m), A1-adjusted."""
    return (e, qa_mod(b + e, m))


def qa_hyperedge(b, e, m):
    """Length-4 Fibonacci window hyperedge (b, e, d, a) for state (b,e).

    A2: d = b+e, a = b+2e derived (never assigned independently).
    """
    d = b + e
    a = b + 2 * e
    return (b, e, qa_mod(d, m), qa_mod(a, m))


def t_shift_hyperedge(h, m):
    """Sliding window shift: (F0, F1, F2, F3) -> (F1, F2, F3, (F2+F3) mod m)."""
    _f0, f1, f2, f3 = h
    f4 = qa_mod(f2 + f3, m)
    return (f1, f2, f3, f4)


def enumerate_state_space(m):
    return [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)]


# -----------------------------------------------------------------------------
# Independent theorem recomputation on S_m
# -----------------------------------------------------------------------------

def compute_sliding_window(m):
    """Return (checked, matched) over all of S_m."""
    checked = 0
    matched = 0
    for b, e in enumerate_state_space(m):
        checked += 1
        h = qa_hyperedge(b, e, m)
        predicted = t_shift_hyperedge(h, m)
        actual = qa_hyperedge(*qa_step(b, e, m), m)
        if predicted == actual:
            matched += 1
    return checked, matched


def compute_vertex_degrees(m):
    """Return dict {v: degree} for the state-residue incidence hypergraph."""
    degrees = {v: 0 for v in range(1, m + 1)}
    for b, e in enumerate_state_space(m):
        h = qa_hyperedge(b, e, m)
        for v in h:
            degrees[v] += 1
    return degrees


def compute_orbit_multiset_stats(m):
    """For each T-orbit of S_m, return (length, distinct_multisets)."""
    seen_global = set()
    results = []
    for start in enumerate_state_space(m):
        if start in seen_global:
            continue
        orbit = []
        cur = start
        cur_set = set()
        while cur not in cur_set:
            cur_set.add(cur)
            orbit.append(cur)
            cur = qa_step(cur[0], cur[1], m)
        seen_global.update(orbit)
        multisets = set()
        for b, e in orbit:
            multisets.add(tuple(sorted(qa_hyperedge(b, e, m))))
        results.append({"length": len(orbit), "distinct_multisets": len(multisets)})
    return results


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # HGR_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"HGR_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # HGR_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("HGR_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("HGR_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # HGR_SRC
    src = str(cert.get("source_attribution", ""))
    if "Fibonacci" not in src:
        warnings.append("HGR_SRC: source_attribution should credit Fibonacci")
    if "Berge" not in src:
        warnings.append("HGR_SRC: source_attribution should credit Claude Berge (hypergraph theory)")

    # HGR_191: prerequisite
    xrefs = cert.get("cross_references", [])
    has_191 = isinstance(xrefs, list) and any(
        isinstance(x, dict) and x.get("family") == 191 for x in xrefs
    )
    if not has_191:
        errors.append("HGR_191: missing cross_reference to family 191 (prerequisite)")

    # HGR_SLIDE: recompute sliding window on S_9
    checked, matched = compute_sliding_window(9)
    if checked != 81 or matched != 81:
        errors.append(
            f"HGR_SLIDE: sliding window theorem recomputation gave {matched}/{checked} on S_9 (expected 81/81)"
        )
    decl_slide = cert.get("sliding_window_s9", {})
    if decl_slide:
        if decl_slide.get("checked") != 81 or decl_slide.get("matched") != 81:
            errors.append(
                f"HGR_SLIDE: declared sliding_window_s9 {decl_slide} != expected checked=81 matched=81"
            )

    # HGR_DEG: vertex degrees
    degrees = compute_vertex_degrees(9)
    if any(d != EXPECTED_VERTEX_DEGREE_S9 for d in degrees.values()):
        errors.append(
            f"HGR_DEG: vertex degrees not uniform: {degrees}"
        )
    total = sum(degrees.values())
    if total != EXPECTED_TOTAL_DEGREE_S9:
        errors.append(
            f"HGR_DEG: total degree {total} != expected {EXPECTED_TOTAL_DEGREE_S9}"
        )
    decl_deg = cert.get("vertex_degree_s9", {})
    if decl_deg:
        if decl_deg.get("per_vertex") != EXPECTED_VERTEX_DEGREE_S9:
            errors.append(
                f"HGR_DEG: declared per_vertex {decl_deg.get('per_vertex')} != expected {EXPECTED_VERTEX_DEGREE_S9}"
            )
        if decl_deg.get("total") != EXPECTED_TOTAL_DEGREE_S9:
            errors.append(
                f"HGR_DEG: declared total {decl_deg.get('total')} != expected {EXPECTED_TOTAL_DEGREE_S9}"
            )

    # HGR_ORB: orbit-multiset distribution
    actual_orb = compute_orbit_multiset_stats(9)
    actual_sorted = sorted(
        ((o["length"], o["distinct_multisets"]) for o in actual_orb), reverse=True
    )
    expected_sorted = sorted(
        ((o["length"], o["distinct_multisets"]) for o in EXPECTED_ORBIT_MULTISETS_S9),
        reverse=True,
    )
    if actual_sorted != expected_sorted:
        errors.append(
            f"HGR_ORB: orbit-multiset distribution {actual_sorted} != expected {expected_sorted}"
        )
    decl_orb = cert.get("orbit_multisets_s9", [])
    if decl_orb:
        decl_sorted = sorted(
            ((int(o.get("length", -1)), int(o.get("distinct_multisets", -1))) for o in decl_orb),
            reverse=True,
        )
        if decl_sorted != expected_sorted:
            errors.append(
                f"HGR_ORB: declared orbit_multisets_s9 {decl_sorted} != expected {expected_sorted}"
            )

    # HGR_FIB: formula string present
    fib_formula = cert.get("fibonacci_recurrence")
    if not fib_formula or "d" not in str(fib_formula) or "+" not in str(fib_formula):
        warnings.append(
            "HGR_FIB: fibonacci_recurrence field should express F_{k+1} = F_{k-1} + F_k (involving d and a)"
        )

    # HGR_WIT: witnesses
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 3:
        errors.append(
            f"HGR_WIT: need >= 3 witnesses (cosmos, satellite, singularity), got {len(witnesses) if isinstance(witnesses, list) else 'none'}"
        )

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("hgr_pass_hypergraph.json", True),
        ("hgr_fail_bad_degree.json", False),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Fibonacci Hypergraph Cert [212] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
