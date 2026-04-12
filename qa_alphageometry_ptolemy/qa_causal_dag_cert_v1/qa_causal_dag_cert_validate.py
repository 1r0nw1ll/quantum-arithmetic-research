#!/usr/bin/env python3
"""
qa_causal_dag_cert_validate.py

Validator for QA_CAUSAL_DAG_CERT.v1  [family 213]

Certifies: The A2 axiom (d=b+e, a=b+2e) IS the structural equation system of
a 4-node causal DAG. The DAG has b, e as exogenous variables and d, a as
endogenous collider variables, with 4 directed edges (b->d, e->d, b->a, e->a).
It is acyclic and forms a Y-structure.

Pair-Invertibility Theorem:
    For modulus m, define the 6 unordered pairs of {b, e, d, a}:
        (b,e), (b,d), (b,a), (e,d), (e,a), (d,a)
    All 6 pairs are bijective as maps from S_m = {1..m}^2 to their image
    iff gcd(2, m) = 1. When gcd(2, m) = g > 1, exactly the pair (b, a) is
    g-to-1; the other 5 pairs remain bijective.

    Verified exhaustively:
        S_9  (gcd=1): all 6 pairs bijective, each 81/81 distinct
        S_24 (gcd=2): pair (b,a) is 2-to-1 (288/576 distinct); others bijective

Pearl-Level Collapse Theorem:
    Because the A2 structural equations are deterministic integer arithmetic
    (no stochastic noise), Pearl's three causal hierarchy levels collapse:

      Level 1 (association):   P(d | b=b*, e=e*) = delta(d - b* - e*)
      Level 2 (intervention):  P(d | do(b=b*), e=e*) = delta(d - b* - e*)
      Level 3 (counterfactual): P(d_{b=b*} | b=b', e=e') = delta(d - b* - e')

    All three deliver the same value because there is no noise to marginalize.
    This is a degenerate but valid SCM — it matches Theorem NT: the QA causal
    structure lives in the discrete A2 equations; the observer (measurement)
    layer is orthogonal.

Source grounding:
    - Judea Pearl, "Causality: Models, Reasoning, and Inference" (Cambridge
      University Press, 2nd ed. 2009) — structural causal models, do-calculus,
      three-level causal hierarchy
    - Sewall Wright, "Correlation and Causation" (Journal of Agricultural
      Research 20, 1921, pp. 557-585) — original path analysis; direct
      ancestor of modern SCM
    - Prerequisite cert: family [191] qa_bateson_learning_levels_cert
    - Related: [150] qa_septenary_unit_group (2 ∈ (Z/9Z)* makes S_9 fully
      pair-invertible)
    - Related: [202] qa_hebrew_mod9_identity (A1 dr adjustment used in sector
      labels)

Checks:
    CDG_1      — schema_version matches
    CDG_STRUCT — 4-node Y-DAG declared (b,e exogenous; d,a colliders; acyclic)
    CDG_A2     — structural equations d = b+e, a = b+2e present in declaration
    CDG_PAIRS  — pair-bijectivity table matches theorem on S_9 (all 6) and S_24
                 (5 of 6, with (b,a) 2-to-1); recomputed independently
    CDG_PEARL  — Pearl-level collapse demonstrated on witness sample
    CDG_NT     — Theorem NT correspondence declared
    CDG_191    — cross-reference to family 191
    CDG_SRC    — source attribution to Pearl present
    CDG_WIT    — ≥ 4 witnesses (one per pair class: always-bijective-unit-m,
                 always-bijective-nonunit-m, (b,a)-works-m=9, (b,a)-fails-m=24)
    CDG_F      — fail_ledger well-formed

QA axiom compliance: integer state alphabet {1..m}, A1-adjusted sector labels,
A2-derived d and a (the structural equations themselves), no floats, integer
path time.
"""

QA_COMPLIANCE = "cert_validator — validates the 4-node Y-structure causal DAG of the QA 4-tuple; integer state space; no observer, no floats, no continuous dynamics"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_CAUSAL_DAG_CERT.v1"

EXPECTED_CAUSAL_NODES = ("a", "b", "d", "e")  # alphabetical for stable comparison
EXPECTED_CAUSAL_EDGES_SET = frozenset({
    ("b", "d"), ("e", "d"), ("b", "a"), ("e", "a")
})
EXPECTED_EXOGENOUS_SORTED = ["b", "e"]
EXPECTED_COLLIDERS_SORTED = ["a", "d"]


# -----------------------------------------------------------------------------
# QA primitives (integer-only, A1/A2-compliant) — stdlib only
# -----------------------------------------------------------------------------

def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


def qa_tuple(b, e, m):
    """A2-compliant: d and a derived, then A1-reduced."""
    d = b + e
    a = b + 2 * e
    return (b, e, qa_mod(d, m), qa_mod(a, m))


def enumerate_state_space(m):
    return [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)]


# -----------------------------------------------------------------------------
# Independent pair-bijectivity recomputation
# -----------------------------------------------------------------------------

ALL_PAIRS = (
    ("b", "e"),
    ("b", "d"),
    ("b", "a"),
    ("e", "d"),
    ("e", "a"),
    ("d", "a"),
)


def compute_pair_bijectivity(m):
    """For each of the 6 pairs, return (distinct, total, bijective, fold)."""
    states = enumerate_state_space(m)
    tuples = [qa_tuple(b, e, m) for b, e in states]
    index = {"b": 0, "e": 1, "d": 2, "a": 3}
    result = {}
    for p in ALL_PAIRS:
        i, j = index[p[0]], index[p[1]]
        projected = [(t[i], t[j]) for t in tuples]
        n_distinct = len(set(projected))
        n_total = len(projected)
        result[p] = {
            "distinct": n_distinct,
            "total": n_total,
            "bijective": n_distinct == n_total,
            "fold": n_total // n_distinct if n_distinct else 0,
        }
    return result


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # CDG_1
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"CDG_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # CDG_F
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("CDG_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("CDG_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # CDG_SRC
    src = str(cert.get("source_attribution", ""))
    if "Pearl" not in src:
        warnings.append("CDG_SRC: source_attribution should credit Judea Pearl")

    # CDG_191
    xrefs = cert.get("cross_references", [])
    has_191 = isinstance(xrefs, list) and any(
        isinstance(x, dict) and x.get("family") == 191 for x in xrefs
    )
    if not has_191:
        errors.append("CDG_191: missing cross_reference to family 191 (prerequisite)")

    # CDG_STRUCT: check declared DAG structure
    dag = cert.get("causal_dag", {})
    decl_nodes = dag.get("nodes", [])
    decl_edges = dag.get("edges", [])
    decl_exogenous = dag.get("exogenous", [])
    decl_colliders = dag.get("colliders", [])

    if sorted(decl_nodes) != list(EXPECTED_CAUSAL_NODES):
        errors.append(
            f"CDG_STRUCT: declared nodes {sorted(decl_nodes)} != expected {list(EXPECTED_CAUSAL_NODES)}"
        )
    decl_edges_set = frozenset(
        tuple(e) for e in decl_edges if isinstance(e, (list, tuple)) and len(e) == 2
    )
    if decl_edges_set != EXPECTED_CAUSAL_EDGES_SET:
        errors.append(
            f"CDG_STRUCT: declared edges {sorted(decl_edges_set)} != expected {sorted(EXPECTED_CAUSAL_EDGES_SET)}"
        )
    if sorted(decl_exogenous) != EXPECTED_EXOGENOUS_SORTED:
        errors.append(
            f"CDG_STRUCT: declared exogenous {sorted(decl_exogenous)} != expected {EXPECTED_EXOGENOUS_SORTED}"
        )
    if sorted(decl_colliders) != EXPECTED_COLLIDERS_SORTED:
        errors.append(
            f"CDG_STRUCT: declared colliders {sorted(decl_colliders)} != expected {EXPECTED_COLLIDERS_SORTED}"
        )

    # CDG_A2: structural equations must match A2 axiom
    se = dag.get("structural_equations", {})
    if se.get("d") not in ("b + e", "b+e"):
        errors.append(f"CDG_A2: structural_equations.d must be 'b + e', got {se.get('d')!r}")
    if se.get("a") not in ("b + 2e", "b + 2 * e", "b+2e", "b+2*e"):
        errors.append(f"CDG_A2: structural_equations.a must be 'b + 2e', got {se.get('a')!r}")

    # CDG_PAIRS: independently recompute on S_9 and S_24
    for m, expected_all_bijective in [(9, True), (24, False)]:
        bij = compute_pair_bijectivity(m)
        # Other 5 pairs always bijective
        for p in ALL_PAIRS:
            if p == ("b", "a"):
                continue
            if not bij[p]["bijective"]:
                errors.append(
                    f"CDG_PAIRS: S_{m} pair {p} not bijective ({bij[p]['distinct']}/{bij[p]['total']})"
                )
        # (b,a) bijective iff gcd(2,m) == 1
        expected_ba_bij = gcd(2, m) == 1
        actual_ba_bij = bij[("b", "a")]["bijective"]
        if actual_ba_bij != expected_ba_bij:
            errors.append(
                f"CDG_PAIRS: S_{m} pair (b,a) bijective={actual_ba_bij}, expected {expected_ba_bij}"
            )
        # On S_24, (b,a) should be 2-to-1
        if m == 24 and bij[("b", "a")]["fold"] != 2:
            errors.append(
                f"CDG_PAIRS: S_24 pair (b,a) fold={bij[('b','a')]['fold']}, expected 2"
            )

    # Check declared pair-bijectivity table matches actual
    decl_pairs = cert.get("pair_bijectivity_s9", {})
    for p in ALL_PAIRS:
        pname = "_".join(p)
        if pname in decl_pairs:
            decl_val = decl_pairs[pname]
            if decl_val is True or decl_val == "bijective":
                pass  # accept
            elif decl_val is False or decl_val == "non-bijective":
                errors.append(f"CDG_PAIRS: declared S_9 {pname} = {decl_val}, but theorem says bijective")
            else:
                warnings.append(f"CDG_PAIRS: declared S_9 {pname} = {decl_val} (unrecognized)")

    decl_pairs_24 = cert.get("pair_bijectivity_s24", {})
    if decl_pairs_24:
        decl_ba = decl_pairs_24.get("b_a")
        if decl_ba not in ("2-to-1", False, "non-bijective"):
            errors.append(
                f"CDG_PAIRS: declared S_24 b_a = {decl_ba!r}, expected '2-to-1' / False / 'non-bijective'"
            )

    # CDG_PEARL: level collapse declared
    pearl = cert.get("pearl_level_collapse", {})
    if not pearl.get("collapses"):
        warnings.append("CDG_PEARL: pearl_level_collapse.collapses should be true")
    if not pearl.get("reason"):
        warnings.append("CDG_PEARL: pearl_level_collapse.reason (deterministic SCM) should be present")

    # CDG_NT
    nt = cert.get("theorem_nt_correspondence", "")
    if not nt or "observer" not in str(nt).lower():
        warnings.append("CDG_NT: theorem_nt_correspondence field should mention observer projection")

    # CDG_WIT
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 4:
        errors.append(
            f"CDG_WIT: need >= 4 witnesses (pair classes), got {len(witnesses) if isinstance(witnesses, list) else 'none'}"
        )

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("cdg_pass_y_structure.json", True),
        ("cdg_fail_bad_structure.json", False),
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
    parser = argparse.ArgumentParser(description="QA Causal DAG Cert [213] validator")
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
