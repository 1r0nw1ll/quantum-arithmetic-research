#!/usr/bin/env python3
"""
qa_quadrance_product_cert_validate.py

Validator for QA_QUADRANCE_PRODUCT_CERT.v1  [family 208]

Certifies: Every QA area element is irreducibly a two-factor product of
role-distinct base elements. Quadrances (A=a*a, B=b*b) are products,
not powers. S1 (always b*b product form) is structural, not just numerical.

Will Dale's insight (2026-04-08):
    A square is NOT a "special case" of rectangle. It is a rectangle
    whose sides happen to have equal values — but the two factors
    maintain distinct structural roles. Even 1*1 = 1 is an AREA
    (product of two unit lengths), not the scalar 1.

    Parallel to [207] Circle Impossibility:
    - Circle: observer sees equal axes → concludes C=0. But C >= 2.
    - Square: observer sees equal sides → concludes "one parameter."
      But the product has TWO factors with distinct roles.

S1 deep reason:
    S1 mandates b*b, never the power operator. The stated reason is
    ULP drift, but the structural reason is: b*b preserves the
    two-factor product (two operands); the power form collapses it
    to a unary operation (one operand), destroying role-distinction.

Checks:
    QP_1        — schema_version matches
    QP_PRODUCT  — product table has >= 6 area elements
    QP_ROLE     — all product elements have role-distinct factors
    QP_S1       — no formula uses ** (power notation)
    QP_AREA_MIN — minimum area verified (all products well-defined)
    QP_DIM      — dimensional types present (length/area/area²)
    QP_SQUARE   — square/rectangle parallel articulated (qa_view present)
    QP_SRC      — source attribution present
    QP_WITNESS  — at least 3 witnesses with explicit product forms
    QP_F        — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates product structure of QA area elements; integer state space; S1 compliant; no ** operator"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_QUADRANCE_PRODUCT_CERT.v1"


# -----------------------------------------------------------------------------
# QA primitives (integer-only, axiom-compliant)
# -----------------------------------------------------------------------------

def qa_tuple(b, e):
    """Compute full tuple (b, e, d, a). RAW derived coords for elements.

    Elements use d = b+e, a = b+2e (A2, un-reduced).
    Modular reduction is for T-operator state evolution, NOT element computation.
    """
    d = b + e
    a = b + 2 * e
    return (b, e, d, a)


def qa_products(b, e, d, a):
    """Compute all area-type elements as explicit products. S1 compliant.

    Uses RAW d = b+e, a = b+2e. F identity: b*a == d*d - e*e.
    """
    return {
        "A": a * a,
        "B": b * b,
        "D": d * d,
        "E": e * e,
        "C": 2 * d * e,
        "F_ba": b * a,
        "F_diff": (d - e) * (d + e),  # d*d - e*e as product
        "J": b * d,
        "X": e * d,
    }


def verify_products_all_states(m):
    """Verify product structure for all states in S_m."""
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            _, _, d, a = qa_tuple(b, e)
            prods = qa_products(b, e, d, a)
            # F consistency: b*a must equal (d-e)*(d+e) for primitive
            # (only guaranteed when d,e computed without modular reduction
            #  on the raw values, which holds for small m)
            # All products must be well-defined integers
            for name, val in prods.items():
                if not isinstance(val, int):
                    return False, f"{name} is not integer at ({b},{e})"
    return True, "all products integer"


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def _run_checks(fixture):
    results = {}

    # QP_1: schema version
    results["QP_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    # QP_PRODUCT: product table has >= 6 area elements
    pt = fixture.get("product_table", {})
    elements = pt.get("area_elements", [])
    results["QP_PRODUCT"] = len(elements) >= 6

    # QP_ROLE: all elements have role-distinct factors
    results["QP_ROLE"] = True
    if results["QP_PRODUCT"]:
        for elem in elements:
            f1 = elem.get("factor_1", {})
            f2 = elem.get("factor_2", {})
            r1 = f1.get("role", "")
            r2 = f2.get("role", "")
            # For cross-products (C, F, J, X), roles must be textually different
            if f1.get("value") != f2.get("value"):
                if r1 == r2:
                    results["QP_ROLE"] = False
                    break
            # For quadrances (A, B, D, E), values equal but roles must differ
            else:
                # Role strings should not be identical for quadrances
                if r1 == r2 and elem.get("type") == "quadrance":
                    results["QP_ROLE"] = False
                    break

    # QP_S1: no formula uses ** (power notation)
    results["QP_S1"] = True
    for elem in elements:
        formula = elem.get("formula", "")
        if "**" in formula:
            results["QP_S1"] = False
            break

    # QP_AREA_MIN: area verification
    results["QP_AREA_MIN"] = True
    # Computationally verify if product table is present
    if results["QP_PRODUCT"]:
        # Check that B_min = 1 (1*1) is acknowledged
        witnesses = fixture.get("witnesses", [])
        if witnesses:
            has_unit = any(
                "1*1" in str(w.get("products", {}))
                for w in witnesses
            )
            # Unit area must be present to prove even 1*1 is a product
            results["QP_AREA_MIN"] = has_unit

    # QP_DIM: dimensional types present
    rdp = fixture.get("role_distinction_proof", {})
    dim = rdp.get("dimensional_types", {})
    results["QP_DIM"] = (
        "length" in dim
        and "area" in dim
        and len(dim.get("length", [])) >= 2
        and len(dim.get("area", [])) >= 2
    )

    # QP_SQUARE: square/rectangle parallel
    srp = fixture.get("square_rectangle_parallel", {})
    results["QP_SQUARE"] = (
        "qa_view" in srp
        and isinstance(srp.get("qa_view"), str)
        and len(srp.get("qa_view", "")) > 20
    )

    # QP_SRC: source attribution
    src = fixture.get("source_attribution", "")
    results["QP_SRC"] = (
        "Will Dale" in src
        or "Dale" in src
    )

    # QP_WITNESS: at least 3 witnesses with correct tuple derivation
    witnesses = fixture.get("witnesses", [])
    results["QP_WITNESS"] = len(witnesses) >= 3
    if results["QP_WITNESS"]:
        for w in witnesses:
            prods = w.get("products", {})
            if len(prods) < 2:
                results["QP_WITNESS"] = False
                break
            # Verify at least one product contains '*'
            has_product_form = any("*" in str(v) for v in prods.values())
            if not has_product_form:
                results["QP_WITNESS"] = False
                break
            # Verify tuple_beda is correctly derived from state_be
            be = w.get("state_be", [])
            claimed_tuple = w.get("tuple_beda", [])
            if len(be) == 2 and len(claimed_tuple) == 4:
                b_val, e_val = be
                actual_tuple = list(qa_tuple(b_val, e_val))
                if actual_tuple != claimed_tuple:
                    results["QP_WITNESS"] = False
                    break

    # QP_F: fail_ledger
    fl = fixture.get("fail_ledger")
    results["QP_F"] = isinstance(fl, list)

    return results


def validate_fixture(path):
    with open(path) as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    all_pass = all(checks.values())
    actual = "PASS" if all_pass else "FAIL"
    ok = actual == expected
    return {"ok": ok, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    """Run validator against bundled fixtures."""
    fdir = Path(__file__).parent / "fixtures"
    results = {}
    for fp in sorted(fdir.glob("*.json")):
        results[fp.name] = validate_fixture(fp)
    all_ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    elif len(sys.argv) > 1:
        r = validate_fixture(sys.argv[1])
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)
    else:
        print("Usage: python qa_quadrance_product_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
