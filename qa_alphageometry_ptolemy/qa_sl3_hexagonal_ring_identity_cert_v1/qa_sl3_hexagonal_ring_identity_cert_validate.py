#!/usr/bin/env python3
"""
qa_sl3_hexagonal_ring_identity_cert_validate.py  [family 245]
"""

QA_COMPLIANCE = "cert_validator - sl(3) hexagonal ring identity ring(a,b) = T_{d+1} + a*b in QA coords; pure integer arithmetic; no pow; no float; exhaustive on [1..14]^2"

import json
import sys
from pathlib import Path

import sympy as sp

SCHEMA_VERSION = "QA_SL3_HEXAGONAL_RING_IDENTITY_CERT.v1"


def triangular(n):
    return n * (n + 1) // 2


def sl3_dim(b, e):
    numerator = (b + 1) * (e + 1) * (b + e + 2)
    if numerator % 2 != 0:
        raise ValueError("sl3 dimension numerator must be even")
    return numerator // 2


def ring(a, b):
    if a < 1 or b < 1:
        raise ValueError("ring identity is certified for a,b >= 1")
    return sl3_dim(a, b) - sl3_dim(a - 1, b - 1)


def ring_closed(a, b):
    total = a + b
    return triangular(total + 1) + a * b


def _expected_entries():
    entries = []
    for a in range(1, 15):
        for b in range(1, 15):
            total = a + b
            t = triangular(total + 1)
            entries.append({
                "a": a,
                "b": b,
                "qa_b": a,
                "qa_e": b,
                "d": total,
                "dim_ab": sl3_dim(a, b),
                "dim_inner": sl3_dim(a - 1, b - 1),
                "ring": ring(a, b),
                "triangular_index": total + 1,
                "triangular_value": t,
                "bilinear_ab": a * b,
                "closed_form": t + a * b,
            })
    return entries


def _symbolic_expansion_ok():
    a, b = sp.symbols("a b", integer=True)
    lhs_num = (a + 1) * (b + 1) * (a + b + 2) - a * b * (a + b)
    rhs_num = (a + b + 1) * (a + b + 2) + 2 * a * b
    return sp.expand(lhs_num - rhs_num) == 0


def _run_checks(fixture):
    checks = {}
    checks["SHR_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    expected_entries = _expected_entries()
    algebraic = fixture.get("algebraic_expansion", {})
    checks["SHR_ALGEBRAIC_EXPANSION"] = algebraic == {
        "lhs": "(a+1)*(b+1)*(a+b+2)//2 - a*b*(a+b)//2",
        "rhs": "(a+b+1)*(a+b+2)//2 + a*b",
        "cleared_denominator_identity": "(a+1)*(b+1)*(a+b+2) - a*b*(a+b) == (a+b+1)*(a+b+2) + 2*a*b",
        "sympy_expand_difference": 0,
    } and _symbolic_expansion_ok()

    exhaustive = fixture.get("exhaustive_grid", {})
    entries = exhaustive.get("entries", [])
    checks["SHR_EXHAUSTIVE"] = exhaustive == {
        "a_range": [1, 14],
        "b_range": [1, 14],
        "entry_count": 196,
        "zero_failures": True,
        "entries": expected_entries,
    } and all(row["ring"] == row["closed_form"] for row in expected_entries)

    qa_form = fixture.get("qa_coord_form", {})
    checks["SHR_QA_COORD_FORM"] = qa_form == {
        "identification": "(b_QA,e_QA)=(a,b)",
        "d_rule": "d=b_QA+e_QA",
        "formula": "ring=T(d+1)+b_QA*e_QA",
        "integer_polynomial": True,
    } and all(row["closed_form"] == triangular(row["d"] + 1) + row["qa_b"] * row["qa_e"] for row in expected_entries)

    samples = fixture.get("known_multiplicities", [])
    expected_samples = [
        {"a": 1, "b": 1, "ring": 7, "interpretation": "adjoint minus trivial"},
        {"a": 2, "b": 1, "ring": 12, "interpretation": "standard sl3 weight-diagram multiplicity"},
        {"a": 2, "b": 2, "ring": 19, "interpretation": "standard sl3 weight-diagram multiplicity"},
    ]
    checks["SHR_KNOWN_MULTIPLICITIES"] = samples == expected_samples and all(
        row["ring"] == ring(row["a"], row["b"]) for row in expected_samples
    )

    src = fixture.get("source_attribution", "")
    checks["SHR_SRC"] = "Wildberger" in src and "Quarks, diamonds" in src and "sl_3" in src

    witnesses = fixture.get("witnesses", [])
    checks["SHR_WITNESS"] = isinstance(witnesses, list) and {
        "ALGEBRAIC_EXPANSION",
        "EXHAUSTIVE",
        "QA_COORD_FORM",
        "KNOWN_MULTIPLICITIES",
    }.issubset({w.get("kind") for w in witnesses})
    checks["SHR_F"] = isinstance(fixture.get("fail_ledger"), list)

    if entries != expected_entries:
        checks["SHR_EXHAUSTIVE"] = False
    return checks


def validate_fixture(path):
    with open(path, encoding="utf-8") as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    actual = "PASS" if all(checks.values()) else "FAIL"
    return {"ok": actual == expected, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    fdir = Path(__file__).parent / "fixtures"
    results = {fp.name: validate_fixture(fp) for fp in sorted(fdir.glob("*.json"))}
    ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": ok, "checks": results}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    if len(sys.argv) > 1:
        result = validate_fixture(sys.argv[1])
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        sys.exit(0 if result["ok"] else 1)
    print("Usage: python qa_sl3_hexagonal_ring_identity_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
