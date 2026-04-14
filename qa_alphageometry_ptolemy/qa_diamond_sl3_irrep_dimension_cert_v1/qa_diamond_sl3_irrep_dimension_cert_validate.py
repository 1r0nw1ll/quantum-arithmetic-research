#!/usr/bin/env python3
"""
qa_diamond_sl3_irrep_dimension_cert_validate.py  [family 240]
"""

QA_COMPLIANCE = "cert_validator - sl(3) irrep dim formula (b+1)(e+1)(d+2)/2 as QA polynomial; pure integer arithmetic; no pow operator; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_DIAMOND_SL3_IRREP_DIMENSION_CERT.v1"

STANDARD_TABLE = [
    {"sl3_a": 0, "sl3_b": 0, "dim": 1},
    {"sl3_a": 1, "sl3_b": 0, "dim": 3},
    {"sl3_a": 0, "sl3_b": 1, "dim": 3},
    {"sl3_a": 1, "sl3_b": 1, "dim": 8},
    {"sl3_a": 2, "sl3_b": 0, "dim": 6},
    {"sl3_a": 0, "sl3_b": 2, "dim": 6},
    {"sl3_a": 2, "sl3_b": 1, "dim": 15},
    {"sl3_a": 1, "sl3_b": 2, "dim": 15},
    {"sl3_a": 3, "sl3_b": 0, "dim": 10},
    {"sl3_a": 0, "sl3_b": 3, "dim": 10},
    {"sl3_a": 2, "sl3_b": 2, "dim": 27},
    {"sl3_a": 3, "sl3_b": 1, "dim": 24},
    {"sl3_a": 1, "sl3_b": 3, "dim": 24},
    {"sl3_a": 3, "sl3_b": 2, "dim": 42},
    {"sl3_a": 2, "sl3_b": 3, "dim": 42},
    {"sl3_a": 4, "sl3_b": 0, "dim": 15},
    {"sl3_a": 0, "sl3_b": 4, "dim": 15},
    {"sl3_a": 3, "sl3_b": 3, "dim": 64},
    {"sl3_a": 4, "sl3_b": 1, "dim": 35},
    {"sl3_a": 1, "sl3_b": 4, "dim": 35},
    {"sl3_a": 4, "sl3_b": 2, "dim": 60},
    {"sl3_a": 2, "sl3_b": 4, "dim": 60},
]


def sl3_dim(b, e):
    d = b + e
    numerator = (b + 1) * (e + 1) * (d + 2)
    if numerator % 2 != 0:
        raise ValueError("sl3 dimension numerator must be even")
    return numerator // 2


def triangular(n):
    return n * (n + 1) // 2


def quark_column_dim(a):
    return triangular(a + 1)


def _standard_rows():
    rows = []
    for row in STANDARD_TABLE:
        b = row["sl3_a"]
        e = row["sl3_b"]
        d = b + e
        rows.append({
            "sl3_a": b,
            "sl3_b": e,
            "qa_b": b,
            "qa_e": e,
            "qa_d": d,
            "expected_dim": row["dim"],
            "computed_dim": sl3_dim(b, e),
        })
    return rows


def _check_polynomial_equivalence(limit):
    for b in range(limit + 1):
        for e in range(limit + 1):
            d = b + e
            direct = (b + 1) * (e + 1) * (b + e + 2)
            qa_polynomial = (b + 1) * (e + 1) * (d + 2)
            if direct != qa_polynomial:
                return False
            if qa_polynomial % 2 != 0:
                return False
    return True


def _run_checks(fixture):
    checks = {}
    checks["DSI_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    dim_formula = fixture.get("dim_formula", {})
    checks["DSI_DIM_FORMULA"] = dim_formula == {
        "identification": "(qa_b,qa_e)=(sl3_a,sl3_b)",
        "qa_d_rule": "qa_d=qa_b+qa_e",
        "formula": "(qa_b+1)*(qa_e+1)*(qa_d+2)//2",
        "standard_entry_count": 22,
        "matches": _standard_rows(),
        "polynomial_equivalence_checked_on_grid_0_to_6": True,
    } and all(row["computed_dim"] == row["expected_dim"] for row in _standard_rows()) and _check_polynomial_equivalence(6)

    adjoint = fixture.get("adjoint", {})
    checks["DSI_ADJOINT"] = adjoint == {
        "sl3_highest_weight": [1, 1],
        "qa_coordinates": {"b": 1, "e": 1, "d": 2},
        "computed_dim": 8,
        "sl3_lie_algebra_dim": 8,
    } and sl3_dim(1, 1) == 8

    triangular_column = fixture.get("triangular_column", {})
    expected_column = [{"a": a, "dim": quark_column_dim(a), "triangular_index": a + 1} for a in range(6)]
    checks["DSI_TRIANGULAR_COLUMN"] = triangular_column == {
        "family": "D(a,0)",
        "a_range": [0, 5],
        "dims": expected_column,
    } and [row["dim"] for row in expected_column] == [1, 3, 6, 10, 15, 21]

    qaaq = fixture.get("quark_antiquark", {})
    checks["DSI_QUARK_ANTIQUARK"] = qaaq == {
        "D(1,0)": {"dim": 3, "heights": [1, 0, 1], "name": "quark triple"},
        "D(0,1)": {"dim": 3, "heights": [-1, -1, 0], "name": "anti-quark triple"},
        "fundamental_decomposition": "D(a,b)=D(a,0)+D(0,b)",
    } and sl3_dim(1, 0) == 3 and sl3_dim(0, 1) == 3

    heisenberg = fixture.get("heisenberg_commutators", {})
    expected_brackets = [
        {"lhs": ["X_alpha", "X_beta"], "rhs": "X_alpha_plus_beta", "coefficient": 1},
        {"lhs": ["X_alpha", "X_alpha_plus_beta"], "rhs": "0", "coefficient": 0},
        {"lhs": ["X_alpha_plus_beta", "X_beta"], "rhs": "0", "coefficient": 0},
    ]
    checks["DSI_HEISENBERG"] = heisenberg == {
        "basis": ["X_alpha", "X_beta", "X_alpha_plus_beta"],
        "integer_coefficients": True,
        "brackets": expected_brackets,
    } and all(isinstance(row["coefficient"], int) for row in expected_brackets)

    src = fixture.get("source_attribution", "")
    checks["DSI_SRC"] = "Wildberger" in src and "Quarks, diamonds" in src and "sl_3" in src

    witnesses = fixture.get("witnesses", [])
    checks["DSI_WITNESS"] = isinstance(witnesses, list) and {
        "DIM_FORMULA",
        "ADJOINT",
        "TRIANGULAR_COLUMN",
        "QUARK_ANTIQUARK",
        "HEISENBERG",
    }.issubset({w.get("kind") for w in witnesses})
    checks["DSI_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_diamond_sl3_irrep_dimension_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
