#!/usr/bin/env python3
"""
qa_super_catalan_diagonal_cert_validate.py  [family 235]
"""

QA_COMPLIANCE = "cert_validator - Super Catalan diagonal cert; integer arithmetic only; raw d=b+e; no pow operator; no float state"

import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SUPER_CATALAN_DIAGONAL_CERT.v1"
A000984_0_TO_10 = [1, 2, 6, 20, 70, 252, 924, 3432, 12870, 48620, 184756]


def super_catalan(b, e):
    numerator = math.factorial(2 * b) * math.factorial(2 * e)
    denominator = math.factorial(b) * math.factorial(e) * math.factorial(b + e)
    return numerator // denominator


def catalan(n):
    return math.comb(2 * n, n) // (n + 1)


def _run_checks(fixture):
    checks = {}
    checks["SCD_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    d1_rows = fixture.get("d1_a000984_values", [])
    d1_ok = isinstance(d1_rows, list) and len(d1_rows) == len(A000984_0_TO_10)
    if d1_ok:
        for b, row in enumerate(d1_rows):
            expected = A000984_0_TO_10[b]
            if row != {"b": b, "S_b_b": super_catalan(b, b), "A000984": expected}:
                d1_ok = False
                break
    checks["SCD_D1_A000984"] = d1_ok

    symmetry_failures = []
    for b in range(0, 8):
        for e in range(0, 8):
            if super_catalan(b, e) != super_catalan(e, b):
                symmetry_failures.append([b, e])
    checks["SCD_SYMMETRY"] = (
        fixture.get("swap_symmetry_range") == "[0..7]^2"
        and fixture.get("swap_symmetry_failures") == len(symmetry_failures) == 0
    )

    recurrence_failures = []
    for b in range(0, 8):
        for e in range(0, 8):
            lhs = 4 * super_catalan(b, e)
            rhs = super_catalan(b + 1, e) + super_catalan(b, e + 1)
            if lhs != rhs:
                recurrence_failures.append([b, e, lhs, rhs])
    checks["SCD_RECURRENCE"] = (
        fixture.get("recurrence_range") == "[0..7]^2"
        and fixture.get("recurrence_failures") == len(recurrence_failures) == 0
    )

    catalan_rows = fixture.get("s1n_catalan_values", [])
    catalan_ok = isinstance(catalan_rows, list) and len(catalan_rows) == 10
    if catalan_ok:
        for n, row in enumerate(catalan_rows):
            c_n = catalan(n)
            if row != {"n": n, "S_1_n": super_catalan(1, n), "Catalan_n": c_n, "two_Catalan_n": 2 * c_n}:
                catalan_ok = False
                break
    checks["SCD_CATALAN"] = catalan_ok

    qa = fixture.get("qa_identification", {})
    checks["SCD_QA_IDENT"] = qa == {
        "b": "m",
        "e": "n",
        "d": "b+e",
        "formula_denominator_factor": "(m+n)! = d!",
    }
    src = fixture.get("source_attribution", "")
    checks["SCD_SRC"] = "Limanta" in src and "Wildberger" in src and "Super Catalan" in src
    checks["SCD_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_super_catalan_diagonal_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
