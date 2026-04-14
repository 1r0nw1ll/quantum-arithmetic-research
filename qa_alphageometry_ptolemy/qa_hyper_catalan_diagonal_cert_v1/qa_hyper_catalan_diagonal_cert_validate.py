#!/usr/bin/env python3
"""
qa_hyper_catalan_diagonal_cert_validate.py  [family 231]

Validator for QA_HYPER_CATALAN_DIAGONAL_CORRESPONDENCE_CERT.v1.
Packages the pre-verified Wildberger-Rubine hyper-Catalan diagonal
correspondence. This validator recomputes only the fixture claims needed for
certificate packaging.
"""

QA_COMPLIANCE = "cert_validator - Hyper-Catalan diagonal packaging; integer state space; A1/A2 compliant; raw d=b+e; no pow operator; no float state"

import json
import sys
from math import factorial
from pathlib import Path

SCHEMA_VERSION = "QA_HYPER_CATALAN_DIAGONAL_CORRESPONDENCE_CERT.v1"

OEIS = {
    2: [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796],
    3: [1, 1, 3, 12, 55, 273, 1428, 7752, 43263],
    4: [1, 1, 4, 22, 140, 969, 7084],
    5: [1, 1, 5, 35, 285, 2530],
}


def hc(mv):
    V = 2 + sum((k + 1) * m for k, m in enumerate(mv))
    E = 1 + sum((k + 2) * m for k, m in enumerate(mv))
    F = sum(mv)
    denom = factorial(V - 1)
    for m in mv:
        denom *= factorial(m)
    return factorial(E - 1) // denom, V, E, F


def single_type_mv(k, n):
    mv = [0] * (k - 1)
    mv[k - 2] = n
    return mv


def _run_checks(fixture):
    checks = {}
    checks["HCD_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    euler_rows = fixture.get("euler_check", [])
    euler_ok = isinstance(euler_rows, list) and len(euler_rows) == 5
    if euler_ok:
        for row in euler_rows:
            mv = row.get("m")
            if not (isinstance(mv, list) and all(isinstance(x, int) and x >= 0 for x in mv)):
                euler_ok = False
                break
            _, V, E, F = hc(mv)
            b = V - 1
            e = F
            expected = {
                "V": V,
                "E": E,
                "F": F,
                "b": b,
                "e": e,
                "d_derived": b + e,
                "V_minus_E_plus_F": V - E + F,
            }
            for key, val in expected.items():
                if row.get(key) != val:
                    euler_ok = False
                    break
            if not euler_ok or row.get("d_derived") != row.get("E") or row.get("V_minus_E_plus_F") != 1:
                euler_ok = False
                break
    checks["HCD_EULER"] = euler_ok

    cat_rows = fixture.get("catalan_oeis_match", [])
    cat_ok = isinstance(cat_rows, list) and len(cat_rows) == len(OEIS[2])
    if cat_ok:
        for row in cat_rows:
            n = row.get("n")
            if not isinstance(n, int) or not (0 <= n < len(OEIS[2])):
                cat_ok = False
                break
            c_val, _, _, _ = hc(single_type_mv(2, n))
            if row.get("C_computed") != c_val or row.get("oeis_A000108") != OEIS[2][n] or row.get("match") is not True:
                cat_ok = False
                break
    checks["HCD_OEIS"] = cat_ok

    fuss = fixture.get("fuss_oeis_match", {})
    fuss_ok = isinstance(fuss, dict)
    if fuss_ok:
        for k in (3, 4, 5):
            rows = fuss.get(str(k))
            if not isinstance(rows, list) or len(rows) != len(OEIS[k]):
                fuss_ok = False
                break
            for row in rows:
                n = row.get("n")
                if not isinstance(n, int) or not (0 <= n < len(OEIS[k])):
                    fuss_ok = False
                    break
                c_val, _, _, _ = hc(single_type_mv(k, n))
                if row.get("C") != c_val or row.get("oeis") != OEIS[k][n] or row.get("match") is not True:
                    fuss_ok = False
                    break
            if not fuss_ok:
                break
    checks["HCD_FUSS"] = fuss_ok

    diag_rows = fixture.get("single_type_diagonal", [])
    diag_ok = isinstance(diag_rows, list) and len(diag_rows) >= 8
    if diag_ok:
        for row in diag_rows:
            k = row.get("k")
            n = row.get("n")
            if not (isinstance(k, int) and isinstance(n, int) and k >= 2 and n >= 0):
                diag_ok = False
                break
            _, V, _, F = hc(single_type_mv(k, n))
            b = V - 1
            e = F
            expected_b = (k - 1) * e + 1
            if row.get("b") != b or row.get("e") != e or row.get("expected_b") != expected_b or row.get("on_sibling_diagonal") is not True:
                diag_ok = False
                break
    checks["HCD_SINGLE_DIAGONAL"] = diag_ok

    d1 = fixture.get("d1_disjointness", {})
    d1_ok = isinstance(d1, dict) and d1.get("k_range") == [2, 7] and d1.get("n_range") == [0, 9] and d1.get("hits") == []
    if d1_ok:
        for k in range(2, 8):
            for n in range(0, 10):
                _, V, _, F = hc(single_type_mv(k, n))
                if V - 1 == F:
                    d1_ok = False
                    break
            if not d1_ok:
                break
    checks["HCD_D1_DISJOINT"] = d1_ok

    src = fixture.get("source_attribution", "")
    checks["HCD_SRC"] = "Wildberger" in src and "Rubine" in src and ("Dale" in src or "Claude" in src)
    witnesses = fixture.get("witnesses", [])
    checks["HCD_WITNESS"] = isinstance(witnesses, list) and {"euler", "oeis", "d1_disjoint"}.issubset({w.get("kind") for w in witnesses})
    checks["HCD_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_hyper_catalan_diagonal_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
