#!/usr/bin/env python3
"""
qa_spread_polynomial_composition_cert_validate.py  [family 236]
"""

QA_COMPLIANCE = "cert_validator - Spread polynomial composition cert; exact symbolic arithmetic; raw d=b+e; no pow operator; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SPREAD_POLYNOMIAL_COMPOSITION_CERT.v1"


def _clean(poly):
    return {degree: coeff for degree, coeff in poly.items() if coeff != 0}


def _add(left, right):
    out = dict(left)
    for degree, coeff in right.items():
        out[degree] = out.get(degree, 0) + coeff
    return _clean(out)


def _scale(poly, factor):
    return _clean({degree: coeff * factor for degree, coeff in poly.items()})


def _mul(left, right):
    out = {}
    for left_degree, left_coeff in left.items():
        for right_degree, right_coeff in right.items():
            degree = left_degree + right_degree
            out[degree] = out.get(degree, 0) + left_coeff * right_coeff
    return _clean(out)


def _pow(poly, exponent):
    out = {0: 1}
    for _ in range(exponent):
        out = _mul(out, poly)
    return out


def _compose(outer, inner):
    out = {}
    for degree, coeff in outer.items():
        out = _add(out, _scale(_pow(inner, degree), coeff))
    return _clean(out)


def _spread_polys(max_n):
    s = {1: 1}
    one_minus_2s = {0: 1, 1: -2}
    polys = {0: {}, 1: s}
    for n in range(1, max_n):
        polys[n + 1] = _add(
            _add(_scale(_mul(one_minus_2s, polys[n]), 2), _scale(polys[n - 1], -1)),
            _scale(s, 2),
        )
    return polys


def _coeff_map(poly):
    return {str(degree): coeff for degree, coeff in sorted(_clean(poly).items())}


def _run_checks(fixture):
    checks = {}
    checks["SPC_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    composition_rows = fixture.get("composition_witnesses", [])
    max_n = 1
    for row in composition_rows:
        if isinstance(row.get("n"), int) and isinstance(row.get("m"), int):
            max_n = max(max_n, row["n"] * row["m"])
    polys = _spread_polys(max_n)

    composition_ok = isinstance(composition_rows, list) and len(composition_rows) == 6
    if composition_ok:
        expected_pairs = {(2, 3), (3, 2), (2, 4), (4, 3), (3, 3), (2, 5)}
        seen_pairs = set()
        for row in composition_rows:
            n = row.get("n")
            m = row.get("m")
            if (n, m) not in expected_pairs:
                composition_ok = False
                break
            lhs = _compose(polys[n], polys[m])
            rhs = polys[n * m]
            lhs_coeffs = _coeff_map(lhs)
            rhs_coeffs = _coeff_map(rhs)
            if lhs_coeffs != rhs_coeffs:
                composition_ok = False
                break
            if row.get("lhs_coeffs") != lhs_coeffs or row.get("rhs_coeffs") != rhs_coeffs:
                composition_ok = False
                break
            seen_pairs.add((n, m))
        composition_ok = composition_ok and seen_pairs == expected_pairs
    checks["SPC_COMPOSITION"] = composition_ok

    closed = fixture.get("integer_closed_forms", {})
    checks["SPC_CLOSED_FORMS"] = closed == {
        "S_2": {"1": 4, "2": -4},
        "S_3": {"1": 9, "2": -24, "3": 16},
        "S_4": {"1": 16, "2": -80, "3": 128, "4": -64},
    } and all(_coeff_map(polys[n]) == closed[f"S_{n}"] for n in (2, 3, 4))

    checks["SPC_LOGISTIC"] = fixture.get("logistic_map") == {
        "S_2_factored": "4*s*(1-s)",
        "S_2_expanded_coeffs": {"1": 4, "2": -4},
    } and _coeff_map({1: 4, 2: -4}) == _coeff_map(polys[2])

    trig_note = fixture.get("rational_trig_identity_note", "")
    checks["SPC_TRIG_NOTE"] = "skipped" in trig_note and "float" in trig_note

    src = fixture.get("source_attribution", "")
    checks["SPC_SRC"] = "Goh" in src and "Wildberger" in src and "Spread polynomials" in src
    checks["SPC_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_spread_polynomial_composition_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
