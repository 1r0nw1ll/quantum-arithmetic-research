#!/usr/bin/env python3
"""
qa_uhg_diagonal_coincidence_cert_validate.py  [family 232]
"""

QA_COMPLIANCE = "cert_validator - UHG zero quadrance equals QA gcd-reduced diagonal class at m=9; integer state space with Fraction rational arithmetic; no pow operator; no float state"

import json
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_UHG_DIAGONAL_COINCIDENCE_CERT.v1"


def uhg(a, b):
    return -(a[0] * b[1] + a[1] * b[0])


def quad(a, b):
    n = uhg(a, b)
    n = n * n
    denom = uhg(a, a) * uhg(b, b)
    return None if denom == 0 else Fraction(1) - Fraction(n, denom)


def dc(p):
    g = gcd(p[0], p[1])
    return (p[0] // g, p[1] // g)


def _enumerate(m):
    points = [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)]
    zero_q = []
    same_diag = []
    intersection = []
    zero_diff = []
    same_nonzero = []
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            q = quad(a, b)
            if q is None:
                continue
            is_zero = q == 0
            is_same = dc(a) == dc(b)
            if is_zero:
                zero_q.append((a, b))
            if is_same:
                same_diag.append((a, b))
            if is_zero and is_same:
                intersection.append((a, b))
            if is_zero and not is_same:
                zero_diff.append((a, b))
            if is_same and not is_zero:
                same_nonzero.append((a, b))
    return zero_q, same_diag, intersection, zero_diff, same_nonzero


def _parse_pair(row):
    return tuple(row["p"]), tuple(row["q"])


def _fraction_text(q):
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def _check_witness_rows(rows, expected_zero):
    if not isinstance(rows, list) or len(rows) < 5:
        return False
    for row in rows:
        try:
            a, b = _parse_pair(row)
        except Exception:
            return False
        q = quad(a, b)
        if q is None:
            return False
        if (q == 0) is not expected_zero:
            return False
        if row.get("diagonal_class_p") != list(dc(a)):
            return False
        if row.get("diagonal_class_q") != list(dc(b)):
            return False
        if row.get("same_diagonal") is not (dc(a) == dc(b)):
            return False
        if row.get("quadrance") != _fraction_text(q):
            return False
    return True


def _run_checks(fixture):
    checks = {}
    checks["UDC_1"] = fixture.get("schema_version") == SCHEMA_VERSION
    m = fixture.get("m")
    checks["UDC_M"] = m == 9
    if m == 9:
        zero_q, same_diag, intersection, zero_diff, same_nonzero = _enumerate(m)
    else:
        zero_q, same_diag, intersection, zero_diff, same_nonzero = [], [], [], [], []
    checks["UDC_COUNTS"] = fixture.get("zero_q_pair_count") == len(zero_q) == 64 and fixture.get("same_diagonal_pair_count") == len(same_diag) == 64
    checks["UDC_INTERSECTION"] = fixture.get("intersection_size") == len(intersection) == 64
    ce = fixture.get("counter_examples", {})
    checks["UDC_COUNTEREXAMPLES"] = isinstance(ce, dict) and ce.get("zero_q_but_diff_diagonal") == [] and ce.get("same_diagonal_but_nonzero_q") == [] and zero_diff == [] and same_nonzero == []
    witnesses = fixture.get("witnesses", {})
    checks["UDC_WITNESS"] = isinstance(witnesses, dict) and _check_witness_rows(witnesses.get("zero_q_pairs"), True) and _check_witness_rows(witnesses.get("non_zero_q_pairs"), False)
    src = fixture.get("source_attribution", "")
    checks["UDC_SRC"] = "Wildberger" in src and ("Dale" in src or "Claude" in src)
    checks["UDC_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_uhg_diagonal_coincidence_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
