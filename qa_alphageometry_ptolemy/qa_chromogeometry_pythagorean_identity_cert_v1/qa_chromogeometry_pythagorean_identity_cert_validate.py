#!/usr/bin/env python3
"""
qa_chromogeometry_pythagorean_identity_cert_validate.py  [family 234]
"""

QA_COMPLIANCE = "cert_validator - Chromogeometry Pythagorean identity over QA coordinates; integer state space; raw d=b+e; no pow operator; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_CHROMOGEOMETRY_PYTHAGOREAN_IDENTITY_CERT.v1"


def quadrances(b, e):
    q_b = b * b + e * e
    q_r = b * b - e * e
    q_g = 2 * b * e
    lhs = q_b * q_b
    rhs = q_r * q_r + q_g * q_g
    return q_b, q_r, q_g, lhs, rhs, lhs - rhs


def _run_checks(fixture):
    checks = {}
    checks["CPI_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    samples = fixture.get("sample_pairs", [])
    sample_ok = isinstance(samples, list) and len(samples) == 20
    if sample_ok:
        for row in samples:
            b = row.get("b")
            e = row.get("e")
            if not (isinstance(b, int) and isinstance(e, int) and 1 <= b <= 19 and 1 <= e <= 19):
                sample_ok = False
                break
            q_b, q_r, q_g, lhs, rhs, diff = quadrances(b, e)
            expected = {"Q_b": q_b, "Q_r": q_r, "Q_g": q_g, "Q_b_sq": lhs, "Q_r_sq_plus_Q_g_sq": rhs, "difference": diff}
            for key, val in expected.items():
                if row.get(key) != val:
                    sample_ok = False
                    break
            if not sample_ok:
                break
    checks["CPI_SAMPLES"] = sample_ok

    failures = []
    for b in range(1, 20):
        for e in range(1, 20):
            if quadrances(b, e)[5] != 0:
                failures.append([b, e])
    checks["CPI_RANGE"] = fixture.get("exhaustive_range") == "[1..19]^2" and fixture.get("exhaustive_failures") == len(failures) == 0

    formulas = fixture.get("qa_coord_formulas_verified", {})
    checks["CPI_FORMULAS"] = formulas == {"Q_r": "(b-e)*d", "Q_g": "2*b*e", "Q_b": "b*b + e*e"}

    triples = fixture.get("plimpton_link", [])
    triples_ok = isinstance(triples, list) and len(triples) == 5
    if triples_ok:
        for row in triples:
            r = row.get("r")
            s = row.get("s")
            if not (isinstance(r, int) and isinstance(s, int)):
                triples_ok = False
                break
            q_b, q_r, q_g, _, _, diff = quadrances(r, s)
            if row.get("triple") != [q_r, q_g, q_b] or diff != 0 or row.get("qa_generated") is not True:
                triples_ok = False
                break
    checks["CPI_PLIMPTON"] = triples_ok
    src = fixture.get("source_attribution", "")
    checks["CPI_SRC"] = "Wildberger" in src and ("Dale" in src or "Claude" in src)
    checks["CPI_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_chromogeometry_pythagorean_identity_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
