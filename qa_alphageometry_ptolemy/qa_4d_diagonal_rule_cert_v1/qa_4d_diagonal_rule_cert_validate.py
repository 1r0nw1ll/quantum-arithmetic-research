#!/usr/bin/env python3
"""
qa_4d_diagonal_rule_cert_validate.py  [family 237]
"""

QA_COMPLIANCE = "cert_validator - QA 4D diagonal rule cert; integer vector arithmetic; raw d=b+e and a=b+2e; no pow operator; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_4D_DIAGONAL_RULE_CERT.v1"
V1 = [1, 0, 1, 1]
V2 = [0, 1, 1, 2]


def qa_tuple(b, e):
    return [b, e, b + e, b + 2 * e]


def lincomb(b, e):
    return [b * V1[i] + e * V2[i] for i in range(4)]


def dot(u, v):
    return sum(u[i] * v[i] for i in range(4))


def quadrance(u):
    return dot(u, u)


def vector_from_be(pair):
    return qa_tuple(pair[0], pair[1])


def _run_checks(fixture):
    checks = {}
    checks["Q4D_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    embedding_failures = []
    for b in range(-5, 6):
        for e in range(-5, 6):
            if qa_tuple(b, e) != lincomb(b, e):
                embedding_failures.append([b, e])
    checks["Q4D_EMBED"] = (
        fixture.get("embedding_range") == "[-5..5]^2"
        and fixture.get("embedding_failures") == len(embedding_failures) == 0
        and fixture.get("basis") == {"v1": V1, "v2": V2}
    )

    gram = [[dot(V1, V1), dot(V1, V2)], [dot(V2, V1), dot(V2, V2)]]
    det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    checks["Q4D_GRAM"] = fixture.get("gram_matrix") == gram and fixture.get("gram_det") == det == 9
    checks["Q4D_MODULUS"] = fixture.get("canonical_modulus") == 9 and fixture.get("gram_det") == fixture.get("canonical_modulus")

    witnesses = fixture.get("diagonal_rule_witnesses", [])
    witness_ok = isinstance(witnesses, list) and len(witnesses) == 2
    if witness_ok:
        for row in witnesses:
            u = vector_from_be(row.get("u_be", []))
            v = vector_from_be(row.get("v_be", []))
            diff = [u[i] - v[i] for i in range(4)]
            q1 = quadrance(u)
            q2 = quadrance(v)
            q3 = quadrance(diff)
            expected = {
                "u_be": row.get("u_be"),
                "v_be": row.get("v_be"),
                "u_tuple": u,
                "v_tuple": v,
                "dot": dot(u, v),
                "Q1": q1,
                "Q2": q2,
                "Q3": q3,
                "Q1_plus_Q2_minus_Q3": q1 + q2 - q3,
            }
            if row != expected or expected["dot"] != 0 or expected["Q1_plus_Q2_minus_Q3"] != 0:
                witness_ok = False
                break
    checks["Q4D_DIAGONAL_RULE"] = witness_ok

    src = fixture.get("source_attribution", "")
    checks["Q4D_SRC"] = "Wildberger" in src and "Diagonal Rule" in src and "KoG" in src
    checks["Q4D_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_4d_diagonal_rule_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
