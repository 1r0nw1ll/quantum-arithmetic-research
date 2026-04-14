#!/usr/bin/env python3
"""
qa_neuberg_cubic_f23_cert_validate.py  [family 242]
"""

QA_COMPLIANCE = "cert_validator - Neuberg cubic F23 cert; finite-field integer arithmetic only; no pow operator; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_NEUBERG_CUBIC_F23_CERT.v1"
P = 23


def modp(n):
    return n % P


def square(n):
    return modp(n * n)


def curve_rhs(x, a, b):
    return modp(x * x * x + a * x + b)


def is_on_curve(x, y, a, b):
    return square(y) == curve_rhs(x, a, b)


def nonsingular_weierstrass(a, b):
    return modp(4 * a * a * a + 27 * b * b) != 0


def curve_points(a, b):
    pts = []
    for x in range(P):
        rhs = curve_rhs(x, a, b)
        for y in range(P):
            if square(y) == rhs:
                pts.append([x, y])
    return pts


def quadratic_residues():
    return {square(i) for i in range(P)}


def tangent_lambda(point, a):
    x0 = point[0]
    return modp(3 * x0 * x0 + a)


def tangent_conic_points(point, lam):
    x0, y0 = point
    pts = []
    for x in range(P):
        dx = modp(x - x0)
        for y in range(P):
            dy = modp(y - y0)
            if square(dy) == modp(lam * square(dx)):
                pts.append([x, y])
    return pts


def conic_relation(points_1, points_2):
    s1 = {tuple(pt) for pt in points_1}
    s2 = {tuple(pt) for pt in points_2}
    if s1 == s2:
        return "identical"
    if not s1.intersection(s2):
        return "disjoint"
    return "partial"


def quadrance(p1, p2):
    dx = modp(p2[0] - p1[0])
    dy = modp(p2[1] - p1[1])
    return modp(dx * dx + dy * dy)


def spread_pair_at_a(triangle):
    a, b, c = triangle
    bax = modp(b[0] - a[0])
    bay = modp(b[1] - a[1])
    cax = modp(c[0] - a[0])
    cay = modp(c[1] - a[1])
    cross = modp(bax * cay - bay * cax)
    q_ab = quadrance(a, b)
    q_ac = quadrance(a, c)
    return {
        "cross_mod_p": cross,
        "spread_num_mod_p": modp(cross * cross),
        "quadrance_ab": q_ab,
        "quadrance_ac": q_ac,
        "spread_den_mod_p": modp(q_ab * q_ac),
    }


def _run_checks(fixture):
    checks = {}
    checks["NCF23_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    curve = fixture.get("elliptic_curve", {})
    points = curve_points(1, 1)
    checks["NCF23_POINT_COUNT"] = curve == {
        "p": 23,
        "a": 1,
        "b": 1,
        "equation": "y^2=x^3+x+1",
        "affine_point_count": 27,
        "projective_point_count": 28,
        "points": points,
    } and nonsingular_weierstrass(1, 1)

    residues = quadratic_residues()
    witness_rows = fixture.get("tangent_conic_witnesses", [])
    conic_ok = isinstance(witness_rows, list) and len(witness_rows) == 2
    if conic_ok:
        seen_cases = set()
        for row in witness_rows:
            c1 = row.get("curve_1", {})
            c2 = row.get("curve_2", {})
            p1 = c1.get("point")
            p2 = c2.get("point")
            if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) == 2 and len(p2) == 2):
                conic_ok = False
                break
            a1 = c1.get("a")
            b1 = c1.get("b")
            a2 = c2.get("a")
            b2 = c2.get("b")
            if not (is_on_curve(p1[0], p1[1], a1, b1) and is_on_curve(p2[0], p2[1], a2, b2)):
                conic_ok = False
                break
            if not (nonsingular_weierstrass(a1, b1) and nonsingular_weierstrass(a2, b2)):
                conic_ok = False
                break
            lam1 = tangent_lambda(p1, a1)
            lam2 = tangent_lambda(p2, a2)
            pts1 = tangent_conic_points(p1, lam1)
            pts2 = tangent_conic_points(p2, lam2)
            relation = conic_relation(pts1, pts2)
            intersection_count = len({tuple(pt) for pt in pts1}.intersection({tuple(pt) for pt in pts2}))
            expected = {
                "case": row.get("case"),
                "curve_1": c1,
                "curve_2": c2,
                "lambda_1": lam1,
                "lambda_2": lam2,
                "lambda_1_is_nonsquare": lam1 not in residues,
                "lambda_2_is_nonsquare": lam2 not in residues,
                "conic_size_1": len(pts1),
                "conic_size_2": len(pts2),
                "intersection_count": intersection_count,
                "set_relation": relation,
            }
            if row != expected or relation not in ("identical", "disjoint"):
                conic_ok = False
                break
            seen_cases.add(row.get("case"))
    checks["NCF23_TANGENT_CONIC_DICHOTOMY"] = conic_ok and seen_cases == {"identical", "disjoint"}

    spread = fixture.get("orthic_spread_polynomial_witness", {})
    triangle = spread.get("triangle", [])
    spread_ok = isinstance(triangle, list) and len(triangle) == 3
    if spread_ok:
        recomputed = spread_pair_at_a(triangle)
        expected_spread = {
            "triangle": triangle,
            "field": "F_23",
            "spread_at_vertex_index": 0,
            "polynomial_form": "cross=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1); numerator=cross*cross; denominator=q12*q13",
            "division_required": False,
        }
        expected_spread.update(recomputed)
        spread_ok = spread == expected_spread and recomputed["spread_den_mod_p"] != 0
    checks["NCF23_SPREAD_POLYNOMIAL"] = spread_ok

    qa = fixture.get("qa_compatibility", {})
    units = qa.get("nonzero_unit_witnesses", [])
    checks["NCF23_QA_COMPAT"] = qa == {
        "field": "F_23",
        "p": 23,
        "char_not_2_or_3": True,
        "integer_arithmetic_only": True,
        "nonzero_unit_witnesses": units,
    } and all(isinstance(v, int) and 1 <= v < P for v in units)

    src = fixture.get("source_attribution", "")
    checks["NCF23_SRC"] = "Wildberger" in src and "Neuberg cubics" in src and "0806.2495" in src
    checks["NCF23_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_neuberg_cubic_f23_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
