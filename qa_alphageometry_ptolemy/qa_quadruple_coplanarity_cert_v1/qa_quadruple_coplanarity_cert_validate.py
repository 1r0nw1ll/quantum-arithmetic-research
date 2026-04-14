#!/usr/bin/env python3
"""
qa_quadruple_coplanarity_cert_validate.py  [family 241]
"""

QA_COMPLIANCE = "cert_validator - four-QA-point Cayley-Menger determinant zero under three chromogeometric quadrances; integer determinant via sympy; QA planarity d=b+e enforced (A2); no pow, no floats"

import json
import sys
from pathlib import Path

import sympy as sp

SCHEMA_VERSION = "QA_QUADRUPLE_COPLANARITY_CERT.v1"


def qa3(point):
    b, e = point
    return [b, e, b + e]


def plane_residual(point3):
    b, e, d = point3
    return d - b - e


def det_int(matrix):
    return int(sp.Matrix(matrix).det())


def parallelepiped_det(points):
    return det_int([qa3(point) for point in points])


def q_blue(p, q):
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    return dx * dx + dy * dy


def q_red(p, q):
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    return dx * dx - dy * dy


def q_green(p, q):
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    return 2 * dx * dy


QUADRANCE_FORMS = {
    "blue": q_blue,
    "red": q_red,
    "green": q_green,
}


def cayley_menger(points, quadrance):
    n = len(points)
    matrix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        matrix[0][i] = 1
        matrix[i][0] = 1
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i + 1][j + 1] = quadrance(points[i], points[j])
    return det_int(matrix)


def _valid_point(point, lo=1, hi=9):
    return (
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(v, int) for v in point)
        and lo <= point[0] <= hi
        and lo <= point[1] <= hi
    )


def _plane_grid_ok(spec):
    b_range = spec.get("b_range")
    e_range = spec.get("e_range")
    if b_range != [-9, 9] or e_range != [-9, 9]:
        return False
    count = 0
    max_abs = 0
    for b in range(b_range[0], b_range[1] + 1):
        for e in range(e_range[0], e_range[1] + 1):
            residual = plane_residual([b, e, b + e])
            if residual != 0:
                return False
            if abs(residual) > max_abs:
                max_abs = abs(residual)
            count += 1
    if spec.get("checked_count") != count or spec.get("max_abs_residual") != max_abs:
        return False
    for point in spec.get("explicit_points", []):
        if not (isinstance(point, list) and len(point) == 3 and all(isinstance(v, int) for v in point)):
            return False
        if plane_residual(point) != 0:
            return False
    return True


def _parallelepiped_ok(spec):
    triples = spec.get("triples", [])
    if spec.get("sample_count") != 30 or spec.get("sample_seed") != 241 or len(triples) != 30:
        return False
    for row in triples:
        points = row.get("points")
        if not (isinstance(points, list) and len(points) == 3 and all(_valid_point(point) for point in points)):
            return False
        det = parallelepiped_det(points)
        if row.get("det") != det or det != 0:
            return False
    return True


def _cm_ok(spec):
    rows = spec.get("quadruples", [])
    if spec.get("sample_count") != 31 or spec.get("random_sample_count") != 30 or len(rows) != 31:
        return False
    if spec.get("quadrance_forms") != ["blue", "red", "green"]:
        return False
    satellite = [[3, 3], [6, 9], [6, 6], [3, 9]]
    if spec.get("satellite_quadruple") != satellite:
        return False
    seen_satellite = False
    for row in rows:
        points = row.get("points")
        if not (isinstance(points, list) and len(points) == 4 and all(_valid_point(point) for point in points)):
            return False
        if len({tuple(point) for point in points}) != 4:
            return False
        if {tuple(point) for point in points} == {tuple(point) for point in satellite}:
            seen_satellite = True
        determinants = row.get("determinants")
        expected = {name: cayley_menger(points, fn) for name, fn in QUADRANCE_FORMS.items()}
        if determinants != expected:
            return False
        if any(value != 0 for value in expected.values()):
            return False
    return seen_satellite


def _run_checks(fixture):
    checks = {}
    checks["QCO_1"] = fixture.get("schema_version") == SCHEMA_VERSION
    checks["QCO_PLANE_IDENTITY"] = _plane_grid_ok(fixture.get("plane_identity", {}))
    checks["QCO_PARALLELEPIPED_VOL"] = _parallelepiped_ok(fixture.get("parallelepiped_volume", {}))
    checks["QCO_CM_4POINT_BLUE_RED_GREEN"] = _cm_ok(fixture.get("cayley_menger", {}))
    checks["QCO_CHROMO_COPLANARITY_PRESERVED"] = fixture.get("chromo_coplanarity_preserved") == {
        "forms": ["blue", "red", "green"],
        "all_cayley_menger_determinants_zero": True,
        "integer_polynomial_identity": True,
    } and checks["QCO_CM_4POINT_BLUE_RED_GREEN"]

    src = fixture.get("source_attribution", "")
    checks["QCO_SRC"] = "Notowidigdo" in src and "Wildberger" in src and "Chromogeometry" in src

    witnesses = fixture.get("witnesses", [])
    checks["QCO_WITNESS"] = isinstance(witnesses, list) and {
        "PLANE_IDENTITY",
        "PARALLELEPIPED_VOL",
        "CM_4POINT_BLUE",
        "CM_4POINT_RED",
        "CM_4POINT_GREEN",
        "CHROMO_COPLANARITY_PRESERVED",
    }.issubset({w.get("kind") for w in witnesses})
    checks["QCO_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_quadruple_coplanarity_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
