#!/usr/bin/env python3
"""
qa_chromogeometric_tqf_symmetry_cert_validate.py  [family 246]
"""

QA_COMPLIANCE = "cert_validator - chromogeometric TQF symmetry TQF_r = TQF_g = -TQF_b; symbolic polynomial identity; integer arithmetic; no float; deterministic 3000-triangle sample plus symbolic proof via sympy"

import itertools
import json
import random
import sys
from pathlib import Path

import sympy as sp

SCHEMA_VERSION = "QA_CHROMOGEOMETRIC_TQF_SYMMETRY_CERT.v1"


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


def tqf(q1, q2, q3):
    s = q1 + q2 + q3
    return s * s - 2 * (q1 * q1 + q2 * q2 + q3 * q3)


def tqf_for_points(points, quadrance):
    q12 = quadrance(points[0], points[1])
    q13 = quadrance(points[0], points[2])
    q23 = quadrance(points[1], points[2])
    return tqf(q12, q13, q23)


def area2(points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    return x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)


def _row_for_points(label, points):
    blue = tqf_for_points(points, q_blue)
    red = tqf_for_points(points, q_red)
    green = tqf_for_points(points, q_green)
    twice_area = area2(points)
    return {
        "label": label,
        "points": points,
        "tqf": {"blue": blue, "red": red, "green": green},
        "area2": twice_area,
        "blue_factored": 4 * twice_area * twice_area,
        "sign_identity": red == -blue and green == -blue,
        "collinear": twice_area == 0,
    }


def _sample_rows():
    points = [[x, y] for x in range(1, 10) for y in range(1, 10)]
    triples = [list(t) for t in itertools.combinations(points, 3)]
    rng = random.Random(246)
    rng.shuffle(triples)
    rows = []
    for idx, triple in enumerate(triples[:3000]):
        rows.append(_row_for_points(f"sample_{idx:04d}", triple))
    return rows


def _symbolic_tqfs():
    x1, y1, x2, y2, x3, y3 = sp.symbols("x1 y1 x2 y2 x3 y3", integer=True)
    p1 = (x1, y1)
    p2 = (x2, y2)
    p3 = (x3, y3)

    def sq(v):
        return v * v

    def qb(p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return sq(dx) + sq(dy)

    def qr(p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return sq(dx) - sq(dy)

    def qg(p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return 2 * dx * dy

    def tq(q1, q2, q3):
        s = q1 + q2 + q3
        return s * s - 2 * (q1 * q1 + q2 * q2 + q3 * q3)

    tb = tq(qb(p1, p2), qb(p1, p3), qb(p2, p3))
    tr = tq(qr(p1, p2), qr(p1, p3), qr(p2, p3))
    tg = tq(qg(p1, p2), qg(p1, p3), qg(p2, p3))
    area_expr = x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2)
    return tb, tr, tg, area_expr


def _symbolic_ok():
    tb, tr, tg, area_expr = _symbolic_tqfs()
    return (
        sp.simplify(tr + tb) == 0
        and sp.simplify(tg + tb) == 0
        and sp.factor(tb - 4 * area_expr * area_expr) == 0
    )


def _sample_ok(spec):
    expected_rows = _sample_rows()
    rows = spec.get("triangles", [])
    if spec != {
        "coordinate_range": [1, 9],
        "sample_seed": 246,
        "sample_count": 3000,
        "zero_violations": True,
        "triangles": expected_rows,
    }:
        return False
    for row in rows:
        points = row.get("points")
        if not (isinstance(points, list) and len(points) == 3):
            return False
        expected = _row_for_points(row.get("label"), points)
        if row != expected:
            return False
        blue = row["tqf"]["blue"]
        red = row["tqf"]["red"]
        green = row["tqf"]["green"]
        if red != -blue or green != -blue:
            return False
    return True


def _collinearity_ok(spec):
    points = [[x, y] for x in range(1, 10) for y in range(1, 10)]
    triples = itertools.combinations(points, 3)
    checked = 0
    zero_count = 0
    for triple in triples:
        tri = [list(p) for p in triple]
        blue = tqf_for_points(tri, q_blue)
        red = tqf_for_points(tri, q_red)
        green = tqf_for_points(tri, q_green)
        col = area2(tri) == 0
        if (blue == 0) != col or (red == 0) != col or (green == 0) != col:
            return False
        if col:
            zero_count += 1
        checked += 1
    return spec == {
        "coordinate_range": [1, 9],
        "exhaustive_triples": checked,
        "zero_tqf_triples": zero_count,
        "all_zero_tqf_iff_collinear": True,
    }


def _run_checks(fixture):
    checks = {}
    checks["CTQF_1"] = fixture.get("schema_version") == SCHEMA_VERSION
    checks["CTQF_SYMBOLIC_RB"] = fixture.get("symbolic_rb") == {
        "identity": "TQF_r + TQF_b == 0",
        "sympy_simplify": 0,
    } and _symbolic_ok()
    checks["CTQF_SYMBOLIC_GB"] = fixture.get("symbolic_gb") == {
        "identity": "TQF_g + TQF_b == 0",
        "sympy_simplify": 0,
    } and _symbolic_ok()
    checks["CTQF_FACTORED_BLUE"] = fixture.get("factored_blue") == {
        "factor": "TQF_b = 4*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)",
        "area_relation": "TQF_b = 16*A*A where A=area2/2",
        "nonnegative": True,
    } and _symbolic_ok()
    checks["CTQF_SAMPLE_EXHAUSTIVE"] = _sample_ok(fixture.get("sample_identity", {}))
    checks["CTQF_COLLINEARITY_INVARIANT"] = _collinearity_ok(fixture.get("collinearity_invariant", {}))

    src = fixture.get("source_attribution", "")
    checks["CTQF_SRC"] = "Wildberger" in src and "Chromogeometry" in src and "Divine Proportions" in src

    witnesses = fixture.get("witnesses", [])
    checks["CTQF_WITNESS"] = isinstance(witnesses, list) and {
        "SYMBOLIC_RB",
        "SYMBOLIC_GB",
        "SAMPLE_EXHAUSTIVE",
        "FACTORED_BLUE",
        "COLLINEARITY_INVARIANT",
    }.issubset({w.get("kind") for w in witnesses})
    checks["CTQF_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_chromogeometric_tqf_symmetry_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
