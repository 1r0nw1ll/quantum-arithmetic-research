#!/usr/bin/env python3
"""
qa_uhg_orbit_diagonal_profile_cert_validate.py  [family 233]
"""

QA_COMPLIANCE = "cert_validator - UHG orbit diagonal profile at m=9; integer state space; T uses explicit mod reduction; no pow operator; no float state"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_UHG_ORBIT_DIAGONAL_PROFILE_CERT.v1"


def dc(p):
    g = gcd(p[0], p[1])
    return (p[0] // g, p[1] // g)


def T(b, e, m):
    return ((b + e - 1) % m) + 1


def step(p, m):
    b, e = p
    nb = T(b, e, m)
    return (nb, T(e, nb, m))


def orbit(start, m):
    out = []
    seen = set()
    p = start
    while p not in seen:
        seen.add(p)
        out.append(p)
        p = step(p, m)
    return out


def all_orbits(m):
    points = [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)]
    seen = set()
    out = []
    for p in points:
        if p in seen:
            continue
        o = orbit(p, m)
        for q in o:
            seen.add(q)
        out.append(o)
    return out


def as_points(rows):
    return [tuple(p) for p in rows]


def class_rows(points):
    return [list(dc(p)) for p in points]


def d1_points(points):
    return [p for p in points if dc(p) == (1, 1)]


def _same_cycle(a, b):
    if len(a) != len(b):
        return False
    doubled = b + b
    for i in range(len(b)):
        if doubled[i:i + len(a)] == a:
            return True
    return False


def _run_checks(fixture):
    checks = {}
    checks["UODP_1"] = fixture.get("schema_version") == SCHEMA_VERSION
    m = fixture.get("m")
    checks["UODP_M"] = m == 9
    generated = all_orbits(m) if m == 9 else []
    lengths = sorted(len(o) for o in generated)
    summary = fixture.get("partition_summary", {})
    checks["UODP_PARTITION"] = (
        lengths == [1, 4, 4, 12, 12, 12, 12, 12, 12]
        and summary.get("singularity_orbits") == 1
        and summary.get("satellite_orbits") == 2
        and summary.get("cosmos_orbits") == 6
        and summary.get("total_points") == 81
    )

    fixture_orbits = fixture.get("orbits", [])
    orbit_data_ok = isinstance(fixture_orbits, list) and len(fixture_orbits) >= 1
    d1_profile_ok = orbit_data_ok
    complement_ok = orbit_data_ok
    for row in fixture_orbits if isinstance(fixture_orbits, list) else []:
        pts = as_points(row.get("points", []))
        if not pts or row.get("cycle_length") != len(pts):
            orbit_data_ok = False
            break
        if not any(_same_cycle(pts, gen) for gen in generated):
            orbit_data_ok = False
            break
        if row.get("diagonal_class_multiset") != class_rows(pts):
            orbit_data_ok = False
            break
        d1 = d1_points(pts)
        if row.get("d1_multiplicity") != len(d1):
            d1_profile_ok = False
        if len(d1) == 1:
            if pts != [(9, 9)]:
                d1_profile_ok = False
            continue
        if len(d1) == 0:
            if row.get("d1_complement_pair") not in (None, []):
                complement_ok = False
            continue
        if len(d1) != 2:
            d1_profile_ok = False
            complement_ok = False
            continue
        pair = row.get("d1_complement_pair")
        if not (isinstance(pair, list) and len(pair) == 2):
            complement_ok = False
            continue
        a, b = tuple(pair[0]), tuple(pair[1])
        if set((a, b)) != set(d1):
            complement_ok = False
            continue
        if a[0] + b[0] != m or a[1] + b[1] != m:
            complement_ok = False
    checks["UODP_ORBIT_DATA"] = orbit_data_ok
    checks["UODP_D1_PROFILE"] = d1_profile_ok
    checks["UODP_COMPLEMENT"] = complement_ok
    src = fixture.get("source_attribution", "")
    checks["UODP_SRC"] = "Dale" in src or "Claude" in src
    checks["UODP_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_uhg_orbit_diagonal_profile_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
