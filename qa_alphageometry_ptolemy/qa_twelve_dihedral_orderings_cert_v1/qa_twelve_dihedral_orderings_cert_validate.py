#!/usr/bin/env python3
"""
qa_twelve_dihedral_orderings_cert_validate.py  [family 239]
"""

QA_COMPLIANCE = "cert_validator - Twelve dihedral orderings cert; exhaustive integer combinatorics; no pow operator; no float state"

import itertools
import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_TWELVE_DIHEDRAL_ORDERINGS_CERT.v1"
OBJECTS = [0, 1, 2, 3, 4]
EXPECTED_REPS = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 4, 3],
    [0, 1, 3, 2, 4],
    [0, 1, 3, 4, 2],
    [0, 1, 4, 2, 3],
    [0, 1, 4, 3, 2],
    [0, 2, 1, 3, 4],
    [0, 2, 1, 4, 3],
    [0, 2, 3, 1, 4],
    [0, 2, 4, 1, 3],
    [0, 3, 1, 2, 4],
    [0, 3, 2, 1, 4],
]


def _rotations(ordering):
    n = len(ordering)
    return [ordering[i:] + ordering[:i] for i in range(n)]


def dihedral_images(ordering):
    seq = list(ordering)
    rev = list(reversed(seq))
    return _rotations(seq) + _rotations(rev)


def canonical_rep(ordering):
    return list(min(tuple(img) for img in dihedral_images(ordering)))


def build_classes():
    classes = {}
    for perm in itertools.permutations(OBJECTS):
        rep = tuple(canonical_rep(perm))
        classes.setdefault(rep, []).append(list(perm))
    return classes


def _run_checks(fixture):
    checks = {}
    checks["TDO_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    classes = build_classes()
    reps = [list(rep) for rep in sorted(classes)]
    sizes = [{"canonical": list(rep), "size": len(members)} for rep, members in sorted(classes.items())]

    group = fixture.get("dihedral_group", {})
    checks["TDO_GROUP"] = group == {
        "object_count": 5,
        "rotation_count": 5,
        "reflection_count": 5,
        "order": 10,
    }

    checks["TDO_PERMUTATIONS"] = (
        fixture.get("object_set") == OBJECTS
        and fixture.get("permutation_count") == math.factorial(5) == 120
    )
    checks["TDO_CANONICAL_REPS"] = fixture.get("canonical_reps") == reps == EXPECTED_REPS
    checks["TDO_CLASS_COUNT"] = fixture.get("dihedral_class_count") == len(classes) == 12
    checks["TDO_CLASS_SIZE"] = fixture.get("class_sizes") == sizes and all(row["size"] == 10 for row in sizes)

    formula = fixture.get("quotient_formula", {})
    checks["TDO_FORMULA"] = formula == {
        "factorial_5": 120,
        "dihedral_order": 10,
        "quotient": 12,
        "expression": "5!/(2*5)",
    } and math.factorial(5) // 10 == 12

    qa = fixture.get("qa_connections", {})
    checks["TDO_QA_CONNECTION"] = qa == {
        "g2_non_identity_root_count": 12,
        "cuboctahedral_shell_s1": 12,
        "icosahedral_vertex_count": 12,
    }

    src = fixture.get("source_attribution", "")
    checks["TDO_SRC"] = "Le" in src and "Wildberger" in src and "Twelve Special Conics" in src
    checks["TDO_F"] = isinstance(fixture.get("fail_ledger"), list)
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
    print("Usage: python qa_twelve_dihedral_orderings_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
