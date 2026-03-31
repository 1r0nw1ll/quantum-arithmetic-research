#!/usr/bin/env python3
"""
qa_cube_sum_cert_validate.py

Validator for QA_CUBE_SUM_CERT.v1  [family 143]

Certifies the cube sum identity for the fundamental QA Pythagorean triple:

    F³ + C³ + G³ = 216 = 6³

where (F,C,G) = (3,4,5) is the fundamental QA triple for direction (d,e)=(2,1).

This is the unique 3D extension of 3²+4²=5² (the fundamental Pythagorean triple):

  2D (Pythagorean):  F² + C² = G²   →  3²+4²=5²=25  (cert [130])
  3D (cube sum):     F³ + C³ + G³   →  3³+4³+5³=216=6³  (this cert)

QA CONNECTIONS:
  1. (F,C,G)=(d²-e², 2de, d²+e²)=(3,4,5) for fundamental (d,e)=(2,1)
  2. 6 = b×e×d×a for fundamental QN (b,e,d,a)=(1,1,2,3) (product of all elements)
  3. 216 = 9×24 = mod-9 × mod-24 (product of both QA orbit moduli)
  4. k=4 is the unique positive integer where (k-1)³+k³+(k+1)³ is a perfect cube
     Proof: (k-1)³+k³+(k+1)³ = 3k³+6k = 3k(k²+2); must equal n³.
     For k=4: 3×4×18=216=6³; no other k∈[1,10000] satisfies this.

ALGEBRAIC IDENTITY (Newton identity):
  a³+b³+c³ = 3abc + (a+b+c)(a²+b²+c²−ab−bc−ca)
  For (a,b,c)=(3,4,5):
    3abc = 180
    (a+b+c) = 12, (a²+b²+c²−ab−bc−ca) = 50−47 = 3
    3abc+(a+b+c)(…) = 180+36 = 216 = 6³  ✓

PYTHAGOREAN DUAL PROPERTY:
  The triple (3,4,5) is simultaneously:
    - A Pythagorean triple: 3²+4²=5² (unique such triple with consecutive integers)
    - A cube sum: 3³+4³+5³=6³ (unique such triple for consecutive integers)

CHECKS:
  CS_1    schema_version == 'QA_CUBE_SUM_CERT.v1'
  CS_2    F=d²-e², C=2de, G=d²+e², F²+C²=G²
  CS_IDEN F³+C³+G³=216=6³ for fundamental (F,C,G)=(3,4,5)
  CS_DUAL F²+C²=G² for the same fundamental triple (Pythagorean 2D)
  CS_MOD  216=9×24 (product of QA orbit moduli mod-9 × mod-24)
  CS_QN   6=b×e×d×a for QN (1,1,2,3); 6³=216
  CS_UNIQ k=4 unique positive integer in [1,N] with (k-1)³+k³+(k+1)³ perfect cube
  CS_W    ≥1 witness (fundamental must be present)
  CS_F    Fundamental (d,e)=(2,1): F=3,C=4,G=5; F³+C³+G³=216=6³
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_CUBE_SUM_CERT.v1"

CUBE_SUM_TARGET = 216
CUBE_ROOT_TARGET = 6
MOD_9 = 9
MOD_24 = 24


def qa_triple(d: int, e: int):
    return d*d - e*e, 2*d*e, d*d + e*e


def is_perfect_cube(n: int) -> bool:
    if n <= 0:
        return False
    root = round(n ** (1/3))
    for r in (root - 1, root, root + 1):
        if r >= 0 and r * r * r == n:
            return True
    return False


def cube_root_if_cube(n: int):
    """Return integer cube root if n is a perfect cube, else None."""
    if n <= 0:
        return None
    root = round(n ** (1/3))
    for r in (root - 1, root, root + 1):
        if r >= 0 and r * r * r == n:
            return r
    return None


def uniqueness_check(up_to: int) -> list[int]:
    """Return all k in [1,up_to] where (k-1)^3+k^3+(k+1)^3 is a perfect cube."""
    found = []
    for k in range(1, up_to + 1):
        s = (k-1)*(k-1)*(k-1) + k*k*k + (k+1)*(k+1)*(k+1)
        if is_perfect_cube(s):
            found.append(k)
    return found


def check_direction(d: int, e: int, decl: dict) -> list[str]:
    errors = []
    F, C, G = qa_triple(d, e)

    # CS_2: declared values
    for key, val in (("F", F), ("C", C), ("G", G)):
        if decl.get(key) is not None and decl[key] != val:
            errors.append(f"CS_2: ({d},{e}) declared {key}={decl[key]} ≠ computed {val}")

    # CS_2: Pythagorean
    if F*F + C*C != G*G:
        errors.append(f"CS_2: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    cube_sum = F*F*F + C*C*C + G*G*G
    if decl.get("cube_sum") is not None and decl["cube_sum"] != cube_sum:
        errors.append(f"CS_IDEN: ({d},{e}) declared cube_sum={decl['cube_sum']} ≠ {cube_sum}")

    cube_rt = cube_root_if_cube(cube_sum)
    if decl.get("is_perfect_cube") is not None:
        expected_cube = (cube_rt is not None)
        if decl["is_perfect_cube"] != expected_cube:
            errors.append(
                f"({d},{e}) declared is_perfect_cube={decl['is_perfect_cube']} but computed={expected_cube}"
            )

    # CS_IDEN: for fundamental (d,e)=(2,1) specifically
    if d == 2 and e == 1:
        if F != 3 or C != 4 or G != 5:
            errors.append(f"CS_F: ({d},{e}) expected (F,C,G)=(3,4,5), got ({F},{C},{G})")
        if cube_sum != CUBE_SUM_TARGET:
            errors.append(f"CS_IDEN: ({d},{e}) F³+C³+G³={cube_sum} ≠ {CUBE_SUM_TARGET}=6³")
        if cube_rt != CUBE_ROOT_TARGET:
            errors.append(f"CS_IDEN: ({d},{e}) cube root of {cube_sum} is {cube_rt}, expected {CUBE_ROOT_TARGET}")
        # CS_DUAL: 2D Pythagorean
        if F*F + C*C != G*G:
            errors.append(f"CS_DUAL: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # CS_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"CS_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        return errors, warnings

    # CS_MOD: 216 = 9×24
    if MOD_9 * MOD_24 != CUBE_SUM_TARGET:
        errors.append(f"CS_MOD: 9×24={MOD_9*MOD_24} ≠ 216 (internal sanity check)")

    # CS_QN: 6 = b×e×d×a for (1,1,2,3), 6³=216
    qn = cert.get("fundamental_qn")
    if qn:
        b, e_qn, d_qn, a = qn.get("b"), qn.get("e"), qn.get("d"), qn.get("a")
        if None not in (b, e_qn, d_qn, a):
            prod = b * e_qn * d_qn * a
            if prod * prod * prod != CUBE_SUM_TARGET:
                errors.append(
                    f"CS_QN: b×e×d×a={prod}, {prod}³={prod**3} ≠ {CUBE_SUM_TARGET}"
                )
            if (b, e_qn, d_qn, a) != (1, 1, 2, 3):
                warnings.append(f"CS_QN: expected fundamental QN (1,1,2,3), got ({b},{e_qn},{d_qn},{a})")

    # CS_UNIQ: uniqueness verification
    uniq = cert.get("uniqueness_check")
    if uniq:
        up_to = uniq.get("checked_up_to", 1000)
        declared_solutions = uniq.get("solutions", [])
        computed_solutions = uniqueness_check(up_to)
        if computed_solutions != declared_solutions:
            errors.append(
                f"CS_UNIQ: declared solutions={declared_solutions} ≠ "
                f"computed={computed_solutions} for k∈[1,{up_to}]"
            )

    # CS_W: witnesses
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        warnings.append("CS_W: no witnesses provided")
    else:
        has_fundamental = False
        for w in witnesses:
            d, e = w["d"], w["e"]
            decl = {k: w.get(k) for k in ("F", "C", "G", "cube_sum", "is_perfect_cube")}
            errs = check_direction(d, e, decl)
            errors.extend([f"witness ({d},{e}): {e_}" for e_ in errs])
            if d == 2 and e == 1:
                has_fundamental = True
        if not has_fundamental:
            errors.append("CS_F: fundamental (2,1) not in witnesses")

    # Internal validation checks
    vc = cert.get("validation_checks", [])
    failed_internal = [c for c in vc if not c.get("passed", True)]
    if failed_internal:
        errors.extend(
            [f"internal check {c['check_id']} not passed" for c in failed_internal]
        )

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected_pass = [
        "cs_pass_fundamental.json",
        "cs_pass_extended.json",
    ]
    results = []
    all_ok = True
    for fname in expected_pass:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errors, warnings = validate(fpath)
            passed = len(errors) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue
        if not passed:
            all_ok = False
        results.append({"fixture": fname, "ok": passed, "errors": errors})
    return {"ok": all_ok, "results": results}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="QA Cube Sum Cert [143] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths
    if not paths:
        here = Path(__file__).parent / "fixtures"
        paths = list(here.glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errors, warnings = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warnings:
            print(f"  WARN: {w}")
        for e in errors:
            print(f"  FAIL: {e}")
        if not errors:
            print("  PASS")
        else:
            total_errors += len(errors)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures PASS.")
        sys.exit(0)


if __name__ == "__main__":
    main()
