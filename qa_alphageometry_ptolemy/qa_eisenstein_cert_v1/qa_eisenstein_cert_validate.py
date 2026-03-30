#!/usr/bin/env python3
"""
qa_eisenstein_cert_validate.py

Validator for QA_EISENSTEIN_CERT.v1  [family 133]

Certifies two universal Eisenstein-norm identities arising from QA elements:

  Identity 1:  F² − F·W + W² = Z²
  Identity 2:  Y² − Y·W + W² = Z²

where, for ALL QA tuples (b,e,d,a) with d=b+e, a=b+2e:
  F = a·b        (semi-latus rectum)
  W = d·(e+a)    (equilateral side; QA Law 15)
  Z = e² + a·d   (Eisenstein companion; QA Law 15)
  Y = a² − d²    (= e(2b+3e); A−D element)

Algebraic proof: let u = b²+3be.
  F+W = b(b+2e)+(b+e)(b+3e) = 2u+3e²
  FW  = b(b+2e)·(b+e)(b+3e) = u(u+2e²)
  Z   = b²+3be+3e² = u+3e²
  (F+W)²−3FW = (2u+3e²)²−3u(u+2e²) = u²+6ue²+9e⁴ = (u+3e²)² = Z²  QED

Checks
------
EIS_1  schema_version == 'QA_EISENSTEIN_CERT.v1'
EIS_2  F = a·b
EIS_3  W = d·(e+a)
EIS_4  Z = e·e + a·d
EIS_5  Y = a·a − d·d
EIS_6  F·F − F·W + W·W = Z·Z
EIS_7  Y·Y − Y·W + W·W = Z·Z
EIS_W  ≥3 witnesses present (witness fixture)
EIS_U  fundamental witness (b,e,d,a)=(1,1,2,3) present
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_EISENSTEIN_CERT.v1"


def check_tuple_elements(b, e, d, a, F, W, Z, Y):
    """Verify element formulas and Eisenstein identities for a single tuple."""
    errors = []

    # EIS_2: F = a*b
    exp_F = a * b
    if F != exp_F:
        errors.append(f"EIS_2 ({b},{e},{d},{a}): F={F}, expected a*b={exp_F}")

    # EIS_3: W = d*(e+a)
    exp_W = d * (e + a)
    if W != exp_W:
        errors.append(f"EIS_3 ({b},{e},{d},{a}): W={W}, expected d*(e+a)={exp_W}")

    # EIS_4: Z = e*e + a*d
    exp_Z = e * e + a * d
    if Z != exp_Z:
        errors.append(f"EIS_4 ({b},{e},{d},{a}): Z={Z}, expected e²+a*d={exp_Z}")

    # EIS_5: Y = a*a - d*d
    exp_Y = a * a - d * d
    if Y != exp_Y:
        errors.append(f"EIS_5 ({b},{e},{d},{a}): Y={Y}, expected a²−d²={exp_Y}")

    # EIS_6: F² - F*W + W² = Z²
    lhs6 = F * F - F * W + W * W
    rhs6 = Z * Z
    if lhs6 != rhs6:
        errors.append(f"EIS_6 ({b},{e},{d},{a}): F²-FW+W²={lhs6} ≠ Z²={rhs6}")

    # EIS_7: Y² - Y*W + W² = Z²
    lhs7 = Y * Y - Y * W + W * W
    rhs7 = Z * Z
    if lhs7 != rhs7:
        errors.append(f"EIS_7 ({b},{e},{d},{a}): Y²-YW+W²={lhs7} ≠ Z²={rhs7}")

    # Also check d=b+e and a=b+2e (tuple validity)
    if d != b + e:
        errors.append(f"tuple ({b},{e},{d},{a}): d={d} ≠ b+e={b+e}")
    if a != b + 2 * e:
        errors.append(f"tuple ({b},{e},{d},{a}): a={a} ≠ b+2e={b+2*e}")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # EIS_1: schema version
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"EIS_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        print("  SKIP detailed checks — cert declares FAIL")
        return errors, warnings

    # --- Single-tuple fixture (fundamental) ---
    if "tuple" in cert and "elements" in cert:
        t = cert["tuple"]
        b, e, d, a = t["b"], t["e"], t["d"], t["a"]
        el = cert["elements"]
        F, W, Z, Y = el["F"], el["W"], el["Z"], el["Y"]
        errs = check_tuple_elements(b, e, d, a, F, W, Z, Y)
        errors.extend(errs)

        # Also cross-check eisenstein_fwz / eisenstein_ywz blocks if present
        for block_key, id_name, lhs_fn in [
            ("eisenstein_fwz", "EIS_6", lambda F, W, Z: F * F - F * W + W * W == Z * Z),
            ("eisenstein_ywz", "EIS_7", lambda Y, W, Z: Y * Y - Y * W + W * W == Z * Z),
        ]:
            blk = cert.get(block_key)
            if blk:
                triple = blk.get("triple", [])
                if len(triple) == 3:
                    p, q, r = triple
                    if block_key == "eisenstein_fwz" and not (
                        p * p - p * q + q * q == r * r
                    ):
                        errors.append(f"{id_name}: triple {triple} fails a²-ab+b²=c²")
                    elif block_key == "eisenstein_ywz" and not (
                        p * p - p * q + q * q == r * r
                    ):
                        errors.append(f"{id_name}: triple {triple} fails a²-ab+b²=c²")

    # --- Witness-list fixture ---
    witnesses = cert.get("witnesses", [])
    if witnesses:
        if len(witnesses) < 3:
            errors.append(f"EIS_W FAIL: need ≥3 witnesses, got {len(witnesses)}")

        has_fundamental = False
        for w in witnesses:
            b, e, d, a = w["b"], w["e"], w["d"], w["a"]
            F, W, Z, Y = w["F"], w["W"], w["Z"], w["Y"]
            werr = check_tuple_elements(b, e, d, a, F, W, Z, Y)
            errors.extend(werr)
            if b == 1 and e == 1 and d == 2 and a == 3:
                has_fundamental = True

        if len(witnesses) >= 3 and not has_fundamental:
            errors.append("EIS_U FAIL: fundamental witness (1,1,2,3) not found")

    # Internal validation_checks block
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
        "eisenstein_pass_fundamental.json",
        "eisenstein_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Eisenstein Cert [133] validator")
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
            print(f"  PASS")
        else:
            total_errors += len(errors)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print(f"\nAll fixtures PASS.")
        sys.exit(0)


if __name__ == "__main__":
    main()
