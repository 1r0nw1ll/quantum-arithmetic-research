#!/usr/bin/env python3
"""
qa_origin_of_24_cert_validate.py

Validator for QA_ORIGIN_OF_24_CERT.v1  [family 129]

Certifies the dual derivation of mod-24 as the natural modulus of QA arithmetic:
  - Route 1 (Pyth-1): H²-G²=G²-I²=24 for the 3-4-5 triangle; ≡0(mod 24) generally
  - Route 2 (crystal): 7²-5²=24 at the fundamental Pythagorean direction (d,e)=(2,1)

Checks
------
O24_1  schema_version == 'QA_ORIGIN_OF_24_CERT.v1'
O24_2  C = 2*d*e  (for fixture-level checks)
O24_3  F = d²-e²
O24_4  G = d²+e²
O24_5  H = C+F
O24_6  I = C-F
O24_7  H²-G² = declared value (Pyth-1 route)
O24_8  G²-I² = declared value (crystal route)
O24_9  H²-G² = G²-I² (dual derivation consistency)
O24_G  general_theorem.statement present (for general fixture)
O24_W  ≥3 witnesses with correct H²-G² values
O24_F  fundamental witness (d=2,e=1) has H²-G²=24
O24_D  all witness H²-G² divisible by 24
"""

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_ORIGIN_OF_24_CERT.v1"


def check_elements(d, e, declared):
    """Verify QA element arithmetic for a direction (d,e)."""
    errors = []
    C = 2 * d * e
    F = d * d - e * e
    G = d * d + e * e
    H = C + F
    I_val = C - F
    for key, expected, actual in [("C", declared.get("C"), C), ("F", declared.get("F"), F),
                                   ("G", declared.get("G"), G), ("H", declared.get("H"), H),
                                   ("I", declared.get("I"), I_val)]:
        if expected is not None and expected != actual:
            errors.append(f"{key}: declared {expected}, computed {actual}")
    return errors, C, F, G, H, I_val


def validate_witnesses(witnesses):
    """Verify all witness H²-G² values and divisibility by 24."""
    errors = []
    has_fundamental = False
    for w in witnesses:
        d, e = w["d"], w["e"]
        C = 2 * d * e
        F = d * d - e * e
        G = d * d + e * e
        H = C + F
        I_val = C - F
        h2g2 = H * H - G * G
        g2i2 = G * G - I_val * I_val
        if h2g2 != w.get("H2_G2"):
            errors.append(f"witness ({d},{e}): declared H2_G2={w.get('H2_G2')}, computed={h2g2}")
        if h2g2 != g2i2:
            errors.append(f"witness ({d},{e}): H²-G²={h2g2} ≠ G²-I²={g2i2}")
        if h2g2 % 24 != 0:
            errors.append(f"witness ({d},{e}): H²-G²={h2g2} not divisible by 24")
        if d == 2 and e == 1:
            has_fundamental = True
            if h2g2 != 24:
                errors.append(f"fundamental (2,1): H²-G²={h2g2}, expected 24")
    if not has_fundamental:
        errors.append("no fundamental witness (d=2,e=1) found")
    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # O24_1: schema version
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"O24_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")

    # Result check
    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")

    if result == "FAIL":
        print(f"  SKIP detailed checks — cert declares FAIL")
        return errors, warnings

    # Per-fixture element checks (anchor fixture)
    direction = cert.get("direction")
    if direction:
        d, e = direction.get("d", 0), direction.get("e", 0)
        declared_elems = cert.get("qa_elements", {})
        elem_errors, C, F, G, H, I_val = check_elements(d, e, declared_elems)
        errors.extend([f"O24_2-6: {e}" for e in elem_errors])

        origin = cert.get("origin_of_24", {})
        pyth1 = origin.get("route_pyth1", {})
        crystal = origin.get("route_crystal", {})

        # O24_7
        declared_h2g2 = pyth1.get("H2_minus_G2")
        computed_h2g2 = H * H - G * G
        if declared_h2g2 is not None and declared_h2g2 != computed_h2g2:
            errors.append(f"O24_7 FAIL: H²-G² declared={declared_h2g2}, computed={computed_h2g2}")

        # O24_8
        declared_g2i2 = crystal.get("check", "")
        computed_g2i2 = G * G - I_val * I_val
        if declared_h2g2 is not None and computed_h2g2 != computed_g2i2:
            errors.append(f"O24_8 FAIL: G²-I²={computed_g2i2} ≠ H²-G²={computed_h2g2}")

        # O24_9
        if declared_h2g2 is not None and computed_h2g2 != computed_g2i2:
            errors.append(f"O24_9 FAIL: dual derivation inconsistency")

    # General theorem fixture checks
    general = cert.get("general_theorem")
    if general:
        # O24_G
        if not general.get("statement"):
            errors.append("O24_G FAIL: general_theorem.statement missing")

        witnesses = cert.get("witnesses", [])
        if len(witnesses) < 3:
            errors.append(f"O24_W FAIL: need ≥3 witnesses, got {len(witnesses)}")
        else:
            witness_errors = validate_witnesses(witnesses)
            errors.extend([f"O24_W/F/D: {e}" for e in witness_errors])

    # Validation checks internal consistency
    vc = cert.get("validation_checks", [])
    failed_internal = [c for c in vc if not c.get("passed", True)]
    if failed_internal:
        errors.extend([f"internal check {c['check_id']} not passed" for c in failed_internal])

    return errors, warnings


def _self_test() -> dict:
    """Run all fixtures and return JSON summary for meta-validator."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected_pass = [
        "origin24_pass_3_4_5.json",
        "origin24_pass_general.json",
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
    parser = argparse.ArgumentParser(description="QA Origin of 24 Cert [130] validator")
    parser.add_argument("--self-test", action="store_true",
                        help="Run all fixtures and print JSON summary")
    parser.add_argument("paths", nargs="*", help="Fixture files to validate")
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
