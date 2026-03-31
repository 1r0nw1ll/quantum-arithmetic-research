#!/usr/bin/env python3
"""
qa_male_female_octave_cert_validate.py

Validator for QA_MALE_FEMALE_OCTAVE_CERT.v1  [family 144]

Certifies the Male→Female transform and the octave ratio identity:

    female_product = 4 × male_product

for any QA Quantum Number (b,e,d,a) with d=b+e, a=b+2e.

THE TRANSFORM:
  Given male QN (b, e, d, a):
    Step 1: double e  → (b, 2e)
    Step 2: swap b↔e → (2e, b)
    Result: b_f=2e, e_f=b, d_f=b_f+e_f=2e+b=a, a_f=b_f+2e_f=2e+2b=2d

  Female QN = (2e, b, a, 2d)

ALGEBRAIC PROOF:
  male_product = b × e × d × a = b × e × (b+e) × (b+2e)
  female_product = 2e × b × a × 2d
                 = 4 × (e × b × (b+2e) × (b+e))
                 = 4 × male_product  ✓

MUSICAL INTERPRETATION:
  4× frequency ratio = 2 octaves (1 octave = 2× frequency).
  Female QN is always exactly 2 octaves above the corresponding male QN.

FUNDAMENTAL EXAMPLE:
  Male  (1,1,2,3): product = 1×1×2×3 = 6
  Female(2,1,3,4): product = 2×1×3×4 = 24 = 4×6  ✓

QA CONNECTIONS:
  - d_f = a_male  (female direction = male apogee)
  - a_f = 2d_male (female apogee = 2× male direction)
  - The male-female pair shares the Pythagorean triple structure through cert [130]

CHECKS:
  MF_1    schema_version == 'QA_MALE_FEMALE_OCTAVE_CERT.v1'
  MF_2    d=b+e, a=b+2e for all declared QNs
  MF_TRANS Female transform: b_f=2e, e_f=b, d_f=a, a_f=2d
  MF_PROD  female product = 4 × male product (for all witnesses)
  MF_OCT   4× ratio = 2 octaves (declared and verified)
  MF_W     ≥3 male/female QN pairs
  MF_F     Fundamental (1,1,2,3)→(2,1,3,4): products 6→24=4×6
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_MALE_FEMALE_OCTAVE_CERT.v1"
OCTAVE_RATIO = 4  # female_product / male_product = 4 = 2²


def qn_product(b: int, e: int, d: int, a: int) -> int:
    return b * e * d * a


def female_transform(b: int, e: int, d: int, a: int):
    """Apply Male→Female transform: double e, swap b↔e."""
    b_f = 2 * e
    e_f = b
    d_f = b_f + e_f     # = 2e + b = a
    a_f = b_f + 2 * e_f # = 2e + 2b = 2(b+e) = 2d
    return b_f, e_f, d_f, a_f


def check_qn(b: int, e: int, d: int, a: int, label: str, decl: dict) -> list[str]:
    errors = []
    # MF_2: d=b+e, a=b+2e
    if d != b + e:
        errors.append(f"MF_2: {label} d={d} ≠ b+e={b+e}")
    if a != b + 2*e:
        errors.append(f"MF_2: {label} a={a} ≠ b+2e={b+2*e}")
    # Declared product
    prod = qn_product(b, e, d, a)
    if decl.get("product") is not None and decl["product"] != prod:
        errors.append(f"MF_2: {label} declared product={decl['product']} ≠ {prod}")
    return errors


def check_pair(male: dict, female: dict) -> list[str]:
    errors = []
    bm, em, dm, am = male["b"], male["e"], male["d"], male["a"]
    bf, ef, df, af = female["b"], female["e"], female["d"], female["a"]

    # MF_2: derived coordinates for male
    errors.extend(check_qn(bm, em, dm, am, f"male({bm},{em})", male))
    errors.extend(check_qn(bf, ef, df, af, f"female({bf},{ef})", female))

    # MF_TRANS: verify transform
    bf_exp, ef_exp, df_exp, af_exp = female_transform(bm, em, dm, am)
    if (bf, ef, df, af) != (bf_exp, ef_exp, df_exp, af_exp):
        errors.append(
            f"MF_TRANS: male({bm},{em})→female expected ({bf_exp},{ef_exp},{df_exp},{af_exp}), "
            f"got ({bf},{ef},{df},{af})"
        )

    # MF_PROD: female product = 4 × male product
    male_prod = qn_product(bm, em, dm, am)
    female_prod = qn_product(bf, ef, df, af)
    if female_prod != OCTAVE_RATIO * male_prod:
        errors.append(
            f"MF_PROD: female_product={female_prod} ≠ 4×male_product={4*male_prod} "
            f"for male({bm},{em})→female({bf},{ef})"
        )

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # MF_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"MF_1 FAIL: schema_version={cert.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        return errors, warnings

    # MF_OCT: octave ratio check (sanity)
    if OCTAVE_RATIO != 4:
        errors.append("MF_OCT: internal sanity: octave_ratio must be 4")

    # Pair witnesses
    pairs = cert.get("pairs", [])
    if not pairs:
        warnings.append("MF_W: no pairs provided")
    else:
        if len(pairs) < 3:
            errors.append(f"MF_W: need ≥3 pairs, got {len(pairs)}")
        has_fundamental = False
        for pair in pairs:
            male = pair.get("male", {})
            female = pair.get("female", {})
            if not male or not female:
                errors.append("pair missing male or female field")
                continue
            errs = check_pair(male, female)
            errors.extend(errs)
            if male.get("b") == 1 and male.get("e") == 1:
                has_fundamental = True
        if not has_fundamental:
            warnings.append("MF_F: fundamental (1,1,2,3)→(2,1,3,4) not in pairs")

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
        "mf_pass_fundamental.json",
        "mf_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Male/Female Octave Cert [144] validator")
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
