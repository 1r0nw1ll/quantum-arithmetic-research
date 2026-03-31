#!/usr/bin/env python3
"""
qa_48_64_cert_validate.py

Validator for QA_48_64_CERT.v1  [family 139]

Certifies two structural constants of QA — 48 and 64 — via:

ALGEBRAIC (cert-grade, universal):
  48L = H²-I² = 4CF  for ALL QA directions (d,e)
  where H=C+F, I=C-F, L=CF/12 (QA L-element)
  Proof: (C+F)²-(C-F)² = 4CF = 48·(CF/12)
  Minimum 48 at fundamental (2,1) with L=1.

ORBIT CONSTANTS (QA system, reference cert [128]):
  48 = 2 × cosmos_period = 2 × 24
  64 = satellite_period² = 8²
  Ratio 48/64 = 3/4 = equilateral spread in Wildberger chromogeometry.

POLYNOMIAL (equilateral null triangle):
  The symmetric solution P=R=T=4 satisfies PR+RT+PT=48 and PRT=64.
  Polynomial: x³-12x²+48x-64 = (x-4)³ (triple root — equilateral).
  This is the UNIQUE positive integer solution with P=R=T.
  Represents the equilateral null triangle (Eisenstein symmetry, cert [133]).

Checks
------
C4864_1     schema_version == 'QA_48_64_CERT.v1'
C4864_2     F=d²-e², C=2de, G=d²+e² for each declared triple
C4864_3     F²+C²=G² for each triple
C4864_ALG   48L = H²-I² = 4CF for each direction
C4864_POLY  Equilateral (4,4,4): PR+RT+PT=48, PRT=64
C4864_ORB   48=2×24, 64=8² stated in cert
C4864_W     ≥3 algebraic witnesses
C4864_F     Fundamental (d,e)=(2,1) present with 48L=48
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_48_64_CERT.v1"


def qa_triple(d: int, e: int):
    return d*d - e*e, 2*d*e, d*d + e*e


def check_direction_48(d: int, e: int, decl: dict) -> list[str]:
    """Validate 48L = H²-I² = 4CF for direction (d,e)."""
    errors = []
    F, C, G = qa_triple(d, e)

    # C4864_2: declared triple
    for key, val in (("F", F), ("C", C), ("G", G)):
        if decl.get(key) is not None and decl[key] != val:
            errors.append(f"C4864_2: ({d},{e}) {key}={decl[key]} ≠ computed {val}")

    # C4864_3: Pythagorean
    if F*F + C*C != G*G:
        errors.append(f"C4864_3: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    # Compute H, I, L, values
    H = C + F
    I = C - F
    four_CF = 4 * C * F
    H2_I2 = H*H - I*I

    # C4864_ALG: 48L = H²-I² = 4CF
    if H2_I2 != four_CF:
        errors.append(f"C4864_ALG: ({d},{e}) H²-I²={H2_I2} ≠ 4CF={four_CF}")
    if (C * F) % 12 != 0:
        errors.append(f"C4864_ALG: ({d},{e}) C*F={C*F} not divisible by 12 → L not integer")
    else:
        L = (C * F) // 12
        if 48 * L != four_CF:
            errors.append(f"C4864_ALG: ({d},{e}) 48*L={48*L} ≠ 4CF={four_CF}")

    # Check declared values
    if decl.get("H") is not None and decl["H"] != H:
        errors.append(f"({d},{e}) declared H={decl['H']} ≠ {H}")
    if decl.get("I") is not None and decl["I"] != I:
        errors.append(f"({d},{e}) declared I={decl['I']} ≠ {I}")
    if (C * F) % 12 == 0:
        L = (C * F) // 12
        if decl.get("L") is not None and decl["L"] != L:
            errors.append(f"({d},{e}) declared L={decl['L']} ≠ {L}")
        if decl.get("forty_eight_L") is not None and decl["forty_eight_L"] != 48 * L:
            errors.append(f"({d},{e}) declared 48L={decl['forty_eight_L']} ≠ {48*L}")
    if decl.get("H_sq_minus_I_sq") is not None and decl["H_sq_minus_I_sq"] != H2_I2:
        errors.append(f"({d},{e}) declared H²-I²={decl['H_sq_minus_I_sq']} ≠ {H2_I2}")
    if decl.get("four_CF") is not None and decl["four_CF"] != four_CF:
        errors.append(f"({d},{e}) declared 4CF={decl['four_CF']} ≠ {four_CF}")

    return errors


def check_polynomial(poly_block: dict) -> list[str]:
    """Validate the equilateral polynomial block."""
    errors = []
    P = poly_block.get("P")
    R = poly_block.get("R")
    T = poly_block.get("T")
    if P is None or R is None or T is None:
        return []  # optional block

    e2 = P*R + R*T + P*T
    e3 = P*R*T

    if poly_block.get("e2_PR_RT_PT") is not None and poly_block["e2_PR_RT_PT"] != e2:
        errors.append(f"C4864_POLY: declared e2={poly_block['e2_PR_RT_PT']} ≠ computed {e2}")
    if e2 != 48:
        errors.append(f"C4864_POLY: PR+RT+PT={e2} ≠ 48")

    if poly_block.get("e3_PRT") is not None and poly_block["e3_PRT"] != e3:
        errors.append(f"C4864_POLY: declared e3={poly_block['e3_PRT']} ≠ computed {e3}")
    if e3 != 64:
        errors.append(f"C4864_POLY: PRT={e3} ≠ 64")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # C4864_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"C4864_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        print("  SKIP detailed checks — cert declares FAIL")
        return errors, warnings

    # --- Single-direction fixture (fundamental) ---
    direction = cert.get("direction")
    if direction:
        d, e = direction["d"], direction["e"]
        decl = {}
        decl.update(cert.get("triple", {}))
        decl.update(cert.get("koenig_elements", {}))
        alg = cert.get("algebraic_48", {})
        decl.update({
            "forty_eight_L": alg.get("forty_eight_L"),
            "H_sq_minus_I_sq": alg.get("H_sq_minus_I_sq"),
            "four_CF": alg.get("four_CF"),
        })
        errors.extend(check_direction_48(d, e, decl))

        # C4864_F
        if d == 2 and e == 1:
            F, C, G = qa_triple(d, e)
            L = (C * F) // 12
            if 48 * L != 48:
                errors.append(f"C4864_F: fundamental (2,1) 48L={48*L} ≠ 48")

        # C4864_POLY
        poly_block = cert.get("equilateral_polynomial", {})
        errors.extend(check_polynomial(poly_block))

        # C4864_ORB: check orbit constants present
        orb = cert.get("orbit_constants", {})
        if orb:
            if orb.get("cosmos_period") != 24:
                errors.append(f"C4864_ORB: cosmos_period={orb.get('cosmos_period')} ≠ 24")
            if orb.get("satellite_period") != 8:
                errors.append(f"C4864_ORB: satellite_period={orb.get('satellite_period')} ≠ 8")
            if orb.get("forty_eight") != 48:
                errors.append(f"C4864_ORB: forty_eight={orb.get('forty_eight')} ≠ 48")
            if orb.get("sixty_four") != 64:
                errors.append(f"C4864_ORB: sixty_four={orb.get('sixty_four')} ≠ 64")

    # --- Witness-list fixture ---
    witnesses = cert.get("algebraic_witnesses", [])
    if witnesses:
        if len(witnesses) < 3:
            errors.append(f"C4864_W FAIL: need ≥3 witnesses, got {len(witnesses)}")
        has_fundamental = False
        for w in witnesses:
            wdecl = {k: w.get(k) for k in ("F","C","G","H","I","L","forty_eight_L","H_sq_minus_I_sq","four_CF")}
            errs = check_direction_48(w["d"], w["e"], wdecl)
            errors.extend([f"witness ({w['d']},{w['e']}): {e_}" for e_ in errs])
            if w["d"] == 2 and w["e"] == 1:
                has_fundamental = True
        if not has_fundamental:
            warnings.append("C4864_F: fundamental (d,e)=(2,1) not found in witnesses")

    # Polynomial witnesses
    poly_witnesses = cert.get("polynomial_witnesses", [])
    for pw in poly_witnesses:
        if pw.get("P") is not None:
            errors.extend(check_polynomial(pw))

    # Internal validation_checks
    vc = cert.get("validation_checks", [])
    failed_internal = [c for c in vc if not c.get("passed", True)]
    if failed_internal:
        errors.extend([f"internal check {c['check_id']} not passed" for c in failed_internal])

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected_pass = [
        "c4864_pass_fundamental.json",
        "c4864_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA 48/64 Cert [139] validator")
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
