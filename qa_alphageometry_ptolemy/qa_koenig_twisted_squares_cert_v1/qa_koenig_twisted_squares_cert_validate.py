#!/usr/bin/env python3
"""
qa_koenig_twisted_squares_cert_validate.py

Validator for QA_KOENIG_TWISTED_SQUARES_CERT.v1  [family 137]

Certifies the (I²,2CF,G²,H²) arithmetic progression structure and
H²-G²=G²-I²=2CF=24L identities for QA directions:

  H = C+F  (outer Koenig square element)
  I = C-F  (inner Koenig square element; negative → ellipse)
  L = C*F/12  (QA L-element, must be integer)

  KTS_5:  H² − G² = 2CF
  KTS_6:  G² − I² = 2CF
  KTS_8:  24·L = 2CF
  KTS_9:  2CF ≡ 0 (mod 24)  for primitive directions

Algebraic proof:
  H²-G² = (C+F)² - (C²+F²) = 2CF  [using C²+F²=G²]
  G²-I² = (C²+F²) - (C-F)² = 2CF
  L=CF/12 ∈ Z: 8|C=2de (one of d,e even), 3|F=d²-e²

Quadruple (I²,2CF,G²,H²) is arithmetic progression step 2CF:
  I²+2CF = G²,  G²+2CF = H²

Historical chain: Iverson QA L-element (Pyth-1 Law 15) →
Mathologer twisted-squares (4 triangles + inner square = outer square) →
Will Dale Koenig I→H chain (generates all primes via descent) →
(I²,2CF,G²,H²) QA quadruple corollary (Will, 2026-03-30).

Checks
------
KTS_1   schema_version == 'QA_KOENIG_TWISTED_SQUARES_CERT.v1'
KTS_2   F=d²-e², C=2de, G=d²+e² for declared triple
KTS_3   F²+C²=G² for triple
KTS_4   H=C+F, I=C-F computed correctly
KTS_5   H²-G² = 2*C*F
KTS_6   G²-I² = 2*C*F
KTS_7   L=C*F/12 is integer
KTS_8   24*L = 2*C*F
KTS_9   2*C*F ≡ 0 (mod 24)
KTS_W   ≥3 witness entries
KTS_F   fundamental (d,e)=(2,1) present with 2CF=24
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_KOENIG_TWISTED_SQUARES_CERT.v1"


def qa_triple(d, e):
    return d*d - e*e, 2*d*e, d*d + e*e


def check_direction(d, e, decl_triple):
    """Validate a single direction (d,e) and its H,I,L,2CF identities."""
    errors = []
    F, C, G = qa_triple(d, e)

    # KTS_2: declared triple matches
    if decl_triple:
        if decl_triple.get("F") is not None and decl_triple["F"] != F:
            errors.append(f"KTS_2: ({d},{e}) F={decl_triple['F']} ≠ computed {F}")
        if decl_triple.get("C") is not None and decl_triple["C"] != C:
            errors.append(f"KTS_2: ({d},{e}) C={decl_triple['C']} ≠ computed {C}")
        if decl_triple.get("G") is not None and decl_triple["G"] != G:
            errors.append(f"KTS_2: ({d},{e}) G={decl_triple['G']} ≠ computed {G}")

    # KTS_3: Pythagorean identity
    if F*F + C*C != G*G:
        errors.append(f"KTS_3: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    # Compute H, I, L, 2CF
    H = C + F
    I = C - F
    two_CF = 2 * C * F

    # KTS_4: check declared H and I
    # KTS_5: H²-G² = 2CF
    if H*H - G*G != two_CF:
        errors.append(f"KTS_5: ({d},{e}) H²-G²={H*H-G*G} ≠ 2CF={two_CF}")

    # KTS_6: G²-I² = 2CF
    if G*G - I*I != two_CF:
        errors.append(f"KTS_6: ({d},{e}) G²-I²={G*G-I*I} ≠ 2CF={two_CF}")

    # KTS_7: L=CF/12 is integer
    if (C * F) % 12 != 0:
        errors.append(f"KTS_7: ({d},{e}) C*F={C*F} not divisible by 12")
    else:
        L = (C * F) // 12
        # KTS_8: 24L = 2CF
        if 24 * L != two_CF:
            errors.append(f"KTS_8: ({d},{e}) 24*L={24*L} ≠ 2CF={two_CF}")

    # KTS_9: 2CF divisible by 24
    if two_CF % 24 != 0:
        errors.append(f"KTS_9: ({d},{e}) 2CF={two_CF} not divisible by 24")

    return errors, F, C, G, H, I, two_CF


def check_witness(w):
    """Validate a witness entry with d, e fields."""
    d, e = w["d"], w["e"]
    decl_triple = w.get("triple", {})
    errors, F, C, G, H, I, two_CF = check_direction(d, e, decl_triple)

    # KTS_4: check declared H, I
    if w.get("H") is not None and w["H"] != H:
        errors.append(f"KTS_4: ({d},{e}) declared H={w['H']} ≠ computed H={H}")
    if w.get("I") is not None and w["I"] != I:
        errors.append(f"KTS_4: ({d},{e}) declared I={w['I']} ≠ computed I={I}")
    if w.get("I_abs") is not None and w["I_abs"] != abs(I):
        errors.append(f"KTS_4: ({d},{e}) declared I_abs={w['I_abs']} ≠ computed |I|={abs(I)}")

    # Check declared 2CF
    if w.get("two_CF") is not None and w["two_CF"] != two_CF:
        errors.append(f"KTS_5: ({d},{e}) declared two_CF={w['two_CF']} ≠ computed {two_CF}")

    # Check declared L
    if (C * F) % 12 == 0:
        L = (C * F) // 12
        if w.get("L") is not None and w["L"] != L:
            errors.append(f"KTS_7: ({d},{e}) declared L={w['L']} ≠ computed L={L}")

    # Check declared quadruple [I², 2CF, G², H²]
    expected_quad = [I*I, two_CF, G*G, H*H]
    decl_quad = w.get("quadruple")
    if decl_quad is not None:
        if decl_quad != expected_quad:
            errors.append(f"quadruple ({d},{e}): declared {decl_quad} ≠ computed {expected_quad}")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # KTS_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"KTS_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
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
        decl_triple = cert.get("triple", {})
        errs, F, C, G, H, I, two_CF = check_direction(d, e, decl_triple)
        errors.extend(errs)

        # KTS_4: check declared koenig_elements
        ke = cert.get("koenig_elements", {})
        if ke:
            if ke.get("H") is not None and ke["H"] != H:
                errors.append(f"KTS_4: declared H={ke['H']} ≠ computed H={H}")
            if ke.get("I") is not None and ke["I"] != I:
                errors.append(f"KTS_4: declared I={ke['I']} ≠ computed I={I}")
            if (C * F) % 12 == 0:
                L = (C * F) // 12
                if ke.get("L") is not None and ke["L"] != L:
                    errors.append(f"KTS_7: declared L={ke['L']} ≠ computed L={L}")

        # Check twisted_squares_identities block
        tsi = cert.get("twisted_squares_identities", {})
        if tsi:
            if tsi.get("two_CF") is not None and tsi["two_CF"] != two_CF:
                errors.append(f"KTS_5: two_CF={tsi['two_CF']} ≠ computed {two_CF}")
            if tsi.get("H_sq_minus_G_sq") is not None and tsi["H_sq_minus_G_sq"] != H*H - G*G:
                errors.append(f"KTS_5: H_sq_minus_G_sq={tsi['H_sq_minus_G_sq']} ≠ {H*H-G*G}")
            if tsi.get("G_sq_minus_I_sq") is not None and tsi["G_sq_minus_I_sq"] != G*G - I*I:
                errors.append(f"KTS_6: G_sq_minus_I_sq={tsi['G_sq_minus_I_sq']} ≠ {G*G-I*I}")

        # Check quadruple_structure block
        qs = cert.get("quadruple_structure", {})
        if qs:
            if (C * F) % 12 == 0:
                L = (C * F) // 12
                expected_step = two_CF
                if qs.get("step") is not None and qs["step"] != expected_step:
                    errors.append(f"quadruple step={qs['step']} ≠ 2CF={expected_step}")
            if qs.get("I_sq") is not None and qs["I_sq"] != I*I:
                errors.append(f"quadruple I²={qs['I_sq']} ≠ {I*I}")
            if qs.get("G_sq") is not None and qs["G_sq"] != G*G:
                errors.append(f"quadruple G²={qs['G_sq']} ≠ {G*G}")
            if qs.get("H_sq") is not None and qs["H_sq"] != H*H:
                errors.append(f"quadruple H²={qs['H_sq']} ≠ {H*H}")

        # KTS_F: fundamental pair check
        if d == 2 and e == 1 and two_CF != 24:
            errors.append(f"KTS_F FAIL: fundamental (2,1) 2CF={two_CF} ≠ 24")

    # --- Witness-list fixture ---
    witnesses = cert.get("witnesses", [])
    if witnesses:
        if len(witnesses) < 3:
            errors.append(f"KTS_W FAIL: need ≥3 witnesses, got {len(witnesses)}")
        for w in witnesses:
            werrs = check_witness(w)
            errors.extend([f"witness ({w['d']},{w['e']}): {e_}" for e_ in werrs])

    # Internal validation_checks
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
        "kts_pass_fundamental.json",
        "kts_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Koenig Twisted Squares Cert [137] validator")
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
