#!/usr/bin/env python3
"""
qa_conic_discriminant_cert_validate.py

Validator for QA_CONIC_DISCRIMINANT_CERT.v1  [family 140]

Certifies I = C - F = Qg(d,e) - Qr(d,e) as the QA conic discriminant:

  I = C - F = 2de - (d²-e²)

  I > 0  →  hyperbola  (C > F, d/e < 1+√2)
  I = 0  →  parabola   (IMPOSSIBLE for integers — requires d/e = 1+√2)
  I < 0  →  ellipse    (C < F, d/e > 1+√2)

Parabola impossibility: I=0 requires d²-2de-e²=0 → (d/e)=1+√2,
the silver ratio. Since √2 is irrational, no integer pair (d,e) satisfies
this. Proof: the quadratic x²-2x-1=0 has discriminant 8 (not a perfect
square) → no rational roots.

The silver-ratio continued fraction [2;2,2,2,...] has convergents
2/1, 5/2, 12/5, 29/12, 70/29, ... which alternate H(I=1)/E(I=-1),
each step halving the gap |d/e - (1+√2)|.

Chromogeometry connection (cert [125]):
  I = Qg(d,e) - Qr(d,e) = green minus red quadrance
  sign(I) = dominance of green (rotation) vs red (stretch)

Checks
------
CD_1      schema_version == 'QA_CONIC_DISCRIMINANT_CERT.v1'
CD_2      I = C-F computed correctly for each direction
CD_3      conic_type matches sign(I): I>0→hyperbola, I<0→ellipse
CD_4      F²+C²=G² for each triple
CD_PARA   parabola_impossibility block present
CD_W      ≥3 witnesses of each type (hyperbola and ellipse)
CD_F      fundamental (d,e)=(2,1) present as hyperbola (I=1)
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_CONIC_DISCRIMINANT_CERT.v1"

CONIC_TYPE = {1: "hyperbola", 0: "parabola", -1: "ellipse"}


def qa_triple(d: int, e: int):
    return d*d - e*e, 2*d*e, d*d + e*e


def sign(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def check_direction(d: int, e: int, decl: dict) -> list[str]:
    errors = []
    F, C, G = qa_triple(d, e)
    I = C - F

    # CD_2: declared F, C, G, I
    for key, val in (("F", F), ("C", C), ("G", G)):
        if decl.get(key) is not None and decl[key] != val:
            errors.append(f"CD_2: ({d},{e}) {key}={decl[key]} ≠ computed {val}")
    if decl.get("I") is not None and decl["I"] != I:
        errors.append(f"CD_2: ({d},{e}) I={decl['I']} ≠ computed {I}")

    # CD_3: conic_type matches sign(I)
    expected_type = CONIC_TYPE[sign(I)]
    declared_type = decl.get("conic_type")
    if declared_type is not None and declared_type != expected_type:
        errors.append(
            f"CD_3: ({d},{e}) I={I} → expected '{expected_type}', declared '{declared_type}'"
        )

    # CD_4: Pythagorean
    if F*F + C*C != G*G:
        errors.append(f"CD_4: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    # No parabola in integer QA
    if I == 0:
        errors.append(
            f"PARA: ({d},{e}) I=0 — parabola should be impossible for primitive integer directions"
        )

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # CD_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"CD_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        return errors, warnings

    # --- Single-direction list (fundamental fixture) ---
    directions = cert.get("directions", [])
    if directions:
        has_fundamental = False
        for d_entry in directions:
            d, e = d_entry["d"], d_entry["e"]
            errors.extend(check_direction(d, e, d_entry))
            if d == 2 and e == 1:
                has_fundamental = True
                F, C, G = qa_triple(d, e)
                I = C - F
                if I != 1:
                    errors.append(f"CD_F: fundamental (2,1) I={I} ≠ 1")
        if not has_fundamental:
            errors.append("CD_F: fundamental (d,e)=(2,1) not found in directions")

        # CD_PARA: parabola impossibility block
        if not cert.get("parabola_impossibility"):
            errors.append("CD_PARA: parabola_impossibility block missing")

    # --- Witness-list fixture ---
    hyp_witnesses = cert.get("hyperbola_witnesses", [])
    ell_witnesses = cert.get("ellipse_witnesses", [])

    if hyp_witnesses or ell_witnesses:
        if len(hyp_witnesses) < 3:
            errors.append(f"CD_W: need ≥3 hyperbola witnesses, got {len(hyp_witnesses)}")
        if len(ell_witnesses) < 3:
            errors.append(f"CD_W: need ≥3 ellipse witnesses, got {len(ell_witnesses)}")

        has_fundamental = False
        for w in hyp_witnesses:
            d, e = w["d"], w["e"]
            errors.extend([f"hyp ({d},{e}): {e_}" for e_ in check_direction(d, e, w)])
            # All hyperbola witnesses must have I>0
            F, C, G = qa_triple(d, e)
            I = C - F
            if I <= 0:
                errors.append(f"CD_3: hyperbola witness ({d},{e}) has I={I} ≤ 0")
            if d == 2 and e == 1:
                has_fundamental = True

        for w in ell_witnesses:
            d, e = w["d"], w["e"]
            errors.extend([f"ell ({d},{e}): {e_}" for e_ in check_direction(d, e, w)])
            # All ellipse witnesses must have I<0
            F, C, G = qa_triple(d, e)
            I = C - F
            if I >= 0:
                errors.append(f"CD_3: ellipse witness ({d},{e}) has I={I} ≥ 0")

        if not has_fundamental:
            warnings.append("CD_F: fundamental (2,1) not in hyperbola witnesses")

        # Validate convergent_sequence if present
        seq = cert.get("convergent_sequence", {})
        convs = seq.get("convergents", [])
        for cv in convs:
            d, e = cv["d"], cv["e"]
            F, C, G = qa_triple(d, e)
            I_comp = C - F
            if cv.get("I") is not None and cv["I"] != I_comp:
                errors.append(f"convergent ({d},{e}): declared I={cv['I']} ≠ computed {I_comp}")
            expected_type = "H" if I_comp > 0 else ("E" if I_comp < 0 else "P")
            if cv.get("type") is not None and cv["type"] != expected_type:
                errors.append(f"convergent ({d},{e}): declared type={cv['type']} ≠ {expected_type}")

    # Internal validation_checks
    vc = cert.get("validation_checks", [])
    failed_internal = [c for c in vc if not c.get("passed", True)]
    if failed_internal:
        errors.extend([f"internal check {c['check_id']} not passed" for c in failed_internal])

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected_pass = [
        "cd_pass_fundamental.json",
        "cd_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Conic Discriminant Cert [140] validator")
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
