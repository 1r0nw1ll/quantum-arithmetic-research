#!/usr/bin/env python3
"""
qa_plimpton322_cert_validate.py

Validator for QA_PLIMPTON322_CERT.v1  [family 138]

Certifies that Babylonian tablet Plimpton 322 (~1800 BCE) encodes QA
chromogeometric triples. Each row is a QA direction (d,e) with both d,e
"regular" (5-smooth: only factors 2,3,5), generating:

  F = d² - e²   (short side β, red quadrance Qr)
  C = 2de        (middle side, green quadrance Qg)
  G = d² + e²   (diagonal δ, blue quadrance Qb)

Key theorem: if d and e are regular, then C=2de is regular, so the
denominator of G/C (in lowest terms) is also 5-smooth, meaning G/C
terminates in sexagesimal base-60 notation — giving exact Babylonian
arithmetic.

Plimpton column 1 = (G/C)² decreases from Row 1 to Row 15.
SPVN no-zero: F,C,G > 0 (QA A1).

Historical: tablet ~1800 BCE Old Babylonian. Mansfield & Wildberger 2017
(Historia Mathematica 44:395-419) identified it as a table of exact
rational trigonometric values. QA connection: F=Qr(d,e), C=Qg(d,e),
G=Qb(d,e) — Wildberger Chromogeometry Thm 6: F²+C²=G².

Checks
------
P322_1      schema_version == 'QA_PLIMPTON322_CERT.v1'
P322_2      F=d²-e², C=2de, G=d²+e² for declared triple
P322_3      F²+C²=G² (Pythagorean identity)
P322_4      gcd(d,e)=1, d>e, d-e odd (primitive direction)
P322_REG    d and e are 5-smooth (regular in base-60)
P322_BASE60 G/C terminates in base-60 (denominator is 5-smooth)
P322_NOZERO F,C,G > 0 (SPVN no-zero = QA A1)
P322_W      ≥3 witness rows
P322_F      fundamental (d,e)=(2,1) present
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_PLIMPTON322_CERT.v1"


def is_regular(n: int) -> bool:
    """5-smooth: only prime factors 2, 3, 5."""
    if n <= 0:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def base60_terminates(num: int, den: int) -> bool:
    """G/C = num/den terminates in base-60 iff denominator (reduced) is 5-smooth."""
    if den == 0:
        return False
    g = gcd(abs(num), abs(den))
    return is_regular(den // g)


def qa_triple(d: int, e: int):
    return d*d - e*e, 2*d*e, d*d + e*e


def check_row(d: int, e: int, decl_triple: dict) -> list[str]:
    """Validate all P322 checks for a single row (d,e)."""
    errors = []
    F, C, G = qa_triple(d, e)

    # P322_2: declared triple
    if decl_triple:
        for key, val in (("F", F), ("C", C), ("G", G)):
            declared = decl_triple.get(key)
            if declared is not None and declared != val:
                errors.append(f"P322_2: ({d},{e}) {key}={declared} ≠ computed {val}")

    # P322_3: Pythagorean
    if F*F + C*C != G*G:
        errors.append(f"P322_3: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    # P322_4: primitive direction
    if gcd(d, e) != 1:
        errors.append(f"P322_4: ({d},{e}) gcd={gcd(d,e)} ≠ 1")
    if d <= e:
        errors.append(f"P322_4: ({d},{e}) d must > e")
    if (d - e) % 2 == 0:
        errors.append(f"P322_4: ({d},{e}) d-e={d-e} must be odd")

    # P322_REG: both d and e regular
    if not is_regular(d):
        errors.append(f"P322_REG: ({d},{e}) d={d} is not 5-smooth (not regular in base-60)")
    if not is_regular(e):
        errors.append(f"P322_REG: ({d},{e}) e={e} is not 5-smooth (not regular in base-60)")

    # P322_BASE60: G/C terminates in base-60
    if not base60_terminates(G, C):
        g = gcd(G, C)
        denom = C // g
        errors.append(f"P322_BASE60: ({d},{e}) G/C={G}/{C} denom_reduced={denom} not 5-smooth → does not terminate in base-60")

    # P322_NOZERO: all positive
    for name, val in (("F", F), ("C", C), ("G", G)):
        if val <= 0:
            errors.append(f"P322_NOZERO: ({d},{e}) {name}={val} ≤ 0 (violates SPVN no-zero = QA A1)")

    return errors


def check_witness(w: dict) -> list[str]:
    d, e = w["d"], w["e"]
    decl_triple = w.get("triple", {})
    errors = check_row(d, e, decl_triple)

    # Check declared base60 list
    F, C, G = qa_triple(d, e)
    if w.get("base60") is not None:
        # Just verify first element (integer part) = G // C
        b60 = w["base60"]
        if b60 and b60[0] != G // C:
            errors.append(f"base60 ({d},{e}): integer part {b60[0]} ≠ G//C={G//C}")

    # Check declared G_over_C denominator
    decl_denom = w.get("denom_reduced")
    if decl_denom is not None:
        g = gcd(G, C)
        computed_denom = C // g
        if decl_denom != computed_denom:
            errors.append(f"P322_BASE60: ({d},{e}) declared denom_reduced={decl_denom} ≠ computed {computed_denom}")

    # Check declared d_regular, e_regular
    if w.get("d_regular") is not None and w["d_regular"] != is_regular(d):
        errors.append(f"P322_REG: ({d},{e}) declared d_regular={w['d_regular']} ≠ computed {is_regular(d)}")
    if w.get("e_regular") is not None and w["e_regular"] != is_regular(e):
        errors.append(f"P322_REG: ({d},{e}) declared e_regular={w['e_regular']} ≠ computed {is_regular(e)}")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # P322_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"P322_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        print("  SKIP detailed checks — cert declares FAIL")
        return errors, warnings

    # --- Single-row fixture (fundamental) ---
    direction = cert.get("direction")
    if direction:
        d, e = direction["d"], direction["e"]
        decl_triple = cert.get("triple", {})
        errs = check_row(d, e, decl_triple)
        errors.extend(errs)

        # Check base60 block
        b60_block = cert.get("base60", {})
        if b60_block:
            g = gcd(d*d+e*e, 2*d*e)
            computed_denom = (2*d*e) // g
            if b60_block.get("denominator_reduced") is not None and b60_block["denominator_reduced"] != computed_denom:
                errors.append(f"P322_BASE60: declared denom={b60_block['denominator_reduced']} ≠ {computed_denom}")
            if b60_block.get("denominator_regular") is not None:
                expected_reg = is_regular(computed_denom)
                if b60_block["denominator_regular"] != expected_reg:
                    errors.append(f"P322_BASE60: declared denom_regular={b60_block['denominator_regular']} ≠ {expected_reg}")

    # --- Witness-list fixture ---
    witnesses = cert.get("witnesses", [])
    if witnesses:
        if len(witnesses) < 3:
            errors.append(f"P322_W FAIL: need ≥3 witnesses, got {len(witnesses)}")
        has_fundamental = False
        for w in witnesses:
            werrs = check_witness(w)
            errors.extend([f"witness ({w['d']},{w['e']}): {e_}" for e_ in werrs])
            if w["d"] == 2 and w["e"] == 1:
                has_fundamental = True
        if not has_fundamental:
            warnings.append("P322_F: fundamental (d,e)=(2,1) not found in witnesses")

    # Validate non_row_counterexample block (should be non-regular)
    nrc = cert.get("non_row_counterexample")
    if nrc and nrc.get("d_regular") is not None:
        d_nr = nrc.get("d", 0)
        if nrc["d_regular"] != is_regular(d_nr):
            errors.append(f"non_row_counterexample: d_regular={nrc['d_regular']} ≠ {is_regular(d_nr)} for d={d_nr}")

    # Internal validation_checks
    vc = cert.get("validation_checks", [])
    failed_internal = [c for c in vc if not c.get("passed", True)]
    if failed_internal:
        errors.extend([f"internal check {c['check_id']} not passed" for c in failed_internal])

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected_pass = [
        "p322_pass_fundamental.json",
        "p322_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Plimpton 322 Cert [138] validator")
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
