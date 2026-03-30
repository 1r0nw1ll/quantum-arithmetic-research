#!/usr/bin/env python3
"""
qa_cyclic_quad_cert_validate.py

Validator for QA_CYCLIC_QUAD_CERT.v1  [family 136]

Certifies Ptolemy's theorem via three integer identities for QA triple pairs:

  BF (Brahmagupta-Fibonacci):
      G₁·G₂ = D² + E²   where D=d₁d₂−e₁e₂, E=d₁e₂+d₂e₁

  PP (Ptolemy Product):
      F₃ = F₁F₂ − C₁C₂   (Ptolemy cos-subtraction)
      C₃ = F₁C₂ + F₂C₁   (Ptolemy sin-addition)
      F₃² + C₃² = (G₁G₂)²

  PC (Ptolemy Conjugate):
      F₄ = F₁F₂ + C₁C₂   (Ptolemy cos-addition)
      C₄ = |F₁C₂ − F₂C₁|  (Ptolemy sin-subtraction)
      F₄² + C₄² = (G₁G₂)²

Both product and conjugate triples lie on the circle G₁G₂ — the two diagonals
of the Ptolemy cyclic quadrilateral. Algebraic proof:
    F₃²+C₃² = (F₁F₂−C₁C₂)²+(F₁C₂+F₂C₁)² = (F₁²+C₁²)(F₂²+C₂²) = G₁²G₂²

Historical chain: Ptolemy ~150 CE (chord-table sin/cos addition) →
Brahmagupta 628 CE (product of sums-of-squares) → QA Gaussian direction
multiplication. Connects to cert [127] UHG null (same null points, now as
cyclic quadrilateral vertices).

Checks
------
CQ_1   schema_version == 'QA_CYCLIC_QUAD_CERT.v1'
CQ_2   F=d²-e², C=2de, G=d²+e² for each declared triple
CQ_3   F²+C²=G² for each triple
CQ_BF  G1*G2 = D²+E² where D=d1*d2-e1*e2, E=d1*e2+d2*e1
CQ_PP  F3=|F1*F2-C1*C2|, C3=F1*C2+F2*C1, F3²+C3²=(G1*G2)²
CQ_PC  F4=F1*F2+C1*C2, C4=|F1*C2-F2*C1|, F4²+C4²=(G1*G2)²
CQ_G3  Both product and conjugate G equal G1*G2
CQ_W   ≥3 witness pairs (witness fixture)
CQ_F   fundamental pair (d1,e1)=(2,1),(d2,e2)=(3,2) present
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_CYCLIC_QUAD_CERT.v1"


def qa_triple(d, e):
    return d*d - e*e, 2*d*e, d*d + e*e


def check_pair(d1, e1, d2, e2, cert_block):
    """Verify all Ptolemy identities for a (d1,e1)×(d2,e2) pair."""
    errors = []

    F1, C1, G1 = qa_triple(d1, e1)
    F2, C2, G2 = qa_triple(d2, e2)

    # CQ_2: declared triples match
    for i, (d, e, F, C, G, key) in enumerate([
        (d1,e1,F1,C1,G1,"triple_1"), (d2,e2,F2,C2,G2,"triple_2")
    ]):
        t = cert_block.get(key, {})
        if t:
            if t.get("F") != F:
                errors.append(f"CQ_2: {key} F={t.get('F')} ≠ computed {F} for ({d},{e})")
            if t.get("C") != C:
                errors.append(f"CQ_2: {key} C={t.get('C')} ≠ computed {C} for ({d},{e})")
            if t.get("G") != G:
                errors.append(f"CQ_2: {key} G={t.get('G')} ≠ computed {G} for ({d},{e})")
        # CQ_3: Pythagorean identity
        if F*F + C*C != G*G:
            errors.append(f"CQ_3: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    # Compute expected values
    D  = d1*d2 - e1*e2
    E  = d1*e2 + d2*e1
    Dp = d1*d2 + e1*e2
    Ep = abs(d1*e2 - d2*e1)
    G3 = G1 * G2

    # CQ_BF
    if D*D + E*E != G3:
        errors.append(f"CQ_BF: D²+E²={D*D+E*E} ≠ G1*G2={G3}")

    # CQ_PP
    F3_raw = F1*F2 - C1*C2
    F3 = abs(F3_raw)
    C3 = F1*C2 + F2*C1
    if F3*F3 + C3*C3 != G3*G3:
        errors.append(f"CQ_PP: F3²+C3²={F3*F3+C3*C3} ≠ (G1G2)²={G3*G3}")

    # Check declared T3 values if present
    t3 = cert_block.get("T3") or cert_block.get("ptolemy_product", {})
    if isinstance(t3, dict):
        for key, expected, actual in [("F", F3, t3.get("F")), ("C", C3, t3.get("C")), ("G", G3, t3.get("G"))]:
            if actual is not None and actual != expected:
                errors.append(f"CQ_PP: T3.{key}={actual} ≠ computed {expected}")

    # CQ_PC
    F4 = F1*F2 + C1*C2
    C4 = abs(F1*C2 - F2*C1)
    if F4*F4 + C4*C4 != G3*G3:
        errors.append(f"CQ_PC: F4²+C4²={F4*F4+C4*C4} ≠ (G1G2)²={G3*G3}")

    # CQ_G3
    if Dp*Dp + Ep*Ep != G3:
        errors.append(f"CQ_G3 conj: D'²+E'²={Dp*Dp+Ep*Ep} ≠ G1*G2={G3}")

    # Check declared BF values
    bf = cert_block.get("brahmagupta_fibonacci", {})
    if bf:
        if bf.get("D") is not None and bf["D"] != D:
            errors.append(f"CQ_BF: declared D={bf['D']} ≠ computed D={D}")
        if bf.get("E") is not None and bf["E"] != E:
            errors.append(f"CQ_BF: declared E={bf['E']} ≠ computed E={E}")
        if bf.get("G3") is not None and bf["G3"] != G3:
            errors.append(f"CQ_BF: declared G3={bf['G3']} ≠ G1*G2={G3}")

    return errors


def check_witness(w):
    """Validate a witness entry with d1,e1,d2,e2 fields."""
    d1, e1, d2, e2 = w["d1"], w["e1"], w["d2"], w["e2"]
    errors = check_pair(d1, e1, d2, e2, w)

    # Also verify declared D,E and T3,T4 values
    D  = d1*d2 - e1*e2
    E  = d1*e2 + d2*e1
    Dp = d1*d2 + e1*e2
    Ep = abs(d1*e2 - d2*e1)
    G3 = (d1*d2 - e1*e2)**2 + (d1*e2 + d2*e1)**2  # = G1*G2

    for key, val in [("D", D), ("E", E), ("G3", G3), ("Dp", Dp), ("Ep", Ep)]:
        declared = w.get(key)
        if declared is not None and declared != val:
            errors.append(f"witness ({d1},{e1})×({d2},{e2}): {key}={declared} ≠ computed {val}")

    F1,C1,G1 = qa_triple(d1,e1)
    F2,C2,G2 = qa_triple(d2,e2)
    F3 = abs(F1*F2 - C1*C2)
    C3 = F1*C2 + F2*C1
    F4 = F1*F2 + C1*C2
    C4 = abs(F1*C2 - F2*C1)

    for key, val in [("T3", (F3,C3,G1*G2)), ("T4", (F4,C4,G1*G2))]:
        t = w.get(key, {})
        if t and isinstance(t, dict):
            expected = {"F": val[0], "C": val[1], "G": val[2]}
            for k,v in expected.items():
                if t.get(k) is not None and t[k] != v:
                    errors.append(f"witness ({d1},{e1})×({d2},{e2}): {key}.{k}={t[k]} ≠ {v}")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # CQ_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"CQ_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        print("  SKIP detailed checks — cert declares FAIL")
        return errors, warnings

    # --- Single-pair fixture (fundamental) ---
    if "triple_1" in cert and "triple_2" in cert:
        t1 = cert["triple_1"]
        t2 = cert["triple_2"]
        errs = check_pair(t1["d"], t1["e"], t2["d"], t2["e"], cert)
        errors.extend(errs)

        # CQ_F: fundamental pair check
        if t1.get("d") == 2 and t1.get("e") == 1 and t2.get("d") == 3 and t2.get("e") == 2:
            bf = cert.get("brahmagupta_fibonacci", {})
            if bf.get("G3") != 65:
                errors.append("CQ_F FAIL: fundamental pair G3 should be 65")

    # --- Witness-list fixture ---
    witnesses = cert.get("witnesses", [])
    if witnesses:
        if len(witnesses) < 3:
            errors.append(f"CQ_W FAIL: need ≥3 witnesses, got {len(witnesses)}")
        for w in witnesses:
            werrs = check_witness(w)
            errors.extend([f"witness ({w['d1']},{w['e1']})×({w['d2']},{w['e2']}): {e}" for e in werrs])

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
        "cq_pass_fundamental.json",
        "cq_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Cyclic Quad Cert [136] validator")
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
