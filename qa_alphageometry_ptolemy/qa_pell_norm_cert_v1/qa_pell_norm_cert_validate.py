#!/usr/bin/env python3
"""
qa_pell_norm_cert_validate.py

Validator for QA_PELL_NORM_CERT.v1  [family 141]

Certifies I = C - F = -(x² - 2y²)  where  x = d-e,  y = e

for any QA direction (d,e) with triple (F,C,G)=(d²-e², 2de, d²+e²).

PROOF:
  I = 2de - (d²-e²)
  Let x=d-e, y=e  →  d=x+y
  I = 2(x+y)y - ((x+y)²-y²)
    = 2xy+2y² - (x²+2xy+y²-y²)
    = 2y² - x²
    = -(x²-2y²)  ✓

The Pell norm P(x,y) = x²-2y² satisfies: I = -P(d-e, e).

PELL BOUNDARY:
  P = -1  →  I = +1  (hyperbola boundary)
  P =  0  →  I =  0  (parabola; impossible for primitive integer d,e)
  P = +1  →  I = -1  (ellipse boundary)

M_B CHAIN:
  The Pythagorean tree move M_B(d,e)=(2d+e, d) corresponds to
  (x,y) → (x+2y, x+y) in Pell variables, which maps P → -P.
  Starting from (2,1) with P=-1, the M_B chain generates:
  (2,1)→(5,2)→(12,5)→(29,12)→(70,29)→...
  with alternating P=-1,+1,-1,+1,...  (|I|=1 at each step)

CHECKS:
  PN_1     schema_version == 'QA_PELL_NORM_CERT.v1'
  PN_2     F=d²-e², C=2de, G=d²+e²
  PN_3     F²+C²=G²
  PN_IDEN  I = -(x²-2y²) where x=d-e, y=e
  PN_MB    M_B step flips Pell sign: consecutive chain entries have P₁=-P₀
  PN_W     ≥3 general witnesses
  PN_F     fundamental (2,1): P=-1, I=1
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_PELL_NORM_CERT.v1"


def qa_triple(d: int, e: int):
    return d*d - e*e, 2*d*e, d*d + e*e


def pell_norm(x: int, y: int) -> int:
    return x*x - 2*y*y


def check_direction(d: int, e: int, decl: dict) -> list[str]:
    errors = []
    F, C, G = qa_triple(d, e)
    I = C - F
    x, y = d - e, e
    P = pell_norm(x, y)

    # PN_2
    for key, val in (("F", F), ("C", C), ("G", G)):
        if decl.get(key) is not None and decl[key] != val:
            errors.append(f"PN_2: ({d},{e}) {key}={decl[key]} ≠ computed {val}")

    # PN_3
    if F*F + C*C != G*G:
        errors.append(f"PN_3: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    # PN_IDEN: I = -P
    if I != -P:
        errors.append(f"PN_IDEN: ({d},{e}) I={I} ≠ -P(d-e,e)={-P}")

    # Check declared pell_norm and I
    if decl.get("pell_norm") is not None and decl["pell_norm"] != P:
        errors.append(f"PN_IDEN: ({d},{e}) declared pell_norm={decl['pell_norm']} ≠ {P}")
    if decl.get("I") is not None and decl["I"] != I:
        errors.append(f"PN_IDEN: ({d},{e}) declared I={decl['I']} ≠ {I}")

    # Check declared x, y
    if decl.get("x") is not None and decl["x"] != x:
        errors.append(f"({d},{e}) declared x={decl['x']} ≠ d-e={x}")
    if decl.get("y") is not None and decl["y"] != y:
        errors.append(f"({d},{e}) declared y={decl['y']} ≠ e={y}")

    # Check declared type
    if decl.get("type") is not None:
        expected = "H" if I > 0 else ("E" if I < 0 else "P")
        if decl["type"] != expected:
            errors.append(f"({d},{e}) declared type={decl['type']} ≠ {expected}")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # PN_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"PN_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        return errors, warnings

    # --- Pell chain fixture ---
    chain = cert.get("pell_chain", {}).get("chain", [])
    if chain:
        has_fundamental = False
        for entry in chain:
            d, e = entry["d"], entry["e"]
            errors.extend(check_direction(d, e, entry))
            if d == 2 and e == 1:
                has_fundamental = True

        if not has_fundamental:
            errors.append("PN_F: fundamental (2,1) not found in pell_chain")

        # PN_MB: consecutive entries must flip Pell sign
        for i in range(len(chain) - 1):
            cur = chain[i]
            nxt = chain[i+1]
            # Verify M_B relationship
            d0, e0 = cur["d"], cur["e"]
            d1, e1 = nxt["d"], nxt["e"]
            if (2*d0 + e0, d0) != (d1, e1):
                errors.append(f"PN_MB: chain[{i}]→chain[{i+1}]: ({d1},{e1}) ≠ M_B({d0},{e0})=({2*d0+e0},{d0})")
            # Pell sign must flip
            P0 = pell_norm(d0-e0, e0)
            P1 = pell_norm(d1-e1, e1)
            if P0 != -P1:
                errors.append(f"PN_MB: P({d0},{e0})={P0} and P({d1},{e1})={P1}: should be negatives")

    # --- General witnesses ---
    witnesses = cert.get("witnesses", [])
    if witnesses:
        if len(witnesses) < 3:
            errors.append(f"PN_W: need ≥3 witnesses, got {len(witnesses)}")
        has_fundamental = False
        for w in witnesses:
            d, e = w["d"], w["e"]
            decl = {k: w.get(k) for k in ("F","C","G","I","x","y","pell_norm","type")}
            if w.get("triple"):
                decl.update(w["triple"])
            errors.extend([f"witness ({d},{e}): {e_}" for e_ in check_direction(d, e, decl)])
            if d == 2 and e == 1:
                has_fundamental = True
        if not has_fundamental:
            warnings.append("PN_F: (2,1) not in witnesses")

    # Internal validation_checks
    vc = cert.get("validation_checks", [])
    failed_internal = [c for c in vc if not c.get("passed", True)]
    if failed_internal:
        errors.extend([f"internal check {c['check_id']} not passed" for c in failed_internal])

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected_pass = [
        "pn_pass_fundamental.json",
        "pn_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Pell Norm Cert [141] validator")
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
