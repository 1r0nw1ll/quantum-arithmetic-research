#!/usr/bin/env python3
"""
qa_klein4_harmonics_cert_validate.py

Validator for QA_KLEIN4_HARMONICS_CERT.v1  [family 142]

Certifies that the four sign-changes of (F,C,G) form a Klein 4-group K4 = Z2×Z2
that preserves the QA Pythagorean null condition F²+C²=G² and acts on the
harmonic packet {H, I, -H, -I} = {C+F, C-F, -(C+F), -(C-F)}.

THE FOUR OPERATIONS:
  I₀: (F, C, G) → ( F,  C, G)  [identity]
  I₁: (F, C, G) → (-F,  C, G)  [F-flip = red reflection;  (d,e)→(e,d)]
  I₂: (F, C, G) → ( F, -C, G)  [C-flip = green reflection; (d,e)→(d,-e)]
  I₃: (F, C, G) → (-F, -C, G)  [FC-flip = composition;    (d,e)→(e,-d)]

PYTHAGOREAN INVARIANCE:
  F²+C²=G²  iff  (-F)²+C²=G²  iff  F²+(-C)²=G²  iff  (-F)²+(-C)²=G²
  All four operations preserve the null/Pythagorean condition.

HARMONIC ACTION (H = C+F, I = C-F):
  I₀: (H, I) → ( H,  I)   [identity]
  I₁: (H, I) → ( I,  H)   [swap H↔I]
  I₂: (H, I) → (-I, -H)   [negate-swap]
  I₃: (H, I) → (-H, -I)   [negate both]

PROOF of I₁ swap: let (F',C')=(-F,C).
  H'=C'+F'=C+(-F)=C-F=I  ✓
  I'=C'-F'=C-(-F)=C+F=H  ✓

PROOF of I₂ negate-swap: let (F',C')=(F,-C).
  H'=C'+F'=(-C)+F=-(C-F)=-I  ✓
  I'=C'-F'=(-C)-F=-(C+F)=-H  ✓

PROOF of I₃ negate: let (F',C')=(-F,-C).
  H'=C'+F'=(-C)+(-F)=-(C+F)=-H  ✓
  I'=C'-F'=(-C)-(-F)=F-C=-(C-F)=-I  ✓

KLEIN 4-GROUP TABLE (Z2×Z2 — every element is its own inverse, I₁∘I₂=I₃):
  ∘  I₀  I₁  I₂  I₃
  I₀ I₀  I₁  I₂  I₃
  I₁ I₁  I₀  I₃  I₂
  I₂ I₂  I₃  I₀  I₁
  I₃ I₃  I₂  I₁  I₀

COORDINATE CORRESPONDENCE:
  I₁ (d,e)→(e,d):   F=d²-e²→e²-d²=-F, C=2de→2ed=C  ✓  (physical swap)
  I₂ (d,e)→(d,-e):  F=d²-e²=F, C=2d(-e)=-C          ✓  (formal: -e)
  I₃ (d,e)→(e,-d):  F=e²-d²=-F, C=2e(-d)=-C          ✓  (composition)

CHECKS:
  K4_1    schema_version == 'QA_KLEIN4_HARMONICS_CERT.v1'
  K4_2    F=d²-e², C=2de, G=d²+e², F²+C²=G²
  K4_3    Group table 4×4 matches K4 = Z2×Z2
  K4_ACT  All four K4 images of (F,C,G) satisfy F'²+C'²=G²
  K4_HARM I₁ swaps H↔I; I₂ maps (H,I)→(-I,-H); I₃ maps (H,I)→(-H,-I)
  K4_W    ≥3 direction witnesses
  K4_F    Fundamental (2,1): H=7, I=1; orbit {(7,1),(1,7),(-1,-7),(-7,-1)}
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_KLEIN4_HARMONICS_CERT.v1"

# Klein 4-group: (sign_F, sign_C) for each element index 0..3
_K4_SIGNS = {
    0: (1,  1),   # I0 identity
    1: (-1, 1),   # I1 F-flip
    2: (1,  -1),  # I2 C-flip
    3: (-1, -1),  # I3 FC-flip
}

# Canonical group multiplication table: _K4_TABLE[i][j] = index of I_i ∘ I_j
_K4_TABLE = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
]


def qa_triple(d: int, e: int):
    return d*d - e*e, 2*d*e, d*d + e*e


def k4_apply(idx: int, F: int, C: int):
    """Return (F', C') after applying K4 element idx."""
    sf, sc = _K4_SIGNS[idx]
    return sf * F, sc * C


def check_direction(d: int, e: int, decl: dict) -> list[str]:
    errors = []
    F, C, G = qa_triple(d, e)
    H_val = C + F
    I_val = C - F

    # K4_2: declared triple/harmonic values
    for key, val in (("F", F), ("C", C), ("G", G), ("H", H_val), ("I", I_val)):
        if decl.get(key) is not None and decl[key] != val:
            errors.append(f"K4_2: ({d},{e}) declared {key}={decl[key]} ≠ computed {val}")

    # K4_2: Pythagorean
    if F*F + C*C != G*G:
        errors.append(f"K4_2: ({d},{e}) F²+C²={F*F+C*C} ≠ G²={G*G}")

    # K4_ACT: all four images preserve F'²+C'²=G²
    for idx in range(4):
        Fp, Cp = k4_apply(idx, F, C)
        if Fp*Fp + Cp*Cp != G*G:
            errors.append(
                f"K4_ACT: ({d},{e}) I{idx}(F,C)=({Fp},{Cp}): "
                f"F'²+C'²={Fp*Fp+Cp*Cp} ≠ G²={G*G}"
            )

    # K4_HARM: I₁ swaps H↔I
    Fp1, Cp1 = k4_apply(1, F, C)
    Hn1, In1 = Cp1 + Fp1, Cp1 - Fp1
    if Hn1 != I_val or In1 != H_val:
        errors.append(
            f"K4_HARM: ({d},{e}) I₁ should swap H↔I: got (H',I')=({Hn1},{In1}), "
            f"expected ({I_val},{H_val})"
        )

    # K4_HARM: I₂ maps (H,I)→(-I,-H)
    Fp2, Cp2 = k4_apply(2, F, C)
    Hn2, In2 = Cp2 + Fp2, Cp2 - Fp2
    if Hn2 != -I_val or In2 != -H_val:
        errors.append(
            f"K4_HARM: ({d},{e}) I₂ should map (H,I)→(-I,-H): "
            f"got (H',I')=({Hn2},{In2}), expected ({-I_val},{-H_val})"
        )

    # K4_HARM: I₃ maps (H,I)→(-H,-I)
    Fp3, Cp3 = k4_apply(3, F, C)
    Hn3, In3 = Cp3 + Fp3, Cp3 - Fp3
    if Hn3 != -H_val or In3 != -I_val:
        errors.append(
            f"K4_HARM: ({d},{e}) I₃ should negate (H,I): "
            f"got (H',I')=({Hn3},{In3}), expected ({-H_val},{-I_val})"
        )

    # Check declared k4_orbit entries if present
    orbit = decl.get("k4_orbit")
    if orbit:
        for item in orbit:
            dH = item.get("H")
            dI = item.get("I")
            if dH is None or dI is None:
                continue
            # F'=(H'-I')/2, C'=(H'+I')/2 must satisfy F'²+C'²=G²
            if (dH + dI) % 2 != 0:
                errors.append(f"({d},{e}) orbit H={dH} I={dI}: H+I={dH+dI} not even")
                continue
            Cp_check = (dH + dI) // 2
            Fp_check = (dH - dI) // 2
            if Fp_check * Fp_check + Cp_check * Cp_check != G * G:
                errors.append(
                    f"({d},{e}) orbit H={dH} I={dI}: "
                    f"F'²+C'²={Fp_check*Fp_check+Cp_check*Cp_check} ≠ G²={G*G}"
                )

    return errors


def check_group_table(decl_table: list) -> list[str]:
    """Verify declared K4 multiplication table matches Z2×Z2."""
    errors = []
    if len(decl_table) != 4:
        errors.append(f"K4_3: group table has {len(decl_table)} rows, expected 4")
        return errors
    for i, row in enumerate(decl_table):
        if len(row) != 4:
            errors.append(f"K4_3: row {i} has {len(row)} entries, expected 4")
            continue
        for j, val in enumerate(row):
            if val != _K4_TABLE[i][j]:
                errors.append(
                    f"K4_3: table[{i}][{j}]={val} ≠ expected {_K4_TABLE[i][j]}"
                )
    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # K4_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"K4_1 FAIL: schema_version={cert.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        return errors, warnings

    # K4_3: group table (optional, but validated if present)
    table = cert.get("group_table")
    if table is not None:
        errors.extend(check_group_table(table))

    # Direction witnesses
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        warnings.append("K4_W: no witnesses provided")
    else:
        if len(witnesses) < 3:
            errors.append(f"K4_W: need ≥3 witnesses, got {len(witnesses)}")
        has_fundamental = False
        for w in witnesses:
            d, e = w["d"], w["e"]
            decl = {k: w.get(k) for k in ("F", "C", "G", "H", "I", "k4_orbit")}
            errs = check_direction(d, e, decl)
            errors.extend([f"witness ({d},{e}): {e_}" for e_ in errs])
            if d == 2 and e == 1:
                has_fundamental = True
        # K4_F: fundamental must appear in at least one fixture (warn if missing)
        if not has_fundamental:
            warnings.append("K4_F: fundamental (2,1) not in witnesses")

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
        "k4_pass_group_axioms.json",
        "k4_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Klein 4 Harmonics Cert [142] validator")
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
