#!/usr/bin/env python3
"""
qa_eisenstein_crystal_cert_validate.py

Validator for QA_EISENSTEIN_CRYSTAL_CERT.v1  [family 183]

Certifies:
  (1) Z - Y = J = bd  (new universal identity)
  (2) Z² - Y² = J·a·(a+e)  (factorization)
  (3) F² - FW + W² = Z²  (Eisenstein norm encodes crystal constants)
  (4) Unity Block {F,G,Z,W} = {3,5,7,8} = Ben's four Forces

Checks:
  EC_1      — schema_version matches
  EC_ZYJ    — Z - Y = J = bd for each witness
  EC_FACTOR — Z² - Y² = J·a·(a+e) for each witness
  EC_EISEN  — F² - FW + W² = Z² for each witness
  EC_TUPLE  — (b,e,d,a) A2-compliant: d=b+e, a=b+2e
  EC_UNITY  — Unity Block special properties (if present)
  EC_W      — at least one witness
  EC_F      — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates algebraic identity claims in submitted JSON; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_EISENSTEIN_CRYSTAL_CERT.v1"


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # EC_1
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"EC_1: schema_version mismatch: got {sv!r}")

    # EC_F
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("EC_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("EC_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # EC_W
    witnesses = cert.get("witnesses")
    if not witnesses or not isinstance(witnesses, list):
        errors.append("EC_W: no witnesses array")
        return errors, warnings

    for idx, w in enumerate(witnesses):
        b = w.get("b")
        e = w.get("e")
        if b is None or e is None:
            errors.append(f"EC_W: witness[{idx}] missing b or e")
            continue

        # EC_TUPLE: derive d, a
        d_exp = b + e
        a_exp = b + 2 * e
        d = w.get("d", d_exp)
        a = w.get("a", a_exp)
        if d != d_exp:
            errors.append(f"EC_TUPLE: witness[{idx}] d={d}, expected {d_exp}")
        if a != a_exp:
            errors.append(f"EC_TUPLE: witness[{idx}] a={a}, expected {a_exp}")

        # Compute expected values
        J_exp = b * d_exp
        Z_exp = e * e + a_exp * d_exp
        Y_exp = a_exp * a_exp - d_exp * d_exp
        F_exp = a_exp * b
        W_exp = d_exp * (e + a_exp)
        G_exp = d_exp * d_exp + e * e

        # EC_ZYJ: Z - Y = J = bd
        J_decl = w.get("J")
        Z_decl = w.get("Z")
        Y_decl = w.get("Y")
        zmy_decl = w.get("Z_minus_Y")

        if J_decl is not None and J_decl != J_exp:
            errors.append(f"EC_ZYJ: witness[{idx}] J={J_decl}, expected bd={J_exp}")
        if Z_decl is not None and Z_decl != Z_exp:
            errors.append(f"EC_ZYJ: witness[{idx}] Z={Z_decl}, expected {Z_exp}")
        if Y_decl is not None and Y_decl != Y_exp:
            errors.append(f"EC_ZYJ: witness[{idx}] Y={Y_decl}, expected {Y_exp}")
        if zmy_decl is not None and zmy_decl != J_exp:
            errors.append(f"EC_ZYJ: witness[{idx}] Z_minus_Y={zmy_decl}, expected J={J_exp}")

        zyj_flag = w.get("Z_minus_Y_eq_J")
        if zyj_flag is not None and zyj_flag is not True:
            errors.append(f"EC_ZYJ: witness[{idx}] Z_minus_Y_eq_J is not True")

        # EC_FACTOR: Z² - Y² = J·a·(a+e)
        z2y2_exp = Z_exp * Z_exp - Y_exp * Y_exp
        jae_exp = J_exp * a_exp * (a_exp + e)
        if z2y2_exp != jae_exp:
            errors.append(f"EC_FACTOR: witness[{idx}] Z²-Y²={z2y2_exp} != J·a·(a+e)={jae_exp} (internal error)")

        z2y2_decl = w.get("Z2_minus_Y2")
        jae_decl = w.get("J_a_ape")
        if z2y2_decl is not None and z2y2_decl != z2y2_exp:
            errors.append(f"EC_FACTOR: witness[{idx}] Z2_minus_Y2={z2y2_decl}, expected {z2y2_exp}")
        if jae_decl is not None and jae_decl != jae_exp:
            errors.append(f"EC_FACTOR: witness[{idx}] J_a_ape={jae_decl}, expected {jae_exp}")

        # EC_EISEN: F² - FW + W² = Z²
        eisen_exp = F_exp * F_exp - F_exp * W_exp + W_exp * W_exp
        z2_exp = Z_exp * Z_exp
        if eisen_exp != z2_exp:
            errors.append(f"EC_EISEN: witness[{idx}] F²-FW+W²={eisen_exp} != Z²={z2_exp} (internal error)")

        eisen_decl = w.get("F2_FW_W2")
        z2_decl = w.get("Z2")
        if eisen_decl is not None and eisen_decl != eisen_exp:
            errors.append(f"EC_EISEN: witness[{idx}] F2_FW_W2={eisen_decl}, expected {eisen_exp}")
        if z2_decl is not None and z2_decl != z2_exp:
            errors.append(f"EC_EISEN: witness[{idx}] Z2={z2_decl}, expected {z2_exp}")

        eisen_flag = w.get("eisenstein_holds")
        if eisen_flag is not None and eisen_flag is not True:
            errors.append(f"EC_EISEN: witness[{idx}] eisenstein_holds is not True")

    # EC_UNITY: check Unity Block section if present
    ec = cert.get("eisenstein_crystal")
    if ec:
        ub = ec.get("unity_block")
        if ub and ub.get("b") == 1 and ub.get("e") == 1:
            ff = ec.get("four_forces", {})
            if ff.get("F") != 3:
                errors.append("EC_UNITY: Unity F should be 3")
            if ff.get("G") != 5:
                errors.append("EC_UNITY: Unity G should be 5")
            if ff.get("Z") != 7:
                errors.append("EC_UNITY: Unity Z should be 7")
            if ff.get("W") != 8:
                errors.append("EC_UNITY: Unity W should be 8")
            cp = ec.get("cosmos_period_from_unity", {})
            if cp.get("equals_cosmos_period") is not True:
                warnings.append("EC_UNITY: cosmos_period_from_unity.equals_cosmos_period not True")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("ec_pass_identity_chain.json", True),
        ("ec_fail_wrong_zyj.json", True),  # declares FAIL, validator skips
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Eisenstein Crystal Cert [183] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))
    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)
    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
