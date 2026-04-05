#!/usr/bin/env python3
"""
qa_equilateral_height_cert_validate.py

Validator for QA_EQUILATERAL_HEIGHT_CERT.v1  [family 190]

Certifies: Element S = d*X = d²*e, Dale Pond's 25th QA element.
Dale Pond extension (#25 in svpwiki.com "Quantum Arithmetic Elements").
Dale labeled this "Height of equilateral triangle" but geometrically
S = d * (C/2) is a RECTANGLE AREA (semi-major × half-base), not a
height. The algebraic identity S = d²*e = d*X = D*e is exact.

  S = d * X     where X = d*e = C/2
  S = d² * e    = D * e
  S = d * C/2

Checks:
  EH_1       — schema_version matches
  EH_S       — S = d*d*e for all witnesses
  EH_DX      — S = d*X verified
  EH_DE      — S = D*e verified (D = d²)
  EH_W       — at least 3 witnesses
  EH_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates equilateral height element S; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_EQUILATERAL_HEIGHT_CERT.v1"


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # EH_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"EH_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # EH_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("EH_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("EH_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # EH_W: witnesses
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        errors.append(f"EH_W: need >= 3 witnesses, got {len(witnesses)}")

    for idx, w in enumerate(witnesses):
        b = w.get("b")
        e = w.get("e")
        if b is None or e is None:
            continue

        d_val = b + e  # A2: derived
        D_val = d_val * d_val
        X_val = d_val * e

        S_decl = w.get("S")
        S_expected = d_val * d_val * e

        # EH_S: S = d²*e
        if S_decl is not None and S_decl != S_expected:
            errors.append(f"EH_S: witness[{idx}] S={S_decl}, expected d*d*e={S_expected}")

        # EH_DX: S = d*X
        if S_decl is not None and S_decl != d_val * X_val:
            errors.append(f"EH_DX: witness[{idx}] S={S_decl}, expected d*X={d_val*X_val}")

        # EH_DE: S = D*e
        if S_decl is not None and S_decl != D_val * e:
            errors.append(f"EH_DE: witness[{idx}] S={S_decl}, expected D*e={D_val*e}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("eh_pass_height.json", True),
        ("eh_fail_wrong_s.json", True),
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
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Equilateral Height Cert [190] validator")
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
