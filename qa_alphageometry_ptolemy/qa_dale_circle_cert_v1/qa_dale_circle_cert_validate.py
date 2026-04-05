#!/usr/bin/env python3
"""
qa_dale_circle_cert_validate.py

Validator for QA_DALE_CIRCLE_CERT.v1  [family 189]

Certifies: Dale Pond's integer circle construction extending Ben Iverson's
Quantum Arithmetic. Three new elements P, Q, R derived from the equilateral
side W.

DALE'S CIRCLE (svpwiki.com, "Quantum Arithmetic Elements", 1998):
  P = 2W          — circle diameter (in QA integer units)
  Q = P = 2W      — circle circumference (in QA integer units)
  R = W*W         — circle area (in QA integer units)

Dale defines QA circular units where pi=1: P=2W (diameter), Q=P
(circumference), R=W² (area). This is a unit convention (analogous to
natural units in physics), not a structural theorem. The algebraic
identities are exact by definition. Dale's contribution is identifying
W as the natural radius-like quantity for QA circles.
"When Ben said there was no way to define a circle with Quantum
Arithmetic, I developed three different ways." — Dale Pond, 1998.

W = d*(e+a) = d*(b+3*e) = X + K  (certified in [152])

Checks:
  DC_1       — schema_version matches
  DC_P       — P = 2*W for all witnesses
  DC_Q       — Q = P for all witnesses (diameter = circumference)
  DC_R       — R = W*W for all witnesses
  DC_W       — W correctly derived from (b,e) tuple
  DC_SRC     — source attribution to Dale Pond present
  DC_WITNESS — at least 3 witnesses with verified elements
  DC_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Dale circle elements P,Q,R; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_DALE_CIRCLE_CERT.v1"


def qa_mod(x, m):
    """A1-compliant: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # DC_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"DC_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # DC_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("DC_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("DC_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # DC_SRC: source attribution
    src = cert.get("source_attribution")
    if not src or "Dale Pond" not in str(src):
        warnings.append("DC_SRC: source_attribution should credit Dale Pond")

    # DC_WITNESS: at least 3 witnesses
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        errors.append(f"DC_WITNESS: need >= 3 witnesses, got {len(witnesses)}")

    for idx, w in enumerate(witnesses):
        b = w.get("b")
        e = w.get("e")
        if b is None or e is None:
            continue

        # Compute derived values
        d_val = b + e  # A2: derived
        a_val = e + d_val  # A2: derived

        # DC_W: W correctly derived
        W_expected = d_val * (e + a_val)
        W_decl = w.get("W")
        if W_decl is not None and W_decl != W_expected:
            errors.append(f"DC_W: witness[{idx}] W={W_decl}, expected d*(e+a)={W_expected}")

        W_val = W_decl if W_decl is not None else W_expected

        # DC_P: P = 2W
        P_decl = w.get("P")
        if P_decl is not None and P_decl != 2 * W_val:
            errors.append(f"DC_P: witness[{idx}] P={P_decl}, expected 2*W={2*W_val}")

        # DC_Q: Q = P
        Q_decl = w.get("Q")
        P_val = P_decl if P_decl is not None else 2 * W_val
        if Q_decl is not None and Q_decl != P_val:
            errors.append(f"DC_Q: witness[{idx}] Q={Q_decl}, expected P={P_val}")

        # DC_R: R = W*W
        R_decl = w.get("R")
        if R_decl is not None and R_decl != W_val * W_val:
            errors.append(f"DC_R: witness[{idx}] R={R_decl}, expected W*W={W_val*W_val}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("dc_pass_circle.json", True),
        ("dc_fail_bad_p.json", True),
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
        description="QA Dale Circle Cert [189] validator")
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
