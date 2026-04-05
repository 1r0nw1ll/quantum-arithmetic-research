#!/usr/bin/env python3
"""
qa_miller_orbit_cert_validate.py

Validator for QA_MILLER_ORBIT_CERT.v1  [family 182]

Certifies structural properties of QA mod-9 orbit classification
applied to crystallographic Miller indices (h,k,l):
  (1) cosmos mean d > satellite mean d (ordering)
  (2) satellite Q_M mod 9 restricted to QR(9) = {0,1,4,7}
  (3) singularity Q_M = perfect squares
  (4) chromogeometric channel shift (satellite greener)

Checks:
  MO_1      — schema_version matches
  MO_ORDER  — ordering claim: fraction_confirmed = 1.0
  MO_QR     — quadratic residue restriction exact
  MO_SQUARE — singularity Q_M all perfect squares
  MO_CHROMO — satellite green > cosmos green
  MO_W      — at least one witness
  MO_F      — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates algebraic crystallography claims in submitted JSON; no float state"

import json
import sys
import math
from pathlib import Path

SCHEMA_VERSION = "QA_MILLER_ORBIT_CERT.v1"
QR_9 = {0, 1, 4, 7}  # quadratic residues mod 9


def validate(path):
    """Validate a Miller orbit certificate.
    Returns (errors, warnings).
    """
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # MO_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"MO_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # MO_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("MO_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("MO_F: fail_ledger must be a list")

    # Handle FAIL result early
    if cert.get("result") == "FAIL":
        return errors, warnings

    # MO_ORDER: d-spacing ordering
    ordering = cert.get("ordering")
    if ordering:
        fc = ordering.get("fraction_confirmed")
        if fc is not None and fc < 1.0:
            errors.append(f"MO_ORDER: fraction_confirmed={fc}, expected 1.0 (all minerals)")
        n_tested = ordering.get("n_minerals_tested", 0)
        n_conf = ordering.get("n_minerals_confirmed", 0)
        if n_tested > 0 and n_conf < n_tested:
            errors.append(f"MO_ORDER: {n_conf}/{n_tested} minerals confirmed, expected all")
        # Check means if provided
        cos_d = ordering.get("cosmos_mean_d")
        sat_d = ordering.get("satellite_mean_d")
        if cos_d is not None and sat_d is not None:
            if cos_d <= sat_d:
                errors.append(f"MO_ORDER: cosmos_mean_d={cos_d} <= satellite_mean_d={sat_d}")
    else:
        warnings.append("MO_ORDER: ordering section missing")

    # MO_QR: quadratic residue restriction
    qr = cert.get("quadratic_residues")
    if qr:
        observed = set(qr.get("satellite_residues_observed", []))
        forbidden = observed - QR_9
        if forbidden:
            errors.append(f"MO_QR: satellite residues {sorted(forbidden)} are not in QR(9)={sorted(QR_9)}")
        if qr.get("restriction_exact") is False:
            errors.append("MO_QR: restriction_exact is False — satellite has forbidden residues")
        fc = qr.get("forbidden_count", 0)
        if fc > 0:
            errors.append(f"MO_QR: forbidden_count={fc}, expected 0")
    else:
        warnings.append("MO_QR: quadratic_residues section missing")

    # MO_SQUARE: singularity perfect squares
    ps = cert.get("perfect_squares")
    if ps:
        vals = ps.get("observed_values", [])
        for v in vals:
            root = int(math.isqrt(v))
            if root * root != v:
                errors.append(f"MO_SQUARE: singularity Q_M={v} is not a perfect square")
        if ps.get("all_perfect_squares") is False:
            errors.append("MO_SQUARE: all_perfect_squares is False")
    else:
        warnings.append("MO_SQUARE: perfect_squares section missing")

    # MO_CHROMO: chromogeometric shift
    ch = cert.get("chromogeometric_shift")
    if ch:
        cos_g = ch.get("cosmos_green_fraction", 0)
        sat_g = ch.get("satellite_green_fraction", 0)
        if sat_g <= cos_g:
            errors.append(f"MO_CHROMO: satellite_green={sat_g} <= cosmos_green={cos_g}")
    else:
        warnings.append("MO_CHROMO: chromogeometric_shift section missing")

    # MO_W: witnesses
    w = cert.get("witnesses")
    if not w:
        warnings.append("MO_W: no witnesses listed")

    return errors, warnings


def _self_test():
    """Test bundled fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("mo_pass_batch21.json", True),
        ("mo_fail_wrong_residues.json", True),  # declares result=FAIL, validator skips
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
        description="QA Miller Orbit Cert [182] validator")
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
