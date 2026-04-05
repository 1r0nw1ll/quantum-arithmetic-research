#!/usr/bin/env python3
"""
qa_keely_dominant_control_cert_validate.py

Validator for QA_KEELY_DOMINANT_CONTROL_CERT.v1  [family 186]

Certifies: Keely's 3 Dominant/Control Laws (Category 3 of Vibes
5-category framework) mapped to QA orbit hierarchy.

Laws: 1 (Matter and Force), 11 (Force), 16 (Oscillating Atomoles)

Core mapping: the invariant substrate ({1,...,N}), the triune
manifestation of the generator (creative/transmissive/attractive),
and the singularity as neutral center (dominant force).

Checks:
  KDC_1       — schema_version matches
  KDC_LAWS    — all 3 law numbers present
  KDC_SUB     — state space is fixed and invariant (Law 1)
  KDC_TRIUNE  — three manifestations of generator declared (Law 11)
  KDC_SING    — singularity identified as neutral center/dominant (Law 16)
  KDC_W       — at least 3 witnesses
  KDC_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Keely dominant/control law mappings; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_KEELY_DOMINANT_CONTROL_CERT.v1"
REQUIRED_LAWS = frozenset([1, 11, 16])


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # KDC_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"KDC_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # KDC_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("KDC_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("KDC_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # KDC_LAWS: all 3 laws present
    laws = cert.get("laws", {})
    declared_nums = set()
    if isinstance(laws, dict):
        declared_nums = {int(k) for k in laws.keys()}
    elif isinstance(laws, list):
        declared_nums = {entry.get("law_number") for entry in laws if isinstance(entry, dict)}
    missing = REQUIRED_LAWS - declared_nums
    if missing:
        errors.append(f"KDC_LAWS: missing law numbers: {sorted(missing)}")

    # KDC_SUB: invariant substrate (Law 1)
    substrate = cert.get("invariant_substrate")
    if substrate:
        modulus = substrate.get("modulus")
        state_count = substrate.get("state_count")
        if modulus and state_count and state_count != modulus * modulus:
            errors.append(f"KDC_SUB: state_count={state_count}, expected {modulus}*{modulus}={modulus*modulus}")
    else:
        warnings.append("KDC_SUB: invariant_substrate block not declared")

    # KDC_TRIUNE: three manifestations (Law 11)
    triune = cert.get("generator_triune")
    if triune:
        required_forms = {"creative", "transmissive", "attractive"}
        declared_forms = set(triune.keys())
        missing_forms = required_forms - declared_forms
        if missing_forms:
            errors.append(f"KDC_TRIUNE: missing generator forms: {sorted(missing_forms)}")
    else:
        warnings.append("KDC_TRIUNE: generator_triune block not declared")

    # KDC_SING: singularity as neutral center (Law 16)
    sing = cert.get("singularity_dominant")
    if sing:
        if sing.get("is_fixed_point") is not True:
            errors.append("KDC_SING: singularity must be declared as fixed point")
        if sing.get("is_neutral_center") is not True:
            errors.append("KDC_SING: singularity must be declared as neutral center")
    else:
        warnings.append("KDC_SING: singularity_dominant block not declared")

    # KDC_W: witnesses
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        errors.append(f"KDC_W: need >= 3 witnesses, got {len(witnesses)}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("kdc_pass_hierarchy.json", True),
        ("kdc_fail_no_center.json", True),
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
        description="QA Keely Dominant Control Cert [186] validator")
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
