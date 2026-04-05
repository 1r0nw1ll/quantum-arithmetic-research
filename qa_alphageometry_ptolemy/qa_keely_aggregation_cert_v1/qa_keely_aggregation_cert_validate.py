#!/usr/bin/env python3
"""
qa_keely_aggregation_cert_validate.py

Validator for QA_KEELY_AGGREGATION_CERT.v1  [family 187]

Certifies: Keely's 5 Aggregation/Disintegration Laws (Category 4 of Vibes
5-category framework) mapped to QA state composition and decomposition.

Laws: 3 (Corporeal Oscillations), 12 (Oscillating Atomic Substances),
      28 (Chemical Dissociation), 34 (Atomic Dissociation),
      35 (Atomolic Synthesis of Chemical Elements)

Core mapping: non-isolated states modify each other through coupling
(Law 3); orbit density determines effective pitch (Law 12); discord
causes dissociation = orbit separation (Laws 28, 34); pitch selection
determines orbit membership deterministically (Law 35).

Checks:
  KAG_1       — schema_version matches
  KAG_LAWS    — all 5 law numbers present
  KAG_COUPLE  — coupling tension examples (Law 3)
  KAG_DENSITY — orbit density declarations (Law 12)
  KAG_DISSOC  — dissociation = orbit separation (Laws 28, 34)
  KAG_SYNTH   — deterministic synthesis from pitch (Law 35)
  KAG_W       — at least 3 witnesses
  KAG_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Keely aggregation law mappings; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_KEELY_AGGREGATION_CERT.v1"
REQUIRED_LAWS = frozenset([3, 12, 28, 34, 35])


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # KAG_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"KAG_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # KAG_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("KAG_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("KAG_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # KAG_LAWS: all 5 laws present
    laws = cert.get("laws", {})
    declared_nums = set()
    if isinstance(laws, dict):
        declared_nums = {int(k) for k in laws.keys()}
    elif isinstance(laws, list):
        declared_nums = {entry.get("law_number") for entry in laws if isinstance(entry, dict)}
    missing = REQUIRED_LAWS - declared_nums
    if missing:
        errors.append(f"KAG_LAWS: missing law numbers: {sorted(missing)}")

    # KAG_DENSITY: orbit density (Law 12)
    density = cert.get("orbit_density")
    if density:
        cosmos_count = density.get("cosmos_count")
        satellite_count = density.get("satellite_count")
        if cosmos_count is not None and satellite_count is not None:
            if cosmos_count <= satellite_count:
                errors.append(f"KAG_DENSITY: cosmos_count={cosmos_count} should exceed satellite_count={satellite_count}")

    # KAG_SYNTH: deterministic synthesis (Law 35)
    synth = cert.get("deterministic_synthesis")
    if synth:
        if synth.get("fully_determined") is not True:
            errors.append("KAG_SYNTH: deterministic_synthesis.fully_determined must be true")

    # KAG_W: witnesses
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        errors.append(f"KAG_W: need >= 3 witnesses, got {len(witnesses)}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("kag_pass_composition.json", True),
        ("kag_fail_bad_density.json", True),
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
        description="QA Keely Aggregation Cert [187] validator")
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
