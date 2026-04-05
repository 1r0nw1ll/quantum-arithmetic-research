#!/usr/bin/env python3
"""
qa_keely_phenomenological_cert_validate.py

Validator for QA_KEELY_PHENOMENOLOGICAL_CERT.v1  [family 188]

Certifies: Keely's 17 Phenomenological Laws (Category 5 of Vibes
5-category framework) classified as observer projections under Theorem NT.

Laws: 13 (Sono-thermity), 14 (Oscillating Atoms), 15 (Vibrating
Atomolic Substances), 19-26 (Variation laws), 30 (Chemical Substitution),
31 (Catalysis), 32 (Molecular Synthesis/Organic), 36 (Heat),
38 (Cohesion), 39 (Refractive Indices)

Core mapping: ALL 17 laws describe continuous measurements (temperature,
pressure, frequency, refractive index, etc.) that are EFFECTS of the
underlying discrete QA structure. Under Theorem NT, these are observer
projections — they reveal but never causally feed back into QA logic.

Checks:
  KPH_1       — schema_version matches
  KPH_LAWS    — all 17 law numbers present
  KPH_NT      — Theorem NT compliance declared for all laws
  KPH_OBS     — each law identifies the continuous observable(s)
  KPH_DISC    — each law identifies the discrete QA source
  KPH_W       — at least 3 witnesses
  KPH_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Keely phenomenological law mappings; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_KEELY_PHENOMENOLOGICAL_CERT.v1"
REQUIRED_LAWS = frozenset([13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 36, 38, 39])


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # KPH_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"KPH_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # KPH_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("KPH_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("KPH_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # KPH_LAWS: all 17 laws present
    laws = cert.get("laws", {})
    declared_nums = set()
    if isinstance(laws, dict):
        declared_nums = {int(k) for k in laws.keys()}
    elif isinstance(laws, list):
        declared_nums = {entry.get("law_number") for entry in laws if isinstance(entry, dict)}
    missing = REQUIRED_LAWS - declared_nums
    if missing:
        errors.append(f"KPH_LAWS: missing law numbers: {sorted(missing)}")

    # KPH_NT: Theorem NT compliance
    nt = cert.get("theorem_nt_compliance")
    if nt:
        if nt.get("all_laws_observer_projections") is not True:
            errors.append("KPH_NT: all_laws_observer_projections must be true")
        if nt.get("no_causal_feedback") is not True:
            errors.append("KPH_NT: no_causal_feedback must be true")
    else:
        errors.append("KPH_NT: theorem_nt_compliance block required")

    # KPH_OBS + KPH_DISC: each law should declare observable and discrete source
    if isinstance(laws, dict):
        for law_num, law_data in laws.items():
            if not isinstance(law_data, dict):
                continue
            obs = law_data.get("continuous_observable")
            disc = law_data.get("discrete_qa_source")
            if obs is None:
                warnings.append(f"KPH_OBS: law {law_num} missing continuous_observable")
            if disc is None:
                warnings.append(f"KPH_DISC: law {law_num} missing discrete_qa_source")

    # KPH_W: witnesses
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        errors.append(f"KPH_W: need >= 3 witnesses, got {len(witnesses)}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("kph_pass_observer.json", True),
        ("kph_fail_no_nt.json", True),
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
        description="QA Keely Phenomenological Cert [188] validator")
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
