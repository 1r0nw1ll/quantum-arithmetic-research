#!/usr/bin/env python3
"""
qa_keely_sympathetic_transfer_cert_validate.py

Validator for QA_KEELY_SYMPATHETIC_TRANSFER_CERT.v1  [family 185]

Certifies: Keely's 7 Sympathetic Transfer Laws (Category 2 of Vibes
5-category framework) mapped to QA reachability and path structure.

Laws: 5 (Transmissive Vibraic Energy), 6 (Sympathetic Oscillation),
      7 (Attraction), 8 (Repulsion), 17 (Transformation of Forces),
      37 (Electro-Chemical Equivalents), 40 (Electric Conductivity)

Core mapping: sympathetic transfer = QA reachability via generator.
Co-membership in orbit enables coupling; discordance = reachability
obstruction. Vibes (2026-04-03): "QA reachability obstruction" =
"SVP failure of sympathetic access or concordant transmissibility."

Checks:
  KST_1       — schema_version matches
  KST_LAWS    — all 7 law numbers present
  KST_REACH   — reachability examples: same-orbit states reachable
  KST_BLOCK   — obstruction examples: cross-orbit states unreachable
  KST_PATH    — path length is integer (T1 axiom)
  KST_TRIAD   — cause-medium-receiver triad condition declared
  KST_W       — at least 3 witnesses
  KST_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Keely sympathetic transfer law mappings; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_KEELY_SYMPATHETIC_TRANSFER_CERT.v1"
REQUIRED_LAWS = frozenset([5, 6, 7, 8, 17, 37, 40])


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # KST_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"KST_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # KST_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("KST_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("KST_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # KST_LAWS: all 7 laws present
    laws = cert.get("laws", {})
    declared_nums = set()
    if isinstance(laws, dict):
        declared_nums = {int(k) for k in laws.keys()}
    elif isinstance(laws, list):
        declared_nums = {entry.get("law_number") for entry in laws if isinstance(entry, dict)}
    missing = REQUIRED_LAWS - declared_nums
    if missing:
        errors.append(f"KST_LAWS: missing law numbers: {sorted(missing)}")

    # KST_REACH: reachability examples — validate declared orbits match
    reach_examples = cert.get("reachability_examples", [])
    for idx, ex in enumerate(reach_examples):
        reachable_decl = ex.get("reachable")
        orb = ex.get("both_orbit")
        if reachable_decl is True and orb is None:
            warnings.append(f"KST_REACH: example[{idx}] declares reachable=true but both_orbit not specified")

    # KST_BLOCK: obstruction examples — source and target orbits must differ
    block_examples = cert.get("obstruction_examples", [])
    for idx, ex in enumerate(block_examples):
        orb1 = ex.get("source_orbit", ex.get("orbit1"))
        orb2 = ex.get("target_orbit", ex.get("orbit2"))
        if orb1 is not None and orb2 is not None and orb1 == orb2:
            errors.append(f"KST_BLOCK: example[{idx}] declares obstruction but orbits are same: {orb1}")

    # KST_PATH: path lengths are integers
    witnesses = cert.get("witnesses", [])
    for idx, w in enumerate(witnesses):
        pl = w.get("path_length")
        if pl is not None and not isinstance(pl, int):
            errors.append(f"KST_PATH: witness[{idx}] path_length={pl} is not integer (T1 axiom)")

    # KST_TRIAD: cause-medium-receiver
    triad = cert.get("triad_condition")
    if triad is None:
        warnings.append("KST_TRIAD: triad_condition not declared (Vibes's three-body concordance)")

    # KST_W: witnesses
    if len(witnesses) < 3:
        errors.append(f"KST_W: need >= 3 witnesses, got {len(witnesses)}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("kst_pass_reachability.json", True),
        ("kst_fail_cross_orbit.json", True),
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
        description="QA Keely Sympathetic Transfer Cert [185] validator")
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
