#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=surrogate_methodology_fixtures"
"""QA Surrogate Methodology Cert family [173] — certifies the corrected
surrogate null design where real targets are fixed and only surrogate QCI
is randomized. Identifies and resolves the circular null problem.
6/8 domains confirmed with corrected design.

Checks: SRM_1 (schema), SRM_DESIGN (corrected null design declared),
SRM_CIRCULAR (circular problem identified), SRM_DOMAINS (>=6 domains confirmed),
SRM_W (witness), SRM_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_SURROGATE_METHODOLOGY_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # SRM_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("SRM_1", f"schema_version must be {SCHEMA}")

    # SRM_DESIGN — corrected null design
    design = cert.get("null_design", {})
    if not design:
        err("SRM_DESIGN", "null_design section missing")
    else:
        if not design.get("targets_fixed"):
            err("SRM_DESIGN", "null_design.targets_fixed must be true")
        if not design.get("surrogate_qci_only"):
            err("SRM_DESIGN", "null_design.surrogate_qci_only must be true")

    # SRM_CIRCULAR — circular problem identified
    circular = cert.get("circular_null_problem", {})
    if not circular:
        err("SRM_CIRCULAR", "circular_null_problem section missing")
    else:
        if not circular.get("identified"):
            err("SRM_CIRCULAR", "circular_null_problem.identified must be true")
        if not circular.get("resolution"):
            err("SRM_CIRCULAR", "circular_null_problem.resolution missing")

    # SRM_DOMAINS — domain count
    domains = cert.get("confirmed_domains", [])
    if len(domains) < 6:
        err("SRM_DOMAINS", f"confirmed_domains has {len(domains)} entries, need >= 6")

    # SRM_W — witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        warnings.append("SRM_W: no witnesses declared")

    # SRM_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("SRM_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("SRM_F: declared FAIL but no fail_ledger entries and all checks pass")

    return {
        "ok": not has_errors,
        "errors": errors,
        "warnings": warnings,
        "schema": SCHEMA,
    }


def self_test():
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    results = {"pass_count": 0, "fail_count": 0, "errors": []}

    for fname in sorted(os.listdir(fixture_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(fixture_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            cert = json.load(f)

        out = validate(cert)
        declared = cert.get("result", "UNKNOWN")

        if declared == "PASS" and out["ok"]:
            results["pass_count"] += 1
        elif declared == "FAIL" and not out["ok"]:
            results["fail_count"] += 1
        else:
            results["errors"].append({
                "fixture": fname,
                "declared": declared,
                "validator_ok": out["ok"],
                "issues": out["errors"],
            })

    results["ok"] = len(results["errors"]) == 0
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{SCHEMA} validator")
    parser.add_argument("--self-test", action="store_true", help="Run self-test against fixtures")
    parser.add_argument("cert_file", nargs="?", help="Path to certificate JSON")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    if args.cert_file:
        with open(args.cert_file, "r", encoding="utf-8") as f:
            cert = json.load(f)
        result = validate(cert)
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
