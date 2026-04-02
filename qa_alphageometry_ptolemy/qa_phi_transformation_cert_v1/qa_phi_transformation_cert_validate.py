#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=phi_transformation_fixtures"
"""QA Phi Transformation Cert family [174] — certifies the Phi(D)
transformation law classifying disorder-stress vs order-stress.

2/2 pre-registered (cardiac, EMG), 6/6 post-hoc consistent.
Domain requirement: temporal multi-channel signals with non-trivial baselines.

Checks: PHI_1 (schema), PHI_CLASS (classification declared),
PHI_PREREG (>=2 pre-registered), PHI_POSTHOC (>=4 post-hoc consistent),
PHI_REQ (domain requirement stated), PHI_W (witness), PHI_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_PHI_TRANSFORMATION_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # PHI_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("PHI_1", f"schema_version must be {SCHEMA}")

    # PHI_CLASS — classification declared
    classification = cert.get("classification", {})
    if not classification:
        err("PHI_CLASS", "classification section missing")
    else:
        if not classification.get("disorder_stress"):
            err("PHI_CLASS", "classification.disorder_stress missing")
        if not classification.get("order_stress"):
            err("PHI_CLASS", "classification.order_stress missing")

    # PHI_PREREG — pre-registered domains
    prereg = cert.get("pre_registered", [])
    if len(prereg) < 2:
        err("PHI_PREREG", f"pre_registered has {len(prereg)} entries, need >= 2")

    # PHI_POSTHOC — post-hoc consistent domains
    posthoc = cert.get("post_hoc_consistent", [])
    if len(posthoc) < 4:
        err("PHI_POSTHOC", f"post_hoc_consistent has {len(posthoc)} entries, need >= 4")

    # PHI_REQ — domain requirement
    req = cert.get("domain_requirement", "")
    if not req:
        err("PHI_REQ", "domain_requirement missing")

    # PHI_W — witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        warnings.append("PHI_W: no witnesses declared")

    # PHI_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("PHI_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("PHI_F: declared FAIL but no fail_ledger entries and all checks pass")

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
