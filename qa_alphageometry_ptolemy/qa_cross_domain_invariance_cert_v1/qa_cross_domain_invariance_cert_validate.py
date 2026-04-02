#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=cross_domain_invariance_fixtures"
"""QA Cross-Domain Invariance Cert family [175] — certifies 3 structural
invariants of the QA observer framework across 6 Tier 3 domains:
1) surrogate survival, 2) independent information, 3) domain-general architecture.

Checks: CDI_1 (schema), CDI_INV1 (surrogate survival), CDI_INV2 (independent info),
CDI_INV3 (domain-general architecture), CDI_W (witness), CDI_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_CROSS_DOMAIN_INVARIANCE_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # CDI_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("CDI_1", f"schema_version must be {SCHEMA}")

    # CDI_INV1 — surrogate survival
    inv1 = cert.get("invariant_surrogate_survival", {})
    if not inv1:
        err("CDI_INV1", "invariant_surrogate_survival section missing")
    else:
        if not inv1.get("description"):
            err("CDI_INV1", "invariant_surrogate_survival.description missing")
        domains = inv1.get("domains_confirmed", [])
        if len(domains) < 6:
            err("CDI_INV1", f"invariant_surrogate_survival.domains_confirmed has {len(domains)}, need >= 6")

    # CDI_INV2 — independent information
    inv2 = cert.get("invariant_independent_info", {})
    if not inv2:
        err("CDI_INV2", "invariant_independent_info section missing")
    else:
        if not inv2.get("description"):
            err("CDI_INV2", "invariant_independent_info.description missing")
        domains = inv2.get("domains_confirmed", [])
        if len(domains) < 6:
            err("CDI_INV2", f"invariant_independent_info.domains_confirmed has {len(domains)}, need >= 6")

    # CDI_INV3 — domain-general architecture
    inv3 = cert.get("invariant_domain_general", {})
    if not inv3:
        err("CDI_INV3", "invariant_domain_general section missing")
    else:
        if not inv3.get("description"):
            err("CDI_INV3", "invariant_domain_general.description missing")
        domains = inv3.get("domains_confirmed", [])
        if len(domains) < 6:
            err("CDI_INV3", f"invariant_domain_general.domains_confirmed has {len(domains)}, need >= 6")

    # CDI_W — witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        warnings.append("CDI_W: no witnesses declared")

    # CDI_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("CDI_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("CDI_F: declared FAIL but no fail_ledger entries and all checks pass")

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
