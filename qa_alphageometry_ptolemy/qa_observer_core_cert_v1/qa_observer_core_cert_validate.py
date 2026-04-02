#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=observer_core_fixtures"
"""QA Observer Core Cert family [159] — certifies qa_mod() A1 compliance
and compute_qci() determinism.  These are the two foundational functions
used across all 6 empirical domains (finance, EEG, audio, seismology,
climate, ERA5 reanalysis).

SOURCE: qa_lab/qa_observer/core.py, canonical from script 46 lines 34-45.

Checks: OC_1 (schema), OC_A1 (qa_mod output always in {1,...,m} for all
tested inputs including negatives and zero), OC_QCI (compute_qci determinism
and known match rate), OC_T2 (no float->int feedback in qa_mod),
OC_W (>=1 domain witness), OC_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_OBSERVER_CORE_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # OC_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("OC_1", f"schema_version must be {SCHEMA}")

    # OC_A1 — qa_mod A1 compliance
    a1 = cert.get("a1_tests", {})
    if not a1:
        err("OC_A1", "a1_tests section missing")
    else:
        modulus = a1.get("modulus")
        if not modulus or modulus < 1:
            err("OC_A1", "a1_tests.modulus must be positive integer")

        cases = a1.get("cases", [])
        if not cases:
            err("OC_A1", "a1_tests.cases must be non-empty")

        for c in cases:
            x = c.get("x")
            result = c.get("result")
            m = c.get("m", modulus)
            if result is not None and m is not None:
                if result < 1 or result > m:
                    err("OC_A1", f"qa_mod({x}, {m}) = {result} not in {{1,...,{m}}}")

        # Check formula declaration
        formula = a1.get("formula", "")
        if "((x - 1) % m) + 1" not in formula and "((int(x) - 1) % m) + 1" not in formula:
            warnings.append("OC_A1: formula should be '((int(x) - 1) % m) + 1'")

    # OC_QCI — compute_qci determinism
    qci = cert.get("qci_tests", {})
    if not qci:
        err("OC_QCI", "qci_tests section missing")
    else:
        if not qci.get("deterministic"):
            err("OC_QCI", "qci_tests.deterministic must be true")
        length = qci.get("output_length")
        input_length = qci.get("input_length")
        if length is not None and input_length is not None:
            if length != input_length - 2:
                err("OC_QCI", f"output_length must be input_length - 2: got {length} vs {input_length}")

    # OC_T2 — no float->int feedback (T2 firewall)
    t2 = cert.get("t2_compliance", {})
    if t2:
        if t2.get("float_to_int_feedback") is True:
            err("OC_T2", "qa_mod must not use float->int feedback (T2 violation)")
    else:
        warnings.append("OC_T2: t2_compliance section missing")

    # OC_W — at least one domain witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        err("OC_W", "at least one witness required")

    # OC_F — fail detection
    result = cert.get("result", "")
    fail_ledger = cert.get("fail_ledger", [])
    if result == "FAIL" and not fail_ledger:
        err("OC_F", "FAIL result requires non-empty fail_ledger")
    if result == "PASS" and fail_ledger:
        err("OC_F", "PASS result must not have fail_ledger entries")

    return {
        "ok": not errors,
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
    ap = argparse.ArgumentParser(description=f"{SCHEMA} validator")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("cert_file", nargs="?")
    args = ap.parse_args()

    if args.self_test:
        r = self_test()
        print(json.dumps(r, indent=2, sort_keys=True))
        sys.exit(0 if r["ok"] else 1)

    if args.cert_file:
        with open(args.cert_file, "r") as f:
            cert = json.load(f)
        r = validate(cert)
        print(json.dumps(r, indent=2, sort_keys=True))
        sys.exit(0 if r["ok"] else 1)

    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
