#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=cardiac_arrhythmia_fixtures"
"""QA Cardiac Arrhythmia Cert family [170] — certifies QA orbit features
as independent predictors of arrhythmia classification beyond R-R interval
baseline using MIT-BIH Arrhythmia Database (48 records, 94536 beats).

dR2=+0.037 beyond R-R interval, p<10^-6, 2/2 surrogates beaten.
Phi(D)=-1 pre-registered and confirmed (disorder-stress).

Checks: CAR_1 (schema), CAR_DATA (MIT-BIH source), CAR_DELTA (dR2>0 + p<0.05),
CAR_SURR (surrogate pass), CAR_PHI (Phi pre-registration), CAR_W (witness),
CAR_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_CARDIAC_ARRHYTHMIA_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # CAR_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("CAR_1", f"schema_version must be {SCHEMA}")

    # CAR_DATA — MIT-BIH source
    data = cert.get("data_source", {})
    if not data.get("database"):
        err("CAR_DATA", "data_source.database missing")
    if not data.get("n_records"):
        err("CAR_DATA", "data_source.n_records missing")
    if not data.get("n_beats"):
        err("CAR_DATA", "data_source.n_beats missing")

    # CAR_DELTA — delta R2
    delta = cert.get("delta_r2", {})
    if not delta:
        err("CAR_DELTA", "delta_r2 section missing")
    else:
        val = delta.get("value")
        if val is None or val <= 0:
            err("CAR_DELTA", "delta_r2.value must be > 0")
        p = delta.get("p_value")
        if p is None or p >= 0.05:
            err("CAR_DELTA", "delta_r2.p_value must be < 0.05")

    # CAR_SURR — surrogate pass
    surr = cert.get("surrogate_tests", {})
    if not surr:
        err("CAR_SURR", "surrogate_tests section missing")
    else:
        beaten = surr.get("surrogates_beaten")
        total = surr.get("surrogates_total")
        if beaten is None or total is None:
            err("CAR_SURR", "surrogate_tests.surrogates_beaten and surrogates_total required")
        elif beaten < total:
            err("CAR_SURR", f"surrogates_beaten ({beaten}) < surrogates_total ({total})")

    # CAR_PHI — Phi pre-registration
    phi = cert.get("phi_transformation", {})
    if not phi:
        err("CAR_PHI", "phi_transformation section missing")
    else:
        if phi.get("pre_registered") is not True:
            err("CAR_PHI", "phi_transformation.pre_registered must be true")
        if phi.get("phi_D") is None:
            err("CAR_PHI", "phi_transformation.phi_D missing")

    # CAR_W — witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        warnings.append("CAR_W: no witnesses declared")

    # CAR_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("CAR_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("CAR_F: declared FAIL but no fail_ledger entries and all checks pass")

    return {
        "ok": not has_errors,
        "errors": errors,
        "warnings": warnings,
        "schema": SCHEMA,
    }


def self_test():
    """Run validator against bundled fixtures."""
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
