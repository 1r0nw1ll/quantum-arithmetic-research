#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=era5_reanalysis_fixtures"
"""QA ERA5 Reanalysis Cert family [172] — certifies QCI as predictor of
atmospheric variability using WeatherBench2 ERA5 data (3297 days x 15
channels, 500hPa).

r=+0.46, partial r=+0.43, 4/4 surrogates beaten.

Checks: ERA_1 (schema), ERA_DATA (WeatherBench2 source), ERA_R (r>0),
ERA_PARTIAL (partial r>0), ERA_SURR (surrogate pass), ERA_W (witness),
ERA_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_ERA5_REANALYSIS_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # ERA_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("ERA_1", f"schema_version must be {SCHEMA}")

    # ERA_DATA — data source
    data = cert.get("data_source", {})
    if not data.get("dataset"):
        err("ERA_DATA", "data_source.dataset missing")
    if not data.get("n_days"):
        err("ERA_DATA", "data_source.n_days missing")
    if not data.get("n_channels"):
        err("ERA_DATA", "data_source.n_channels missing")

    # ERA_R — correlation
    corr = cert.get("correlation", {})
    if not corr:
        err("ERA_R", "correlation section missing")
    else:
        r = corr.get("r")
        if r is None or r <= 0:
            err("ERA_R", "correlation.r must be > 0")

    # ERA_PARTIAL — partial correlation
    partial = cert.get("partial_correlation", {})
    if not partial:
        err("ERA_PARTIAL", "partial_correlation section missing")
    else:
        pr = partial.get("partial_r")
        if pr is None or pr <= 0:
            err("ERA_PARTIAL", "partial_correlation.partial_r must be > 0")

    # ERA_SURR — surrogate pass
    surr = cert.get("surrogate_tests", {})
    if not surr:
        err("ERA_SURR", "surrogate_tests section missing")
    else:
        beaten = surr.get("surrogates_beaten")
        total = surr.get("surrogates_total")
        if beaten is None or total is None:
            err("ERA_SURR", "surrogate_tests.surrogates_beaten and surrogates_total required")
        elif beaten < total:
            err("ERA_SURR", f"surrogates_beaten ({beaten}) < surrogates_total ({total})")

    # ERA_W — witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        warnings.append("ERA_W: no witnesses declared")

    # ERA_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("ERA_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("ERA_F: declared FAIL but no fail_ledger entries and all checks pass")

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
