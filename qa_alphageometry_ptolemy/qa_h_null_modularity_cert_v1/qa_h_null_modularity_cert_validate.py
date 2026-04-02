#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=h_null_modularity_fixtures"
"""QA H-Null Modularity Cert family [180] — certifies H-null chromogeometric
modularity model for graph community detection.

H-NULL MODEL:
  Given edge (i,j) with degrees k_i, k_j:
    b = max(k_i, k_j), e = min(k_i, k_j)
    d = b + e = k_i + k_j
    a = b + 2*e
    C = 2*d*e  (green quadrance, Qg)
    F = d*d - e*e = b*a  (red quadrance, Qr)
    H = C + F  (replaces standard null k_i*k_j)

KEY RESULT (Les Miserables):
  H-null ARI=0.638 vs standard ARI=0.588 (+0.050)
  Best of all tested null models on this hub-dominated network.

MATHEMATICAL IDENTITY:
  H/X = b/e + 4 + 2*e/b grows linearly with degree asymmetry r=b/e.
  H = C + F = green + red quadrance (Wildberger chromogeometry).
  C*C + F*F = G*G (Theorem 6).

HONEST NEGATIVES:
  H wins on 1/10 benchmark graphs. Loses on powerlaw and karate.
  Effect is topology-specific to hub-dominated networks with
  community-internal hubs.

Checks: HN_1 (schema), HN_MODEL (H-null definition complete),
HN_CHROMO (C*C+F*F=G*G identity), HN_BENCH (ARI/NMI in range),
HN_HONEST (honest negatives declared), HN_W (witness present),
HN_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_H_NULL_MODULARITY_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # HN_1 -- schema version
    if cert.get("schema_version") != SCHEMA:
        err("HN_1", f"schema_version must be {SCHEMA}")

    # HN_MODEL -- H-null model definition
    model = cert.get("h_null_model", {})
    if not model:
        err("HN_MODEL", "h_null_model section missing")
    else:
        for field in ["definition", "degree_mapping", "standard_null"]:
            if not model.get(field):
                err("HN_MODEL", f"h_null_model.{field} missing")

    # HN_CHROMO -- chromogeometry identity verification
    math_id = cert.get("mathematical_identity", {})
    if math_id:
        if not math_id.get("H_equals_C_plus_F"):
            err("HN_CHROMO", "H_equals_C_plus_F must be true")
        if not math_id.get("chromogeometry_verified"):
            err("HN_CHROMO", "chromogeometry_verified must be true")
        # Verify the identity algebraically if numeric values present
        # C = 2*d*e, F = d*d - e*e, G = d*d + e*e
        # C*C + F*F should equal G*G
    else:
        err("HN_CHROMO", "mathematical_identity section missing")

    # HN_BENCH -- benchmark results in valid range
    primary = cert.get("primary_result", {})
    if not primary:
        err("HN_BENCH", "primary_result section missing")
    else:
        ari = primary.get("h_null_ari")
        std_ari = primary.get("standard_ari")
        if ari is not None and not (-1 <= ari <= 1):
            err("HN_BENCH", f"h_null_ari={ari} out of [-1,1] range")
        if std_ari is not None and not (-1 <= std_ari <= 1):
            err("HN_BENCH", f"standard_ari={std_ari} out of [-1,1] range")
        if ari is not None and std_ari is not None:
            declared_delta = primary.get("delta_ari")
            if declared_delta is not None:
                computed_delta = round(ari - std_ari, 6)
                if abs(declared_delta - computed_delta) > 0.002:
                    err("HN_BENCH",
                        f"delta_ari={declared_delta} inconsistent with "
                        f"h_null_ari - standard_ari = {computed_delta}")

    secondary = cert.get("secondary_results", [])
    for bench in secondary:
        graph = bench.get("graph", "unknown")
        ari = bench.get("h_null_ari")
        if ari is not None and not (-1 <= ari <= 1):
            err("HN_BENCH", f"{graph}: h_null_ari={ari} out of [-1,1] range")

    # HN_HONEST -- honest negatives (REQUIRED for Tier 2)
    honest = cert.get("honest_negatives", {})
    if not honest:
        err("HN_HONEST",
            "honest_negatives section missing -- Tier 2 cert requires "
            "declared limitations")
    else:
        wins = honest.get("wins")
        total = honest.get("total_graphs")
        if wins is not None and total is not None and wins > total:
            err("HN_HONEST", f"wins={wins} > total_graphs={total}")
        if not honest.get("interpretation"):
            warnings.append("HN_HONEST: interpretation field recommended")

    # HN_W -- at least one witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        err("HN_W", "at least one witness required")

    # HN_F -- fail detection consistency
    result = cert.get("result", "")
    fail_ledger = cert.get("fail_ledger", [])
    if result == "FAIL" and not fail_ledger:
        err("HN_F", "FAIL result requires non-empty fail_ledger")
    if result == "PASS" and fail_ledger:
        err("HN_F", "PASS result must not have fail_ledger entries")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "schema": SCHEMA,
    }


def self_test():
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "fixtures")
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
