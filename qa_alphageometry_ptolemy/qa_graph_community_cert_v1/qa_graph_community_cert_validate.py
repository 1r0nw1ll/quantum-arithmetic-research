#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=graph_community_fixtures"
"""QA Graph Community Cert family [158] — certifies QA feature map
dimensions and community detection on benchmark graphs (football,
karate, dolphins).

SOURCE: codex_on_QA/ spectral clustering framework (Rust + Python),
now consolidated in qa_lab/qa_graph/.

Checks: GC_1 (schema), GC_DIM (feature vector dimensions: qa21=21,
qa27=27, qa83=83), GC_BENCH (benchmark graph metrics in expected range),
GC_CHROMO (C^2+F^2=G^2 identity holds), GC_W (>=1 benchmark witness),
GC_F (fail detection).
"""

import json
import os
import sys

SCHEMA = "QA_GRAPH_COMMUNITY_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # GC_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("GC_1", f"schema_version must be {SCHEMA}")

    # GC_DIM — feature map dimensions
    dims = cert.get("feature_dimensions", {})
    if not dims:
        err("GC_DIM", "feature_dimensions section missing")
    else:
        expected = {"qa21": 21, "qa27": 27, "qa83": 83}
        for mode, exp_dim in expected.items():
            actual = dims.get(mode)
            if actual != exp_dim:
                err("GC_DIM", f"{mode}: expected {exp_dim}, got {actual}")

    # GC_CHROMO — chromogeometry identity
    chromo = cert.get("chromogeometry_check", {})
    if chromo:
        for test in chromo.get("tests", []):
            b, e = test.get("b"), test.get("e")
            residual = test.get("C2_plus_F2_minus_G2", None)
            if residual is not None and abs(residual) > 1e-8:
                err("GC_CHROMO", f"C^2+F^2-G^2 != 0 for ({b},{e}): residual={residual}")
    else:
        warnings.append("GC_CHROMO: chromogeometry_check section missing")

    # GC_BENCH — benchmark results
    benchmarks = cert.get("benchmark_results", [])
    if not benchmarks:
        err("GC_BENCH", "benchmark_results must be non-empty")
    else:
        for bench in benchmarks:
            graph = bench.get("graph", "unknown")
            ari = bench.get("ari")
            nmi = bench.get("nmi")
            if ari is not None and not (-1 <= ari <= 1):
                err("GC_BENCH", f"{graph}: ARI={ari} out of [-1,1] range")
            if nmi is not None and not (0 <= nmi <= 1):
                err("GC_BENCH", f"{graph}: NMI={nmi} out of [0,1] range")

    # GC_W — at least one benchmark witness
    witnesses = cert.get("witnesses", [])
    if not witnesses:
        err("GC_W", "at least one witness required")

    # GC_F — fail detection
    result = cert.get("result", "")
    fail_ledger = cert.get("fail_ledger", [])
    if result == "FAIL" and not fail_ledger:
        err("GC_F", "FAIL result requires non-empty fail_ledger")
    if result == "PASS" and fail_ledger:
        err("GC_F", "PASS result must not have fail_ledger entries")

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
