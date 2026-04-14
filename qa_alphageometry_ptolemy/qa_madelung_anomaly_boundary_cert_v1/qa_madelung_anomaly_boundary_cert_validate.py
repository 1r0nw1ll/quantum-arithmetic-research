#!/usr/bin/env python3
"""
qa_madelung_anomaly_boundary_cert_validate.py  [family 222]

QA_MADELUNG_ANOMALY_BOUNDARY_CERT.v1 validator.

Claim: every known atomic Madelung anomaly has |d(src) - d(dst)| <= 1
where d = n + l is the QA d-coordinate under (b, e) = (n, l).
"""

QA_COMPLIANCE = "cert_validator - Madelung anomaly boundary zone |Δd|<=1 on QA d-coordinate; integer state space (n, l); A1/A2 compliant; no ** operator; no float QA state (null test uses Fraction for binomial)"

import json
import sys
from fractions import Fraction
from pathlib import Path

SCHEMA_VERSION = "QA_MADELUNG_ANOMALY_BOUNDARY_CERT.v1"


def _run_checks(fixture):
    results = {}

    results["MAB_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    anoms = fixture.get("anomalies", [])
    a_ok = isinstance(anoms, list) and len(anoms) >= 1
    results["MAB_ANOMALIES"] = a_ok

    # MAB_MAPPING: check d = n+l and delta_d = |d_src - d_dst|
    mapping_ok = a_ok
    if mapping_ok:
        for row in anoms:
            src = row.get("source", {})
            dst = row.get("destination", {})
            n_s, l_s, d_s = src.get("n"), src.get("l"), src.get("d")
            n_d, l_d, d_d = dst.get("n"), dst.get("l"), dst.get("d")
            if not all(isinstance(x, int) for x in (n_s, l_s, d_s, n_d, l_d, d_d)):
                mapping_ok = False
                break
            if d_s != n_s + l_s or d_d != n_d + l_d:
                mapping_ok = False
                break
            if row.get("delta_d") != abs(d_s - d_d):
                mapping_ok = False
                break
    results["MAB_MAPPING"] = mapping_ok

    # MAB_ZONE: all entries have |Δd| <= 1
    zone_ok = mapping_ok
    if zone_ok:
        for row in anoms:
            if row.get("delta_d", 99) > 1:
                zone_ok = False
                break
    results["MAB_ZONE"] = zone_ok

    # MAB_COVERAGE: distribution counts match
    dist = fixture.get("distribution", {})
    cov_ok = isinstance(dist, dict)
    if cov_ok:
        c0 = sum(1 for r in anoms if r.get("delta_d") == 0)
        c1 = sum(1 for r in anoms if r.get("delta_d") == 1)
        cge2 = sum(1 for r in anoms if r.get("delta_d", 99) >= 2)
        if (dist.get("delta_d_0") != c0 or dist.get("delta_d_1") != c1
                or dist.get("delta_d_ge_2") != cge2
                or dist.get("total") != len(anoms)):
            cov_ok = False
    results["MAB_COVERAGE"] = cov_ok

    # MAB_NULL
    null = fixture.get("null_test", {})
    null_ok = (
        isinstance(null, dict)
        and "baseline_zone_rate" in null
        and "observed_zone_rate" in null
        and "binomial_p_value" in null
    )
    if null_ok:
        p = null.get("binomial_p_value")
        if not (isinstance(p, (int, float)) and 0 < p < 0.01):
            null_ok = False
    results["MAB_NULL"] = null_ok

    # MAB_COUNTEREX: at least one counterexample listed (zone member without anomaly)
    cex = fixture.get("counterexamples_in_zone_but_no_anomaly", [])
    results["MAB_COUNTEREX"] = isinstance(cex, list) and len(cex) >= 1

    # MAB_SRC
    src = fixture.get("source_attribution", "")
    results["MAB_SRC"] = (
        ("NIST" in src or "Sato" in src)
        and ("Will Dale" in src or "Dale" in src)
    )

    # MAB_WITNESS
    ws = fixture.get("witnesses", [])
    w_ok = isinstance(ws, list) and len(ws) >= 3
    if w_ok:
        kinds = {w.get("kind") for w in ws}
        if not {"zone_coverage", "null_significance", "necessity_not_sufficiency"}.issubset(kinds):
            w_ok = False
    results["MAB_WITNESS"] = w_ok

    # MAB_F
    fl = fixture.get("fail_ledger")
    results["MAB_F"] = isinstance(fl, list)

    return results


def validate_fixture(path):
    with open(path) as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    all_pass = all(checks.values())
    actual = "PASS" if all_pass else "FAIL"
    ok = actual == expected
    return {"ok": ok, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    fdir = Path(__file__).parent / "fixtures"
    results = {}
    for fp in sorted(fdir.glob("*.json")):
        results[fp.name] = validate_fixture(fp)
    all_ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    elif len(sys.argv) > 1:
        r = validate_fixture(sys.argv[1])
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)
    else:
        print("Usage: python qa_madelung_anomaly_boundary_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
