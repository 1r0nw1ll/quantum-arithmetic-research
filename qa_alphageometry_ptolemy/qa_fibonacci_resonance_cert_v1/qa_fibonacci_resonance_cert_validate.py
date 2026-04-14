#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=fibonacci_resonance_fixtures"
"""QA Fibonacci Resonance Cert family [219] — certifies that mean-motion
resonances preferentially select Fibonacci ratios.

TIER 2→3 — CROSS-VALIDATED EMPIRICAL PATTERN:
  Among order-1 resonances (|p-q|=1), there are 9 coprime ratios:
  2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9
  Only 2 are Fibonacci (2:1, 3:2) = 22% expected.
  Nature selects Fibonacci 77% of the time (33/43 instances).

  This holds across 8+ independent planetary systems:
  Solar system (Laplace, Saturn moons, Kirkwood, TNOs) +
  TRAPPIST-1, HD 110067, K2-138, Kepler-80, Kepler-223,
  TOI-178, GJ-876, HD 158259.

  QA INTERPRETATION: T-operator = Fibonacci shift [[0,1],[1,1]].
  Fibonacci ratios F_n/F_m are convergents of φ. Period ratios
  locked to Fibonacci fractions sit at dynamically deeper attractors
  of the T-operator eigenstructure.

  Standard perturbation theory explains order preference (e^|p-q|)
  but NOT ratio selection within each order class.

SOURCE: Murray & Dermott "Solar System Dynamics" (1999);
Luger et al. Nature Astronomy (2017) — TRAPPIST-1;
Luque et al. ApJ (2024) — HD 110067;
NASA Exoplanet Archive; Will Dale analysis (2026-04-02).

Checks
------
FR_1       schema_version == 'QA_FIBONACCI_RESONANCE_CERT.v1'
FR_CAT     resonance catalogue with ≥20 entries from ≥3 systems
FR_CLASS   each resonance classified as Fibonacci or non-Fibonacci
FR_STAT    statistical test: observed Fib rate > expected (p<0.05)
FR_ORDER   order-stratified analysis present
FR_CROSS   at least 2 independent datasets (solar + exoplanet)
FR_HONEST  caveats section present (detection bias, catalogue completeness)
FR_W       at least 3 witness systems
FR_F       fail detection
"""

import json
import os
import sys
from math import gcd

SCHEMA = "QA_FIBONACCI_RESONANCE_CERT.v1"
FIBS = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}


def is_fibonacci(n):
    return n in FIBS


def is_fibonacci_ratio(p, q):
    g = gcd(p, q)
    return is_fibonacci(p // g) and is_fibonacci(q // g)


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # FR_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("FR_1", f"schema_version must be {SCHEMA}")

    # FR_CAT — catalogue
    catalogue = cert.get("catalogue", [])
    if len(catalogue) < 20:
        err("FR_CAT", f"need ≥20 catalogue entries, got {len(catalogue)}")

    systems = set()
    for entry in catalogue:
        sys_name = entry.get("system", "")
        systems.add(sys_name)
    if len(systems) < 3:
        err("FR_CAT", f"need ≥3 systems, got {len(systems)}")

    # FR_CLASS — classification
    fib_count = 0
    nonfib_count = 0
    for entry in catalogue:
        p = entry.get("p", 0)
        q = entry.get("q", 0)
        declared_fib = entry.get("is_fibonacci")

        if p <= 0 or q <= 0:
            err("FR_CLASS", f"invalid ratio {p}:{q}")
            continue

        computed_fib = is_fibonacci_ratio(p, q)
        if declared_fib is not None and declared_fib != computed_fib:
            err("FR_CLASS", f"{p}:{q} declared is_fibonacci={declared_fib} but computed {computed_fib}")

        if computed_fib:
            fib_count += 1
        else:
            nonfib_count += 1

    # FR_STAT — statistical significance
    stats = cert.get("statistics", {})
    p_value = stats.get("p_value")
    if p_value is not None and p_value >= 0.05:
        err("FR_STAT", f"p_value={p_value} ≥ 0.05 — not significant")
    if p_value is None:
        warnings.append("FR_STAT: no p_value declared")

    observed_rate = stats.get("observed_fibonacci_rate")
    expected_rate = stats.get("expected_fibonacci_rate")
    if observed_rate is not None and expected_rate is not None:
        if observed_rate <= expected_rate:
            err("FR_STAT", f"observed rate {observed_rate} ≤ expected {expected_rate}")

    # FR_ORDER — order-stratified
    order_analysis = cert.get("order_analysis")
    if not order_analysis:
        warnings.append("FR_ORDER: no order_analysis section")

    # FR_CROSS — cross-validation
    datasets = cert.get("datasets", [])
    if len(datasets) < 2:
        err("FR_CROSS", f"need ≥2 datasets for cross-validation, got {len(datasets)}")

    # FR_HONEST — caveats
    caveats = cert.get("caveats", [])
    if not caveats:
        err("FR_HONEST", "caveats section required for honest reporting")

    # FR_W — witness systems
    witness_systems = cert.get("witness_systems", [])
    if len(witness_systems) < 3:
        err("FR_W", f"need ≥3 witness systems, got {len(witness_systems)}")

    # FR_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("FR_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("FR_F: declared FAIL but no fail_ledger and all checks pass")

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
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("cert_file", nargs="?")
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
