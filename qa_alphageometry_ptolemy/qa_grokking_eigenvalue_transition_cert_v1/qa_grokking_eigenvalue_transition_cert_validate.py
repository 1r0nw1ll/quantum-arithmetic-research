#!/usr/bin/env python3
"""Validator for QA_GROKKING_EIGENVALUE_TRANSITION_CERT.v1 [family 199]."""

QA_COMPLIANCE = "cert_validator - grokking eigenvalue transition partial cert; observer projection evidence kept separate from QA orbit-family count"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_GROKKING_EIGENVALUE_TRANSITION_CERT.v1"


def _run_checks(fixture):
    results = {}
    results["GET_1"] = fixture.get("schema_version") == SCHEMA_VERSION
    results["GET_STATUS"] = fixture.get("status") == "PARTIALLY_VERIFIED"

    prime = fixture.get("prime_control", {})
    results["GET_PRIME"] = (
        prime.get("modulus") == 97
        and prime.get("grokked") is True
        and prime.get("test_accuracy", 0) >= 0.99
        and prime.get("unit_circle_eigenvalues") == 97
        and prime.get("post_grokking_modes") == 17
    )

    composite = fixture.get("composite_target", {})
    results["GET_COMPOSITE"] = (
        composite.get("modulus") == 9
        and composite.get("grokked") is False
        and composite.get("no_grokking_epochs", 0) >= 100000
    )

    correction = fixture.get("mode_count_correction", {})
    results["GET_CORRECTION"] = (
        correction.get("dft_frequency_pairs_m9") == 5
        and correction.get("qa_orbit_families_m9") == 9
        and correction.get("distinct_quantities") is True
    )

    src = fixture.get("source_attribution", "")
    results["GET_SRC"] = "Schiffman" in src and "2602.22600" in src

    witnesses = fixture.get("witnesses", [])
    kinds = {w.get("kind") for w in witnesses if isinstance(w, dict)}
    results["GET_WITNESS"] = {"prime_control", "composite_target", "correction"}.issubset(kinds)

    fail_ledger = fixture.get("fail_ledger")
    results["GET_F"] = isinstance(fail_ledger, list) and len(fail_ledger) >= 1
    return results


def validate_fixture(path):
    with open(path, encoding="utf-8") as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    actual = "PASS" if all(checks.values()) else "FAIL"
    return {"ok": actual == expected, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    fdir = Path(__file__).parent / "fixtures"
    results = {fp.name: validate_fixture(fp) for fp in sorted(fdir.glob("*.json"))}
    ok = all(item["ok"] for item in results.values())
    print(json.dumps({"ok": ok, "results": results}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    if len(sys.argv) == 2:
        result = validate_fixture(sys.argv[1])
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        sys.exit(0 if result["ok"] else 1)
    print("Usage: python qa_grokking_eigenvalue_transition_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
