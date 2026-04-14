#!/usr/bin/env python3
"""Validator for QA_PUDELKO_MODULAR_PERIODICITY_CERT.v1 [family 198]."""

QA_COMPLIANCE = "cert_validator - Pudelko modular periodicity bridge; integer counts only; partial-verification honesty gate; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_PUDELKO_MODULAR_PERIODICITY_CERT.v1"


def _run_checks(fixture):
    results = {}
    results["PUD_1"] = fixture.get("schema_version") == SCHEMA_VERSION
    results["PUD_STATUS"] = fixture.get("status") == "PARTIALLY_VERIFIED"

    counts = fixture.get("orbit_family_counts", [])
    expected = {3: 3, 9: 9, 27: 27}
    actual = {}
    for row in counts:
        if isinstance(row, dict):
            actual[row.get("modulus")] = row.get("families")
    results["PUD_ORBIT"] = actual == expected

    sim = fixture.get("fractal_self_similarity", {})
    results["PUD_SELF_SIM"] = (
        sim.get("pattern") == "3^k families for mod-3^k"
        and sim.get("satellite_invariant") is True
        and sim.get("singularity_invariant") is True
        and sim.get("cosmos_scales") is True
    )

    weight = fixture.get("weight_preservation", {})
    results["PUD_WEIGHT"] = weight.get("non_singularity_weight_preserved") is True

    status = fixture.get("verification_status", {})
    results["PUD_HONEST"] = (
        status.get("V1") == "VERIFIED"
        and status.get("V3") == "VERIFIED"
        and status.get("V4") == "VERIFIED"
        and status.get("V6") == "VERIFIED"
        and status.get("V7") == "VERIFIED"
        and status.get("V8") == "VERIFIED"
        and status.get("V2") in {"PARTIALLY_VERIFIED", "NEEDS_REFINEMENT"}
        and status.get("V5") == "OPEN"
    )

    src = fixture.get("source_attribution", "")
    results["PUD_SRC"] = "Pudelko" in src and "2510.24882" in src

    witnesses = fixture.get("witnesses", [])
    kinds = {w.get("kind") for w in witnesses if isinstance(w, dict)}
    results["PUD_WITNESS"] = {"orbit_count", "self_similarity", "weight"}.issubset(kinds)

    fail_ledger = fixture.get("fail_ledger")
    results["PUD_F"] = isinstance(fail_ledger, list) and len(fail_ledger) >= 1
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
    print("Usage: python qa_pudelko_modular_periodicity_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
