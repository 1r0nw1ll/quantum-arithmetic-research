#!/usr/bin/env python3
"""Validator for QA_SPHERICAL_GROKKING_THEOREM_NT_CERT.v1 [family 200]."""

QA_COMPLIANCE = "cert_validator - spherical grokking Theorem NT partial cert; continuous magnitude treated as observer-layer quantity"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SPHERICAL_GROKKING_THEOREM_NT_CERT.v1"


def _run_checks(fixture):
    results = {}
    results["SGT_1"] = fixture.get("schema_version") == SCHEMA_VERSION
    results["SGT_STATUS"] = fixture.get("status") == "PARTIALLY_VERIFIED"

    prime = fixture.get("prime_control", {})
    speedup = prime.get("speedup_x", 0)
    results["SGT_SPEEDUP"] = (
        prime.get("modulus") == 97
        and speedup >= 2.0
        and prime.get("standard_epoch", 0) > prime.get("spherical_epoch", 0)
    )

    norm = fixture.get("residual_norm", {})
    results["SGT_NORM"] = norm.get("spherical_constant") == 1.0 and norm.get("s2_compliant") is True

    uniform = fixture.get("uniform_attention", {})
    results["SGT_UNIFORM"] = uniform.get("confirmed") is True and uniform.get("accuracy", 0) >= 0.99

    m9 = fixture.get("m9_composite_target", {})
    results["SGT_M9"] = (
        m9.get("modulus") == 9
        and m9.get("no_model_grokked") is True
        and m9.get("interpretation") == "not_applicable"
    )

    honesty = fixture.get("honesty_gate", {})
    results["SGT_HONEST"] = (
        honesty.get("s5_local_tested") is False
        and honesty.get("s5_status") == "UNTESTED_LOCAL"
        and honesty.get("local_speedup_weaker_than_yildirim") is True
    )

    src = fixture.get("source_attribution", "")
    results["SGT_SRC"] = "Yildirim" in src and "2603.05228" in src

    witnesses = fixture.get("witnesses", [])
    kinds = {w.get("kind") for w in witnesses if isinstance(w, dict)}
    results["SGT_WITNESS"] = {"speedup", "norm", "uniform_attention", "m9"}.issubset(kinds)

    fail_ledger = fixture.get("fail_ledger")
    results["SGT_F"] = isinstance(fail_ledger, list) and len(fail_ledger) >= 1
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
    print("Usage: python qa_spherical_grokking_theorem_nt_cert_validate.py [--self-test | fixture.json]")
    sys.exit(1)
