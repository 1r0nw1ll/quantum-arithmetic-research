#!/usr/bin/env python3
"""
QA Obstruction-Aware Planner Certificate Validator
Family [113] — family_extension of QA_CORE_SPEC.v1 [107]

Certifies that a planner correctly applies the arithmetic obstruction test
from [112] before expanding its search frontier:
  - OBSTRUCTION_PRESENT => pruned_before_search=True, nodes_expanded=0
  - OBSTRUCTION_ABSENT  => pruned_before_search=False, search may proceed

Checks: IH1-IH3 (kernel), BR1-BR5 (bridge/arithmetic), PA1-PA3 (planner behavior)
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

SCHEMA_VERSION = "QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1"
CERT_TYPE = "obstruction_aware_planner"
KERNEL_VERSION = "QA_CORE_SPEC.v1"
BRIDGE_FAMILY = "QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1"
REQUIRED_SPEC_SCOPE = "family_extension"
REQUIRED_GATES = list(range(6))


class _Out:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str): self.errors.append(msg)
    def warn(self, msg: str): self.warnings.append(msg)


def _legendre(a: int, p: int) -> int:
    val = pow(a, (p - 1) // 2, p)
    return -1 if val == p - 1 else val


def _is_inert(p: int) -> bool:
    if p == 2 or p == 5:
        return False
    return _legendre(5, p) == -1


def _vp(r: int, p: int) -> float:
    if r == 0:
        return float('inf')
    v = 0
    while r % p == 0:
        r //= p
        v += 1
    return v


def validate_planner_cert(cert: dict) -> dict:
    out = _Out()

    if cert.get("schema_version") != SCHEMA_VERSION:
        out.fail(f"schema_version must be '{SCHEMA_VERSION}'")
        return _build_result(cert, out, set())
    if cert.get("cert_type") != CERT_TYPE:
        out.fail(f"cert_type must be '{CERT_TYPE}'")
        return _build_result(cert, out, set())

    declared_result = cert.get("result", "")
    if declared_result not in ("PASS", "FAIL"):
        out.fail("result must be PASS or FAIL")
        return _build_result(cert, out, set())

    detected: set[str] = set()

    # IH1-IH3: kernel inheritance
    if cert.get("inherits_from") != KERNEL_VERSION:
        detected.add("INVALID_KERNEL_REFERENCE")
    if cert.get("spec_scope") != REQUIRED_SPEC_SCOPE:
        detected.add("SPEC_SCOPE_MISMATCH")
    compat = cert.get("core_kernel_compatibility", {})
    gates = compat.get("gate_policy_respected", [])
    if not all(g in gates for g in REQUIRED_GATES):
        detected.add("GATE_POLICY_INCOMPATIBLE")

    # BR1: obstruction_ref
    obs_ref = cert.get("obstruction_ref", {})
    if obs_ref.get("schema_version") != BRIDGE_FAMILY:
        detected.add("OBSTRUCTION_REF_MISMATCH")

    # BR2-BR4: arithmetic params
    params = cert.get("arithmetic_params", {})
    prime_p = params.get("prime_p")
    k = params.get("k")
    modulus = params.get("modulus")

    if prime_p and k and modulus:
        if modulus != prime_p ** k:
            detected.add("MODULUS_MISMATCH")
    else:
        detected.add("MODULUS_MISMATCH")

    if prime_p and not _is_inert(prime_p):
        detected.add("PRIME_NOT_INERT")

    target = cert.get("target_arithmetic_class")
    if target is None or not isinstance(target, int) or modulus is None:
        detected.add("TARGET_OUT_OF_RANGE")
    elif not (0 <= target < modulus):
        detected.add("TARGET_OUT_OF_RANGE")

    # BR5 + PA checks
    if "TARGET_OUT_OF_RANGE" not in detected and "PRIME_NOT_INERT" not in detected:
        vp_val = _vp(target, prime_p)
        expected_verdict = "OBSTRUCTION_PRESENT" if vp_val == 1 else "OBSTRUCTION_ABSENT"
        declared_verdict = cert.get("obstruction_verdict", "")

        # BR5: verdict matches v_p
        if declared_verdict != expected_verdict:
            detected.add("OBSTRUCTION_VERDICT_WRONG")

        behavior = cert.get("planner_behavior", {})
        pruned = behavior.get("pruned_before_search", None)
        nodes = behavior.get("nodes_expanded", -1)
        plan_found = behavior.get("plan_found", "MISSING")

        if declared_verdict == "OBSTRUCTION_PRESENT":
            # PA1: must prune before search
            if pruned is not True:
                detected.add("OBSTRUCTION_NOT_APPLIED")
            # PA2: must have expanded 0 nodes
            elif nodes != 0:
                detected.add("OBSTRUCTION_NOT_APPLIED")
        elif declared_verdict == "OBSTRUCTION_ABSENT":
            # PA: must NOT have pruned a valid target
            if pruned is True:
                detected.add("PRUNE_DECISION_INCONSISTENT")

    return _build_result(cert, out, detected)


def _build_result(cert: dict, out: _Out, detected: set[str]) -> dict:
    declared_result = cert.get("result", "")
    declared_fail_types = {
        e["fail_type"] for e in cert.get("fail_ledger", []) if "fail_type" in e
    }

    if declared_result == "PASS":
        if detected:
            out.fail(f"cert declares PASS but detected: {sorted(detected)}")
        elif declared_fail_types:
            out.fail(f"cert declares PASS but fail_ledger non-empty: {sorted(declared_fail_types)}")
    elif declared_result == "FAIL":
        if not declared_fail_types:
            out.fail("cert declares FAIL but fail_ledger empty")
        elif detected != declared_fail_types:
            out.fail(
                f"declared fail_ledger {sorted(declared_fail_types)} != "
                f"detected {sorted(detected)}"
            )

    ok = len(out.errors) == 0
    label = "PASS" if ok else "FAIL"
    if ok and out.warnings:
        label = "PASS_WITH_WARNINGS"

    return {
        "ok": ok,
        "label": label,
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "errors": out.errors,
        "warnings": out.warnings,
        "detected_fails": sorted(detected),
    }


def _self_test() -> dict:
    fixtures_dir = HERE / "fixtures"
    expected = {
        "planner_pass_pruned_class_3.json":         True,
        "planner_pass_search_class_4.json":          True,
        "planner_fail_obstruction_not_applied.json": True,
    }

    results = []
    all_ok = True

    for fname, expect_ok in expected.items():
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        with open(fpath) as f:
            cert = json.load(f)
        r = validate_planner_cert(cert)
        passed = r["ok"] == expect_ok
        if not passed:
            all_ok = False
        results.append({
            "fixture": fname,
            "ok": passed,
            "label": r["label"],
            "errors": r["errors"],
            "warnings": r["warnings"],
        })

    return {"ok": all_ok, "results": results}


def main():
    if "--self-test" in sys.argv:
        result = _self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        fpath = pathlib.Path(sys.argv[idx + 1])
        with open(fpath) as f:
            cert = json.load(f)
        r = validate_planner_cert(cert)
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)

    print("Usage: qa_obstruction_aware_planner_validate.py [--self-test] [--file <path>]")
    sys.exit(1)


if __name__ == "__main__":
    main()
