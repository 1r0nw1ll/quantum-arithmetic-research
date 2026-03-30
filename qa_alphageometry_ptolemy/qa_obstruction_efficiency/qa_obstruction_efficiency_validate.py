"""
QA Obstruction Efficiency Certificate Validator — Family [114]
Schema: QA_OBSTRUCTION_EFFICIENCY_CERT.v1

Checks (IH = inheritance, EF = efficiency):
  IH1  inherits_from == 'QA_CORE_SPEC.v1'
  IH2  spec_scope == 'family_extension'
  IH3  gate_policy_respected ⊇ [0,1,2,3,4,5]
  EF1  planner_ref.schema_version == 'QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1'
  EF2  modulus == prime_p ^ k
  EF3  prime_p is inert in Z[phi] (Legendre(5, p) == -1)
  EF4  target_arithmetic_class in {0 .. modulus-1}
  EF5  obstruction_verdict matches v_p(target)
  EF6  efficiency_claim.saved_nodes == baseline_trace.nodes_expanded - aware_trace.nodes_expanded
  EF7  efficiency_claim.pruning_ratio == saved_nodes / baseline (0.0 if baseline == 0)
  EF8  OBSTRUCTION_PRESENT → aware_trace.pruned_before_search==true, nodes_expanded==0
       OBSTRUCTION_ABSENT  → aware_trace.pruned_before_search==false
  EF9  false_pruning==false for PASS (OBSTRUCTION_ABSENT target never pruned)

Validator recomputes all derived efficiency metrics from raw traces — does not trust cert values.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

def _legendre(a: int, p: int) -> int:
    """Legendre symbol (a/p) for odd prime p."""
    val = pow(a, (p - 1) // 2, p)
    return -1 if val == p - 1 else val


def _is_inert(p: int) -> bool:
    """True iff p is inert in Z[phi] (x^2+x-1 irreducible mod p, i.e. Legendre(5,p)==-1)."""
    if p < 2 or p == 2 or p == 5:
        return False
    return _legendre(5, p) == -1


def _v_p(r: int, p: int) -> int:
    """p-adic valuation of r: largest k s.t. p^k | r; v_p(0) = infinity (returned as -1)."""
    if r == 0:
        return -1  # infinity sentinel
    v = 0
    while r % p == 0:
        v += 1
        r //= p
    return v


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

REQUIRED_GATES = {0, 1, 2, 3, 4, 5}

KNOWN_FAIL_TYPES = {
    "INVALID_KERNEL_REFERENCE",
    "SPEC_SCOPE_MISMATCH",
    "GATE_POLICY_INCOMPATIBLE",
    "PLANNER_REF_MISMATCH",
    "MODULUS_MISMATCH",
    "PRIME_NOT_INERT",
    "TARGET_OUT_OF_RANGE",
    "OBSTRUCTION_VERDICT_WRONG",
    "EFFICIENCY_CLAIM_INCORRECT",
    "FALSE_PRUNING_EFFICIENCY",
    "AWARE_TRACE_MISMATCH",
}


def validate(cert: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a QA_OBSTRUCTION_EFFICIENCY_CERT.v1 certificate.

    Returns (ok, errors) where ok==True iff structural validation passes
    (PASS cert with zero detected failures, or FAIL cert with matching fail_ledger).
    """
    errors: list[str] = []
    detected: set[str] = set()

    # -----------------------------------------------------------------------
    # Schema version guard
    # -----------------------------------------------------------------------
    if cert.get("schema_version") != "QA_OBSTRUCTION_EFFICIENCY_CERT.v1":
        errors.append(
            f"schema_version mismatch: expected 'QA_OBSTRUCTION_EFFICIENCY_CERT.v1', "
            f"got {cert.get('schema_version')!r}"
        )
        return False, errors

    # -----------------------------------------------------------------------
    # IH1 – inherits_from
    # -----------------------------------------------------------------------
    if cert.get("inherits_from") != "QA_CORE_SPEC.v1":
        detected.add("INVALID_KERNEL_REFERENCE")

    # IH2 – spec_scope
    if cert.get("spec_scope") != "family_extension":
        detected.add("SPEC_SCOPE_MISMATCH")

    # IH3 – gate_policy_respected
    gates = set(cert.get("core_kernel_compatibility", {}).get("gate_policy_respected", []))
    if not REQUIRED_GATES.issubset(gates):
        detected.add("GATE_POLICY_INCOMPATIBLE")

    # -----------------------------------------------------------------------
    # EF1 – planner_ref
    # -----------------------------------------------------------------------
    planner_ref = cert.get("planner_ref", {})
    if planner_ref.get("schema_version") != "QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1":
        detected.add("PLANNER_REF_MISMATCH")

    # -----------------------------------------------------------------------
    # EF2 – modulus == p^k
    # -----------------------------------------------------------------------
    ap = cert.get("arithmetic_params", {})
    prime_p = ap.get("prime_p")
    k = ap.get("k")
    modulus = ap.get("modulus")

    if (
        prime_p is None or k is None or modulus is None
        or modulus != prime_p ** k
    ):
        detected.add("MODULUS_MISMATCH")

    # -----------------------------------------------------------------------
    # EF3 – prime_p inert
    # -----------------------------------------------------------------------
    if prime_p is not None and not _is_inert(prime_p):
        detected.add("PRIME_NOT_INERT")

    # -----------------------------------------------------------------------
    # EF4 – target in range
    # -----------------------------------------------------------------------
    target = cert.get("target_arithmetic_class")
    if (
        target is None
        or modulus is None
        or not (0 <= target < modulus)
    ):
        detected.add("TARGET_OUT_OF_RANGE")

    # -----------------------------------------------------------------------
    # EF5 – obstruction_verdict vs v_p(target)
    # -----------------------------------------------------------------------
    declared_verdict = cert.get("obstruction_verdict")
    if prime_p is not None and target is not None and "MODULUS_MISMATCH" not in detected:
        vp = _v_p(target, prime_p)
        # vp == -1 means r==0 → v_p(0)=∞ → never equals exactly 1
        expected_verdict = "OBSTRUCTION_PRESENT" if vp == 1 else "OBSTRUCTION_ABSENT"
        if declared_verdict != expected_verdict:
            detected.add("OBSTRUCTION_VERDICT_WRONG")

    # -----------------------------------------------------------------------
    # EF6 / EF7 / EF8 / EF9 – efficiency metrics
    # -----------------------------------------------------------------------
    baseline = cert.get("baseline_trace", {})
    aware = cert.get("aware_trace", {})
    claim = cert.get("efficiency_claim", {})

    baseline_nodes: int | None = baseline.get("nodes_expanded")
    aware_nodes: int | None = aware.get("nodes_expanded")
    pruned_before: bool | None = aware.get("pruned_before_search")
    saved_claimed: int | None = claim.get("saved_nodes")
    ratio_claimed: float | None = claim.get("pruning_ratio")
    false_pruning_claimed: bool | None = claim.get("false_pruning")

    if baseline_nodes is not None and aware_nodes is not None:
        # EF6: recompute saved_nodes
        saved_computed = baseline_nodes - aware_nodes
        if saved_claimed != saved_computed:
            detected.add("EFFICIENCY_CLAIM_INCORRECT")

        # EF7: recompute pruning_ratio
        if baseline_nodes > 0:
            ratio_computed = saved_computed / baseline_nodes
        else:
            ratio_computed = 0.0
        # compare with tolerance for floating point
        if ratio_claimed is None or abs(ratio_claimed - ratio_computed) > 1e-9:
            detected.add("EFFICIENCY_CLAIM_INCORRECT")

    # EF8: aware trace consistency with verdict
    if declared_verdict == "OBSTRUCTION_PRESENT":
        if pruned_before is not True or aware_nodes != 0:
            detected.add("AWARE_TRACE_MISMATCH")
    elif declared_verdict == "OBSTRUCTION_ABSENT":
        if pruned_before is True:
            detected.add("AWARE_TRACE_MISMATCH")

    # EF9: false_pruning only allowed when target is genuinely forbidden
    if declared_verdict == "OBSTRUCTION_ABSENT" and pruned_before is True:
        detected.add("FALSE_PRUNING_EFFICIENCY")
    if false_pruning_claimed is True and declared_verdict == "OBSTRUCTION_ABSENT":
        detected.add("FALSE_PRUNING_EFFICIENCY")

    # -----------------------------------------------------------------------
    # Reconcile detected failures against declared result / fail_ledger
    # -----------------------------------------------------------------------
    declared_result = cert.get("result")
    fail_ledger_types = {e["fail_type"] for e in cert.get("fail_ledger", [])}

    if detected:
        # Should be a FAIL cert with matching fail_ledger
        if declared_result != "FAIL":
            errors.append(
                f"Detected failures {sorted(detected)} but cert declares result=PASS."
            )
        else:
            missing = detected - fail_ledger_types
            extra = fail_ledger_types - detected
            if missing:
                errors.append(f"fail_ledger missing detected failure types: {sorted(missing)}")
            if extra:
                errors.append(f"fail_ledger contains undeclared failure types: {sorted(extra)}")
    else:
        if declared_result != "PASS":
            errors.append(
                "No failures detected but cert declares result=FAIL."
            )

    ok = len(errors) == 0
    return ok, errors


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test(fixtures_dir: Path) -> dict:
    expected = {
        "efficiency_pass_forbidden_class_6.json": True,
        "efficiency_pass_valid_class_4.json":      True,
        "efficiency_fail_false_pruning.json":       True,
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
        ok, errs = validate(cert)
        passed = ok == expect_ok
        if not passed:
            all_ok = False
        results.append({
            "fixture": fname,
            "ok": passed,
            "label": "PASS" if passed else "FAIL",
            "errors": errs,
        })

    return {"ok": all_ok, "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate QA_OBSTRUCTION_EFFICIENCY_CERT.v1 certificates"
    )
    parser.add_argument("cert_files", nargs="*", help="JSON cert files to validate")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test against fixtures/ directory",
    )
    args = parser.parse_args()

    if args.self_test:
        fixtures_dir = Path(__file__).parent / "fixtures"
        result = _run_self_test(fixtures_dir)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if not args.cert_files:
        parser.print_help()
        sys.exit(0)

    all_ok = True
    for path in args.cert_files:
        with open(path) as f:
            cert = json.load(f)
        ok, errs = validate(cert)
        if ok:
            print(f"PASS  {path}")
        else:
            all_ok = False
            print(f"FAIL  {path}")
            for e in errs:
                print(f"      {e}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
