"""
QA Obstruction Stack Certificate Validator — Family [115]
Schema: QA_OBSTRUCTION_STACK_CERT.v1

Synthesis spine for the [111]–[114] obstruction chain. Recomputes every layer
independently from first principles — no layer trusts the adjacent layer's
claimed outputs.

Checks:
  IH1   inherits_from == 'QA_CORE_SPEC.v1'
  IH2   spec_scope == 'family_extension'
  IH3   gate_policy_respected ⊇ [0,1,2,3,4,5]

  OS1   family_refs.arithmetic_ref == 'QA_AREA_QUANTIZATION_PK_CERT.v1'
  OS2   family_refs.control_ref == 'QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1'
  OS3   family_refs.planner_ref == 'QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1'
  OS4   family_refs.efficiency_ref == 'QA_OBSTRUCTION_EFFICIENCY_CERT.v1'

  OS5   modulus == prime_p ^ k
  OS6   prime_p is inert in Z[phi]: Legendre(5, p) == -1
  OS7   target_arithmetic_class in {0 .. modulus-1}

  OS8   arithmetic_layer.obstruction_verdict matches independently recomputed v_p(r)

  OS9   OBSTRUCTION_PRESENT →
          control_layer.claimed_reachable == false
          control_layer.control_verdict == 'UNREACHABLE'

  OS10  OBSTRUCTION_PRESENT →
          planner_layer.pruned_before_search == true
          planner_layer.nodes_expanded == 0
        OBSTRUCTION_ABSENT →
          planner_layer.pruned_before_search == false

  OS11  efficiency_layer: recompute saved_nodes and pruning_ratio from baseline/aware
        OBSTRUCTION_PRESENT → aware_nodes == 0, pruning_ratio == 1.0
        (saved_nodes == baseline_nodes, ratio == 1.0)

  OS12  stack_conclusion.full_chain_holds is consistent with all four layers passing
        (true iff OS8–OS11 all green)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

def _legendre(a: int, p: int) -> int:
    val = pow(a, (p - 1) // 2, p)
    return -1 if val == p - 1 else val


def _is_inert(p: int) -> bool:
    if p < 2 or p == 2 or p == 5:
        return False
    return _legendre(5, p) == -1


def _v_p(r: int, p: int) -> int:
    """p-adic valuation; returns -1 for r==0 (infinity sentinel)."""
    if r == 0:
        return -1
    v = 0
    while r % p == 0:
        v += 1
        r //= p
    return v


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

REQUIRED_GATES = {0, 1, 2, 3, 4, 5}


def validate(cert: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a QA_OBSTRUCTION_STACK_CERT.v1 certificate.

    Returns (ok, errors).
    ok==True: structural validation passes (PASS cert with zero failures,
              or FAIL cert with fail_ledger matching detected set).
    """
    errors: list[str] = []
    detected: set[str] = set()

    # Schema guard
    if cert.get("schema_version") != "QA_OBSTRUCTION_STACK_CERT.v1":
        errors.append(
            f"schema_version mismatch: expected 'QA_OBSTRUCTION_STACK_CERT.v1', "
            f"got {cert.get('schema_version')!r}"
        )
        return False, errors

    # -------------------------------------------------------------------
    # IH1-IH3 — kernel inheritance
    # -------------------------------------------------------------------
    if cert.get("inherits_from") != "QA_CORE_SPEC.v1":
        detected.add("INVALID_KERNEL_REFERENCE")

    if cert.get("spec_scope") != "family_extension":
        detected.add("SPEC_SCOPE_MISMATCH")

    gates = set(cert.get("core_kernel_compatibility", {}).get("gate_policy_respected", []))
    if not REQUIRED_GATES.issubset(gates):
        detected.add("GATE_POLICY_INCOMPATIBLE")

    # -------------------------------------------------------------------
    # OS1-OS4 — family refs
    # -------------------------------------------------------------------
    refs = cert.get("family_refs", {})
    if refs.get("arithmetic_ref", {}).get("schema_version") != "QA_AREA_QUANTIZATION_PK_CERT.v1":
        detected.add("ARITHMETIC_REF_MISMATCH")
    if refs.get("control_ref", {}).get("schema_version") != "QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1":
        detected.add("CONTROL_REF_MISMATCH")
    if refs.get("planner_ref", {}).get("schema_version") != "QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1":
        detected.add("PLANNER_REF_MISMATCH")
    if refs.get("efficiency_ref", {}).get("schema_version") != "QA_OBSTRUCTION_EFFICIENCY_CERT.v1":
        detected.add("EFFICIENCY_REF_MISMATCH")

    # -------------------------------------------------------------------
    # OS5 — modulus
    # -------------------------------------------------------------------
    ap = cert.get("arithmetic_params", {})
    prime_p = ap.get("prime_p")
    k = ap.get("k")
    modulus = ap.get("modulus")

    if prime_p is None or k is None or modulus is None or modulus != prime_p ** k:
        detected.add("MODULUS_MISMATCH")

    # -------------------------------------------------------------------
    # OS6 — inert prime
    # -------------------------------------------------------------------
    if prime_p is not None and not _is_inert(prime_p):
        detected.add("PRIME_NOT_INERT")

    # -------------------------------------------------------------------
    # OS7 — target in range
    # -------------------------------------------------------------------
    target = cert.get("target_arithmetic_class")
    if target is None or modulus is None or not (0 <= target < modulus):
        detected.add("TARGET_OUT_OF_RANGE")

    # -------------------------------------------------------------------
    # OS8 — arithmetic layer: recompute v_p(r)
    # -------------------------------------------------------------------
    declared_verdict = cert.get("arithmetic_layer", {}).get("obstruction_verdict")
    layer_failures: set[str] = set()   # track which layer checks pass (for OS12)

    if prime_p is not None and target is not None and "MODULUS_MISMATCH" not in detected:
        vp = _v_p(target, prime_p)
        expected_verdict = "OBSTRUCTION_PRESENT" if vp == 1 else "OBSTRUCTION_ABSENT"
        if declared_verdict != expected_verdict:
            detected.add("OBSTRUCTION_VERDICT_WRONG")
            layer_failures.add("arithmetic")

    # -------------------------------------------------------------------
    # OS9 — control layer consistency
    # -------------------------------------------------------------------
    ctl = cert.get("control_layer", {})
    claimed_reachable = ctl.get("claimed_reachable")
    control_verdict = ctl.get("control_verdict")

    if declared_verdict == "OBSTRUCTION_PRESENT":
        if claimed_reachable is not False or control_verdict != "UNREACHABLE":
            detected.add("STACK_INCONSISTENCY")
            layer_failures.add("control")

    # -------------------------------------------------------------------
    # OS10 — planner layer consistency
    # -------------------------------------------------------------------
    pl = cert.get("planner_layer", {})
    pruned = pl.get("pruned_before_search")
    nodes = pl.get("nodes_expanded")

    if declared_verdict == "OBSTRUCTION_PRESENT":
        if pruned is not True or nodes != 0:
            detected.add("PRUNING_CONCLUSION_MISMATCH")
            layer_failures.add("planner")
    elif declared_verdict == "OBSTRUCTION_ABSENT":
        if pruned is True:
            detected.add("PRUNING_CONCLUSION_MISMATCH")
            layer_failures.add("planner")

    # -------------------------------------------------------------------
    # OS11 — efficiency layer consistency
    # -------------------------------------------------------------------
    ef = cert.get("efficiency_layer", {})
    baseline_nodes = ef.get("baseline_nodes")
    aware_nodes = ef.get("aware_nodes")
    saved_claimed = ef.get("saved_nodes")
    ratio_claimed = ef.get("pruning_ratio")

    if baseline_nodes is not None and aware_nodes is not None:
        # Recompute
        saved_computed = baseline_nodes - aware_nodes
        ratio_computed = (saved_computed / baseline_nodes) if baseline_nodes > 0 else 0.0

        if saved_claimed != saved_computed or (
            ratio_claimed is None or abs(ratio_claimed - ratio_computed) > 1e-9
        ):
            detected.add("EFFICIENCY_CONCLUSION_MISMATCH")
            layer_failures.add("efficiency")

        # OBSTRUCTION_PRESENT: aware_nodes must be 0
        if declared_verdict == "OBSTRUCTION_PRESENT" and aware_nodes != 0:
            detected.add("EFFICIENCY_CONCLUSION_MISMATCH")
            layer_failures.add("efficiency")

    # -------------------------------------------------------------------
    # OS12 — stack_conclusion.full_chain_holds
    # -------------------------------------------------------------------
    sc = cert.get("stack_conclusion", {})
    full_chain = sc.get("full_chain_holds")
    # full_chain_holds should be True iff no layer failures
    chain_should_hold = len(layer_failures) == 0
    if full_chain != chain_should_hold:
        detected.add("STACK_INCONSISTENCY")

    # -------------------------------------------------------------------
    # Reconcile
    # -------------------------------------------------------------------
    declared_result = cert.get("result")
    fail_ledger_types = {e["fail_type"] for e in cert.get("fail_ledger", [])}

    if detected:
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
            errors.append("No failures detected but cert declares result=FAIL.")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test(fixtures_dir: Path) -> dict:
    expected = {
        "stack_pass_forbidden_class_6.json":   True,
        "stack_fail_inconsistent_class_6.json": True,
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
        description="Validate QA_OBSTRUCTION_STACK_CERT.v1 certificates"
    )
    parser.add_argument("cert_files", nargs="*", help="JSON cert files to validate")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test against fixtures/ directory")
    args = parser.parse_args()

    if args.self_test:
        result = _run_self_test(Path(__file__).parent / "fixtures")
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
