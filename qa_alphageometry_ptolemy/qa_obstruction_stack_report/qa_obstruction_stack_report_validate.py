"""
QA Obstruction Stack Report Validator — Family [116]
Schema: QA_OBSTRUCTION_STACK_REPORT.v1

Reader-facing report packaging [115] for external audiences.
Validator recomputes the summary table from arithmetic params and checks
that the report faithfully represents the machine-verified results.

Checks:
  IH1   inherits_from == 'QA_CORE_SPEC.v1'
  IH2   spec_scope == 'family_extension'
  IH3   gate_policy_respected ⊇ [0,1,2,3,4,5]

  RP1   stack_ref.schema_version == 'QA_OBSTRUCTION_STACK_CERT.v1'
  RP2   modulus == prime_p ^ k
  RP3   prime_p is inert: Legendre(5, p) == -1
  RP4   theorem_statement is non-empty (len >= 20)

  RP5   layer_summaries covers all five families by schema_version:
          QA_AREA_QUANTIZATION_PK_CERT.v1
          QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1
          QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1
          QA_OBSTRUCTION_EFFICIENCY_CERT.v1
          QA_OBSTRUCTION_STACK_CERT.v1

  RP6   for each summary_table row, independently recompute:
          v_p_r, forbidden, reachable, pruned, saved_nodes, pruning_ratio
        and check that claimed values match

  RP7   all forbidden rows must have pruned=true, aware_nodes=0, pruning_ratio=1.0

  RP8   canonical_pass_witness.target_r appears in summary_table with forbidden=true

  RP9   canonical_fail_witness.fail_types is non-empty
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
    """p-adic valuation; -1 sentinel for r==0."""
    if r == 0:
        return -1
    v = 0
    while r % p == 0:
        v += 1
        r //= p
    return v


# ---------------------------------------------------------------------------
# Required layer families
# ---------------------------------------------------------------------------

REQUIRED_LAYER_SCHEMAS = {
    "QA_AREA_QUANTIZATION_PK_CERT.v1",
    "QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1",
    "QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1",
    "QA_OBSTRUCTION_EFFICIENCY_CERT.v1",
    "QA_OBSTRUCTION_STACK_CERT.v1",
}

REQUIRED_GATES = {0, 1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

def validate(cert: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    detected: set[str] = set()

    if cert.get("schema_version") != "QA_OBSTRUCTION_STACK_REPORT.v1":
        errors.append(
            f"schema_version mismatch: expected 'QA_OBSTRUCTION_STACK_REPORT.v1', "
            f"got {cert.get('schema_version')!r}"
        )
        return False, errors

    # -----------------------------------------------------------------------
    # IH1–IH3
    # -----------------------------------------------------------------------
    if cert.get("inherits_from") != "QA_CORE_SPEC.v1":
        detected.add("INVALID_KERNEL_REFERENCE")
    if cert.get("spec_scope") != "family_extension":
        detected.add("SPEC_SCOPE_MISMATCH")
    gates = set(cert.get("core_kernel_compatibility", {}).get("gate_policy_respected", []))
    if not REQUIRED_GATES.issubset(gates):
        detected.add("GATE_POLICY_INCOMPATIBLE")

    # -----------------------------------------------------------------------
    # RP1 — stack_ref
    # -----------------------------------------------------------------------
    if cert.get("stack_ref", {}).get("schema_version") != "QA_OBSTRUCTION_STACK_CERT.v1":
        detected.add("STACK_REF_MISMATCH")

    # -----------------------------------------------------------------------
    # RP2 — modulus
    # -----------------------------------------------------------------------
    ap = cert.get("arithmetic_params", {})
    prime_p = ap.get("prime_p")
    k = ap.get("k")
    modulus = ap.get("modulus")

    if prime_p is None or k is None or modulus is None or modulus != prime_p ** k:
        detected.add("MODULUS_MISMATCH")

    # -----------------------------------------------------------------------
    # RP3 — inert prime
    # -----------------------------------------------------------------------
    if prime_p is not None and not _is_inert(prime_p):
        detected.add("PRIME_NOT_INERT")

    # -----------------------------------------------------------------------
    # RP4 — theorem_statement
    # -----------------------------------------------------------------------
    stmt = cert.get("theorem_statement", "")
    if not isinstance(stmt, str) or len(stmt) < 20:
        detected.add("THEOREM_STATEMENT_MISSING")

    # -----------------------------------------------------------------------
    # RP5 — layer_summaries completeness
    # -----------------------------------------------------------------------
    summaries = cert.get("layer_summaries", [])
    found_schemas = {s.get("schema_version") for s in summaries}
    if not REQUIRED_LAYER_SCHEMAS.issubset(found_schemas):
        detected.add("LAYER_SUMMARY_INCOMPLETE")

    # -----------------------------------------------------------------------
    # RP6 + RP7 — summary_table faithfulness
    # -----------------------------------------------------------------------
    table = cert.get("summary_table", [])
    table_mismatches: list[str] = []

    for row in table:
        r = row.get("target_r")
        if r is None or prime_p is None or "MODULUS_MISMATCH" in detected:
            continue

        # Recompute
        vp = _v_p(r, prime_p)
        forbidden_computed = (vp == 1)
        reachable_computed = not forbidden_computed
        pruned_computed = forbidden_computed  # pruned iff forbidden

        # Check v_p_r
        if row.get("v_p_r") != vp and not (vp == -1 and row.get("v_p_r", 0) < 0):
            table_mismatches.append(f"r={r}: v_p_r claimed={row.get('v_p_r')}, computed={vp}")

        # Check forbidden / reachable / pruned
        if row.get("forbidden") != forbidden_computed:
            table_mismatches.append(
                f"r={r}: forbidden claimed={row.get('forbidden')}, computed={forbidden_computed}"
            )
        if row.get("reachable") != reachable_computed:
            table_mismatches.append(
                f"r={r}: reachable claimed={row.get('reachable')}, computed={reachable_computed}"
            )
        if row.get("pruned") != pruned_computed:
            table_mismatches.append(
                f"r={r}: pruned claimed={row.get('pruned')}, computed={pruned_computed}"
            )

        # RP7: for forbidden rows, aware_nodes must be 0 → saved = baseline, ratio = 1.0
        if forbidden_computed:
            if row.get("aware_nodes") != 0:
                table_mismatches.append(
                    f"r={r}: forbidden target but aware_nodes={row.get('aware_nodes')}, must be 0"
                )
            baseline = row.get("baseline_nodes", 0)
            if row.get("saved_nodes") != baseline:
                table_mismatches.append(
                    f"r={r}: saved_nodes claimed={row.get('saved_nodes')}, must be baseline={baseline}"
                )
            if row.get("pruning_ratio") is None or abs(row.get("pruning_ratio", 0) - 1.0) > 1e-9:
                table_mismatches.append(
                    f"r={r}: pruning_ratio claimed={row.get('pruning_ratio')}, must be 1.0"
                )

        # Recompute saved_nodes / pruning_ratio regardless
        baseline_n = row.get("baseline_nodes")
        aware_n = row.get("aware_nodes")
        if baseline_n is not None and aware_n is not None:
            saved_computed = baseline_n - aware_n
            ratio_computed = (saved_computed / baseline_n) if baseline_n > 0 else 0.0
            if row.get("saved_nodes") != saved_computed:
                table_mismatches.append(
                    f"r={r}: saved_nodes claimed={row.get('saved_nodes')}, computed={saved_computed}"
                )
            if row.get("pruning_ratio") is None or abs(row.get("pruning_ratio") - ratio_computed) > 1e-9:
                table_mismatches.append(
                    f"r={r}: pruning_ratio claimed={row.get('pruning_ratio')}, computed={ratio_computed:.4f}"
                )

    if table_mismatches:
        detected.add("SUMMARY_TABLE_MISMATCH")

    # -----------------------------------------------------------------------
    # RP8 — canonical_pass_witness target in table as forbidden
    # -----------------------------------------------------------------------
    pw = cert.get("canonical_pass_witness", {})
    pw_target = pw.get("target_r")
    if pw_target is not None:
        table_map = {row.get("target_r"): row for row in table}
        pw_row = table_map.get(pw_target)
        if pw_row is None or not pw_row.get("forbidden", False):
            detected.add("WITNESS_MISMATCH")

    # -----------------------------------------------------------------------
    # RP9 — canonical_fail_witness has ≥1 fail_type
    # -----------------------------------------------------------------------
    fw = cert.get("canonical_fail_witness", {})
    if not fw.get("fail_types"):
        detected.add("WITNESS_MISMATCH")

    # -----------------------------------------------------------------------
    # Reconcile
    # -----------------------------------------------------------------------
    declared_result = cert.get("result")
    fail_ledger_types = {e["fail_type"] for e in cert.get("fail_ledger", [])}

    if detected:
        if declared_result != "FAIL":
            errors.append(
                f"Detected failures {sorted(detected)} but cert declares result=PASS."
            )
            if table_mismatches:
                errors.append("  Summary table mismatches:\n" + "\n".join(f"    {m}" for m in table_mismatches))
        else:
            missing = detected - fail_ledger_types
            extra = fail_ledger_types - detected
            if missing:
                errors.append(f"fail_ledger missing detected failure types: {sorted(missing)}")
                if table_mismatches:
                    errors.append("  Summary table mismatches:\n" + "\n".join(f"    {m}" for m in table_mismatches))
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
        "report_pass_canonical_r6.json":         True,
        "report_fail_inconsistent_summary.json":  True,
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
        description="Validate QA_OBSTRUCTION_STACK_REPORT.v1 certificates"
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
