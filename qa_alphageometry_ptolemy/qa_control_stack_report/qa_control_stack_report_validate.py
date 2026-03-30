"""
QA Control Stack Report Validator — Family [118]
Schema: QA_CONTROL_STACK_REPORT.v1

Reader-facing report packaging [117] for external audiences.
Validator recomputes cross-domain consistency from the comparison_table
and checks that the report faithfully represents [117].

Checks:
  IH1   inherits_from == 'QA_CORE_SPEC.v1'
  IH2   spec_scope == 'family_extension'
  IH3   gate_policy_respected ⊇ [0,1,2,3,4,5]
  CR1   control_stack_ref.schema_version == 'QA_CONTROL_STACK_CERT.v1'
  CR2   theorem_statement non-empty (len >= 20)
  CR3   domain_summaries covers required schemas: [106], [105], [110]
  CR4   all comparison_table rows have consistent orbit_path
  CR5   all comparison_table rows have equal path_length_k
  CR6   canonical_pass_witness.result == 'cross_domain_equivalence_holds'
  CR7   canonical_fail_witness.fail_types non-empty
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_GATES = {0, 1, 2, 3, 4, 5}

REQUIRED_DOMAIN_SCHEMAS = {
    "QA_PLAN_CONTROL_COMPILER_CERT.v1",
    "QA_CYMATIC_CONTROL_CERT.v1",
    "QA_SEISMIC_CONTROL_CERT.v1",
}


def validate(cert: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    detected: set[str] = set()

    if cert.get("schema_version") != "QA_CONTROL_STACK_REPORT.v1":
        errors.append(
            f"schema_version mismatch: expected 'QA_CONTROL_STACK_REPORT.v1', "
            f"got {cert.get('schema_version')!r}"
        )
        return False, errors

    # IH1–IH3
    if cert.get("inherits_from") != "QA_CORE_SPEC.v1":
        detected.add("INVALID_KERNEL_REFERENCE")
    if cert.get("spec_scope") != "family_extension":
        detected.add("SPEC_SCOPE_MISMATCH")
    gates = set(cert.get("core_kernel_compatibility", {}).get("gate_policy_respected", []))
    if not REQUIRED_GATES.issubset(gates):
        detected.add("GATE_POLICY_INCOMPATIBLE")

    # CR1
    if cert.get("control_stack_ref", {}).get("schema_version") != "QA_CONTROL_STACK_CERT.v1":
        detected.add("CONTROL_STACK_REF_MISMATCH")

    # CR2
    stmt = cert.get("theorem_statement", "")
    if not isinstance(stmt, str) or len(stmt) < 20:
        detected.add("THEOREM_STATEMENT_MISSING")

    # CR3
    summaries = cert.get("domain_summaries", [])
    found = {s.get("schema_version") for s in summaries}
    if not REQUIRED_DOMAIN_SCHEMAS.issubset(found):
        detected.add("DOMAIN_SUMMARY_INCOMPLETE")

    # CR4 — orbit_path uniform across comparison_table
    table = cert.get("comparison_table", [])
    orbit_paths  = [tuple(row.get("orbit_path", [])) for row in table]
    path_lengths = [row.get("path_length_k") for row in table]

    if len(set(orbit_paths)) > 1:
        detected.add("COMPARISON_TABLE_MISMATCH")

    # CR5 — path_length_k uniform
    if len(set(path_lengths)) > 1:
        detected.add("COMPARISON_TABLE_MISMATCH")

    # CR6
    pw = cert.get("canonical_pass_witness", {})
    if pw.get("result") != "cross_domain_equivalence_holds":
        detected.add("WITNESS_MISMATCH")

    # CR7
    fw = cert.get("canonical_fail_witness", {})
    if not fw.get("fail_types"):
        detected.add("WITNESS_MISMATCH")

    # Reconcile
    declared_result = cert.get("result")
    fail_ledger_types = {e["fail_type"] for e in cert.get("fail_ledger", [])}

    if detected:
        if declared_result != "FAIL":
            errors.append(f"Detected failures {sorted(detected)} but cert declares result=PASS.")
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


def _run_self_test(fixtures_dir: Path) -> dict:
    expected = {
        "report_pass_cross_domain.json":    True,
        "report_fail_table_mismatch.json":   True,
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate QA_CONTROL_STACK_REPORT.v1 certificates")
    parser.add_argument("cert_files", nargs="*")
    parser.add_argument("--self-test", action="store_true")
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
        print(f"{'PASS' if ok else 'FAIL'}  {path}")
        for e in errs:
            print(f"      {e}")
        if not ok:
            all_ok = False
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
