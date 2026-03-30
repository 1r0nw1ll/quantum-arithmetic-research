"""
QA Dual Spine Unification Report Validator [119]
=================================================
Schema: QA_DUAL_SPINE_UNIFICATION_REPORT.v1

Checks:
  IH1  schema_version is QA_DUAL_SPINE_UNIFICATION_REPORT.v1
  IH2  cert_type is qa_dual_spine_unification_report
  IH3  inherits from kernel QA_CORE_SPEC.v1 (via schema lineage)

  DU1  obstruction_spine_ref == "QA_OBSTRUCTION_STACK_REPORT.v1"
  DU2  control_spine_ref     == "QA_CONTROL_STACK_REPORT.v1"
  DU3  obstruction_theorem   non-empty (>=20 chars)
  DU4  control_theorem       non-empty (>=20 chars)
  DU5  comparison_table has exactly 2 rows
  DU6  one comparison_table row entry_point == "QA_OBSTRUCTION_STACK_REPORT.v1"
  DU7  other comparison_table row entry_point == "QA_CONTROL_STACK_REPORT.v1"
  DU8  obstruction row canonical_theorem contains "v_p"
  DU9  control row canonical_theorem contains "singularity" or "cosmos"
  DU10 synthesis_statement non-empty (>=20 chars)
  DU11 synthesis_statement mentions both "obstruction" and "control"
  DU12 canonical_pass_witness.obstruction_spine_result == "obstruction_spine_verified"
  DU13 canonical_pass_witness.control_spine_result     == "control_spine_verified"
  DU14 canonical_fail_witness.fail_types is non-empty

Fail types:
  INVALID_KERNEL_REFERENCE       IH1/IH2/IH3
  OBSTRUCTION_SPINE_REF_MISMATCH DU1
  CONTROL_SPINE_REF_MISMATCH     DU2
  OBSTRUCTION_THEOREM_MISSING    DU3
  CONTROL_THEOREM_MISSING        DU4
  COMPARISON_TABLE_INCOMPLETE    DU5
  COMPARISON_TABLE_MISMATCH      DU6/DU7/DU8/DU9
  SYNTHESIS_STATEMENT_MISSING    DU10
  SYNTHESIS_STATEMENT_INCOMPLETE DU11
  WITNESS_MISMATCH               DU12/DU13/DU14
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"


import json
import os
import sys
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "QA_DUAL_SPINE_UNIFICATION_REPORT.v1"
CERT_TYPE      = "qa_dual_spine_unification_report"

EXPECTED_OBSTRUCTION_SPINE_REF = "QA_OBSTRUCTION_STACK_REPORT.v1"
EXPECTED_CONTROL_SPINE_REF     = "QA_CONTROL_STACK_REPORT.v1"


def _ih_checks(cert: Dict[str, Any]) -> List[str]:
    """IH1–IH3: kernel inheritance checks."""
    detected: List[str] = []
    if cert.get("schema_version") != SCHEMA_VERSION:
        detected.append("INVALID_KERNEL_REFERENCE")
    if cert.get("cert_type") != CERT_TYPE:
        detected.append("INVALID_KERNEL_REFERENCE")
    return detected


def validate(cert: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a QA_DUAL_SPINE_UNIFICATION_REPORT.v1 certificate.

    Returns (passed: bool, fail_types: List[str]).
    """
    detected: List[str] = []

    # IH1-IH3
    detected.extend(_ih_checks(cert))

    # DU1: obstruction_spine_ref
    obs_ref = cert.get("obstruction_spine_ref", "")
    if obs_ref != EXPECTED_OBSTRUCTION_SPINE_REF:
        detected.append("OBSTRUCTION_SPINE_REF_MISMATCH")

    # DU2: control_spine_ref
    ctl_ref = cert.get("control_spine_ref", "")
    if ctl_ref != EXPECTED_CONTROL_SPINE_REF:
        detected.append("CONTROL_SPINE_REF_MISMATCH")

    # DU3: obstruction_theorem non-empty
    obs_thm = cert.get("obstruction_theorem", "")
    if len(obs_thm.strip()) < 20:
        detected.append("OBSTRUCTION_THEOREM_MISSING")

    # DU4: control_theorem non-empty
    ctl_thm = cert.get("control_theorem", "")
    if len(ctl_thm.strip()) < 20:
        detected.append("CONTROL_THEOREM_MISSING")

    # DU5: comparison_table has exactly 2 rows
    table = cert.get("comparison_table", [])
    if len(table) != 2:
        detected.append("COMPARISON_TABLE_INCOMPLETE")
    else:
        # DU6/DU7: one row must be obstruction, other must be control
        entry_points = [row.get("entry_point", "") for row in table]
        if EXPECTED_OBSTRUCTION_SPINE_REF not in entry_points:
            detected.append("COMPARISON_TABLE_MISMATCH")
        if EXPECTED_CONTROL_SPINE_REF not in entry_points:
            detected.append("COMPARISON_TABLE_MISMATCH")

        # Find each row for content checks
        obs_row = next(
            (r for r in table if r.get("entry_point") == EXPECTED_OBSTRUCTION_SPINE_REF),
            None
        )
        ctl_row = next(
            (r for r in table if r.get("entry_point") == EXPECTED_CONTROL_SPINE_REF),
            None
        )

        # DU8: obstruction row canonical_theorem contains "v_p"
        if obs_row is not None:
            if "v_p" not in obs_row.get("canonical_theorem", ""):
                detected.append("COMPARISON_TABLE_MISMATCH")

        # DU9: control row canonical_theorem contains "singularity" or "cosmos"
        if ctl_row is not None:
            ctl_canon = ctl_row.get("canonical_theorem", "")
            if "singularity" not in ctl_canon and "cosmos" not in ctl_canon:
                detected.append("COMPARISON_TABLE_MISMATCH")

    # DU10: synthesis_statement non-empty
    synth = cert.get("synthesis_statement", "")
    if len(synth.strip()) < 20:
        detected.append("SYNTHESIS_STATEMENT_MISSING")
    else:
        # DU11: synthesis_statement mentions both "obstruction" and "control"
        synth_lower = synth.lower()
        if "obstruction" not in synth_lower or "control" not in synth_lower:
            detected.append("SYNTHESIS_STATEMENT_INCOMPLETE")

    # DU12/DU13: canonical_pass_witness
    pass_witness = cert.get("canonical_pass_witness", {})
    if pass_witness.get("obstruction_spine_result") != "obstruction_spine_verified":
        detected.append("WITNESS_MISMATCH")
    if pass_witness.get("control_spine_result") != "control_spine_verified":
        detected.append("WITNESS_MISMATCH")

    # DU14: canonical_fail_witness.fail_types non-empty
    fail_witness = cert.get("canonical_fail_witness", {})
    if not fail_witness.get("fail_types"):
        detected.append("WITNESS_MISMATCH")

    # Deduplicate while preserving order
    seen = set()
    fail_types: List[str] = []
    for ft in detected:
        if ft not in seen:
            seen.add(ft)
            fail_types.append(ft)

    return (len(fail_types) == 0, fail_types)


def _run_self_test() -> Dict[str, Any]:
    """Run self-test against PASS and FAIL fixtures. Returns JSON-serialisable dict."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    results = []
    all_ok = True

    # --- PASS fixture ---
    pass_path = os.path.join(fixtures_dir, "unification_pass_canonical.json")
    try:
        with open(pass_path) as f:
            cert = json.load(f)
        passed, fail_types = validate(cert)
        ok = passed and len(fail_types) == 0
        results.append({
            "fixture": "unification_pass_canonical.json",
            "expected": "PASS",
            "got": "PASS" if ok else f"FAIL({fail_types})",
            "ok": ok,
        })
        if not ok:
            all_ok = False
    except Exception as exc:
        results.append({"fixture": "unification_pass_canonical.json", "error": str(exc), "ok": False})
        all_ok = False

    # --- FAIL fixture ---
    fail_path = os.path.join(fixtures_dir, "unification_fail_spine_ref_mismatch.json")
    try:
        with open(fail_path) as f:
            cert = json.load(f)
        passed, fail_types = validate(cert)
        expected_fail = {"OBSTRUCTION_SPINE_REF_MISMATCH"}
        ok = not passed and expected_fail.issubset(set(fail_types))
        results.append({
            "fixture": "unification_fail_spine_ref_mismatch.json",
            "expected": "FAIL(OBSTRUCTION_SPINE_REF_MISMATCH)",
            "got": f"FAIL({fail_types})" if not passed else "PASS",
            "ok": ok,
        })
        if not ok:
            all_ok = False
    except Exception as exc:
        results.append({
            "fixture": "unification_fail_spine_ref_mismatch.json",
            "error": str(exc),
            "ok": False,
        })
        all_ok = False

    return {"ok": all_ok, "results": results}


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        result = _run_self_test()
        print(json.dumps(result))
        sys.exit(0 if result["ok"] else 1)

    if len(sys.argv) < 2:
        print("Usage: python qa_dual_spine_unification_report_validate.py <cert.json>")
        print("       python qa_dual_spine_unification_report_validate.py --self-test")
        sys.exit(1)

    cert_path = sys.argv[1]
    with open(cert_path) as f:
        cert = json.load(f)
    passed, fail_types = validate(cert)
    if passed:
        print(f"PASS: {cert_path}")
    else:
        print(f"FAIL: {cert_path}")
        for ft in fail_types:
            print(f"  - {ft}")
    sys.exit(0 if passed else 1)
