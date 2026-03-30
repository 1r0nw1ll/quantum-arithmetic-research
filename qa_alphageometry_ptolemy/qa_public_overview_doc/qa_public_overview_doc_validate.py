"""
QA Public Overview Doc Validator [120]
=======================================
Schema: QA_PUBLIC_OVERVIEW_DOC.v1

Derived from: QA_DUAL_SPINE_UNIFICATION_REPORT.v1 [119]

Checks:
  IH1  schema_version is QA_PUBLIC_OVERVIEW_DOC.v1
  IH2  cert_type is qa_public_overview_doc
  IH3  inherits from kernel QA_CORE_SPEC.v1 (via schema lineage)

  PO1  overview_ref == "QA_DUAL_SPINE_UNIFICATION_REPORT.v1"
  PO2  executive_summary non-empty (>=20 chars)
  PO3  spine_diagram present with both obstruction_spine and control_spine; each has chain+theorem
  PO4  obstruction_example present, has canonical_r/p/k integers and result string
  PO4b obstruction_example description mentions "v_p" or ratio claim
  PO5  control_example present, domains list has >=2 entries
  PO5b control_example mentions both "cymatics" and "seismology" in domain names
  PO6  why_it_matters non-empty (>=20 chars)
  PO7  spine_entry_points contains both QA_OBSTRUCTION_STACK_REPORT.v1 and QA_CONTROL_STACK_REPORT.v1
  PO8  canonical_pass_witness.overview_result == "public_overview_verified"
  PO9  canonical_fail_witness.fail_types non-empty

Fail types:
  INVALID_KERNEL_REFERENCE      IH1/IH2/IH3
  OVERVIEW_REF_MISMATCH         PO1
  EXECUTIVE_SUMMARY_MISSING     PO2
  SPINE_DIAGRAM_MISSING         PO3
  OBSTRUCTION_EXAMPLE_MISSING   PO4
  OBSTRUCTION_EXAMPLE_INCOMPLETE PO4b
  CONTROL_EXAMPLE_MISSING       PO5
  CONTROL_EXAMPLE_INCOMPLETE    PO5b
  WHY_IT_MATTERS_MISSING        PO6
  SPINE_ENTRY_POINTS_INCOMPLETE PO7
  WITNESS_MISMATCH              PO8/PO9
"""

from __future__ import annotations
import json
import os
import sys
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "QA_PUBLIC_OVERVIEW_DOC.v1"
CERT_TYPE      = "qa_public_overview_doc"

EXPECTED_OVERVIEW_REF            = "QA_DUAL_SPINE_UNIFICATION_REPORT.v1"
EXPECTED_OBSTRUCTION_ENTRY_POINT = "QA_OBSTRUCTION_STACK_REPORT.v1"
EXPECTED_CONTROL_ENTRY_POINT     = "QA_CONTROL_STACK_REPORT.v1"


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
    Validate a QA_PUBLIC_OVERVIEW_DOC.v1 certificate.

    Returns (passed: bool, fail_types: List[str]).
    """
    detected: List[str] = []

    # IH1-IH3
    detected.extend(_ih_checks(cert))

    # PO1: overview_ref
    if cert.get("overview_ref") != EXPECTED_OVERVIEW_REF:
        detected.append("OVERVIEW_REF_MISMATCH")

    # PO2: executive_summary
    if len(cert.get("executive_summary", "").strip()) < 20:
        detected.append("EXECUTIVE_SUMMARY_MISSING")

    # PO3: spine_diagram with both spines
    diagram = cert.get("spine_diagram")
    if not isinstance(diagram, dict):
        detected.append("SPINE_DIAGRAM_MISSING")
    else:
        obs_spine = diagram.get("obstruction_spine", {})
        ctl_spine = diagram.get("control_spine", {})
        if (
            not isinstance(obs_spine, dict)
            or len(obs_spine.get("chain", "").strip()) == 0
            or len(obs_spine.get("theorem", "").strip()) < 10
        ):
            detected.append("SPINE_DIAGRAM_MISSING")
        if (
            not isinstance(ctl_spine, dict)
            or len(ctl_spine.get("chain", "").strip()) == 0
            or len(ctl_spine.get("theorem", "").strip()) < 10
        ):
            detected.append("SPINE_DIAGRAM_MISSING")

    # PO4: obstruction_example
    obs_ex = cert.get("obstruction_example")
    if not isinstance(obs_ex, dict):
        detected.append("OBSTRUCTION_EXAMPLE_MISSING")
    else:
        # Required integer fields
        if not isinstance(obs_ex.get("canonical_r"), int):
            detected.append("OBSTRUCTION_EXAMPLE_MISSING")
        if not isinstance(obs_ex.get("canonical_p"), int):
            detected.append("OBSTRUCTION_EXAMPLE_MISSING")
        if not isinstance(obs_ex.get("canonical_k"), int):
            detected.append("OBSTRUCTION_EXAMPLE_MISSING")
        # PO4b: description mentions v_p or ratio
        desc = obs_ex.get("description", "") + " " + obs_ex.get("result", "")
        if "v_p" not in desc and "ratio" not in desc and "pruning" not in desc:
            detected.append("OBSTRUCTION_EXAMPLE_INCOMPLETE")

    # PO5: control_example with >=2 domains
    ctl_ex = cert.get("control_example")
    if not isinstance(ctl_ex, dict):
        detected.append("CONTROL_EXAMPLE_MISSING")
    else:
        domains = ctl_ex.get("domains", [])
        if not isinstance(domains, list) or len(domains) < 2:
            detected.append("CONTROL_EXAMPLE_MISSING")
        else:
            # PO5b: domain names include both cymatics and seismology
            domain_names = [str(d.get("name", "")).lower() for d in domains]
            if "cymatics" not in domain_names or "seismology" not in domain_names:
                detected.append("CONTROL_EXAMPLE_INCOMPLETE")

    # PO6: why_it_matters
    if len(cert.get("why_it_matters", "").strip()) < 20:
        detected.append("WHY_IT_MATTERS_MISSING")

    # PO7: spine_entry_points contains both expected refs
    entry_points = cert.get("spine_entry_points", [])
    if not isinstance(entry_points, list):
        detected.append("SPINE_ENTRY_POINTS_INCOMPLETE")
    else:
        combined = " ".join(entry_points)
        if EXPECTED_OBSTRUCTION_ENTRY_POINT not in combined:
            detected.append("SPINE_ENTRY_POINTS_INCOMPLETE")
        if EXPECTED_CONTROL_ENTRY_POINT not in combined:
            detected.append("SPINE_ENTRY_POINTS_INCOMPLETE")

    # PO8: canonical_pass_witness
    pass_witness = cert.get("canonical_pass_witness", {})
    if pass_witness.get("overview_result") != "public_overview_verified":
        detected.append("WITNESS_MISMATCH")

    # PO9: canonical_fail_witness.fail_types non-empty
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
    pass_path = os.path.join(fixtures_dir, "overview_pass_canonical.json")
    try:
        with open(pass_path) as f:
            cert = json.load(f)
        passed, fail_types = validate(cert)
        ok = passed and len(fail_types) == 0
        results.append({
            "fixture": "overview_pass_canonical.json",
            "expected": "PASS",
            "got": "PASS" if ok else f"FAIL({fail_types})",
            "ok": ok,
        })
        if not ok:
            all_ok = False
    except Exception as exc:
        results.append({"fixture": "overview_pass_canonical.json", "error": str(exc), "ok": False})
        all_ok = False

    # --- FAIL fixture ---
    fail_path = os.path.join(fixtures_dir, "overview_fail_entry_points_incomplete.json")
    try:
        with open(fail_path) as f:
            cert = json.load(f)
        passed, fail_types = validate(cert)
        expected_fail = {"SPINE_ENTRY_POINTS_INCOMPLETE"}
        ok = not passed and expected_fail.issubset(set(fail_types))
        results.append({
            "fixture": "overview_fail_entry_points_incomplete.json",
            "expected": "FAIL(SPINE_ENTRY_POINTS_INCOMPLETE)",
            "got": f"FAIL({fail_types})" if not passed else "PASS",
            "ok": ok,
        })
        if not ok:
            all_ok = False
    except Exception as exc:
        results.append({
            "fixture": "overview_fail_entry_points_incomplete.json",
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
        print("Usage: python qa_public_overview_doc_validate.py <cert.json>")
        print("       python qa_public_overview_doc_validate.py --self-test")
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
