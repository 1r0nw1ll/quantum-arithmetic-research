#!/usr/bin/env python3
"""QA Inheritance Compat [109] validator — QA_INHERITANCE_COMPAT_CERT.v1

Certifies a single inheritance edge between two QA cert families as a
first-class object. Turns '[108] extends [107]' from a validator convention
into a certified edge in the QA spec graph.

Checks:
  IC1  parent_family.schema_version is recognised        → PARENT_CERT_MISSING
  IC2  child_family.schema_version is recognised         → CHILD_CERT_MISSING
  IC3  declared_inherits_from == parent schema_version   → INVALID_INHERITANCE_EDGE
  IC4  child gate policy ⊇ parent gate policy            → GATE_POLICY_INCOMPATIBLE
  IC5  child failure algebra ⊇ parent types (if both non-empty) → FAILURE_ALGEBRA_BREAKS_PARENT
  IC6  child logging ⊇ {move, fail_type, invariant_diff} (if non-empty) → LOGGING_CONTRACT_INCOMPATIBLE
  IC7  child preserves_invariants_refs ⊆ parent invariant_names (if both non-empty) → INVARIANT_REFERENCE_UNRESOLVED
  IC8  scope_transition is in VALID_SCOPE_TRANSITIONS     → SCOPE_TRANSITION_INVALID

Usage:
  python qa_inheritance_compat_validate.py --self-test
  python qa_inheritance_compat_validate.py --file fixtures/inherit_pass_107_to_108.json
"""

import json
import sys
import argparse
from pathlib import Path


# ── known QA cert families ────────────────────────────────────────────────────
KNOWN_FAMILIES: dict = {
    "QA_CORE_SPEC.v1":                    {"spec_scope": "kernel"},
    "QA_AREA_QUANTIZATION_CERT.v1":        {"spec_scope": "family_extension"},
    "QA_PLAN_CONTROL_COMPILER_CERT.v1":   {"spec_scope": "family_extension"},
    "QA_CYMATIC_CONTROL_CERT.v1":          {"spec_scope": "domain_instance"},
    "QA_CYMATIC_PLANNER_CERT.v1":          {"spec_scope": "domain_instance"},
    "QA_CYMATICS_CERT.v1":                 {"spec_scope": "domain_instance"},
    "QA_CYMATIC_FARADAY_CERT.v1":          {"spec_scope": "domain_instance"},
    "QA_SEISMIC_CONTROL_CERT.v1":          {"spec_scope": "domain_instance"},
    "QA_AREA_QUANTIZATION_PK_CERT.v1":    {"spec_scope": "family_extension"},
    "QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1": {"spec_scope": "family_extension"},
    "QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1":  {"spec_scope": "family_extension"},
    "QA_OBSTRUCTION_EFFICIENCY_CERT.v1":     {"spec_scope": "family_extension"},
    "QA_OBSTRUCTION_STACK_CERT.v1":          {"spec_scope": "family_extension"},
    "QA_OBSTRUCTION_STACK_REPORT.v1":        {"spec_scope": "family_extension"},
    "QA_CONTROL_STACK_CERT.v1":              {"spec_scope": "family_extension"},
    "QA_CONTROL_STACK_REPORT.v1":            {"spec_scope": "family_extension"},
    "QA_DUAL_SPINE_UNIFICATION_REPORT.v1":  {"spec_scope": "family_extension"},
    "QA_PUBLIC_OVERVIEW_DOC.v1":            {"spec_scope": "family_extension"},
}

# ── valid scope transitions ───────────────────────────────────────────────────
VALID_SCOPE_TRANSITIONS = frozenset([
    "kernel -> family_extension",
    "kernel -> domain_kernel",
    "domain_kernel -> family_extension",
    "domain_kernel -> domain_instance",
    "family_extension -> family_extension",
    "family_extension -> domain_instance",
])

# ── required logging fields ───────────────────────────────────────────────────
REQUIRED_LOG_FIELDS = frozenset(["move", "fail_type", "invariant_diff"])

# ── cert fail types ───────────────────────────────────────────────────────────
INHERIT_FAIL_TYPES = frozenset([
    "PARENT_CERT_MISSING",
    "CHILD_CERT_MISSING",
    "INVALID_INHERITANCE_EDGE",
    "GATE_POLICY_INCOMPATIBLE",
    "FAILURE_ALGEBRA_BREAKS_PARENT",
    "LOGGING_CONTRACT_INCOMPATIBLE",
    "INVARIANT_REFERENCE_UNRESOLVED",
    "SCOPE_TRANSITION_INVALID",
])


# ── output accumulator ────────────────────────────────────────────────────────
class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


# ── validator ─────────────────────────────────────────────────────────────────
def validate_inheritance_compat_cert(cert: dict) -> dict:
    out = _Out()
    detected: set[str] = set()

    # ── schema / cert_type guards ───────────────────────────────────────────
    if cert.get("schema_version") != "QA_INHERITANCE_COMPAT_CERT.v1":
        out.fail(f"schema_version must be 'QA_INHERITANCE_COMPAT_CERT.v1', got {cert.get('schema_version')!r}")
    if cert.get("cert_type") != "qa_inheritance_compat":
        out.fail(f"cert_type must be 'qa_inheritance_compat', got {cert.get('cert_type')!r}")

    for field in ["certificate_id", "parent_family", "child_family",
                  "declared_inherits_from", "scope_transition",
                  "extracted_parent", "extracted_child",
                  "compatibility_claims", "validation_checks", "fail_ledger", "result"]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")

    if out.errors:
        return _reconcile(cert, out, detected)

    parent  = cert.get("parent_family", {})
    child   = cert.get("child_family", {})
    ep      = cert.get("extracted_parent", {})
    ec      = cert.get("extracted_child", {})

    parent_sv = parent.get("schema_version", "")
    child_sv  = child.get("schema_version", "")

    # ── IC1: parent recognised ─────────────────────────────────────────────
    if parent_sv not in KNOWN_FAMILIES:
        detected.add("PARENT_CERT_MISSING")

    # ── IC2: child recognised ──────────────────────────────────────────────
    if child_sv not in KNOWN_FAMILIES:
        detected.add("CHILD_CERT_MISSING")

    # ── IC3: declared_inherits_from matches parent schema_version ──────────
    if cert.get("declared_inherits_from") != parent_sv:
        detected.add("INVALID_INHERITANCE_EDGE")

    # ── IC4: child gate policy ⊇ parent gate policy ────────────────────────
    parent_gates = set(ep.get("gate_policy", []))
    child_gates  = set(ec.get("gate_policy", []))
    if not parent_gates.issubset(child_gates):
        detected.add("GATE_POLICY_INCOMPATIBLE")

    # ── IC5: child failure algebra ⊇ parent (only when both non-empty) ─────
    parent_fa = set(ep.get("failure_algebra_types", []))
    child_fa  = set(ec.get("failure_algebra_types", []))
    if parent_fa and not parent_fa.issubset(child_fa):
        detected.add("FAILURE_ALGEBRA_BREAKS_PARENT")

    # ── IC6: child logging ⊇ required (only when child logging non-empty) ──
    child_log = set(ec.get("logging_required_fields", []))
    if child_log and not REQUIRED_LOG_FIELDS.issubset(child_log):
        detected.add("LOGGING_CONTRACT_INCOMPATIBLE")

    # ── IC7: child invariant refs ⊆ parent names (when both non-empty) ─────
    parent_inv_names = set(ep.get("invariant_names", []))
    child_inv_refs   = set(ec.get("preserves_invariants_refs", []))
    if child_inv_refs and parent_inv_names:
        if not child_inv_refs.issubset(parent_inv_names):
            detected.add("INVARIANT_REFERENCE_UNRESOLVED")

    # ── IC8: scope transition is valid ─────────────────────────────────────
    scope_transition = cert.get("scope_transition", "")
    if scope_transition not in VALID_SCOPE_TRANSITIONS:
        detected.add("SCOPE_TRANSITION_INVALID")

    return _reconcile(cert, out, detected)


def _reconcile(cert: dict, out: _Out, detected: set) -> dict:
    declared_result = cert.get("result", "")
    declared_ledger = cert.get("fail_ledger", [])
    declared_fail_types = {e.get("fail_type") for e in declared_ledger
                           if isinstance(e, dict)}

    for ft in declared_fail_types:
        if ft not in INHERIT_FAIL_TYPES:
            out.warn(f"unrecognised fail_type in fail_ledger: {ft!r}")

    if declared_result == "PASS":
        if detected:
            for ft in sorted(detected):
                out.fail(f"cert declares PASS but detected: {ft}")
    elif declared_result == "FAIL":
        missing_from_ledger = detected - declared_fail_types
        for ft in sorted(missing_from_ledger):
            out.warn(f"detected {ft} but not declared in fail_ledger")
        phantom = declared_fail_types - detected
        for ft in sorted(phantom):
            out.warn(f"fail_ledger declares {ft} but validator did not detect it")
    else:
        out.fail(f"result must be 'PASS' or 'FAIL', got {declared_result!r}")

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


# ── file entry point ──────────────────────────────────────────────────────────
def validate_file(path: Path) -> dict:
    with open(path) as f:
        cert = json.load(f)
    return validate_inheritance_compat_cert(cert)


# ── self-test ─────────────────────────────────────────────────────────────────
def self_test() -> dict:
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = {
        "inherit_pass_107_to_108.json":         True,
        "inherit_pass_107_to_111.json":         True,
        "inherit_pass_107_to_112.json":         True,
        "inherit_pass_107_to_113.json":         True,
        "inherit_pass_107_to_114.json":         True,
        "inherit_pass_107_to_115.json":         True,
        "inherit_pass_107_to_116.json":         True,
        "inherit_pass_107_to_117.json":         True,
        "inherit_pass_107_to_118.json":         True,
        "inherit_pass_107_to_119.json":         True,
        "inherit_pass_107_to_120.json":         True,
        "inherit_pass_106_to_105.json":          True,
        "inherit_pass_106_to_110.json":          True,
        "inherit_fail_gate_policy_deleted.json": True,
    }

    results = []
    all_ok = True

    for fname, expect_ok in expected.items():
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        r = validate_file(fpath)
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


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QA Inheritance Compat [109] validator")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test against fixture suite")
    parser.add_argument("--file", type=Path,
                        help="Validate a single cert file")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if args.file:
        result = validate_file(args.file)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
