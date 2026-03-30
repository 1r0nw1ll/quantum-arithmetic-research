#!/usr/bin/env python3
"""
QA Seismic Pattern Control Certificate Validator
Family [110] — domain_instance of QA_PLAN_CONTROL_COMPILER_CERT.v1 [106]

Checks: IH1, IH2, S1-S6
"""

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"


import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parent.parent

SCHEMA_VERSION = "QA_SEISMIC_CONTROL_CERT.v1"
CERT_TYPE = "seismic_control"
PARENT_FAMILY = "QA_PLAN_CONTROL_COMPILER_CERT.v1"
REQUIRED_SPEC_SCOPE = "domain_instance"
KNOWN_ORBIT_FAMILIES = frozenset(["singularity", "satellite", "cosmos", "out_of_orbit"])
DISORDERED_CLASS = "disordered"


class ValidationOutput:
    def __init__(self):
        self.issues: list[str] = []
        self._ok = True

    def fail(self, msg: str):
        self.issues.append(msg)
        self._ok = False

    def warn(self, msg: str):
        self.issues.append(f"WARN: {msg}")

    def ok(self) -> bool:
        return self._ok


def _build_edge_set(control_graph: dict) -> set[tuple[str, str, str]]:
    edges = set()
    for e in control_graph.get("edges", []):
        edges.add((e["from"], e["move"], e["to"]))
    return edges


def validate_seismic_cert(cert: dict) -> ValidationOutput:
    out = ValidationOutput()

    # --- Schema gate ---
    if cert.get("schema_version") != SCHEMA_VERSION:
        out.fail(f"schema_version must be '{SCHEMA_VERSION}', got '{cert.get('schema_version')}'")
        return out
    if cert.get("cert_type") != CERT_TYPE:
        out.fail(f"cert_type must be '{CERT_TYPE}', got '{cert.get('cert_type')}'")
        return out

    declared_result = cert.get("result", "")
    if declared_result not in ("PASS", "FAIL"):
        out.fail(f"result must be PASS or FAIL, got '{declared_result}'")
        return out

    # --- Collect detected failures ---
    detected: set[str] = set()

    # IH1: inherits_from
    inherits_from = cert.get("inherits_from", "")
    if inherits_from != PARENT_FAMILY:
        detected.add("INVALID_KERNEL_REFERENCE")

    # IH2: spec_scope
    spec_scope = cert.get("spec_scope", "")
    if spec_scope != REQUIRED_SPEC_SCOPE:
        detected.add("SPEC_SCOPE_MISMATCH")

    # Extract structural fields
    generator_sequence = cert.get("generator_sequence", [])
    target_spec = cert.get("target_spec", {})
    observed = cert.get("observed_final_state", {})
    control_graph = cert.get("control_graph", {})
    qa_mapping = cert.get("qa_mapping", {})

    path_length_k = observed.get("path_length_k", -1)
    max_path_length_k = target_spec.get("max_path_length_k", -1)
    target_class = target_spec.get("target_pattern_class", "")
    final_class = observed.get("pattern_class", "")
    final_orbit = qa_mapping.get("final_orbit_family", "")

    # S1: path_length_k == len(generator_sequence)
    if path_length_k != len(generator_sequence):
        detected.add("PATH_LENGTH_EXCEEDED")

    # S2: path_length_k <= max_path_length_k
    if isinstance(path_length_k, int) and isinstance(max_path_length_k, int):
        if path_length_k > max_path_length_k:
            detected.add("PATH_LENGTH_EXCEEDED")

    # S3: final pattern class == target
    if final_class != target_class:
        detected.add("GOAL_NOT_REACHED")

    # S4: all transitions are legal edges in control_graph
    if control_graph:
        legal_edges = _build_edge_set(control_graph)
        # Walk the generator sequence from initial_state
        current = cert.get("initial_state", {}).get("pattern_class", "")
        for step in generator_sequence:
            move = step.get("move", "")
            expected = step.get("expected_transition", "")
            # Parse expected_transition "X -> Y"
            parts = [p.strip() for p in expected.split("->")]
            if len(parts) == 2:
                from_cls, to_cls = parts[0], parts[1]
            else:
                from_cls, to_cls = current, ""
            edge = (from_cls, move, to_cls)
            if edge not in legal_edges:
                detected.add("ILLEGAL_TRANSITION")
            current = to_cls

    # S5: final pattern class maps to recognized orbit family
    if final_orbit not in KNOWN_ORBIT_FAMILIES:
        detected.add("ORBIT_CLASS_MISMATCH")

    # S6: final orbit family is not out_of_orbit (not disordered)
    if final_orbit == "out_of_orbit" or final_class == DISORDERED_CLASS:
        detected.add("NONLINEAR_ESCAPE")

    # --- Reconcile with declared result ---
    declared_fail_types = {e["fail_type"] for e in cert.get("fail_ledger", []) if "fail_type" in e}

    if declared_result == "PASS":
        if detected:
            out.fail(f"Cert declares PASS but validator detected failures: {sorted(detected)}")
        elif declared_fail_types:
            out.fail(f"Cert declares PASS but fail_ledger is non-empty: {sorted(declared_fail_types)}")
    elif declared_result == "FAIL":
        if not declared_fail_types:
            out.fail("Cert declares FAIL but fail_ledger is empty")
        elif detected != declared_fail_types:
            out.fail(
                f"Declared fail_ledger {sorted(declared_fail_types)} does not match "
                f"detected failures {sorted(detected)}"
            )

    # --- Validate validation_checks array reconciles ---
    checks = cert.get("validation_checks", [])
    check_ids_failed = {c["check_id"] for c in checks if not c.get("passed", True)}
    # Just warn on mismatch — not a structural failure
    if declared_result == "PASS" and check_ids_failed:
        out.warn(f"PASS cert has failed checks in validation_checks: {sorted(check_ids_failed)}")

    return out


def _self_test() -> dict:
    fixtures_dir = HERE / "fixtures"
    results = {}
    for fpath in sorted(fixtures_dir.glob("*.json")):
        with open(fpath) as f:
            cert = json.load(f)
        out = validate_seismic_cert(cert)
        declared = cert.get("result", "UNKNOWN")
        structural_ok = out.ok()
        status = "PASS" if structural_ok else "FAIL"
        results[fpath.name] = {
            "declared_result": declared,
            "structural_validation": status,
            "issues": out.issues,
        }
    return results


def main():
    if "--self-test" in sys.argv:
        results = _self_test()
        print(json.dumps(results, indent=2))
        all_ok = all(r["structural_validation"] == "PASS" for r in results.values())
        sys.exit(0 if all_ok else 1)

    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        fpath = pathlib.Path(sys.argv[idx + 1])
        with open(fpath) as f:
            cert = json.load(f)
        out = validate_seismic_cert(cert)
        if out.ok():
            print(f"OK: {fpath.name} ({cert.get('result')})")
        else:
            print(f"FAIL: {fpath.name}")
            for issue in out.issues:
                print(f"  - {issue}")
        sys.exit(0 if out.ok() else 1)

    print("Usage: qa_seismic_control_validate.py [--self-test] [--file <path>]")
    sys.exit(1)


if __name__ == "__main__":
    main()
