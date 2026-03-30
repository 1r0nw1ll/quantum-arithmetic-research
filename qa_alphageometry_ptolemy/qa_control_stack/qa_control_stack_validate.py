"""
QA Control Stack Certificate Validator — Family [117]
Schema: QA_CONTROL_STACK_CERT.v1

Synthesis cert for the control/compiler spine. Asserts that QA_PLAN_CONTROL_COMPILER_CERT.v1
[106] is domain-generic: the orbit trajectory and path_length_k are preserved across distinct
physical domain instances under the same kernel-governed compiler law.

Checks:
  IH1   inherits_from == 'QA_CORE_SPEC.v1'
  IH2   spec_scope == 'family_extension'
  IH3   gate_policy_respected ⊇ [0,1,2,3,4,5]

  CS1   compiler_ref.schema_version == 'QA_PLAN_CONTROL_COMPILER_CERT.v1'

  CS2   each domain_instance_refs entry is a known domain_instance family of [106]:
          QA_CYMATIC_CONTROL_CERT.v1
          QA_SEISMIC_CONTROL_CERT.v1

  CS3   all domain_traces have the same orbit_trajectory list (cross-domain consistency)

  CS4   all domain_traces have the same path_length_k

  CS5   all domain_traces have the same initial_orbit_family

  CS6   all domain_traces have the same final_orbit_family

  CS7   cross_domain_claim.orbit_trajectory_preserved consistent with CS3+CS5+CS6

  CS8   cross_domain_claim.path_length_equal consistent with CS4

  CS9   cross_domain_claim.compiler_law_domain_generic consistent with CS7+CS8

  CS10  cross_domain_claim.canonical_orbit_trajectory matches all domain trace orbit_trajectories

  CS11  cross_domain_claim.canonical_path_length_k matches all domain trace path_length_k
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"



import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_GATES = {0, 1, 2, 3, 4, 5}

KNOWN_DOMAIN_INSTANCE_SCHEMAS = {
    "QA_CYMATIC_CONTROL_CERT.v1",
    "QA_SEISMIC_CONTROL_CERT.v1",
}


def validate(cert: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    detected: set[str] = set()

    if cert.get("schema_version") != "QA_CONTROL_STACK_CERT.v1":
        errors.append(
            f"schema_version mismatch: expected 'QA_CONTROL_STACK_CERT.v1', "
            f"got {cert.get('schema_version')!r}"
        )
        return False, errors

    # -------------------------------------------------------------------
    # IH1–IH3
    # -------------------------------------------------------------------
    if cert.get("inherits_from") != "QA_CORE_SPEC.v1":
        detected.add("INVALID_KERNEL_REFERENCE")
    if cert.get("spec_scope") != "family_extension":
        detected.add("SPEC_SCOPE_MISMATCH")
    gates = set(cert.get("core_kernel_compatibility", {}).get("gate_policy_respected", []))
    if not REQUIRED_GATES.issubset(gates):
        detected.add("GATE_POLICY_INCOMPATIBLE")

    # -------------------------------------------------------------------
    # CS1 — compiler_ref
    # -------------------------------------------------------------------
    if cert.get("compiler_ref", {}).get("schema_version") != "QA_PLAN_CONTROL_COMPILER_CERT.v1":
        detected.add("COMPILER_REF_MISMATCH")

    # -------------------------------------------------------------------
    # CS2 — domain_instance_refs
    # -------------------------------------------------------------------
    for ref in cert.get("domain_instance_refs", []):
        sv = ref.get("schema_version", "")
        if sv not in KNOWN_DOMAIN_INSTANCE_SCHEMAS:
            detected.add("DOMAIN_INSTANCE_MISMATCH")

    # -------------------------------------------------------------------
    # CS3–CS6 — cross-domain trace consistency
    # -------------------------------------------------------------------
    traces = cert.get("domain_traces", [])
    layer_failures: set[str] = set()

    trajectories = [tuple(t.get("orbit_trajectory", [])) for t in traces]
    path_lengths  = [t.get("path_length_k") for t in traces]
    init_orbits   = [t.get("initial_orbit_family") for t in traces]
    final_orbits  = [t.get("final_orbit_family") for t in traces]

    # CS3: orbit_trajectory uniform
    if len(set(trajectories)) > 1:
        detected.add("ORBIT_TRAJECTORY_MISMATCH")
        layer_failures.add("orbit_trajectory")

    # CS4: path_length_k uniform
    if len(set(path_lengths)) > 1:
        detected.add("PATH_LENGTH_MISMATCH")
        layer_failures.add("path_length")

    # CS5: initial_orbit_family uniform
    if len(set(init_orbits)) > 1:
        detected.add("ORBIT_TRAJECTORY_MISMATCH")
        layer_failures.add("orbit_trajectory")

    # CS6: final_orbit_family uniform
    if len(set(final_orbits)) > 1:
        detected.add("ORBIT_TRAJECTORY_MISMATCH")
        layer_failures.add("orbit_trajectory")

    # -------------------------------------------------------------------
    # CS7–CS9 — cross_domain_claim consistency
    # -------------------------------------------------------------------
    claim = cert.get("cross_domain_claim", {})
    orbits_preserved  = claim.get("orbit_trajectory_preserved")
    paths_equal       = claim.get("path_length_equal")
    law_generic       = claim.get("compiler_law_domain_generic")

    # CS7: orbit_trajectory_preserved iff orbits actually uniform
    orbit_uniform = "orbit_trajectory" not in layer_failures
    if orbits_preserved != orbit_uniform:
        detected.add("CROSS_DOMAIN_CLAIM_INCONSISTENT")
        layer_failures.add("claim")

    # CS8: path_length_equal iff path lengths actually uniform
    path_uniform = "path_length" not in layer_failures
    if paths_equal != path_uniform:
        detected.add("CROSS_DOMAIN_CLAIM_INCONSISTENT")
        layer_failures.add("claim")

    # CS9: compiler_law_domain_generic requires both CS7 and CS8 to be clean
    law_should_hold = orbit_uniform and path_uniform
    if law_generic != law_should_hold:
        detected.add("STACK_INCONSISTENCY")

    # CS10: canonical_orbit_trajectory matches all traces
    canonical_traj = tuple(claim.get("canonical_orbit_trajectory", []))
    if canonical_traj and any(t != canonical_traj for t in trajectories):
        detected.add("ORBIT_TRAJECTORY_MISMATCH")

    # CS11: canonical_path_length_k matches all traces
    canonical_k = claim.get("canonical_path_length_k")
    if canonical_k is not None and any(pl != canonical_k for pl in path_lengths):
        detected.add("PATH_LENGTH_MISMATCH")

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


def _run_self_test(fixtures_dir: Path) -> dict:
    expected = {
        "control_stack_pass_cross_domain.json":    True,
        "control_stack_fail_orbit_mismatch.json":   True,
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
    parser = argparse.ArgumentParser(description="Validate QA_CONTROL_STACK_CERT.v1 certificates")
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
