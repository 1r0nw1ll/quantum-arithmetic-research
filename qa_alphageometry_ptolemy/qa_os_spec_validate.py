"""
qa_os_spec_validate.py

Validator for QA_OS_SPEC.v1 - the QA Operating System unification artifact.

Performs:
1. JSON Schema validation (structural)
2. Semantic constraint validation (no redefinition drift)
3. Cross-reference validation (declared modules exist)

Usage:
    python qa_os_spec_validate.py                 # Validate spec
    python qa_os_spec_validate.py --json          # Output as JSON
    python qa_os_spec_validate.py --check-refs    # Also verify file references exist
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Set

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_cert_core import (
    canonical_json_compact, sha256_canonical, sha256_file,
    ValidationResult,
)


# ============================================================================
# REQUIRED CONSTRAINTS (drift prevention)
# ============================================================================

# These must appear in kernel.state_space.no_redefinition_constraints
REQUIRED_NO_REDEF = {
    "J=b*d",
    "K=d*a",
    "X=d*e",
    "C is base leg",
    "F is altitude leg",
    "Quantum ellipse major axis = 2D",
}

# Required generators
REQUIRED_GENERATORS = {"sigma", "mu", "lambda", "nu"}

# Required trace contract fields
REQUIRED_TRACE_FIELDS = {"move", "fail_type", "invariant_diff"}

# Required canonical invariants (subset check)
REQUIRED_INVARIANTS = {"b", "e", "d", "a", "C", "F", "G"}

# Required principles (all 5 are core OS axioms)
REQUIRED_PRINCIPLES = {
    "QA_GEOMETRIC_ACTION_PRINCIPLE",
    "QA_NON_REDUCTION_AXIOM",
    "QA_TIME_AXIOM",
    "QA_FAILURE_FIRST_CLASS",
    "QA_CAPABILITY_EQUALS_REACHABILITY",
}


# ============================================================================
# SEMANTIC VALIDATION
# ============================================================================

def validate_semantics(spec: Dict[str, Any]) -> ValidationResult:
    """
    Validate QA-specific semantic constraints beyond schema.

    These are the "no drift" checks that prevent redefinition.
    """
    v = ValidationResult()

    # Check schema_id
    v.check(
        spec.get("schema_id") == "QA_OS_SPEC.v1",
        f"schema_id must be 'QA_OS_SPEC.v1', got: {spec.get('schema_id')}"
    )

    # Check principles include required set
    principles = {p.get("id") for p in spec.get("principles", [])}
    missing_principles = REQUIRED_PRINCIPLES - principles
    v.check(
        len(missing_principles) == 0,
        f"Missing required principles: {sorted(missing_principles)}"
    )

    # Check kernel.state_space.no_redefinition_constraints
    ss = spec.get("kernel", {}).get("state_space", {})
    nrd = set(ss.get("no_redefinition_constraints", []) or [])
    # Check that required constraints are present (substring match for flexibility)
    for req in REQUIRED_NO_REDEF:
        found = any(req in constraint for constraint in nrd)
        v.check(found, f"Missing no_redefinition_constraint containing: {req}")

    # Check kernel.state_space.canonical_invariants includes required set
    invariants = set(ss.get("canonical_invariants", []) or [])
    missing_inv = REQUIRED_INVARIANTS - invariants
    v.check(
        len(missing_inv) == 0,
        f"Missing required canonical_invariants: {sorted(missing_inv)}"
    )

    # Check kernel.generators.declared includes required set
    gens = spec.get("kernel", {}).get("generators", {})
    declared = set(gens.get("declared", []) or [])
    missing_gens = REQUIRED_GENERATORS - declared
    v.check(
        len(missing_gens) == 0,
        f"Missing required generators: {sorted(missing_gens)}"
    )

    # Check generator semantics exist for each declared generator
    semantics_names = {s.get("name") for s in gens.get("semantics", [])}
    for g in REQUIRED_GENERATORS:
        v.check(
            g in semantics_names,
            f"Generator '{g}' declared but missing from semantics"
        )

    # Check reachability flags
    reach = gens.get("reachability", {})
    v.check(
        reach.get("bounded_return_in_k") is True,
        "kernel.generators.reachability.bounded_return_in_k must be true"
    )
    v.check(
        reach.get("connected_components_are_first_class") is True,
        "kernel.generators.reachability.connected_components_are_first_class must be true"
    )
    v.check(
        reach.get("failure_modes_are_first_class") is True,
        "kernel.generators.reachability.failure_modes_are_first_class must be true"
    )

    # Check runtime.trace_contract.leaf_fields
    tc = spec.get("runtime", {}).get("trace_contract", {})
    leaf_fields = set(tc.get("leaf_fields", []) or [])
    missing_fields = REQUIRED_TRACE_FIELDS - leaf_fields
    v.check(
        len(missing_fields) == 0,
        f"Missing required trace_contract.leaf_fields: {sorted(missing_fields)}"
    )
    v.check(
        tc.get("required") is True,
        "runtime.trace_contract.required must be true"
    )

    # Check filesystem.hashing
    hashing = spec.get("filesystem", {}).get("hashing", {})
    v.check(
        hashing.get("algorithm") == "sha256",
        f"filesystem.hashing.algorithm must be 'sha256', got: {hashing.get('algorithm')}"
    )
    v.check(
        hashing.get("self_hash_paradox_avoided") is True,
        "filesystem.hashing.self_hash_paradox_avoided must be true"
    )

    # Check security.golden_fixtures_required
    security = spec.get("security", {})
    v.check(
        security.get("golden_fixtures_required") is True,
        "security.golden_fixtures_required must be true"
    )

    # Check security.threat_model.covered_threats is non-empty
    threats = security.get("threat_model", {}).get("covered_threats", [])
    v.check(
        len(threats) >= 1,
        "security.threat_model.covered_threats must have at least 1 entry"
    )

    # Check security.failure_algebra.core_fail_types is non-empty
    fail_types = security.get("failure_algebra", {}).get("core_fail_types", [])
    v.check(
        len(fail_types) >= 1,
        "security.failure_algebra.core_fail_types must have at least 1 entry"
    )

    # Check network.multi_agent.agents is non-empty
    agents = spec.get("network", {}).get("multi_agent", {}).get("agents", [])
    v.check(
        len(agents) >= 1,
        "network.multi_agent.agents must have at least 1 entry"
    )

    return v


def validate_file_references(spec: Dict[str, Any], base_dir: str) -> ValidationResult:
    """
    Validate that file references in the spec actually exist.
    """
    v = ValidationResult()

    # Check validator paths
    validators = spec.get("runtime", {}).get("validators", [])
    for validator in validators:
        path = validator.get("path", "")
        # Handle paths relative to project root
        if path.startswith("qa_alphageometry_ptolemy/"):
            # Relative to parent of base_dir
            full_path = os.path.join(os.path.dirname(base_dir), path)
        else:
            full_path = os.path.join(base_dir, path)

        # Normalize and check
        full_path = os.path.normpath(full_path)
        if not os.path.exists(full_path):
            v.check(False, f"Validator path not found: {path}")

    # Check cert_roots exist
    cert_roots = spec.get("filesystem", {}).get("cert_roots", [])
    project_root = os.path.dirname(base_dir)
    for root in cert_roots:
        full_path = os.path.normpath(os.path.join(project_root, root))
        if not os.path.isdir(full_path):
            v.check(False, f"cert_root directory not found: {root}")

    # Check schema_roots exist
    schema_roots = spec.get("filesystem", {}).get("schema_roots", [])
    for root in schema_roots:
        full_path = os.path.normpath(os.path.join(project_root, root))
        if not os.path.isdir(full_path):
            v.check(False, f"schema_root directory not found: {root}")

    return v


def validate_schema(spec: Dict[str, Any], schema_path: str) -> ValidationResult:
    """
    Validate spec against JSON Schema.
    """
    v = ValidationResult()

    try:
        import jsonschema
    except ImportError:
        v.check(False, "jsonschema not installed; cannot validate schema")
        return v

    try:
        with open(schema_path) as f:
            schema = json.load(f)
        jsonschema.validate(instance=spec, schema=schema)
    except jsonschema.ValidationError as e:
        v.check(False, f"Schema validation failed: {e.message}")
    except Exception as e:
        v.check(False, f"Schema validation error: {e}")

    return v


# ============================================================================
# MAIN VALIDATOR
# ============================================================================

def validate_qa_os_spec(
    spec_path: str = None,
    check_refs: bool = False
) -> Dict[str, Any]:
    """
    Full validation of QA_OS_SPEC.v1.

    Returns:
        Dict with 'ok', 'result', 'checks', 'errors', 'hashes' fields.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if spec_path is None:
        spec_path = os.path.join(base_dir, "qa_os_spec.json")

    schema_path = os.path.join(base_dir, "schemas", "QA_OS_SPEC.v1.schema.json")

    results: Dict[str, Any] = {
        "ok": True,
        "result": "UNKNOWN",
        "checks": [],
        "errors": [],
        "hashes": {},
    }

    # Load spec
    try:
        with open(spec_path) as f:
            spec = json.load(f)
        results["checks"].append("load: OK")
    except Exception as e:
        results["ok"] = False
        results["result"] = "FAIL"
        results["errors"].append(f"Failed to load spec: {e}")
        return results

    # Compute hashes
    results["hashes"]["file_sha256"] = sha256_file(spec_path)
    results["hashes"]["canonical_sha256"] = sha256_canonical(spec)

    # Schema validation
    schema_result = validate_schema(spec, schema_path)
    if schema_result.is_valid:
        results["checks"].append("schema: OK")
    else:
        results["ok"] = False
        results["errors"].extend(schema_result.issues)

    # Semantic validation
    semantic_result = validate_semantics(spec)
    if semantic_result.is_valid:
        results["checks"].append("semantics: OK")
    else:
        results["ok"] = False
        results["errors"].extend(semantic_result.issues)

    # File reference validation (optional)
    if check_refs:
        ref_result = validate_file_references(spec, base_dir)
        if ref_result.is_valid:
            results["checks"].append("file_refs: OK")
        else:
            # File ref issues are warnings, not hard failures
            for issue in ref_result.issues:
                results["checks"].append(f"file_refs: WARN - {issue}")

    # Final result
    if results["ok"]:
        results["result"] = "PASS"
    else:
        results["result"] = "FAIL"

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QA_OS_SPEC.v1 Validator")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--check-refs", action="store_true",
                        help="Also verify file references exist")
    parser.add_argument("spec", nargs="?", help="Path to spec file (default: qa_os_spec.json)")
    args = parser.parse_args()

    result = validate_qa_os_spec(
        spec_path=args.spec,
        check_refs=args.check_refs
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"QA_OS_SPEC.v1 Validation: {result['result']}")
        print()
        print("Checks:")
        for check in result["checks"]:
            print(f"  - {check}")
        if result["errors"]:
            print()
            print("Errors:")
            for err in result["errors"]:
                print(f"  - {err}")
        print()
        print("Hashes:")
        print(f"  file_sha256:      {result['hashes'].get('file_sha256', 'N/A')[:32]}...")
        print(f"  canonical_sha256: {result['hashes'].get('canonical_sha256', 'N/A')[:32]}...")

    sys.exit(0 if result["ok"] else 1)
