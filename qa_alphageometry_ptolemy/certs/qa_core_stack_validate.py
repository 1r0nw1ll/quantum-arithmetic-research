#!/usr/bin/env python3
"""
QA Core System Stack Validator

Validates the QA_CORE_SYSTEM_STACK.v1 certificate and its witness files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent
CERT_PATH = SCRIPT_DIR / "QA_CORE_SYSTEM_STACK.v1.json"
WITNESS_DIR = SCRIPT_DIR / "witness" / "core_stack"
SCHEMA_PATH = SCRIPT_DIR.parent / "schemas" / "QA_CORE_SYSTEM_STACK.v1.schema.json"


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def validate_schema(cert: Dict, schema: Dict) -> Tuple[bool, List[str]]:
    """Validate certificate against schema."""
    errors = []

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in cert:
            errors.append(f"Missing required field: {field}")

    # Check schema_id
    if cert.get("schema_id") != "QA_CORE_SYSTEM_STACK.v1":
        errors.append(f"Invalid schema_id: {cert.get('schema_id')}")

    return len(errors) == 0, errors


def validate_witnesses(cert: Dict) -> Tuple[bool, List[str]]:
    """Validate that all referenced witnesses exist."""
    errors = []

    for witness_ref in cert.get("witnesses", []):
        witness_id = witness_ref.get("witness_id", "")
        # Map witness_id to filename
        name_map = {
            "WITNESS__CLAUDE_SKILLS__v1": "claude_skills.witness.json",
            "WITNESS__CONTEXT_GEOMETRY__v1": "context_geometry.witness.json",
            "WITNESS__KIMI_K2__v1": "kimi_k2.witness.json",
            "WITNESS__KOSMOS__v1": "kosmos.witness.json",
            "WITNESS__CLARA__v1": "clara.witness.json",
            "WITNESS__STAT_MECH_NN__v1": "stat_mech.witness.json",
            "WITNESS__QUANTUM_MEMORY__v1": "quantum_memory.witness.json",
        }

        filename = name_map.get(witness_id)
        if not filename:
            errors.append(f"Unknown witness_id: {witness_id}")
            continue

        witness_path = WITNESS_DIR / filename
        if not witness_path.exists():
            errors.append(f"Missing witness file: {witness_path}")
            continue

        # Load and validate witness
        try:
            witness = load_json(witness_path)
            if witness.get("witness_id") != witness_id:
                errors.append(f"Witness ID mismatch in {filename}")
        except Exception as e:
            errors.append(f"Error loading {filename}: {e}")

    return len(errors) == 0, errors


def validate_cross_invariants(cert: Dict) -> Tuple[bool, List[str]]:
    """Validate cross-layer invariants are properly structured."""
    errors = []
    valid_layers = {"AGENT_OS", "MODEL_SUBSTRATE", "PHYSICAL_SUBSTRATE"}

    for ci in cert.get("cross_invariants", []):
        ci_id = ci.get("id", "UNKNOWN")
        layers = ci.get("layers", [])

        if not ci.get("statement"):
            errors.append(f"Cross-invariant {ci_id} missing statement")

        for layer in layers:
            if layer not in valid_layers:
                errors.append(f"Cross-invariant {ci_id} has invalid layer: {layer}")

    return len(errors) == 0, errors


def validate_layer_consistency(cert: Dict) -> Tuple[bool, List[str]]:
    """Validate that each layer has required fields."""
    errors = []

    for layer_name in ["agent_layer", "model_layer", "physics_layer"]:
        layer = cert.get(layer_name, {})

        if not layer.get("description"):
            errors.append(f"{layer_name} missing description")

        if not layer.get("components"):
            errors.append(f"{layer_name} missing components")

    return len(errors) == 0, errors


def run_validation() -> Tuple[bool, Dict]:
    """Run all validations and return results."""
    results = {
        "cert_exists": False,
        "schema_valid": False,
        "witnesses_valid": False,
        "cross_invariants_valid": False,
        "layers_valid": False,
        "errors": [],
    }

    # Check cert exists
    if not CERT_PATH.exists():
        results["errors"].append(f"Certificate not found: {CERT_PATH}")
        return False, results
    results["cert_exists"] = True

    # Load certificate
    cert = load_json(CERT_PATH)

    # Schema validation
    if SCHEMA_PATH.exists():
        schema = load_json(SCHEMA_PATH)
        ok, errs = validate_schema(cert, schema)
        results["schema_valid"] = ok
        results["errors"].extend(errs)
    else:
        results["errors"].append(f"Schema not found: {SCHEMA_PATH}")

    # Witness validation
    ok, errs = validate_witnesses(cert)
    results["witnesses_valid"] = ok
    results["errors"].extend(errs)

    # Cross-invariant validation
    ok, errs = validate_cross_invariants(cert)
    results["cross_invariants_valid"] = ok
    results["errors"].extend(errs)

    # Layer validation
    ok, errs = validate_layer_consistency(cert)
    results["layers_valid"] = ok
    results["errors"].extend(errs)

    all_valid = all([
        results["cert_exists"],
        results["schema_valid"],
        results["witnesses_valid"],
        results["cross_invariants_valid"],
        results["layers_valid"],
    ])

    return all_valid, results


def main():
    print("=" * 60)
    print("QA Core System Stack Validator")
    print("=" * 60)

    valid, results = run_validation()

    print(f"\nCertificate exists:     {'PASS' if results['cert_exists'] else 'FAIL'}")
    print(f"Schema valid:           {'PASS' if results['schema_valid'] else 'FAIL'}")
    print(f"Witnesses valid:        {'PASS' if results['witnesses_valid'] else 'FAIL'}")
    print(f"Cross-invariants valid: {'PASS' if results['cross_invariants_valid'] else 'FAIL'}")
    print(f"Layers valid:           {'PASS' if results['layers_valid'] else 'FAIL'}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  - {err}")

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if valid else 'FAIL'}")
    print(f"{'=' * 60}")

    return 0 if valid else 1


if __name__ == "__main__":
    exit(main())
