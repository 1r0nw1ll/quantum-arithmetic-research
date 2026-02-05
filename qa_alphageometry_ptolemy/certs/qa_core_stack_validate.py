#!/usr/bin/env python3
"""
QA Core System Stack Validator

Validates the QA_CORE_SYSTEM_STACK.v1 certificate and its witness files.
Includes drift detection and golden fixture tests.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent
CERT_PATH = SCRIPT_DIR / "QA_CORE_SYSTEM_STACK.v1.json"
WITNESS_DIR = SCRIPT_DIR / "witness" / "core_stack"
SCHEMA_PATH = SCRIPT_DIR.parent / "schemas" / "QA_CORE_SYSTEM_STACK.v1.schema.json"
HASHES_PATH = SCRIPT_DIR / "expected_hashes.json"
FIXTURES_DIR = SCRIPT_DIR / "fixtures"

VALID_LAYERS = {"AGENT_OS", "MODEL_SUBSTRATE", "PHYSICAL_SUBSTRATE"}


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of file."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


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

            # Validate layer is valid
            layer = witness.get("qa_mapping", {}).get("layer")
            if layer and layer not in VALID_LAYERS:
                errors.append(f"Invalid layer '{layer}' in {filename}")
        except Exception as e:
            errors.append(f"Error loading {filename}: {e}")

    return len(errors) == 0, errors


def validate_cross_invariants(cert: Dict) -> Tuple[bool, List[str]]:
    """Validate cross-layer invariants are properly structured."""
    errors = []

    for ci in cert.get("cross_invariants", []):
        ci_id = ci.get("id", "UNKNOWN")
        layers = ci.get("layers", [])

        if not ci.get("statement"):
            errors.append(f"Cross-invariant {ci_id} missing statement")

        for layer in layers:
            if layer not in VALID_LAYERS:
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


def validate_drift() -> Tuple[bool, List[str]]:
    """Check for drift against expected hashes."""
    errors = []

    if not HASHES_PATH.exists():
        return True, []  # Skip if no hashes file

    expected = load_json(HASHES_PATH)

    # Check cert hash
    if CERT_PATH.exists():
        actual = sha256_file(CERT_PATH)
        expected_hash = expected.get("QA_CORE_SYSTEM_STACK.v1.json")
        if expected_hash and actual != expected_hash:
            errors.append(f"DRIFT: QA_CORE_SYSTEM_STACK.v1.json hash changed")

    # Check schema hash
    if SCHEMA_PATH.exists():
        actual = sha256_file(SCHEMA_PATH)
        expected_hash = expected.get("schema")
        if expected_hash and actual != expected_hash:
            errors.append(f"DRIFT: schema hash changed")

    # Check witness hashes
    expected_witnesses = expected.get("witnesses", {})
    for filename, expected_hash in expected_witnesses.items():
        witness_path = WITNESS_DIR / filename
        if witness_path.exists():
            actual = sha256_file(witness_path)
            if actual != expected_hash:
                errors.append(f"DRIFT: {filename} hash changed")

    return len(errors) == 0, errors


def run_golden_fixtures() -> Tuple[bool, List[str]]:
    """Run golden fixture tests."""
    errors = []

    if not FIXTURES_DIR.exists():
        return True, []  # Skip if no fixtures

    for fixture_path in FIXTURES_DIR.glob("golden_fail_*.json"):
        fixture = load_json(fixture_path)

        # This fixture should fail validation
        layer = fixture.get("qa_mapping", {}).get("layer")
        if layer in VALID_LAYERS:
            errors.append(f"FIXTURE FAIL: {fixture_path.name} should have invalid layer")
        # If layer is invalid, that's expected - test passes

    return len(errors) == 0, errors


def run_validation() -> Tuple[bool, Dict]:
    """Run all validations and return results."""
    results = {
        "cert_exists": False,
        "schema_valid": False,
        "witnesses_valid": False,
        "cross_invariants_valid": False,
        "layers_valid": False,
        "drift_check": False,
        "fixtures_pass": False,
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

    # Drift detection
    ok, errs = validate_drift()
    results["drift_check"] = ok
    results["errors"].extend(errs)

    # Golden fixtures
    ok, errs = run_golden_fixtures()
    results["fixtures_pass"] = ok
    results["errors"].extend(errs)

    all_valid = all([
        results["cert_exists"],
        results["schema_valid"],
        results["witnesses_valid"],
        results["cross_invariants_valid"],
        results["layers_valid"],
        results["drift_check"],
        results["fixtures_pass"],
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
    print(f"Drift check:            {'PASS' if results['drift_check'] else 'FAIL'}")
    print(f"Golden fixtures:        {'PASS' if results['fixtures_pass'] else 'FAIL'}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  - {err}")

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if valid else 'FAIL'}")
    print(f"{'=' * 60}")

    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
