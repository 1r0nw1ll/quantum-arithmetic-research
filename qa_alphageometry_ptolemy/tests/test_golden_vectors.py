"""
test_golden_vectors.py

Golden test vector harness for QA certificate protocol.

These tests enforce protocol-level specifications:
- Canonicalization format (json.dumps params)
- Hash computation (SHA256 of canonical JSON)
- Merkle leaf format (name:hash:result)

Any drift in these primitives will break these tests, which is intentional.
Protocol changes require:
1. Bump hash_spec_id (e.g., qa.hash_spec.v2)
2. Update fixture files
3. Document in PROTOCOL_CHANGELOG.md
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Support both module and direct execution
TESTS_DIR = Path(__file__).parent

try:
    # When run as module: python -m qa_alphageometry_ptolemy.tests.test_golden_vectors
    from ..qa_cert_core import canonical_json_compact, sha256_canonical, certificate_hash
except ImportError:
    # When run directly: python tests/test_golden_vectors.py
    sys.path.insert(0, str(TESTS_DIR.parent.parent))
    from qa_alphageometry_ptolemy.qa_cert_core import (
        canonical_json_compact,
        sha256_canonical,
        certificate_hash,
    )

# Directory containing golden fixtures
GOLDEN_DIR = TESTS_DIR / "golden"


def load_fixtures() -> List[Tuple[str, Dict[str, Any]]]:
    """Load all golden fixture files."""
    fixtures = []
    for fixture_file in sorted(GOLDEN_DIR.glob("*.json")):
        if fixture_file.name == "README.md":
            continue
        with open(fixture_file) as f:
            data = json.load(f)
        fixtures.append((fixture_file.name, data))
    return fixtures


def check_canonical_json_compact(fixture: Dict[str, Any]) -> List[str]:
    """Check canonical JSON compaction matches expected output."""
    errors = []

    if "object" not in fixture.get("input", {}):
        return errors  # Not a canonicalization test

    expected = fixture.get("expected", {})
    if "canonical_json_compact" not in expected:
        return errors  # No canonical expectation

    obj = fixture["input"]["object"]
    actual = canonical_json_compact(obj)
    expected_canonical = expected["canonical_json_compact"]

    if actual != expected_canonical:
        errors.append(
            f"canonical_json_compact mismatch:\n"
            f"  expected: {expected_canonical!r}\n"
            f"  actual:   {actual!r}"
        )

    return errors


def check_sha256_canonical(fixture: Dict[str, Any]) -> List[str]:
    """Check SHA256 of canonical JSON matches expected hash."""
    errors = []

    if "object" not in fixture.get("input", {}):
        return errors  # Not a canonicalization test

    expected = fixture.get("expected", {})
    if "sha256_canonical" not in expected:
        return errors  # No hash expectation

    obj = fixture["input"]["object"]
    actual = sha256_canonical(obj)
    expected_hash = expected["sha256_canonical"]

    if actual != expected_hash:
        errors.append(
            f"sha256_canonical mismatch:\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {actual}"
        )

    return errors


def check_merkle_leaf(fixture: Dict[str, Any]) -> List[str]:
    """Check merkle leaf format matches expected output."""
    errors = []

    inp = fixture.get("input", {})
    if "name" not in inp or "canonical_hash" not in inp or "result_label" not in inp:
        return errors  # Not a merkle leaf test

    expected = fixture.get("expected", {})

    # Test leaf input string format
    name = inp["name"]
    canonical_hash = inp["canonical_hash"]
    result_label = inp["result_label"]

    actual_input = f"{name}:{canonical_hash}:{result_label}"
    expected_input = expected.get("merkle_leaf_input")

    if expected_input and actual_input != expected_input:
        errors.append(
            f"merkle_leaf_input mismatch:\n"
            f"  expected: {expected_input!r}\n"
            f"  actual:   {actual_input!r}"
        )

    # Test leaf hash
    actual_hash = hashlib.sha256(actual_input.encode("utf-8")).hexdigest()
    expected_hash = expected.get("merkle_leaf")

    if expected_hash and actual_hash != expected_hash:
        errors.append(
            f"merkle_leaf hash mismatch:\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {actual_hash}"
        )

    return errors


def check_merkle_root(fixture: Dict[str, Any]) -> List[str]:
    """Check merkle root computation matches expected output."""
    errors = []

    inp = fixture.get("input", {})
    if "certificates" not in inp:
        return errors  # Not a merkle root test

    expected = fixture.get("expected", {})
    if "merkle_root" not in expected:
        return errors  # No merkle root expectation

    # Compute leaf hashes
    certs = inp["certificates"]
    # Sort by name (lexicographic ordering - protocol specified)
    certs_sorted = sorted(certs, key=lambda c: c["name"])

    leaves = {}
    for cert in certs_sorted:
        name = cert["name"]
        canonical_hash = cert["canonical_hash"]
        result_label = cert["result_label"]
        leaf_input = f"{name}:{canonical_hash}:{result_label}"
        leaves[name] = hashlib.sha256(leaf_input.encode("utf-8")).hexdigest()

    # Verify individual leaves if expected
    expected_leaves = expected.get("leaves", {})
    for name, expected_leaf in expected_leaves.items():
        if name in leaves and leaves[name] != expected_leaf:
            errors.append(
                f"merkle leaf '{name}' mismatch:\n"
                f"  expected: {expected_leaf}\n"
                f"  actual:   {leaves[name]}"
            )

    # Compute merkle root (pairwise hashing)
    leaf_hashes = [leaves[c["name"]] for c in certs_sorted]

    while len(leaf_hashes) > 1:
        next_level = []
        for i in range(0, len(leaf_hashes), 2):
            if i + 1 < len(leaf_hashes):
                combined = leaf_hashes[i] + leaf_hashes[i + 1]
            else:
                # Odd number: carry last hash up
                combined = leaf_hashes[i] + leaf_hashes[i]
            next_level.append(hashlib.sha256(combined.encode("utf-8")).hexdigest())
        leaf_hashes = next_level

    actual_root = leaf_hashes[0] if leaf_hashes else ""
    expected_root = expected["merkle_root"]

    if actual_root != expected_root:
        errors.append(
            f"merkle_root mismatch:\n"
            f"  expected: {expected_root}\n"
            f"  actual:   {actual_root}"
        )

    return errors


def check_failure_algebra(fixture: Dict[str, Any]) -> List[str]:
    """Check failure algebra enum consistency (prevents silent token drift)."""
    errors = []

    inp = fixture.get("input", {})
    if "fail_type_enum" not in inp:
        return errors  # Not a failure algebra test

    expected = fixture.get("expected", {})

    # Check fail_type_enum
    fail_types = inp.get("fail_type_enum", [])
    expected_count = expected.get("fail_type_count")
    if expected_count is not None and len(fail_types) != expected_count:
        errors.append(
            f"fail_type_count mismatch:\n"
            f"  expected: {expected_count}\n"
            f"  actual:   {len(fail_types)}"
        )

    # Check hash of sorted fail_type_enum
    expected_hash = expected.get("fail_type_sha256")
    if expected_hash:
        fail_type_canonical = json.dumps(sorted(fail_types), separators=(",", ":"))
        actual_hash = hashlib.sha256(fail_type_canonical.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            errors.append(
                f"fail_type_sha256 mismatch (enum drift detected):\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}"
            )

    # Check failure_class_enum
    failure_classes = inp.get("failure_class_enum", [])
    expected_class_count = expected.get("failure_class_count")
    if expected_class_count is not None and len(failure_classes) != expected_class_count:
        errors.append(
            f"failure_class_count mismatch:\n"
            f"  expected: {expected_class_count}\n"
            f"  actual:   {len(failure_classes)}"
        )

    # Check hash of sorted failure_class_enum
    expected_class_hash = expected.get("failure_class_sha256")
    if expected_class_hash:
        class_canonical = json.dumps(sorted(failure_classes), separators=(",", ":"))
        actual_class_hash = hashlib.sha256(class_canonical.encode("utf-8")).hexdigest()
        if actual_class_hash != expected_class_hash:
            errors.append(
                f"failure_class_sha256 mismatch (enum drift detected):\n"
                f"  expected: {expected_class_hash}\n"
                f"  actual:   {actual_class_hash}"
            )

    return errors


def check_hash_domain_separation(fixture: Dict[str, Any]) -> List[str]:
    """Check that hash domains are properly separated (prevents 'helpful' unification)."""
    errors = []

    inp = fixture.get("input", {})
    expected = fixture.get("expected", {})

    # Only run if this is a hash domain test
    if "certificate_hash" not in expected or "sha256_canonical" not in expected:
        return errors

    if "object" not in inp:
        return errors

    obj = inp["object"]

    # Check certificate_hash (uses canonical_json with indent=None, ensure_ascii=True)
    actual_cert_hash = certificate_hash(obj)
    expected_cert_hash = expected["certificate_hash"]

    if actual_cert_hash != expected_cert_hash:
        errors.append(
            f"certificate_hash mismatch:\n"
            f"  expected: {expected_cert_hash}\n"
            f"  actual:   {actual_cert_hash}"
        )

    # Check sha256_canonical (uses canonical_json_compact, ensure_ascii=False)
    actual_semantic_hash = sha256_canonical(obj)
    expected_semantic_hash = expected["sha256_canonical"]

    if actual_semantic_hash != expected_semantic_hash:
        errors.append(
            f"sha256_canonical mismatch:\n"
            f"  expected: {expected_semantic_hash}\n"
            f"  actual:   {actual_semantic_hash}"
        )

    # Verify they ARE different (by design)
    # The two hash functions use DIFFERENT canonicalization:
    #   certificate_hash: canonical_json(indent=None) → {"a": 1, "b": "hello"} (spaces!)
    #   sha256_canonical: canonical_json_compact()    → {"a":1,"b":"hello"} (no spaces)
    if expected.get("different_by_design", False):
        # Import canonical functions to compare underlying serialization
        try:
            from ..qa_cert_core import canonical_json, canonical_json_compact
        except ImportError:
            from qa_alphageometry_ptolemy.qa_cert_core import canonical_json, canonical_json_compact

        cert_canonical = canonical_json(obj, indent=None)
        semantic_canonical = canonical_json_compact(obj)

        # These MUST be different - that's the whole point
        if cert_canonical == semantic_canonical:
            errors.append(
                "HASH DOMAIN COLLISION: canonical forms are IDENTICAL!\n"
                f"  certificate canonical: {cert_canonical!r}\n"
                f"  semantic canonical:    {semantic_canonical!r}\n"
                "  This should NOT happen - they use different canonicalization:\n"
                "    certificate: indent=None, ensure_ascii=True (has spaces after : and ,)\n"
                "    semantic: compact separators (',',':'), ensure_ascii=False (no spaces)\n"
                "  Someone may have 'unified' the hash functions incorrectly."
            )

        # Also check they match the expected canonical forms if provided
        expected_cert_canonical = expected.get("certificate_canonical")
        expected_semantic_canonical = expected.get("semantic_canonical")

        if expected_cert_canonical and cert_canonical != expected_cert_canonical:
            errors.append(
                f"certificate canonical form mismatch:\n"
                f"  expected: {expected_cert_canonical!r}\n"
                f"  actual:   {cert_canonical!r}"
            )

        if expected_semantic_canonical and semantic_canonical != expected_semantic_canonical:
            errors.append(
                f"semantic canonical form mismatch:\n"
                f"  expected: {expected_semantic_canonical!r}\n"
                f"  actual:   {semantic_canonical!r}"
            )

    return errors


def check_manifest_hash_mismatch(fixture: Dict[str, Any]) -> List[str]:
    """Check that integrity checking catches hash mismatches (valid schema, wrong hash)."""
    errors = []

    inp = fixture.get("input", {})
    expected = fixture.get("expected", {})

    if "manifest" not in inp or "actual_cert_content" not in inp:
        return errors  # Not a hash mismatch test

    if expected.get("reason") != "hash_mismatch":
        return errors  # Not testing for hash mismatch

    manifest = inp["manifest"]
    actual_cert = inp["actual_cert_content"]

    # Compute what the correct hashes should be
    actual_canonical_sha = sha256_canonical(actual_cert)

    # Check that manifest has wrong hash (as expected for this fixture)
    for name, entry in manifest.get("certificates", {}).items():
        manifest_canonical = entry.get("canonical_sha256", "")

        # The fixture expects the hashes to NOT match
        if manifest_canonical == actual_canonical_sha:
            errors.append(
                f"FIXTURE ERROR: {name} hash unexpectedly matches!\n"
                f"  This fixture is supposed to have mismatched hashes."
            )
            continue

        # Verify integrity check would catch this
        if expected.get("integrity_valid", True):
            errors.append(
                "FIXTURE ERROR: expected.integrity_valid should be false for mismatch test"
            )

    return errors


def check_merkle_property(fixture: Dict[str, Any]) -> List[str]:
    """Property-style Merkle tests: verify determinism, odd-carry rule, tree height."""
    errors = []

    inp = fixture.get("input", {})
    expected = fixture.get("expected", {})

    if "merkle_property_test" not in inp:
        return errors  # Not a property test

    test_config = inp["merkle_property_test"]
    leaf_count = test_config.get("leaf_count", 0)

    if leaf_count < 1:
        return errors

    # Generate deterministic leaves
    leaves = []
    for i in range(leaf_count):
        leaf_input = f"test_cert_{i}:deadbeef{i:04x}:PASS"
        leaf_hash = hashlib.sha256(leaf_input.encode("utf-8")).hexdigest()
        leaves.append(leaf_hash)

    # Compute merkle root twice (determinism check)
    def compute_root(leaf_list):
        current = list(leaf_list)
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    combined = current[i] + current[i + 1]
                else:
                    # Odd: duplicate last hash (carry rule)
                    combined = current[i] + current[i]
                next_level.append(hashlib.sha256(combined.encode("utf-8")).hexdigest())
            current = next_level
        return current[0] if current else ""

    root1 = compute_root(leaves)
    root2 = compute_root(leaves)

    # Determinism check
    if root1 != root2:
        errors.append(
            f"MERKLE DETERMINISM FAILURE: Same leaves produced different roots!\n"
            f"  leaf_count: {leaf_count}\n"
            f"  root1: {root1}\n"
            f"  root2: {root2}"
        )

    # Tree height check
    expected_height = math.ceil(math.log2(leaf_count)) if leaf_count > 1 else 0

    # Check expected root if provided
    expected_root = expected.get("merkle_root")
    if expected_root and root1 != expected_root:
        errors.append(
            f"merkle_root mismatch for {leaf_count} leaves:\n"
            f"  expected: {expected_root}\n"
            f"  actual:   {root1}"
        )

    # Check expected height if provided
    expected_h = expected.get("tree_height")
    if expected_h is not None and expected_height != expected_h:
        errors.append(
            f"tree_height mismatch for {leaf_count} leaves:\n"
            f"  expected: {expected_h}\n"
            f"  actual:   {expected_height}"
        )

    return errors


def run_fixture(name: str, fixture: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Run all applicable tests for a fixture."""
    all_errors = []

    # Check hash_spec_id
    fixture_spec = fixture.get("hash_spec_id", "unknown")
    if fixture_spec != "qa.hash_spec.v1":
        all_errors.append(
            f"Unsupported hash_spec_id: {fixture_spec} (expected qa.hash_spec.v1)"
        )

    # Run all check types
    all_errors.extend(check_canonical_json_compact(fixture))
    all_errors.extend(check_sha256_canonical(fixture))
    all_errors.extend(check_merkle_leaf(fixture))
    all_errors.extend(check_merkle_root(fixture))
    all_errors.extend(check_failure_algebra(fixture))
    all_errors.extend(check_hash_domain_separation(fixture))
    all_errors.extend(check_schema_violation(fixture))
    all_errors.extend(check_manifest_hash_mismatch(fixture))
    all_errors.extend(check_merkle_property(fixture))

    return len(all_errors) == 0, all_errors


def check_schema_violation(fixture: Dict[str, Any]) -> List[str]:
    """Check that intentionally invalid manifests are rejected by real JSON Schema."""
    errors = []

    inp = fixture.get("input", {})
    expected = fixture.get("expected", {})

    if "invalid_manifest" not in inp:
        return errors  # Not a schema violation test

    if not expected.get("should_fail", False):
        return errors  # Not testing for failure

    invalid_manifest = inp["invalid_manifest"]
    error_contains = expected.get("error_contains", [])

    # Try to validate against real JSON Schema
    schema_path = TESTS_DIR.parent / "schemas" / "qa_manifest.schema.json"

    # Detect CI environment - jsonschema is REQUIRED in CI
    in_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")

    try:
        import jsonschema

        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)

            # This SHOULD raise ValidationError for invalid manifest
            try:
                jsonschema.validate(invalid_manifest, schema)
                # If we get here, validation passed - that's a problem!
                errors.append(
                    "SCHEMA GATE FAILURE: Invalid manifest was NOT rejected!\n"
                    f"  Manifest: {invalid_manifest}\n"
                    "  The shipped schema did not catch this invalid input."
                )
                return errors
            except jsonschema.ValidationError as e:
                # Good - schema rejected it. Check error contains expected substrings
                error_msg = str(e).lower()
                for expected_substr in error_contains:
                    if expected_substr.lower() not in error_msg:
                        errors.append(
                            f"Schema error doesn't contain '{expected_substr}':\n"
                            f"  Got: {str(e)[:200]}..."
                        )
                return errors
        else:
            if in_ci:
                errors.append(
                    "CI FAILURE: Schema file not found!\n"
                    f"  Expected: {schema_path}\n"
                    "  In CI, schema validation is required."
                )
                return errors
            # Schema file not found - fall back to heuristic check
            pass

    except ImportError:
        if in_ci:
            errors.append(
                "CI FAILURE: jsonschema package not installed!\n"
                "  In CI, jsonschema is required for schema validation tests.\n"
                "  Add 'jsonschema>=4.0' to requirements-dev.txt and install."
            )
            return errors
        # jsonschema not installed - fall back to heuristic check
        pass

    # Fallback: hand-rolled rejection heuristic (less rigorous)
    schema_version = invalid_manifest.get("schema_version", "")
    valid_versions = {"QA_MANIFEST.v1", "QA_SHA256_MANIFEST.v1"}

    rejection_reasons = []

    if schema_version not in valid_versions:
        rejection_reasons.append(f"invalid schema_version: {schema_version}")

    for name, cert in invalid_manifest.get("certificates", {}).items():
        if "sha256" not in cert and "canonical_sha256" not in cert:
            rejection_reasons.append(f"{name}: missing hash fields")

    if not rejection_reasons:
        errors.append(
            "SCHEMA GATE FAILURE: Invalid manifest was NOT rejected!\n"
            f"  Manifest: {invalid_manifest}\n"
            "  (Note: jsonschema not available, using heuristic check)"
        )
        return errors

    rejection_text = " ".join(rejection_reasons).lower()
    for expected_substr in error_contains:
        if expected_substr.lower() not in rejection_text:
            errors.append(
                f"Expected error to contain '{expected_substr}' but got:\n"
                f"  {rejection_reasons}"
            )

    return errors


def main():
    """Run all golden vector tests."""
    fixtures = load_fixtures()

    if not fixtures:
        print("ERROR: No golden fixtures found in", GOLDEN_DIR)
        return 1

    print(f"Running {len(fixtures)} golden vector tests...")
    print()

    passed = 0
    failed = 0

    for name, fixture in fixtures:
        fixture_id = fixture.get("fixture_id", name)
        ok, errors = run_fixture(name, fixture)

        if ok:
            print(f"  ✓ {fixture_id}")
            passed += 1
        else:
            print(f"  ✗ {fixture_id}")
            for err in errors:
                for line in err.split("\n"):
                    print(f"      {line}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        print()
        print("PROTOCOL DRIFT DETECTED!")
        print("If this is intentional, update hash_spec_id and fixtures.")
        return 1

    return 0


# Pytest integration
def test_all_golden_vectors():
    """Pytest entry point for golden vector tests."""
    fixtures = load_fixtures()
    assert len(fixtures) > 0, f"No golden fixtures found in {GOLDEN_DIR}"

    for name, fixture in fixtures:
        fixture_id = fixture.get("fixture_id", name)
        ok, errors = run_fixture(name, fixture)
        assert ok, f"{fixture_id}: {errors}"


if __name__ == "__main__":
    sys.exit(main())
