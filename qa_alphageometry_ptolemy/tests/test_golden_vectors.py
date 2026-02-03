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
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path for imports
TESTS_DIR = Path(__file__).parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR.parent))

from qa_alphageometry_ptolemy.qa_cert_core import (
    canonical_json_compact,
    sha256_canonical,
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

    return len(all_errors) == 0, all_errors


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
