"""
qa_cert_core.py

Shared primitives for all QA certificate modules.

Used by:
- qa_generator_injection_certificate.py
- qa_diversity_collapse_certificate.py
- qa_generalization_certificate.py
- (future certificate modules)

Provides:
- Exact scalar arithmetic (int/Fraction, no floats)
- Deterministic JSON serialization
- Certificate hashing
- Failure-complete validation framework
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
from fractions import Fraction
import hashlib
import json
from datetime import datetime, timezone


# ============================================================================
# EXACT SCALAR TYPES
# ============================================================================

Scalar = Union[int, Fraction]


def to_scalar(x: Any) -> Scalar:
    """
    Convert to exact scalar.

    Floats are converted via Fraction.limit_denominator to preserve
    exact arithmetic discipline while accepting measured values.
    """
    if isinstance(x, bool):
        raise TypeError("Cannot convert bool to exact scalar")
    if isinstance(x, (int, Fraction)):
        return x
    if isinstance(x, float):
        return Fraction(x).limit_denominator(10**9)
    if isinstance(x, str):
        s = x.strip()
        if "/" in s or "." in s:
            return Fraction(s)
        return int(s)
    raise TypeError(f"Cannot convert {type(x)} to exact scalar (got {x})")


def to_scalar_strict(x: Any) -> Scalar:
    """Strict conversion -- no floats allowed."""
    if isinstance(x, float):
        raise TypeError(f"Float not allowed in strict mode: {x}")
    return to_scalar(x)


def scalar_to_str(x: Scalar) -> str:
    """Convert scalar to deterministic string representation."""
    return str(x)


# ============================================================================
# DETERMINISTIC SERIALIZATION
# ============================================================================
#
# HASH DOMAINS (intentional separation - do not "fix"):
#
# 1. certificate_hash() / full_hash():
#    - Uses canonical_json(indent=None, ensure_ascii=True)
#    - Purpose: Certificate ID hash for tetrad/conjecture certs
#    - Domain: Pretty-canonical with ASCII escaping
#
# 2. sha256_canonical():
#    - Uses canonical_json_compact(separators=(',',':'), ensure_ascii=False)
#    - Purpose: Manifest semantic identity + merkle root inputs
#    - Domain: Compact canonical with UTF-8
#
# These are DIFFERENT by design. Changing one will invalidate existing hashes.
# ============================================================================

def canonical_json(obj: Dict[str, Any], indent: int = 2) -> str:
    """
    Produce deterministic JSON with sorted keys (human-readable).

    This is the canonical serialization for all certificates.
    Hashing this output yields the certificate hash.

    Hash domain: certificate_hash(), full_hash()
    """
    return json.dumps(obj, sort_keys=True, indent=indent, ensure_ascii=True)


def canonical_json_compact(obj: Any) -> str:
    """
    Produce deterministic compact JSON for hashing/manifest purposes.

    Spec (hash_spec.version 1.0):
    - sort_keys=True (deterministic key ordering)
    - separators=(',', ':') (no whitespace)
    - ensure_ascii=False (UTF-8 friendly, stable across systems)

    This is the canonical serialization for:
    - canonical_sha256 in manifests
    - Merkle root computation
    - Semantic identity hashing

    Do NOT use for human-readable output - use canonical_json() instead.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def certificate_hash(cert_dict: Dict[str, Any]) -> str:
    """
    Compute SHA-256 short hash of a certificate's canonical JSON.

    Returns first 16 hex chars (64 bits).
    """
    content = canonical_json(cert_dict, indent=None)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def full_hash(cert_dict: Dict[str, Any]) -> str:
    """Compute full SHA-256 hash of a certificate's canonical JSON."""
    content = canonical_json(cert_dict, indent=None)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_canonical(obj: Any) -> str:
    """
    Compute SHA-256 hash of an object's compact canonical JSON.

    Uses canonical_json_compact() for consistent hashing across modules.
    This is the basis for canonical_sha256 in manifests.
    """
    return hashlib.sha256(canonical_json_compact(obj).encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of file bytes."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def state_hash(label: str) -> str:
    """Hash a state label to a short identifier."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# MANIFEST INTEGRITY CHECK (shared across modules)
# ============================================================================

def check_manifest_integrity(
    manifest: Dict[str, Any],
    cert_dir: str,
    load_cert_fn=None,
    schema_path: str = None
) -> Dict[str, Any]:
    """
    Generic manifest integrity checker for certificate modules.

    Verifies:
    1. Each certificate file exists
    2. File SHA256 matches manifest sha256 (file bytes)
    3. Canonical SHA256 matches manifest canonical_sha256 (semantic identity)
    4. (Optional) Manifest matches JSON Schema if schema_path provided

    Args:
        manifest: Loaded manifest dict with 'certificates' section
        cert_dir: Directory containing certificate files
        load_cert_fn: Optional function to load cert JSON (default: json.load)
        schema_path: Optional path to JSON Schema for manifest validation

    Returns:
        Dict with 'ok', 'checks', 'errors' fields
    """
    import os

    if load_cert_fn is None:
        def load_cert_fn(path):
            with open(path) as f:
                return json.load(f)

    results = {"ok": True, "checks": [], "errors": []}

    # Optional JSON Schema validation
    if schema_path:
        try:
            import jsonschema
            with open(schema_path) as f:
                schema = json.load(f)
            jsonschema.validate(manifest, schema)
            results["checks"].append("schema: OK (manifest validates)")
        except ImportError:
            results["checks"].append("schema: WARN - jsonschema not installed, skipping")
        except Exception as e:
            results["errors"].append(f"schema: FAIL - {e}")
            results["ok"] = False

    # Check hash_spec is present
    hash_spec = manifest.get("hash_spec", {})
    hash_spec_id = hash_spec.get("id", "unknown")
    results["hash_spec_id"] = hash_spec_id

    # Check each certificate
    for name, entry in manifest.get("certificates", {}).items():
        cert_file = entry.get("file")
        if not cert_file:
            results["errors"].append(f"{name}: missing 'file' in manifest")
            results["ok"] = False
            continue

        cert_path = os.path.join(cert_dir, cert_file)
        if not os.path.exists(cert_path):
            results["errors"].append(f"{name}: file not found: {cert_file}")
            results["ok"] = False
            continue

        # Check file bytes SHA256
        manifest_sha = entry.get("sha256")
        if manifest_sha:
            actual_sha = sha256_file(cert_path)
            if actual_sha == manifest_sha:
                results["checks"].append(f"{name}: OK (file sha256)")
            else:
                results["errors"].append(
                    f"{name}: FILE SHA256 MISMATCH\n"
                    f"  manifest: {manifest_sha[:16]}...\n"
                    f"  actual:   {actual_sha[:16]}..."
                )
                results["ok"] = False
        else:
            results["checks"].append(f"{name}: WARN - no sha256 in manifest")

        # Check canonical JSON SHA256
        manifest_canonical = entry.get("canonical_sha256")
        if manifest_canonical:
            try:
                cert = load_cert_fn(cert_path)
                actual_canonical = sha256_canonical(cert)
                if actual_canonical == manifest_canonical:
                    results["checks"].append(f"{name}: OK (canonical sha256)")
                else:
                    results["errors"].append(
                        f"{name}: CANONICAL SHA256 MISMATCH\n"
                        f"  manifest: {manifest_canonical[:16]}...\n"
                        f"  actual:   {actual_canonical[:16]}..."
                    )
                    results["ok"] = False
            except Exception as e:
                results["errors"].append(f"{name}: failed to load cert: {e}")
                results["ok"] = False
        else:
            results["checks"].append(f"{name}: WARN - no canonical_sha256 in manifest")

    return results


# ============================================================================
# TIMESTAMP
# ============================================================================

def utc_now_iso() -> str:
    """Current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def cert_id(prefix: str) -> str:
    """Generate a certificate ID with prefix and timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{ts}"


# ============================================================================
# FAILURE-COMPLETE VALIDATION
# ============================================================================

class ValidationResult:
    """
    Result of certificate validation.

    Always returns a structured verdict:
    - is_valid: bool
    - issues: list of human-readable issue descriptions

    This is failure-complete: there is never a silent failure.
    """

    def __init__(self):
        self._issues: List[str] = []

    def check(self, condition: bool, issue_if_false: str) -> None:
        """Add issue if condition is False."""
        if not condition:
            self._issues.append(issue_if_false)

    def fail(self, issue: str) -> None:
        """Unconditionally add an issue."""
        self._issues.append(issue)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> List[str]:
        return list(self._issues)

    def as_tuple(self) -> Tuple[bool, List[str]]:
        return self.is_valid, self.issues

    def __repr__(self) -> str:
        if self.is_valid:
            return "ValidationResult(VALID)"
        return f"ValidationResult(INVALID, {len(self._issues)} issues)"


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    # Test scalar conversion
    assert to_scalar(42) == 42
    assert to_scalar("3/7") == Fraction(3, 7)
    assert to_scalar(0.5) == Fraction(1, 2)
    assert isinstance(to_scalar(3.14159), Fraction)

    try:
        to_scalar_strict(3.14)
        assert False, "Should have raised"
    except TypeError:
        pass

    # Test hashing
    d = {"a": 1, "b": "hello"}
    h1 = certificate_hash(d)
    h2 = certificate_hash(d)
    assert h1 == h2, "Deterministic hash failed"
    assert len(h1) == 16

    # Test canonical_json_compact
    compact = canonical_json_compact(d)
    assert compact == '{"a":1,"b":"hello"}', f"Expected compact, got: {compact}"
    assert " " not in compact, "Compact JSON should have no spaces"

    # Test sha256_canonical
    h_canon = sha256_canonical(d)
    assert len(h_canon) == 64, "Full hash should be 64 hex chars"

    # Test validation
    v = ValidationResult()
    v.check(True, "should not appear")
    v.check(False, "this is an issue")
    assert not v.is_valid
    assert len(v.issues) == 1
    assert v.issues[0] == "this is an issue"

    v2 = ValidationResult()
    v2.check(True, "nope")
    assert v2.is_valid

    # Test cert_id
    cid = cert_id("TEST")
    assert cid.startswith("TEST-")
    assert len(cid) == 5 + 14  # prefix + dash + YYYYMMDDHHMMSS

    print("qa_cert_core: all self-tests passed")
