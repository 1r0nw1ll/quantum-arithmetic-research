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

def canonical_json(obj: Dict[str, Any], indent: int = 2) -> str:
    """
    Produce deterministic JSON with sorted keys.

    This is the canonical serialization for all certificates.
    Hashing this output yields the certificate hash.
    """
    return json.dumps(obj, sort_keys=True, indent=indent, ensure_ascii=True)


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


def state_hash(label: str) -> str:
    """Hash a state label to a short identifier."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]


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
