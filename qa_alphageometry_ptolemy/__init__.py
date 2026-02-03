"""
qa_alphageometry_ptolemy

QA Certificate Infrastructure package.

Provides:
- qa_cert_core: Shared primitives (canonicalization, hashing, validation)
- qa_meta_validator: Cross-certificate validation
- Module-specific validators (qa_kayser, qa_fst, etc.)
"""

from .qa_cert_core import (
    canonical_json,
    canonical_json_compact,
    sha256_canonical,
    sha256_file,
    certificate_hash,
    full_hash,
    check_manifest_integrity,
    ValidationResult,
)

__all__ = [
    "canonical_json",
    "canonical_json_compact",
    "sha256_canonical",
    "sha256_file",
    "certificate_hash",
    "full_hash",
    "check_manifest_integrity",
    "ValidationResult",
]
