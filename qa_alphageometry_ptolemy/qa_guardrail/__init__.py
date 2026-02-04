"""
qa_guardrail - QA Guardrail MVP for Agent Security

Provides:
- guard(planned_move, ctx) -> GuardrailResult
- GuardrailContext for specifying active generators, policy, capabilities
- Threat scanning via Gemini patterns (scan_for_threats, verify_ic_cert)
- Golden fixtures for regression testing
"""

from .qa_guardrail import (
    guard,
    guard_batch,
    GuardrailContext,
    GuardrailResult,
    AuditLogger,
    make_fail_record,
    validate_fixtures,
    run_self_tests,
    FAIL_TYPES,
    GUARDRAIL_FAIL_TYPES,
    AUTHORIZED_GENERATORS,
)

from .threat_scanner import (
    ThreatScanner,
    scan_for_threats,
    is_content_safe,
    verify_ic_cert,
    create_verification_receipt,
    verify_receipt,
    patterns_sha256,
    policy_sha256,
    content_sha256,
    get_current_patterns_hash,
    SCANNER_ID,
    SCANNER_VERSION,
    MALICIOUS_PATTERNS,
    MALFORMED_PATTERNS,
    ADVERSARIAL_PATTERNS,
)

__all__ = [
    # Guardrail core
    "guard",
    "guard_batch",
    "GuardrailContext",
    "GuardrailResult",
    "AuditLogger",
    "make_fail_record",
    "validate_fixtures",
    "run_self_tests",
    "FAIL_TYPES",
    "GUARDRAIL_FAIL_TYPES",
    "AUTHORIZED_GENERATORS",
    # Threat scanner (Gemini integration)
    "ThreatScanner",
    "scan_for_threats",
    "is_content_safe",
    "verify_ic_cert",
    "create_verification_receipt",
    "verify_receipt",
    "patterns_sha256",
    "policy_sha256",
    "content_sha256",
    "get_current_patterns_hash",
    "SCANNER_ID",
    "SCANNER_VERSION",
    "MALICIOUS_PATTERNS",
    "MALFORMED_PATTERNS",
    "ADVERSARIAL_PATTERNS",
]
