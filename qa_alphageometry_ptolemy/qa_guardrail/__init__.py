"""
qa_guardrail - QA Guardrail MVP for Agent Security

Provides:
- guard(planned_move, ctx) -> GuardrailResult
- GuardrailContext for specifying active generators, policy, capabilities
- Golden fixtures for regression testing
"""

from .qa_guardrail import (
    guard,
    guard_batch,
    GuardrailContext,
    GuardrailResult,
    make_fail_record,
    validate_fixtures,
    run_self_tests,
    FAIL_TYPES,
    GUARDRAIL_FAIL_TYPES,
    AUTHORIZED_GENERATORS,
)

__all__ = [
    "guard",
    "guard_batch",
    "GuardrailContext",
    "GuardrailResult",
    "make_fail_record",
    "validate_fixtures",
    "run_self_tests",
    "FAIL_TYPES",
    "GUARDRAIL_FAIL_TYPES",
    "AUTHORIZED_GENERATORS",
]
