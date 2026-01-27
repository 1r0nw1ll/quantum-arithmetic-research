"""
qa_meta_validator.py

Cross-certificate meta-validator for the QA Certificate Tetrad + Conjectures.

Accepts any certificate JSON (Injection, Collapse, Field, Beyond Neurons,
or Conjecture),
dispatches to the correct validator, and enforces the shared
failure-complete contract:

    Every validation produces:
        success: bool
        OR {fail_type, invariant_diff, barrier} structured failure

Uses qa_cert_core exclusively for shared operations.
"""

from __future__ import annotations

import json
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qa_cert_core import (
    canonical_json, certificate_hash, full_hash,
    ValidationResult, utc_now_iso, cert_id,
)


# ============================================================================
# CERTIFICATE TYPE REGISTRY
# ============================================================================

KNOWN_CERT_TYPES = {
    "GENERATOR_INJECTION",
    "DIVERSITY_COLLAPSE_OBSTRUCTION",
    "FIELD_COMPUTATION_CERT",
    "BEYOND_NEURONS_INTELLIGENCE_CERT",
    "QA_CONJECTURE",
}

# Allowed conjecture types
KNOWN_CONJECTURE_TYPES = {
    "CONJ_SUBSTRATE_INVARIANCE",
    "CONJ_CONSTRAINT_INJECTION_DUALITY",
    "CONJ_HORIZON_HIERARCHY",
    "CONJ_GOAL_COLLAPSE_EQUIVALENCE",
    "CONJ_COMPETENCY_DEPTH_BOUND",
}

# Fields that EVERY certificate must have (structural contract)
REQUIRED_SHARED_FIELDS = {
    "certificate_id",
    "certificate_type",
    "timestamp",
    # Note: result field varies by type ("result" vs "verdict")
}

# Fields by certificate type (beyond shared)
REQUIRED_BY_TYPE = {
    "GENERATOR_INJECTION": {
        "before_generators", "after_generators", "injected_generators",
        "barrier_crossed", "before_witness", "after_witness",
        "invariants_preserved", "invariant_verification_method",
        "result",
    },
    "DIVERSITY_COLLAPSE_OBSTRUCTION": {
        "diversity_invariant", "collapsed_trace", "collapse_witness",
        "missing_generator", "verdict",
    },
    "FIELD_COMPUTATION_CERT": {
        "claim", "task_class", "target_operator", "domain",
        "baseline_generators", "extended_generators", "injected_generators",
        "invariants", "barrier", "before_witness", "after_witness",
        "result",
    },
    "BEYOND_NEURONS_INTELLIGENCE_CERT": {
        "claim", "substrate", "scale", "problem_space",
        "search_efficiency", "result",
    },
    "QA_CONJECTURE": {
        "conjecture_type", "title", "claim",
        "validator_contract", "failure_taxonomy", "status",
    },
}


# ============================================================================
# META-VALIDATION RESULT
# ============================================================================

class FailType(Enum):
    """Structured failure types for meta-validation."""
    UNKNOWN_CERT_TYPE = "unknown_cert_type"
    MISSING_FIELD = "missing_field"
    INVALID_JSON = "invalid_json"
    HASH_MISMATCH = "hash_mismatch"
    STRUCTURAL_ERROR = "structural_error"
    TYPE_SPECIFIC_ERROR = "type_specific_error"


@dataclass
class MetaValidationResult:
    """
    Full meta-validation output.

    Always structured. Never silent.
    On failure: fail_type + invariant_diff + barrier description.
    """
    certificate_id: str
    certificate_type: str
    is_valid: bool
    content_hash: str
    content_hash_full: str

    # Failure details (populated only if invalid)
    fail_type: Optional[FailType] = None
    issues: Optional[List[str]] = None
    invariant_diff: Optional[Dict[str, Any]] = None
    barrier: Optional[str] = None

    # Type-specific validation (from delegated validator)
    type_validation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "certificate_id": self.certificate_id,
            "certificate_type": self.certificate_type,
            "is_valid": self.is_valid,
            "content_hash": self.content_hash,
            "content_hash_full": self.content_hash_full,
        }
        if not self.is_valid:
            d["fail_type"] = self.fail_type.value if self.fail_type else None
            d["issues"] = self.issues or []
            d["invariant_diff"] = self.invariant_diff
            d["barrier"] = self.barrier
        if self.type_validation:
            d["type_validation"] = self.type_validation
        return d

    def to_json(self, indent: int = 2) -> str:
        return canonical_json(self.to_dict(), indent=indent)


# ============================================================================
# STRUCTURAL VALIDATOR (shared across all types)
# ============================================================================

def validate_structure(cert_dict: Dict[str, Any]) -> ValidationResult:
    """
    Validate shared structural requirements for any certificate.
    """
    v = ValidationResult()

    # Check shared required fields
    for field in REQUIRED_SHARED_FIELDS:
        v.check(field in cert_dict, f"Missing required field: {field}")

    # Check certificate type is known
    cert_type = cert_dict.get("certificate_type", "")
    v.check(cert_type in KNOWN_CERT_TYPES, f"Unknown certificate_type: {cert_type}")

    # Check type-specific required fields
    if cert_type in REQUIRED_BY_TYPE:
        for field in REQUIRED_BY_TYPE[cert_type]:
            v.check(field in cert_dict, f"Missing {cert_type} field: {field}")

    # Check certificate_id is non-empty
    v.check(bool(cert_dict.get("certificate_id")), "certificate_id is empty")

    # Check timestamp is non-empty
    v.check(bool(cert_dict.get("timestamp")), "timestamp is empty")

    return v


# ============================================================================
# TYPE-SPECIFIC DELEGATION
# ============================================================================

def validate_injection(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate GENERATOR_INJECTION-specific semantics."""
    v = ValidationResult()

    before = cert_dict.get("before_generators", {})
    after = cert_dict.get("after_generators", {})
    injected = cert_dict.get("injected_generators", [])

    # Extract generator names
    before_names = {g.get("name") for g in before.get("generators", [])}
    after_names = {g.get("name") for g in after.get("generators", [])}
    injected_names = {g.get("name") for g in injected}

    # Subset check
    v.check(before_names.issubset(after_names),
            f"before_generators not subset of after: {before_names - after_names}")

    # Injection difference check
    expected = after_names - before_names
    v.check(injected_names == expected,
            f"injected mismatch: got {injected_names}, expected {expected}")

    # Path uses only after generators
    after_witness = cert_dict.get("after_witness", {})
    for step in after_witness.get("path", []):
        gen = step.get("generator")
        v.check(gen in after_names, f"Path uses unknown generator: {gen}")

    # Result consistency
    result = cert_dict.get("result")
    if result == "barrier_crossed":
        path = after_witness.get("path", [])
        v.check(len(path) > 0, "BARRIER_CROSSED but empty path")

    return v


def validate_collapse(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate DIVERSITY_COLLAPSE_OBSTRUCTION-specific semantics."""
    v = ValidationResult()

    witness = cert_dict.get("collapse_witness", {})
    verdict = cert_dict.get("verdict", "")

    if verdict == "collapse_detected":
        # Mode collapse signature check
        is_mode_collapse = witness.get("is_mode_collapse", False)
        v.check(is_mode_collapse,
                "COLLAPSE_DETECTED but witness.is_mode_collapse is False")

        v.check(witness.get("diversity_collapsed", False),
                "COLLAPSE_DETECTED but diversity_collapsed is False")

    # Trace strategy check
    trace = cert_dict.get("collapsed_trace", {})
    strategy = trace.get("strategy", "")
    v.check(strategy in ("rl_policy_gradient", "beam_search"),
            f"Collapsed trace uses non-collapse-susceptible strategy: {strategy}")

    # Preservation trace check (if present)
    preserved = cert_dict.get("preserved_trace")
    if preserved:
        pstrat = preserved.get("strategy", "")
        v.check(pstrat in ("evolutionary", "random_search"),
                f"Preserved trace uses {pstrat}, expected evolutionary/random")

    return v


def validate_field(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate FIELD_COMPUTATION_CERT-specific semantics."""
    v = ValidationResult()

    baseline = cert_dict.get("baseline_generators", {})
    extended = cert_dict.get("extended_generators", {})
    injected = cert_dict.get("injected_generators", [])

    # Generator subset check
    baseline_names = {g.get("name") for g in baseline.get("generators", [])}
    extended_names = {g.get("name") for g in extended.get("generators", [])}
    injected_names = {g.get("name") for g in injected}

    v.check(baseline_names.issubset(extended_names),
            f"Baseline not subset of extended: {baseline_names - extended_names}")

    expected = extended_names - baseline_names
    v.check(injected_names == expected,
            f"Injected mismatch: got {injected_names}, expected {expected}")

    # Trajectory check
    after_witness = cert_dict.get("after_witness", {})
    for step in after_witness.get("trajectory", []):
        gen = step.get("generator")
        v.check(gen in extended_names, f"Trajectory uses unknown generator: {gen}")

    # Operator realization check
    result = cert_dict.get("result")
    if result == "barrier_crossed":
        realized = after_witness.get("operator_realized")
        v.check(realized is True,
                "BARRIER_CROSSED but operator_realized is not True")

    # Domain consistency
    cert_domain = cert_dict.get("domain")
    v.check(baseline.get("domain") == cert_domain,
            "Baseline generator domain doesn't match certificate domain")
    v.check(extended.get("domain") == cert_domain,
            "Extended generator domain doesn't match certificate domain")

    return v


def validate_beyond_neurons(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate BEYOND_NEURONS_INTELLIGENCE_CERT-specific semantics."""
    v = ValidationResult()

    # Search efficiency checks
    se = cert_dict.get("search_efficiency", {})
    tau_blind = se.get("tau_blind")
    tau_agent = se.get("tau_agent")
    K = se.get("K")

    v.check(tau_blind is not None, "search_efficiency missing tau_blind")
    v.check(tau_agent is not None, "search_efficiency missing tau_agent")
    v.check(K is not None, "search_efficiency missing K")

    # Problem space completeness
    ps = cert_dict.get("problem_space", {})
    v.check(bool(ps.get("operators")), "problem_space missing operators")
    v.check(bool(ps.get("constraints")), "problem_space missing constraints")
    v.check(bool(ps.get("evaluation")), "problem_space missing evaluation")
    v.check(ps.get("horizon", 0) > 0, "problem_space horizon must be positive")

    # Result consistency
    result = cert_dict.get("result", "")
    if result == "constraint_edited":
        v.check(cert_dict.get("edited_constraints") is not None,
                "CONSTRAINT_EDITED but no edited_constraints provided")
    if result == "goal_decoupled":
        v.check(cert_dict.get("competency_architecture") is not None,
                "GOAL_DECOUPLED but no competency_architecture provided")

    # If operator injection present, verify subset
    baseline_ops = cert_dict.get("baseline_operators")
    extended_ops = cert_dict.get("extended_operators")
    injected_ops = cert_dict.get("injected_operators")
    if baseline_ops and extended_ops and injected_ops:
        base_names = {o.get("name") for o in baseline_ops.get("operators", [])}
        ext_names = {o.get("name") for o in extended_ops.get("operators", [])}
        inj_names = {o.get("name") for o in injected_ops}
        v.check(base_names.issubset(ext_names),
                f"Baseline operators not subset of extended: {base_names - ext_names}")
        expected = ext_names - base_names
        v.check(inj_names == expected,
                f"Injected mismatch: got {inj_names}, expected {expected}")

    # Competency architecture ordering (if present)
    arch = cert_dict.get("competency_architecture")
    if arch:
        levels = arch.get("levels", [])
        v.check(len(levels) >= 2,
                "Competency architecture needs at least 2 levels")
        for i in range(1, len(levels)):
            v.check(levels[i].get("level_index", 0) > levels[i - 1].get("level_index", 0),
                    f"Competency levels not ordered at index {i}")

    return v


def validate_conjecture(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate QA_CONJECTURE-specific semantics."""
    v = ValidationResult()

    # Conjecture type must be in allowed set
    conj_type = cert_dict.get("conjecture_type", "")
    v.check(conj_type in KNOWN_CONJECTURE_TYPES,
            f"Unknown conjecture_type: {conj_type}")

    # Title must be non-empty
    v.check(bool(cert_dict.get("title")), "title is empty")

    # Claim must have required subfields
    claim = cert_dict.get("claim", {})
    v.check(bool(claim.get("statement")), "claim missing statement")

    # Validator contract checks
    vc = cert_dict.get("validator_contract", {})
    v.check(vc.get("deterministic") is True,
            "validator_contract.deterministic must be true")

    steps = vc.get("steps", [])
    v.check(len(steps) > 0, "validator_contract.steps is empty")
    for i, step in enumerate(steps):
        v.check(bool(step.get("op")),
                f"validator_contract.steps[{i}] missing op")

    # Failure taxonomy checks
    ft = cert_dict.get("failure_taxonomy", [])
    v.check(len(ft) > 0, "failure_taxonomy is empty")
    for i, entry in enumerate(ft):
        v.check(bool(entry.get("fail_type")),
                f"failure_taxonomy[{i}] missing fail_type")
        v.check(isinstance(entry.get("invariant_diff_schema"), dict),
                f"failure_taxonomy[{i}] missing or non-dict invariant_diff_schema")

    # Status must be in allowed set
    status = cert_dict.get("status", "")
    v.check(status in ("open", "supported", "refuted"),
            f"status must be open|supported|refuted, got: {status}")

    return v


TYPE_VALIDATORS = {
    "GENERATOR_INJECTION": validate_injection,
    "DIVERSITY_COLLAPSE_OBSTRUCTION": validate_collapse,
    "FIELD_COMPUTATION_CERT": validate_field,
    "BEYOND_NEURONS_INTELLIGENCE_CERT": validate_beyond_neurons,
    "QA_CONJECTURE": validate_conjecture,
}


# ============================================================================
# META-VALIDATOR ENTRY POINT
# ============================================================================

def validate(cert_dict: Dict[str, Any]) -> MetaValidationResult:
    """
    Validate any QA certificate.

    Performs:
    1. Canonical JSON serialization + hash computation
    2. Structural validation (shared fields)
    3. Type-specific validation (delegated)
    4. Failure-complete output: success OR {fail_type, issues, invariant_diff, barrier}
    """
    # Compute hashes
    short_hash = certificate_hash(cert_dict)
    long_hash = full_hash(cert_dict)

    cert_type = cert_dict.get("certificate_type", "UNKNOWN")
    cert_id_val = cert_dict.get("certificate_id", "UNKNOWN")

    # Structural validation
    struct_result = validate_structure(cert_dict)

    if not struct_result.is_valid:
        return MetaValidationResult(
            certificate_id=cert_id_val,
            certificate_type=cert_type,
            is_valid=False,
            content_hash=short_hash,
            content_hash_full=long_hash,
            fail_type=FailType.STRUCTURAL_ERROR,
            issues=struct_result.issues,
            barrier="Structural validation failed",
        )

    # Type-specific validation
    validator = TYPE_VALIDATORS.get(cert_type)
    if validator is None:
        return MetaValidationResult(
            certificate_id=cert_id_val,
            certificate_type=cert_type,
            is_valid=False,
            content_hash=short_hash,
            content_hash_full=long_hash,
            fail_type=FailType.UNKNOWN_CERT_TYPE,
            issues=[f"No validator for type: {cert_type}"],
            barrier=f"Unknown certificate type: {cert_type}",
        )

    type_result = validator(cert_dict)

    if not type_result.is_valid:
        return MetaValidationResult(
            certificate_id=cert_id_val,
            certificate_type=cert_type,
            is_valid=False,
            content_hash=short_hash,
            content_hash_full=long_hash,
            fail_type=FailType.TYPE_SPECIFIC_ERROR,
            issues=type_result.issues,
            type_validation={
                "validator": cert_type,
                "issue_count": len(type_result.issues),
            },
            barrier="Type-specific validation failed",
        )

    # All checks passed
    return MetaValidationResult(
        certificate_id=cert_id_val,
        certificate_type=cert_type,
        is_valid=True,
        content_hash=short_hash,
        content_hash_full=long_hash,
        type_validation={
            "validator": cert_type,
            "issue_count": 0,
        },
    )


def validate_json(json_str: str) -> MetaValidationResult:
    """Validate from raw JSON string."""
    try:
        cert_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        return MetaValidationResult(
            certificate_id="UNKNOWN",
            certificate_type="UNKNOWN",
            is_valid=False,
            content_hash="",
            content_hash_full="",
            fail_type=FailType.INVALID_JSON,
            issues=[f"JSON parse error: {e}"],
            barrier="Input is not valid JSON",
        )
    return validate(cert_dict)


# ============================================================================
# CLI + SELF-TEST
# ============================================================================

if __name__ == "__main__":
    # If a file path is provided, validate that file
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path) as f:
            result = validate_json(f.read())
        print(result.to_json())
        sys.exit(0 if result.is_valid else 1)

    # Otherwise run self-tests against all four certificate types
    print("=== META-VALIDATOR SELF-TEST (TETRAD + CONJECTURES) ===\n")

    # Import all four certificate modules
    from qa_generator_injection_certificate import create_llm_sandbox_certificate
    from qa_diversity_collapse_certificate import (
        create_collapse_certificate, SearchTrace, PopulationSnapshot,
        SearchStrategy, DiversityMetric, DiversityInvariant,
    )
    from qa_field_computation_certificate import create_wise_desync_certificate
    from qa_beyond_neurons_certificate import (
        create_planaria_certificate, create_cancer_certificate,
        create_non_neural_ai_certificate,
    )

    from fractions import Fraction as F

    results = []

    # --- Test 1: Generator Injection ---
    inj_cert = create_llm_sandbox_certificate(
        before_tools=["text_generation"],
        after_tools=["text_generation", "execute_bash", "file_read", "file_write"],
        task_class="long_context_qa",
        unreachable_evidence="Context exceeds model window",
        reachable_path=[
            ("file_write", "stored"), ("file_read", "loaded"), ("text_generation", "answered")
        ],
        invariants=["valid_response", "task_completion"],
    )
    r1 = validate(inj_cert.to_dict())
    results.append(("GENERATOR_INJECTION", r1))
    print(f"[1] {r1.certificate_type}: valid={r1.is_valid} hash={r1.content_hash}")

    # --- Test 2: Diversity Collapse ---
    rl_snaps = []
    for step in range(11):
        rl_snaps.append(PopulationSnapshot(
            step=step, mean_reward=F(50 + step * 3, 1),
            max_reward=F(70 + min(step, 2), 1),
            min_reward=F(30 + step, 1), reward_std=F(15 - step, 1),
            diversity_value=F(max(100 - step * 12, 5), 1),
            diversity_metric=DiversityMetric.EMBEDDING_DISPERSION,
            population_size=64,
        ))
    rl_trace = SearchTrace(
        strategy=SearchStrategy.RL_POLICY_GRADIENT,
        snapshots=tuple(rl_snaps), total_steps=10,
    )
    invariant = DiversityInvariant(
        metric=DiversityMetric.EMBEDDING_DISPERSION,
        threshold=F(30), collapse_window=3,
        plateau_epsilon=F(2), plateau_window=4,
    )
    col_cert = create_collapse_certificate(
        rl_trace=rl_trace, evo_trace=None, invariant=invariant,
    )
    r2 = validate(col_cert.to_dict())
    results.append(("DIVERSITY_COLLAPSE", r2))
    print(f"[2] {r2.certificate_type}: valid={r2.is_valid} hash={r2.content_hash}")

    # --- Test 3: Field Computation ---
    field_cert = create_wise_desync_certificate()
    r3 = validate(field_cert.to_dict())
    results.append(("FIELD_COMPUTATION", r3))
    print(f"[3] {r3.certificate_type}: valid={r3.is_valid} hash={r3.content_hash}")

    # --- Test 4: Beyond Neurons (Planaria) ---
    planaria_cert = create_planaria_certificate()
    r4 = validate(planaria_cert.to_dict())
    results.append(("BEYOND_NEURONS_PLANARIA", r4))
    print(f"[4] {r4.certificate_type}: valid={r4.is_valid} hash={r4.content_hash}")

    # --- Test 5: Beyond Neurons (Cancer) ---
    cancer_cert = create_cancer_certificate()
    r5 = validate(cancer_cert.to_dict())
    results.append(("BEYOND_NEURONS_CANCER", r5))
    print(f"[5] {r5.certificate_type}: valid={r5.is_valid} hash={r5.content_hash}")

    # --- Test 6: Beyond Neurons (Silicon) ---
    silicon_cert = create_non_neural_ai_certificate()
    r6 = validate(silicon_cert.to_dict())
    results.append(("BEYOND_NEURONS_SILICON", r6))
    print(f"[6] {r6.certificate_type}: valid={r6.is_valid} hash={r6.content_hash}")

    # --- Test 7-9: QA Conjectures (from canonical ledger) ---
    conj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "qa_ledger", "conjectures")
    conj_files = [
        ("CONJ_SUBSTRATE_INVARIANCE", "QA_CONJ__SUBSTRATE_INVARIANCE__v1.json"),
        ("CONJ_HORIZON_HIERARCHY", "QA_CONJ__HORIZON_HIERARCHY__v1.json"),
        ("CONJ_GOAL_COLLAPSE_EQUIVALENCE", "QA_CONJ__GOAL_COLLAPSE_EQUIVALENCE__v1.json"),
    ]
    for i, (label, fname) in enumerate(conj_files, start=7):
        fpath = os.path.join(conj_dir, fname)
        with open(fpath) as f:
            conj_result = validate_json(f.read())
        results.append((label, conj_result))
        print(f"[{i}] {conj_result.certificate_type} ({label}): "
              f"valid={conj_result.is_valid} hash={conj_result.content_hash}")

    # --- Test 10: Invalid certificate (missing fields) ---
    r10 = validate({"certificate_type": "GENERATOR_INJECTION", "certificate_id": "test"})
    results.append(("INVALID_MISSING_FIELDS", r10))
    print(f"[10] INVALID (missing fields): valid={r10.is_valid} fail_type={r10.fail_type}")

    # --- Test 11: Unknown type ---
    r11 = validate({"certificate_type": "UNKNOWN_TYPE", "certificate_id": "x",
                    "timestamp": "now", "result": "ok"})
    results.append(("UNKNOWN_TYPE", r11))
    print(f"[11] UNKNOWN type: valid={r11.is_valid} fail_type={r11.fail_type}")

    # --- Test 12: Bad JSON ---
    r12 = validate_json("{not valid json")
    results.append(("BAD_JSON", r12))
    print(f"[12] BAD JSON: valid={r12.is_valid} fail_type={r12.fail_type}")

    print()
    valid_count = sum(1 for _, r in results if r.is_valid)
    invalid_count = sum(1 for _, r in results if not r.is_valid)
    print(f"Results: {valid_count} valid, {invalid_count} invalid (expected 9 valid, 3 invalid)")

    assert valid_count == 9, f"Expected 9 valid, got {valid_count}"
    assert invalid_count == 3, f"Expected 3 invalid, got {invalid_count}"
    print("\nAll meta-validator self-tests PASSED (TETRAD + CONJECTURES)")
