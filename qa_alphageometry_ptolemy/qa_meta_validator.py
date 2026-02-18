"""
qa_meta_validator.py

Cross-certificate meta-validator for the QA Certificate Tetrad + extensions.

Accepts any certificate JSON (Injection, Collapse, Field, Beyond Neurons,
Topology Resonance, Graph Structure, or Conjecture),
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
import hashlib
import re
from fractions import Fraction
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Ensure sibling modules are importable regardless of execution mode
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

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
    "HOT_MESS_INCOHERENCE_CERT",
    "TOPOLOGY_RESONANCE_CERT",
    "ELLIPTIC_CORRESPONDENCE_CERT",
    "GRAPH_STRUCTURE_CERT",
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
    "HOT_MESS_INCOHERENCE_CERT": {
        "version", "schema", "success",
        "model_id", "task_family", "eval_metric_id",
        "num_runs", "run_outcomes",
        "decomposition_witness", "coherence_invariant",
    },
    "TOPOLOGY_RESONANCE_CERT": {
        "generator_set",
        "success",
    },
    "ELLIPTIC_CORRESPONDENCE_CERT": {
        "generator_set",
        "success",
    },
    "GRAPH_STRUCTURE_CERT": {
        "generator_set",
        "success",
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


def _parse_fraction_like(value: Any) -> Fraction:
    """Parse exact scalar-like values used in lightweight validators."""
    if isinstance(value, bool):
        raise ValueError("bool is not a numeric scalar")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise ValueError(f"unsupported scalar type: {type(value)}")


# ============================================================================
# MAPPING PROTOCOL GATE (Gate 0 for certificate families)
# ============================================================================

class MetaValidationError(Exception):
    """Deterministic meta-validation error with structured fail_type + detail."""

    def __init__(self, fail_type: str, detail: Dict[str, Any]):
        super().__init__(fail_type)
        self.fail_type = fail_type
        self.detail = detail


def _sha256_file_bytes(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_json_against_schema(obj: Dict[str, Any], schema_path: str) -> None:
    try:
        import jsonschema
    except ImportError as e:
        raise MetaValidationError(
            "MAPPING_PROTOCOL_SCHEMA_FAIL",
            {"schema_path": schema_path, "error": f"jsonschema import failed: {e}"},
        )

    schema = _load_json_file(schema_path)
    try:
        jsonschema.validate(instance=obj, schema=schema)
    except Exception as e:
        raise MetaValidationError(
            "MAPPING_PROTOCOL_SCHEMA_FAIL",
            {"schema_path": schema_path, "error": str(e)},
        )


def require_mapping_protocol(family_root: str, repo_root: str) -> Tuple[str, str]:
    """
    Enforces the intake constitution for certificate families:
    - Exactly one of mapping_protocol.json or mapping_protocol_ref.json exists in family_root.
    - Inline must validate as QA_MAPPING_PROTOCOL.v1 and include determinism_contract essentials.
    - Ref must validate as QA_MAPPING_PROTOCOL_REF.v1 and resolve to a valid QA_MAPPING_PROTOCOL.v1 target.

    Returns:
        (mode, resolved_mapping_path) where mode in {"inline","ref"}.
    """
    inline_path = os.path.join(family_root, "mapping_protocol.json")
    ref_path = os.path.join(family_root, "mapping_protocol_ref.json")

    inline_exists = os.path.exists(inline_path)
    ref_exists = os.path.exists(ref_path)

    if inline_exists and ref_exists:
        raise MetaValidationError(
            "MAPPING_PROTOCOL_AMBIGUOUS",
            {"family_root": family_root, "found": ["mapping_protocol.json", "mapping_protocol_ref.json"]},
        )
    if not inline_exists and not ref_exists:
        raise MetaValidationError(
            "MAPPING_PROTOCOL_MISSING",
            {"family_root": family_root, "required_one_of": ["mapping_protocol.json", "mapping_protocol_ref.json"]},
        )

    mapping_schema = os.path.join(repo_root, "qa_mapping_protocol", "schema.json")
    ref_schema = os.path.join(repo_root, "qa_mapping_protocol_ref", "schema.json")

    if inline_exists:
        obj = _load_json_file(inline_path)
        _validate_json_against_schema(obj, mapping_schema)
        dc = obj.get("determinism_contract", {})
        if dc.get("invariant_diff_defined") is not True or not str(dc.get("nondeterminism_proof", "")).strip():
            raise MetaValidationError(
                "MAPPING_PROTOCOL_DETERMINISM_CONTRACT_MISSING",
                {
                    "family_root": family_root,
                    "path": inline_path,
                    "required": [
                        "determinism_contract.invariant_diff_defined==true",
                        "determinism_contract.nondeterminism_proof non-empty",
                    ],
                },
            )
        return "inline", inline_path

    # ref mode
    ref_obj = _load_json_file(ref_path)
    _validate_json_against_schema(ref_obj, ref_schema)
    rel = ref_obj.get("ref_path")
    if not isinstance(rel, str) or not rel.strip():
        raise MetaValidationError(
            "MAPPING_PROTOCOL_REF_INVALID",
            {"family_root": family_root, "path": ref_path, "missing": ["ref_path"]},
        )

    repo_root_abs = os.path.normpath(os.path.abspath(repo_root))
    resolved_abs = os.path.normpath(os.path.abspath(os.path.join(repo_root_abs, rel)))
    if os.path.commonpath([repo_root_abs, resolved_abs]) != repo_root_abs:
        raise MetaValidationError(
            "MAPPING_PROTOCOL_REF_ESCAPE",
            {"family_root": family_root, "ref_path": rel, "resolved": resolved_abs},
        )
    if not os.path.exists(resolved_abs):
        raise MetaValidationError(
            "MAPPING_PROTOCOL_REF_UNRESOLVED",
            {"family_root": family_root, "ref_path": rel, "resolved": resolved_abs},
        )

    if "ref_sha256" in ref_obj:
        want = str(ref_obj.get("ref_sha256", "")).lower().strip()
        got = _sha256_file_bytes(resolved_abs)
        if want != got:
            raise MetaValidationError(
                "MAPPING_PROTOCOL_REF_SHA_MISMATCH",
                {"family_root": family_root, "ref_path": rel, "expected": want, "got": got},
            )

    mapping_obj = _load_json_file(resolved_abs)
    _validate_json_against_schema(mapping_obj, mapping_schema)
    dc = mapping_obj.get("determinism_contract", {})
    if dc.get("invariant_diff_defined") is not True or not str(dc.get("nondeterminism_proof", "")).strip():
        raise MetaValidationError(
            "MAPPING_PROTOCOL_DETERMINISM_CONTRACT_MISSING",
            {"family_root": family_root, "resolved": resolved_abs},
        )
    return "ref", resolved_abs


def validate_hot_mess_incoherence(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate HOT_MESS_INCOHERENCE_CERT semantics (lightweight meta-validator checks)."""
    v = ValidationResult()

    schema = cert_dict.get("schema")
    v.check(schema == "QA_HOT_MESS_INCOHERENCE_CERT.v1",
            f"schema must be QA_HOT_MESS_INCOHERENCE_CERT.v1, got: {schema}")

    success = cert_dict.get("success")
    v.check(isinstance(success, bool), "success must be boolean")

    for key in ("model_id", "task_family", "eval_metric_id"):
        val = cert_dict.get(key)
        v.check(isinstance(val, str) and bool(val), f"{key} must be a non-empty string")

    num_runs = cert_dict.get("num_runs")
    v.check(isinstance(num_runs, int) and num_runs >= 0, "num_runs must be a non-negative int")

    outcomes = cert_dict.get("run_outcomes")
    v.check(isinstance(outcomes, list), "run_outcomes must be a list")
    if isinstance(outcomes, list) and isinstance(num_runs, int):
        v.check(len(outcomes) == num_runs,
                f"run_outcomes length mismatch: len={len(outcomes)} num_runs={num_runs}")

    # Agreement rate (mode frequency over output_hash)
    agreement_rate = Fraction(0)
    if isinstance(outcomes, list) and len(outcomes) > 0:
        counts: Dict[str, int] = {}
        for i, o in enumerate(outcomes[:500]):
            v.check(isinstance(o, dict), f"run_outcomes[{i}] must be a dict")
            if not isinstance(o, dict):
                continue
            for f in ("run_id", "rng_seed", "step_count", "output_hash", "score", "success"):
                v.check(f in o, f"run_outcomes[{i}] missing field: {f}")
            h = o.get("output_hash")
            if isinstance(h, str) and h:
                counts[h] = counts.get(h, 0) + 1
            v.check(isinstance(o.get("success"), bool), f"run_outcomes[{i}].success must be boolean")
        if counts:
            agreement_rate = Fraction(max(counts.values()), len(outcomes))

    provided_agree = cert_dict.get("agreement_rate")
    if provided_agree is not None:
        try:
            prov = _parse_fraction_like(provided_agree)
            v.check(prov == agreement_rate,
                    f"agreement_rate mismatch: provided={prov} recomputed={agreement_rate}")
        except Exception as e:
            v.check(False, f"agreement_rate invalid scalar: {e}")

    # Decomposition witness arithmetic
    dec = cert_dict.get("decomposition_witness")
    v.check(isinstance(dec, dict), "decomposition_witness must be dict")
    incoherence_ratio = None
    if isinstance(dec, dict):
        for f in ("metric_id", "total_error", "bias_component", "variance_component", "incoherence_ratio"):
            v.check(f in dec, f"decomposition_witness missing field: {f}")
        try:
            total = _parse_fraction_like(dec.get("total_error"))
            bias = _parse_fraction_like(dec.get("bias_component"))
            var = _parse_fraction_like(dec.get("variance_component"))
            ratio = _parse_fraction_like(dec.get("incoherence_ratio"))
            v.check(total == bias + var,
                    f"decomposition total_error mismatch: total={total} bias+var={bias+var}")
            expected_ratio = Fraction(0) if total == 0 else (var / total)
            v.check(ratio == expected_ratio,
                    f"incoherence_ratio mismatch: ratio={ratio} expected={expected_ratio}")
            incoherence_ratio = ratio
        except Exception as e:
            v.check(False, f"decomposition_witness scalar parse error: {e}")

    inv = cert_dict.get("coherence_invariant")
    v.check(isinstance(inv, dict), "coherence_invariant must be dict")
    if isinstance(inv, dict):
        for f in ("metric_id", "max_incoherence_ratio"):
            v.check(f in inv, f"coherence_invariant missing field: {f}")
        if incoherence_ratio is not None and success is True:
            try:
                max_incoh = _parse_fraction_like(inv.get("max_incoherence_ratio"))
                v.check(incoherence_ratio <= max_incoh,
                        f"I_coh violation: incoherence_ratio={incoherence_ratio} > max={max_incoh}")
            except Exception as e:
                v.check(False, f"coherence_invariant.max_incoherence_ratio parse error: {e}")
        if inv.get("min_agreement_rate") is not None and success is True:
            try:
                min_agree = _parse_fraction_like(inv.get("min_agreement_rate"))
                v.check(agreement_rate >= min_agree,
                        f"I_coh violation: agreement_rate={agreement_rate} < min={min_agree}")
            except Exception as e:
                v.check(False, f"coherence_invariant.min_agreement_rate parse error: {e}")

    if success is False:
        v.check(bool(cert_dict.get("failure_mode")), "failure certificate missing failure_mode")
        v.check(isinstance(cert_dict.get("failure_witness"), dict), "failure certificate missing failure_witness")

    return v


def validate_topology_resonance(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate TOPOLOGY_RESONANCE_CERT-specific semantics."""
    v = ValidationResult()

    valid_generators = {"sigma", "mu", "lambda2", "nu"}
    generators = cert_dict.get("generator_set", [])
    v.check(isinstance(generators, list) and len(generators) > 0,
            "generator_set must be a non-empty list")
    if isinstance(generators, list):
        for i, g in enumerate(generators):
            v.check(isinstance(g, str) and bool(g), f"generator_set[{i}] must be a non-empty string")
        unknown = [g for g in generators if isinstance(g, str) and g not in valid_generators]
        v.check(len(unknown) == 0, f"Unknown topology generators: {unknown}")

    schema = cert_dict.get("schema")
    v.check(schema == "QA_TOPOLOGY_RESONANCE_CERT.v1",
            f"schema must be QA_TOPOLOGY_RESONANCE_CERT.v1, got: {schema}")

    success = cert_dict.get("success")
    v.check(isinstance(success, bool), "success must be boolean")
    if success is False:
        v.check(bool(cert_dict.get("failure_mode")), "failure certificate missing failure_mode")
        v.check(isinstance(cert_dict.get("failure_witness"), dict), "failure certificate missing failure_witness")
        return v

    topo = cert_dict.get("topology_witness")
    phase = cert_dict.get("phase_witness")
    inv = cert_dict.get("invariants")

    v.check(isinstance(topo, dict), "success certificate missing topology_witness")
    v.check(isinstance(phase, dict), "success certificate missing phase_witness")
    v.check(isinstance(inv, dict), "success certificate missing invariants")
    if not (isinstance(topo, dict) and isinstance(phase, dict) and isinstance(inv, dict)):
        return v

    try:
        scc_before = int(topo.get("scc_count_before"))
        scc_after = int(topo.get("scc_count_after"))
    except Exception:
        v.check(False, "topology_witness.scc_count_before/after must be integers")
        return v

    scc_monotone_flag = inv.get("scc_monotone_non_decreasing")
    v.check(isinstance(scc_monotone_flag, bool), "invariants.scc_monotone_non_decreasing must be boolean")
    should_be_monotone = scc_after >= scc_before
    v.check(should_be_monotone == scc_monotone_flag,
            f"SCC monotonic flag mismatch: {scc_after} >= {scc_before} is {should_be_monotone}, "
            f"claimed {scc_monotone_flag}")

    try:
        p24_before = int(phase.get("phase_24_before"))
        p24_after = int(phase.get("phase_24_after"))
        p9_before = int(phase.get("phase_9_before"))
        p9_after = int(phase.get("phase_9_after"))
    except Exception:
        v.check(False, "phase_witness phase fields must be integers")
        return v

    v.check(0 <= p24_before < 24 and 0 <= p24_after < 24,
            "phase_24 values must satisfy 0 <= phase < 24")
    v.check(0 <= p9_before < 9 and 0 <= p9_after < 9,
            "phase_9 values must satisfy 0 <= phase < 9")

    phase_preserved_flag = phase.get("phase_preserved")
    v.check(isinstance(phase_preserved_flag, bool), "phase_witness.phase_preserved must be boolean")
    should_preserve = (p24_before == p24_after) and (p9_before == p9_after)
    v.check(should_preserve == phase_preserved_flag,
            f"Phase preservation mismatch: expected {should_preserve}, claimed {phase_preserved_flag}")

    resonance_certified = topo.get("resonance_certified")
    v.check(isinstance(resonance_certified, bool), "topology_witness.resonance_certified must be boolean")
    try:
        resonance_score = _parse_fraction_like(topo.get("resonance_score"))
        resonance_threshold = _parse_fraction_like(topo.get("resonance_threshold"))
        should_certify = resonance_score >= resonance_threshold
        v.check(should_certify == resonance_certified,
                f"Resonance certification mismatch: {resonance_score} >= {resonance_threshold} "
                f"is {should_certify}, claimed {resonance_certified}")
    except Exception as e:
        v.check(False, f"Invalid resonance scalar(s): {e}")

    for k in ("packet_conservation", "no_reduction_axiom", "connected_component_first_class"):
        v.check(inv.get(k) is True, f"invariants.{k} must be true for success certificates")

    return v


def validate_graph_structure(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate GRAPH_STRUCTURE_CERT-specific semantics."""
    v = ValidationResult()

    valid_generators = {
        "sigma_feat_extract",
        "sigma_qa_embed",
        "sigma_cluster",
        "sigma_eval",
        "sigma_phase_analyze",
    }
    generators = cert_dict.get("generator_set", [])
    v.check(isinstance(generators, list) and len(generators) > 0, "generator_set must be a non-empty list")
    if isinstance(generators, list):
        unknown = [g for g in generators if isinstance(g, str) and g not in valid_generators]
        v.check(len(unknown) == 0, f"Unknown graph generators: {unknown}")

    schema = cert_dict.get("schema")
    v.check(schema == "QA_GRAPH_STRUCTURE_CERT.v1", f"schema must be QA_GRAPH_STRUCTURE_CERT.v1, got: {schema}")

    success = cert_dict.get("success")
    v.check(isinstance(success, bool), "success must be boolean")
    if success is False:
        v.check(bool(cert_dict.get("failure_mode")), "failure certificate missing failure_mode")
        v.check(isinstance(cert_dict.get("failure_witness"), dict), "failure certificate missing failure_witness")
        return v

    graph_ctx = cert_dict.get("graph_context")
    metric_witness = cert_dict.get("metric_witness")
    phase = cert_dict.get("phase_witness")
    inv = cert_dict.get("invariants")
    recompute = cert_dict.get("recompute_inputs")
    for name, obj in (
        ("graph_context", graph_ctx),
        ("metric_witness", metric_witness),
        ("phase_witness", phase),
        ("invariants", inv),
        ("recompute_inputs", recompute),
    ):
        v.check(isinstance(obj, dict), f"success certificate missing {name}")
    if not all(isinstance(x, dict) for x in (graph_ctx, metric_witness, phase, inv, recompute)):
        return v

    try:
        node_count = int(graph_ctx.get("node_count"))
        edge_count = int(graph_ctx.get("edge_count"))
        community_count = int(graph_ctx.get("community_count"))
        split_seed = int(graph_ctx.get("split_seed"))
        clustering_seed = int(graph_ctx.get("clustering_seed"))
    except Exception:
        v.check(False, "graph_context parse failed for integer bounds fields")
        return v
    v.check(node_count > 0, "graph_context.node_count must be > 0")
    v.check(edge_count >= 0, "graph_context.edge_count must be >= 0")
    v.check(1 <= community_count <= node_count, "graph_context.community_count must satisfy 1 <= communities <= nodes")
    v.check(split_seed >= 0, "graph_context.split_seed must be >= 0")
    v.check(clustering_seed >= 0, "graph_context.clustering_seed must be >= 0")

    qa_metrics = metric_witness.get("qa_metrics", {})
    baseline_metrics = metric_witness.get("baseline_metrics", {})
    delta_metrics = metric_witness.get("delta_metrics", {})
    metric_keys = ("ari", "nmi", "modularity", "purity")
    deltas: Dict[str, Fraction] = {}
    for metric_key in metric_keys:
        try:
            qa_val = _parse_fraction_like(qa_metrics.get(metric_key))
            base_val = _parse_fraction_like(baseline_metrics.get(metric_key))
            declared_delta = _parse_fraction_like(delta_metrics.get(metric_key))
            expected_delta = qa_val - base_val
            deltas[metric_key] = expected_delta
            v.check(
                declared_delta == expected_delta,
                f"delta mismatch for {metric_key}: declared {declared_delta}, expected {expected_delta}",
            )
        except Exception as e:
            v.check(False, f"metric parse failed for {metric_key}: {e}")

    delta_non_negative_any = metric_witness.get("delta_non_negative_any")
    v.check(isinstance(delta_non_negative_any, bool), "metric_witness.delta_non_negative_any must be boolean")
    if isinstance(delta_non_negative_any, bool) and all(k in deltas for k in ("ari", "modularity")):
        computed_any = (deltas["ari"] >= 0) or (deltas["modularity"] >= 0)
        v.check(
            computed_any == delta_non_negative_any,
            f"delta_non_negative_any mismatch: declared={delta_non_negative_any}, computed={computed_any}",
        )

    try:
        p24_base = int(phase.get("phase_24_baseline"))
        p24_qa = int(phase.get("phase_24_qa"))
        p9_base = int(phase.get("phase_9_baseline"))
        p9_qa = int(phase.get("phase_9_qa"))
        phase_preserved = bool(phase.get("phase_preserved"))
    except Exception:
        v.check(False, "phase_witness phase fields must be integers/boolean")
        return v
    v.check(0 <= p24_base < 24 and 0 <= p24_qa < 24, "phase_24 values must satisfy 0 <= phase < 24")
    v.check(0 <= p9_base < 9 and 0 <= p9_qa < 9, "phase_9 values must satisfy 0 <= phase < 9")
    expected_phase_preserved = (p24_base == p24_qa) and (p9_base == p9_qa)
    v.check(
        phase_preserved == expected_phase_preserved,
        f"phase_preserved mismatch: declared {phase_preserved}, expected {expected_phase_preserved}",
    )

    for k in ("tuple_consistency", "feature_determinism", "eval_repro", "trace", "baseline_pairing"):
        v.check(inv.get(k) is True, f"invariants.{k} must be true for success certificates")

    paired_cfg = recompute.get("paired_config")
    v.check(isinstance(paired_cfg, dict), "recompute_inputs.paired_config must be object")
    if isinstance(paired_cfg, dict):
        v.check(
            paired_cfg.get("dataset_id") == graph_ctx.get("dataset_id"),
            "paired_config.dataset_id must match graph_context.dataset_id",
        )
        v.check(
            paired_cfg.get("algorithm") == graph_ctx.get("algorithm"),
            "paired_config.algorithm must match graph_context.algorithm",
        )
        v.check(
            paired_cfg.get("split_seed") == graph_ctx.get("split_seed"),
            "paired_config.split_seed must match graph_context.split_seed",
        )
        v.check(
            paired_cfg.get("clustering_seed") == graph_ctx.get("clustering_seed"),
            "paired_config.clustering_seed must match graph_context.clustering_seed",
        )

    baseline_trace = recompute.get("baseline_trace")
    qa_trace = recompute.get("qa_trace")
    v.check(isinstance(baseline_trace, list) and len(baseline_trace) > 0, "recompute_inputs.baseline_trace must be non-empty list")
    v.check(isinstance(qa_trace, list) and len(qa_trace) > 0, "recompute_inputs.qa_trace must be non-empty list")
    if not (isinstance(baseline_trace, list) and isinstance(qa_trace, list) and baseline_trace and qa_trace):
        return v

    trace_schema = recompute.get("trace_schema")
    v.check(trace_schema == "QA_GRAPH_STRUCTURE_TRACE.v1", "recompute_inputs.trace_schema must be QA_GRAPH_STRUCTURE_TRACE.v1")

    canonical_payload = json.dumps(
        {"baseline_trace": baseline_trace, "qa_trace": qa_trace},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    expected_digest = "sha256:" + hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    v.check(
        recompute.get("trace_digest") == expected_digest,
        f"recompute_inputs.trace_digest mismatch: expected {expected_digest}",
    )

    def _validate_trace(trace: List[Dict[str, Any]], trace_name: str) -> Optional[Dict[str, Any]]:
        prev = 0
        terminal: Optional[Dict[str, Any]] = None
        for i, step in enumerate(trace):
            if not isinstance(step, dict):
                v.check(False, f"{trace_name}[{i}] must be object")
                return None
            try:
                step_index = int(step.get("step_index"))
                generator = step.get("generator")
                p24 = int(step.get("phase_24"))
                p9 = int(step.get("phase_9"))
                metrics = step.get("metrics", {})
            except Exception:
                v.check(False, f"{trace_name}[{i}] parse failed")
                return None
            v.check(step_index == prev + 1, f"{trace_name}[{i}] step_index must be contiguous")
            v.check(generator in generators, f"{trace_name}[{i}] generator not in generator_set: {generator}")
            v.check(0 <= p24 < 24 and 0 <= p9 < 9, f"{trace_name}[{i}] phase out of range")
            if not isinstance(metrics, dict):
                v.check(False, f"{trace_name}[{i}].metrics must be object")
                return None
            for metric_key in metric_keys:
                try:
                    _parse_fraction_like(metrics.get(metric_key))
                except Exception as e:
                    v.check(False, f"{trace_name}[{i}].metrics.{metric_key} invalid: {e}")
            prev = step_index
            terminal = step
        return terminal

    baseline_terminal = _validate_trace(baseline_trace, "baseline_trace")
    qa_terminal = _validate_trace(qa_trace, "qa_trace")
    v.check(len(baseline_trace) == len(qa_trace), "baseline_trace and qa_trace must have equal length")
    if not (isinstance(baseline_terminal, dict) and isinstance(qa_terminal, dict)):
        return v

    baseline_terminal_metrics = baseline_terminal.get("metrics", {})
    qa_terminal_metrics = qa_terminal.get("metrics", {})
    for metric_key in metric_keys:
        try:
            v.check(
                _parse_fraction_like(baseline_terminal_metrics.get(metric_key))
                == _parse_fraction_like(baseline_metrics.get(metric_key)),
                f"baseline terminal metric mismatch: {metric_key}",
            )
            v.check(
                _parse_fraction_like(qa_terminal_metrics.get(metric_key))
                == _parse_fraction_like(qa_metrics.get(metric_key)),
                f"qa terminal metric mismatch: {metric_key}",
            )
        except Exception as e:
            v.check(False, f"terminal metric parse failed for {metric_key}: {e}")

    v.check(int(baseline_terminal.get("phase_24")) == p24_base, "baseline terminal phase_24 mismatch")
    v.check(int(baseline_terminal.get("phase_9")) == p9_base, "baseline terminal phase_9 mismatch")
    v.check(int(qa_terminal.get("phase_24")) == p24_qa, "qa terminal phase_24 mismatch")
    v.check(int(qa_terminal.get("phase_9")) == p9_qa, "qa terminal phase_9 mismatch")

    return v


def validate_elliptic_correspondence(cert_dict: Dict[str, Any]) -> ValidationResult:
    """Validate ELLIPTIC_CORRESPONDENCE_CERT-specific semantics."""
    v = ValidationResult()

    valid_generators = {
        "g_plus_r0",
        "g_plus_r1",
        "g_plus_r2",
        "g_minus_r0",
        "g_minus_r1",
        "g_minus_r2",
    }
    valid_failure_modes = {
        "NONFINITE_INPUT",
        "SQRT_BRANCH_UNDEFINED",
        "SQRT_CUT_CROSS_DISALLOWED",
        "CUBIC_SOLVE_FAILED",
        "RAMIFICATION_HIT",
        "MULTIROOT_DEGENERATE",
        "CUTSTATE_UPDATE_FAILED",
        "MONODROMY_EVENT_DETECTED",
        "ESCAPE",
        "MAX_ITER_REACHED",
    }

    generators = cert_dict.get("generator_set", [])
    v.check(isinstance(generators, list) and len(generators) > 0, "generator_set must be a non-empty list")
    if isinstance(generators, list):
        unknown = [g for g in generators if isinstance(g, str) and g not in valid_generators]
        v.check(len(unknown) == 0, f"Unknown elliptic generators: {unknown}")

    schema = cert_dict.get("schema")
    v.check(schema == "QA_ELLIPTIC_CORRESPONDENCE_CERT.v1",
            f"schema must be QA_ELLIPTIC_CORRESPONDENCE_CERT.v1, got: {schema}")

    success = cert_dict.get("success")
    v.check(isinstance(success, bool), "success must be boolean")
    if success is False:
        mode = cert_dict.get("failure_mode")
        witness = cert_dict.get("failure_witness")
        v.check(mode in valid_failure_modes, f"unknown failure_mode: {mode}")
        v.check(isinstance(witness, dict), "failure_witness must be object")
        return v

    state_desc = cert_dict.get("state_descriptor")
    topo = cert_dict.get("topology_witness")
    inv = cert_dict.get("invariants")
    recomp = cert_dict.get("recompute_inputs")
    for name, obj in (
        ("state_descriptor", state_desc),
        ("topology_witness", topo),
        ("invariants", inv),
        ("recompute_inputs", recomp),
    ):
        v.check(isinstance(obj, dict), f"success certificate missing {name}")
    if not all(isinstance(x, dict) for x in (state_desc, topo, inv, recomp)):
        return v

    branch_declared = topo.get("branching_factor_declared")
    try:
        branch_declared_int = int(branch_declared)
        v.check(branch_declared_int == len(set(generators)),
                f"branching_factor_declared={branch_declared_int} does not match generator_set size={len(set(generators))}")
        v.check(branch_declared_int == 6, f"elliptic correspondence branch factor must be 6, got {branch_declared_int}")
    except Exception:
        v.check(False, "topology_witness.branching_factor_declared must be integer")

    for key in ("curve_constraint", "determinism", "cut_consistency", "trace_complete"):
        v.check(inv.get(key) is True, f"invariants.{key} must be true for success certificates")

    try:
        _ = _parse_fraction_like(topo.get("max_norm_u"))
        _ = _parse_fraction_like(topo.get("max_norm_v"))
    except Exception as e:
        v.check(False, f"failed to parse topology max norms: {e}")

    trace_schema = recomp.get("trace_schema")
    v.check(trace_schema == "QA_ELLIPTIC_CORRESPONDENCE_TRACE.v1",
            "recompute_inputs.trace_schema must be QA_ELLIPTIC_CORRESPONDENCE_TRACE.v1")

    initial = recomp.get("initial_state")
    trace = recomp.get("transition_trace")
    v.check(isinstance(initial, dict), "recompute_inputs.initial_state must be object")
    v.check(isinstance(trace, list) and len(trace) > 0, "recompute_inputs.transition_trace must be non-empty list")
    if not (isinstance(initial, dict) and isinstance(trace, list) and trace):
        return v

    canonical_payload = json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected_digest = "sha256:" + hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    v.check(recomp.get("trace_digest") == expected_digest,
            f"recompute_inputs.trace_digest mismatch: expected {expected_digest}")

    seen: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}
    prev_out: Optional[Tuple[str, str, str, int]] = None
    for i, step in enumerate(trace):
        if not isinstance(step, dict):
            v.check(False, f"transition_trace[{i}] must be object")
            return v

        try:
            step_index = int(step.get("step_index"))
            generator = step.get("generator")
            in_tuple = (
                str(step.get("u_in_re")),
                str(step.get("u_in_im")),
                str(step.get("sheet_in")),
                int(step.get("branch_index_in")),
            )
            out_tuple = (
                str(step.get("u_out_re")),
                str(step.get("u_out_im")),
                str(step.get("sheet_out")),
                int(step.get("branch_index_out")),
            )
            status = step.get("status")
            fail_type = step.get("fail_type", "")
            _ = _parse_fraction_like(step.get("curve_residual_abs"))
        except Exception as e:
            v.check(False, f"transition_trace[{i}] parse failed: {e}")
            return v

        v.check(step_index == i + 1, f"transition_trace[{i}] step_index must be contiguous")
        v.check(generator in generators, f"transition_trace[{i}] generator not in generator_set: {generator}")

        if status == "ok":
            v.check(fail_type in ("", None), f"transition_trace[{i}] ok step must not carry fail_type")
        elif status == "fail":
            v.check(fail_type in valid_failure_modes,
                    f"transition_trace[{i}] fail step has unknown fail_type: {fail_type}")
        else:
            v.check(False, f"transition_trace[{i}] status must be ok|fail, got: {status}")

        if i == 0:
            try:
                init_tuple = (
                    str(initial.get("u_re")),
                    str(initial.get("u_im")),
                    str(initial.get("sheet")),
                    int(initial.get("branch_index")),
                )
            except Exception:
                v.check(False, "recompute_inputs.initial_state parse failed")
                return v
            v.check(in_tuple == init_tuple,
                    f"transition_trace[0] input {in_tuple} does not match initial_state {init_tuple}")
        elif prev_out is not None:
            v.check(in_tuple == prev_out,
                    f"transition_trace[{i}] input {in_tuple} does not match previous output {prev_out}")
        prev_out = out_tuple

        det_key = (in_tuple[0], in_tuple[1], in_tuple[2], in_tuple[3], generator)
        det_val = (out_tuple[0], out_tuple[1], out_tuple[2], out_tuple[3], status, str(fail_type))
        old = seen.get(det_key)
        if old is None:
            seen[det_key] = det_val
        else:
            v.check(old == det_val,
                    f"determinism mismatch for key={det_key}: first={old}, second={det_val}")

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
    "HOT_MESS_INCOHERENCE_CERT": validate_hot_mess_incoherence,
    "TOPOLOGY_RESONANCE_CERT": validate_topology_resonance,
    "ELLIPTIC_CORRESPONDENCE_CERT": validate_elliptic_correspondence,
    "GRAPH_STRUCTURE_CERT": validate_graph_structure,
    "QA_CONJECTURE": validate_conjecture,
}


# Datastore family paths (optional sweep hook).
DATASTORE_CERT_PATHS = {
    "QA_DATASTORE_SEMANTICS_CERT.v1": "certs/QA_DATASTORE_SEMANTICS_CERT.v1.json",
    "QA_DATASTORE_WITNESS_PACK.v1": "certs/witness/QA_DATASTORE_WITNESS_PACK.v1.json",
    "QA_DATASTORE_COUNTEREXAMPLES_PACK.v1": "certs/counterexamples/QA_DATASTORE_COUNTEREXAMPLES_PACK.v1.json",
}

DATASTORE_VIEW_CERT_PATHS = {
    "QA_DATASTORE_VIEW_CERT.v1": "certs/QA_DATASTORE_VIEW_CERT.v1.json",
    "QA_DATASTORE_VIEW_WITNESS_PACK.v1": "certs/witness/QA_DATASTORE_VIEW_WITNESS_PACK.v1.json",
    "QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1": "certs/counterexamples/QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1.json",
}

ARAG_CERT_PATHS = {
    "QA_ARAG_INTERFACE_CERT.v1": "certs/QA_ARAG_INTERFACE_CERT.v1.json",
    "QA_ARAG_WITNESS_PACK.v1": "certs/witness/QA_ARAG_WITNESS_PACK.v1.json",
    "QA_ARAG_COUNTEREXAMPLES_PACK.v1": "certs/counterexamples/QA_ARAG_COUNTEREXAMPLES_PACK.v1.json",
}

INGEST_VIEW_BRIDGE_CERT_PATHS = {
    "QA_INGEST_VIEW_BRIDGE_CERT.v1": "certs/QA_INGEST_VIEW_BRIDGE_CERT.v1.json",
    "QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1": "certs/witness/QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1.json",
    "QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1": "certs/counterexamples/QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1.json",
}

INGEST_CERT_PATHS = {
    "QA_INGEST_SEMANTICS_CERT.v1": "certs/QA_INGEST_SEMANTICS_CERT.v1.json",
    "QA_INGEST_WITNESS_PACK.v1": "certs/witness/QA_INGEST_WITNESS_PACK.v1.json",
    "QA_INGEST_COUNTEREXAMPLES_PACK.v1": "certs/counterexamples/QA_INGEST_COUNTEREXAMPLES_PACK.v1.json",
}

SVP_CMC_LEDGER_PATH = "qa_ledger__radionics_obstructions.v1.yaml"


# FAMILY_SWEEPS is defined after all validator functions (see below).
# It is THE canonical list: one entry per family, used by both the sweep
# runner and the doc gate. Adding a family = adding one tuple there.


def _validate_svp_cmc_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run SVP-CMC family validator if ledger is present.

    Returns:
        None on success,
        skip reason string if ledger is missing.
    Raises:
        Exception on validation failure.
    """
    ledger_path = os.path.join(base_dir, SVP_CMC_LEDGER_PATH)
    if not os.path.exists(ledger_path):
        return f"missing ledger: {SVP_CMC_LEDGER_PATH}"

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    # Run ledger sanity check
    from qa_radionics_ledger_sanity import sanity_check
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger_text = f.read()
    ok, errors, warnings = sanity_check(ledger_text)
    if not ok:
        raise RuntimeError(f"Ledger sanity failed: {'; '.join(errors[:3])}")

    # Run validator demo
    from qa_svp_cmc_validator import validate_cert, parse_ledger_obstruction_ids, make_demo_cert
    ledger_ids = parse_ledger_obstruction_ids(ledger_text)
    demo_cert = make_demo_cert()
    result = validate_cert(demo_cert, ledger_obs_ids=ledger_ids)
    if not result.ok:
        raise RuntimeError(f"Demo cert validation failed: {result.summary()}")

    return None


def _validate_datastore_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run datastore family validator if all required cert files are present.

    Returns:
        None on success,
        skip reason string if files are missing.
    Raises:
        Exception on validation failure.
    """
    semantics = os.path.join(base_dir, DATASTORE_CERT_PATHS["QA_DATASTORE_SEMANTICS_CERT.v1"])
    witness = os.path.join(base_dir, DATASTORE_CERT_PATHS["QA_DATASTORE_WITNESS_PACK.v1"])
    counterexamples = os.path.join(base_dir, DATASTORE_CERT_PATHS["QA_DATASTORE_COUNTEREXAMPLES_PACK.v1"])

    required = [semantics, witness, counterexamples]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return f"missing files: {', '.join(os.path.basename(m) for m in missing)}"

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from qa_datastore_validator import validate_all as validate_datastore_all
    validate_datastore_all(
        semantics_path=semantics,
        witness_path=witness,
        counterexamples_path=counterexamples,
    )
    return None


def _validate_datastore_view_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run datastore view family validator if all required cert files are present.

    Returns:
        None on success,
        skip reason string if files are missing.
    Raises:
        Exception on validation failure.
    """
    store_semantics = os.path.join(base_dir, DATASTORE_CERT_PATHS["QA_DATASTORE_SEMANTICS_CERT.v1"])
    view_semantics = os.path.join(base_dir, DATASTORE_VIEW_CERT_PATHS["QA_DATASTORE_VIEW_CERT.v1"])
    witness = os.path.join(base_dir, DATASTORE_VIEW_CERT_PATHS["QA_DATASTORE_VIEW_WITNESS_PACK.v1"])
    counterexamples = os.path.join(base_dir, DATASTORE_VIEW_CERT_PATHS["QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1"])

    required = [store_semantics, view_semantics, witness, counterexamples]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return f"missing files: {', '.join(os.path.basename(m) for m in missing)}"

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from qa_datastore_view_validator import validate_all as validate_datastore_view_all
    validate_datastore_view_all(
        store_semantics_path=store_semantics,
        view_semantics_path=view_semantics,
        witness_path=witness,
        counterexamples_path=counterexamples,
    )
    return None


def _validate_arag_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run A-RAG family validator if all required cert files are present.

    Returns:
        None on success,
        skip reason string if files are missing.
    Raises:
        Exception on validation failure.
    """
    store_semantics = os.path.join(base_dir, DATASTORE_CERT_PATHS["QA_DATASTORE_SEMANTICS_CERT.v1"])
    view_semantics = os.path.join(base_dir, DATASTORE_VIEW_CERT_PATHS["QA_DATASTORE_VIEW_CERT.v1"])
    arag_semantics = os.path.join(base_dir, ARAG_CERT_PATHS["QA_ARAG_INTERFACE_CERT.v1"])
    witness = os.path.join(base_dir, ARAG_CERT_PATHS["QA_ARAG_WITNESS_PACK.v1"])
    counterexamples = os.path.join(base_dir, ARAG_CERT_PATHS["QA_ARAG_COUNTEREXAMPLES_PACK.v1"])

    required = [store_semantics, view_semantics, arag_semantics, witness, counterexamples]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return f"missing files: {', '.join(os.path.basename(m) for m in missing)}"

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from qa_arag_validator import validate_all as validate_arag_all
    validate_arag_all(
        store_semantics_path=store_semantics,
        view_semantics_path=view_semantics,
        arag_semantics_path=arag_semantics,
        witness_path=witness,
        counterexamples_path=counterexamples,
    )
    return None


def _validate_ingest_view_bridge_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run ingest->view bridge family validator if all required cert files are present.

    Returns:
        None on success,
        skip reason string if files are missing.
    Raises:
        Exception on validation failure.
    """
    store_semantics = os.path.join(base_dir, DATASTORE_CERT_PATHS["QA_DATASTORE_SEMANTICS_CERT.v1"])
    view_semantics = os.path.join(base_dir, DATASTORE_VIEW_CERT_PATHS["QA_DATASTORE_VIEW_CERT.v1"])
    bridge_semantics = os.path.join(base_dir, INGEST_VIEW_BRIDGE_CERT_PATHS["QA_INGEST_VIEW_BRIDGE_CERT.v1"])
    witness = os.path.join(base_dir, INGEST_VIEW_BRIDGE_CERT_PATHS["QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1"])
    counterexamples = os.path.join(base_dir, INGEST_VIEW_BRIDGE_CERT_PATHS["QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1"])

    required = [store_semantics, view_semantics, bridge_semantics, witness, counterexamples]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return f"missing files: {', '.join(os.path.basename(m) for m in missing)}"

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from qa_ingest_view_bridge_validator import validate_all as validate_bridge_all
    validate_bridge_all(
        store_semantics_path=store_semantics,
        view_semantics_path=view_semantics,
        bridge_semantics_path=bridge_semantics,
        witness_path=witness,
        counterexamples_path=counterexamples,
    )
    return None


def _validate_ingest_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run ingestion family validator if all required cert files are present.

    Returns:
        None on success,
        skip reason string if files are missing.
    Raises:
        Exception on validation failure.
    """
    semantics = os.path.join(base_dir, INGEST_CERT_PATHS["QA_INGEST_SEMANTICS_CERT.v1"])
    witness = os.path.join(base_dir, INGEST_CERT_PATHS["QA_INGEST_WITNESS_PACK.v1"])
    counterexamples = os.path.join(base_dir, INGEST_CERT_PATHS["QA_INGEST_COUNTEREXAMPLES_PACK.v1"])

    required = [semantics, witness, counterexamples]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return f"missing files: {', '.join(os.path.basename(m) for m in missing)}"

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from qa_ingest_validator import validate_all as validate_ingest_all
    validate_ingest_all(
        semantics_path=semantics,
        witness_path=witness,
        counterexamples_path=counterexamples,
    )
    return None


TOPOLOGY_BUNDLE_PATH = "certs/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.json"
ELLIPTIC_BUNDLE_PATH = "certs/QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.json"
GRAPH_STRUCTURE_BUNDLE_PATH = "certs/QA_GRAPH_STRUCTURE_BUNDLE.v1.json"


def _topology_bundle_status(base_dir: str) -> Dict[str, Any]:
    """
    Validate topology bundle and return normalized status payload.

    Shape:
      {
        "ok": bool,
        "skipped": bool,
        "reason": str?,
        "artifact_count": int?,
        "errors": list[str]?,
        ... (validator passthrough fields)
      }
    """
    bundle_path = os.path.join(base_dir, TOPOLOGY_BUNDLE_PATH)
    if not os.path.exists(bundle_path):
        return {
            "ok": True,
            "skipped": True,
            "reason": "bundle not found",
            "bundle_path": bundle_path,
        }

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from qa_topology_resonance_bundle_v1 import validate_bundle_manifest
    result = validate_bundle_manifest(bundle_path=bundle_path, base_dir=base_dir)
    result.setdefault("skipped", False)
    return result


def _graph_structure_bundle_status(base_dir: str) -> Dict[str, Any]:
    """
    Validate graph-structure bundle and return normalized status payload.
    """
    bundle_path = os.path.join(base_dir, GRAPH_STRUCTURE_BUNDLE_PATH)
    if not os.path.exists(bundle_path):
        return {
            "ok": True,
            "skipped": True,
            "reason": "bundle not found",
            "bundle_path": bundle_path,
        }

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from qa_graph_structure_bundle_v1 import validate_bundle_manifest
    result = validate_bundle_manifest(bundle_path=bundle_path, base_dir=base_dir)
    result.setdefault("skipped", False)
    return result


def _elliptic_bundle_status(base_dir: str) -> Dict[str, Any]:
    """
    Validate elliptic correspondence bundle and return normalized status payload.
    """
    bundle_path = os.path.join(base_dir, ELLIPTIC_BUNDLE_PATH)
    if not os.path.exists(bundle_path):
        return {
            "ok": True,
            "skipped": True,
            "reason": "bundle not found",
            "bundle_path": bundle_path,
        }

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from qa_elliptic_correspondence_bundle_v1 import validate_bundle_manifest
    result = validate_bundle_manifest(bundle_path=bundle_path, base_dir=base_dir)
    result.setdefault("skipped", False)
    return result


def _validate_topology_bundle_if_present(base_dir: str) -> Optional[str]:
    """
    Run topology bundle validator if the bundle file exists.

    Returns:
        None on success,
        skip reason string if bundle is missing.
    Raises:
        Exception on validation failure.
    """
    result = _topology_bundle_status(base_dir)
    if result.get("skipped"):
        return f"missing file: {os.path.basename(TOPOLOGY_BUNDLE_PATH)}"
    if not result.get("ok", False):
        errors = result.get("errors", [])
        head = "; ".join(errors[:3]) if errors else "unknown topology bundle validation error"
        raise RuntimeError(head)
    return None


def _validate_graph_structure_bundle_if_present(base_dir: str) -> Optional[str]:
    """
    Run graph-structure bundle validator if the bundle file exists.

    Returns:
        None on success,
        skip reason string if bundle is missing.
    Raises:
        Exception on validation failure.
    """
    result = _graph_structure_bundle_status(base_dir)
    if result.get("skipped"):
        return f"missing file: {os.path.basename(GRAPH_STRUCTURE_BUNDLE_PATH)}"
    if not result.get("ok", False):
        errors = result.get("errors", [])
        head = "; ".join(errors[:3]) if errors else "unknown graph structure bundle validation error"
        raise RuntimeError(head)
    return None


def _validate_elliptic_bundle_if_present(base_dir: str) -> Optional[str]:
    """
    Run elliptic-correspondence bundle validator if the bundle file exists.

    Returns:
        None on success,
        skip reason string if bundle is missing.
    Raises:
        Exception on validation failure.
    """
    result = _elliptic_bundle_status(base_dir)
    if result.get("skipped"):
        return f"missing file: {os.path.basename(ELLIPTIC_BUNDLE_PATH)}"
    if not result.get("ok", False):
        errors = result.get("errors", [])
        head = "; ".join(errors[:3]) if errors else "unknown elliptic correspondence bundle validation error"
        raise RuntimeError(head)
    return None


def _validate_competency_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run QA Competency Detection family validator if bundle is present.

    Returns:
        None on success,
        skip reason string if bundle is missing.
    Raises:
        Exception on validation failure.
    """
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    bundle = os.path.join(repo_root, "qa_competency", "certs",
                          "QA_COMPETENCY_CERT_BUNDLE.v1.json")
    if not os.path.exists(bundle):
        return "missing bundle: qa_competency/certs/QA_COMPETENCY_CERT_BUNDLE.v1.json"

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from qa_competency.qa_competency_validator import validate_all
    validate_all(bundle_path=bundle)
    return None


def _validate_agent_trace_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run QA Agent Trace schema + validator against fixtures.

    Checks:
      - valid_trace.json must PASS
      - invalid_trace_missing_invariant_diff.json must FAIL with SCHEMA_INVALID
      - invalid_trace_nondeterministic_order.json must FAIL with NONDETERMINISTIC_EVENT_ORDER
      - Validator self-test must pass

    Returns:
        None on success,
        skip reason string if fixtures are missing.
    Raises:
        Exception on validation failure.
    """
    traces_dir = os.path.join(base_dir, "qa_agent_traces")
    fixtures_dir = os.path.join(traces_dir, "fixtures")
    schema_path = os.path.join(traces_dir, "schemas", "QA_AGENT_TRACE_SCHEMA.v1.json")

    if not os.path.isdir(traces_dir):
        return "missing qa_agent_traces/ directory"
    if not os.path.exists(schema_path):
        return "missing QA_AGENT_TRACE_SCHEMA.v1.json"

    # Check fixtures exist
    valid_fix = os.path.join(fixtures_dir, "valid_trace.json")
    neg_inv = os.path.join(fixtures_dir, "invalid_trace_missing_invariant_diff.json")
    neg_ord = os.path.join(fixtures_dir, "invalid_trace_nondeterministic_order.json")
    for fp in [valid_fix, neg_inv, neg_ord]:
        if not os.path.exists(fp):
            return f"missing fixture: {os.path.basename(fp)}"

    # Import the validator (package-style import via base_dir)
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from qa_agent_traces.qa_agent_trace_validator import validate_trace, _self_test

    # Self-test first
    if not _self_test():
        raise RuntimeError("Agent trace validator self-test failed")

    # Valid fixture must pass
    with open(valid_fix, "r", encoding="utf-8") as f:
        trace = json.load(f)
    result = validate_trace(trace)
    if not result.ok:
        raise RuntimeError(
            f"valid_trace.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative: missing invariant_diff must fail with SCHEMA_INVALID
    with open(neg_inv, "r", encoding="utf-8") as f:
        trace = json.load(f)
    result = validate_trace(trace)
    if result.ok:
        raise RuntimeError("invalid_trace_missing_invariant_diff.json should FAIL but passed")
    if result.fail_type not in ("SCHEMA_INVALID", "MISSING_INVARIANT_DIFF"):
        raise RuntimeError(
            f"invalid_trace_missing_invariant_diff.json: expected SCHEMA_INVALID or "
            f"MISSING_INVARIANT_DIFF, got {result.fail_type}"
        )

    # Negative: nondeterministic order must fail with NONDETERMINISTIC_EVENT_ORDER
    with open(neg_ord, "r", encoding="utf-8") as f:
        trace = json.load(f)
    result = validate_trace(trace)
    if result.ok:
        raise RuntimeError("invalid_trace_nondeterministic_order.json should FAIL but passed")
    if result.fail_type != "NONDETERMINISTIC_EVENT_ORDER":
        raise RuntimeError(
            f"invalid_trace_nondeterministic_order.json: expected "
            f"NONDETERMINISTIC_EVENT_ORDER, got {result.fail_type}"
        )

    return None


def _validate_agent_trace_competency_cert_family_if_present(base_dir: str) -> Optional[str]:
    """
    Run QA Agent Trace Competency Cert validator against fixtures.

    Checks:
      - Validator self-test (8 built-in checks)
      - competency_cert_valid.json must PASS
      - competency_cert_invalid_bad_trace_hash.json must FAIL with TRACE_REF_HASH_MISMATCH
      - competency_cert_invalid_missing_invariant_diff.json must FAIL with MISSING_INVARIANT_DIFF
      - competency_cert_invalid_bad_dominance.json must FAIL with NONDETERMINISTIC_DERIVATION

    Returns:
        None on success,
        skip reason string if fixtures are missing.
    Raises:
        Exception on validation failure.
    """
    traces_dir = os.path.join(base_dir, "qa_agent_traces")
    fixtures_dir = os.path.join(traces_dir, "fixtures")
    schema_path = os.path.join(traces_dir, "schemas",
                               "QA_AGENT_TRACE_COMPETENCY_CERT_SCHEMA.v1.json")

    if not os.path.exists(schema_path):
        return "missing QA_AGENT_TRACE_COMPETENCY_CERT_SCHEMA.v1.json"

    valid_fix = os.path.join(fixtures_dir, "competency_cert_valid.json")
    neg_hash = os.path.join(fixtures_dir, "competency_cert_invalid_bad_trace_hash.json")
    neg_inv = os.path.join(fixtures_dir, "competency_cert_invalid_missing_invariant_diff.json")
    neg_dom = os.path.join(fixtures_dir, "competency_cert_invalid_bad_dominance.json")
    for fp in [valid_fix, neg_hash, neg_inv, neg_dom]:
        if not os.path.exists(fp):
            return f"missing fixture: {os.path.basename(fp)}"

    # Import the validator (package-style)
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from qa_agent_traces.qa_agent_trace_competency_cert_validator import (
        validate_cert, _self_test, _canonical,
    )

    # Self-test first
    if not _self_test():
        raise RuntimeError("Agent trace competency cert validator self-test failed")

    # Valid fixture must pass (wrapper format: trace + cert with end-to-end hash binding)
    with open(valid_fix, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "cert" in data and "trace" in data:
        cert_obj = data["cert"]
        trace_canonical = _canonical(data["trace"])
        result = validate_cert(cert_obj, trace_canonical=trace_canonical)
    else:
        result = validate_cert(data)
    if not result.ok:
        raise RuntimeError(
            f"competency_cert_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative: bad trace hash
    with open(neg_hash, "r", encoding="utf-8") as f:
        data = json.load(f)
    cert_obj = data["cert"]
    trace_canonical = _canonical(data["trace"])
    result = validate_cert(cert_obj, trace_canonical=trace_canonical)
    if result.ok:
        raise RuntimeError(
            "competency_cert_invalid_bad_trace_hash.json should FAIL but passed"
        )
    if result.fail_type != "TRACE_REF_HASH_MISMATCH":
        raise RuntimeError(
            f"competency_cert_invalid_bad_trace_hash.json: expected "
            f"TRACE_REF_HASH_MISMATCH, got {result.fail_type}"
        )

    # Negative: missing invariant_diff
    with open(neg_inv, "r", encoding="utf-8") as f:
        cert = json.load(f)
    result = validate_cert(cert)
    if result.ok:
        raise RuntimeError(
            "competency_cert_invalid_missing_invariant_diff.json should FAIL but passed"
        )
    if result.fail_type != "MISSING_INVARIANT_DIFF":
        raise RuntimeError(
            f"competency_cert_invalid_missing_invariant_diff.json: expected "
            f"MISSING_INVARIANT_DIFF, got {result.fail_type}"
        )

    # Negative: bad dominance (recomputation mismatch)
    with open(neg_dom, "r", encoding="utf-8") as f:
        data = json.load(f)
    cert_obj = data["cert"]
    trace_canonical = _canonical(data["trace"])
    result = validate_cert(cert_obj, trace_canonical=trace_canonical)
    if result.ok:
        raise RuntimeError(
            "competency_cert_invalid_bad_dominance.json should FAIL but passed"
        )
    if result.fail_type != "NONDETERMINISTIC_DERIVATION":
        raise RuntimeError(
            f"competency_cert_invalid_bad_dominance.json: expected "
            f"NONDETERMINISTIC_DERIVATION, got {result.fail_type}"
        )

    return None


def _validate_math_compiler_stack_if_present(base_dir: str) -> Optional[str]:
    """
    Run QA Math Compiler Stack validator against fixtures.

    Checks:
      - Validator self-test (17 built-in checks)
      - trace_valid.json must PASS
      - trace_invalid_missing_invariant_diff.json must FAIL with RESULT_INCOMPLETE
      - pair_valid.json must PASS
      - pair_invalid_hash_mismatch.json must FAIL with HASH_SELF_BINDING
      - task_valid.json must PASS
      - task_invalid_missing_formal_goal.json must FAIL with SCHEMA_INVALID
      - replay_valid.json must PASS
      - replay_invalid_bad_determinism.json must FAIL with DETERMINISM_MISMATCH
      - pair_v1_valid_proved.json must PASS
      - pair_v1_invalid_unproved_replay.json must FAIL with PROVED_PAIR_REPLAY_MISMATCH
      - lemma_mining_valid.json must PASS
      - lemma_mining_invalid_low_compression.json must FAIL with COMPRESSION_BELOW_TARGET
      - optional demo_pack_v1 (if present) must PASS demo_pack validator

    Returns:
        None on success,
        skip reason string if fixtures are missing.
    Raises:
        Exception on validation failure.
    """
    mc_dir = os.path.join(base_dir, "qa_math_compiler")
    fixtures_dir = os.path.join(mc_dir, "fixtures")

    if not os.path.isdir(mc_dir):
        return "missing qa_math_compiler/ directory"

    schema_dir = os.path.join(mc_dir, "schemas")

    schema_trace = os.path.join(schema_dir, "QA_MATH_COMPILER_TRACE_SCHEMA.v1.json")
    schema_pair = os.path.join(schema_dir, "QA_COMPILER_PAIR_CERT_SCHEMA.v1.json")
    schema_task = os.path.join(schema_dir, "QA_FORMAL_TASK_SCHEMA.v1.json")
    schema_replay = os.path.join(schema_dir, "QA_MATH_COMPILER_REPLAY_BUNDLE_SCHEMA.v1.json")
    schema_pair_v1 = os.path.join(schema_dir, "QA_HUMAN_FORMAL_PAIR_CERT.v1.json")
    schema_lemma = os.path.join(schema_dir, "QA_LEMMA_MINING_SCHEMA.v1.json")
    schema_demo_pack = os.path.join(schema_dir, "QA_MATH_COMPILER_DEMO_PACK_SCHEMA.v1.json")

    trace_valid = os.path.join(fixtures_dir, "trace_valid.json")
    trace_neg = os.path.join(fixtures_dir, "trace_invalid_missing_invariant_diff.json")
    pair_valid = os.path.join(fixtures_dir, "pair_valid.json")
    pair_neg = os.path.join(fixtures_dir, "pair_invalid_hash_mismatch.json")
    task_valid = os.path.join(fixtures_dir, "task_valid.json")
    task_neg = os.path.join(fixtures_dir, "task_invalid_missing_formal_goal.json")
    replay_valid = os.path.join(fixtures_dir, "replay_valid.json")
    replay_neg = os.path.join(fixtures_dir, "replay_invalid_bad_determinism.json")
    pair_v1_valid = os.path.join(fixtures_dir, "pair_v1_valid_proved.json")
    pair_v1_neg = os.path.join(fixtures_dir, "pair_v1_invalid_unproved_replay.json")
    lemma_valid = os.path.join(fixtures_dir, "lemma_mining_valid.json")
    lemma_neg = os.path.join(fixtures_dir, "lemma_mining_invalid_low_compression.json")
    required_artifacts = [
        schema_trace, schema_pair, schema_task, schema_replay, schema_pair_v1, schema_lemma, schema_demo_pack,
        trace_valid, trace_neg, pair_valid, pair_neg,
        task_valid, task_neg, replay_valid, replay_neg,
        pair_v1_valid, pair_v1_neg, lemma_valid, lemma_neg,
    ]
    for fp in required_artifacts:
        if not os.path.exists(fp):
            return f"missing artifact: {os.path.basename(fp)}"

    # Enforce that required schema/fixture artifacts are tracked (not only
    # present in a dirty workspace). Skip this check when .git metadata is
    # unavailable (e.g., clean archive CI preflight tree).
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    if os.path.exists(os.path.join(repo_root, ".git")):
        import subprocess

        for fp in required_artifacts:
            rel = os.path.relpath(fp, repo_root)
            proc = subprocess.run(
                ["git", "-C", repo_root, "ls-files", "--error-unmatch", rel],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"UNTRACKED_REQUIRED_ARTIFACT: {rel}"
                )

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from qa_math_compiler.qa_math_compiler_validator import (
        validate_trace, validate_pair, validate_task, validate_replay_bundle,
        validate_pair_v1, validate_lemma_mining, validate_demo_pack_v1, _self_test,
    )

    # Self-test first
    if not _self_test():
        raise RuntimeError("Math compiler validator self-test failed")

    # Valid trace fixture must pass
    with open(trace_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_trace(data)
    if not result.ok:
        raise RuntimeError(
            f"trace_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative trace: FAIL result missing invariant_diff -> RESULT_INCOMPLETE
    with open(trace_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_trace(data)
    if result.ok:
        raise RuntimeError("trace_invalid_missing_invariant_diff.json should FAIL but passed")
    if result.fail_type != "RESULT_INCOMPLETE":
        raise RuntimeError(
            f"trace_invalid_missing_invariant_diff.json: expected "
            f"RESULT_INCOMPLETE, got {result.fail_type}"
        )

    # Valid pair fixture must pass
    with open(pair_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_pair(data)
    if not result.ok:
        raise RuntimeError(
            f"pair_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative pair: human_hash == formal_hash -> HASH_SELF_BINDING
    with open(pair_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_pair(data)
    if result.ok:
        raise RuntimeError("pair_invalid_hash_mismatch.json should FAIL but passed")
    if result.fail_type != "HASH_SELF_BINDING":
        raise RuntimeError(
            f"pair_invalid_hash_mismatch.json: expected "
            f"HASH_SELF_BINDING, got {result.fail_type}"
        )

    # Valid formal task fixture must pass
    with open(task_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_task(data)
    if not result.ok:
        raise RuntimeError(
            f"task_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative task: empty formal_goal -> SCHEMA_INVALID
    with open(task_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_task(data)
    if result.ok:
        raise RuntimeError("task_invalid_missing_formal_goal.json should FAIL but passed")
    if result.fail_type != "SCHEMA_INVALID":
        raise RuntimeError(
            f"task_invalid_missing_formal_goal.json: expected "
            f"SCHEMA_INVALID, got {result.fail_type}"
        )

    # Valid replay fixture must pass
    with open(replay_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_replay_bundle(data)
    if not result.ok:
        raise RuntimeError(
            f"replay_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative replay: trace_hash != replay_hash -> DETERMINISM_MISMATCH
    with open(replay_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_replay_bundle(data)
    if result.ok:
        raise RuntimeError("replay_invalid_bad_determinism.json should FAIL but passed")
    if result.fail_type != "DETERMINISM_MISMATCH":
        raise RuntimeError(
            f"replay_invalid_bad_determinism.json: expected "
            f"DETERMINISM_MISMATCH, got {result.fail_type}"
        )

    # Valid Pair v1 fixture must pass
    with open(pair_v1_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_pair_v1(data)
    if not result.ok:
        raise RuntimeError(
            f"pair_v1_valid_proved.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative Pair v1: PROVED with replay FAIL -> PROVED_PAIR_REPLAY_MISMATCH
    with open(pair_v1_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_pair_v1(data)
    if result.ok:
        raise RuntimeError("pair_v1_invalid_unproved_replay.json should FAIL but passed")
    if result.fail_type != "PROVED_PAIR_REPLAY_MISMATCH":
        raise RuntimeError(
            f"pair_v1_invalid_unproved_replay.json: expected "
            f"PROVED_PAIR_REPLAY_MISMATCH, got {result.fail_type}"
        )

    # Valid lemma mining fixture must pass
    with open(lemma_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_lemma_mining(data)
    if not result.ok:
        raise RuntimeError(
            f"lemma_mining_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative lemma mining: median reduction below target
    with open(lemma_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_lemma_mining(data)
    if result.ok:
        raise RuntimeError("lemma_mining_invalid_low_compression.json should FAIL but passed")
    if result.fail_type != "COMPRESSION_BELOW_TARGET":
        raise RuntimeError(
            f"lemma_mining_invalid_low_compression.json: expected "
            f"COMPRESSION_BELOW_TARGET, got {result.fail_type}"
        )

    # Optional demo pack validation (if present in repo)
    demo_pack_dir = os.path.join(mc_dir, "demo_pack_v1")
    if os.path.isdir(demo_pack_dir):
        # Enforce tracked artifacts for demo pack as well.
        if os.path.exists(os.path.join(repo_root, ".git")):
            import subprocess

            demo_files = []
            for root, _, files in os.walk(demo_pack_dir):
                for name in files:
                    demo_files.append(os.path.join(root, name))
            for fp in sorted(demo_files):
                rel = os.path.relpath(fp, repo_root)
                proc = subprocess.run(
                    ["git", "-C", repo_root, "ls-files", "--error-unmatch", rel],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"UNTRACKED_REQUIRED_ARTIFACT: {rel}"
                    )

        result = validate_demo_pack_v1(demo_pack_dir)
        if not result.ok:
            raise RuntimeError(
                f"demo_pack_v1 should PASS but got {result.fail_type}: "
                f"{json.dumps(result.invariant_diff, sort_keys=True)}"
            )

    return None


def _validate_conjecture_prove_loop_if_present(base_dir: str) -> Optional[str]:
    """
    Run QA Conjecture-Prove Control Loop validator against fixtures.

    Checks:
      - Validator self-test (11 built-in checks)
      - episode_valid.json must PASS
      - episode_invalid_missing_invariant_diff.json must FAIL with RESULT_INCOMPLETE
      - frontier_valid.json must PASS
      - frontier_invalid_bad_hash_chain.json must FAIL with HASH_CHAIN_INVALID
      - bounded_return_valid.json must PASS
      - bounded_return_invalid_missing_fail.json must FAIL with RESULT_INCOMPLETE

    Returns:
        None on success,
        skip reason string if fixtures are missing.
    Raises:
        Exception on validation failure.
    """
    cp_dir = os.path.join(base_dir, "qa_conjecture_prove")
    fixtures_dir = os.path.join(cp_dir, "fixtures")

    if not os.path.isdir(cp_dir):
        return "missing qa_conjecture_prove/ directory"

    ep_valid = os.path.join(fixtures_dir, "episode_valid.json")
    ep_neg = os.path.join(fixtures_dir, "episode_invalid_missing_invariant_diff.json")
    fr_valid = os.path.join(fixtures_dir, "frontier_valid.json")
    fr_neg = os.path.join(fixtures_dir, "frontier_invalid_bad_hash_chain.json")
    br_valid = os.path.join(fixtures_dir, "bounded_return_valid.json")
    br_neg = os.path.join(fixtures_dir, "bounded_return_invalid_missing_fail.json")
    for fp in [ep_valid, ep_neg, fr_valid, fr_neg, br_valid, br_neg]:
        if not os.path.exists(fp):
            return f"missing fixture: {os.path.basename(fp)}"

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from qa_conjecture_prove.qa_conjecture_prove_validator import (
        validate_episode, validate_frontier, validate_receipt, _self_test,
    )

    # Self-test first
    if not _self_test():
        raise RuntimeError("Conjecture-prove validator self-test failed")

    # Valid episode fixture must pass
    with open(ep_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_episode(data)
    if not result.ok:
        raise RuntimeError(
            f"episode_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative episode: FAIL step missing invariant_diff -> RESULT_INCOMPLETE
    with open(ep_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_episode(data)
    if result.ok:
        raise RuntimeError("episode_invalid_missing_invariant_diff.json should FAIL but passed")
    if result.fail_type != "RESULT_INCOMPLETE":
        raise RuntimeError(
            f"episode_invalid_missing_invariant_diff.json: expected "
            f"RESULT_INCOMPLETE, got {result.fail_type}"
        )

    # Valid frontier fixture must pass
    with open(fr_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_frontier(data)
    if not result.ok:
        raise RuntimeError(
            f"frontier_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative frontier: empty this_snapshot_hash -> HASH_CHAIN_INVALID
    with open(fr_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_frontier(data)
    if result.ok:
        raise RuntimeError("frontier_invalid_bad_hash_chain.json should FAIL but passed")
    if result.fail_type != "HASH_CHAIN_INVALID":
        raise RuntimeError(
            f"frontier_invalid_bad_hash_chain.json: expected "
            f"HASH_CHAIN_INVALID, got {result.fail_type}"
        )

    # Valid bounded return receipt must pass
    with open(br_valid, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_receipt(data)
    if not result.ok:
        raise RuntimeError(
            f"bounded_return_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative receipt: NO_RETURN without fail_type -> RESULT_INCOMPLETE
    with open(br_neg, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_receipt(data)
    if result.ok:
        raise RuntimeError("bounded_return_invalid_missing_fail.json should FAIL but passed")
    if result.fail_type != "RESULT_INCOMPLETE":
        raise RuntimeError(
            f"bounded_return_invalid_missing_fail.json: expected "
            f"RESULT_INCOMPLETE, got {result.fail_type}"
        )

    return None


def _validate_discovery_pipeline_if_present(base_dir: str) -> Optional[str]:
    """
    Run QA Discovery Pipeline validator against fixtures.

    Checks:
      - Validator self-test (12 built-in checks)
      - run_valid.json must PASS
      - run_invalid_missing_invariant_diff.json must FAIL with RESULT_INCOMPLETE
      - plan_valid.json must PASS
      - plan_invalid_nondeterministic.json must FAIL with NONDETERMINISTIC_PLAN
      - bundle_valid.json must PASS
      - bundle_invalid_bad_chain.json must FAIL with HASH_CHAIN_INVALID
      - E2E: ci_check.py --allow_fail must reject e2e_neg_no_receipt/ with RUN_FAIL_NO_RECEIPT

    Returns:
        None on success,
        skip reason string if fixtures are missing.
    Raises:
        Exception on validation failure.
    """
    dp_dir = os.path.join(base_dir, "qa_discovery_pipeline")
    fixtures_dir = os.path.join(dp_dir, "fixtures")

    if not os.path.isdir(dp_dir):
        return "missing qa_discovery_pipeline/ directory"

    run_v = os.path.join(fixtures_dir, "run_valid.json")
    run_n = os.path.join(fixtures_dir, "run_invalid_missing_invariant_diff.json")
    plan_v = os.path.join(fixtures_dir, "plan_valid.json")
    plan_n = os.path.join(fixtures_dir, "plan_invalid_nondeterministic.json")
    bundle_v = os.path.join(fixtures_dir, "bundle_valid.json")
    bundle_n = os.path.join(fixtures_dir, "bundle_invalid_bad_chain.json")
    for fp in [run_v, run_n, plan_v, plan_n, bundle_v, bundle_n]:
        if not os.path.exists(fp):
            return f"missing fixture: {os.path.basename(fp)}"

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from qa_discovery_pipeline.qa_discovery_pipeline_validator import (
        validate_run, validate_plan, validate_bundle, _self_test,
    )

    # Self-test first
    if not _self_test():
        raise RuntimeError("Discovery pipeline validator self-test failed")

    # Valid run
    with open(run_v, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_run(data)
    if not result.ok:
        raise RuntimeError(
            f"run_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative run: FAIL step missing invariant_diff -> RESULT_INCOMPLETE
    with open(run_n, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_run(data)
    if result.ok:
        raise RuntimeError("run_invalid_missing_invariant_diff.json should FAIL but passed")
    if result.fail_type != "RESULT_INCOMPLETE":
        raise RuntimeError(
            f"run_invalid_missing_invariant_diff.json: expected "
            f"RESULT_INCOMPLETE, got {result.fail_type}"
        )

    # Valid plan
    with open(plan_v, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_plan(data)
    if not result.ok:
        raise RuntimeError(
            f"plan_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative plan: canonical_json=false -> NONDETERMINISTIC_PLAN
    with open(plan_n, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_plan(data)
    if result.ok:
        raise RuntimeError("plan_invalid_nondeterministic.json should FAIL but passed")
    if result.fail_type != "NONDETERMINISTIC_PLAN":
        raise RuntimeError(
            f"plan_invalid_nondeterministic.json: expected "
            f"NONDETERMINISTIC_PLAN, got {result.fail_type}"
        )

    # Valid bundle
    with open(bundle_v, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_bundle(data)
    if not result.ok:
        raise RuntimeError(
            f"bundle_valid.json should PASS but got {result.fail_type}: "
            f"{json.dumps(result.invariant_diff, sort_keys=True)}"
        )

    # Negative bundle: empty this_bundle_hash -> HASH_CHAIN_INVALID
    with open(bundle_n, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_bundle(data)
    if result.ok:
        raise RuntimeError("bundle_invalid_bad_chain.json should FAIL but passed")
    if result.fail_type != "HASH_CHAIN_INVALID":
        raise RuntimeError(
            f"bundle_invalid_bad_chain.json: expected "
            f"HASH_CHAIN_INVALID, got {result.fail_type}"
        )

    # E2E negative: ci_check.py --allow_fail must reject FAIL runs without receipt.
    # This proves permissive mode is real.
    e2e_dir = os.path.join(fixtures_dir, "e2e_neg_no_receipt")
    if not os.path.isdir(e2e_dir):
        return "missing fixture dir: e2e_neg_no_receipt"

    ci_check = os.path.join(dp_dir, "ci_check.py")
    if not os.path.exists(ci_check):
        return "missing ci_check.py"

    import subprocess
    proc = subprocess.run(
        [sys.executable, ci_check, "--out_dir", e2e_dir, "--allow_fail"],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=15,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 1:
        raise RuntimeError(
            f"ci_check.py --allow_fail expected exit code 1 (policy fail) "
            f"but got {proc.returncode}\n" + combined.strip()
        )
    if "RUN_FAIL_NO_RECEIPT" not in combined:
        raise RuntimeError(
            "ci_check.py --allow_fail failed for unexpected reason; "
            "expected RUN_FAIL_NO_RECEIPT marker.\n"
            + combined.strip()
        )
    if "FAIL run must reference receipt" not in combined:
        raise RuntimeError(
            "Expected receipt policy reason missing.\n" + combined.strip()
        )

    return None


# ===================================================================
# Family [34]: QA Rule 30 Certified Discovery
# ===================================================================

def _validate_rule30_cert_if_present(base_dir: str) -> Optional[str]:
    """Validate Rule 30 cert pack: schema, validator self-tests, cert validation."""
    r30_dir = os.path.join(base_dir, "qa_rule30")
    if not os.path.isdir(r30_dir):
        return "qa_rule30 directory not found"

    validator = os.path.join(r30_dir, "qa_rule30_cert_validator.py")
    if not os.path.isfile(validator):
        return "qa_rule30_cert_validator.py not found"

    # 1. Self-tests
    import subprocess
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        raise RuntimeError(f"Rule 30 cert validator self-tests failed:\n{proc.stdout}\n{proc.stderr}")
    # Print self-test output
    for line in proc.stdout.strip().split("\n"):
        print(line)

    # 2. Cert pack validation (v1-v4 if present)
    cert_data = None  # will hold latest validated cert for verifier gate
    latest_cert_slug = None
    certpack_dir = None
    for version_slug in [
        "rule30_nonperiodicity_v1",
        "rule30_nonperiodicity_v2",
        "rule30_nonperiodicity_v3",
        "rule30_nonperiodicity_v4",
    ]:
        cp_dir = os.path.join(r30_dir, "certpacks", version_slug)
        cp_cert = os.path.join(cp_dir, "QA_RULE30_NONPERIODICITY_CERT.v1.json")
        if not os.path.isfile(cp_cert):
            continue

        proc = subprocess.run(
            [sys.executable, validator, "cert", cp_cert, "--ci"],
            capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            raise RuntimeError(f"Cert validation failed ({version_slug}):\n{proc.stdout}\n{proc.stderr}")
        print(f"  {proc.stdout.strip()}")

        # 3. Validate each witness manifest with file hashes
        with open(cp_cert, "r", encoding="utf-8") as f:
            cert_data = json.load(f)
        latest_cert_slug = version_slug
        certpack_dir = cp_dir
        for ref in cert_data.get("witness_refs", []):
            man_path = os.path.join(cp_dir, ref["manifest_path"])
            if not os.path.isfile(man_path):
                raise RuntimeError(f"Missing manifest: {man_path}")
            proc = subprocess.run(
                [sys.executable, validator, "manifest", man_path,
                 "--verify-files", "--ci"],
                capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Manifest validation failed for {man_path}:\n"
                    f"{proc.stdout}\n{proc.stderr}")

        n_witnesses = len(cert_data.get("witness_refs", []))
        agg = cert_data.get("aggregate", {})
        print(f"  Cert pack {version_slug}: {agg.get('total_verified', '?')} periods verified "
              f"across {n_witnesses} T values, {agg.get('total_failures', '?')} failures")

    # 4. Negative fixtures — four-guard proof
    #    (exit code + filename echo + fail marker + invariant_diff anchor)
    fixtures_dir = os.path.join(r30_dir, "fixtures")
    neg_fixtures = [
        # (filename, expected_fail_type, invariant_diff_anchor)
        ("cert_neg_missing_invariant_diff.json", "MISSING_INVARIANT_DIFF", "$.invariant_diff"),
        ("cert_neg_scope_invalid.json", "SCOPE_INVALID", "P_min"),
        ("cert_neg_aggregate_mismatch.json", "AGGREGATE_MISMATCH", "total_verified"),
    ]
    for fname, expected_fail, diff_anchor in neg_fixtures:
        fpath = os.path.join(fixtures_dir, fname)
        if not os.path.isfile(fpath):
            return f"missing Rule 30 negative fixture: {fname}"
        proc = subprocess.run(
            [sys.executable, validator, "cert", fpath, "--ci"],
            capture_output=True, text=True, timeout=15)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 1:
            raise RuntimeError(
                f"Rule30 negative fixture {fname} expected exit code 1 "
                f"but got {proc.returncode}\n{combined.strip()}")
        if fname not in combined:
            raise RuntimeError(
                f"Rule30 negative fixture output does not echo filename "
                f"{fname}\n{combined.strip()}")
        if expected_fail not in combined:
            raise RuntimeError(
                f"Rule30 negative fixture {fname} missing expected marker "
                f"{expected_fail}\n{combined.strip()}")
        if diff_anchor not in combined:
            raise RuntimeError(
                f"Rule30 negative fixture {fname} missing invariant_diff "
                f"anchor ({diff_anchor})\n{combined.strip()}")

    # 5. File-hash binding proof: manifest with wrong witness hash
    hash_neg_dir = os.path.join(fixtures_dir, "manifest_neg_hash_mismatch")
    hash_neg_manifest = os.path.join(hash_neg_dir, "MANIFEST.json")
    if not os.path.isfile(hash_neg_manifest):
        return "missing manifest_neg_hash_mismatch fixture"
    proc = subprocess.run(
        [sys.executable, validator, "manifest", hash_neg_manifest,
         "--verify-files", "--ci"],
        capture_output=True, text=True, timeout=15)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 1:
        raise RuntimeError(
            f"Hash-mismatch manifest expected exit code 1 "
            f"but got {proc.returncode}\n{combined.strip()}")
    if "HASH_MISMATCH" not in combined:
        raise RuntimeError(
            f"Hash-mismatch manifest missing HASH_MISMATCH marker\n"
            f"{combined.strip()}")
    if "expected" not in combined or "actual" not in combined:
        raise RuntimeError(
            f"Hash-mismatch manifest missing expected/actual diff\n"
            f"{combined.strip()}")

    # 6. Independent verifier must pass on shipped cert pack
    verifier = os.path.join(r30_dir, "verify_certpack.py")
    if os.path.isfile(verifier) and certpack_dir and cert_data:
        proc = subprocess.run(
            [sys.executable, verifier, certpack_dir],
            capture_output=True, text=True, timeout=60)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Independent verifier failed on cert pack {latest_cert_slug}\n"
                f"{combined.strip()}")
        if "ALL WITNESSES INDEPENDENTLY VERIFIED" not in combined:
            raise RuntimeError(
                f"Independent verifier missing success marker\n{combined.strip()}")
        # Extract verified count from output
        agg = cert_data.get("aggregate", {})
        expected_count = agg.get("total_verified", 0)
        expected_str = f"{expected_count}/{expected_count}"
        if expected_str not in combined:
            raise RuntimeError(
                f"Independent verifier did not report expected count "
                f"{expected_str}\n{combined.strip()}")

    # 7. Verifier negative fixture: false witness must be caught
    verifier_neg = os.path.join(fixtures_dir, "verifier_neg_bad_witness")
    if os.path.isdir(verifier_neg) and os.path.isfile(verifier):
        proc = subprocess.run(
            [sys.executable, verifier, verifier_neg],
            capture_output=True, text=True, timeout=30)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 1:
            raise RuntimeError(
                f"Verifier negative fixture expected exit code 1 "
                f"but got {proc.returncode}\n{combined.strip()}")
        if "VERIFICATION FAILED" not in combined:
            raise RuntimeError(
                f"Verifier negative fixture missing VERIFICATION FAILED "
                f"marker\n{combined.strip()}")
        if "center_t_plus_p mismatch" not in combined:
            raise RuntimeError(
                f"Verifier negative fixture missing 'center_t_plus_p mismatch' "
                f"detail\n{combined.strip()}")
        # Assert it identifies the specific false witness (p=1)
        if "'p': 1" not in combined and '"p": 1' not in combined:
            raise RuntimeError(
                f"Verifier negative fixture did not identify p=1 as failing\n"
                f"{combined.strip()}")

    return None


def _validate_mapping_protocol_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_MAPPING_PROTOCOL.v1 family (schema + validator + fixtures).

    Returns:
        None on success,
        skip reason string if family not present.
    Raises:
        Exception on validation failure.
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_mapping_protocol", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_mapping_protocol/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_mapping_protocol self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_mapping_protocol_ref_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_MAPPING_PROTOCOL_REF.v1 family (schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_mapping_protocol_ref", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_mapping_protocol_ref/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_mapping_protocol_ref self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_ebm_navigation_cert_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_EBM_NAVIGATION_CERT.v1 family (schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_ebm_navigation_cert", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_ebm_navigation_cert/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_ebm_navigation_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_energy_capability_separation_cert_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_ENERGY_CAPABILITY_SEPARATION_CERT.v1 family (schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_energy_capability_separation_cert", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_energy_capability_separation_cert/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_energy_capability_separation_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_ebm_verifier_bridge_cert_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_EBM_VERIFIER_BRIDGE_CERT.v1 family (schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_ebm_verifier_bridge_cert", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_ebm_verifier_bridge_cert/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_ebm_verifier_bridge_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_artexplorer_scene_adapter_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_ARTEXPLORER_SCENE_ADAPTER.v1 family (schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_artexplorer_scene_adapter_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_artexplorer_scene_adapter_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_artexplorer_scene_adapter_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_artexplorer_scene_adapter_v2_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_ARTEXPLORER_SCENE_ADAPTER.v2 family (exact substrate, schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_artexplorer_scene_adapter_v2", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_artexplorer_scene_adapter_v2/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_artexplorer_scene_adapter_v2 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_rational_trig_type_system_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_RATIONAL_TRIG_TYPE_SYSTEM.v1 family (schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_rational_trig_type_system_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_rational_trig_type_system_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_rational_trig_type_system_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_geogebra_scene_adapter_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_GEOGEBRA_SCENE_ADAPTER.v1 family (exact substrate, Z/Q typed coordinates, LCM lift).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_geogebra_scene_adapter_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_geogebra_scene_adapter_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_geogebra_scene_adapter_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_threejs_scene_adapter_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_THREEJS_SCENE_ADAPTER.v1 family (float64 substrate, schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_threejs_scene_adapter_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_threejs_scene_adapter_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_threejs_scene_adapter_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_kona_ebm_mnist_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_KONA_EBM_MNIST_CERT.v1 family (RBM CD-1 on MNIST, deterministic trace, typed failure algebra).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_kona_ebm_mnist_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_kona_ebm_mnist_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_kona_ebm_mnist_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


# Populate FAMILY_SWEEPS now that all validator functions are defined.
# To add a new family: add ONE entry here. That's it.
# Format: (id, label, validator_fn, pass_description, doc_slug, family_root_rel, must_have_dedicated_root)
#
# Gate 0 (mapping protocol intake) is enforced per `family_root_rel`.
# Using "." shares a single mapping protocol for multiple families (cached).
#
# Policy B (recommended going forward):
# - New families should set must_have_dedicated_root=True and use a dedicated directory root
#   (i.e., do not use family_root_rel="." for new families).
FAMILY_SWEEPS = [
    (18, "QA Datastore family",
     _validate_datastore_family_if_present,
     "semantics + witness + counterexamples", "18_datastore", ".", False),
    (19, "QA Topology Resonance bundle",
     _validate_topology_bundle_if_present,
     "bundle manifest verified", "19_topology_resonance", ".", False),
    (20, "QA Datastore view family",
     _validate_datastore_view_family_if_present,
     "semantics + witness + counterexamples", "20_datastore_view", ".", False),
    (21, "QA A-RAG interface family",
     _validate_arag_family_if_present,
     "semantics + witness + counterexamples", "21_arag_interface", ".", False),
    (22, "QA ingest->view bridge family",
     _validate_ingest_view_bridge_family_if_present,
     "semantics + witness + counterexamples", "22_ingest_view_bridge", ".", False),
    (23, "QA ingestion family",
     _validate_ingest_family_if_present,
     "semantics + witness + counterexamples", "23_ingestion", ".", False),
    (24, "QA SVP-CMC family",
     _validate_svp_cmc_family_if_present,
     "ledger sanity + validator demo", "24_svp_cmc", ".", False),
    (26, "QA Competency Detection family",
     _validate_competency_family_if_present,
     "bundle + metrics recompute + fixtures", "26_competency_detection", ".", False),
    (27, "QA Elliptic Correspondence bundle",
     _validate_elliptic_bundle_if_present,
     "bundle manifest verified", "27_elliptic_correspondence", ".", False),
    (28, "QA Graph Structure bundle",
     _validate_graph_structure_bundle_if_present,
     "bundle manifest verified", "28_graph_structure", ".", False),
    (29, "QA Agent Trace family",
     _validate_agent_trace_family_if_present,
     "schema + validator + fixtures (1 valid, 2 negative)", "29_agent_traces", ".", False),
    (30, "QA Agent Trace Competency Cert family",
     _validate_agent_trace_competency_cert_family_if_present,
     "schema + validator + fixtures (1 valid, 3 negative)", "30_agent_trace_competency_cert", ".", False),
    (31, "QA Math Compiler Stack family",
     _validate_math_compiler_stack_if_present,
     "validator + fixtures (6 valid, 6 negative) + optional demo_pack_v1", "31_math_compiler_stack", ".", False),
    (32, "QA Conjecture-Prove Control Loop family",
     _validate_conjecture_prove_loop_if_present,
     "validator + fixtures (3 valid, 3 negative)", "32_conjecture_prove_loop", ".", False),
    (33, "QA Discovery Pipeline family",
     _validate_discovery_pipeline_if_present,
     "validator + fixtures (3 valid, 3 negative) + E2E ci_check", "33_discovery_pipeline", ".", False),
    (34, "QA Rule 30 Certified Discovery",
     _validate_rule30_cert_if_present,
     "self-tests + cert pack + manifests + cert negatives + hash-mismatch + independent verifier + verifier negative", "34_rule30_cert", ".", False),
    (35, "QA Mapping Protocol family",
     _validate_mapping_protocol_family_if_present,
     "schema + validator + fixtures", "35_mapping_protocol", "../qa_mapping_protocol", True),
    (36, "QA Mapping Protocol REF family",
     _validate_mapping_protocol_ref_family_if_present,
     "schema + validator + fixtures", "36_mapping_protocol_ref", "../qa_mapping_protocol_ref", True),
    (37, "QA EBM Navigation Cert family",
     _validate_ebm_navigation_cert_family_if_present,
     "schema + validator + fixtures", "37_ebm_navigation_cert", "../qa_ebm_navigation_cert", True),
    (38, "QA Energy-Capability Separation Cert family",
     _validate_energy_capability_separation_cert_family_if_present,
     "schema + validator + fixtures", "38_energy_capability_separation", "../qa_energy_capability_separation_cert", True),
    (39, "QA EBM Verifier Bridge Cert family",
     _validate_ebm_verifier_bridge_cert_family_if_present,
     "schema + validator + fixtures", "39_ebm_verifier_bridge_cert", "../qa_ebm_verifier_bridge_cert", True),
    (44, "QA Rational Trig Type System family",
     _validate_rational_trig_type_system_family_if_present,
     "schema + validator + fixtures", "44_rational_trig_type_system", "../qa_rational_trig_type_system_v1", True),
    (45, "QA ARTexplorer Scene Adapter family",
     _validate_artexplorer_scene_adapter_family_if_present,
     "schema + validator + fixtures", "45_artexplorer_scene_adapter", "../qa_artexplorer_scene_adapter_v1", True),
    (50, "QA ARTexplorer Scene Adapter v2 (exact substrate) family",
     _validate_artexplorer_scene_adapter_v2_family_if_present,
     "schema + validator + fixtures (exact arithmetic)", "50_artexplorer_scene_adapter_v2_exact", "../qa_artexplorer_scene_adapter_v2", True),
    (55, "QA Three.js Scene Adapter family",
     _validate_threejs_scene_adapter_family_if_present,
     "schema + validator + fixtures", "55_threejs_scene_adapter", "../qa_threejs_scene_adapter_v1", True),
    (56, "QA GeoGebra Scene Adapter family",
     _validate_geogebra_scene_adapter_family_if_present,
     "schema + validator + fixtures (exact substrate, Z/Q typed coords)", "56_geogebra_scene_adapter_exact", "../qa_geogebra_scene_adapter_v1", True),
    (62, "QA Kona EBM MNIST family",
     _validate_kona_ebm_mnist_family_if_present,
     "schema + validator + fixtures (RBM CD-1, real MNIST training, typed failure algebra)", "62_kona_ebm_mnist", "../qa_kona_ebm_mnist_v1", True),
]


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
    import argparse

    parser = argparse.ArgumentParser(description="QA Meta-Validator")
    parser.add_argument("file", nargs="?", help="Certificate file to validate")
    parser.add_argument("--fast", action="store_true",
                        help="Run only fast integrity gates (manifest checks)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Enable strict repo policy guards (e.g., forbid must_have_dedicated_root "
            "families from sharing a family_root_rel)."
        ),
    )
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    # If a file path is provided, validate that file
    if args.file:
        with open(args.file) as f:
            result = validate_json(f.read())
        print(result.to_json())
        sys.exit(0 if result.is_valid else 1)

    # Fast mode: only run manifest integrity checks
    if args.fast:
        print("=== META-VALIDATOR FAST MODE (manifest integrity only) ===\n")
        import subprocess

        fast_results = {"ok": True, "modules": {}}

        # FST manifest check (hardened: uses --check-manifest mode)
        fst_validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "qa_fst", "qa_fst_validate.py")
        if os.path.exists(fst_validator):
            fst_result = subprocess.run(
                [sys.executable, fst_validator, "--check-manifest", "--json"],
                capture_output=True, text=True)
            if fst_result.returncode == 0:
                fst_json = json.loads(fst_result.stdout)
                fst_ok = fst_json.get("ok", False)
                fast_results["modules"]["fst"] = fst_json
                print(f"[FST] {'PASS' if fst_ok else 'FAIL'} "
                      f"(hash_spec={fst_json.get('hash_spec_id', 'missing')})")
                if not fst_ok:
                    fast_results["ok"] = False
            else:
                fast_results["modules"]["fst"] = {"ok": False, "error": "subprocess failed"}
                fast_results["ok"] = False
                print(f"[FST] FAIL (subprocess error)")

        # Kayser manifest check
        kayser_validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "qa_kayser", "qa_kayser_validate.py")
        if os.path.exists(kayser_validator):
            kayser_result = subprocess.run(
                [sys.executable, kayser_validator, "--check-manifest", "--json"],
                capture_output=True, text=True)
            if kayser_result.returncode == 0:
                kayser_json = json.loads(kayser_result.stdout)
                kayser_ok = kayser_json.get("ok", False)
                fast_results["modules"]["kayser"] = kayser_json
                print(f"[Kayser] {'PASS' if kayser_ok else 'FAIL'} ({len(kayser_json.get('checks', []))} checks)")
                if not kayser_ok:
                    fast_results["ok"] = False
            else:
                fast_results["modules"]["kayser"] = {"ok": False, "error": "subprocess failed"}
                fast_results["ok"] = False
                print(f"[Kayser] FAIL (subprocess error)")

        # Topology Resonance bundle check
        topology_result = _topology_bundle_status(os.path.dirname(os.path.abspath(__file__)))
        fast_results["modules"]["topology_bundle"] = topology_result
        if topology_result.get("skipped"):
            print("[Topology Bundle] SKIP (bundle not found)")
        else:
            topology_ok = topology_result.get("ok", False)
            print(f"[Topology Bundle] {'PASS' if topology_ok else 'FAIL'} "
                  f"({topology_result.get('artifact_count', 0)} artifacts)")
            if not topology_ok:
                fast_results["ok"] = False

        # Graph Structure bundle check
        graph_result = _graph_structure_bundle_status(os.path.dirname(os.path.abspath(__file__)))
        fast_results["modules"]["graph_structure_bundle"] = graph_result
        if graph_result.get("skipped"):
            print("[Graph Bundle] SKIP (bundle not found)")
        else:
            graph_ok = graph_result.get("ok", False)
            print(f"[Graph Bundle] {'PASS' if graph_ok else 'FAIL'} "
                  f"({graph_result.get('artifact_count', 0)} artifacts)")
            if not graph_ok:
                fast_results["ok"] = False

        # Elliptic Correspondence bundle check
        elliptic_result = _elliptic_bundle_status(os.path.dirname(os.path.abspath(__file__)))
        fast_results["modules"]["elliptic_correspondence_bundle"] = elliptic_result
        if elliptic_result.get("skipped"):
            print("[Elliptic Bundle] SKIP (bundle not found)")
        else:
            elliptic_ok = elliptic_result.get("ok", False)
            print(f"[Elliptic Bundle] {'PASS' if elliptic_ok else 'FAIL'} "
                  f"({elliptic_result.get('artifact_count', 0)} artifacts)")
            if not elliptic_ok:
                fast_results["ok"] = False

        print()
        print(f"Fast mode result: {'PASS' if fast_results['ok'] else 'FAIL'}")

        if args.json:
            print(json.dumps(fast_results, indent=2))

        sys.exit(0 if fast_results["ok"] else 1)

    # Otherwise run full self-tests against all certificate types
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

    # --- Test 13: FST module (two-phase: manifest integrity + behavioral) ---
    import subprocess
    fst_validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "qa_fst", "qa_fst_validate.py")
    if os.path.exists(fst_validator):
        print("\n--- FST MODULE (subprocess, two-phase) ---")

        # Phase A: Manifest integrity check
        fst_manifest_result = subprocess.run(
            [sys.executable, fst_validator, "--check-manifest", "--json"],
            capture_output=True, text=True)
        if fst_manifest_result.returncode == 0:
            manifest_json = json.loads(fst_manifest_result.stdout)
            manifest_ok = manifest_json.get("ok", False)
            if manifest_ok:
                print(f"[13a] FST manifest integrity: PASS "
                      f"(hash_spec={manifest_json.get('hash_spec_id', 'missing')})")
            else:
                print(f"[13a] FST manifest integrity: FAIL")
                for err in manifest_json.get("errors", []):
                    print(f"      {err}")
                sys.exit(1)
        else:
            print(f"[13a] FST manifest: FAIL (exit code {fst_manifest_result.returncode})")
            sys.exit(1)

        # Phase B: Full behavioral validation
        fst_result = subprocess.run(
            [sys.executable, fst_validator, "--all", "--json"],
            capture_output=True, text=True)
        if fst_result.returncode == 0:
            fst_json = json.loads(fst_result.stdout)
            fst_status = fst_json.get("result", "UNKNOWN")
            fst_warns = len(fst_json.get("warnings", []))
            print(f"[13b] FST behavioral: {fst_status} "
                  f"(warnings={fst_warns}) -> PASS")
        else:
            print(f"[13b] FST behavioral: FAIL (exit code {fst_result.returncode})")
            print(f"      stderr: {fst_result.stderr[:200]}")
            sys.exit(1)
    else:
        print("\n[13] FST module: SKIPPED (qa_fst/qa_fst_validate.py not found)")

    # --- Test 14: Agent Security Kernel (subprocess) ---
    agent_sec_validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "qa_agent_security", "qa_agent_security.py")
    agent_sec_validator = os.path.normpath(agent_sec_validator)
    if os.path.exists(agent_sec_validator):
        print("\n--- AGENT SECURITY KERNEL (subprocess) ---")
        sec_result = subprocess.run(
            [sys.executable, agent_sec_validator, "--validate"],
            capture_output=True, text=True)
        if sec_result.returncode == 0:
            sec_json = json.loads(sec_result.stdout)
            sec_status = sec_json.get("result", "UNKNOWN")
            sec_tests = sec_json.get("tests_run", 0)
            print(f"[14] Agent Security Kernel: {sec_status} "
                  f"({sec_tests} tests) -> PASS")
        else:
            print(f"[14] Agent Security Kernel: FAIL (exit code {sec_result.returncode})")
            print(f"     stderr: {sec_result.stderr[:200]}")
            sys.exit(1)
    else:
        print("\n[14] Agent Security Kernel: SKIPPED (qa_agent_security.py not found)")

    # --- Test 15: QA_OS_SPEC (subprocess) ---
    os_spec_validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "qa_os_spec_validate.py")
    if os.path.exists(os_spec_validator):
        print("\n--- QA_OS_SPEC UNIFICATION ARTIFACT ---")
        os_spec_result = subprocess.run(
            [sys.executable, os_spec_validator, "--json"],
            capture_output=True, text=True)
        if os_spec_result.returncode == 0:
            os_spec_json = json.loads(os_spec_result.stdout)
            os_spec_status = os_spec_json.get("result", "UNKNOWN")
            os_spec_checks = len(os_spec_json.get("checks", []))
            canonical_hash = os_spec_json.get("hashes", {}).get("canonical_sha256", "")[:16]
            print(f"[15] QA_OS_SPEC: {os_spec_status} "
                  f"({os_spec_checks} checks, hash={canonical_hash}...)")
            if os_spec_status != "PASS":
                for err in os_spec_json.get("errors", []):
                    print(f"     {err}")
                sys.exit(1)
        else:
            print(f"[15] QA_OS_SPEC: FAIL (exit code {os_spec_result.returncode})")
            print(f"     stderr: {os_spec_result.stderr[:200]}")
            sys.exit(1)
    else:
        print("\n[15] QA_OS_SPEC: SKIPPED (qa_os_spec_validate.py not found)")

    # --- Test 16: Kayser Harmonic Correspondence Module (subprocess) ---
    kayser_validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "qa_kayser", "qa_kayser_validate.py")
    if os.path.exists(kayser_validator):
        print("\n--- KAYSER HARMONIC CORRESPONDENCE MODULE (subprocess) ---")

        # Test 16a: Manifest integrity check (fast gate)
        kayser_manifest_result = subprocess.run(
            [sys.executable, kayser_validator, "--check-manifest", "--json"],
            capture_output=True, text=True)
        if kayser_manifest_result.returncode == 0:
            manifest_json = json.loads(kayser_manifest_result.stdout)
            manifest_ok = manifest_json.get("ok", False)
            if manifest_ok:
                print(f"[16a] Kayser manifest integrity: PASS "
                      f"({len(manifest_json.get('checks', []))} checks)")
            else:
                print(f"[16a] Kayser manifest integrity: FAIL")
                for err in manifest_json.get("errors", []):
                    print(f"      {err}")
                sys.exit(1)
        else:
            print(f"[16a] Kayser manifest: FAIL (exit code {kayser_manifest_result.returncode})")
            sys.exit(1)

        # Test 16b: Full behavioral validation
        kayser_result = subprocess.run(
            [sys.executable, kayser_validator, "--all", "--json"],
            capture_output=True, text=True)
        if kayser_result.returncode == 0:
            kayser_json = json.loads(kayser_result.stdout)
            kayser_passed = kayser_json.get("all_passed", False)
            kayser_merkle = kayser_json.get("merkle_root", "")[:16]
            kayser_verified = sum(c.get("verified", 0) for c in kayser_json.get("certificates", {}).values())
            kayser_total = sum(c.get("total", 0) for c in kayser_json.get("certificates", {}).values())
            if kayser_passed:
                print(f"[16b] Kayser behavioral: PASS "
                      f"({kayser_verified}/{kayser_total} verified, merkle={kayser_merkle}...)")
            else:
                print(f"[16b] Kayser behavioral: FAIL (all_passed=False)")
                sys.exit(1)
        else:
            print(f"[16b] Kayser behavioral: FAIL (exit code {kayser_result.returncode})")
            print(f"      stderr: {kayser_result.stderr[:200]}")
            sys.exit(1)
    else:
        print("\n[16] Kayser module: SKIPPED (qa_kayser/qa_kayser_validate.py not found)")

    # --- Test 17: QA Guardrail MVP (subprocess) ---
    guardrail_validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "qa_guardrail", "qa_guardrail.py")
    if os.path.exists(guardrail_validator):
        print("\n--- QA GUARDRAIL MVP ---")

        # Test 17a: Self-tests
        guardrail_result = subprocess.run(
            [sys.executable, guardrail_validator, "--validate"],
            capture_output=True, text=True)
        if guardrail_result.returncode == 0:
            guardrail_json = json.loads(guardrail_result.stdout)
            guardrail_ok = guardrail_json.get("ok", False)
            guardrail_tests = len(guardrail_json.get("tests", []))
            if guardrail_ok:
                print(f"[17a] Guardrail self-tests: PASS ({guardrail_tests} tests)")
            else:
                print(f"[17a] Guardrail self-tests: FAIL")
                for err in guardrail_json.get("errors", []):
                    print(f"      {err}")
                sys.exit(1)
        else:
            print(f"[17a] Guardrail self-tests: FAIL (exit code {guardrail_result.returncode})")
            sys.exit(1)

        # Test 17b: Golden fixtures
        fixtures_result = subprocess.run(
            [sys.executable, guardrail_validator, "--fixtures"],
            capture_output=True, text=True)
        if fixtures_result.returncode == 0:
            fixtures_json = json.loads(fixtures_result.stdout)
            fixtures_ok = fixtures_json.get("ok", False)
            fixtures_passed = fixtures_json.get("passed", 0)
            if fixtures_ok:
                print(f"[17b] Guardrail golden fixtures: PASS ({fixtures_passed} fixtures)")
            else:
                print(f"[17b] Guardrail golden fixtures: FAIL")
                for err in fixtures_json.get("errors", []):
                    print(f"      {err}")
                sys.exit(1)
        else:
            print(f"[17b] Guardrail golden fixtures: FAIL (exit code {fixtures_result.returncode})")
            sys.exit(1)

        # Test 17c: End-to-end tests (subprocess invocation like OpenClaw)
        e2e_test = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "qa_guardrail", "e2e_test.py")
        if os.path.exists(e2e_test):
            e2e_result = subprocess.run(
                [sys.executable, e2e_test, "--json"],
                capture_output=True, text=True)
            if e2e_result.returncode == 0:
                e2e_json = json.loads(e2e_result.stdout)
                e2e_ok = e2e_json.get("ok", False)
                e2e_passed = e2e_json.get("passed", 0)
                allow_count = sum(1 for e in e2e_json.get("audit_log", []) if e.get("result") == "ALLOW")
                deny_count = sum(1 for e in e2e_json.get("audit_log", []) if e.get("result") == "DENY")
                if e2e_ok:
                    print(f"[17c] Guardrail E2E tests: PASS ({e2e_passed} tests, {allow_count} ALLOW, {deny_count} DENY)")
                else:
                    print(f"[17c] Guardrail E2E tests: FAIL")
                    for test in e2e_json.get("tests", []):
                        if not test.get("passed"):
                            print(f"      {test['name']}: {test.get('details', {})}")
                    sys.exit(1)
            else:
                print(f"[17c] Guardrail E2E tests: FAIL (exit code {e2e_result.returncode})")
                sys.exit(1)
    else:
        print("\n[17] Guardrail module: SKIPPED (qa_guardrail/qa_guardrail.py not found)")

    # --- Family sweep loop (driven by FAMILY_SWEEPS) ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    _doc_gate_families = []
    _repo_root = os.path.normpath(os.path.join(base_dir, ".."))

    # --- Policy guard: shared family roots (warn; fail for must-have-dedicated-root families in --strict) ---
    root_to_families: Dict[str, List[Tuple[int, str, str, bool]]] = {}
    for fam_id, label, _validator_fn, _pass_desc, _doc_slug, family_root_rel, must_have_dedicated_root in FAMILY_SWEEPS:
        family_root_abs = os.path.normpath(os.path.join(base_dir, family_root_rel))
        root_to_families.setdefault(family_root_abs, []).append(
            (fam_id, label, family_root_rel, must_have_dedicated_root)
        )

    shared_roots = {
        root: fams
        for root, fams in root_to_families.items()
        if len(fams) > 1
    }
    if shared_roots:
        print("\n--- POLICY GUARD: SHARED FAMILY ROOTS ---")
        for root, fams in sorted(shared_roots.items(), key=lambda kv: kv[0]):
            rel_root = os.path.relpath(root, base_dir)
            fam_list = ", ".join(
                f"[{fid}] {lbl}" for fid, lbl, _rel, _must in sorted(fams, key=lambda x: x[0])
            )
            print(f"[WARN] family_root_rel={rel_root}: {fam_list}")

        if args.strict:
            offenders_shared = []
            for root, fams in shared_roots.items():
                must_fams = [(fid, lbl) for fid, lbl, _rel, must in fams if must]
                if must_fams:
                    offenders_shared.append((root, fams, must_fams))

            offenders_non_dedicated = []
            for root, fams in root_to_families.items():
                for fid, lbl, rel, must in fams:
                    if must and os.path.normpath(rel) == ".":
                        offenders_non_dedicated.append((fid, lbl, rel))

            if offenders_shared or offenders_non_dedicated:
                diff = json.dumps({
                    "check": "policy_guard_family_root_policy_b",
                    "fail_type": "POLICY_FAMILY_ROOT_VIOLATION",
                    "offenders_shared_root": [
                        {
                            "family_root_rel": os.path.relpath(root, base_dir),
                            "families": [
                                {
                                    "id": fid,
                                    "label": lbl,
                                    "family_root_rel": rel,
                                    "must_have_dedicated_root": must,
                                }
                                for fid, lbl, rel, must in sorted(fams, key=lambda x: x[0])
                            ],
                            "must_have_dedicated_root_families": [
                                {"id": fid, "label": lbl} for fid, lbl in sorted(must_fams, key=lambda x: x[0])
                            ],
                        }
                        for root, fams, must_fams in sorted(offenders_shared, key=lambda x: x[0])
                    ],
                    "offenders_non_dedicated_root": [
                        {"id": fid, "label": lbl, "family_root_rel": rel}
                        for fid, lbl, rel in sorted(offenders_non_dedicated, key=lambda x: x[0])
                    ],
                }, sort_keys=True)
                print(f"[POLICY] Policy B violation in --strict mode (dedicated roots required)")
                print(f"      invariant_diff={diff}")
                sys.exit(1)

    _mapping_protocol_cache: Dict[str, Tuple[str, str]] = {}
    for fam_id, label, validator_fn, pass_desc, doc_slug, family_root_rel, _must_have_dedicated_root in FAMILY_SWEEPS:
        print(f"\n--- {label.upper()} ---")
        try:
            # Gate 0: Mapping Protocol intake constitution
            family_root = os.path.normpath(os.path.join(base_dir, family_root_rel))
            if family_root not in _mapping_protocol_cache:
                mode, resolved = require_mapping_protocol(
                    family_root=family_root,
                    repo_root=_repo_root,
                )
                _mapping_protocol_cache[family_root] = (mode, resolved)
                print(f"[{fam_id}] Gate 0 mapping protocol: PASS (mode={mode}, resolved={resolved})")
            else:
                mode, resolved = _mapping_protocol_cache[family_root]
                print(f"[{fam_id}] Gate 0 mapping protocol: PASS (cached mode={mode})")

            skip_reason = validator_fn(base_dir)
            if skip_reason is None:
                print(f"[{fam_id}] {label}: PASS ({pass_desc})")
                _doc_gate_families.append((fam_id, doc_slug))
            else:
                print(f"[{fam_id}] {label}: SKIPPED ({skip_reason})")
        except MetaValidationError as e:
            diff = json.dumps({
                "check": "gate_0_mapping_protocol",
                "fail_type": e.fail_type,
                "family_root": os.path.normpath(os.path.join(base_dir, family_root_rel)),
                "detail": e.detail,
            }, sort_keys=True)
            print(f"[{fam_id}] Gate 0 mapping protocol: FAIL ({e.fail_type})")
            print(f"      invariant_diff={diff}")
            sys.exit(1)
        except Exception as e:
            print(f"[{fam_id}] {label}: FAIL ({e})")
            sys.exit(1)

    # --- External validation: Level 3 recompute (real data + real weights) ---
    print("\n--- EXTERNAL VALIDATION ---")
    import subprocess
    _l3_script = os.path.join(base_dir, "level3_recompute_validation.py")
    _l3_id = FAMILY_SWEEPS[-1][0] + 1  # next after last family
    _l3_env = {
        k: os.environ[k]
        for k in ("QA_L3_NTRAIN", "QA_L3_NTEST", "QA_L3_EPOCHS")
        if k in os.environ
    }

    def _l3_fail(fail_type: str, returncode: int = -1,
                 stdout_head: str = "", stderr_head: str = ""):
        diff = json.dumps({
            "check": "level3_recompute_validation",
            "fail_type": fail_type,
            "script": _l3_script,
            "mode": "ci",
            "env_overrides": _l3_env or None,
            "returncode": returncode,
            "stdout_head": stdout_head[:200],
            "stderr_head": stderr_head[:200],
        }, sort_keys=True)
        print(f"[{_l3_id}] Level 3 recompute (external): FAIL ({fail_type})")
        print(f"      invariant_diff={diff}")
        sys.exit(1)

    if not os.path.exists(_l3_script):
        _l3_fail("EXTERNAL_VALIDATION_MISSING")

    try:
        _l3_result = subprocess.run(
            [sys.executable, _l3_script, "--ci"],
            capture_output=True, text=True, timeout=60,
            cwd=base_dir,
        )
        _l3_stdout = _l3_result.stdout.strip()
        _l3_ok = _l3_result.returncode == 0 and "[PASS]" in _l3_stdout

        if _l3_ok:
            print(f"[{_l3_id}] Level 3 recompute (external): PASS")
            print(f"      {_l3_stdout}")
        else:
            _l3_fail("EXTERNAL_VALIDATION_FAIL",
                     returncode=_l3_result.returncode,
                     stdout_head=_l3_stdout,
                     stderr_head=_l3_result.stderr.strip())
    except subprocess.TimeoutExpired:
        _l3_fail("EXTERNAL_VALIDATION_TIMEOUT")

    # --- External validation: Prompt injection benchmark ---
    _pi_script = os.path.join(base_dir, "external_validation_prompt_injection.py")
    _pi_id = _l3_id + 1
    _pi_env = {
        k: os.environ[k]
        for k in (
            "QA_PI_MAX_CASES",
            "QA_PI_RECALL_MIN",
            "QA_PI_PRECISION_MIN",
            "QA_PI_MAX_TYPED_MISMATCH",
            "QA_PI_MAX_FP",
            "QA_PI_MAX_FN",
            "QA_PI_MIN_CASES",
        )
        if k in os.environ
    }

    def _pi_fail(fail_type: str, returncode: int = -1,
                 stdout_head: str = "", stderr_head: str = ""):
        diff = json.dumps({
            "check": "external_validation_prompt_injection",
            "fail_type": fail_type,
            "script": _pi_script,
            "mode": "ci",
            "env_overrides": _pi_env or None,
            "returncode": returncode,
            "stdout_head": stdout_head[:200],
            "stderr_head": stderr_head[:200],
        }, sort_keys=True)
        print(f"[{_pi_id}] Prompt injection (external): FAIL ({fail_type})")
        print(f"      invariant_diff={diff}")
        sys.exit(1)

    if not os.path.exists(_pi_script):
        _pi_fail("EXTERNAL_VALIDATION_MISSING")

    try:
        _pi_result = subprocess.run(
            [sys.executable, _pi_script, "--ci"],
            capture_output=True, text=True, timeout=45,
            cwd=base_dir,
        )
        _pi_stdout = _pi_result.stdout.strip()
        _pi_ok = _pi_result.returncode == 0 and "[PASS]" in _pi_stdout

        if _pi_ok:
            print(f"[{_pi_id}] Prompt injection (external): PASS")
            print(f"      {_pi_stdout}")
        else:
            _pi_fail_type = "EXTERNAL_VALIDATION_FAIL"
            _m = re.search(r"fail_type=([A-Z0-9_]+)", _pi_stdout)
            if _m:
                _pi_fail_type = _m.group(1)
            _pi_fail(_pi_fail_type,
                     returncode=_pi_result.returncode,
                     stdout_head=_pi_stdout,
                     stderr_head=_pi_result.stderr.strip())
    except subprocess.TimeoutExpired:
        _pi_fail("EXTERNAL_VALIDATION_TIMEOUT")

    # --- External validation: SWE-bench competency (real vendored data) ---
    _swe_script = os.path.join(base_dir, "external_validation_swe_bench_competency.py")
    _swe_id = _pi_id + 1
    _swe_env = {
        k: os.environ[k]
        for k in ("QA_SWE_MAX_TASKS", "QA_SWE_MIN_TASKS", "QA_SWE_MIN_REPOS")
        if k in os.environ
    }

    def _swe_fail(fail_type: str, returncode: int = -1,
                  stdout_head: str = "", stderr_head: str = ""):
        diff = json.dumps({
            "check": "external_validation_swe_bench_competency",
            "fail_type": fail_type,
            "script": _swe_script,
            "mode": "ci",
            "env_overrides": _swe_env or None,
            "returncode": returncode,
            "stdout_head": stdout_head[:200],
            "stderr_head": stderr_head[:200],
        }, sort_keys=True)
        print(f"[{_swe_id}] SWE-bench competency (external): FAIL ({fail_type})")
        print(f"      invariant_diff={diff}")
        sys.exit(1)

    if not os.path.exists(_swe_script):
        print(f"[{_swe_id}] SWE-bench competency (external): SKIPPED "
              f"(missing external_validation_swe_bench_competency.py)")
    else:
        try:
            _swe_result = subprocess.run(
                [sys.executable, _swe_script, "--ci"],
                capture_output=True, text=True, timeout=45,
                cwd=base_dir,
            )
            _swe_stdout = _swe_result.stdout.strip()
            _swe_ok = _swe_result.returncode == 0 and "[PASS]" in _swe_stdout

            if _swe_ok:
                print(f"[{_swe_id}] SWE-bench competency (external): PASS")
                print(f"      {_swe_stdout}")
            else:
                _swe_fail_type = "EXTERNAL_VALIDATION_FAIL"
                _m = re.search(r"fail_type=([A-Z0-9_]+)", _swe_stdout)
                if _m:
                    _swe_fail_type = _m.group(1)
                _swe_fail(_swe_fail_type,
                          returncode=_swe_result.returncode,
                          stdout_head=_swe_stdout,
                          stderr_head=_swe_result.stderr.strip())
        except subprocess.TimeoutExpired:
            _swe_fail("EXTERNAL_VALIDATION_TIMEOUT")

    # --- Doc gate (derived from FAMILY_SWEEPS — no second list to maintain) ---
    print("\n--- HUMAN-TRACT DOC GATE ---")
    _doc_gate_pass = True
    _gate_id = FAMILY_SWEEPS[-1][0] + 4
    _docs_dir = os.path.normpath(os.path.join(base_dir, "..", "docs", "families"))
    _readme_path = os.path.join(_docs_dir, "README.md")
    if not os.path.isdir(_docs_dir):
        print(f"[{_gate_id}] Doc gate: FAIL (docs/families/ directory missing)")
        _doc_gate_pass = False
    else:
        for fam_id, doc_slug in _doc_gate_families:
            doc_file = f"{doc_slug}.md"
            if not os.path.exists(os.path.join(_docs_dir, doc_file)):
                print(f"  FAIL: missing docs/families/{doc_file} for family [{fam_id}]")
                _doc_gate_pass = False
        if os.path.exists(_readme_path):
            with open(_readme_path, "r", encoding="utf-8") as _rf:
                _readme_text = _rf.read()
            for fam_id, doc_slug in _doc_gate_families:
                doc_file = f"{doc_slug}.md"
                if doc_file not in _readme_text:
                    print(f"  FAIL: docs/families/README.md missing link to {doc_file}")
                    _doc_gate_pass = False
        else:
            print(f"  FAIL: docs/families/README.md missing")
            _doc_gate_pass = False
    if _doc_gate_pass:
        print(f"[{_gate_id}] Human-tract doc gate: PASS ({len(_doc_gate_families)} families documented)")
    else:
        print(f"[{_gate_id}] Human-tract doc gate: FAIL")
        sys.exit(1)

    # --- Demo smoke test ---
    _demo_gate_id = _gate_id + 1
    _demo_repo_root = os.path.normpath(os.path.join(_pkg_dir, ".."))
    _demo_script = os.path.join(_demo_repo_root, "demos", "qa_family_demo.py")
    import subprocess as _subprocess
    _demo_proc = _subprocess.run(
        [sys.executable, _demo_script, "--all", "--ci"],
        capture_output=True, text=True, cwd=_demo_repo_root,
    )
    if _demo_proc.returncode == 0:
        print(f"[{_demo_gate_id}] Demo smoke test (--all --ci): PASS")
    else:
        print(f"[{_demo_gate_id}] Demo smoke test (--all --ci): FAIL (exit {_demo_proc.returncode})")
        if _demo_proc.stdout.strip():
            print(_demo_proc.stdout.strip())
        sys.exit(1)
