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

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"

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
        # The bundled execution environment does not always include jsonschema.
        # Treat schema validation as a soft skip so the deterministic structural
        # checks can still run.
        return

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

SPINE_V1_FAMILIES = [
    "qa_kona_ebm_qa_native_orbit_reg_v1",
]
_SPINE_V1_REQUIRED_FIXTURES = [
    "fixtures/invalid_negative_generator_curvature.json",
    "fixtures/invalid_max_dev_spike_epoch.json",
    "fixtures/invalid_min_kappa_epoch.json",
]


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
                capture_output=True, text=True, timeout=120)
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


def _validate_experiment_protocol_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_EXPERIMENT_PROTOCOL.v1 family (schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_experiment_protocol", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_experiment_protocol/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test", "--json"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_experiment_protocol self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    payload = json.loads((proc.stdout or "").strip() or "{}")
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_experiment_protocol self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_benchmark_protocol_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_BENCHMARK_PROTOCOL.v1 family (schema + validator + fixtures).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_benchmark_protocol", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_benchmark_protocol/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test", "--json"],
        capture_output=True, text=True, timeout=30,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_benchmark_protocol self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    payload = json.loads((proc.stdout or "").strip() or "{}")
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_benchmark_protocol self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
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


def _validate_reachability_descent_run_cert_family_if_present(base_dir: str) -> Optional[str]:
    """QA Reachability Descent Run Cert family (generator-relative training traces)."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_reachability_descent_run_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_reachability_descent_run_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_reachability_descent_run_cert_v1 self-test failed:\n"
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


def _validate_kona_ebm_qa_native_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_KONA_EBM_QA_NATIVE_CERT.v1 family (QA orbit manifold as RBM latent space,
    orbit alignment analysis, ORBIT_MAP_VIOLATION and TRACE_HASH_MISMATCH obstructions).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_kona_ebm_qa_native_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_kona_ebm_qa_native_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_kona_ebm_qa_native_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_kona_ebm_qa_native_orbit_reg_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_KONA_EBM_QA_NATIVE_ORBIT_REG_CERT.v1 family (QA orbit-coherence regularizer,
    orbit alignment + permutation gap analysis, REGULARIZER_NUMERIC_INSTABILITY obstruction).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_kona_ebm_qa_native_orbit_reg_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_kona_ebm_qa_native_orbit_reg_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_kona_ebm_qa_native_orbit_reg_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None




def _validate_curvature_stress_test_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_CURVATURE_STRESS_TEST_BUNDLE.v1 family (cross-family κ universality,
    monoidal bottleneck law, kappa sign prediction alignment).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_curvature_stress_test_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_curvature_stress_test_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_curvature_stress_test_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_guarded_operator_category_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_GUARDED_OPERATOR_CATEGORY_CERT.v1 family (matrix embedding +
    guarded partial-map obstructions).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_guarded_operator_category_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_guarded_operator_category_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_guarded_operator_category_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_structural_algebra_cert_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_STRUCTURAL_ALGEBRA_CERT.v1 family (bounded normal forms,
    uniqueness audits, scaling components, guarded contraction checks).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_structural_algebra_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_structural_algebra_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_structural_algebra_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_component_decomposition_cert_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_COMPONENT_DECOMPOSITION_CERT.v1 family (gcd component decomposition,
    scaled-seed roundtrip, and nu power-of-two contraction characterization).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_component_decomposition_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_component_decomposition_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_component_decomposition_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_algebra_bridge_cert_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_ALGEBRA_BRIDGE_CERT.v1 family (semantic anchor for generator
    definitions, word application convention, and component bridge invariants).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_algebra_bridge_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_algebra_bridge_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_algebra_bridge_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_failure_algebra_structure_cert_family_if_present(base_dir: str) -> Optional[str]:
    """
    Validate QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1 family (finite failure tag
    algebra: poset + join-semilattice + monotone associative composition).
    """
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_failure_algebra_structure_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_failure_algebra_structure_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_failure_algebra_structure_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_neighborhood_sufficiency_cert_family_if_present(base_dir: str) -> Optional[str]:
    """QA Neighborhood Sufficiency Cert family [77]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_neighborhood_sufficiency_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_neighborhood_sufficiency_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test", "--json"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_neighborhood_sufficiency_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    data = json.loads(proc.stdout)
    if not data.get("ok"):
        failures = [f["fixture"] for f in data.get("fixtures", []) if not (f["ok"] == f["expected_ok"])]
        raise RuntimeError(f"self-test fixture mismatches: {failures}")
    return None


def _validate_locality_boundary_cert_family_if_present(base_dir: str) -> Optional[str]:
    """QA Locality Boundary Cert family [78]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_locality_boundary_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_locality_boundary_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test", "--json"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_locality_boundary_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    data = json.loads(proc.stdout)
    if not data.get("ok"):
        failures = [f["fixture"] for f in data.get("fixtures", []) if not (f["ok"] == f["expected_ok"])]
        raise RuntimeError(f"self-test fixture mismatches: {failures}")
    return None


def _validate_locality_regime_sep_cert_family_if_present(base_dir: str) -> Optional[str]:
    """QA Locality Regime Separator Cert family [79]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_locality_regime_sep_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_locality_regime_sep_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test", "--json"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_locality_regime_sep_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    data = json.loads(proc.stdout)
    if not data.get("ok"):
        failures = [f["fixture"] for f in data.get("fixtures", []) if not (f["ok"] == f["expected_ok"])]
        raise RuntimeError(f"self-test fixture mismatches: {failures}")
    return None


def _validate_energy_cert_v1_1_family_if_present(base_dir: str) -> Optional[str]:
    """QA Energy Cert v1.1 family [80]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_energy_cert_v1_1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_energy_cert_v1_1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test", "--json"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_energy_cert_v1_1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    data = json.loads(proc.stdout)
    if not data.get("ok"):
        failures = [f["fixture"] for f in data.get("fixtures", []) if not (f["ok"] == f["expected_ok"])]
        raise RuntimeError(f"self-test fixture mismatches: {failures}")
    return None


def _validate_episode_regime_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA Episode Regime Transitions Cert family [81]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_episode_regime_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_episode_regime_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test", "--json"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_episode_regime_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    data = json.loads(proc.stdout)
    if not data.get("ok"):
        failures = [f["fixture"] for f in data.get("fixtures", []) if not (f["ok"] == f["expected_ok"])]
        raise RuntimeError(f"self-test fixture mismatches: {failures}")
    return None


def _validate_bsd_local_euler_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA BSD Local Euler Cert family [82]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_bsd_local_euler_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_bsd_local_euler_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_bsd_local_euler_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_bsd_local_euler_batch_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA BSD Local Euler Batch Cert family [83]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_bsd_local_euler_batch_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_bsd_local_euler_batch_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_bsd_local_euler_batch_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_bsd_partial_lseries_proxy_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA BSD Partial L-series Proxy Cert family [84]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_bsd_partial_lseries_proxy_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_bsd_partial_lseries_proxy_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_bsd_partial_lseries_proxy_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_bsd_rank_squeeze_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA BSD Rank Squeeze Cert family [85]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_bsd_rank_squeeze_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_bsd_rank_squeeze_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_bsd_rank_squeeze_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_generator_failure_unification_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA Generator-Failure Algebra Unification Cert family [86]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_generator_failure_unification_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_generator_failure_unification_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_generator_failure_unification_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_failure_compose_operator_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA Failure Compose Operator Cert family [87]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_failure_compose_operator_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_failure_compose_operator_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_failure_compose_operator_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_failure_algebra_structure_classification_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA Failure Algebra Structure Classification Cert family [88]."""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_failure_algebra_structure_classification_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_failure_algebra_structure_classification_cert_v1/validator.py"

    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_failure_algebra_structure_classification_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_qalm_curvature_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA QALM Curvature Cert family [89]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_qalm_curvature_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_qalm_curvature_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_qalm_curvature_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_fairness_demographic_parity_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA Fairness Demographic Parity Cert family [90]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_fairness_demographic_parity_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_fairness_demographic_parity_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_fairness_demographic_parity_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_fairness_equalized_odds_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA Fairness Equalized Odds Cert family [91]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_fairness_equalized_odds_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_fairness_equalized_odds_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_fairness_equalized_odds_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_safety_prompt_injection_refusal_cert_v1_family_if_present(base_dir: str) -> Optional[str]:
    """QA Safety Prompt Injection Refusal Cert family [92]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_safety_prompt_injection_refusal_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_safety_prompt_injection_refusal_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_safety_prompt_injection_refusal_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_gnn_mp_curvature_cert_family(base_dir: str) -> Optional[str]:
    """QA GNN Message-Passing Curvature Cert family [93]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_gnn_message_passing_curvature_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_gnn_message_passing_curvature_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_gnn_message_passing_curvature_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_attn_curvature_cert_family(base_dir: str) -> Optional[str]:
    """QA Attention Layer Curvature Cert family [94]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_attn_curvature_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_attn_curvature_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_attn_curvature_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_qarm_curvature_cert_family(base_dir: str) -> Optional[str]:
    """QA QARM Curvature Cert family [95]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_qarm_curvature_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_qarm_curvature_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_qarm_curvature_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    return None


def _validate_symbolic_search_curvature_cert_family(base_dir: str) -> Optional[str]:
    """QA Symbolic Search Curvature Cert family [96]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_symbolic_search_curvature_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_symbolic_search_curvature_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_symbolic_search_curvature_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_symbolic_search_curvature_cert_v1 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_symbolic_search_curvature_cert_v1 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None












def _validate_obstruction_aware_planner_family(base_dir: str) -> Optional[str]:
    """QA Obstruction-Aware Planner family [113] — planner must prune before search on forbidden targets."""
    import subprocess
    pl_dir = os.path.join(base_dir, "qa_obstruction_aware_planner")
    validator = os.path.join(pl_dir, "qa_obstruction_aware_planner_validate.py")
    if not os.path.exists(validator):
        return "missing qa_obstruction_aware_planner/qa_obstruction_aware_planner_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=pl_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_obstruction_aware_planner self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_obstruction_aware_planner self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_obstruction_aware_planner self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_public_overview_doc_family(base_dir: str) -> Optional[str]:
    """QA Public Overview Doc family [120] — presentation-grade export derived from [119]."""
    import subprocess
    po_dir = os.path.join(base_dir, "qa_public_overview_doc")
    validator = os.path.join(po_dir, "qa_public_overview_doc_validate.py")
    if not os.path.exists(validator):
        return "missing qa_public_overview_doc/qa_public_overview_doc_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=po_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_public_overview_doc self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_public_overview_doc self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_public_overview_doc self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_engineering_core_cert_family(base_dir: str) -> Optional[str]:
    """QA Engineering Core Cert family [121] — maps classical engineering systems to QA spec; EC11 catches arithmetic obstructions invisible to Kalman rank analysis."""
    import subprocess
    ec_dir = os.path.join(base_dir, "qa_engineering_core_cert")
    validator = os.path.join(ec_dir, "qa_engineering_core_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_engineering_core_cert/qa_engineering_core_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=ec_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_engineering_core_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_engineering_core_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_engineering_core_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_spread_period_cert_family(base_dir: str) -> Optional[str]:
    """QA Spread Period Cert family [128] — certifies that the QA cosmos orbit period for modulus m equals the Pisano period π(m) of the Fibonacci sequence mod m (= order of Fibonacci shift F=[[0,1],[1,1]] in GL₂(Z/mZ)); π(9)=24, π(7)=16, π(3)=8; spread polynomial S_n(s) cycles after π(m) steps; checks SP1-SP5 (schema, Pisano period, F^P≡I, minimality, orbit_type); 2 PASS (m=9 period=24, m=7 period=16) + 1 FAIL (PISANO_PERIOD_MISMATCH+MATRIX_PERIOD_WRONG: claimed period=12 for m=9 — projective vs linear order confusion); self-test ok"""
    import subprocess
    spc_dir   = os.path.join(base_dir, "qa_spread_period_cert_v1")
    validator = os.path.join(spc_dir, "qa_spread_period_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_spread_period_cert_v1/qa_spread_period_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=spc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_spread_period_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_spread_period_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_spread_period_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_projection_obstruction_cert_family(base_dir: str) -> Optional[str]:
    """QA Projection Obstruction Cert family [129] — certifies the distinction between native symbolic closure, discrete representation-basis mismatch, and physical device realization; representation debt is not itself a physical-device failure. Checks IH1-IH3 + PO1-PO9 (native invariants, resolved references, representation layer structure/tags/verdicts, physical layer structure/verdict, obstruction ledger, overall verdict); 1 PASS (Arto ternary: lawful native arithmetic + representation debt + physical INCONCLUSIVE) + 2 FAIL (physical conflation under UNASSESSED status, unresolved invariant reference); self-test ok"""
    import subprocess
    poc_dir = os.path.abspath(os.path.join(base_dir, "qa_projection_obstruction_cert"))
    validator = os.path.abspath(os.path.join(poc_dir, "qa_projection_obstruction_cert_validate.py"))
    if not os.path.exists(validator):
        return "missing qa_projection_obstruction_cert/qa_projection_obstruction_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=poc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_projection_obstruction_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_projection_obstruction_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_projection_obstruction_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_hat_cert_family(base_dir: str) -> Optional[str]:
    """QA HAT Cert family [131] — certifies H. Lee Price half-angle tangents bridge to QA: HAT₁=e/d=C/(G+F), HAT₂=(d-e)/(d+e)=F/(G+C), spread s=E/G=HAT₁²/(1+HAT₁²); Fibonacci box [[e,d-e],[d,d+e]]; checks HAT_1-HAT_8+HAT_W/F; 2 PASS (fundamental 3-4-5, 5-witness general); self-test ok"""
    import subprocess
    hat_dir   = os.path.join(base_dir, "qa_hat_cert_v1")
    validator = os.path.join(hat_dir, "qa_hat_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_hat_cert_v1/qa_hat_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=hat_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_hat_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_hat_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_hat_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_pythagorean_tree_cert_family(base_dir: str) -> Optional[str]:
    """QA Pythagorean Tree Cert family [135] — certifies three Barning-Hall/Berggren generator moves in direction space: M_A=(2d-e,d), M_B=(2d+e,d), M_C=(d+2e,e); each preserves gcd=1 (Euclidean proof) + opposite parity + Pythagorean triple F²+C²=G²; k-identification theorem: ⌈d'/e'⌉=2 iff M_A, =3 iff M_B, ≥4 iff M_C (proofs: 2-e/d∈(1,2), 2+e/d∈(2,3), d/e+2>3); root (2,1) unique (all inverses give e=0 or d=0); Barning 1963 / Hall 1970 / Price 2008 (Fibonacci boxes=same moves) / Iverson Pyth-1 (Koenig series=same tree); inverse of [134] Egyptian fraction; checks PT_1-4+PT_A/B/C+PT_ROOT/W/F; 2 PASS; self-test ok"""
    import subprocess
    pt_dir    = os.path.join(base_dir, "qa_pythagorean_tree_cert_v1")
    validator = os.path.join(pt_dir, "qa_pythagorean_tree_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_pythagorean_tree_cert_v1/qa_pythagorean_tree_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=pt_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_pythagorean_tree_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_pythagorean_tree_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_pythagorean_tree_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_fibonacci_resonance_cert_family(base_dir: str) -> Optional[str]:
    """QA Fibonacci Resonance Cert family [219] — certifies MMRs preferentially select Fibonacci ratios. 60 resonances across 8+ systems (solar+exoplanet). Order-1: 33/43 (77%) Fib vs 22% expected, p<10⁻⁶. Unique ratios: 8/14 (57%) vs 31% expected, p=0.040. Fisher combined p<10⁻⁶. QA: T-operator=Fib shift makes Fib ratios deeper attractors. Tier 2→3. Corrective renumber: [163] is reserved by QA Dead Reckoning. Checks FR_1+CAT/CLASS/STAT/ORDER/CROSS/HONEST/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fr_dir    = os.path.join(base_dir, "qa_fibonacci_resonance_cert_v1")
    validator = os.path.join(fr_dir, "qa_fibonacci_resonance_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_fibonacci_resonance_cert_v1/qa_fibonacci_resonance_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=fr_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_fibonacci_resonance_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_fibonacci_resonance_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_fibonacci_resonance_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_ecef_rational_cert_family(base_dir: str) -> Optional[str]:
    """QA ECEF Rational Cert family [161] — certifies geodetic→ECEF via spreads/crosses only. X²=(N+h)²·c_φ·c_λ, Y²=(N+h)²·c_φ·s_λ, Z²=(N(1-e²)+h)²·s_φ, N²=a²/(1-e²·s_φ). 6 cities all hemispheres. X²+Y²=(N+h)²·c_φ identity. Tier 1. Checks ECEF_1+SPREAD/N/XYZ/SUM/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    ecef_dir  = os.path.join(base_dir, "qa_ecef_rational_cert_v1")
    validator = os.path.join(ecef_dir, "qa_ecef_rational_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_ecef_rational_cert_v1/qa_ecef_rational_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=ecef_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_ecef_rational_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_ecef_rational_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_ecef_rational_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_bragg_rt_cert_family(base_dir: str) -> Optional[str]:
    """QA Bragg RT Cert family [160] — certifies Bragg's law as rational trigonometry: n²Q_λ=4Q_d·s. Derivation: square nλ=2d·sinθ. Miller Q(h,k,l)=h²+k²+l². Crystal spreads: cubic=1, hexagonal γ=3/4. NaCl Cu Kα 4 reflections exact. Tier 1 algebraic identity. Checks BRT_1+BRAGG/MILLER/SPREAD/PYTH/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    brt_dir   = os.path.join(base_dir, "qa_bragg_rt_cert_v1")
    validator = os.path.join(brt_dir, "qa_bragg_rt_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_bragg_rt_cert_v1/qa_bragg_rt_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=brt_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_bragg_rt_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_bragg_rt_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_bragg_rt_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_human_needs_sdt_cert_family(base_dir: str) -> Optional[str]:
    """QA Human Needs SDT Cert family [162] — certifies structural alignment between Ryan & Deci Self-Determination Theory (3 validated needs: Autonomy, Competence, Relatedness) and QA paired architecture (generators, state+derivative, reach+integral). Canonical mapping: Certainty=b, Variety=e, Significance=d, Connection=a, Growth=DeltaT, Contribution=SigmaT. 5/5 structural predictions confirmed against SDT literature (n=48550, 27 countries). Theorem NT compliant (observer projection). Source: Will Dale Apr/Oct 2025, Ryan & Deci 2022, Tony Robbins/Cloe Madanes. Checks HN_1+MAP/SDT/TYPE/PRED/NT/SRC/W/F/DERIV/DELTA/SIGMA/FT; 2 PASS; self-test ok"""
    import subprocess
    hn_dir    = os.path.join(base_dir, "qa_human_needs_sdt_cert_v1")
    validator = os.path.join(hn_dir, "qa_human_needs_sdt_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_human_needs_sdt_cert_v1/qa_human_needs_sdt_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=hn_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_human_needs_sdt_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_human_needs_sdt_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_human_needs_sdt_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_dead_reckoning_cert_family(base_dir: str) -> Optional[str]:
    """QA Dead Reckoning Cert family [163] — certifies QA T-operator as exact dead reckoning engine on mod-m lattice. T^k·(b₀,e₀) mod m via augmented matrix exponentiation. Zero computational drift (classical DR accumulates sin/cos float error). Three chromogeometric metrics per direction: G=d²+e² (blue/Euclidean), F=d²-e² (red/Minkowski), C=2de (green/area), C²+F²=G². Compass rose mod-24 = QA orbit partition. Theorem NT compliant. Tier 2. Checks DR_1+TOP/EXACT/DRIFT/CHROMO/COMPASS/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    dr_dir    = os.path.join(base_dir, "qa_dead_reckoning_cert_v1")
    validator = os.path.join(dr_dir, "qa_dead_reckoning_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_dead_reckoning_cert_v1/qa_dead_reckoning_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=dr_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_dead_reckoning_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_dead_reckoning_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_dead_reckoning_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_celestial_nav_cert_family(base_dir: str) -> Optional[str]:
    """QA Celestial Nav Cert family [165] — certifies celestial navigation sight reduction as rational trigonometry. sin²(h) = [σ₁√(s_φ·s_δ) + σ₂√(c_φ·c_δ·c_LHA)]² where σ₁,σ₂ ∈ {±1} discrete orientation flags. Sextant = spread instrument. Position circle = spread locus. Two-star fix = algebraic spread intersection. Theorem NT: continuous angles → spreads + discrete σ. 7 star witnesses. Tier 1. Checks CN_1+SIGHT/SPREAD/SIGMA/AZIMUTH/FIX/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    cn_dir    = os.path.join(base_dir, "qa_celestial_nav_cert_v1")
    validator = os.path.join(cn_dir, "qa_celestial_nav_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_celestial_nav_cert_v1/qa_celestial_nav_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=cn_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_celestial_nav_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_celestial_nav_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_celestial_nav_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_loxodrome_cert_family(base_dir: str) -> Optional[str]:
    """QA Loxodrome Cert family [166] — certifies loxodromes (rhumb lines) as QA T-operator constant-bearing paths on mod-m lattice. Period=Pisano π(m). Bearing spread=e²/G. Mercator identity s_φ=tanh²(ψ). Three orbit types partition loxodromes: cosmos/satellite/singularity. 4 path witnesses + 5 Mercator points. Tier 2 structural. Checks LX_1+PATH/BEARING/MERCATOR/ORBIT/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    lx_dir    = os.path.join(base_dir, "qa_loxodrome_cert_v1")
    validator = os.path.join(lx_dir, "qa_loxodrome_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_loxodrome_cert_v1/qa_loxodrome_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=lx_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_loxodrome_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_loxodrome_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_loxodrome_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_historical_nav_cert_family(base_dir: str) -> Optional[str]:
    """QA Historical Nav Cert family [167] — certifies 5 historical navigation systems as proto-QA integer arithmetic. Babylon (Plimpton 322 = Berggren tree), Egypt (seked = spread ratio), Polynesia (star compass = mod-32), Norse (sun stones = spread measurement), Arab (kamal = integer spread increments). Common structure: discrete states + integer arithmetic + observer projection at boundaries. Tier 2 structural. Checks HN_1+SYSTEM/SEKED/KAMAL/TRIPLE/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    hn_dir    = os.path.join(base_dir, "qa_historical_nav_cert_v1")
    validator = os.path.join(hn_dir, "qa_historical_nav_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_historical_nav_cert_v1/qa_historical_nav_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=hn_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_historical_nav_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_historical_nav_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_historical_nav_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_inertial_nav_cert_family(base_dir: str) -> Optional[str]:
    """QA Inertial Nav Cert family [170] — zero drift proof. Classical INS O(ε√N) vs QA exact 0. 3 routes × 4 noise levels. Tier 1 computational. Checks IN_1+QA_EXACT/DRIFT/ZERO/RATIO/W/F; 1 PASS+1 FAIL; self-test ok"""
    import subprocess
    d = os.path.join(base_dir, "qa_inertial_nav_cert_v1")
    v = os.path.join(d, "qa_inertial_nav_cert_validate.py")
    if not os.path.exists(v):
        return "missing qa_inertial_nav_cert_v1/qa_inertial_nav_cert_validate.py"
    proc = subprocess.run([sys.executable, v, "--self-test"], capture_output=True, text=True, timeout=60, cwd=d)
    if proc.returncode != 0:
        raise RuntimeError(f"qa_inertial_nav_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    payload = json.loads((proc.stdout or "").strip() or "{}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_inertial_nav_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_planetary_qn_cert_family(base_dir: str) -> Optional[str]:
    """QA Planetary QN Cert family [171] — solar system QN catalog. 10 bodies shape+orbital. Earth-Jupiter b=59, Earth-Uranus b=101 harmonics. Saturn prime AP. Tier 2. Checks PQ_1+TUPLE/TRIPLE/ECC/HARMONIC/W/F; 1 PASS+1 FAIL; self-test ok"""
    import subprocess
    d = os.path.join(base_dir, "qa_planetary_qn_cert_v1")
    v = os.path.join(d, "qa_planetary_qn_cert_validate.py")
    if not os.path.exists(v):
        return "missing qa_planetary_qn_cert_v1/qa_planetary_qn_cert_validate.py"
    proc = subprocess.run([sys.executable, v, "--self-test"], capture_output=True, text=True, timeout=60, cwd=d)
    if proc.returncode != 0:
        raise RuntimeError(f"qa_planetary_qn_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    payload = json.loads((proc.stdout or "").strip() or "{}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_planetary_qn_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_megalithic_cert_family(base_dir: str) -> Optional[str]:
    """QA Megalithic Cert family [178] — MY p=0.0096, Fathom p<0.0001, Fibonacci QN triangles. Tier 2+3. Checks MG_1+MY/FATHOM/TRIANGLE/HONEST/W/F; 1 PASS+1 FAIL; self-test ok"""
    import subprocess
    d = os.path.join(base_dir, "qa_megalithic_cert_v1")
    v = os.path.join(d, "qa_megalithic_cert_validate.py")
    if not os.path.exists(v):
        return "missing qa_megalithic_cert_v1/qa_megalithic_cert_validate.py"
    proc = subprocess.run([sys.executable, v, "--self-test"], capture_output=True, text=True, timeout=60, cwd=d)
    if proc.returncode != 0:
        raise RuntimeError(f"qa_megalithic_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    payload = json.loads((proc.stdout or "").strip() or "{}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_megalithic_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_paired_pisano_cert_family(base_dir: str) -> Optional[str]:
    """QA Paired Pisano Cert family [179] — Fib pairs 2.25x higher both-divide rate (0.526 vs 0.234), p=0.0017. Tier 3. Checks PP_1+PAIRS/STAT/RATIO/ORDER/MECH/HONEST/W/F; 1 PASS+1 FAIL; self-test ok"""
    import subprocess
    d = os.path.join(base_dir, "qa_paired_pisano_cert_v1")
    v = os.path.join(d, "qa_paired_pisano_cert_validate.py")
    if not os.path.exists(v):
        return "missing qa_paired_pisano_cert_v1/qa_paired_pisano_cert_validate.py"
    proc = subprocess.run([sys.executable, v, "--self-test"], capture_output=True, text=True, timeout=60, cwd=d)
    if proc.returncode != 0:
        raise RuntimeError(f"qa_paired_pisano_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    payload = json.loads((proc.stdout or "").strip() or "{}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_paired_pisano_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_ellipsoid_geodesic_cert_family(base_dir: str) -> Optional[str]:
    """QA Ellipsoid Geodesic Cert family [168] — certifies geodesic properties of WGS84 quantum ellipse in QN arithmetic. M/N=F/(d²-e²·s_φ). b/a=√F/d. I=C-F=-10039<0→ellipse. Quantum lattice: s_φ=C/G≈Tropic, e²/G=eccentricity resonance. Tier 1. Checks EG_1+QN/CURV/AXIS/DISC/LATTICE/W/F; 1 PASS+1 FAIL; self-test ok"""
    import subprocess
    eg_dir = os.path.join(base_dir, "qa_ellipsoid_geodesic_cert_v1")
    validator = os.path.join(eg_dir, "qa_ellipsoid_geodesic_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_ellipsoid_geodesic_cert_v1/qa_ellipsoid_geodesic_cert_validate.py"
    proc = subprocess.run([sys.executable, validator, "--self-test"], capture_output=True, text=True, timeout=60, cwd=eg_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"qa_ellipsoid_geodesic_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    payload = json.loads((proc.stdout or "").strip() or "{}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_ellipsoid_geodesic_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_ellipsoid_slice_cert_family(base_dir: str) -> Optional[str]:
    """QA Ellipsoid Slice Cert family [169] — certifies QA slicing of WGS84 quantum ellipse. Latitude circles R²=a²d²c_φ/(d²-e²s_φ). Meridian ellipse axis=√F/d. Chromo slices C/F/G constant curves. 24 Pisano longitude bands=time zones. Tier 1+2. Checks SL_1+LAT/MER/CHROMO/BAND/W/F; 1 PASS+1 FAIL; self-test ok"""
    import subprocess
    sl_dir = os.path.join(base_dir, "qa_ellipsoid_slice_cert_v1")
    validator = os.path.join(sl_dir, "qa_ellipsoid_slice_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_ellipsoid_slice_cert_v1/qa_ellipsoid_slice_cert_validate.py"
    proc = subprocess.run([sys.executable, validator, "--self-test"], capture_output=True, text=True, timeout=60, cwd=sl_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"qa_ellipsoid_slice_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    payload = json.loads((proc.stdout or "").strip() or "{}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_ellipsoid_slice_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_gnomonic_rt_cert_family(base_dir: str) -> Optional[str]:
    """QA Gnomonic RT Cert family [164] — certifies gnomonic map projection via rational trigonometry. Gnomonic quadrance Q=spread_c/cross_c=tan²(angular_dist). Great circles → straight lines (collinearity cross product <10⁻¹²). Berggren tree generators produce Pythagorean triples C²+F²=G² = discrete geodesic steps on cone. London tangent point, 5 cities. Tier 1 (quadrance) + Tier 2 (Berggren). Checks GN_1+QUAD/SPREAD/COLLINEAR/BERGGREN/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    gn_dir    = os.path.join(base_dir, "qa_gnomonic_rt_cert_v1")
    validator = os.path.join(gn_dir, "qa_gnomonic_rt_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_gnomonic_rt_cert_v1/qa_gnomonic_rt_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=gn_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_gnomonic_rt_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_gnomonic_rt_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_gnomonic_rt_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_wgs84_ellipse_cert_family(base_dir: str) -> Optional[str]:
    """QA WGS84 Ellipse Cert family [156] — certifies WGS84 reference ellipsoid = QA quantum ellipse. Shape QN (101,9,110,119): ecc=9/110 matches WGS84 0.081819 to 0.001%; axis ratio sqrt(12019)/110 matches to 7 sig figs. Orbit QN (59,1,60,61): ecc=1/60 matches orbital 0.01671 to 0.25%. Triple (1980,12019,12181): C*C+F*F=G*G. Tier 1 exact reformulation. Checks WGS_1+QN/TRIPLE/ECC/AXIS/ORBIT/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    wgs_dir   = os.path.join(base_dir, "qa_wgs84_ellipse_cert_v1")
    validator = os.path.join(wgs_dir, "qa_wgs84_ellipse_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_wgs84_ellipse_cert_v1/qa_wgs84_ellipse_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=wgs_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_wgs84_ellipse_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_wgs84_ellipse_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_wgs84_ellipse_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_bearden_phase_conjugate_cert_family(base_dir: str) -> Optional[str]:
    """QA Bearden Phase Conjugate Cert family [155] — certifies structural parallel between Bearden's pumped phase conjugate mirror theory ('stress is a pumper') and the QCI opposite-sign discovery. Global QCI rises (pump=coupling tightens) while local QCI drops (conjugate=trajectories scatter). QCI_gap = phase conjugation signature. Source: Will Dale 2026-04-01, Bearden scalar EM, SVP-adjacent. Checks BPC_1+MODEL/MAP/SIGN/EMP/SVP/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    bpc_dir   = os.path.join(base_dir, "qa_bearden_phase_conjugate_cert_v1")
    validator = os.path.join(bpc_dir, "qa_bearden_phase_conjugate_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_bearden_phase_conjugate_cert_v1/qa_bearden_phase_conjugate_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=bpc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_bearden_phase_conjugate_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_bearden_phase_conjugate_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_bearden_phase_conjugate_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_t_operator_coherence_cert_family(base_dir: str) -> Optional[str]:
    """QA T-Operator Coherence Cert family [154] — certifies QCI (QA Coherence Index) as domain-general structural indicator. Topographic observer → QA states → T-operator prediction → rolling accuracy = QCI. Finance Tier A: partial r=-0.22 beyond lagged RV, 84% robustness, permutation-validated. Cross-domain: EEG (dR²=+0.21), audio (r=+0.75). Frozen scripts 30-37. Checks TC_1+OBS/QCI/OOS/PARTIAL/ROBUST/W/F; 2 PASS; self-test ok"""
    import subprocess
    tc_dir    = os.path.join(base_dir, "qa_t_operator_coherence_cert_v1")
    validator = os.path.join(tc_dir, "qa_t_operator_coherence_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_t_operator_coherence_cert_v1/qa_t_operator_coherence_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=tc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_t_operator_coherence_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_t_operator_coherence_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_t_operator_coherence_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_keely_triune_cert_family(base_dir: str) -> Optional[str]:
    """QA Keely Triune Cert family [153] — maps Keely's three vibratory modes (Enharmonic/Dominant/Harmonic from svpwiki.com) to QA three orbits (Satellite/Singularity/Cosmos). DOMINANT=SINGULARITY (neutral center=fixed point), ENHARMONIC=SATELLITE (bounded=8-cycle), HARMONIC=COSMOS (expansive=24-cycle). {0,3,6} mod 9=singularity=Tesla 3-6-9. Brinton three Laws of Being. Source: svpwiki.com + Iverson; checks KT_1+MAP/PART/PERIOD/369/LCM/W; 2 PASS; self-test ok"""
    import subprocess
    kt_dir    = os.path.join(base_dir, "qa_keely_triune_cert_v1")
    validator = os.path.join(kt_dir, "qa_keely_triune_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_keely_triune_cert_v1/qa_keely_triune_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=kt_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_keely_triune_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_keely_triune_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_keely_triune_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_equilateral_triangle_cert_family(base_dir: str) -> Optional[str]:
    """QA Equilateral Triangle Cert family [152] — certifies W=d(e+a)=X+K, Y=A-D=C+E (dual definition bridging square and product layers), Z=E+K; two Eisenstein norms F²-FW+W²=Z² and Y²-YW+W²=Z²; sum rule F+Y=W; source: Iverson QA-2 Ch 7; checks ET_1+DEF/DUAL/EIS/SUM/W/F; 2 PASS; self-test ok"""
    import subprocess
    et_dir    = os.path.join(base_dir, "qa_equilateral_triangle_cert_v1")
    validator = os.path.join(et_dir, "qa_equilateral_triangle_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_equilateral_triangle_cert_v1/qa_equilateral_triangle_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=et_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_equilateral_triangle_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_equilateral_triangle_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_equilateral_triangle_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_par_number_cert_family(base_dir: str) -> Optional[str]:
    """QA Par Number Cert family [151] — certifies Iverson's Double Parity (par) system: 4-way mod-4 classification (2-par=4k+2, 3-par=4k+3, 4-par=4k, 5-par=4k+1); male²=5-par always; C=4-par, G=5-par for primitive directions; par multiplication table closed; Fib_hits observations; source: Iverson QA-2 Ch 3; checks PN_1+CLASS/SQ/QA/FIB/MULT/W/F; 2 PASS; self-test ok"""
    import subprocess
    pn_dir    = os.path.join(base_dir, "qa_par_number_cert_v1")
    validator = os.path.join(pn_dir, "qa_par_number_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_par_number_cert_v1/qa_par_number_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=pn_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_par_number_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_par_number_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_par_number_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_septenary_cert_family(base_dir: str) -> Optional[str]:
    """QA Septenary Cert family [150] — certifies {1,2,4,5,7,8}=(Z/9Z)* multiplicative group; doubling mod 9 period 6=phi(9); complement {0,3,6}=singularity; diagonal pairs sum to 9; parity cross-over; 2 is primitive root mod 9; source: Iverson QA mod-9 orbit + Grant/Ghannam Philomath Ch 1; checks SP_1+GROUP/CYCLE/COMP/DIAG/PAR/W/F; 2 PASS; self-test ok"""
    import subprocess
    sp_dir    = os.path.join(base_dir, "qa_septenary_cert_v1")
    validator = os.path.join(sp_dir, "qa_septenary_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_septenary_cert_v1/qa_septenary_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=sp_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_septenary_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_septenary_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_septenary_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_law_of_harmonics_cert_family(base_dir: str) -> Optional[str]:
    """QA Law of Harmonics Cert family [149] — certifies Iverson's formal law of harmonic resonance: two QN products sharing all but one prime factor each are harmonic; harmony ratio=min(id1,id2)/max(id1,id2); all QN products divisible by 6; Fibonacci QN chain shows adjacent harmonic pattern; source: Iverson QA-3 Ch 4; checks LH_1+ALIQ/IDEN/RATIO/DIV6/W/F; 2 PASS; self-test ok"""
    import subprocess
    lh_dir    = os.path.join(base_dir, "qa_law_of_harmonics_cert_v1")
    validator = os.path.join(lh_dir, "qa_law_of_harmonics_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_law_of_harmonics_cert_v1/qa_law_of_harmonics_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=lh_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_law_of_harmonics_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_law_of_harmonics_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_law_of_harmonics_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_sixteen_identities_cert_family(base_dir: str) -> Optional[str]:
    """QA Sixteen Identities Cert family [148] — certifies the 16 named quantities (A-L,X,W,Y,Z) of a prime Pythagorean direction and 9 universal algebraic relations: G+C=A, G-C=B, G=(A+B)/2, F²+C²=G², H²+I²=2G², L=CF/12 integer, C=4-par, G=5-par, W=X+K. Fundamental (2,1): F=3,C=4,G=5,H=7,I=1,L=1. Source: Iverson Pyth-1 Ch V; checks SI_1/2+IDEN/REL/PAR/L/W/F; 2 PASS; self-test ok"""
    import subprocess
    si_dir    = os.path.join(base_dir, "qa_sixteen_identities_cert_v1")
    validator = os.path.join(si_dir, "qa_sixteen_identities_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_sixteen_identities_cert_v1/qa_sixteen_identities_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=si_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_sixteen_identities_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_sixteen_identities_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_sixteen_identities_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_synchronous_harmonics_cert_family(base_dir: str) -> Optional[str]:
    """QA Synchronous Harmonics Cert family [147] — certifies coprime synchronization (periods m,n coprime → sync at m×n; non-coprime → LCM<product), par-based interference (3-par LOW at 1/4, 5-par HIGH at 1/4; same-par SUPPORT, cross-par OPPOSE), and QN product divisibility by 6 (among b,e,d at least one even and one div by 3). Source: Iverson Pyth-2 Ch XIII, QA-2 Ch 6, QA-3 Ch 4; checks SH_1+SYNC/PAR/PROD6/W/F; 2 PASS; self-test ok"""
    import subprocess
    sh_dir    = os.path.join(base_dir, "qa_synchronous_harmonics_cert_v1")
    validator = os.path.join(sh_dir, "qa_synchronous_harmonics_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_synchronous_harmonics_cert_v1/qa_synchronous_harmonics_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=sh_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_synchronous_harmonics_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_synchronous_harmonics_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_synchronous_harmonics_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_path_scale_cert_family(base_dir: str) -> Optional[str]:
    """QA Path Scale Cert family [146] — certifies G=d^2+e^2 growth profiles along Pythagorean-tree generator paths. UNIFORM_B paths grow exponentially: G ratio converges to 3+2*sqrt(2)=5.828... (silver ratio squared, from M_B dominant eigenvalue 1+sqrt(2)). UNIFORM_A and UNIFORM_C paths grow polynomially (G ratio -> 1). All forward paths from (2,1) have G monotone increasing. 8-step Pell chain converges by step 4 (6 d.p.). Source: Pell equation theory, certs [135] Pythagorean Tree, [141] Pell Norm, [145] Path Shape; checks SC_1/2+GROWTH/RATIO/CONV_B/W/F; 2 PASS; self-test ok"""
    import subprocess
    sc_dir    = os.path.join(base_dir, "qa_path_scale_cert_v1")
    validator = os.path.join(sc_dir, "qa_path_scale_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_path_scale_cert_v1/qa_path_scale_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=sc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_path_scale_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_path_scale_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_path_scale_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_path_shape_cert_family(base_dir: str) -> Optional[str]:
    """QA Path Shape Cert family [145] — classifies generator-sequence shapes in the Pythagorean tree into four classes: UNIFORM_A (only M_A, consecutive integers), UNIFORM_B (only M_B, Pell chain with alternating norm), UNIFORM_C (only M_C, constant e, arithmetic d), MIXED (two or more generators). Invariants: primitivity preserved at every step; UNIFORM_B: Pell norm alternates sign; UNIFORM_C: e constant. Source: Barning 1963/Hall 1970/Price 2008, certs [134] Egyptian Fraction, [135] Pythagorean Tree, [141] Pell Norm; checks PS_1/2+CLASS/INV_B/INV_C/W/F; 2 PASS; self-test ok"""
    import subprocess
    ps_dir    = os.path.join(base_dir, "qa_path_shape_cert_v1")
    validator = os.path.join(ps_dir, "qa_path_shape_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_path_shape_cert_v1/qa_path_shape_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=ps_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_path_shape_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_path_shape_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_path_shape_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_male_female_octave_cert_family(base_dir: str) -> Optional[str]:
    """QA Male/Female Octave Cert family [144] — certifies Male→Female transform on QA Quantum Numbers (b,e,d,a): female=(2e,b,a,2d) [double e then swap b↔e]; female_product=4×male_product (algebraic proof: 2e×b×a×2d=4×b×e×d×a); 4×=2 octaves (musical interpretation); fundamental (1,1,2,3)→(2,1,3,4): 6→24=4×6; transform chains: each step ×4 adds 2 octaves; 5 Fibonacci+arbitrary pair witnesses; source: Ben Iverson QA framework + Dale Pond SVP male/female vibration; checks MF_1-2+TRANS/PROD/OCT/W/F; 2 PASS; self-test ok"""
    import subprocess
    mf_dir    = os.path.join(base_dir, "qa_male_female_octave_cert_v1")
    validator = os.path.join(mf_dir, "qa_male_female_octave_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_male_female_octave_cert_v1/qa_male_female_octave_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=mf_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_male_female_octave_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_male_female_octave_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_male_female_octave_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_cube_sum_cert_family(base_dir: str) -> Optional[str]:
    """QA Cube Sum Cert family [143] — certifies F³+C³+G³=216=6³ for fundamental QA direction (d,e)=(2,1) with (F,C,G)=(3,4,5): the unique 3D extension of 3²+4²=5² (Pythagorean); (k-1)³+k³+(k+1)³=3k(k²+2) for k=4 gives 12×18=216=6³; k=4 unique in [1,10000]; QA connections: 6=b×e×d×a=1×1×2×3 (fundamental QN product); 216=9×24=mod-9×mod-24 (product of both QA orbit moduli); 4 non-cube witnesses (3,2),(4,1),(4,3),(5,2) confirm uniqueness in QA direction space; checks CS_1-2+IDEN/DUAL/MOD/QN/UNIQ/W/F; 2 PASS; self-test ok"""
    import subprocess
    cs_dir    = os.path.join(base_dir, "qa_cube_sum_cert_v1")
    validator = os.path.join(cs_dir, "qa_cube_sum_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_cube_sum_cert_v1/qa_cube_sum_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=cs_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_cube_sum_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_cube_sum_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_cube_sum_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_klein4_harmonics_cert_family(base_dir: str) -> Optional[str]:
    """QA Klein 4 Harmonics Cert family [142] — certifies that the four sign-changes of (F,C,G) form K4=Z2×Z2 preserving F²+C²=G² and permuting harmonic packet {H,I,-H,-I}: I₀=identity, I₁=(F,C)→(-F,C) [F-flip; (d,e)→(e,d)], I₂=(F,C)→(F,-C) [C-flip], I₃=(F,C)→(-F,-C) [composition]; harmonic action: I₁ swaps H↔I, I₂ maps (H,I)→(-I,-H), I₃ negates both; every element self-inverse; I₁∘I₂=I₃; fundamental (2,1) orbit {(7,1),(1,7),(-1,-7),(-7,-1)}; source: elements.txt H=C+F/I=C-F, cert [137] Koenig, cert [125] chromogeometry; checks K4_1-3+ACT/HARM/W/F; 2 PASS (group axioms + 3 witnesses; 6 general witnesses H/E/Pell-boundary); self-test ok"""
    import subprocess
    k4_dir    = os.path.join(base_dir, "qa_klein4_harmonics_cert_v1")
    validator = os.path.join(k4_dir, "qa_klein4_harmonics_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_klein4_harmonics_cert_v1/qa_klein4_harmonics_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=k4_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_klein4_harmonics_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_klein4_harmonics_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_klein4_harmonics_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_pell_norm_cert_family(base_dir: str) -> Optional[str]:
    """QA Pell Norm Cert family [141] — certifies I=C-F=-(x²-2y²) where x=d-e, y=e: the QA conic discriminant equals the negated Pell norm P(x,y)=x²-2y² for D=2; Pell boundary: P=-1→I=+1 (hyperbola boundary), P=+1→I=-1 (ellipse boundary); P=0 impossible for primitive integers (would require d/e=silver ratio); M_B Pythagorean tree move (cert [135]) M_B(d,e)=(2d+e,d) corresponds to (x,y)→(x+2y,x+y) in Pell variables, which maps P→-P (Pell-sign-flip); M_B chain from (2,1): 2/1→5/2→12/5→29/12→70/29→169/70 generates Pell solution sequence x²-2y²=±1 with alternating H/E; fundamental (2,1): P=-1, I=+1; Wildberger arXiv:0806.2490; checks PN_1-3+IDEN/MB/W/F; 2 PASS (chain 6 steps; 6 general witnesses at P=±1,±7,±17); self-test ok"""
    import subprocess
    pn_dir    = os.path.join(base_dir, "qa_pell_norm_cert_v1")
    validator = os.path.join(pn_dir, "qa_pell_norm_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_pell_norm_cert_v1/qa_pell_norm_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=pn_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_pell_norm_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_pell_norm_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_pell_norm_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_conic_discriminant_cert_family(base_dir: str) -> Optional[str]:
    """QA Conic Discriminant Cert family [140] — certifies I=C-F=Qg-Qr as the conic discriminant: I>0→hyperbola (C>F, d/e<1+√2), I=0→parabola (impossible for integers: d/e=1+√2=silver ratio, irrational; disc(x²-2x-1)=8 non-square), I<0→ellipse (F>C, d/e>1+√2); silver-ratio convergents [2;2,2,2,...]=2/1,5/2,12/5,29/12,70/29 alternate H/E with |I|=1; Plimpton Row 1 (12,5) has I=1 barely hyperbolic; chromogeometry: I=Qg-Qr=green minus red quadrance; checks CD_1-4+PARA/W/F; 2 PASS (4 directions straddling boundary; 4+4 type witnesses + convergent sequence); self-test ok"""
    import subprocess
    cd_dir    = os.path.join(base_dir, "qa_conic_discriminant_cert_v1")
    validator = os.path.join(cd_dir, "qa_conic_discriminant_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_conic_discriminant_cert_v1/qa_conic_discriminant_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=cd_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_conic_discriminant_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_conic_discriminant_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_conic_discriminant_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_48_64_cert_family(base_dir: str) -> Optional[str]:
    """QA 48/64 Cert family [139] — certifies structural constants 48 and 64: ALGEBRAIC: 48L=H²-I²=4CF for all QA directions (proof: (C+F)²-(C-F)²=4CF=48L); min 48 at (2,1) L=1; ORBIT: 48=2×cosmos_period=2×24, 64=satellite_period²=8²; POLYNOMIAL: equilateral null triangle (P,R,T)=(4,4,4) satisfies PR+RT+PT=48, PRT=64 → polynomial (x-4)³=x³-12x²+48x-64; unique symmetric positive integer solution; 48/64=3/4=equilateral spread (Wildberger chromo §6.4); connects to [137] (H²-I²=4CF corollary), [128] (cosmos/satellite periods), [133] (Eisenstein equilateral); checks C4864_1-3+ALG/POLY/ORB/W/F; 2 PASS (fundamental (2,1) 48L=48; 6-witness incl. 5040=7! at (7,2)); self-test ok"""
    import subprocess
    c4864_dir = os.path.join(base_dir, "qa_48_64_cert_v1")
    validator  = os.path.join(c4864_dir, "qa_48_64_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_48_64_cert_v1/qa_48_64_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=c4864_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_48_64_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_48_64_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_48_64_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_plimpton322_cert_family(base_dir: str) -> Optional[str]:
    """QA Plimpton 322 Cert family [138] — certifies Babylonian tablet Plimpton 322 (~1800 BCE) encodes QA chromogeometric triples: each row = direction (d,e) with d,e regular (5-smooth), F=d²-e² (short side β), C=2de, G=d²+e² (diagonal δ); regularity → C=2de regular → G/C terminates in base-60 (exact sexagesimal); F²+C²=G² (Pythagorean); SPVN no-zero = QA A1; counterexample: (7,3) irregular → G/C non-terminating → absent from tablet; checks P322_1-4+P322_REG/BASE60/NOZERO/W/F; source: Mansfield & Wildberger 2017 Historia Mathematica 44:395-419; QA: F=Qr=red quadrance, C=Qg=green quadrance, G=Qb=blue quadrance (Wildberger Chromo Thm 6); 2 PASS (Row 1 (12,5)→(119,120,169); 5-witness Rows 1,5,6,9,11); self-test ok"""
    import subprocess
    p322_dir  = os.path.join(base_dir, "qa_plimpton322_cert_v1")
    validator = os.path.join(p322_dir, "qa_plimpton322_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_plimpton322_cert_v1/qa_plimpton322_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=p322_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_plimpton322_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_plimpton322_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_plimpton322_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_koenig_twisted_squares_cert_family(base_dir: str) -> Optional[str]:
    """QA Koenig Twisted Squares Cert family [137] — certifies H²-G²=G²-I²=2CF=24L for any QA direction: H=C+F (outer Koenig square), I=C-F (inner; sign=conic type), L=CF/12 (always integer for primitive); (I²,2CF,G²,H²) is arithmetic progression step 2CF; proof: H²-G²=(C+F)²-(C²+F²)=2CF, G²-I²=(C²+F²)-(C-F)²=2CF; div-24: 8|C=2de (one even), 3|F=(d-e)(d+e) (one div by 3); geometric: twisted-squares outer²-inner²=4×triangle_area=2CF; Iverson QA Law 15 / Mathologer 2024 / Will Dale 2026-03-30 (I²,2CF,G²,H²) corollary; checks KTS_1-9+KTS_W/F; 2 PASS (fundamental (2,1) 2CF=24; 5-witness general); self-test ok"""
    import subprocess
    kts_dir   = os.path.join(base_dir, "qa_koenig_twisted_squares_cert_v1")
    validator = os.path.join(kts_dir, "qa_koenig_twisted_squares_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_koenig_twisted_squares_cert_v1/qa_koenig_twisted_squares_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=kts_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_koenig_twisted_squares_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_koenig_twisted_squares_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_koenig_twisted_squares_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_cyclic_quad_cert_family(base_dir: str) -> Optional[str]:
    """QA Cyclic Quad Cert family [136] — certifies Ptolemy's theorem via three integer identities for QA direction pairs: BF (Brahmagupta-Fibonacci) G₁G₂=D²+E² where D=d₁d₂-e₁e₂, E=d₁e₂+d₂e₁; PP (Ptolemy Product) F₃=|F₁F₂-C₁C₂|, C₃=F₁C₂+F₂C₁, F₃²+C₃²=(G₁G₂)²; PC (Ptolemy Conjugate) F₄=F₁F₂+C₁C₂, C₄=|F₁C₂-F₂C₁|, F₄²+C₄²=(G₁G₂)²; both triples on circle G₁G₂ = two diagonals of Ptolemy cyclic quadrilateral; algebraic proof: (F₁F₂-C₁C₂)²+(F₁C₂+F₂C₁)²=(F₁²+C₁²)(F₂²+C₂²)=G₁²G₂²; historical: Ptolemy ~150 CE chord tables → Brahmagupta 628 CE → Gaussian Z[i] multiplication; connects to [127] UHG null (same null points); checks CQ_1/2/3/BF/PP/PC/G3/W/F; 2 PASS (fundamental (2,1)×(3,2) G₃=65; 5-witness general); self-test ok"""
    import subprocess
    cq_dir    = os.path.join(base_dir, "qa_cyclic_quad_cert_v1")
    validator = os.path.join(cq_dir, "qa_cyclic_quad_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_cyclic_quad_cert_v1/qa_cyclic_quad_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=cq_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_cyclic_quad_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_cyclic_quad_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_cyclic_quad_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_egyptian_fraction_cert_family(base_dir: str) -> Optional[str]:
    """QA Egyptian Fraction Cert family [134] — certifies greedy Egyptian fraction expansion of HAT₁=e/d: e/d=1/k₁+...+1/kₙ where kᵢ=⌈dᵢ/eᵢ⌉; denominators strictly increasing; all intermediate pairs coprime; terminates when eₙ=1; Koenig descent path (d₀,e₀)→...→(kₙ,1) = Egyptian fraction steps; Rhind Papyrus ~1600 BCE greedy algorithm = Koenig tree navigation; checks EF_1-8+EF_W/F; 2 PASS (fundamental (2,1) expansion=[2]; 6-witness general covering lengths 1/2/3); self-test ok"""
    import subprocess
    ef_dir    = os.path.join(base_dir, "qa_egyptian_fraction_cert_v1")
    validator = os.path.join(ef_dir, "qa_egyptian_fraction_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_egyptian_fraction_cert_v1/qa_egyptian_fraction_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=ef_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_egyptian_fraction_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_egyptian_fraction_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_egyptian_fraction_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_eisenstein_cert_family(base_dir: str) -> Optional[str]:
    """QA Eisenstein Cert family [133] — certifies two universal Eisenstein-norm identities from QA elements: F²-FW+W²=Z² and Y²-YW+W²=Z² for all QA tuples (b,e,d,a); F=ab, W=d(e+a), Z=e²+ad, Y=A-D=a²-d²; algebraic proof via u=b²+3be: (F+W)²-3FW=(u+3e²)²=Z²; W (equilateral side) and Z (Eisenstein companion) per QA Law 15; fundamental (1,1,2,3): (F,W,Z)=(3,8,7), (Y,W,Z)=(5,8,7), both give 49=7²; checks EIS_1-EIS_7+EIS_W/U; 2 PASS (fundamental + 6-witness general); self-test ok"""
    import subprocess
    eis_dir   = os.path.join(base_dir, "qa_eisenstein_cert_v1")
    validator = os.path.join(eis_dir, "qa_eisenstein_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_eisenstein_cert_v1/qa_eisenstein_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=eis_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_eisenstein_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_eisenstein_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_eisenstein_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_origin_of_24_cert_family(base_dir: str) -> Optional[str]:
    """QA Origin of 24 Cert family [130] — certifies dual derivation of mod-24: H²-G²=G²-I²=2CF for direction (d,e), where C=2de (green quadrance) and F=d²-e² (red quadrance); minimum value 24 at fundamental direction (d,e)=(2,1) for 3-4-5 triangle; always divisible by 24 for all primitive Pythagorean directions; checks O24_1-O24_9 (schema, elements C/F/G/H/I, dual routes Pyth-1 and Crystal) + O24_G/W/F/D (general theorem); 2 PASS (anchor 3-4-5, general theorem 6 witnesses); self-test ok"""
    import subprocess
    o24_dir   = os.path.join(base_dir, "qa_origin_of_24_cert_v1")
    validator = os.path.join(o24_dir, "qa_origin_of_24_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_origin_of_24_cert_v1/qa_origin_of_24_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=o24_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_origin_of_24_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_origin_of_24_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_origin_of_24_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_uhg_null_cert_family(base_dir: str) -> Optional[str]:
    """QA UHG Null Cert family [127] — certifies that every QA triple (F,C,G)=(d²-e²,2de,d²+e²) is a null point [F:C:G] in Universal Hyperbolic Geometry satisfying F²+C²-G²=0; equivalent to Wildberger Chromogeometric Theorem 6; Gaussian integer interpretation: Z=d+ei, Z²=(d²-e²)+2dei, |Z|²=d²+e², so (F,C,G)=(Re(Z²),Im(Z²),|Z|²); checks UN1-UN7 (schema, green/red/blue quadrance, null condition, Gaussian decomp, null_quadrance field); 2 PASS (d=2e=1 → 3-4-5 null point, d=3e=2 → 5-12-13 null point) + 1 FAIL (BLUE_QUADRANCE_MISMATCH+NULL_CONDITION_VIOLATED+GAUSSIAN_DECOMP_MISMATCH: G claimed as 6 instead of 5); self-test ok"""
    import subprocess
    unc_dir   = os.path.join(base_dir, "qa_uhg_null_cert_v1")
    validator = os.path.join(unc_dir, "qa_uhg_null_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_uhg_null_cert_v1/qa_uhg_null_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=unc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_uhg_null_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_uhg_null_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_uhg_null_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_red_group_cert_family(base_dir: str) -> Optional[str]:
    """QA Red Group Cert family [126] — certifies that the QA T-operator is the Fibonacci shift F=[[0,1],[1,1]], representing multiplication by φ in the split-complex ring Z[√5]/mZ[√5] (Wildberger red isometry group); det(F)=-1=N_red(φ) (red norm), trace(F)=1 (φ+ψ=1); orbit period = ord(F) in GL₂(Z/mZ); cosmos period=24 for m=9, cosmos period=8 for m=3; checks RG1-RG7 (schema, T_matrix, det, trace, F^P≡I, minimality, orbit_type); 2 PASS (m=9 cosmos period=24, m=3 cosmos period=8) + 1 FAIL (ORBIT_PERIOD_WRONG: claimed period=12 but F^12=-I≢I for m=9); self-test ok"""
    import subprocess
    rgc_dir   = os.path.join(base_dir, "qa_red_group_cert_v1")
    validator = os.path.join(rgc_dir, "qa_red_group_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_red_group_cert_v1/qa_red_group_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=rgc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_red_group_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_red_group_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_red_group_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_chromogeometry_cert_family(base_dir: str) -> Optional[str]:
    """QA Chromogeometry Cert family [125] — certifies that the three QA invariants C, F, G of any generator (b,e) with d=b+e, a=b+2e are exactly the three chromogeometric quadrances of direction vector (d,e): C=Q_green(d,e)=2de, F=Q_red(d,e)=d²-e²=ab, G=Q_blue(d,e)=d²+e²; C²+F²=G² is Wildberger Chromogeometric Theorem 6; I=|C-F| conic discriminant: C>F→hyperbola, C=F→parabola, C<F→ellipse; checks CG1-CG7; 2 PASS (3-4-5 hyperbola b=1e=1, 20-21-29 ellipse b=3e=2) + 1 FAIL (GREEN_QUADRANCE_MISMATCH+PYTHAGORAS_VIOLATED: claimed C=2 instead of 4); self-test ok"""
    import subprocess
    cgc_dir   = os.path.join(base_dir, "qa_chromogeometry_cert_v1")
    validator = os.path.join(cgc_dir, "qa_chromogeometry_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_chromogeometry_cert_v1/qa_chromogeometry_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=cgc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_chromogeometry_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_chromogeometry_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_chromogeometry_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_security_competency_cert_family(base_dir: str) -> Optional[str]:
    """QA Security Competency Cert family [124] — immune system architecture; extends [123] with security_role (identity/membrane/integrity/self_nonself/healing/collective), immune_function (detection/containment/recovery), pq_readiness (fips_final/in_progress/classical_only/hybrid_transitional); SC5 quantum resilience invariant: identity/membrane + classical_only → pq_migration_path required; SC6 FIPS designation required for fips_final; SC9 CELL_ORBIT_MISMATCH inherited from [123]; 2 PASS (ml_kem membrane/fips_final, ed25519 identity/classical_only+migration_path) + 1 FAIL (SC5_PQ_MIGRATION_REQUIRED: rsa_1024 no migration path); self-test ok"""
    import subprocess
    scc_dir   = os.path.join(base_dir, "qa_security_competency_cert")
    validator = os.path.join(scc_dir, "qa_security_competency_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_security_competency_cert/qa_security_competency_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=scc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_security_competency_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_security_competency_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_security_competency_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_agent_competency_cert_family(base_dir: str) -> Optional[str]:
    """QA Agent Competency Cert family [123] — formalizes Levin morphogenetic agent architecture; certifies that a QA Lab agent competency profile (goal, cognitive_horizon, convergence, orbit_signature, levin_cell_type, failure_modes, composition_rules, dedifferentiation_cond, recommitment_cond) is structurally valid; V8 CELL_ORBIT_MISMATCH enforces: differentiated↔cosmos, progenitor↔satellite/mixed, stem↔singularity; 2 PASS (merge_sort_agent cosmos/differentiated, gradient_descent_agent mixed/progenitor) + 1 FAIL (CELL_ORBIT_MISMATCH: stem agent declaring cosmos orbit); self-test ok"""
    import subprocess
    acc_dir = os.path.join(base_dir, "qa_agent_competency_cert")
    validator = os.path.join(acc_dir, "qa_agent_competency_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_agent_competency_cert/qa_agent_competency_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=acc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_agent_competency_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_agent_competency_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_agent_competency_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_empirical_observation_cert_family(base_dir: str) -> Optional[str]:
    """QA Empirical Observation Cert family [122] — bridges Open Brain / experiment results to cert ecosystem; certifies verdict (CONSISTENT/CONTRADICTS/PARTIAL/INCONCLUSIVE) against named parent cert claim."""
    import subprocess
    eo_dir = os.path.join(base_dir, "qa_empirical_observation_cert")
    validator = os.path.join(eo_dir, "qa_empirical_observation_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_empirical_observation_cert/qa_empirical_observation_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=eo_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_empirical_observation_cert self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_empirical_observation_cert self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_empirical_observation_cert self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_prime_bounded_certificate_scaling_cert_family(base_dir: str) -> Optional[str]:
    """QA Prime Bounded Certificate Scaling Cert family [131] — certifies the tested-endpoint scaling law for bounded factor-certificate witness caps; checks schema, canonical hash, artifact parity, row recomputation, row-level honesty, and overall PASS/FAIL honesty; 1 PASS (100,250,500,1000 exact-match cert) + 1 FAIL (mock 500 mismatch cert) as valid fixtures; self-test ok"""
    import subprocess

    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_prime_bounded_certificate_scaling_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_prime_bounded_certificate_scaling_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test", "--json"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_prime_bounded_certificate_scaling_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    data = json.loads(proc.stdout)
    if not data.get("ok"):
        failures = [f["fixture"] for f in data.get("fixtures", []) if not (f["ok"] == f["expected_ok"])]
        raise RuntimeError(f"self-test fixture mismatches: {failures}")
    return None


def _validate_dual_spine_unification_report_family(base_dir: str) -> Optional[str]:
    """QA Dual Spine Unification Report family [119] — top-level validated overview of both public spines."""
    import subprocess
    du_dir = os.path.join(base_dir, "qa_dual_spine_unification_report")
    validator = os.path.join(du_dir, "qa_dual_spine_unification_report_validate.py")
    if not os.path.exists(validator):
        return "missing qa_dual_spine_unification_report/qa_dual_spine_unification_report_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=du_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_dual_spine_unification_report self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_dual_spine_unification_report self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_dual_spine_unification_report self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_control_stack_report_family(base_dir: str) -> Optional[str]:
    """QA Control Stack Report family [118] — reader-facing report packaging [117]."""
    import subprocess
    rp_dir = os.path.join(base_dir, "qa_control_stack_report")
    validator = os.path.join(rp_dir, "qa_control_stack_report_validate.py")
    if not os.path.exists(validator):
        return "missing qa_control_stack_report/qa_control_stack_report_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=rp_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_control_stack_report self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_control_stack_report self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_control_stack_report self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_control_stack_family(base_dir: str) -> Optional[str]:
    """QA Control Stack family [117] — synthesis cert for the control/compiler spine."""
    import subprocess
    cs_dir = os.path.join(base_dir, "qa_control_stack")
    validator = os.path.join(cs_dir, "qa_control_stack_validate.py")
    if not os.path.exists(validator):
        return "missing qa_control_stack/qa_control_stack_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=cs_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_control_stack self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_control_stack self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_control_stack self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_obstruction_stack_report_family(base_dir: str) -> Optional[str]:
    """QA Obstruction Stack Report family [116] — reader-facing report packaging [115]."""
    import subprocess
    rp_dir = os.path.join(base_dir, "qa_obstruction_stack_report")
    validator = os.path.join(rp_dir, "qa_obstruction_stack_report_validate.py")
    if not os.path.exists(validator):
        return "missing qa_obstruction_stack_report/qa_obstruction_stack_report_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=rp_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_obstruction_stack_report self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_obstruction_stack_report self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_obstruction_stack_report self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_obstruction_stack_family(base_dir: str) -> Optional[str]:
    """QA Obstruction Stack family [115] — synthesis spine compressing the full [111]–[114] chain."""
    import subprocess
    st_dir = os.path.join(base_dir, "qa_obstruction_stack")
    validator = os.path.join(st_dir, "qa_obstruction_stack_validate.py")
    if not os.path.exists(validator):
        return "missing qa_obstruction_stack/qa_obstruction_stack_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=st_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_obstruction_stack self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_obstruction_stack self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_obstruction_stack self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_obstruction_efficiency_family(base_dir: str) -> Optional[str]:
    """QA Obstruction Efficiency family [114] — quantifies search-cost savings from obstruction-aware pruning."""
    import subprocess
    ef_dir = os.path.join(base_dir, "qa_obstruction_efficiency")
    validator = os.path.join(ef_dir, "qa_obstruction_efficiency_validate.py")
    if not os.path.exists(validator):
        return "missing qa_obstruction_efficiency/qa_obstruction_efficiency_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=ef_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_obstruction_efficiency self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_obstruction_efficiency self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_obstruction_efficiency self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_obstruction_compiler_bridge_family(base_dir: str) -> Optional[str]:
    """QA Obstruction-Compiler Bridge family [112] — bridges [111] arithmetic to [106] control."""
    import subprocess
    bridge_dir = os.path.join(base_dir, "qa_obstruction_compiler_bridge")
    validator = os.path.join(bridge_dir, "qa_obstruction_compiler_bridge_validate.py")
    if not os.path.exists(validator):
        return "missing qa_obstruction_compiler_bridge/qa_obstruction_compiler_bridge_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=bridge_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_obstruction_compiler_bridge self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_obstruction_compiler_bridge self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_obstruction_compiler_bridge self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_area_quantization_pk_family(base_dir: str) -> Optional[str]:
    """QA Inert Prime Area Quantization family [111] — generalization of [108] to mod p^k."""
    import subprocess
    pk_dir = os.path.join(base_dir, "qa_area_quantization_pk")
    validator = os.path.join(pk_dir, "qa_area_quantization_pk_validate.py")
    if not os.path.exists(validator):
        return "missing qa_area_quantization_pk/qa_area_quantization_pk_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=pk_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_area_quantization_pk self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_area_quantization_pk self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_area_quantization_pk self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_seismic_control_family(base_dir: str) -> Optional[str]:
    """QA Seismic Pattern Control family [110] — domain_instance of QA_PLAN_CONTROL_COMPILER_CERT.v1."""
    import subprocess
    sc_dir = os.path.join(base_dir, "qa_seismic_control")
    validator = os.path.join(sc_dir, "qa_seismic_control_validate.py")
    if not os.path.exists(validator):
        return "missing qa_seismic_control/qa_seismic_control_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=sc_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_seismic_control self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_seismic_control self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    # self-test returns dict of {filename: {structural_validation, ...}}
    # all entries must have structural_validation == "PASS"
    all_ok = all(v.get("structural_validation") == "PASS" for v in payload.values())
    if not all_ok:
        raise RuntimeError(
            "qa_seismic_control self-test has failing fixtures:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_inheritance_compat_family(base_dir: str) -> Optional[str]:
    """QA Inheritance Compat family [109] — certifies inheritance edges as first-class objects."""
    import subprocess
    ic_dir = os.path.join(base_dir, "qa_inheritance_compat")
    validator = os.path.join(ic_dir, "qa_inheritance_compat_validate.py")
    if not os.path.exists(validator):
        return "missing qa_inheritance_compat/qa_inheritance_compat_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=ic_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_inheritance_compat self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_inheritance_compat self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_inheritance_compat self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_area_quantization_family(base_dir: str) -> Optional[str]:
    """QA Area Quantization family [108] — first family_extension of QA_CORE_SPEC.v1."""
    import subprocess
    aq_dir = os.path.join(base_dir, "qa_area_quantization")
    validator = os.path.join(aq_dir, "qa_area_quantization_validate.py")
    if not os.path.exists(validator):
        return "missing qa_area_quantization/qa_area_quantization_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=aq_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_area_quantization self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_area_quantization self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_area_quantization self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_core_spec_family(base_dir: str) -> Optional[str]:
    """QA Core Spec Kernel family [107] — base executable ontology cert."""
    import subprocess
    core_spec_dir = os.path.join(base_dir, "qa_core_spec")
    validator = os.path.join(core_spec_dir, "qa_core_spec_validate.py")
    if not os.path.exists(validator):
        return "missing qa_core_spec/qa_core_spec_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=core_spec_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_core_spec self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_core_spec self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_core_spec self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_plan_control_compiler_family(base_dir: str) -> Optional[str]:
    """QA Plan-Control Compiler family [106] — generic compilation relation cert."""
    import subprocess
    compiler_dir = os.path.join(base_dir, "qa_plan_control_compiler")
    validator = os.path.join(compiler_dir, "qa_plan_control_compiler_validate.py")
    if not os.path.exists(validator):
        return "missing qa_plan_control_compiler/qa_plan_control_compiler_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=compiler_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_plan_control_compiler self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_plan_control_compiler self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_plan_control_compiler self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_cymatics_family_if_present(base_dir: str) -> Optional[str]:
    """QA Cymatics Correspondence family [105] — mode + Faraday + control certs."""
    import subprocess
    cymatics_dir = os.path.join(base_dir, "qa_cymatics")
    validator = os.path.join(cymatics_dir, "qa_cymatics_validate.py")
    if not os.path.exists(validator):
        return "missing qa_cymatics/qa_cymatics_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60,
        cwd=cymatics_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_cymatics self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_cymatics self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_cymatics self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_feuerbach_parent_scale_family(base_dir: str) -> Optional[str]:
    """QA Feuerbach Parent Scale family [104] — scale=4 interior law + root exception (3,4,5)."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_feuerbach_parent_scale_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_feuerbach_parent_scale_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120, cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_feuerbach_parent_scale_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_feuerbach_parent_scale_v1 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_feuerbach_parent_scale_v1 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_lojasiewicz_orbit_cert_v2_family(base_dir: str) -> Optional[str]:
    """QA Lojasiewicz Orbit Descent Cert v2 family [103] — intrinsic (H-crit derived)."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_lojasiewicz_orbit_cert_v2", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_lojasiewicz_orbit_cert_v2/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120, cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_lojasiewicz_orbit_cert_v2 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_lojasiewicz_orbit_cert_v2 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_lojasiewicz_orbit_cert_v2 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_lojasiewicz_orbit_cert_family(base_dir: str) -> Optional[str]:
    """QA Lojasiewicz Orbit Descent Cert family [102]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_lojasiewicz_orbit_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_lojasiewicz_orbit_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120, cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_lojasiewicz_orbit_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_lojasiewicz_orbit_cert_v1 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_lojasiewicz_orbit_cert_v1 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None


def _validate_gradient_lipschitz_gain_cert_family(base_dir: str) -> Optional[str]:
    """QA Gradient Lipschitz Gain Cert family [101]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_gradient_lipschitz_gain_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_gradient_lipschitz_gain_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120, cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_gradient_lipschitz_gain_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_gradient_lipschitz_gain_cert_v1 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_gradient_lipschitz_gain_cert_v1 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None

def _validate_e8_alignment_audit_cert_family(base_dir: str) -> Optional[str]:
    """QA E8 Alignment Audit Cert family [100]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_e8_alignment_audit_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_e8_alignment_audit_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120, cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_e8_alignment_audit_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_e8_alignment_audit_cert_v1 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_e8_alignment_audit_cert_v1 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None

def _validate_attn_spectral_gain_cert_family(base_dir: str) -> Optional[str]:
    """QA Attention Spectral Gain Cert family [99]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_attn_spectral_gain_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_attn_spectral_gain_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_attn_spectral_gain_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_attn_spectral_gain_cert_v1 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_attn_spectral_gain_cert_v1 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None

def _validate_gnn_spectral_gain_cert_family(base_dir: str) -> Optional[str]:
    """QA GNN Spectral Gain Cert family [98]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_gnn_spectral_gain_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_gnn_spectral_gain_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_gnn_spectral_gain_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_gnn_spectral_gain_cert_v1 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_gnn_spectral_gain_cert_v1 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None

def _validate_orbit_curvature_cert_family(base_dir: str) -> Optional[str]:
    """QA Orbit Curvature Cert family [97]."""
    import subprocess
    repo_root = os.path.normpath(os.path.join(base_dir, ".."))
    validator = os.path.join(repo_root, "qa_orbit_curvature_cert_v1", "validator.py")
    if not os.path.exists(validator):
        return "missing qa_orbit_curvature_cert_v1/validator.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "qa_orbit_curvature_cert_v1 self-test failed:\n"
            f"{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            "qa_orbit_curvature_cert_v1 self-test returned non-JSON output:\n"
            f"error={exc}\nstdout={(proc.stdout or '').strip()}\nstderr={(proc.stderr or '').strip()}"
        )
    if payload.get("ok") is not True:
        raise RuntimeError(
            "qa_orbit_curvature_cert_v1 self-test returned ok=false:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return None

def test_spine_v1_compliance() -> bool:
    """[70] QA Dynamics Spine v1 compliance linter.

    For each family in SPINE_V1_FAMILIES, verify the required
    Dynamics-Compatible fixture files exist. Failing here means
    a spine-declared family is missing its tamper-evidence fixtures.
    """
    import os as _os
    _repo_root = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    missing = []
    for family in SPINE_V1_FAMILIES:
        for fixture_rel in _SPINE_V1_REQUIRED_FIXTURES:
            path = _os.path.join(_repo_root, family, fixture_rel)
            if not _os.path.isfile(path):
                missing.append(path)
    if missing:
        print(f"[FAIL] Spine v1 linter: missing fixtures: {missing}")
        return False
    print(f"[PASS] Spine v1 linter: all {len(SPINE_V1_FAMILIES)} families x {len(_SPINE_V1_REQUIRED_FIXTURES)} fixtures found")
    return True


def _validate_pim_kernel_cert_family(base_dir: str) -> Optional[str]:
    """QA PIM Kernel Cert family [157] — certifies PIM kernel correctness: CRT (coprime+non-coprime), RESIDUE_SELECT, TORUS_SHIFT, ROLLING_SUM_PHASE, A1 coordinate-layer documentation. Checks PIM_1+CRT/KERNEL/A1/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_pim_kernel_cert_v1")
    validator = os.path.join(fam_dir, "qa_pim_kernel_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_pim_kernel_cert_v1/qa_pim_kernel_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_pim_kernel_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_pim_kernel_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_pim_kernel_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_graph_community_cert_family(base_dir: str) -> Optional[str]:
    """QA Graph Community Cert family [158] — certifies QA feature map dimensions (qa21=21, qa27=27, qa83=83) and community detection benchmarks on standard networks. Chromogeometry check: C*C+F*F=G*G. Checks GC_1+DIM/CHROMO/BENCH/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_graph_community_cert_v1")
    validator = os.path.join(fam_dir, "qa_graph_community_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_graph_community_cert_v1/qa_graph_community_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_graph_community_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_graph_community_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_graph_community_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_observer_core_cert_family(base_dir: str) -> Optional[str]:
    """QA Observer Core Cert family [159] — certifies qa_mod() A1 compliance (output in {1,...,m}, never 0) and compute_qci() determinism across all 6 empirical domains. T2 firewall: no float->int feedback. Checks OC_1+A1/QCI/T2/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_observer_core_cert_v1")
    validator = os.path.join(fam_dir, "qa_observer_core_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_observer_core_cert_v1/qa_observer_core_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_observer_core_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_observer_core_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_observer_core_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_cardiac_arrhythmia_cert_family(base_dir: str) -> Optional[str]:
    """QA Cardiac Arrhythmia Cert family [170] — certifies QA orbit features as independent predictors of arrhythmia classification beyond R-R interval baseline using MIT-BIH (48 records, 94536 beats). dR2=+0.037 p<10^-6; 2/2 surrogates beaten; Phi(D)=-1 pre-registered. Checks CAR_1+DATA/DELTA/SURR/PHI/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_cardiac_arrhythmia_cert_v1")
    validator = os.path.join(fam_dir, "qa_cardiac_arrhythmia_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_cardiac_arrhythmia_cert_v1/qa_cardiac_arrhythmia_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_cardiac_arrhythmia_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_cardiac_arrhythmia_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_cardiac_arrhythmia_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_emg_pathology_cert_family(base_dir: str) -> Optional[str]:
    """QA EMG Pathology Cert family [171] — certifies QA orbit features as independent predictors of EMG pathology classification beyond RMS using PhysioNet EMG (3 records, 1203 windows). dR2=+0.608 p<10^-6; 2/2 surrogates beaten; Phi(D)=-1 pre-registered. Checks EMG_1+DATA/DELTA/SURR/PHI/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_emg_pathology_cert_v1")
    validator = os.path.join(fam_dir, "qa_emg_pathology_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_emg_pathology_cert_v1/qa_emg_pathology_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_emg_pathology_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_emg_pathology_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_emg_pathology_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_era5_reanalysis_cert_family(base_dir: str) -> Optional[str]:
    """QA ERA5 Reanalysis Cert family [172] — certifies QCI as predictor of atmospheric variability using WeatherBench2 ERA5 (3297 days x 15 channels, 500hPa). r=+0.46, partial r=+0.43; 4/4 surrogates beaten. Checks ERA_1+DATA/R/PARTIAL/SURR/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_era5_reanalysis_cert_v1")
    validator = os.path.join(fam_dir, "qa_era5_reanalysis_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_era5_reanalysis_cert_v1/qa_era5_reanalysis_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_era5_reanalysis_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_era5_reanalysis_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_era5_reanalysis_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_surrogate_methodology_cert_family(base_dir: str) -> Optional[str]:
    """QA Surrogate Methodology Cert family [173] — certifies corrected surrogate null design: real targets fixed, surrogate QCI only. Circular null problem identified and resolved. 6/8 domains confirmed. Checks SRM_1+DESIGN/CIRCULAR/DOMAINS/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_surrogate_methodology_cert_v1")
    validator = os.path.join(fam_dir, "qa_surrogate_methodology_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_surrogate_methodology_cert_v1/qa_surrogate_methodology_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_surrogate_methodology_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_surrogate_methodology_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_surrogate_methodology_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_phi_transformation_cert_family(base_dir: str) -> Optional[str]:
    """QA Phi Transformation Cert family [174] — certifies Phi(D) transformation law: disorder-stress vs order-stress classification. 2/2 pre-registered (cardiac, EMG), 6/6 post-hoc consistent. Domain requirement: temporal multi-channel signals with non-trivial baselines. Checks PHI_1+CLASS/PREREG/POSTHOC/REQ/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_phi_transformation_cert_v1")
    validator = os.path.join(fam_dir, "qa_phi_transformation_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_phi_transformation_cert_v1/qa_phi_transformation_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_phi_transformation_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_phi_transformation_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_phi_transformation_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_cross_domain_invariance_cert_family(base_dir: str) -> Optional[str]:
    """QA Cross-Domain Invariance Cert family [175] — certifies 3 structural invariants (surrogate survival, independent information, domain-general architecture) across 6 Tier 3 domains. Checks CDI_1+INV1/INV2/INV3/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_cross_domain_invariance_cert_v1")
    validator = os.path.join(fam_dir, "qa_cross_domain_invariance_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_cross_domain_invariance_cert_v1/qa_cross_domain_invariance_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_cross_domain_invariance_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_cross_domain_invariance_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_cross_domain_invariance_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_h_null_modularity_cert_family(base_dir: str) -> Optional[str]:
    """QA H-Null Modularity Cert family [180] — certifies H-null chromogeometric modularity model for graph community detection. H(b,e)=C+F where C=2de (green quadrance) and F=d*d-e*e (red quadrance). Les Miserables ARI=0.638 vs standard ARI=0.588 (+0.050). HONEST: wins on 1/10 graphs only — topology-specific to hub-dominated networks. Tier 2. Checks HN_1+MODEL/CHROMO/BENCH/HONEST/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_h_null_modularity_cert_v1")
    validator = os.path.join(fam_dir, "qa_h_null_modularity_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_h_null_modularity_cert_v1/qa_h_null_modularity_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_h_null_modularity_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_h_null_modularity_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_h_null_modularity_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_satellite_product_sum_cert_family(base_dir: str) -> Optional[str]:
    """QA Satellite Product Sum Cert family [181] — certifies that the sum of QA tuple products (b*e*d*a) over all satellite pairs equals M^4 for any modulus M divisible by 3. Proof: satellite sub-lattice is Fibonacci-closed with normalized product sum = 81 = 3^4, giving total = s^4 * 3^4 = M^4. Corollary: satellite total volume = singularity volume. Tier 1 algebraic identity. Verified for 33 moduli (M=3..99). Checks SPS_1+PROOF/COUNT/SUM/TUPLES/CLOSURE/COROL/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_satellite_product_sum_cert_v1")
    validator = os.path.join(fam_dir, "qa_satellite_product_sum_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_satellite_product_sum_cert_v1/qa_satellite_product_sum_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_satellite_product_sum_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_satellite_product_sum_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_satellite_product_sum_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_miller_orbit_cert_family(base_dir: str) -> Optional[str]:
    """QA Miller Orbit Cert family [182] — certifies structural properties of QA mod-9 orbit classification applied to crystallographic Miller indices. Four Tier 1 results: (1) cosmos d > satellite d universal across all crystal systems (proved from Q_M ≥ 9 for satellite); (2) satellite Q_M mod 9 ∈ QR(9) = {0,1,4,7} (quadratic residue restriction); (3) singularity Q_M = perfect squares (h=k=0 → Q_M=l²); (4) satellite green channel 3× cosmos (chromogeometric shift). Verified 21 minerals, 13055 reflections. Checks MO_1+ORDER/QR/SQUARE/CHROMO/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_miller_orbit_cert_v1")
    validator = os.path.join(fam_dir, "qa_miller_orbit_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_miller_orbit_cert_v1/qa_miller_orbit_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_miller_orbit_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_miller_orbit_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_miller_orbit_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_eisenstein_crystal_cert_family(base_dir: str) -> Optional[str]:
    """QA Eisenstein Crystal Cert family [183] — certifies Z-Y=J=bd (new universal identity), Z²-Y²=J·a·(a+e) factorization, and Eisenstein norm F²-FW+W²=Z² as crystal constant encoder. Unity Block {F,G,Z,W}={3,5,7,8}=Ben Iverson's 'four Forces' (QA-4 Crystal Universe 1990). Tier 1 algebraic. 10 witnesses. Connects [133] Eisenstein, [182] Miller orbit. Checks EC_1+ZYJ/FACTOR/EISEN/TUPLE/UNITY/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_eisenstein_crystal_cert_v1")
    validator = os.path.join(fam_dir, "qa_eisenstein_crystal_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_eisenstein_crystal_cert_v1/qa_eisenstein_crystal_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_eisenstein_crystal_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_eisenstein_crystal_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_eisenstein_crystal_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_keely_structural_ratio_cert_family(base_dir: str) -> Optional[str]:
    """QA Keely Structural Ratio Cert family [184] — certifies Keely's 8 structural ratio laws (Laws 2,4,9,10,18,27,29,33) as QA modular invariants. Category 1 of Vibes 5-category framework. Maps pitch=f-value, period divisibility 1|8|24, concordance coupling, chromogeometry C*C+F*F=G*G. Checks KSR_1+LAWS/PERIOD/FVAL/LCM/CHROMO/CLOSURE/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_keely_structural_ratio_cert_v1")
    validator = os.path.join(fam_dir, "qa_keely_structural_ratio_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_keely_structural_ratio_cert_v1/qa_keely_structural_ratio_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_keely_structural_ratio_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_keely_structural_ratio_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_keely_structural_ratio_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_keely_sympathetic_transfer_cert_family(base_dir: str) -> Optional[str]:
    """QA Keely Sympathetic Transfer Cert family [185] — certifies Keely's 7 sympathetic transfer laws (Laws 5,6,7,8,17,37,40) as QA reachability and path structure. Category 2 of Vibes framework. Sympathetic oscillation=orbit co-membership; discord=reachability obstruction; triad concordance condition. Checks KST_1+LAWS/REACH/BLOCK/PATH/TRIAD/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_keely_sympathetic_transfer_cert_v1")
    validator = os.path.join(fam_dir, "qa_keely_sympathetic_transfer_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_keely_sympathetic_transfer_cert_v1/qa_keely_sympathetic_transfer_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_keely_sympathetic_transfer_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_keely_sympathetic_transfer_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_keely_sympathetic_transfer_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_keely_dominant_control_cert_family(base_dir: str) -> Optional[str]:
    """QA Keely Dominant Control Cert family [186] — certifies Keely's 3 dominant/control laws (Laws 1,11,16) as QA orbit hierarchy. Category 3 of Vibes framework. Invariant substrate, triune generator manifestation, singularity as neutral center/dominant. Checks KDC_1+LAWS/SUB/TRIUNE/SING/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_keely_dominant_control_cert_v1")
    validator = os.path.join(fam_dir, "qa_keely_dominant_control_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_keely_dominant_control_cert_v1/qa_keely_dominant_control_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_keely_dominant_control_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_keely_dominant_control_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_keely_dominant_control_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_keely_aggregation_cert_family(base_dir: str) -> Optional[str]:
    """QA Keely Aggregation Cert family [187] — certifies Keely's 5 aggregation/disintegration laws (Laws 3,12,28,34,35) as QA state composition/decomposition. Category 4 of Vibes framework. Coupling tension, orbit density, discord dissociation, deterministic synthesis. Checks KAG_1+LAWS/COUPLE/DENSITY/DISSOC/SYNTH/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_keely_aggregation_cert_v1")
    validator = os.path.join(fam_dir, "qa_keely_aggregation_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_keely_aggregation_cert_v1/qa_keely_aggregation_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_keely_aggregation_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_keely_aggregation_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_keely_aggregation_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_keely_phenomenological_cert_family(base_dir: str) -> Optional[str]:
    """QA Keely Phenomenological Cert family [188] — certifies Keely's 17 phenomenological laws (Laws 13-15,19-26,30-32,36,38,39) as Theorem NT observer projections. Category 5 of Vibes framework. All 17 describe continuous measurements that reveal but never causally feed back into QA. Largest category (42.5%). Checks KPH_1+LAWS/NT/OBS/DISC/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_keely_phenomenological_cert_v1")
    validator = os.path.join(fam_dir, "qa_keely_phenomenological_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_keely_phenomenological_cert_v1/qa_keely_phenomenological_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_keely_phenomenological_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_keely_phenomenological_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_keely_phenomenological_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_dale_circle_cert_family(base_dir: str) -> Optional[str]:
    """QA Dale Circle Cert family [189] — certifies Dale Pond's integer circle construction. Three new elements: P=2W (diameter), Q=P (circumference=diameter in QA units), R=W² (area). Pi disappears in QA circular units. Source: svpwiki.com 1998. Checks DC_1+P/Q/R/W/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_dale_circle_cert_v1")
    validator = os.path.join(fam_dir, "qa_dale_circle_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_dale_circle_cert_v1/qa_dale_circle_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_dale_circle_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_dale_circle_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_dale_circle_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_equilateral_height_cert_family(base_dir: str) -> Optional[str]:
    """QA Equilateral Height Cert family [190] — certifies element S=d²e=d*X=D*e, Dale Pond's equilateral triangle height (#25 in svpwiki.com QA Elements). Three equivalent definitions verified for 7 directions. Connects to [152] W side and [189] circle. Checks EH_1+S/DX/DE/W/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_equilateral_height_cert_v1")
    validator = os.path.join(fam_dir, "qa_equilateral_height_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_equilateral_height_cert_v1/qa_equilateral_height_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_equilateral_height_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_equilateral_height_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_equilateral_height_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_dual_extremality_24_cert_family(base_dir: str) -> Optional[str]:
    """QA Dual Extremality 24 Cert family [192] — certifies the joint extremality of m=24 under the Pisano period operator pi and the Carmichael lambda function. (1) pi(24)=24: minimum non-trivial Pisano fixed point (OEIS A235702). (2) lambda(24)=2 and max{m : lambda(m)=2}=24 (structurally proved: m | 24). (3) pi(9)=24: QA theoretical modulus maps to applied modulus in one Pisano step. (4) Basin of 24 in [1,30] = {6,9,12,16,18,24}. (5) Cannonball identity 1^2+...+24^2=70^2. (6) 24-theorem: p^2-1 div by 24 for primes p>=5. Closes item 5 of [191] Bateson sketch (Level-III self-improvement fixed point). ORIGINAL: joint (pi, lambda) extremality observation. Source: Wall 1960, OEIS A235702, Carmichael 1910, Watson 1918, Baez 2008. Checks DE_1+PISANO/MIN_FP/CARMICHAEL/MAX_LAM/JOINT/BRIDGE/BASIN/CANNON/24THM/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_dual_extremality_24_cert_v1")
    validator = os.path.join(fam_dir, "qa_dual_extremality_24_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_dual_extremality_24_cert_v1/qa_dual_extremality_24_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_dual_extremality_24_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_dual_extremality_24_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_dual_extremality_24_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_conversation_arag_cert_family(base_dir: str) -> Optional[str]:
    """QA Conversation A-RAG Cert family [210] — QA-native conversation retrieval datastore. Three sources (ChatGPT, Claude.ai, Gemini) mapped to integer tuples via Candidate F: b=dr(sum(ord(c))), e=role_rank, using [202] Aiq Bekar digital root. Three A-RAG views: keyword (FTS5), semantic (PPR over parent/cite/succ/ref edges, alpha=0.5), chunk (direct raw_text). Role-diagonal property: (a_label-d_label) mod 9 = e mod 9. Cross-source invariance verified on 5361 messages. T2 firewall: BM25/PPR = observer measurements, never QA inputs. A2: only b,e stored, d/a derived on read. Composes with [18], [20], QA_ARAG_INTERFACE, [202], [122], [191]. Checks CAV_1+SCHEMA/TUPLE/DIAG/CROSS/PROMO/A1/A2/T2/VIEWS/W/F; 2 fixtures PASS+FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_conversation_arag_cert_v1")
    validator = os.path.join(fam_dir, "qa_conversation_arag_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_conversation_arag_cert_v1/qa_conversation_arag_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_conversation_arag_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_conversation_arag_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_conversation_arag_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_norm_flip_signed_cert_family(base_dir: str) -> Optional[str]:
    """QA Norm-Flip Signed-Temporal Cert family [214] — Eisenstein quadratic form f(b,e) = b*b + b*e - e*e satisfies integer identity f(e,b+e) = -f(b,e). T^2 preserves f mod m. On S_9, 5 T-orbits decompose into 3 signed cosmos orbits with norm pairs {1,8}/{4,5}/{2,7} (Fibonacci/Lucas/Phibonacci) and 2 null orbits (Tribonacci satellite, Ninbonacci singularity). Temporal sign formula: (-1)^t * sign(f(s_0)) on integer lift. Connects [133] Eisenstein, [155] Bearden phase conjugate (QCI opposite-sign IS this flip), [191] stratification. Source: Eisenstein 1844. Checks NFS_1+FLIP/T2/PAIRS/TEMPORAL/155/133/SRC/WIT/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_norm_flip_signed_cert_v1")
    validator = os.path.join(fam_dir, "qa_norm_flip_signed_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_norm_flip_signed_cert_v1/qa_norm_flip_signed_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_norm_flip_signed_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_norm_flip_signed_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_norm_flip_signed_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_causal_dag_cert_family(base_dir: str) -> Optional[str]:
    """QA Causal DAG Cert family [213] — The A2 axiom (d=b+e, a=b+2e) IS the structural equation system of a 4-node Y-structure causal DAG with b,e exogenous and d,a endogenous colliders. Pair-invertibility theorem: all 6 pairs bijective on S_m iff gcd(2,m)=1; on S_9 all 6 bijective, on S_24 pair (b,a) is 2-to-1. Pearl-level collapse theorem: deterministic SCM collapses association/intervention/counterfactual to the A2 identities — SCM form of Theorem NT. Source: Pearl 2009, Wright 1921. Prerequisites: [191], [150], [202]. Checks CDG_1+STRUCT/A2/PAIRS/PEARL/NT/191/SRC/WIT/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_causal_dag_cert_v1")
    validator = os.path.join(fam_dir, "qa_causal_dag_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_causal_dag_cert_v1/qa_causal_dag_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_causal_dag_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_causal_dag_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_causal_dag_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_fibonacci_hypergraph_cert_family(base_dir: str) -> Optional[str]:
    """QA Fibonacci Hypergraph Cert family [212] — proves three structural theorems about the QA state-residue incidence hypergraph H(m) whose hyperedges are length-4 Fibonacci windows (b,e,d,a). Theorem 1: sliding window h(T(s))=(e,d,a,(d+a) mod m), 81/81 on S_9. Theorem 2: uniform vertex degree 4m=36 per residue, total 324 on S_9. Theorem 3: orbit-multiset collapse on S_9 T-orbits (24,24,24,8,1) yielding (22,22,22,4,1) distinct multisets. Source: Fibonacci 1202, Lucas 1878, Wall 1960, Berge 1989. Prerequisites: [191], [192], [211]. Checks HGR_1+SLIDE/DEG/ORB/FIB/191/SRC/WIT/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_fibonacci_hypergraph_cert_v1")
    validator = os.path.join(fam_dir, "qa_fibonacci_hypergraph_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_fibonacci_hypergraph_cert_v1/qa_fibonacci_hypergraph_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_fibonacci_hypergraph_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_fibonacci_hypergraph_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_fibonacci_hypergraph_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_cayley_bateson_filtration_cert_family(base_dir: str) -> Optional[str]:
    """QA Cayley Bateson Filtration Cert family [211] — proves the tiered reachability classes of [191] are exactly the connected components of nested undirected Cayley graphs on S_9. Generator sets Gamma_L1={T}, Gamma_L2a=Gamma_L1 U (Z/9Z)*-scalars U {swap}, Gamma_L2b=Gamma_L2a U zero-divisor-scalars U {const_(9,9)} yield components (24,24,24,8,1)/(72,8,1)/(81). Cumulative sums-of-squares 1793/5249/6561 and non-cumulative differences 1712/3456/1312 match [191] EXPECTED_TIER_COUNTS_S9. Undirected convention essential. Source: Cayley 1878, Dehn 1911. Prerequisite: [191]. Checks CBF_1+GEN/COMP/CUMU/DIFF/L1/L2A/L2B/191/SRC/WIT/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_cayley_bateson_filtration_cert_v1")
    validator = os.path.join(fam_dir, "qa_cayley_bateson_filtration_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_cayley_bateson_filtration_cert_v1/qa_cayley_bateson_filtration_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_cayley_bateson_filtration_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_cayley_bateson_filtration_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_cayley_bateson_filtration_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_ebm_equivalence_cert_family(base_dir: str) -> Optional[str]:
    """QA EBM Equivalence Cert family [216] — formal claim that QA coherence is a discrete-native, Theorem NT-compliant Energy-Based Model. Pointwise energy E_QA(b,e,next) = 0 if T(b,e)==next else 1; window energy = 1 - QCI. Five EBM axioms verified in-cert: (E1) non-negativity exhaustive on S_9^2 × {1..9}; (E2) data-manifold zero — E(deterministic T-trajectory)=0 exactly; (E3) monotonicity — injecting k% mismatch grows E linearly (0→0.19→0.46→0.66→0.83 at 0/10/30/50/80%); (E4) Boltzmann occupancy well-formed, T=2π/m per cert [215]; (E5) score identity — argmax of Boltzmann over next_state equals T-operator step (exhaustive S_9, 81/81). Structural consequence: qa_detect IS a trained EBM; MCMC-free sampling = T-operator walk; no gradient approximation; reproducible by integer arithmetic. Source: Will Dale + Claude 2026-04-12; LeCun et al. 2006 EBM tutorial, Hinton 2002 CD. Cross-refs [154] QCI empirical, [191] Bateson filtration, [215] bin-resonance temperature identity, Theorem NT. Checks EBM_1+SCHEMA/NONNEG/ZERO/MONOTONE/BOLTZMANN/SCORE/INT_ONLY/SELFTEST; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_ebm_equivalence_cert_v1")
    validator = os.path.join(fam_dir, "qa_ebm_equivalence_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_ebm_equivalence_cert_v1/qa_ebm_equivalence_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_ebm_equivalence_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_ebm_equivalence_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_ebm_equivalence_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_resonance_bin_correspondence_cert_family(base_dir: str) -> Optional[str]:
    """QA Resonance-Bin Correspondence Cert family [215] — formalizes the bin-width ≡ resonance-tolerance isomorphism as a candidate permissibility filter connecting QA syntax (integer equivalence classes under modulus m) to SVP semantics (sympathetic transmission windows). Three witnesses: (1) Arnold-tongue phase-lock width matches QA bin width at corresponding modulus — critical coupling K* scales monotonically with m (m=6→K*=0.06, m=18→0.08, m=48→0.10 empirically); (2) Hensel lift mod 3→9→27 progressive bandwidth narrowing via external reference qa_brainca_selforg_v2.py; (3) integer-only round-trip preserves bin assignment with zero fractions.Fraction usage (S2/A1 compliant). Closes the permissibility-filter gap flagged in docs/theory/QA_SYNTAX_SVP_SEMANTICS.md from Dale Pond + Vibes letter 2026-04-05 (OB a9307705). Source: Will Dale + Claude 2026-04-12; Arnold tongue theory (Arnold 1961), Q-factor resonance. Checks RBC_1+SCHEMA/ARNOLD/BINWIDTH/HENSEL/ROUNDTRIP/INT_ONLY/SELFTEST; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_resonance_bin_correspondence_cert_v1")
    validator = os.path.join(fam_dir, "qa_resonance_bin_correspondence_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_resonance_bin_correspondence_cert_v1/qa_resonance_bin_correspondence_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_resonance_bin_correspondence_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_resonance_bin_correspondence_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_resonance_bin_correspondence_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_bateson_learning_levels_cert_family(base_dir: str) -> Optional[str]:
    """QA Bateson Learning Levels Cert family [191] — formalizes Gregory Bateson's learning hierarchy (0/I/II/III) as a strict invariant filtration on QA state spaces. Four invariants (orbit ⊂ family ⊂ modulus ⊂ ambient category) define L_0/L_1/L_2a/L_2b/L_3 operator classes. Tiered Reachability Theorem exhaustively verified on S_9: only 26% of 6561 pairs are Level-I reachable; 52.67% require L_2a, 20% require L_2b. Witnesses at every tier (qa_step, scalar_mult k=2, scalar_mult k=3, modulus_reduction). Source: Bateson (1972), Ashby (1956). Checks BLL_1+FILT/TIER/L1/L2A/L2B/L3/STRICT/DB/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_bateson_learning_levels_cert_v1")
    validator = os.path.join(fam_dir, "qa_bateson_learning_levels_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_bateson_learning_levels_cert_v1/qa_bateson_learning_levels_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_bateson_learning_levels_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_bateson_learning_levels_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_bateson_learning_levels_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_levin_cognitive_lightcone_cert_family(base_dir: str) -> Optional[str]:
    """QA Levin Cognitive Lightcone Cert family [193] — maps Michael Levin's cognitive light cone (CLC) to QA orbit radius. Singularity=radius 0 (fixed point, no goals), Satellite=radius 8 (8-cycle, local goals), Cosmos=radius 24 (24-cycle, far-reaching goals). Cancer = CLC shrinkage = Cosmos->Satellite orbit transition. Tiered Reachability [191]: 26% L1-reachable = structural CLC ceiling. Source: Levin & Resnik 'Mind Everywhere' (Biological Theory 2026); Lyons/Pio-Lopez/Levin 'Cancer to AI Alignment' (Preprints 2026). Checks CLC_1+ORBIT/RADIUS/CANCER/CEIL/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_levin_cognitive_lightcone_cert_v1")
    validator = os.path.join(fam_dir, "qa_levin_cognitive_lightcone_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_levin_cognitive_lightcone_cert_v1/qa_levin_cognitive_lightcone_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_levin_cognitive_lightcone_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_levin_cognitive_lightcone_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_levin_cognitive_lightcone_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_cognition_space_morphospace_cert_family(base_dir: str) -> Optional[str]:
    """QA Cognition Space Morphospace Cert family [194] — maps Sole, Seoane et al. 'Cognition spaces' (arXiv:2601.12837) qualitative morphospace to QA exact discrete morphospace. Three clusters (basal/neural/human-AI) = three QA orbits. Voids = algebraically necessary. Agency = |reachable set|/|total states|: Singularity=1/81, Satellite=8/81, Cosmos=72/81. Source: Sole et al. arXiv:2601.12837. Checks CSM_1+CLUSTERS/VOIDS/AGENCY/ENUM/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_cognition_space_morphospace_cert_v1")
    validator = os.path.join(fam_dir, "qa_cognition_space_morphospace_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_cognition_space_morphospace_cert_v1/qa_cognition_space_morphospace_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_cognition_space_morphospace_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_cognition_space_morphospace_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_cognition_space_morphospace_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_pezzulo_levin_bootstrap_cert_family(base_dir: str) -> Optional[str]:
    """QA Pezzulo Levin Bootstrap Cert family [195] — maps Pezzulo & Levin 'Bootstrapping Life-Inspired Machine Intelligence' (arXiv:2602.08079) 7-stage pipeline to QA architecture levels. Chemistry=A1, Metabolic=single-step T(b,e), Transcriptional=v_3(f) orbit classification, Anatomical=orbit+E8, Behavioral=observer projection (Theorem NT), Abstract=multi-modulus L_2, Creativity=L_3 pi(9)=24. Intelligence ratchet = Pisano FP [192]. 5 design principles map to QA axioms. Source: Pezzulo & Levin arXiv:2602.08079. Checks PLB_1+STAGES/RATCHET/PRINCIPLES/PIPELINE/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_pezzulo_levin_bootstrap_cert_v1")
    validator = os.path.join(fam_dir, "qa_pezzulo_levin_bootstrap_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_pezzulo_levin_bootstrap_cert_v1/qa_pezzulo_levin_bootstrap_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_pezzulo_levin_bootstrap_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_pezzulo_levin_bootstrap_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_pezzulo_levin_bootstrap_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_see_capture_convergence_cert_family(base_dir: str) -> Optional[str]:
    """QA See Capture Convergence Cert family [196] — maps T.J.J. See's capture theory of cosmical evolution (1909-1910) to QA transient-to-periodic orbit convergence. Free body = arbitrary (b,e); resisting medium = modular reduction; eccentricity decay = transient steps; stable capture = orbit membership. All 81 S_9 states instantly captured (tau=0); extended conditions tau=1. Distribution: cosmos=72, satellite=8, singularity=1. Source: See, 'Capture Theory' (1910). Checks SCC_1+CONV/MEAN/MAX/DIST/MED/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_see_capture_convergence_cert_v1")
    validator = os.path.join(fam_dir, "qa_see_capture_convergence_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_see_capture_convergence_cert_v1/qa_see_capture_convergence_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_see_capture_convergence_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_see_capture_convergence_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_see_capture_convergence_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_see_longitudinal_transverse_cert_family(base_dir: str) -> Optional[str]:
    """QA See Longitudinal Transverse Cert family [197] — maps T.J.J. See's wave duality (longitudinal/compression=gravity, transverse/shear=light; same medium, orthogonal modes) to QA generator/observer duality (T-operator=discrete causal, projection=continuous measurement). Theorem NT = mode orthogonality. Complementary to [153] Keely triune (3-fold within longitudinal) vs See (2-mode between generator/observer). Source: See, 'Electrodynamic Wave-Theory' (1917). Checks SLT_1+LONG/TRANS/ORTH/NT/KEELY/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_see_longitudinal_transverse_cert_v1")
    validator = os.path.join(fam_dir, "qa_see_longitudinal_transverse_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_see_longitudinal_transverse_cert_v1/qa_see_longitudinal_transverse_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_see_longitudinal_transverse_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_see_longitudinal_transverse_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_see_longitudinal_transverse_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_snell_manuscript_cert_family(base_dir: str) -> Optional[str]:
    """QA Snell Manuscript Cert family [201] — certifies 7 structural claims from the Snell Manuscript (Keely's own notes, compiled 1934 by C.W. Snell) as QA invariants: 7x3=21 hierarchy, frequency scaling by 3/9, Trexar orbit encoding (Ag/Au/Pt={3,6,9}), mass-as-difference=f-value, polarity inversion at 2/3, triple dissociation=orbit descent, rotation from 3:9 ratio. Checks SNM_1+21/FREQ/SCALE/TREX/FVAL/POL/DISS/ROT/CHORD/W/F; 2 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_snell_manuscript_cert_v1")
    validator = os.path.join(fam_dir, "qa_snell_manuscript_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_snell_manuscript_cert_v1/qa_snell_manuscript_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_snell_manuscript_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_snell_manuscript_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_snell_manuscript_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_hebrew_mod9_identity_cert_family(base_dir: str) -> Optional[str]:
    """QA Hebrew Mod-9 Identity Cert family [202] — certifies the structural identity between Hebrew gematria mod-9 reduction (Aiq Bekar / pythmen) and QA A1 axiom state space {1,...,9}. Maps three enneads (27 signs), digital root homomorphism (Izmirli 2014), Sefer Yetzirah 4!=24, Skinner metrological 9^4=6561 kernel, 9->24 bridge via factor 6, Pythagorean transmission (Iamblichus), base-9 hypothesis (Kreinovich 2018). Checks HM9_1+AIQ/DR/ENNEAD/SY24/SKIN/BRIDGE/NUM/W/F; 2 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_hebrew_mod9_identity_cert_v1")
    validator = os.path.join(fam_dir, "qa_hebrew_mod9_identity_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_hebrew_mod9_identity_cert_v1/qa_hebrew_mod9_identity_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_hebrew_mod9_identity_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_hebrew_mod9_identity_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_hebrew_mod9_identity_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_self_tested_family(base_dir: str, fam_root: str, validator_name: str, label: str) -> Optional[str]:
    import subprocess
    fam_dir = os.path.join(base_dir, fam_root)
    validator = os.path.join(fam_dir, validator_name)
    if not os.path.exists(validator):
        return f"missing {fam_root}/{validator_name}"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"{label} self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"{label} self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_pudelko_modular_periodicity_cert_family(base_dir: str) -> Optional[str]:
    """QA Pudelko Modular Periodicity Cert family [198] — partial-verification cert with explicit V2/V5 open-item honesty gates; self-test ok"""
    return _validate_self_tested_family(
        base_dir,
        "qa_pudelko_modular_periodicity_cert_v1",
        "qa_pudelko_modular_periodicity_cert_validate.py",
        "qa_pudelko_modular_periodicity_cert",
    )


def _validate_grokking_eigenvalue_transition_cert_family(base_dir: str) -> Optional[str]:
    """QA Grokking Eigenvalue Transition Cert family [199] — partial-verification cert separating DFT mode count from QA orbit-family count; self-test ok"""
    return _validate_self_tested_family(
        base_dir,
        "qa_grokking_eigenvalue_transition_cert_v1",
        "qa_grokking_eigenvalue_transition_cert_validate.py",
        "qa_grokking_eigenvalue_transition_cert",
    )


def _validate_spherical_grokking_theorem_nt_cert_family(base_dir: str) -> Optional[str]:
    """QA Spherical Grokking Theorem NT Cert family [200] — partial-verification cert preserving local 3x speedup and untested S5 scope boundary; self-test ok"""
    return _validate_self_tested_family(
        base_dir,
        "qa_spherical_grokking_theorem_nt_cert_v1",
        "qa_spherical_grokking_theorem_nt_cert_validate.py",
        "qa_spherical_grokking_theorem_nt_cert",
    )


def _validate_sefer_yetzirah_combinatorics_cert_family(base_dir: str) -> Optional[str]:
    """QA Sefer Yetzirah Combinatorics Cert family [203] — certifies combinatorial structures in the Sefer Yetzirah (Book of Formation, c. 2nd-6th century CE): 231 gates = C(22,2) = K_22 complete graph, factorial computation n! for n=2..7 (earliest known), 3-7-12 letter partition, 32 paths = 10+22 = 2^5, oscillating circle, Pythagorean transmission (Iamblichus), tzeruf permutation groups. Checks SYC_1+GATES/FACT/PART/PATHS/NUM/W/F; 2 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_sefer_yetzirah_combinatorics_cert_v1")
    validator = os.path.join(fam_dir, "qa_sefer_yetzirah_combinatorics_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_sefer_yetzirah_combinatorics_cert_v1/qa_sefer_yetzirah_combinatorics_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_sefer_yetzirah_combinatorics_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_sefer_yetzirah_combinatorics_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_sefer_yetzirah_combinatorics_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_skinner_hebrew_metrology_cert_family(base_dir: str) -> Optional[str]:
    """QA Skinner Hebrew Metrology Cert family [204] — certifies 7 verified metrological claims from Skinner's 'Source of Measures' (1875): Parker kernel 6561=9^4, Garden-Eden=24 via digital roots, solar day 5184=72^2 (Cosmos pairs), Adam/Woman dr=9, factor 6 bridge mod-9->mod-24, Metius dr-closure dr(113)+dr(355)=9, T2 compliance (pi as observer output). 3 qualified: El=31 subgroup, palindrome trivial, Parker pi mediocre. Checks SKM_1+PARK/EDEN/SOLAR/ADAM/BRIDGE/MET/NUM/W/F; 2 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_skinner_hebrew_metrology_cert_v1")
    validator = os.path.join(fam_dir, "qa_skinner_hebrew_metrology_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_skinner_hebrew_metrology_cert_v1/qa_skinner_hebrew_metrology_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_skinner_hebrew_metrology_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_skinner_hebrew_metrology_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_skinner_hebrew_metrology_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_grid_cell_rns_cert_family(base_dir: str) -> Optional[str]:
    """QA Grid Cell RNS Cert family [205] — certifies structural isomorphism between entorhinal grid cell residue number system and QA modular arithmetic: RNS isomorphism, CRT reconstruction, 24/9 ratio within 2% of optimal e (Wei 2015), LCM(9,24)=72=Cosmos orbit, golden ratio for two-module (Vago 2018), carry-free=axiom independence, abstract domain (Constantinescu 2016), toroidal state space, hex27 encoding. Checks GCR_1+RATIO/LCM/PHI6/HEX/NUM/W/F; 2 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_grid_cell_rns_cert_v1")
    validator = os.path.join(fam_dir, "qa_grid_cell_rns_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_grid_cell_rns_cert_v1/qa_grid_cell_rns_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_grid_cell_rns_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_grid_cell_rns_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_grid_cell_rns_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_hera_orchestration_evolution_cert_family(base_dir: str) -> Optional[str]:
    """QA HERA Orchestration Evolution Cert family [206] — certifies structural correspondence between HERA multi-agent orchestration evolution (Li & Ramakrishnan VT 2026, arXiv:2604.00901) and QA orbit dynamics: RoPE dual-axes = Bateson [191] L1/L2a, four-phase topology = orbit descent (Satellite->Cosmos->Satellite, NOT Singularity), entropy plateau = Satellite convergence, sparse exploration = orbit discovery, Theorem NT compliance. 38.69% improvement over SOTA. Checks HOE_1+ROPE/PHASE/ENT/PERF/W/F; 2 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_hera_orchestration_evolution_cert_v1")
    validator = os.path.join(fam_dir, "qa_hera_orchestration_evolution_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_hera_orchestration_evolution_cert_v1/qa_hera_orchestration_evolution_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_hera_orchestration_evolution_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_hera_orchestration_evolution_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_hera_orchestration_evolution_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_circle_impossibility_cert_family(base_dir: str) -> Optional[str]:
    """QA Circle Impossibility Cert family [207] — proves no QA state has C=0 (true circle impossible in integer arithmetic). C=2de, A1 says d,e>=1, so C>=2 always. The circle is an observer projection of an ellipsoid where C lies along the viewing axis (Will Dale 2026-04-08). Extends [140] conic discriminant impossibility (parabola). Connects [189] Dale Circle, [125] Chromogeometry, Theorem NT. Checks CI_1+C_MIN/EXHAUSTIVE/PROJECTION/HIERARCHY/CHROMO/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_circle_impossibility_cert_v1")
    validator = os.path.join(fam_dir, "qa_circle_impossibility_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_circle_impossibility_cert_v1/qa_circle_impossibility_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_circle_impossibility_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_circle_impossibility_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_circle_impossibility_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_quadrance_product_cert_family(base_dir: str) -> Optional[str]:
    """QA Quadrance Product Cert family [208] — proves every QA area element is irreducibly a two-factor product of role-distinct base elements. Quadrances (A=a*a, B=b*b) are products, not powers. S1 (always b*b product form) is structural. Even 1*1=1 is an area, not scalar. Square = rectangle with equal sides, product never collapses. Parallels [207] circle impossibility. Source: Will Dale 2026-04-08. Checks QP_1+PRODUCT/ROLE/S1/AREA_MIN/DIM/SQUARE/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_quadrance_product_cert_v1")
    validator = os.path.join(fam_dir, "qa_quadrance_product_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_quadrance_product_cert_v1/qa_quadrance_product_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_quadrance_product_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_quadrance_product_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_quadrance_product_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_fuller_ve_diagonal_decomposition_cert_family(base_dir: str) -> Optional[str]:
    """QA Fuller VE Diagonal Decomposition Cert family [217] — Fuller's cuboctahedral / vector-equilibrium shell count S_n = 10n^2+2 (12, 42, 92, 162, 252, 362, ...) decomposes across QA integer diagonals by n mod 3: n not divisible by 3 => on b=e diagonal D_1 with tuple (S_n/3, S_n/3, 2*S_n/3, S_n); n divisible by 3 => off D_1 on sibling odd-divisor diagonal D_k with (2k+1)|S_n. Proof: S_n mod 3 = (n^2+2) mod 3 = 0 iff n not div by 3. First documented hierarchy whose QA decomposition is mixed across diagonal classes (complements FST/STF entirely on D_1). Mod-3 selection is QA-native triune; 2:1 density ratio of on- vs off-diagonal shells. Source: Will Dale + Claude 2026-04-13, Buckminster Fuller Synergetics (1975). Checks FVDD_1+FORMULA/MOD3/DIAGONAL/OFFDIAGONAL/COMPUTATIONAL/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_fuller_ve_diagonal_decomposition_cert_v1")
    validator = os.path.join(fam_dir, "qa_fuller_ve_diagonal_decomposition_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_fuller_ve_diagonal_decomposition_cert_v1/qa_fuller_ve_diagonal_decomposition_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_fuller_ve_diagonal_decomposition_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_fuller_ve_diagonal_decomposition_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_fuller_ve_diagonal_decomposition_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None



def _validate_haramein_scaling_diagonal_cert_family(base_dir):
    """QA Haramein Scaling Diagonal Cert family [218] — Haramein-Rauscher-Hyson 2008 Table 1 (Big Bang/Planck, Atomic, Stellar Solar, Galactic G1/G2, Universe) encoded as integer (log10 R cm, log10 nu Hz) tuples sit on a QA fixed-d hyperbola (b+e = const, the Schwarzschild line R*nu = c after decade-rounding) and exhibit four structural segment-ratios on 2D Euclidean distances whose integer quadratic-form quotients approximate phi^2 or 1/phi^2 to <= 7%: (25^2+25^2)/(16^2+15^2) = 1250/481 ~ phi^2; (6^2+7^2)/(4^2+4^2) = 85/32 ~ phi^2; (2^2+3^2)/(4^2+4^2) = 13/32 ~ 1/phi^2; (16^2+16^2)/(25^2+25^2) = 512/1250 ~ 1/phi^2. Null (N=200000 random slope-minus-1 6-point placements, same structural pair positions): p < 5e-6. Places Haramein hierarchy in Q(sqrt 5) = Z[phi] algebraic family on fixed-d diagonal (distinct from [217] b=e D_1, companion to [163]). Primary source: Documents/haramein_rsf/scale_unification_2008.pdf. Theory: docs/theory/QA_HARAMEIN_SCALING_DIAGONAL.md. Source: Will Dale + Claude 2026-04-13. Checks HSD_1+TABLE/FIXED_D/SEGMENTS/QUADRATIC/PHI_RATIOS/NULL/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_haramein_scaling_diagonal_cert_v1")
    validator = os.path.join(fam_dir, "qa_haramein_scaling_diagonal_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_haramein_scaling_diagonal_cert_v1/qa_haramein_scaling_diagonal_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_haramein_scaling_diagonal_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_haramein_scaling_diagonal_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_haramein_scaling_diagonal_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None



def _validate_madelung_d_ordering_cert_family(base_dir):
    """QA Madelung d-Ordering Cert family [220] — atomic subshells (n,l), n>=1, 0<=l<=n-1, identified as QA (b,e)=(n,l). Then d=b+e=n+l IS the Madelung quantum. Aufbau filling order EXACTLY = QA (d,-e) ascending sort, verified over first 36 Janet-extended subshells (1s through 11s) with zero mismatches. Deterministic selection rule: within-d antidiagonal (b,e)->(b+1,e-1) when e>0; between-d jump (b,0)->(ceil((b+2)/2), floor(b/2)) when e=0; holds for 35/35 transitions. Derived: subshell pop = 4e+2 = 2(2l+1); period-k pop = 2*ceil((k+1)/2)^2 = {2,8,8,18,18,32,32,50,...} (matches physical periodic table periods 1-7 + Janet predictions 8+); shell-n cumulative total = 2*Sum k^2 through n. Distinct QA class from [217]/[218]/[219]: d-ordering walk across ALL d-classes, NOT a Q(sqrt 5) / Z[phi] structure. Madelung rule was empirical (Madelung 1936, Klechkowski 1962); QA promotes it from aufbau heuristic to structural consequence of A2 axiom. Does NOT claim derivation from Schrodinger or prediction of Madelung anomalies (Cr/Cu/lanthanides) — those live in SVP/permissibility semantic layer. Source: Will Dale + Claude 2026-04-13. Theory: docs/theory/QA_MADELUNG_D_ORDERING.md. Checks MAD_1+MAPPING/ORDER/RULE/POP/PERIODS/SHELL/CLASS/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_madelung_d_ordering_cert_v1")
    validator = os.path.join(fam_dir, "qa_madelung_d_ordering_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_madelung_d_ordering_cert_v1/qa_madelung_d_ordering_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_madelung_d_ordering_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_madelung_d_ordering_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_madelung_d_ordering_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None



def _validate_nuclear_magic_spin_extension_cert_family(base_dir):
    """QA Nuclear Magic Spin-Extension Cert family [221] — extends QA with Dirac axiom D1 (sigma in {1,2} encodes spin alignment, j = l + (2*sigma-3)/2, pop = 2j+1 = 2(e+sigma-1)). Maps (b,e)=(n,l), HO shell N=2b-e-2. Fractional-1/2 promotion: when sigma=2 AND b=e+1 AND l>=l*, N_eff = N - 1/2. The 1/2 is Dirac spin unit (derived from D1). Threshold l* derived from single physics input P1: r=alpha/hbar_omega in [1/3, 1/2), giving l*=ceil(1/r)=3 by integer-ceiling. Empirical nuclear r~0.3-0.4 inside window (Mayer-Jensen 1950, Bohr-Mottelson 1969). Atomic r~0.01-0.02 gives l*>50 never reached, explaining why [220] Madelung needs no extension. Magic-shell criterion: N_eff in {0,1,2} OR N_eff half-integer. Cumulative populations at magic closures = {2,8,20,28,50,82,126}, all 7 experimental nuclear magic numbers exactly. Non-magic integer N_eff>=3 closures at 40,70,112,168 (HO residues, physically smaller gaps). Scope: combinatorial identity plus one discrete physical ratio input; honest framing of what QA derives vs physics supplies. Will Dale + Claude 2026-04-13. Theory note: docs/theory/QA_NUCLEAR_MAGIC_SPIN_EXTENSION.md. Checks NMS_1+D1/HO/PROMOTION/THRESHOLD/MAGIC/P1/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_nuclear_magic_spin_extension_cert_v1")
    validator = os.path.join(fam_dir, "qa_nuclear_magic_spin_extension_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_nuclear_magic_spin_extension_cert_v1/qa_nuclear_magic_spin_extension_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_nuclear_magic_spin_extension_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_nuclear_magic_spin_extension_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_nuclear_magic_spin_extension_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None



def _validate_madelung_anomaly_boundary_cert_family(base_dir):
    """QA Madelung Anomaly Boundary Cert family [222] — every known atomic Madelung anomaly (20 total: Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag, La, Ce, Gd, Pt, Au, Ac, Th, Pa, U, Np, Cm, Lr) satisfies |d(src) - d(dst)| <= 1 in QA (n,l) = (b,e). 10 at |Δd|=0 (intra-class f↔d in lanthanides/actinides, d↔p in Lr); 10 at |Δd|=1 (inter-class s↔d). Null (uniform random 2-subshell pairs from first 20 Madelung positions): 36.8% in zone; observed 100% (20/20); enrichment 2.71x; binomial p = 2.1e-9. Necessary but not sufficient — Ti/V/Mn/Fe/Co/Ni (d=4↔5) and Ta/W/Re/Os/Ir (d=6↔7) in zone follow Madelung. Falsifiable: newly-discovered anomaly with |Δd|>=2 breaks claim. Extends [220] d-ordering; complements [221] nuclear-magic-spin-extension. QA provides boundary structure; exchange/relativistic mechanisms in semantic layer. Sources: NIST Atomic Spectra Database + Sato et al. 2015 Nature 520 (Lr 7p1). Will Dale + Claude 2026-04-13. Theory: docs/theory/QA_MADELUNG_ANOMALY_BOUNDARY.md. Checks MAB_1+ANOMALIES/MAPPING/ZONE/COVERAGE/NULL/COUNTEREX/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_madelung_anomaly_boundary_cert_v1")
    validator = os.path.join(fam_dir, "qa_madelung_anomaly_boundary_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_madelung_anomaly_boundary_cert_v1/qa_madelung_anomaly_boundary_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_madelung_anomaly_boundary_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_madelung_anomaly_boundary_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_madelung_anomaly_boundary_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_hyper_catalan_diagonal_cert_family(base_dir):
    """QA Hyper-Catalan Diagonal Correspondence Cert family [231] — under b=V_m-1 and e=F_m for Wildberger-Rubine hyper-Catalan multi-index m, d=b+e equals E_m exactly and Euler V_m-E_m+F_m=1 follows. Single-type m_k=n sits on sibling diagonal b=(k-1)e+1; Catalan/Fuss single-type values match OEIS A000108/A001764/A002293/A002294; no single-type case k in [2,7], n in [0,9] sits on D_1. Source: Wildberger-Rubine 2025; Will Dale + Claude 2026-04-13. Checks HCD_1+EULER/OEIS/FUSS/SINGLE_DIAGONAL/D1_DISJOINT/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_hyper_catalan_diagonal_cert_v1")
    validator = os.path.join(fam_dir, "qa_hyper_catalan_diagonal_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_hyper_catalan_diagonal_cert_v1/qa_hyper_catalan_diagonal_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_hyper_catalan_diagonal_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_hyper_catalan_diagonal_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_hyper_catalan_diagonal_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_uhg_diagonal_coincidence_cert_family(base_dir):
    """QA UHG Diagonal Coincidence Cert family [232] — at m=9 on {1,...,9}^2, UHG zero quadrance under <a,b>=-(b1e2+e1b2) coincides exactly with QA gcd-reduced diagonal class: 64 unordered zero-quadrance pairs, 64 same-diagonal pairs, intersection 64, zero counterexamples either direction. Source: Wildberger UHG I 2013; Will Dale + Claude 2026-04-13. Checks UDC_1+M/COUNTS/INTERSECTION/COUNTEREXAMPLES/WITNESS/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_uhg_diagonal_coincidence_cert_v1")
    validator = os.path.join(fam_dir, "qa_uhg_diagonal_coincidence_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_uhg_diagonal_coincidence_cert_v1/qa_uhg_diagonal_coincidence_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_uhg_diagonal_coincidence_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_uhg_diagonal_coincidence_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_uhg_diagonal_coincidence_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_uhg_orbit_diagonal_profile_cert_family(base_dir):
    """QA UHG Orbit Diagonal Profile Cert family [233] — at m=9, QA T-step partitions 81 points into 1 singularity length 1, 2 satellite orbits length 4, and 6 cosmos orbits length 12. Every non-singular D_1-containing orbit has exactly two D_1 points summing to (9,9): Sat#1, Cos#1, Cos#3, Cos#4. Source: Will Dale + Claude 2026-04-13. Checks UODP_1+M/PARTITION/ORBIT_DATA/D1_PROFILE/COMPLEMENT/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_uhg_orbit_diagonal_profile_cert_v1")
    validator = os.path.join(fam_dir, "qa_uhg_orbit_diagonal_profile_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_uhg_orbit_diagonal_profile_cert_v1/qa_uhg_orbit_diagonal_profile_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_uhg_orbit_diagonal_profile_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_uhg_orbit_diagonal_profile_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_uhg_orbit_diagonal_profile_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_g_equivariant_cnn_structural_cert_family(base_dir):
    """QA G-Equivariant CNN Structural Cert family [247] — closed-form Cohen-Welling rotation-index algebra: phi(b)=b mod n is a bijection {1,...,n}->Z/nZ with explicit inverse; qa_step(b1,b2,n)=((b1+b2-1) mod n)+1 preserves addition under phi exhaustively at n=9 and n=24; single-generator n=9 iteration partitions 81 pairs into singularity/satellite/cosmos with counts 9/18/54 and zero exceptions; Eq. 10 lifting = observer IN, Eq. 11 G-correlation = QA-layer resonance, §6.3 coset pooling = observer OUT. Primary source: Cohen and Welling 2016. Checks GECS_1+BIJECT/COMPOSE/ORBIT/CORR/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir = os.path.join(base_dir, "qa_g_equivariant_cnn_structural_cert_v1")
    validator = os.path.join(fam_dir, "qa_g_equivariant_cnn_structural_cert_v1.py")
    if not os.path.exists(validator):
        return "missing qa_g_equivariant_cnn_structural_cert_v1/qa_g_equivariant_cnn_structural_cert_v1.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_g_equivariant_cnn_structural_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_g_equivariant_cnn_structural_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_g_equivariant_cnn_structural_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_formal_conjecture_resolution_cert_family(base_dir):
    """QA Formal Conjecture Resolution Cert family [248] — QA-native record of conjecture-resolution attempts with typed obstruction (proved / formal_gap / qa_obstruction / generator_insufficient / inconclusive). Mirrors the Ju et al. (2026) Rethlas+Archon pipeline structure; QA contribution is the typed failure_mode layer that distinguishes formal_gap from qa_obstruction from generator_insufficient. Primary source: Ju, Gao, Jiang, Wu, Sun, Chen, Wang, Wang, Wang, He, Wu, Xiao, Liu, Dai, Dong (2026), 'Automated Conjecture Resolution with Formal Verification,' arXiv:2604.03789. Theory: docs/theory/QA_AUTOMATED_CONJECTURE_RESOLUTION.md. Checks FCR_1 schema, FCR_2 generator_set non-empty, FCR_3 typed failure_mode required when not proved, FCR_4 NT compliance, FCR_5 verdict vocabulary, FCR_6 witness path, FCR_7 lean4_stub open-questions; 2 PASS (proved + formal_gap) + 1 FAIL (missing failure label); self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_formal_conjecture_resolution_cert_v1")
    validator = os.path.join(fam_dir, "qa_formal_conjecture_resolution_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_formal_conjecture_resolution_cert_v1/qa_formal_conjecture_resolution_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_formal_conjecture_resolution_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_formal_conjecture_resolution_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_formal_conjecture_resolution_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_chromogeometry_pythagorean_identity_cert_family(base_dir):
    """QA Chromogeometry Pythagorean Identity Cert family [234] — with Q_b=b*b+e*e, Q_r=b*b-e*e, and Q_g=2*b*e, Wildberger's chromogeometry identity Q_b square = Q_r square + Q_g square holds exhaustively over (b,e) in [1..19]^2 with zero failures; QA coordinate forms Q_r=(b-e)d, Q_g=2be, Q_b=b*b+e*e are verified. Source: Wildberger Chromogeometry 2008; Will Dale + Claude 2026-04-13. Checks CPI_1+SAMPLES/RANGE/FORMULAS/PLIMPTON/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_chromogeometry_pythagorean_identity_cert_v1")
    validator = os.path.join(fam_dir, "qa_chromogeometry_pythagorean_identity_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_chromogeometry_pythagorean_identity_cert_v1/qa_chromogeometry_pythagorean_identity_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_chromogeometry_pythagorean_identity_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_chromogeometry_pythagorean_identity_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_chromogeometry_pythagorean_identity_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_super_catalan_diagonal_cert_family(base_dir):
    """QA Super Catalan Diagonal Cert family [235] — Limanta-Wildberger super Catalan numbers S(m,n) identify with QA coordinates (b,e)=(m,n), so d=b+e gives the formula denominator factor (m+n)! = d!. S(b,b) for b=0..10 matches OEIS A000984 central binomials; swap symmetry and recurrence 4*S(b,e)=S(b+1,e)+S(b,e+1) hold exhaustively on [0..7]^2; S(1,n)=2*Catalan(n) for n=0..9. Source: Limanta + Wildberger 2021/2022; Will Dale + Claude 2026-04-13. Checks SCD_1+D1_A000984/SYMMETRY/RECURRENCE/CATALAN/QA_IDENT/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_super_catalan_diagonal_cert_v1")
    validator = os.path.join(fam_dir, "qa_super_catalan_diagonal_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_super_catalan_diagonal_cert_v1/qa_super_catalan_diagonal_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_super_catalan_diagonal_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_super_catalan_diagonal_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_super_catalan_diagonal_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_spread_polynomial_composition_cert_family(base_dir):
    """QA Spread Polynomial Composition Cert family [236] — Goh-Wildberger spread polynomials S_0=0, S_1=s, S_{n+1}=2(1-2s)*S_n-S_{n-1}+2s satisfy S_n composed with S_m = S_{n*m}. Exact SymPy composition verifies pairs (2,3), (3,2), (2,4), (4,3), (3,3), (2,5); S_2=4*s*(1-s) is the logistic map; integer closed forms for S_2/S_3/S_4 match. Trig identity recorded but skipped as float-dependent. Source: Goh + Wildberger 2009; Will Dale + Claude 2026-04-13. Checks SPC_1+COMPOSITION/CLOSED_FORMS/LOGISTIC/TRIG_NOTE/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_spread_polynomial_composition_cert_v1")
    validator = os.path.join(fam_dir, "qa_spread_polynomial_composition_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_spread_polynomial_composition_cert_v1/qa_spread_polynomial_composition_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_spread_polynomial_composition_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_spread_polynomial_composition_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_spread_polynomial_composition_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_4d_diagonal_rule_cert_family(base_dir):
    """QA 4D Diagonal Rule Cert family [237] — QA tuple (b,e,d,a) with d=b+e and a=b+2e is exactly b*v1+e*v2 in the 2-plane of R^4 spanned by v1=(1,0,1,1) and v2=(0,1,1,2). Integer embedding verified for b,e in [-5..5]; Gram matrix [[3,3],[3,6]] has determinant 9 equal to QA canonical modulus m; two concrete perpendicular QA-tuple witnesses satisfy Wildberger's Diagonal Rule Q1+Q2=Q3. Source: Wildberger KoG 21:47-54, 2017; Will Dale + Claude 2026-04-13. Checks Q4D_1+EMBED/GRAM/MODULUS/DIAGONAL_RULE/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_4d_diagonal_rule_cert_v1")
    validator = os.path.join(fam_dir, "qa_4d_diagonal_rule_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_4d_diagonal_rule_cert_v1/qa_4d_diagonal_rule_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_4d_diagonal_rule_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_4d_diagonal_rule_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_4d_diagonal_rule_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_twelve_dihedral_orderings_cert_family(base_dir):
    """QA Twelve Dihedral Orderings Cert family [239] - five objects under D_5 give 5!/(2*5)=12 dihedral classes, each of size 10, by exhaustive permutation enumeration and canonicalization via 5 rotations plus 5 reflections. The 12-count is recorded as G_2 non-identity root count, cuboctahedral S_1, and icosahedral vertex count. Source: Le + Wildberger 2020; Will Dale + Claude 2026-04-14. Checks TDO_1+GROUP/PERMUTATIONS/CANONICAL_REPS/CLASS_COUNT/CLASS_SIZE/FORMULA/QA_CONNECTION/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_twelve_dihedral_orderings_cert_v1")
    validator = os.path.join(fam_dir, "qa_twelve_dihedral_orderings_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_twelve_dihedral_orderings_cert_v1/qa_twelve_dihedral_orderings_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_twelve_dihedral_orderings_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_twelve_dihedral_orderings_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_twelve_dihedral_orderings_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_diamond_sl3_irrep_dimension_cert_family(base_dir):
    """QA Diamond sl3 Irrep Dimension Cert family [240] - Wildberger's diamond model bridge: under (qa_b,qa_e)=(sl3_a,sl3_b), d=b+e gives dim pi[a,b]=(b+1)(e+1)(d+2)/2 as an integer QA polynomial. Verifies 22 standard entries, adjoint dim 8, quark/anti-quark dim 3 triples, triangular D(a,0) column, and integer Heisenberg commutators. Source: Wildberger 2003; Will Dale + Claude 2026-04-14. Checks DSI_1+DIM_FORMULA/ADJOINT/TRIANGULAR_COLUMN/QUARK_ANTIQUARK/HEISENBERG/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_diamond_sl3_irrep_dimension_cert_v1")
    validator = os.path.join(fam_dir, "qa_diamond_sl3_irrep_dimension_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_diamond_sl3_irrep_dimension_cert_v1/qa_diamond_sl3_irrep_dimension_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_diamond_sl3_irrep_dimension_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_diamond_sl3_irrep_dimension_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_diamond_sl3_irrep_dimension_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_quadruple_coplanarity_cert_family(base_dir):
    """QA Quadruple Coplanarity Cert family [241] - every QA point (b,e,d) with d=b+e lies in the plane d-b-e=0 in R3, so every four QA points have zero 4-point Cayley-Menger determinant under blue, red, and green chromogeometric quadrances. Uses SymPy integer determinants; checks [-9..9]^2 plane identity, 30 triples, 30 quadruples plus Satellite #1. Source: Notowidigdo + Wildberger 2019/2021 and Wildberger Chromogeometry 2008; Will Dale + Claude 2026-04-14. Checks QCO_1+PLANE_IDENTITY/PARALLELEPIPED_VOL/CM_4POINT_BLUE_RED_GREEN/CHROMO_COPLANARITY_PRESERVED/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_quadruple_coplanarity_cert_v1")
    validator = os.path.join(fam_dir, "qa_quadruple_coplanarity_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_quadruple_coplanarity_cert_v1/qa_quadruple_coplanarity_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_quadruple_coplanarity_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_quadruple_coplanarity_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_quadruple_coplanarity_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_sl3_hexagonal_ring_identity_cert_family(base_dir):
    """QA SL3 Hexagonal Ring Identity Cert family [245] - Wildberger sl3 diamond follow-up: ring(a,b)=dim pi[a,b]-dim pi[a-1,b-1]=T_{d+1}+a*b under QA coordinates (b_QA,e_QA)=(a,b). Verifies cleared-denominator symbolic expansion, all 196 entries on [1..14]^2, QA coordinate form, and known multiplicities ring(1,1)=7, ring(2,1)=12, ring(2,2)=19. Source: Wildberger 2003; Will Dale + Claude 2026-04-14. Checks SHR_1+ALGEBRAIC_EXPANSION/EXHAUSTIVE/QA_COORD_FORM/KNOWN_MULTIPLICITIES/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_sl3_hexagonal_ring_identity_cert_v1")
    validator = os.path.join(fam_dir, "qa_sl3_hexagonal_ring_identity_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_sl3_hexagonal_ring_identity_cert_v1/qa_sl3_hexagonal_ring_identity_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_sl3_hexagonal_ring_identity_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_sl3_hexagonal_ring_identity_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_sl3_hexagonal_ring_identity_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_chromogeometric_tqf_symmetry_cert_family(base_dir):
    """QA Chromogeometric TQF Symmetry Cert family [246] - Wildberger chromogeometry Triple Quad Formula sign symmetry: TQF_r=TQF_g=-TQF_b for integer-coordinate triangles, with TQF_b=4*area2*area2=16*A*A. Uses SymPy symbolic identities, deterministic 3000-triangle sample from [1..9]^2, and exhaustive C(81,3) collinearity invariant. Source: Wildberger Chromogeometry 2008 and Divine Proportions 2005; Will Dale + Claude 2026-04-14. Checks CTQF_1+SYMBOLIC_RB/SYMBOLIC_GB/SAMPLE_EXHAUSTIVE/FACTORED_BLUE/COLLINEARITY_INVARIANT/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_chromogeometric_tqf_symmetry_cert_v1")
    validator = os.path.join(fam_dir, "qa_chromogeometric_tqf_symmetry_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_chromogeometric_tqf_symmetry_cert_v1/qa_chromogeometric_tqf_symmetry_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_chromogeometric_tqf_symmetry_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_chromogeometric_tqf_symmetry_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_chromogeometric_tqf_symmetry_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_neuberg_cubic_f23_cert_family(base_dir):
    """QA Neuberg Cubic F23 Cert family [242] - Wildberger's finite-field Neuberg setting over F_23: E:y^2=x^3+x+1 has 27 affine points plus infinity, Weierstrass tangent-conic witnesses are enumerated as identical or disjoint F_23 point sets, and the spread witness is an integer-polynomial pair with no division required in the fixture. Source: Wildberger 2008; Will Dale + Claude 2026-04-14. Checks NCF23_1+POINT_COUNT/TANGENT_CONIC_DICHOTOMY/SPREAD_POLYNOMIAL/QA_COMPAT/SRC/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_neuberg_cubic_f23_cert_v1")
    validator = os.path.join(fam_dir, "qa_neuberg_cubic_f23_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_neuberg_cubic_f23_cert_v1/qa_neuberg_cubic_f23_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_neuberg_cubic_f23_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_neuberg_cubic_f23_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_neuberg_cubic_f23_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_mutation_game_root_lattice_cert_family(base_dir):
    """QA Mutation Game Root Lattice Cert family [244] - Wildberger integer Mutation Game on the E_8 root lattice: Cartan determinant 1, BFS orbit closure of size 240, sign split 120+120, every root has norm 2 under G=2I-A, and Weyl involution/braid relations hold on integer populations. Uses only tuple/set BFS, exact Cartan determinant, and integer quadratic form checks. Source: Wildberger 2020 pp.10-11; Will Dale + Claude 2026-04-14. Checks MGR_1+CARTAN/BFS/ROOT_NORM/SIGN_SPLIT/INVOLUTION_BRAID/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_mutation_game_root_lattice_cert_v1")
    validator = os.path.join(fam_dir, "qa_mutation_game_root_lattice_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_mutation_game_root_lattice_cert_v1/qa_mutation_game_root_lattice_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_mutation_game_root_lattice_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_mutation_game_root_lattice_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_mutation_game_root_lattice_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_e8_embedding_orbit_classifier_cert_family(base_dir):
    """QA E8 Embedding Orbit Classifier Cert family [249] - canonical (b,e,d,a)→ℤ^8 embedding into the Wildberger E_8 root lattice [244]; verifies m=9 T-orbit partition {1,8,24,24,24}, closed-form Q(E_diag(b,e))=2(b²+e²+d²+a²)-2(bd+ea+da) symbolic + exhaustive on [1..9]², per-orbit min Q under E_diag = (8,16,28,72,162) is a 5-distinct complete T-orbit classifier, and per-orbit Q-multisets are pairwise distinct under both E_diag and E_tri. E_diag canonical, E_tri recorded informationally. Source: Wildberger 2020 + cert [244]; Will Dale + Claude 2026-04-15. Checks E8E_1+CARTAN_LOAD/T_ORBITS/DIAG_FORMULA/DIAG_MIN_Q/DIAG_MULTISET/TRI_PROFILE/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_e8_embedding_orbit_classifier_cert_v1")
    validator = os.path.join(fam_dir, "qa_e8_embedding_orbit_classifier_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_e8_embedding_orbit_classifier_cert_v1/qa_e8_embedding_orbit_classifier_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_e8_embedding_orbit_classifier_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_e8_embedding_orbit_classifier_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_e8_embedding_orbit_classifier_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_ade_mutation_game_cert_family(base_dir):
    """QA ADE Mutation Game Cert family [250] - extends [244] (E_8 only) to the full simply-laced ADE classification (A_5, D_5, E_6, E_7, E_8) using the same integer Wildberger 2020 Mutation Game BFS. Verifies for each type Cartan determinant = order of center (6,4,3,2,1), Weyl orbit size (30,40,72,126,240) per Humphreys GTM 9 §9.3 Table 1, exhaustive v^T G v = 2, and equal positive/negative split with R-=-R+. Source: Wildberger 2020 + Humphreys 1972 + cert [244]; Will Dale + Claude 2026-04-15. Checks ADE_1+CARTAN_DETS/BFS_SIZES/ROOT_NORM/SIGN_SPLIT/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_ade_mutation_game_cert_v1")
    validator = os.path.join(fam_dir, "qa_ade_mutation_game_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_ade_mutation_game_cert_v1/qa_ade_mutation_game_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=120, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_ade_mutation_game_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_ade_mutation_game_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_ade_mutation_game_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_g2_mutation_game_cert_family(base_dir):
    """QA G2 Mutation Game Cert family [251] - first non-simply-laced Mutation Game cert, extending [244] and [250] to G_2 using directed edge counts A(0->1)=3, A(1->0)=1 for Cartan [[2,-1],[-3,2]]. Verifies BFS closure at 12 integer populations, six positive plus six negative with R-=-R+, Humphreys §12.1 coordinate swap with three short and three long positive roots under G_sr=[[2,-3],[-3,6]], s0^2=s1^2=I, and strict Coxeter order 6. Source: Wildberger 2020 + Humphreys 1972 §12.1 + theory docs/theory/QA_G2_MUTATION_GAME.md commit b86442f; Will Dale + Claude 2026-04-15. Checks G2M_1/G2M_2/G2M_3/G2M_4/G2M_5/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_g2_mutation_game_cert_v1")
    validator = os.path.join(fam_dir, "qa_g2_mutation_game_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_g2_mutation_game_cert_v1/qa_g2_mutation_game_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_g2_mutation_game_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_g2_mutation_game_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_g2_mutation_game_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_signal_generator_inference_cert_family(base_dir: str) -> Optional[str]:
    """QA Signal Generator Inference Cert family [209] — for any m-valued time series, e_t = ((b_{t+1} - b_t - 1) % m) + 1 is the unique A1-compliant generator. The signal IS the orbit; the generator IS the dynamics. b (amplitude state) and e (transition generator) are role-distinct per [208]. Cross-series generator synchrony measures coupling per [207]. Supersedes hardcoded CMAP/MICROSTATE_STATES lookups. EEG chb01: DR2=+0.157 p=0.0003 beyond delta; DR2=+0.085 p=0.024 beyond Observer 3. Source: Will Dale + Claude 2026-04-08. Checks SGI_1+CLOSURE/UNIQUE/ROLE/SYNC/EMPIRICAL/SUPERSEDE/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok"""
    import subprocess
    fam_dir   = os.path.join(base_dir, "qa_signal_generator_inference_cert_v1")
    validator = os.path.join(fam_dir, "qa_signal_generator_inference_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_signal_generator_inference_cert_v1/qa_signal_generator_inference_cert_validate.py"
    proc = subprocess.run(
        [sys.executable, validator, "--self-test"],
        capture_output=True, text=True, timeout=60, cwd=fam_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"qa_signal_generator_inference_cert self-test failed:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(f"qa_signal_generator_inference_cert self-test returned non-JSON:\nerror={exc}\nstdout={(proc.stdout or '').strip()}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"qa_signal_generator_inference_cert self-test ok=false:\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return None


def _validate_kg_consistency_cert_v3(base_dir):
    """QA-KG Consistency Cert [225] v3: validates graph consistency after Phase 1 epistemic fields + alias removal. Schema v3 (epistemic columns authority/epistemic_status/method/source_locator/lifecycle_state). Gates: KG1 no self-vetting, KG2 no contradicts cycles, KG3 Theorem NT firewall tri-state, KG4 satellite orphan aging WARN, KG5 tier ≡ canonical orbit classifier, KG6 Candidate F integrity [202], KG7 epistemic fields non-null, KG8 frozen certs not in FAMILY_SWEEPS, KG9 AXIOM_CODES canonical, KG10 no except-Exception-continue swallows. Source: docs/specs/QA_MEM_SCOPE.md; QA_AXIOMS_BLOCK.md (Dale 2026); CLAUDE.md; cert [226]. Supersedes v2 (frozen)."""
    import subprocess
    validator = os.path.join(base_dir, "qa_kg_consistency_cert_v3", "qa_kg_consistency_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_kg_consistency_cert_v3/qa_kg_consistency_cert_validate.py"
    db_path = os.path.join(os.path.dirname(base_dir), "tools", "qa_kg", "qa_kg.db")
    if not os.path.exists(db_path):
        return None  # DB not built yet — skip
    proc = subprocess.run(
        [sys.executable, validator, "--db", db_path],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"[225] v3 FAIL:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    return None


def _validate_kg_epistemic_fields_cert(base_dir):
    """QA-KG Epistemic Fields Cert [252] v1: validates authority/epistemic_status/method/source_locator/lifecycle_state correctness per Phase 1 QA-MEM. Single source of truth for the allowed authority × epistemic_status matrix is allowed_matrix.json. Gates: EF1 authority non-null, EF2 epistemic_status non-null, EF3 matrix enforcement, EF4 primary source_locator resolves, EF5 agent-authority count (WARN), EF6 Axiom ⇒ primary+axiom. Source: docs/specs/QA_MEM_SCOPE.md; QA_AXIOMS_BLOCK.md (Dale 2026); CLAUDE.md."""
    import subprocess
    validator = os.path.join(base_dir, "qa_kg_epistemic_fields_cert_v1", "qa_kg_epistemic_fields_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_kg_epistemic_fields_cert_v1/qa_kg_epistemic_fields_cert_validate.py"
    db_path = os.path.join(os.path.dirname(base_dir), "tools", "qa_kg", "qa_kg.db")
    if not os.path.exists(db_path):
        return None  # DB not built yet — skip
    proc = subprocess.run(
        [sys.executable, validator, "--db", db_path],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"[252] v1 FAIL:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
    return None


def _validate_kg_firewall_effective_cert(base_dir):
    """QA-KG Firewall Effective Cert [227] v1: validates Phase 2 Theorem NT firewall effectiveness. DB-backed promoted-from check, _meta_ledger.json staleness, broadcast corroboration provenance. Gates: FE1 no unpromoted agent causal edges, FE2 via_cert ledger freshness, FE3 no promoted-from cycles, FE4 ephemeral firewall test, FE5 oldest unpromoted WARN, FE6 provenance snapshot. Source: docs/specs/QA_MEM_SCOPE.md (Dale, 2026); tools/qa_kg/kg.py (promote protocol)."""
    import subprocess
    validator = os.path.join(base_dir, "qa_kg_firewall_effective_cert_v1", "qa_kg_firewall_effective_cert_validate.py")
    if not os.path.exists(validator):
        return "missing qa_kg_firewall_effective_cert_v1/qa_kg_firewall_effective_cert_validate.py"
    db_path = os.path.join(os.path.dirname(base_dir), "tools", "qa_kg", "qa_kg.db")
    if not os.path.exists(db_path):
        return None  # DB not built yet — skip
    proc = subprocess.run(
        [sys.executable, validator, "--db", db_path],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"[227] v1 FAIL:\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}")
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
    (40, "QA Reachability Descent Run Cert family",
     _validate_reachability_descent_run_cert_family_if_present,
     "schema + validator + fixtures (PASS + negative fixtures)", "40_reachability_descent_run_cert", "../qa_reachability_descent_run_cert_v1", True),
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
    (63, "QA Kona EBM QA-Native family",
     _validate_kona_ebm_qa_native_family_if_present,
     "schema + validator + fixtures (QA orbit manifold as latent space, orbit alignment analysis)", "63_kona_ebm_qa_native", "../qa_kona_ebm_qa_native_v1", True),
    (64, "QA Kona EBM QA-Native Orbit Reg family",
     _validate_kona_ebm_qa_native_orbit_reg_family_if_present,
     "schema + validator + fixtures (orbit-coherence regularizer, permutation gap analysis)", "64_kona_ebm_qa_native_orbit_reg", "../qa_kona_ebm_qa_native_orbit_reg_v1", True),
    (71, "QA Curvature Stress-Test Bundle (cross-family \u03ba universality)",
     _validate_curvature_stress_test_family_if_present,
     "schema + validator + 4 fixtures (1 valid, 3 negative)", "71_curvature_stress_test", "../qa_curvature_stress_test_v1", True),
    (72, "QA Guarded Operator Category family",
     _validate_guarded_operator_category_family_if_present,
     "schema + validator + fixtures (matrix embedding + guarded obstructions)", "72_guarded_operator_category", "../qa_guarded_operator_category_v1", True),
    (73, "QA Structural Algebra Cert family",
     _validate_structural_algebra_cert_family_if_present,
     "schema + validator + fixtures (normal forms + bounded uniqueness + scaling + guards)", "73_structural_algebra_cert", "../qa_structural_algebra_cert_v1", True),
    (74, "QA Component Decomposition Cert family",
     _validate_component_decomposition_cert_family_if_present,
     "schema + validator + fixtures (gcd decomposition + scaled roundtrip + nu power-of-two characterization)", "74_component_decomposition_cert", "../qa_component_decomposition_cert_v1", True),
    (75, "QA Algebra Bridge Cert family",
     _validate_algebra_bridge_cert_family_if_present,
     "schema + validator + fixtures (semantics anchor + word convention + component bridge + hash binding)", "75_algebra_bridge_cert", "../qa_algebra_bridge_cert_v1", True),
    (76, "QA Failure Algebra Structure Cert family",
     _validate_failure_algebra_structure_cert_family_if_present,
     "schema + validator + fixtures (finite poset + join-semilattice + monotone composition + propagation law)", "76_failure_algebra_structure_cert", "../qa_failure_algebra_structure_cert_v1", True),
    (77, "QA Neighborhood Sufficiency Cert family",
     _validate_neighborhood_sufficiency_cert_family_if_present,
     "schema v1.1 + validator (branching Gate 3) + 8 fixtures (4 valid: houston, indian_pines, salinas, ksc_failure; 4 negative: not_dominant, no_plateau, digest, claims_dominant_but_negative_delta)", "77_neighborhood_sufficiency_cert",
     "../qa_neighborhood_sufficiency_cert_v1", True),
    (78, "QA Locality Boundary Cert family",
     _validate_locality_boundary_cert_family_if_present,
     "schema v1.2 + validator (6-gate: schema, hash, failure curve, delta flag, fragmentation, adjacency witness Mode A/B) + 7 fixtures (3 valid: ksc_boundary v1/v1.1/v1.2 path mode; 4 negative: not_a_boundary_case, digest_mismatch, adj_rate_wrong, gt_mask_sha_mismatch)", "78_locality_boundary_cert",
     "../qa_locality_boundary_cert_v1", True),
    (79, "QA Locality Regime Separator Cert family",
     _validate_locality_regime_sep_cert_family_if_present,
     "schema v1.2 + validator (7-gate incl. Gate 6 adj witness + Gate 7 predictive threshold adj_crit/epsilon/TRANSITION) + 12 fixtures (4 v1 + 4 v1.1 witness + 4 v1.2: salinas DOMINANT pred, ksc BOUNDARY pred, pred_mismatch, pred_conflicts_empirical)", "79_locality_regime_sep_cert",
     "../qa_locality_regime_sep_cert_v1", True),
    (80, "QA Energy Cert v1.1 (CAPS_TR cognitive domain)",
     _validate_energy_cert_v1_1_family_if_present,
     "schema + validator (7-gate incl. Gate 7 episode_samples consistency) + 7 fixtures (PASS_FEAR, PASS_LOVE, PASS_MIXED+episode, FAIL_POWER, FAIL_INTERACTION, FAIL_HORIZON, FAIL_EPISODE)", "80_energy_cert",
     "../qa_energy_cert_v1_1", True),
    (81, "QA Episode Regime Transitions Cert family",
     _validate_episode_regime_cert_v1_family_if_present,
     "schema + validator (5-gate) + 6 fixtures (PASS_RECOVERING, PASS_ESCALATING, PASS_MIXED, FAIL_LABEL, FAIL_TRANSITION, FAIL_DRIFT)", "81_episode_regime_transitions",
     "../qa_episode_regime_cert_v1", True),
    (82, "QA BSD Local Euler Cert family",
     _validate_bsd_local_euler_cert_v1_family_if_present,
     "schema v1/v1.1 + validator (schema/recompute/reduction-type gates; optional delta_mod_p/is_good_reduction checks) + 3 fixtures (pass_good_p5, pass_good_p7_v1_1, fail_wrong_ap)", "82_bsd_local_euler_cert",
     "../qa_bsd_local_euler_cert_v1", True),
    (83, "QA BSD Local Euler Batch Cert family",
     _validate_bsd_local_euler_batch_cert_v1_family_if_present,
     "schema + validator (per-prime recompute + manifest hash binding) + 3 fixtures (pass_batch_p5_p7, pass_batch_p5_p11, fail_corrupt_record_p7_ap)", "83_bsd_local_euler_batch_cert",
     "../qa_bsd_local_euler_batch_cert_v1", True),
    (84, "QA BSD Partial L-series Proxy Cert family",
     _validate_bsd_partial_lseries_proxy_cert_v1_family_if_present,
     "schema + validator (exact non-reduced Π(#E(F_p)/p) proxy + manifest binding) + 3 fixtures (pass_proxy_p5_p7, pass_proxy_p5_p11, fail_wrong_proxy_denominator)", "84_bsd_partial_lseries_proxy_cert",
     "../qa_bsd_partial_lseries_proxy_cert_v1", True),
    (85, "QA BSD Rank Squeeze Cert family",
     _validate_bsd_rank_squeeze_cert_v1_family_if_present,
     "schema + validator (local recompute + manifest binding + exact proxy + monotone rank-trace consistency) + 5 fixtures (pass_closed_p5_p7, pass_open_p5_p11, fail_bad_trace_crossing, fail_wrong_proxy_denominator, fail_wrong_ap_p7)", "85_bsd_rank_squeeze_cert",
     "../qa_bsd_rank_squeeze_cert_v1", True),
    (86, "QA Generator-Failure Algebra Unification Cert family",
     _validate_generator_failure_unification_cert_v1_family_if_present,
     "schema + validator (5-gate: carrier cross-check, digest, T1 finite image, T2 SCC + T3 path propagation, T4 energy monotonicity) + 3 fixtures (valid_caps_tr_fear_love, invalid_tag_not_in_carrier, invalid_energy_drift) + cross-binding to [76] failure algebra ref + [80] energy cert ref",
     "86_generator_failure_unification_cert",
     "../qa_generator_failure_unification_cert_v1", True),
    (87, "QA Failure Compose Operator Cert family",
     _validate_failure_compose_operator_cert_v1_family_if_present,
     "schema + validator (formal compose(Fi,Fj,form) with closure/table completeness + per-form associativity checks) + 3 fixtures (pass_feedback_escalation, fail_closure_incomplete_table, fail_associativity_feedback_violation)",
     "87_failure_compose_operator_cert",
     "../qa_failure_compose_operator_cert_v1", True),
    (88, "QA Failure Algebra Structure Classification Cert family",
     _validate_failure_algebra_structure_classification_cert_v1_family_if_present,
     "schema + validator (form-indexed semigroup/monoid classification with identity/absorber/commutativity and optional monotonicity checks) + 5 fixtures (pass_classify_from_family87_tables, fail_identity_claim_wrong, fail_absorber_claim_wrong, fail_commutative_claim_wrong, fail_monotonicity_violation)",
     "88_failure_algebra_structure_classification_cert",
     "../qa_failure_algebra_structure_classification_cert_v1", True),
    (89, "QA QALM Curvature Cert family",
     _validate_qalm_curvature_cert_v1_family_if_present,
     "schema + validator (H_QA recompute and curvature-scaled update-rule pin) + 3 fixtures (pass_default_tuple, fail_h_qa_mismatch, fail_update_sign)",
     "89_qalm_curvature_cert",
     "../qa_qalm_curvature_cert_v1", True),
    (90, "QA Fairness Demographic Parity Cert family",
     _validate_fairness_demographic_parity_cert_v1_family_if_present,
     "schema + validator (demographic parity gap with constructive failure witness) + 2 fixtures (valid_min, invalid_gap)",
     "90_fairness_demographic_parity_cert",
     "../qa_fairness_demographic_parity_cert_v1", True),
    (91, "QA Fairness Equalized Odds Cert family",
     _validate_fairness_equalized_odds_cert_v1_family_if_present,
     "schema + validator (equalized odds TPR/FPR gap with constructive failure witness) + 2 fixtures (valid_min, invalid_gap)",
     "91_fairness_equalized_odds_cert",
     "../qa_fairness_equalized_odds_cert_v1", True),
    (92, "QA Safety Prompt Injection Refusal Cert family",
     _validate_safety_prompt_injection_refusal_cert_v1_family_if_present,
     "schema + validator (prompt injection refusal rate with judge contract hash and failure witness) + 2 fixtures (valid_min, invalid_rate)",
     "92_safety_prompt_injection_refusal_cert",
     "../qa_safety_prompt_injection_refusal_cert_v1", True),
    (93, "QA GNN Message-Passing Curvature Cert family",
     _validate_gnn_mp_curvature_cert_family,
     "3/3 fixtures + gate checks",
     "93_gnn_mp_curvature_cert",
     "../qa_gnn_message_passing_curvature_cert_v1", True),
    (94, "QA Attention Layer Curvature Cert family",
     _validate_attn_curvature_cert_family,
     "3/3 fixtures + gate checks",
     "94_attn_curvature_cert",
     "../qa_attn_curvature_cert_v1", True),
    (95, "QA QARM Curvature Cert family",
     _validate_qarm_curvature_cert_family,
     "3/3 fixtures + gate checks",
     "95_qarm_curvature_cert",
     "../qa_qarm_curvature_cert_v1", True),
    (96, "QA Symbolic Search Curvature Cert family",
     _validate_symbolic_search_curvature_cert_family,
     "QA symbolic search curvature cert (sym_gain, beam_width, search_depth, rule_count)",
     "96_symbolic_search_curvature_cert",
     "../qa_symbolic_search_curvature_cert_v1", True),
    (97, "QA Orbit Curvature Cert family",
     _validate_orbit_curvature_cert_family,
     "QA orbit curvature cert (orbit enumeration, kappa_min stability margin)",
     "97_orbit_curvature_cert",
     "../qa_orbit_curvature_cert_v1", True),
    (98, "QA GNN Spectral Gain Cert family",
     _validate_gnn_spectral_gain_cert_family,
     "QA GNN spectral gain cert (sigma_max derived from weight matrix, not free witness)",
     "98_gnn_spectral_gain_cert",
     "../qa_gnn_spectral_gain_cert_v1", True),
    (99, "QA Attention Spectral Gain Cert family",
     _validate_attn_spectral_gain_cert_family,
     "QA attention spectral gain cert (sigma_max(QK^T/sqrt(d_k)), natural Lipschitz constant)",
     "99_attn_spectral_gain_cert",
     "../qa_attn_spectral_gain_cert_v1", True),
    (100, "QA E8 Alignment Audit Cert family",
     _validate_e8_alignment_audit_cert_family,
     "E8 alignment audit (pre-registered rule, mod-9 full population, verdict: INCIDENTAL)",
     "100_e8_alignment_audit_cert",
     "../qa_e8_alignment_audit_cert_v1", True),
    (101, "QA Gradient Lipschitz Gain Cert family",
     _validate_gradient_lipschitz_gain_cert_family,
     "QA gradient Lipschitz gain cert (gain = min(||grad||_2, 2.0), derived from gradient vector)",
     "101_gradient_lipschitz_gain_cert",
     "../qa_gradient_lipschitz_gain_cert_v1", True),
    (102, "QA Lojasiewicz Orbit Descent Cert family",
     _validate_lojasiewicz_orbit_cert_family,
     "QA Lojasiewicz orbit cert (phi-contraction phi_{t+L} <= phi_t - (1-alpha)*C(O), C(O) orbit-computable)",
     "102_lojasiewicz_orbit_cert",
     "../qa_lojasiewicz_orbit_cert_v1", True),
    (103, "QA Lojasiewicz Orbit Descent Cert v2 (intrinsic) family",
     _validate_lojasiewicz_orbit_cert_v2_family,
     "QA Lojasiewicz orbit cert v2 (H-crit derived from phi_t>0 via B3; h_crit_witnessed field removed)",
     "103_lojasiewicz_orbit_cert_v2",
     "../qa_lojasiewicz_orbit_cert_v2", True),
    (104, "QA Feuerbach Parent Scale family",
     _validate_feuerbach_parent_scale_family,
     "Feuerbach parent-scale law: scale=4 for all non-root primitive triples; (3,4,5) unique boundary exception with QA closure scale 2G=10",
     "104_feuerbach_parent_scale",
     "../qa_feuerbach_parent_scale_v1", True),
    (105, "QA Cymatics Correspondence family",
     _validate_cymatics_family_if_present,
     "mode witness (Chladni+QA tuple echo) + Faraday reachability + control cert (lawful generator sequence → target pattern); 6 fixtures, 3 schemas, self-test ok",
     "105_cymatics",
     "qa_cymatics", True),
    (106, "QA Plan-Control Compiler family",
     _validate_plan_control_compiler_family,
     "generic certifiable compilation relation (search→plan→execution→witness); cymatics first instantiation; GENERATOR_SEQUENCE_MISMATCH, COMPILATION_HASH_MISMATCH, TARGET_INVARIANT_MISMATCH; 1 schema, 2 fixtures, self-test ok",
     "106_plan_control_compiler",
     "qa_plan_control_compiler", True),
    (107, "QA Core Spec Kernel family",
     _validate_core_spec_family,
     "base executable ontology (state_space, generators, invariants, reachability, failure_algebra, logging, gates [0..5]); kernel witness + 3 FAIL fixtures; V1-V5 checks; self-test ok",
     "107_qa_core_spec",
     "qa_core_spec", True),
    (108, "QA Area Quantization family",
     _validate_area_quantization_family,
     "first family_extension of QA_CORE_SPEC.v1; certifies discrete quadrea spectrum of Q(sqrt(5)) norm form b^2+be-e^2 mod m; mod-9 spectrum={0,1,2,4,5,7,8} forbidden={3,6} (3 inert in Z[phi]); IH1-IH4 inheritance checks + AQ1-AQ2 domain checks; 1 PASS + 1 FAIL fixture; self-test ok",
     "108_qa_area_quantization",
     "qa_area_quantization", True),
    (109, "QA Inheritance Compat family",
     _validate_inheritance_compat_family,
     "certifies inheritance edges as first-class objects in the QA spec graph; IC1-IC8 checks (parent/child recognition, inherits_from match, gate policy superset, failure algebra extension, logging contract, invariant ref resolution, scope transition validity); 13 PASS edges ([107]→[108], [107]→[111]–[120], [106]→[105], [106]→[110]) + 1 FAIL (gate policy deleted); self-test ok",
     "109_qa_inheritance_compat",
     "qa_inheritance_compat", True),
    (110, "QA Seismic Pattern Control family",
     _validate_seismic_control_family,
     "second domain_instance of QA_PLAN_CONTROL_COMPILER_CERT.v1 [106]; proves compiler is cross-domain (seismology vs cymatics); seismic wave states quiet/p_wave/s_wave/surface_wave/coda/disordered → QA orbits singularity/satellite/cosmos; S1-S6 checks; 1 PASS (quiet→p_wave→surface_wave, k=2) + 1 FAIL (illegal direct quiet→surface_wave); self-test ok",
     "110_qa_seismic_control",
     "qa_seismic_control", True),
    (111, "QA Inert Prime Area Quantization family",
     _validate_area_quantization_pk_family,
     "family_extension of QA_CORE_SPEC.v1; generalises [108] to mod p^k for all inert primes p; theorem: Im(f)={r: v_p(r)!=1}, forbidden={r: v_p(r)=1}; validator exhaustively recomputes spectrum (O(p^2k) loop) and verifies theorem prediction; IH1-IH3 inheritance + PK1-PK4 domain checks; 3 PASS (p=3/k=2 anchor, p=3/k=3, p=7/k=2 second inert prime) + 1 FAIL (wrong forbidden set); self-test ok",
     "111_qa_area_quantization_pk",
     "qa_area_quantization_pk", True),
    (112, "QA Obstruction-Compiler Bridge family",
     _validate_obstruction_compiler_bridge_family,
     "family_extension of QA_CORE_SPEC.v1; bridges arithmetic obstruction ([111]) to control reachability ([106]); theorem: v_p(r)=1 => no valid plan/control PASS cert may claim r as reachable target; validator recomputes v_p and checks claimed_reachable consistency; IH1-IH3 inheritance + B1-B7 bridge checks; 2 PASS (forbidden class 3 blocked, valid class 4 unblocked) + 1 FAIL (forbidden class 6 claimed reachable); self-test ok",
     "112_qa_obstruction_compiler_bridge",
     "qa_obstruction_compiler_bridge", True),
    (113, "QA Obstruction-Aware Planner family",
     _validate_obstruction_aware_planner_family,
     "family_extension of QA_CORE_SPEC.v1; certifies planner correctly applies [112] bridge before search: forbidden targets (v_p=1) must be pruned with nodes_expanded=0; valid targets must not be pruned; correctness not just impossibility — obstruction actively governs computation; IH1-IH3 + BR1-BR5 + PA1-PA3 checks; 2 PASS (pruned class 3, search class 4 plan found) + 1 FAIL (forbidden class 6 but 47 nodes expanded: OBSTRUCTION_NOT_APPLIED); self-test ok",
     "113_qa_obstruction_aware_planner",
     "qa_obstruction_aware_planner", True),
    (114, "QA Obstruction Efficiency family",
     _validate_obstruction_efficiency_family,
     "family_extension of QA_CORE_SPEC.v1; quantifies search-cost savings from obstruction-aware pruning ([113]); for forbidden targets (v_p=1): naive planner expands N>0 nodes, aware planner expands 0 (saved_nodes=N, pruning_ratio=1.0); for valid targets: no false pruning (pruning_ratio=0, false_pruning=false); validator recomputes saved_nodes and pruning_ratio from raw traces; IH1-IH3 + EF1-EF9 checks; 2 PASS (forbidden class 6 100% saved, valid class 4 zero savings) + 1 FAIL (valid class 4 falsely pruned: FALSE_PRUNING_EFFICIENCY + AWARE_TRACE_MISMATCH); self-test ok",
     "114_qa_obstruction_efficiency",
     "qa_obstruction_efficiency", True),
    (115, "QA Obstruction Stack family",
     _validate_obstruction_stack_family,
     "synthesis spine compressing the full [111]–[114] obstruction chain into one theorem-bearing cert; recomputes all four layers independently (arithmetic v_p, control reachability, planner pruning, efficiency savings); OBSTRUCTION_PRESENT → control_verdict=UNREACHABLE + pruned(0 nodes) + pruning_ratio=1.0; validator checks cross-layer consistency and stack_conclusion.full_chain_holds; IH1-IH3 + OS1-OS12; 1 PASS (canonical forbidden r=6, full chain holds) + 1 FAIL (r=6 obstruction declared but planner expanded 12 nodes, ratio=0.74: PRUNING_CONCLUSION_MISMATCH + EFFICIENCY_CONCLUSION_MISMATCH + STACK_INCONSISTENCY); self-test ok",
     "115_qa_obstruction_stack",
     "qa_obstruction_stack", True),
    (116, "QA Obstruction Stack Report family",
     _validate_obstruction_stack_report_family,
     "reader-facing report artifact packaging [115] for external audiences; contains theorem statement, one-line layer summaries for [111]–[115], recomputed summary table (v_p, forbidden, reachable, pruned, baseline/aware/saved nodes, pruning_ratio), canonical PASS+FAIL witnesses, source refs; validator recomputes entire table from arithmetic params and checks faithfulness; IH1-IH3 + RP1-RP9; 1 PASS (canonical r=6, two-row table verified) + 1 FAIL (r=6 row claims pruned=false, aware_nodes=12, ratio=0.74: SUMMARY_TABLE_MISMATCH); self-test ok",
     "116_qa_obstruction_stack_report",
     "qa_obstruction_stack_report", True),
    (117, "QA Control Stack family",
     _validate_control_stack_family,
     "synthesis cert for the control/compiler spine; asserts QA_PLAN_CONTROL_COMPILER_CERT.v1 [106] is domain-generic: orbit trajectory singularity→satellite→cosmos and path_length_k=2 preserved across cymatics [105] (flat→stripes→hexagons) and seismology [110] (quiet→p_wave→surface_wave); validator recomputes cross-domain trace consistency (CS1-CS11); 1 PASS (both domains share orbit trajectory + k=2) + 1 FAIL (seismology declares final=satellite not cosmos: ORBIT_TRAJECTORY_MISMATCH + CROSS_DOMAIN_CLAIM_INCONSISTENT + STACK_INCONSISTENCY); self-test ok",
     "117_qa_control_stack",
     "qa_control_stack", True),
    (118, "QA Control Stack Report family",
     _validate_control_stack_report_family,
     "reader-facing report packaging [117] for external audiences; contains theorem statement, one-line summaries for [106]/[105]/[110], comparison table (domain, initial/intermediate/target state, orbit_path, path_length_k, move_sequence), canonical PASS+FAIL witnesses; validator recomputes cross-row consistency from comparison table; IH1-IH3 + CR1-CR7; 1 PASS (two-row table consistent) + 1 FAIL (seismology row orbit_path ends in satellite, path_length_k=3: COMPARISON_TABLE_MISMATCH); self-test ok",
     "118_qa_control_stack_report",
     "qa_control_stack_report", True),
    (119, "QA Dual Spine Unification Report family",
     _validate_dual_spine_unification_report_family,
     "top-level validated overview placing [116] (obstruction spine) and [118] (control spine) side by side; contains theorem paragraphs for each spine, two-row comparison table (entry_point, kernel, main_family_chain, canonical_theorem, pass/fail witness summaries, what_is_recomputed), synthesis statement unifying both spines; validator checks spine refs (DU1-DU2), theorem presence (DU3-DU4), comparison table structure and content (DU5-DU9), synthesis completeness (DU10-DU11), witness values (DU12-DU14); 1 PASS (canonical two-spine table) + 1 FAIL (obstruction_spine_ref points to cert not report: OBSTRUCTION_SPINE_REF_MISMATCH); self-test ok",
     "119_qa_dual_spine_unification_report",
     "qa_dual_spine_unification_report", True),
    (120, "QA Public Overview Doc family",
     _validate_public_overview_doc_family,
     "presentation-grade export derived from [119]; designed to be handed to a reviewer as the first artifact they read; contains executive summary, two-spine diagram (chain + theorem for each spine), canonical obstruction example (r=6, p=3, k=2, pruning_ratio=1.0), canonical cross-domain control example (cymatics + seismology, shared orbit singularity→satellite→cosmos, k=2), why-it-matters section, spine entry-point pointers; validator checks faithfulness to [119] (PO1-PO9): overview_ref, executive summary, spine diagram completeness, obstruction example substantive content (v_p/ratio), control example domain names (cymatics+seismology), why_it_matters, both spine entry points present, witness values; 1 PASS (canonical overview) + 1 FAIL (spine_entry_points missing control spine: SPINE_ENTRY_POINTS_INCOMPLETE); self-test ok",
     "120_qa_public_overview_doc",
     "qa_public_overview_doc", True),
    (121, "QA Engineering Core Cert family",
     _validate_engineering_core_cert_family,
     "family_extension of QA_CORE_SPEC.v1 [107]; certifies that any classical engineering system (state-space model + stability conditions + controllability claim) maps validly to a QA specification; validator recomputes IH1-IH3 (kernel inheritance), EC1 (state encoding 1<=b,e<=N), EC2 (all transitions have generator names), EC3 (all failure modes map to QA fail types), EC4 (target orbit family valid), EC5 (declared orbit_family matches recomputed f(b,e) 3-adic valuation), EC6 (lyapunov_function mentions QA invariant), EC7 (orbit_contraction_factor < 1.0), EC8 (equilibrium maps to singularity), EC9 (reachability_witness present for full_rank controllability), EC10 (minimality_witness present with optimization_claim), EC11 (obstruction_check.obstructed matches recomputed v_p(target_r) for inert primes — catches arithmetic obstructions invisible to Kalman rank analysis); fail types: STATE_ENCODING_INVALID, TRANSITION_NOT_GENERATOR, FAILURE_TAXONOMY_INCOMPLETE, TARGET_NOT_ORBIT_FAMILY, ORBIT_FAMILY_CLASSIFICATION_FAILURE, LYAPUNOV_QA_MISMATCH, CONTROLLABILITY_QA_MISMATCH, ARITHMETIC_OBSTRUCTION_IGNORED; 1 PASS (spring-mass oscillator: still/transient/steady_oscillation → singularity/satellite/cosmos, k=2, target_r=2 v3=0 not obstructed) + 1 FAIL (ARITHMETIC_OBSTRUCTION_IGNORED: target_r=3 v3(3)=1 inert, cert declares obstructed=false) + 1 FAIL (STATE_ENCODING_INVALID: b=0 outside domain {1,...,N}); self-test ok",
     "121_qa_engineering_core_cert",
     "qa_engineering_core_cert", True),
    (122, "QA Empirical Observation Cert family",
     _validate_empirical_observation_cert_family,
     "bridge: Open Brain / experiment results → cert ecosystem; V1-V5 checks (source, parent ref, verdict, CONTRADICTS→fail_ledger, evidence); 2 PASS (audio orbit CONSISTENT, finance script 26 CONTRADICTS) + 1 FAIL (EMPTY_EVIDENCE); self-test ok",
     "122_qa_empirical_observation_cert",
     "qa_empirical_observation_cert", True),
    (123, "QA Agent Competency Cert family",
     _validate_agent_competency_cert_family,
     "Levin morphogenetic agent architecture; V1-V10 checks (schema, horizon, convergence, orbit, cell_type, failure_modes, dediff_cond, CELL_ORBIT_MISMATCH, goal_length, composition_rules); 2 PASS (merge_sort cosmos/differentiated, gradient_descent mixed/progenitor) + 1 FAIL (CELL_ORBIT_MISMATCH); self-test ok",
     "123_qa_agent_competency_cert",
     "qa_agent_competency_cert", True),
    (124, "QA Security Competency Cert family",
     _validate_security_competency_cert_family,
     "Immune system architecture; SC1-SC11 (schema, security_role, immune_function, pq_readiness, SC5 quantum resilience invariant, FIPS designation, failure_modes, composition_rules, CELL_ORBIT_MISMATCH, goal_length, result_match); 2 PASS (ml_kem membrane/fips_final, ed25519 identity/classical_only+migration) + 1 FAIL (SC5_PQ_MIGRATION_REQUIRED); self-test ok",
     "124_qa_security_competency_cert",
     "qa_security_competency_cert", True),
    (129, "QA Projection Obstruction Cert family",
     _validate_projection_obstruction_cert_family,
     "separates native symbolic closure, discrete representation-basis mismatch, and physical device realization; representation debt is not itself device failure; checks IH1-IH3 + PO1-PO9; 1 PASS (Arto ternary lawful natively + representation debt + physical INCONCLUSIVE) + 2 FAIL (physical conflation, unresolved invariant ref); self-test ok",
     "129_qa_projection_obstruction_cert",
     "qa_projection_obstruction_cert", True),
    (128, "QA Spread Period Cert family",
     _validate_spread_period_cert_family,
     "Pisano period = QA cosmos orbit: cosmos period for modulus m = π(m) = Fibonacci period mod m = ord(F) in GL₂(Z/mZ); π(9)=24, π(7)=16, π(3)=8; checks SP1-SP5 (schema, Pisano period, F^P≡I, minimality, orbit_type); 2 PASS (m=9 period=24, m=7 period=16) + 1 FAIL (PISANO_PERIOD_MISMATCH+MATRIX_PERIOD_WRONG: claimed period=12 for m=9); self-test ok",
     "128_qa_spread_period",
     "qa_spread_period_cert_v1", True),
    (127, "QA UHG Null Cert family",
     _validate_uhg_null_cert_family,
     "UHG null points: every QA triple (F,C,G)=(d²-e²,2de,d²+e²) satisfies F²+C²-G²=0 (null condition in UHG); Gaussian integer interpretation Z=d+ei, Z²=(d²-e²)+2dei; checks UN1-UN7 (schema, green/red/blue quadrance, null condition, Gaussian decomp, null_quadrance); 2 PASS (3-4-5, 5-12-13) + 1 FAIL (BLUE_QUADRANCE_MISMATCH+NULL_CONDITION_VIOLATED+GAUSSIAN_DECOMP_MISMATCH: G=6 instead of 5); self-test ok",
     "127_qa_uhg_null",
     "qa_uhg_null_cert_v1", True),
    (126, "QA Red Group Cert family",
     _validate_red_group_cert_family,
     "Wildberger red isometry group: QA T-operator = Fibonacci shift F=[[0,1],[1,1]] = red-rotation by φ in Z[√5]/mZ[√5]; det(F)=-1=N_red(φ), trace(F)=1; orbit period = ord(F) in GL₂(Z/mZ); checks RG1-RG7 (schema, T_matrix, det, trace, F^P≡I, minimality, orbit_type); 2 PASS (m=9 period=24, m=3 period=8) + 1 FAIL (ORBIT_PERIOD_WRONG: claimed period=12, F^12=-I≢I); self-test ok",
     "126_qa_red_group",
     "qa_red_group_cert_v1", True),
    (125, "QA Chromogeometry Cert family",
     _validate_chromogeometry_cert_family,
     "Wildberger chromogeometric quadrances: C=Q_green(d,e)=2de, F=Q_red(d,e)=d²-e²=ab, G=Q_blue(d,e)=d²+e²; C²+F²=G² (Wildberger Thm 6); I=|C-F| conic discriminant; checks CG1-CG7 (schema, green/red/blue quadrance, Pythagoras, semi-latus, conic type); 2 PASS (3-4-5 hyperbola b=1e=1, 20-21-29 ellipse b=3e=2) + 1 FAIL (GREEN_QUADRANCE_MISMATCH+PYTHAGORAS_VIOLATED); self-test ok",
     "125_qa_chromogeometry",
     "qa_chromogeometry_cert_v1", True),
    (132, "QA HAT Cert family",
     _validate_hat_cert_family,
     "H. Lee Price half-angle tangents bridge to QA: HAT₁=e/d=C/(G+F) [primary], HAT₂=(d-e)/(d+e)=F/(G+C) [secondary]; spread s=E/G=HAT₁²/(1+HAT₁²) [Wildberger]; Fibonacci box [[e,d-e],[d,d+e]]; Price Fibonacci box cols = QA generation matrix entries; proportionality: HAT fractions carry QA element meaning; checks HAT_1-8+HAT_W/F; 2 PASS; self-test ok",
     "132_qa_hat",
     "qa_hat_cert_v1", True),
    (135, "QA Pythagorean Tree Cert family",
     _validate_pythagorean_tree_cert_family,
     "three Barning-Hall/Berggren generator moves in QA direction space: M_A=(2d-e,d) k=2, M_B=(2d+e,d) k=3, M_C=(d+2e,e) k≥4; each preserves gcd=1 + opposite parity + F²+C²=G²; k-identification theorem links each move to Egyptian fraction first step k; root (2,1) has no valid parent; Barning 1963/Hall 1970/Price 2008 Fibonacci-boxes/Iverson Koenig = same tree; inverse of cert [134]; checks PT_1-4+PT_A/B/C+PT_ROOT/W/F; 2 PASS; self-test ok",
     "135_qa_pythagorean_tree",
     "qa_pythagorean_tree_cert_v1", True),
    (136, "QA Cyclic Quad Cert family",
     _validate_cyclic_quad_cert_family,
     "Ptolemy theorem via three integer identities for QA direction pairs: BF G₁G₂=D²+E² (Brahmagupta-Fibonacci); PP F₃=|F₁F₂-C₁C₂|, C₃=F₁C₂+F₂C₁, F₃²+C₃²=(G₁G₂)²; PC F₄=F₁F₂+C₁C₂, C₄=|F₁C₂-F₂C₁|, F₄²+C₄²=(G₁G₂)²; both triples = two diagonals of Ptolemy cyclic quadrilateral on circle G₁G₂; proof: (F₁F₂-C₁C₂)²+(F₁C₂+F₂C₁)²=(F₁²+C₁²)(F₂²+C₂²); Ptolemy ~150 CE→Brahmagupta 628 CE→Gaussian Z[i]; connects to [127] UHG null; checks CQ_1/2/3/BF/PP/PC/G3/W/F; 2 PASS; self-test ok",
     "136_qa_cyclic_quad_cert",
     "qa_cyclic_quad_cert_v1", True),
    (155, "QA Bearden Phase Conjugate Cert family",
     _validate_bearden_phase_conjugate_cert_family,
     "Bearden 'stress is a pumper' = QCI opposite-sign: global QCI+ (pump/coupling tightens) + local QCI- (conjugate/trajectories scatter); QCI_gap partial r=-0.17 to -0.42; 100% robust; permutation-validated; SVP lineage Keely→Pond→Bearden; checks BPC_1+MODEL/MAP/SIGN/EMP/SVP/W/F; 1 PASS + 1 FAIL; self-test ok",
     "155_qa_bearden_phase_conjugate_cert",
     "qa_bearden_phase_conjugate_cert_v1", True),
    (156, "QA WGS84 Ellipse Cert family",
     _validate_wgs84_ellipse_cert_family,
     "WGS84 reference ellipsoid = QA quantum ellipse; shape QN (101,9,110,119) ecc=9/110 matches WGS84 0.08182 to 0.001%; axis ratio sqrt(12019)/110 matches 7 sig figs; orbit QN (59,1,60,61) ecc=1/60 matches 0.01671 to 0.25%; triple (1980,12019,12181) C²+F²=G²; Tier 1 exact reformulation; checks WGS_1+QN/TRIPLE/ECC/AXIS/ORBIT/W/F; 1 PASS + 1 FAIL; self-test ok",
     "156_qa_wgs84_ellipse_cert",
     "qa_wgs84_ellipse_cert_v1", True),
    (219, "QA Fibonacci Resonance Cert family",
     _validate_fibonacci_resonance_cert_family,
     "MMRs preferentially select Fibonacci ratios; 60 resonances across 8+ planetary systems (solar+exoplanet); order-1: 33/43 (77%) Fib vs 22% expected p<10⁻⁶; unique 8/14 (57%) vs 31% p=0.040; Fisher combined p<10⁻⁶; QA T-operator=Fibonacci shift makes Fib ratios deeper attractors; three-body problem selection principle; Tier 2→3; corrective renumber from duplicate [163], which is reserved by Dead Reckoning; checks FR_1+CAT/CLASS/STAT/ORDER/CROSS/HONEST/W/F; 1 PASS + 1 FAIL; self-test ok",
     "219_qa_fibonacci_resonance_cert",
     "qa_fibonacci_resonance_cert_v1", True),
    (161, "QA ECEF Rational Cert family",
     _validate_ecef_rational_cert_family,
     "Geodetic→ECEF via spreads/crosses: X²=(N+h)²·c_φ·c_λ, Y²=(N+h)²·c_φ·s_λ, Z²=(N(1-e²)+h)²·s_φ; 6 cities all hemispheres + poles; X²+Y²=(N+h)²·c_φ identity; Tier 1 exact reformulation; checks ECEF_1+SPREAD/N/XYZ/SUM/W/F; 1 PASS + 1 FAIL; self-test ok",
     "161_qa_ecef_rational_cert",
     "qa_ecef_rational_cert_v1", True),
    (160, "QA Bragg RT Cert family",
     _validate_bragg_rt_cert_family,
     "Bragg's law as rational trigonometry: n²Q_λ=4Q_d·s (square both sides of nλ=2d·sinθ); Miller Q(h,k,l)=h²+k²+l² for cubic; crystal spreads: cubic all=1, hexagonal γ=3/4; NaCl Cu Kα 4 reflections exact integer arithmetic; Tier 1 algebraic identity; checks BRT_1+BRAGG/MILLER/SPREAD/PYTH/W/F; 1 PASS + 1 FAIL; self-test ok",
     "160_qa_bragg_rt_cert",
     "qa_bragg_rt_cert_v1", True),
    (157, "QA PIM Kernel Cert family",
     _validate_pim_kernel_cert_family,
     "PIM kernel correctness: CRT coprime+non-coprime, RESIDUE_SELECT, TORUS_SHIFT, ROLLING_SUM_PHASE; A1 coordinate-layer note; checks PIM_1+CRT/KERNEL/A1/W/F; 1 PASS + 1 FAIL; self-test ok",
     "157_qa_pim_kernel_cert",
     "qa_pim_kernel_cert_v1", True),
    (158, "QA Graph Community Cert family",
     _validate_graph_community_cert_family,
     "QA feature map dimensions qa21=21 qa27=27 qa83=83; chromogeometry C*C+F*F=G*G; benchmark graphs football/karate; checks GC_1+DIM/CHROMO/BENCH/W/F; 1 PASS + 1 FAIL; self-test ok",
     "158_qa_graph_community_cert",
     "qa_graph_community_cert_v1", True),
    (159, "QA Observer Core Cert family",
     _validate_observer_core_cert_family,
     "qa_mod() A1 compliance (output in {1,...,m} never 0) + compute_qci() determinism; 6 domain witnesses; T2 no float->int feedback; checks OC_1+A1/QCI/T2/W/F; 1 PASS + 1 FAIL; self-test ok",
     "159_qa_observer_core_cert",
     "qa_observer_core_cert_v1", True),
    (154, "QA T-Operator Coherence Cert family",
     _validate_t_operator_coherence_cert_family,
     "QCI = rolling T-operator prediction accuracy; finance partial r=-0.22 beyond RV (Tier A hardened); 84% robustness grid; cross-domain: EEG dR²=+0.21, audio r=+0.75; checks TC_1+OBS/QCI/OOS/PARTIAL/ROBUST/W/F; 2 PASS; self-test ok",
     "154_qa_t_operator_coherence_cert",
     "qa_t_operator_coherence_cert_v1", True),
    (153, "QA Keely Triune Cert family",
     _validate_keely_triune_cert_family,
     "Keely triune (Enharmonic/Dominant/Harmonic) → QA orbits (Satellite/Singularity/Cosmos); {0,3,6}=singularity=Tesla 3-6-9; LCM(1,8,24)=24; Brinton Laws of Being; checks KT_1+MAP/PART/PERIOD/369/LCM/W; 2 PASS; self-test ok",
     "153_qa_keely_triune_cert",
     "qa_keely_triune_cert_v1", True),
    (152, "QA Equilateral Triangle Cert family",
     _validate_equilateral_triangle_cert_family,
     "W=d(e+a), Y=A-D=C+E (dual definition), Z=E+K; Eisenstein norms F²-FW+W²=Z² and Y²-YW+W²=Z²; sum F+Y=W; checks ET_1+DEF/DUAL/EIS/SUM/W/F; 2 PASS; self-test ok",
     "152_qa_equilateral_triangle_cert",
     "qa_equilateral_triangle_cert_v1", True),
    (151, "QA Par Number Cert family",
     _validate_par_number_cert_family,
     "Iverson Double Parity: 2/3/4/5-par mod-4 classification; male²=5-par; C=4-par, G=5-par; multiplication table; Fib_hits observations; checks PN_1+CLASS/SQ/QA/FIB/MULT/W/F; 2 PASS; self-test ok",
     "151_qa_par_number_cert",
     "qa_par_number_cert_v1", True),
    (150, "QA Septenary Cert family",
     _validate_septenary_cert_family,
     "{1,2,4,5,7,8}=(Z/9Z)* mod-9 unit group; doubling cycle period 6=phi(9); complement {0,3,6}=singularity; diagonal pairs sum to 9; parity cross-over; checks SP_1+GROUP/CYCLE/COMP/DIAG/PAR/W/F; 2 PASS; self-test ok",
     "150_qa_septenary_cert",
     "qa_septenary_cert_v1", True),
    (149, "QA Law of Harmonics Cert family",
     _validate_law_of_harmonics_cert_family,
     "Iverson's formal law: two QN products sharing all but one prime factor each are harmonic; ratio=min(id)/max(id); all products÷6; Fibonacci chain adjacency pattern; checks LH_1+ALIQ/IDEN/RATIO/DIV6/W/F; 2 PASS; self-test ok",
     "149_qa_law_of_harmonics_cert",
     "qa_law_of_harmonics_cert_v1", True),
    (148, "QA Sixteen Identities Cert family",
     _validate_sixteen_identities_cert_family,
     "16 named quantities (A-L,X,W,Y,Z) of prime Pythagorean direction + 9 algebraic relations: G+C=A, G-C=B, F²+C²=G², H²+I²=2G², L=CF/12 integer, C=4-par, G=5-par; checks SI_1/2+IDEN/REL/PAR/L/W/F; 2 PASS; self-test ok",
     "148_qa_sixteen_identities_cert",
     "qa_sixteen_identities_cert_v1", True),
    (147, "QA Synchronous Harmonics Cert family",
     _validate_synchronous_harmonics_cert_family,
     "coprime periods sync at product; non-coprime at LCM<product; 3-par LOW at 1/4, 5-par HIGH at 1/4; same-par SUPPORT, cross-par OPPOSE; QN products divisible by 6; checks SH_1+SYNC/PAR/PROD6/W/F; 2 PASS; self-test ok",
     "147_qa_synchronous_harmonics_cert",
     "qa_synchronous_harmonics_cert_v1", True),
    (146, "QA Path Scale Cert family",
     _validate_path_scale_cert_family,
     "G=d^2+e^2 growth profiles along Pythagorean-tree paths; UNIFORM_B exponential (ratio->3+2sqrt(2)=5.828), UNIFORM_A/C polynomial (ratio->1); all forward paths G monotone increasing; 8-step Pell convergence witness; checks SC_1/2+GROWTH/RATIO/CONV_B/W/F; 2 PASS; self-test ok",
     "146_qa_path_scale_cert",
     "qa_path_scale_cert_v1", True),
    (145, "QA Path Shape Cert family",
     _validate_path_shape_cert_family,
     "four shape classes: UNIFORM_A (consecutive integers), UNIFORM_B (Pell chain, norm alternates), UNIFORM_C (constant e, arithmetic d), MIXED (2+ generators); primitivity preserved; checks PS_1/2+CLASS/INV_B/INV_C/W/F; 2 PASS; self-test ok",
     "145_qa_path_shape_cert",
     "qa_path_shape_cert_v1", True),
    (144, "QA Male/Female Octave Cert family",
     _validate_male_female_octave_cert_family,
     "Male→Female transform (double e, swap b↔e) gives female=(2e,b,a,2d); female_product=4×male_product; 4×=2 octaves; fundamental (1,1,2,3)→(2,1,3,4): 6→24=4×6; chains indefinitely (+2 octaves per step); checks MF_1-2+TRANS/PROD/OCT/W/F; 2 PASS; self-test ok",
     "144_qa_male_female_octave_cert",
     "qa_male_female_octave_cert_v1", True),
    (143, "QA Cube Sum Cert family",
     _validate_cube_sum_cert_family,
     "F³+C³+G³=216=6³ for fundamental (F,C,G)=(3,4,5); k=4 unique in [1,10000] for (k-1)³+k³+(k+1)³ perfect cube; 216=9×24; 6=b×e×d×a; checks CS_1-2+IDEN/DUAL/MOD/QN/UNIQ/W/F; 2 PASS; self-test ok",
     "143_qa_cube_sum_cert",
     "qa_cube_sum_cert_v1", True),
    (142, "QA Klein 4 Harmonics Cert family",
     _validate_klein4_harmonics_cert_family,
     "sign-changes of (F,C,G) form K4=Z2×Z2 preserving F²+C²=G²; I₁ swaps H↔I; I₂ maps (H,I)→(-I,-H); I₃ negates; every element self-inverse; fundamental (2,1) orbit {(7,1),(1,7),(-1,-7),(-7,-1)}; checks K4_1-3+ACT/HARM/W/F; 2 PASS; self-test ok",
     "142_qa_klein4_harmonics_cert",
     "qa_klein4_harmonics_cert_v1", True),
    (141, "QA Pell Norm Cert family",
     _validate_pell_norm_cert_family,
     "I=C-F=-(x²-2y²) where x=d-e, y=e: discriminant = negated Pell norm P(x,y)=x²-2y²; Pell boundary P=±1→|I|=1 (minimum nonzero); M_B(d,e)=(2d+e,d) = Pell-sign-flip in (x,y) space; M_B chain from (2,1) generates full Pell solution sequence alternating H/E; checks PN_1-3+IDEN/MB/W/F; 2 PASS; self-test ok",
     "141_qa_pell_norm_cert",
     "qa_pell_norm_cert_v1", True),
    (140, "QA Conic Discriminant Cert family",
     _validate_conic_discriminant_cert_family,
     "I=C-F=Qg-Qr as QA conic discriminant: I>0→hyperbola (d/e<1+√2), I=0→parabola (impossible: d/e=silver ratio=1+√2, irrational; disc(x²-2x-1)=8 non-square), I<0→ellipse (d/e>1+√2); silver-ratio CF [2;2,2,2,...] convergents 2/1,5/2,12/5,29/12 alternate H/E with |I|=1; Plimpton Row 1 (12,5) I=1 barely hyperbolic (d/e=2.4 vs 2.414); chromogeometry: I=Qg-Qr=green minus red; checks CD_1-4+PARA/W/F; 2 PASS; self-test ok",
     "140_qa_conic_discriminant_cert",
     "qa_conic_discriminant_cert_v1", True),
    (139, "QA 48/64 Cert family",
     _validate_48_64_cert_family,
     "structural constants 48 and 64: ALGEBRAIC 48L=H²-I²=4CF for all QA directions (proof: (C+F)²-(C-F)²=4CF=48L; min at (2,1) L=1→48); ORBIT 48=2×cosmos_period=2×24, 64=satellite_period²=8²; POLYNOMIAL equilateral (4,4,4): PR+RT+PT=48, PRT=64, polynomial (x-4)³=x³-12x²+48x-64, unique symmetric positive integer solution; 48/64=3/4=equilateral spread; 5040=7! at (7,2) L=105; connects to [137] Koenig (H²-I²=4CF corollary), [128] spread period (cosmos/satellite), [133] Eisenstein (equilateral); checks C4864_1-3+ALG/POLY/ORB/W/F; 2 PASS; self-test ok",
     "139_qa_48_64_cert",
     "qa_48_64_cert_v1", True),
    (138, "QA Plimpton 322 Cert family",
     _validate_plimpton322_cert_family,
     "Babylonian tablet Plimpton 322 (~1800 BCE) encodes QA chromogeometric triples: each row = direction (d,e) with d,e regular (5-smooth=only factors 2,3,5); F=d²-e² (short side β=Qr), C=2de (Qg), G=d²+e² (diagonal δ=Qb); regularity → C regular → G/C terminates in base-60 (exact sexagesimal); F²+C²=G²; SPVN no-zero=QA A1; counterexample (7,3): d=7 irregular → absent from tablet; Mansfield & Wildberger 2017 Historia Mathematica 44:395-419; Wildberger Chromo Thm 6: G²=F²+C²; checks P322_1-4+REG/BASE60/NOZERO/W/F; 2 PASS; self-test ok",
     "138_qa_plimpton322_cert",
     "qa_plimpton322_cert_v1", True),
    (137, "QA Koenig Twisted Squares Cert family",
     _validate_koenig_twisted_squares_cert_family,
     "H²-G²=G²-I²=2CF=24L for all QA directions: H=C+F (outer Koenig square), I=C-F (inner; sign=conic), L=CF/12 integer; (I²,2CF,G²,H²) arithmetic progression step 2CF; proof: H²-G²=(C+F)²-(C²+F²)=2CF using C²+F²=G²; divisibility: 8|C=2de (one of d,e even), 3|F=(d-e)(d+e); twisted-squares: outer²-inner²=4×area; Iverson QA Law 15 / Mathologer 2024 / Will Dale 2026-03-30 quadruple corollary; connects to [130] origin of 24; checks KTS_1-9+KTS_W/F; 2 PASS; self-test ok",
     "137_qa_koenig_twisted_squares_cert",
     "qa_koenig_twisted_squares_cert_v1", True),
    (134, "QA Egyptian Fraction Cert family",
     _validate_egyptian_fraction_cert_family,
     "greedy Egyptian fraction expansion of HAT₁=e/d: e/d=1/k₁+...+1/kₙ, kᵢ=⌈dᵢ/eᵢ⌉; strictly increasing denominators; all intermediate pairs coprime; terminates at eₙ=1 (unit-fraction direction); Koenig descent path = Egyptian fraction steps; Rhind Papyrus ~1600 BCE = Ben Iverson Koenig = H. Lee Price Fibonacci-box navigation; checks EF_1-8+EF_W/F; 2 PASS; self-test ok",
     "134_qa_egyptian_fraction",
     "qa_egyptian_fraction_cert_v1", True),
    (133, "QA Eisenstein Cert family",
     _validate_eisenstein_cert_family,
     "universal Eisenstein-norm identities from QA elements: F²-FW+W²=Z² and Y²-YW+W²=Z² for ALL tuples (b,e,d,a); F=ab, W=d(e+a), Z=e²+ad, Y=a²-d²=e(2b+3e); proof via u=b²+3be: (F+W)²-3FW=(u+3e²)²=Z²; W (equilateral side) and Z (Eisenstein companion) per QA Law 15; (1,1,2,3) gives (3,8,7) and (5,8,7), both Eisenstein triples; checks EIS_1-7+EIS_W/U; 2 PASS; self-test ok",
     "133_qa_eisenstein",
     "qa_eisenstein_cert_v1", True),
    (130, "QA Origin of 24 Cert family",
     _validate_origin_of_24_cert_family,
     "dual derivation of mod-24: H²-G²=G²-I²=2CF for any direction (d,e) [C=2de=green quadrance, F=d²-e²=red quadrance]; C²+F²=G² (Pythagorean) → (C+F)²-G²=2CF; always ÷24 for primitive Pythagorean directions; minimum=24 at fundamental (d,e)=(2,1) for 3-4-5; 7²-5²=24 (Crystal route); checks O24_1-O24_9 + O24_G/W/F/D; 2 PASS (anchor 3-4-5, general theorem 6 witnesses d≤5); self-test ok",
     "130_qa_origin_of_24",
     "qa_origin_of_24_cert_v1", True),
    (131, "QA Prime Bounded Certificate Scaling Cert family",
     _validate_prime_bounded_certificate_scaling_cert_family,
     "empirical scaling cert for bounded factor-certificate witness caps on tested intervals [2,N]; checks schema, canonical hash, artifact parity, row recomputation, row-level honesty, and overall PASS/FAIL honesty; 1 PASS (100,250,500,1000 exact-match cert) + 1 FAIL (mock 500 mismatch cert), both validator-valid; self-test ok",
     "131_qa_prime_bounded_certificate_scaling_cert",
     "../qa_prime_bounded_certificate_scaling_cert_v1", True),
    (162, "QA Human Needs SDT Cert family",
     _validate_human_needs_sdt_cert_family,
     "SDT 3 basic needs (Autonomy/Competence/Relatedness) = QA 3 paired types: (b,e) generators, (d,DeltaT) state+derivative, (a,SigmaT) reach+integral; canonical mapping: certainty=b, variety=e, significance=d, connection=a, growth=DeltaT, contribution=SigmaT; 5/5 structural predictions confirmed (SDT n=48550); Theorem NT compliant (observer projection); checks HN_1+MAP/SDT/TYPE/PRED/NT/SRC/W/F/DERIV/DELTA/SIGMA/FT; 2 PASS; self-test ok",
     "162_qa_human_needs_sdt_cert",
     "qa_human_needs_sdt_cert_v1", True),
    (163, "QA Dead Reckoning Cert family",
     _validate_dead_reckoning_cert_family,
     "T-operator exact DR on mod-m lattice; T^k·(b₀,e₀) via augmented 3x3 matrix exponentiation; zero computational drift vs classical sin/cos accumulation; chromogeometric triple C²+F²=G² per direction: G=position/F=hyperbolic/C=cross-track; compass rose mod-24 = orbit partition; Theorem NT; Tier 2; checks DR_1+TOP/EXACT/DRIFT/CHROMO/COMPASS/W/F; 1 PASS + 1 FAIL; self-test ok",
     "163_qa_dead_reckoning_cert",
     "qa_dead_reckoning_cert_v1", True),
    (164, "QA Gnomonic RT Cert family",
     _validate_gnomonic_rt_cert_family,
     "Gnomonic projection via spreads/crosses; Q=spread_c/cross_c=tan²(angular_dist); great circles→straight lines (collinearity <10⁻¹²); Berggren tree C²+F²=G² = discrete geodesic steps projecting to lines; London tangent 5 cities; Plimpton 322 = nav table; Tier 1+2; checks GN_1+QUAD/SPREAD/COLLINEAR/BERGGREN/W/F; 1 PASS + 1 FAIL; self-test ok",
     "164_qa_gnomonic_rt_cert",
     "qa_gnomonic_rt_cert_v1", True),
    (165, "QA Celestial Nav Cert family",
     _validate_celestial_nav_cert_family,
     "Sight reduction as RT: s_h=[σ₁√(s_φ·s_δ)+σ₂√(c_φ·c_δ·c_LHA)]²; σ₁,σ₂∈{±1} discrete orientation; sextant=spread instrument; position circle=spread locus; two-star fix=algebraic; 7 stars London; Tier 1; checks CN_1+SIGHT/SPREAD/SIGMA/AZIMUTH/FIX/W/F; 1 PASS + 1 FAIL; self-test ok",
     "165_qa_celestial_nav_cert",
     "qa_celestial_nav_cert_v1", True),
    (166, "QA Loxodrome Cert family",
     _validate_loxodrome_cert_family,
     "Loxodromes=constant-bearing T-operator paths on mod-m lattice; period=Pisano π(m); bearing spread=e²/G; Mercator identity s_φ=tanh²(ψ); 3 orbit types; cosmos/satellite/singularity partition; 4 paths + 5 Mercator points; Tier 2; checks LX_1+PATH/BEARING/MERCATOR/ORBIT/W/F; 1 PASS + 1 FAIL; self-test ok",
     "166_qa_loxodrome_cert",
     "qa_loxodrome_cert_v1", True),
    (167, "QA Historical Nav Cert family",
     _validate_historical_nav_cert_family,
     "5 civilizations proto-QA: Babylon P322=Berggren tree; Egypt seked=spread ratio; Polynesia star compass=mod-32; Norse sun stones=spread measurement; Arab kamal=integer spread increments; common: discrete states+integer arithmetic+observer projection; Tier 2; checks HN_1+SYSTEM/SEKED/KAMAL/TRIPLE/W/F; 1 PASS + 1 FAIL; self-test ok",
     "167_qa_historical_nav_cert",
     "qa_historical_nav_cert_v1", True),
    (168, "QA Ellipsoid Geodesic Cert family",
     _validate_ellipsoid_geodesic_cert_family,
     "WGS84 quantum ellipse geodesics in QN arithmetic; M/N=F/(d²-e²·s_φ); b/a=√F/d=√12019/110; I=C-F=-10039<0=ellipse; quantum lattice s_φ=C/G≈Tropic(23.78°); shape/orbit QN harmonically independent; Tier 1; checks EG_1+QN/CURV/AXIS/DISC/LATTICE/W/F; 1 PASS+1 FAIL; self-test ok",
     "168_qa_ellipsoid_geodesic_cert",
     "qa_ellipsoid_geodesic_cert_v1", True),
    (169, "QA Ellipsoid Slice Cert family",
     _validate_ellipsoid_slice_cert_family,
     "QA slicing: latitude circles R²=a²d²c_φ/(d²-e²s_φ); meridian=shape QN ellipse (self-similar); chromo C/F/G curve families; 24 Pisano longitude bands=time zones; Tropic≈C/G quantum point; Tier 1+2; checks SL_1+LAT/MER/CHROMO/BAND/W/F; 1 PASS+1 FAIL; self-test ok",
     "169_qa_ellipsoid_slice_cert",
     "qa_ellipsoid_slice_cert_v1", True),
    (170, "QA Cardiac Arrhythmia Cert family",
     _validate_cardiac_arrhythmia_cert_family,
     "MIT-BIH 48 records 94536 beats; dR2=+0.037 beyond R-R interval p<10^-6; 2/2 surrogates beaten; Phi(D)=-1 pre-registered and confirmed (disorder-stress); checks CAR_1+DATA/DELTA/SURR/PHI/W/F; 1 PASS + 1 FAIL; self-test ok",
     "170_qa_cardiac_arrhythmia_cert",
     "qa_cardiac_arrhythmia_cert_v1", True),
    (171, "QA EMG Pathology Cert family",
     _validate_emg_pathology_cert_family,
     "PhysioNet EMG 3 records 1203 windows (healthy/myopathy/neuropathy); dR2=+0.608 beyond RMS p<10^-6; 2/2 surrogates beaten; Phi(D)=-1 pre-registered; checks EMG_1+DATA/DELTA/SURR/PHI/W/F; 1 PASS + 1 FAIL; self-test ok",
     "171_qa_emg_pathology_cert",
     "qa_emg_pathology_cert_v1", True),
    (172, "QA ERA5 Reanalysis Cert family",
     _validate_era5_reanalysis_cert_family,
     "WeatherBench2 ERA5 3297 days x 15 channels 500hPa; r=+0.46 partial r=+0.43; 4/4 surrogates beaten; checks ERA_1+DATA/R/PARTIAL/SURR/W/F; 1 PASS + 1 FAIL; self-test ok",
     "172_qa_era5_reanalysis_cert",
     "qa_era5_reanalysis_cert_v1", True),
    (173, "QA Surrogate Methodology Cert family",
     _validate_surrogate_methodology_cert_family,
     "Corrected null design: real targets fixed, surrogate QCI only; circular null problem identified and resolved; 6/8 domains confirmed; checks SRM_1+DESIGN/CIRCULAR/DOMAINS/W/F; 1 PASS + 1 FAIL; self-test ok",
     "173_qa_surrogate_methodology_cert",
     "qa_surrogate_methodology_cert_v1", True),
    (174, "QA Phi Transformation Cert family",
     _validate_phi_transformation_cert_family,
     "Phi(D) transformation law: disorder-stress vs order-stress; 2/2 pre-registered (cardiac, EMG); 6/6 post-hoc consistent; domain requirement: temporal multi-channel signals; checks PHI_1+CLASS/PREREG/POSTHOC/REQ/W/F; 1 PASS + 1 FAIL; self-test ok",
     "174_qa_phi_transformation_cert",
     "qa_phi_transformation_cert_v1", True),
    (175, "QA Cross-Domain Invariance Cert family",
     _validate_cross_domain_invariance_cert_family,
     "3 structural invariants: surrogate survival, independent information, domain-general architecture; 6 Tier 3 domains confirmed; checks CDI_1+INV1/INV2/INV3/W/F; 1 PASS + 1 FAIL; self-test ok",
     "175_qa_cross_domain_invariance_cert",
     "qa_cross_domain_invariance_cert_v1", True),
    (176, "QA Inertial Nav Cert family",
     _validate_inertial_nav_cert_family,
     "Zero drift proof: classical INS O(ε√N) vs QA exact 0; 3 routes × 4 noise levels (ULP/trig/MEMS/cheap); ratio→∞; FPGA: ~10% logic of sin/cos pipeline; Theorem NT: only error = observer projection; Tier 1 computational; checks IN_1+QA_EXACT/DRIFT/ZERO/RATIO/W/F; 1 PASS+1 FAIL; self-test ok",
     "176_qa_inertial_nav_cert",
     "qa_inertial_nav_cert_v1", True),
    (177, "QA Planetary QN Cert family",
     _validate_planetary_qn_cert_family,
     "Solar system QN catalog: 10 bodies shape+orbital; Earth-Jupiter b=59, Earth-Uranus b=101 harmonics; Saturn (79,139,199) prime AP; char latitudes via 2ε/(1+ε²); Earth Tropic match Tier 2 p≈0.013; Tier 2 structural; checks PQ_1+TUPLE/TRIPLE/ECC/HARMONIC/W/F; 1 PASS+1 FAIL; self-test ok",
     "177_qa_planetary_qn_cert",
     "qa_planetary_qn_cert_v1", True),
    (178, "QA Megalithic Cert family",
     _validate_megalithic_cert_family,
     "Megalithic Yard p=0.00022 z=-3.54 (Thom 1962+1967 combined 202 circles); Fathom p<10⁻⁸ (74% even of 202); 3:4:5→QN(1,1,2,3) 5:12:13→QN(1,2,3,5) Fibonacci; honest: Fib ratios ns, mod-9 uniform, mod-24 explained by fathom; Tier 2+3; checks MG_1+MY/FATHOM/TRIANGLE/HONEST/W/F; 1 PASS+1 FAIL; self-test ok",
     "178_qa_megalithic_cert",
     "qa_megalithic_cert_v1", True),
    (179, "QA Paired Pisano Cert family",
     _validate_paired_pisano_cert_family,
     "Paired Pisano divisibility: Fib pairs 2.25x higher both-divide rate (0.526 vs 0.234), Mann-Whitney p=0.0017; order-1 3.77x p=0.028; mechanism: lcm(p,q)=p*q smaller for Fib; product-matched 56%; 4:1 exception (Kirkwood); Tier 3; checks PP_1+PAIRS/STAT/RATIO/ORDER/MECH/HONEST/W/F; 1 PASS+1 FAIL; self-test ok",
     "179_qa_paired_pisano_cert",
     "qa_paired_pisano_cert_v1", True),
    (180, "QA H-Null Modularity Cert family",
     _validate_h_null_modularity_cert_family,
     "H-null chromogeometric modularity: H(b,e)=C+F where C=2de (green) F=d*d-e*e (red); Les Miserables ARI=0.638 vs standard 0.588 (+0.050); HONEST: 1/10 graphs improved, topology-specific to hub-dominated networks; H/X=b/e+4+2e/b linear in degree asymmetry; Tier 2; checks HN_1+MODEL/CHROMO/BENCH/HONEST/W/F; 1 PASS + 1 FAIL; self-test ok",
     "180_qa_h_null_modularity_cert",
     "qa_h_null_modularity_cert_v1", True),
    (181, "QA Satellite Product Sum Cert family",
     _validate_satellite_product_sum_cert_family,
     "satellite product sum identity: sum_{satellite} b*e*d*a = M^4 for all M divisible by 3; proof via Fibonacci closure of satellite sub-lattice; normalized product sum = 81 = 3^4; corollary: satellite total volume = singularity volume; Tier 1 algebraic; verified 33 moduli; checks SPS_1+PROOF/COUNT/SUM/TUPLES/CLOSURE/COROL/W/F; 1 PASS + 1 FAIL; self-test ok",
     "181_qa_satellite_product_sum_cert",
     "qa_satellite_product_sum_cert_v1", True),
    (182, "QA Miller Orbit Cert family",
     _validate_miller_orbit_cert_family,
     "Miller orbit crystallography: cosmos d > satellite d universal (21/21 minerals, p<10⁻⁶); satellite Q_M mod 9 ∈ QR(9)={0,1,4,7} (quadratic residue restriction); singularity Q_M = perfect squares; satellite green 3× cosmos; all Tier 1 algebraic; 13055 reflections, 4 crystal systems; checks MO_1+ORDER/QR/SQUARE/CHROMO/W/F; 1 PASS + 1 FAIL; self-test ok",
     "182_qa_miller_orbit_cert",
     "qa_miller_orbit_cert_v1", True),
    (183, "QA Eisenstein Crystal Cert family",
     _validate_eisenstein_crystal_cert_family,
     "Eisenstein crystal bridge: Z-Y=J=bd (new identity); Z²-Y²=J·a·(a+e) factorization; F²-FW+W²=Z² encodes crystal constants; Unity Block {F,G,Z,W}={3,5,7,8}=Ben's four Forces; J·a·(a+e)=2·3·4=24=cosmos period; Tier 1 algebraic; 10 witnesses; checks EC_1+ZYJ/FACTOR/EISEN/TUPLE/UNITY/W/F; 1 PASS + 1 FAIL; self-test ok",
     "183_qa_eisenstein_crystal_cert",
     "qa_eisenstein_crystal_cert_v1", True),
    (184, "QA Keely Structural Ratio Cert family",
     _validate_keely_structural_ratio_cert_family,
     "Keely's 8 structural ratio laws (2,4,9,10,18,27,29,33) → QA modular invariants; Category 1 Vibes framework; pitch=f-value, period 1|8|24, concordance, chromogeometry C*C+F*F=G*G; checks KSR_1+LAWS/PERIOD/FVAL/LCM/CHROMO/CLOSURE/W/F; 1 PASS + 1 FAIL; self-test ok",
     "184_qa_keely_structural_ratio_cert",
     "qa_keely_structural_ratio_cert_v1", True),
    (185, "QA Keely Sympathetic Transfer Cert family",
     _validate_keely_sympathetic_transfer_cert_family,
     "Keely's 7 sympathetic transfer laws (5,6,7,8,17,37,40) → QA reachability/path structure; Category 2 Vibes framework; orbit co-membership=sympathy, discord=obstruction, triad concordance; checks KST_1+LAWS/REACH/BLOCK/PATH/TRIAD/W/F; 1 PASS + 1 FAIL; self-test ok",
     "185_qa_keely_sympathetic_transfer_cert",
     "qa_keely_sympathetic_transfer_cert_v1", True),
    (186, "QA Keely Dominant Control Cert family",
     _validate_keely_dominant_control_cert_family,
     "Keely's 3 dominant/control laws (1,11,16) → QA orbit hierarchy; Category 3 Vibes framework; invariant substrate, triune generator, singularity=neutral center; checks KDC_1+LAWS/SUB/TRIUNE/SING/W/F; 1 PASS + 1 FAIL; self-test ok",
     "186_qa_keely_dominant_control_cert",
     "qa_keely_dominant_control_cert_v1", True),
    (187, "QA Keely Aggregation Cert family",
     _validate_keely_aggregation_cert_family,
     "Keely's 5 aggregation/disintegration laws (3,12,28,34,35) → QA state composition/decomposition; Category 4 Vibes framework; coupling tension, orbit density, discord dissociation, deterministic synthesis; checks KAG_1+LAWS/COUPLE/DENSITY/DISSOC/SYNTH/W/F; 1 PASS + 1 FAIL; self-test ok",
     "187_qa_keely_aggregation_cert",
     "qa_keely_aggregation_cert_v1", True),
    (188, "QA Keely Phenomenological Cert family",
     _validate_keely_phenomenological_cert_family,
     "Keely's 17 phenomenological laws (13-15,19-26,30-32,36,38,39) → Theorem NT observer projections; Category 5 Vibes framework; 42.5% of laws; continuous measurements reveal but never feed back into QA; checks KPH_1+LAWS/NT/OBS/DISC/W/F; 1 PASS + 1 FAIL; self-test ok",
     "188_qa_keely_phenomenological_cert",
     "qa_keely_phenomenological_cert_v1", True),
    (189, "QA Dale Circle Cert family",
     _validate_dale_circle_cert_family,
     "Dale Pond's integer circle: P=2W (diameter), Q=P (circumference=diameter in QA units), R=W² (area); pi disappears; svpwiki.com 1998; checks DC_1+P/Q/R/W/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "189_qa_dale_circle_cert",
     "qa_dale_circle_cert_v1", True),
    (190, "QA Equilateral Height Cert family",
     _validate_equilateral_height_cert_family,
     "Element S=d²e=d*X=D*e; Dale Pond equilateral triangle height (#25 svpwiki.com); three equivalent definitions; 7 directions verified; connects [152] W side + [189] circle; checks EH_1+S/DX/DE/W/F; 1 PASS + 1 FAIL; self-test ok",
     "190_qa_equilateral_height_cert",
     "qa_equilateral_height_cert_v1", True),
    (191, "QA Bateson Learning Levels Cert family",
     _validate_bateson_learning_levels_cert_family,
     "Bateson Learning 0/I/II/III as strict invariant filtration (orbit ⊂ family ⊂ modulus ⊂ ambient category); L_0/L_1/L_2a/L_2b/L_3 operator classes; Tiered Reachability Theorem exhaustively verified on S_9 (81+1712+3456+1312=6561); only 26% Level-I reachable; witnesses: qa_step, scalar_mult k=2 (L_2a), scalar_mult k=3 (L_2b), modulus_reduction (L_3); source Bateson 1972 + Ashby 1956; checks BLL_1+FILT/TIER/L1/L2A/L2B/L3/STRICT/DB/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "191_qa_bateson_learning_levels_cert",
     "qa_bateson_learning_levels_cert_v1", True),
    (192, "QA Dual Extremality 24 Cert family",
     _validate_dual_extremality_24_cert_family,
     "Joint extremality of m=24: simultaneously minimum non-trivial Pisano fixed point (OEIS A235702 = {24, 120, 600, ...}) AND maximum Carmichael lambda=2 modulus (structurally proved: m | 24, set = {3,4,6,8,12,24}); pi(9)=24 bridges theoretical to applied modulus; basin {6,9,12,16,18,24}; cannonball 1^2+..+24^2=70^2; 24-theorem p^2-1 div 24; closes item 5 of [191] Bateson sketch (Level-III self-improvement fixed point); ORIGINAL joint (pi,lambda) observation; source Wall 1960 + OEIS + Carmichael 1910 + Watson 1918 + Baez 2008; checks DE_1+PISANO/MIN_FP/CARMICHAEL/MAX_LAM/JOINT/BRIDGE/BASIN/CANNON/24THM/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "192_qa_dual_extremality_24_cert",
     "qa_dual_extremality_24_cert_v1", True),
    (193, "QA Levin Cognitive Lightcone Cert family",
     _validate_levin_cognitive_lightcone_cert_family,
     "Levin CLC mapped to QA orbit radius: Singularity=0 (fixed point), Satellite=8 (local goals), Cosmos=24 (far-reaching goals); cancer = CLC shrinkage = Cosmos->Satellite orbit demotion (L_2a); Tiered Reachability [191] gives 26% L1-reachable structural ceiling; source Levin & Resnik 2026 + Lyons/Pio-Lopez/Levin 2026; checks CLC_1+ORBIT/RADIUS/CANCER/CEIL/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "193_qa_levin_cognitive_lightcone_cert",
     "qa_levin_cognitive_lightcone_cert_v1", True),
    (194, "QA Cognition Space Morphospace Cert family",
     _validate_cognition_space_morphospace_cert_family,
     "Sole/Seoane et al. qualitative morphospace realized as QA exact discrete morphospace; three clusters (basal/neural/human-AI) = three orbits; voids algebraically necessary (missing divisors {2,3,4,6,12}); agency = |orbit|/81; constructive enumerable cognition space; source Sole et al. arXiv:2601.12837; checks CSM_1+CLUSTERS/VOIDS/AGENCY/ENUM/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "194_qa_cognition_space_morphospace_cert",
     "qa_cognition_space_morphospace_cert_v1", True),
    (195, "QA Pezzulo Levin Bootstrap Cert family",
     _validate_pezzulo_levin_bootstrap_cert_family,
     "Pezzulo & Levin 7-stage bootstrapping pipeline mapped to QA architecture levels (Chemistry=A1 through Creativity=L_3 pi(9)=24); intelligence ratchet = Pisano FP [192]; 5 design principles map to QA axioms (autonomy=A1, self-assembly=orbit emergence, rebuilding=T1, constraints=S1+S2, signaling=resonance); source Pezzulo & Levin arXiv:2602.08079; checks PLB_1+STAGES/RATCHET/PRINCIPLES/PIPELINE/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "195_qa_pezzulo_levin_bootstrap_cert",
     "qa_pezzulo_levin_bootstrap_cert_v1", True),
    (196, "QA See Capture Convergence Cert family",
     _validate_see_capture_convergence_cert_family,
     "T.J.J. See capture theory (1910) mapped to QA orbit convergence; free body=arbitrary (b,e); resisting medium=modular reduction; eccentricity decay=transient steps; stable capture=orbit membership; S_9: cosmos=72, satellite=8, singularity=1, all tau=0 (instant capture); source See 'Capture Theory' 1910; checks SCC_1+CONV/MEAN/MAX/DIST/MED/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "196_qa_see_capture_convergence_cert",
     "qa_see_capture_convergence_cert_v1", True),
    (197, "QA See Longitudinal Transverse Cert family",
     _validate_see_longitudinal_transverse_cert_family,
     "T.J.J. See wave duality (1917) mapped to QA generator/observer duality; longitudinal=T-operator (discrete causal), transverse=observer projection (continuous measurement); Theorem NT = mode orthogonality; complementary to [153] Keely triune (3-fold within longitudinal); source See 'Electrodynamic Wave-Theory' 1917; checks SLT_1+LONG/TRANS/ORTH/NT/KEELY/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "197_qa_see_longitudinal_transverse_cert",
     "qa_see_longitudinal_transverse_cert_v1", True),
    (198, "QA Pudelko Modular Periodicity Cert family",
     _validate_pudelko_modular_periodicity_cert_family,
     "Pudelko modular Fibonacci periodicity mapped to QA T-operator family counts; partial-verification honesty gate preserves V2 Legendre-bridge refinement and V5 mirror-symmetry open item; checks PUD_1+STATUS/ORBIT/SELF_SIM/WEIGHT/HONEST/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "198_qa_pudelko_modular_periodicity_cert",
     "qa_pudelko_modular_periodicity_cert_v1", True),
    (199, "QA Grokking Eigenvalue Transition Cert family",
     _validate_grokking_eigenvalue_transition_cert_family,
     "Schiffman grokking eigenvalue transition mapped to QA with m=97 prime-control evidence, m=9 non-grokking composite target, and explicit DFT frequency-pair vs QA orbit-family correction; checks GET_1+STATUS/PRIME/COMPOSITE/CORRECTION/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "199_qa_grokking_eigenvalue_transition_cert",
     "qa_grokking_eigenvalue_transition_cert_v1", True),
    (200, "QA Spherical Grokking Theorem NT Cert family",
     _validate_spherical_grokking_theorem_nt_cert_family,
     "Yildirim spherical grokking mapped to Theorem NT; partial-verification honesty gate preserves local m=97 3x speedup, residual norm bound, uniform-attention result, m=9 non-applicability, and untested S5 boundary; checks SGT_1+STATUS/SPEEDUP/NORM/UNIFORM/M9/HONEST/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "200_qa_spherical_grokking_theorem_nt_cert",
     "qa_spherical_grokking_theorem_nt_cert_v1", True),
    (201, "QA Snell Manuscript Cert family",
     _validate_snell_manuscript_cert_family,
     "Snell Manuscript (Keely, 1934) 7 structural claims mapped to QA: 7x3=21 hierarchy, frequency scaling by 3/9, Trexar Ag/Au/Pt={3,6,9} orbit encoding, mass-as-difference=f-value, polarity inversion 2/3 threshold, triple dissociation=orbit descent, rotation 3:9=cosmos/satellite ratio. Checks SNM_1+21/FREQ/SCALE/TREX/FVAL/POL/DISS/ROT/CHORD/W/F; 2 PASS + 1 FAIL; self-test ok",
     "201_qa_snell_manuscript_cert",
     "qa_snell_manuscript_cert_v1", True),
    (202, "QA Hebrew Mod-9 Identity Cert family",
     _validate_hebrew_mod9_identity_cert_family,
     "Hebrew gematria Aiq Bekar / pythmen = QA A1 mod-9 identity. Three enneads (27=3x9), digital root homomorphism (Izmirli 2014), Sefer Yetzirah 4!=24, Skinner 9^4=6561 kernel, 9->24 bridge, Pythagorean transmission (Iamblichus), base-9 hypothesis (Kreinovich 2018). Checks HM9_1+AIQ/DR/ENNEAD/SY24/SKIN/BRIDGE/NUM/W/F; 2 PASS + 1 FAIL; self-test ok",
     "202_qa_hebrew_mod9_identity_cert",
     "qa_hebrew_mod9_identity_cert_v1", True),
    (203, "QA Sefer Yetzirah Combinatorics Cert family",
     _validate_sefer_yetzirah_combinatorics_cert_family,
     "Sefer Yetzirah (Book of Formation, c. 2nd-6th century CE) combinatorial structures: 231 gates = C(22,2) = K_22 (earliest binomial coefficient), n! for n=2..7 (earliest factorial), 3-7-12 letter partition, 32 paths = 2^5, oscillating circle, Pythagorean transmission (Iamblichus/Carmel), tzeruf permutation groups. Checks SYC_1+GATES/FACT/PART/PATHS/NUM/W/F; 2 PASS + 1 FAIL; self-test ok",
     "203_qa_sefer_yetzirah_combinatorics_cert",
     "qa_sefer_yetzirah_combinatorics_cert_v1", True),
    (204, "QA Skinner Hebrew Metrology Cert family",
     _validate_skinner_hebrew_metrology_cert_family,
     "Skinner 'Source of Measures' (1875) 7 verified metrological claims: Parker kernel 6561=9^4, Garden-Eden=24 (digital roots), solar day 5184=72^2 (Cosmos pairs), Adam/Woman dr=9, factor 6 bridge 9->24, Metius dr-closure dr(113)+dr(355)=9, T2 compliance. 3 qualified: El=31 subgroup, palindrome trivial, Parker pi mediocre. Checks SKM_1+PARK/EDEN/SOLAR/ADAM/BRIDGE/MET/NUM/W/F; 2 PASS + 1 FAIL; self-test ok",
     "204_qa_skinner_hebrew_metrology_cert",
     "qa_skinner_hebrew_metrology_cert_v1", True),
    (205, "QA Grid Cell RNS Cert family",
     _validate_grid_cell_rns_cert_family,
     "Grid cell residue number system (Fiete 2008, Wei 2015, Vago 2018, Constantinescu 2016) = QA modular arithmetic: RNS isomorphism, CRT reconstruction, 24/9 within 2% of optimal e, LCM(9,24)=72=Cosmos, phi optimal for two-module, carry-free, abstract domains, toroidal state space, hex27 encoding. Checks GCR_1+RATIO/LCM/PHI6/HEX/NUM/W/F; 2 PASS + 1 FAIL; self-test ok",
     "205_qa_grid_cell_rns_cert",
     "qa_grid_cell_rns_cert_v1", True),
    (206, "QA HERA Orchestration Evolution Cert family",
     _validate_hera_orchestration_evolution_cert_family,
     "HERA multi-agent orchestration (Li & Ramakrishnan VT 2026) = QA orbit dynamics: RoPE = Bateson [191] L1/L2a filtration, 4-phase topology = orbit descent (never Singularity), entropy plateau = Satellite convergence, sparse exploration = orbit discovery, Theorem NT compliance. 38.69% over SOTA. Checks HOE_1+ROPE/PHASE/ENT/PERF/W/F; 2 PASS + 1 FAIL; self-test ok",
     "206_qa_hera_orchestration_evolution_cert",
     "qa_hera_orchestration_evolution_cert_v1", True),
    (207, "QA Circle Impossibility Cert family",
     _validate_circle_impossibility_cert_family,
     "No QA state has C=0 (circle impossible). C=2de>=2 always by A1 No-Zero. Circle = observer projection of ellipsoid where C lies along viewing axis (Will Dale 2026-04-08). Extends [140] parabola impossibility. Connects [189] Dale Circle, [125] Chromogeometry, Theorem NT. Checks CI_1+C_MIN/EXHAUSTIVE/PROJECTION/HIERARCHY/CHROMO/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "207_qa_circle_impossibility_cert",
     "qa_circle_impossibility_cert_v1", True),
    (208, "QA Quadrance Product Cert family",
     _validate_quadrance_product_cert_family,
     "Every QA area element is irreducibly a two-factor product of role-distinct base elements. S1 (b*b product form) is structural: product preserves two-factor roles, power operator collapses them. 1*1=1 is area, not scalar. Square = rectangle with equal values, distinct roles. Parallels [207] circle impossibility. Will Dale 2026-04-08. Checks QP_1+PRODUCT/ROLE/S1/AREA_MIN/DIM/SQUARE/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "208_qa_quadrance_product_cert",
     "qa_quadrance_product_cert_v1", True),

    (209, "QA Signal Generator Inference Cert family",
     _validate_signal_generator_inference_cert_family,
     "For any m-valued time series, e_t = ((b_{t+1} - b_t - 1) % m) + 1 is the unique A1-compliant generator. Signal IS orbit; generator IS dynamics. Role-distinct per [208]. Cross-series synchrony per [207]. Supersedes hardcoded CMAP/MICROSTATE_STATES. EEG chb01: directionally correct (synch seizure>baseline), ns on single patient, needs multi-patient. Will Dale + Claude 2026-04-08. Checks SGI_1+CLOSURE/UNIQUE/ROLE/SYNC/EMPIRICAL/SUPERSEDE/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "209_qa_signal_generator_inference_cert",
     "qa_signal_generator_inference_cert_v1", True),

    (211, "QA Cayley Bateson Filtration Cert family",
     _validate_cayley_bateson_filtration_cert_family,
     "Proves that the tiered reachability classes of family [191] are exactly the connected components of nested undirected Cayley graphs on S_9 under stratified generator sets. Gamma_L1={T} -> (24,24,24,8,1); Gamma_L2a adds (Z/9Z)* scalars + swap -> (72,8,1); Gamma_L2b adds non-unit scalars + const_(9,9) -> (81). Cumulative sums-of-squares 1793/5249/6561 and non-cumulative differences 1712/3456/1312 match [191] EXPECTED_TIER_COUNTS_S9 byte-for-byte. Undirected Cayley convention is essential because scalar_3 and const are non-bijective. Validator independently recomputes components on S_9 via stdlib BFS. Source: Cayley 1878 (graphical representation of groups), Dehn 1911 (word problem). Prerequisite: [191]. Will Dale + Claude 2026-04-11. Checks CBF_1+GEN/COMP/CUMU/DIFF/L1/L2A/L2B/191/SRC/WIT/F; 1 PASS + 1 FAIL; self-test ok",
     "211_qa_cayley_bateson_filtration_cert",
     "qa_cayley_bateson_filtration_cert_v1", True),

    (210, "QA Conversation A-RAG Cert family",
     _validate_conversation_arag_cert_family,
     "QA-native conversation retrieval datastore: ChatGPT/Claude.ai/Gemini exports mapped to integer tuples via Candidate F b=dr(sum(ord(c))) + e=role_rank, grounded in [202] Aiq Bekar digital root. Three A-RAG views (keyword_search=FTS5, semantic_search=PPR over parent/cite/succ/ref edges, chunk_read). Role-diagonal property: each canonical role occupies one mod-9 diagonal because (a_label-d_label) mod 9 = e mod 9. Cross-source invariance verified on 5361 msgs (Claude+Gemini). T2 firewall: FTS5/PPR scores are observer measurements, never QA inputs. A2 compliance: only b,e stored, d/a derived on read. Composes with [18], [20], QA_ARAG_INTERFACE, [202] dr, [122] OB bridge, [191] Bateson. Checks CAV_1+SCHEMA/TUPLE/DIAG/CROSS/PROMO/A1/A2/T2/VIEWS/W/F; 2 fixtures PASS+FAIL; self-test ok",
     "210_qa_conversation_arag_cert",
     "qa_conversation_arag_cert_v1", True),

    (212, "QA Fibonacci Hypergraph Cert family",
     _validate_fibonacci_hypergraph_cert_family,
     "Every QA state (b,e) defines a length-4 Fibonacci window hyperedge (b,e,d,a). The resulting state-residue incidence hypergraph H(m) satisfies three structural theorems on S_9: (1) SLIDING WINDOW h(T(s))=(e,d,a,(d+a) mod m), 81/81 states match; (2) UNIFORM VERTEX DEGREE every residue in {1..9} has degree 36=4m, total 324; (3) ORBIT-MULTISET COLLAPSE cosmos/satellite/singularity T-orbits (24,24,24,8,1) produce (22,22,22,4,1) distinct multiset hyperedges. Validator independently recomputes all three theorems in stdlib. Source: Fibonacci 1202 (recurrence), Lucas 1878 + Wall 1960 (Pisano periods), Berge 1989 (hypergraph theory). Prerequisites: [191] Bateson Learning Levels, [192] Dual Extremality 24, [211] Cayley view. Will Dale + Claude 2026-04-11. Checks HGR_1+SLIDE/DEG/ORB/FIB/191/SRC/WIT/F; 1 PASS + 1 FAIL; self-test ok",
     "212_qa_fibonacci_hypergraph_cert",
     "qa_fibonacci_hypergraph_cert_v1", True),

    (213, "QA Causal DAG Cert family",
     _validate_causal_dag_cert_family,
     "The A2 axiom (d=b+e, a=b+2e) IS the structural equation system of a 4-node Y-structure causal DAG with b,e exogenous and d,a endogenous colliders. Pair-invertibility theorem: all 6 unordered pairs of {b,e,d,a} bijective as maps from S_m=(1..m)^2 to their image iff gcd(2,m)=1. On S_9 (gcd=1): all 6 bijective (81/81 each). On S_24 (gcd=2): 5 bijective, pair (b,a) is 2-to-1 (288/576). Pearl-level collapse theorem: because A2 is deterministic integer arithmetic, Pearl's association/intervention/counterfactual hierarchy collapses to the A2 identities themselves. This is the SCM form of Theorem NT observer projection firewall. Source: Pearl 2009 (Causality), Wright 1921 (path analysis). Prerequisites: [191], [150], [202]. Will Dale + Claude 2026-04-11. Checks CDG_1+STRUCT/A2/PAIRS/PEARL/NT/191/SRC/WIT/F; 1 PASS + 1 FAIL; self-test ok",
     "213_qa_causal_dag_cert",
     "qa_causal_dag_cert_v1", True),

    (214, "QA Norm-Flip Signed-Temporal Cert family",
     _validate_norm_flip_signed_cert_family,
     "The Eisenstein quadratic form f(b,e) = b*b + b*e - e*e satisfies the integer identity f(e,b+e) = -f(b,e) where T(b,e)=(e,b+e). Corollary: T^2 preserves f mod m, giving T-orbit graph of S_m a signed-temporal structure. On S_9, 5 T-orbits decompose into 3 signed cosmos orbits with norm pairs {1,8}/{4,5}/{2,7} (Fibonacci/Lucas/Phibonacci) and 2 null orbits (satellite Tribonacci, singularity Ninbonacci) where f is identically 0 mod 9. Three cosmos orbits are bipartite signed (12 + 12 states alternating sign under T). Temporal sign formula: sign(f(T^t(s_0))) = (-1)^t * sign(f(s_0)) on integer lift. This is the SIGNED-TEMPORAL view of the same T dynamic certified structurally by [211]-[213] and operationally by [210]. Source: Eisenstein 1844, Pythagorean Families paper (Will Dale + Claude 2026-03). Connects [133] Eisenstein cert, [155] Bearden phase conjugate (QCI opposite-sign = norm-sign flip), [191] cosmos/satellite/singularity stratification. Will Dale + Claude 2026-04-11. Checks NFS_1+FLIP/T2/PAIRS/TEMPORAL/155/133/SRC/WIT/F; 1 PASS + 1 FAIL; self-test ok",
     "214_qa_norm_flip_signed_cert",
     "qa_norm_flip_signed_cert_v1", True),
    (215, "QA Resonance-Bin Correspondence Cert family",
     _validate_resonance_bin_correspondence_cert_family,
     "Bin-width ≡ resonance-tolerance isomorphism: at modulus m, equivalence class [k]_m = {x ∈ R : quantize(x,m)=k} is isomorphic to a resonance tolerance bandwidth; Hensel lift mod 3→9→27 = progressive bandwidth narrowing. Three witnesses: (W1) Arnold tongue empirical — critical coupling K* for phase-lock mode-dominance rises monotone in m (m=6→0.06, m=18→0.08, m=48→0.10); (W2) Hensel external-artifact reference to qa_brainca_selforg_v2.py; (W3) integer-only round-trip (S2/A1). Candidate permissibility-filter formalization flagged as open in docs/theory/QA_SYNTAX_SVP_SEMANTICS.md (Dale Pond + Vibes letter 2026-04-05, OB a9307705). Source: Will Dale + Claude 2026-04-12. Arnold 1961, Q-factor resonance theory. Checks RBC_1+SCHEMA/ARNOLD/BINWIDTH/HENSEL/ROUNDTRIP/INT_ONLY/SELFTEST; 1 PASS + 1 FAIL; self-test ok",
     "215_qa_resonance_bin_correspondence_cert",
     "qa_resonance_bin_correspondence_cert_v1", True),
    (216, "QA EBM Equivalence Cert family",
     _validate_ebm_equivalence_cert_family,
     "QA coherence is a discrete-native, Theorem NT-compliant Energy-Based Model. Pointwise energy E_QA(b,e,next)=0 if T(b,e)==next else 1; window energy = 1-QCI. Five EBM axioms verified: (E1) non-negativity exhaustive on S_9; (E2) data-manifold zero on deterministic T-trajectory; (E3) monotonicity — E grows 0→0.19→0.46→0.66→0.83 with 0/10/30/50/80% injected mismatch; (E4) Boltzmann occupancy well-formed with T=2π/m per cert [215]; (E5) score identity — argmax of Boltzmann over next_state = T-operator (exhaustive S_9, 81/81). qa_detect IS a trained EBM; MCMC-free sampling = T-operator walk; no gradient approximation. Source: Will Dale + Claude 2026-04-12; LeCun et al. 2006, Hinton 2002 CD. Cross-refs [154][191][215], Theorem NT. Checks EBM_1+SCHEMA/NONNEG/ZERO/MONOTONE/BOLTZMANN/SCORE/INT_ONLY/SELFTEST; 1 PASS + 1 FAIL; self-test ok",
     "216_qa_ebm_equivalence_cert",
     "qa_ebm_equivalence_cert_v1", True),
    (217, "QA Fuller VE Diagonal Decomposition Cert family",
     _validate_fuller_ve_diagonal_decomposition_cert_family,
     "Fuller's cuboctahedral / vector-equilibrium shell count S_n = 10n^2+2 (n=1: 12, n=2: 42, n=3: 92, n=4: 162, n=5: 252, n=6: 362, n=7: 492, n=8: 642, n=9: 812) decomposes across QA integer diagonals by n mod 3: on b=e diagonal D_1 iff n mod 3 != 0 (tuple (S_n/3, S_n/3, 2*S_n/3, S_n)); else on sibling odd-divisor diagonal D_k with (2k+1)|S_n. Proof: S_n mod 3 = (n^2+2) mod 3 = 0 iff n not divisible by 3. First documented physical hierarchy whose QA decomposition is mixed across diagonal classes (complements FST/STF, which sits entirely on D_1 per Briddell). Mod-3 selection is QA-native triune structure on canonical Sierpinski diagonal; 2:1 density ratio of on- vs off-diagonal shells. Foundation note: docs/theory/QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md. Source: Will Dale + Claude 2026-04-13; Buckminster Fuller, Synergetics (1975). Checks FVDD_1+FORMULA/MOD3/DIAGONAL/OFFDIAGONAL/COMPUTATIONAL/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "217_qa_fuller_ve_diagonal_decomposition_cert",
     "qa_fuller_ve_diagonal_decomposition_cert_v1", True),
    (218, "QA Haramein Scaling Diagonal Cert family",
     _validate_haramein_scaling_diagonal_cert_family,
     "Haramein-Rauscher-Hyson 2008 Table 1 (Big Bang/Planck, Atomic, Stellar Solar, Galactic G1, Galactic G2, Universe) encoded as integer (log10 R cm, log10 nu Hz) sits on QA fixed-d hyperbola (b+e = const, Schwarzschild line R*nu=c after decade-rounding). Four structural segment-ratios on 2D Euclidean distances in (log R, log nu) have integer quadratic-form quotients approximating {phi^2, 1/phi^2} within 7%: (25^2+25^2)/(16^2+15^2)=1250/481~phi^2 (0.7% off); (6^2+7^2)/(4^2+4^2)=85/32~phi^2 (1.4%); (2^2+3^2)/(4^2+4^2)=13/32~1/phi^2 (6.3%); (16^2+16^2)/(25^2+25^2)=512/1250~1/phi^2 (7.3%). Null (N=200000 random slope-minus-1 6-point placements, same structural pair positions): p < 5e-6. Places Haramein hierarchy in Q(sqrt 5)=Z[phi] algebraic family on fixed-d diagonal (distinct from [217] b=e D_1, companion to [219] Fibonacci Resonance). Primary source: Documents/haramein_rsf/scale_unification_2008.pdf. Theory: docs/theory/QA_HARAMEIN_SCALING_DIAGONAL.md. Source: Will Dale + Claude 2026-04-13. Checks HSD_1+TABLE/FIXED_D/SEGMENTS/QUADRATIC/PHI_RATIOS/NULL/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "218_qa_haramein_scaling_diagonal_cert",
     "qa_haramein_scaling_diagonal_cert_v1", True),
    (220, "QA Madelung d-Ordering Cert family",
     _validate_madelung_d_ordering_cert_family,
     "Atomic subshells (n,l) identified as QA (b,e)=(n,l); d=b+e=n+l IS Madelung quantum. Aufbau = QA (d,-e) ascending sort, verified exactly through Janet-extended 36-subshell sequence (1s through 11s, zero mismatches). Selection rule: within-d antidiagonal (b,e)->(b+1,e-1) + between-d jump (b,0)->(ceil((b+2)/2), floor(b/2)); holds 35/35. Derived: subshell pop=4e+2; period-k pop=2*ceil((k+1)/2)^2={2,8,8,18,18,32,32,50,...} matching physical periodic table + Janet predictions; shell-n cumulative total=2*Sum k^2. Distinct QA class from [217] b=e D_1, [218] fixed-d hyperbola, [219] Fibonacci T-orbit: this is d-ordering walk across ALL d-classes, NOT a Q(sqrt 5) structure (polynomial-not-exponential). Madelung rule (Madelung 1936, Klechkowski 1962) was empirical aufbau heuristic; QA promotes it to structural consequence of A2 axiom (d=b+e). Does NOT claim derivation from Schrodinger or prediction of anomalies (Cr/Cu/lanthanides) — permissibility semantics live in SVP layer. Source: Will Dale + Claude 2026-04-13. Theory: docs/theory/QA_MADELUNG_D_ORDERING.md. Checks MAD_1+MAPPING/ORDER/RULE/POP/PERIODS/SHELL/CLASS/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "220_qa_madelung_d_ordering_cert",
     "qa_madelung_d_ordering_cert_v1", True),
    (221, "QA Nuclear Magic Spin-Extension Cert family",
     _validate_nuclear_magic_spin_extension_cert_family,
     "Extends QA with Dirac axiom D1 (sigma in {1,2}, j=l+(2sigma-3)/2, pop=2(e+sigma-1)). Identifies (b,e)=(n,l); HO shell N=2b-e-2. Fractional-1/2 promotion: sigma=2 AND b=e+1 AND l>=l* forces N_eff = N-1/2 (Dirac spin unit, not free parameter). Threshold l*=ceil(1/r) derived from physics input P1: r=alpha/hbar_omega in [1/3, 1/2), empirical nuclear ~0.3-0.4. Gives l*=3 by integer-ceiling. Magic criterion: N_eff in {0,1,2} OR half-integer. Cumulative sequence reproduces {2,8,20,28,50,82,126} exactly, zero tunable parameters beyond one discrete physical ratio in narrow window. Non-magic integer N_eff>=3 closures (40,70,112,168) are HO residues, smaller physical gaps. Explains why atomic [220] Madelung needs no extension (atomic r<<1/100). Will Dale + Claude 2026-04-13. Mayer 1950, Jensen 1950, Bohr-Mottelson 1969. Theory: docs/theory/QA_NUCLEAR_MAGIC_SPIN_EXTENSION.md. Checks NMS_1+D1/HO/PROMOTION/THRESHOLD/MAGIC/P1/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "221_qa_nuclear_magic_spin_extension_cert",
     "qa_nuclear_magic_spin_extension_cert_v1", True),
    (222, "QA Madelung Anomaly Boundary Cert family",
     _validate_madelung_anomaly_boundary_cert_family,
     "Every known atomic Madelung anomaly (20 total: Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag, La, Ce, Gd, Pt, Au, Ac, Th, Pa, U, Np, Cm, Lr) has |d(src)-d(dst)|<=1 under QA (b,e)=(n,l). 10 intra-class + 10 inter-class; 0 at |Δd|>=2. Null (uniform 2-subshell pairs from first 20 Madelung positions): 36.8% in zone; observed 100%; enrichment 2.71x; binomial p=2.1e-9. Necessary-not-sufficient: Ti/V/Mn/Fe/Co/Ni (d=4↔5) and Ta/W/Re/Os/Ir (d=6↔7) in zone follow Madelung. Falsifiable on any future |Δd|>=2 anomaly. Extends [220]; companion [221]. Exchange/relativistic mechanisms semantic-layer only. Sources: NIST ASD + Sato 2015 (Lr). Will Dale + Claude 2026-04-13. Theory: docs/theory/QA_MADELUNG_ANOMALY_BOUNDARY.md. Checks MAB_1+ANOMALIES/MAPPING/ZONE/COVERAGE/NULL/COUNTEREX/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "222_qa_madelung_anomaly_boundary_cert",
     "qa_madelung_anomaly_boundary_cert_v1", True),
    (231, "QA Hyper-Catalan Diagonal Correspondence Cert family",
     _validate_hyper_catalan_diagonal_cert_family,
     "Wildberger-Rubine hyper-Catalan multi-indices map to QA coordinates b=V_m-1, e=F_m with d=b+e=E_m exactly; Euler V_m-E_m+F_m=1 follows. Single-type m_k=n lies on sibling diagonal b=(k-1)e+1; OEIS A000108/A001764/A002293/A002294 match; no single-type k in [2,7], n in [0,9] sits on D_1. Checks HCD_1+EULER/OEIS/FUSS/SINGLE_DIAGONAL/D1_DISJOINT/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "231_qa_hyper_catalan_diagonal_cert",
     "qa_hyper_catalan_diagonal_cert_v1", True),
    (232, "QA UHG Diagonal Coincidence Cert family",
     _validate_uhg_diagonal_coincidence_cert_family,
     "At m=9 over {1,...,9}^2, UHG zero quadrance coincides exactly with QA gcd-reduced diagonal class: 64 zero-q pairs, 64 same-diagonal pairs, intersection 64, no counterexamples. Checks UDC_1+M/COUNTS/INTERSECTION/COUNTEREXAMPLES/WITNESS/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "232_qa_uhg_diagonal_coincidence_cert",
     "qa_uhg_diagonal_coincidence_cert_v1", True),
    (233, "QA UHG Orbit Diagonal Profile Cert family",
     _validate_uhg_orbit_diagonal_profile_cert_family,
     "At m=9, the QA T-step partitions 81 points into 1 singularity length 1, 2 satellite orbits length 4, and 6 cosmos orbits length 12; every non-singular D_1-containing orbit has two D_1 points summing to (9,9). Checks UODP_1+M/PARTITION/ORBIT_DATA/D1_PROFILE/COMPLEMENT/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "233_qa_uhg_orbit_diagonal_profile_cert",
     "qa_uhg_orbit_diagonal_profile_cert_v1", True),
    (234, "QA Chromogeometry Pythagorean Identity Cert family",
     _validate_chromogeometry_pythagorean_identity_cert_family,
     "Wildberger chromogeometry under QA coordinates Q_b=b*b+e*e, Q_r=b*b-e*e, Q_g=2*b*e satisfies Q_b square = Q_r square + Q_g square exhaustively on (b,e) in [1..19]^2 with zero failures; QA coordinate formulas and five Pythagorean triples verified. Checks CPI_1+SAMPLES/RANGE/FORMULAS/PLIMPTON/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "234_qa_chromogeometry_pythagorean_identity_cert",
     "qa_chromogeometry_pythagorean_identity_cert_v1", True),
    (223, "QA Experiment Protocol family",
     _validate_experiment_protocol_family_if_present,
     "Enforceable design contract for empirical QA studies. Validates QA_EXPERIMENT_PROTOCOL.v1 JSON against nine gates: schema validity, null design (generating process + held-fixed + permuted + independence), pre-registration, decision rules, observer projection, real-data status, source-mapping cross-reference, ablation declaration, reproducibility manifest. Harvested from MEMORY.md Hard Rules (Adversarial Testing 2026-04-01, No Stochastic 2026-04-02, QA Always Applies 2026-04-08, Primary Sources 2026-04-10) and hardened after 2026-04-13 design-gate audit. Linter gate EXP-1 requires EXPERIMENT_PROTOCOL_REF or sibling experiment_protocol.json on any script with statistical-test call sites. Authority: EXPERIMENT_AXIOMS_BLOCK.md Part A (E1-E6) + Part C (N1-N3). Source: Will Dale + Claude + Codex 2026-04-14. schema + validator + fixtures; self-test ok",
     "223_qa_experiment_protocol",
     "../qa_experiment_protocol", True),
    (224, "QA Benchmark Protocol family",
     _validate_benchmark_protocol_family_if_present,
     "Enforceable design contract for benchmarks comparing a QA method against baseline methods. Validates QA_BENCHMARK_PROTOCOL.v1 JSON against nine gates: schema validity, baseline parity, calibration provenance, framework inheritance, metrics, source-mapping cross-reference, SOTA/null-result baseline, ablation declaration, reproducibility manifest. Addresses 2026-04-13 cmap-tuned-for-finance silent failure, 2026-04-05 Bearden observer-framework port lesson, and benchmark overclaim risk where a null result is accepted without an explicit threshold or reason. Linter gate BENCH-1 requires BENCHMARK_PROTOCOL_REF or sibling benchmark_protocol.json on scripts importing sklearn baselines alongside metric calls or declaring baselines/methods structures. Authority: EXPERIMENT_AXIOMS_BLOCK.md Part B (B1-B4). Source: Will Dale + Claude + Codex 2026-04-14. schema + validator + fixtures; self-test ok",
     "224_qa_benchmark_protocol",
     "../qa_benchmark_protocol", True),
    (235, "QA Super Catalan Diagonal Cert family",
     _validate_super_catalan_diagonal_cert_family,
     "Limanta-Wildberger super Catalan S(m,n) maps to QA (b,e)=(m,n), so d=b+e gives (m+n)! = d!; S(b,b) b=0..10 matches OEIS A000984, swap symmetry and recurrence hold on [0..7]^2, and S(1,n)=2*Catalan(n) for n=0..9. Checks SCD_1+D1_A000984/SYMMETRY/RECURRENCE/CATALAN/QA_IDENT/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "235_qa_super_catalan_diagonal_cert",
     "qa_super_catalan_diagonal_cert_v1", True),
    (236, "QA Spread Polynomial Composition Cert family",
     _validate_spread_polynomial_composition_cert_family,
     "Goh-Wildberger spread polynomials satisfy S_n composed with S_m = S_{n*m}; exact SymPy composition verifies pairs (2,3), (3,2), (2,4), (4,3), (3,3), (2,5); S_2=4*s*(1-s) logistic map and integer closed forms S_2/S_3/S_4 match. Checks SPC_1+COMPOSITION/CLOSED_FORMS/LOGISTIC/TRIG_NOTE/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "236_qa_spread_polynomial_composition_cert",
     "qa_spread_polynomial_composition_cert_v1", True),
    (237, "QA 4D Diagonal Rule Cert family",
     _validate_4d_diagonal_rule_cert_family,
     "QA tuple (b,e,d,a) with d=b+e and a=b+2e equals b*(1,0,1,1)+e*(0,1,1,2); embedding verified on [-5..5]^2; Gram matrix [[3,3],[3,6]] det=9 equals QA canonical modulus m; two perpendicular QA-tuple witnesses satisfy Q1+Q2=Q3. Checks Q4D_1+EMBED/GRAM/MODULUS/DIAGONAL_RULE/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "237_qa_4d_diagonal_rule_cert",
     "qa_4d_diagonal_rule_cert_v1", True),
    (239, "QA Twelve Dihedral Orderings Cert family",
     _validate_twelve_dihedral_orderings_cert_family,
     "Five objects under D_5 give 5!/(2*5)=12 dihedral classes, each of size 10; exhaustive 120-permutation canonicalization verifies the listed reps and the 12-count links to G_2 roots, cuboctahedral S_1, and icosahedral vertices. Checks TDO_1+GROUP/PERMUTATIONS/CANONICAL_REPS/CLASS_COUNT/CLASS_SIZE/FORMULA/QA_CONNECTION/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "239_qa_twelve_dihedral_orderings_cert",
     "qa_twelve_dihedral_orderings_cert_v1", True),
    (240, "QA Diamond sl3 Irrep Dimension Cert family",
     _validate_diamond_sl3_irrep_dimension_cert_family,
     "Wildberger diamond sl3 bridge: dim pi[a,b]=(b+1)(e+1)(d+2)/2 under QA coordinates; verifies 22 standard entries, adjoint dim 8, quark/anti-quark triples, triangular column, and integer Heisenberg commutators. Checks DSI_1+DIM_FORMULA/ADJOINT/TRIANGULAR_COLUMN/QUARK_ANTIQUARK/HEISENBERG/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "240_qa_diamond_sl3_irrep_dimension_cert",
     "qa_diamond_sl3_irrep_dimension_cert_v1", True),
    (241, "QA Quadruple Coplanarity Cert family",
     _validate_quadruple_coplanarity_cert_family,
     "Every QA point (b,e,d=b+e) lies in the plane d-b-e=0 in R3; 30 triples have zero parallelepiped determinant and 30 quadruples plus Satellite #1 have zero Cayley-Menger determinant under blue/red/green chromogeometric quadrances. Checks QCO_1+PLANE_IDENTITY/PARALLELEPIPED_VOL/CM_4POINT_BLUE_RED_GREEN/CHROMO_COPLANARITY_PRESERVED/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "241_qa_quadruple_coplanarity_cert",
     "qa_quadruple_coplanarity_cert_v1", True),
    (242, "QA Neuberg Cubic F23 Cert family",
     _validate_neuberg_cubic_f23_cert_family,
     "Wildberger Neuberg finite-field bridge over F_23: E:y^2=x^3+x+1 has 27 affine points plus infinity; tangent-conic witnesses enumerate F_23 point sets and are identical or disjoint; spread witness is integer-polynomial and QA-compatible for char not 2 or 3. Checks NCF23_1+POINT_COUNT/TANGENT_CONIC_DICHOTOMY/SPREAD_POLYNOMIAL/QA_COMPAT/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "242_qa_neuberg_cubic_f23_cert",
     "qa_neuberg_cubic_f23_cert_v1", True),
    (244, "QA Mutation Game Root Lattice Cert family",
     _validate_mutation_game_root_lattice_cert_family,
     "Wildberger integer Mutation Game on the E_8 root lattice: Cartan determinant 1, BFS orbit closure of size 240, sign split 120+120, every root has norm 2 under G=2I-A, and Weyl involution/braid relations hold on integer populations. Checks MGR_1+CARTAN/BFS/ROOT_NORM/SIGN_SPLIT/INVOLUTION_BRAID/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "244_qa_mutation_game_root_lattice_cert",
     "qa_mutation_game_root_lattice_cert_v1", True),
    (245, "QA SL3 Hexagonal Ring Identity Cert family",
     _validate_sl3_hexagonal_ring_identity_cert_family,
     "Wildberger sl3 diamond follow-up: ring(a,b)=dim pi[a,b]-dim pi[a-1,b-1]=T_{d+1}+a*b under QA coordinates; verifies symbolic expansion, 196/196 grid entries, QA coordinate form, and known multiplicities. Checks SHR_1+ALGEBRAIC_EXPANSION/EXHAUSTIVE/QA_COORD_FORM/KNOWN_MULTIPLICITIES/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "245_qa_sl3_hexagonal_ring_identity_cert",
     "qa_sl3_hexagonal_ring_identity_cert_v1", True),
    (246, "QA Chromogeometric TQF Symmetry Cert family",
     _validate_chromogeometric_tqf_symmetry_cert_family,
     "Wildberger chromogeometry Triple Quad Formula sign symmetry TQF_r=TQF_g=-TQF_b; symbolic polynomial proof, TQF_b=4*area2*area2, deterministic 3000-triangle sample, and exhaustive C(81,3) collinearity invariant. Checks CTQF_1+SYMBOLIC_RB/SYMBOLIC_GB/SAMPLE_EXHAUSTIVE/FACTORED_BLUE/COLLINEARITY_INVARIANT/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "246_qa_chromogeometric_tqf_symmetry_cert",
     "qa_chromogeometric_tqf_symmetry_cert_v1", True),
    (247, "QA G-Equivariant CNN Structural Cert family",
     _validate_g_equivariant_cnn_structural_cert_family,
     "Closed-form Cohen-Welling rotation-index algebra: phi(b)=b mod n bijects {1,...,n} to Z/nZ; qa_step preserves addition under phi exhaustively at n=9 and n=24; n=9 generator partition gives singularity/satellite/cosmos counts 9/18/54 with zero exceptions; Eq. 10 lifting = observer IN, Eq. 11 G-correlation = QA-layer resonance, §6.3 coset pooling = observer OUT. Primary source: Cohen and Welling 2016. Checks GECS_1+BIJECT/COMPOSE/ORBIT/CORR/SRC/F; 1 PASS + 1 FAIL; self-test ok",
     "247_qa_g_equivariant_cnn_structural_cert",
     "qa_g_equivariant_cnn_structural_cert_v1", True),
    (248, "QA Formal Conjecture Resolution Cert family",
     _validate_formal_conjecture_resolution_cert_family,
     "QA-native record of conjecture-resolution attempts with typed obstruction (proved / formal_gap / qa_obstruction / generator_insufficient / inconclusive). QA contribution over Ju et al. (2026) Rethlas+Archon pipeline is the typed failure_mode layer that distinguishes formal_gap from qa_obstruction from generator_insufficient. Primary source: Ju, Gao, Jiang, Wu, Sun, Chen, Wang, Wang, Wang, He, Wu, Xiao, Liu, Dai, Dong (2026), 'Automated Conjecture Resolution with Formal Verification,' arXiv:2604.03789. Theory: docs/theory/QA_AUTOMATED_CONJECTURE_RESOLUTION.md. Checks FCR_1+SCHEMA/GENERATOR_SET/FAILURE_MODE/NT/VERDICT/WITNESS/LEAN4_STUB; 2 PASS (proved + formal_gap) + 1 FAIL (missing failure label); self-test ok",
     "248_qa_formal_conjecture_resolution_cert",
     "qa_formal_conjecture_resolution_cert_v1", True),
    (249, "QA E8 Embedding Orbit Classifier Cert family",
     _validate_e8_embedding_orbit_classifier_cert_family,
     "Canonical (b,e,d,a)→ℤ^8 embedding into the Wildberger E_8 root lattice [244]; m=9 T-orbit partition {1,8,24,24,24}, closed-form Q(E_diag)=2(b²+e²+d²+a²)-2(bd+ea+da), per-orbit min Q under E_diag = (8,16,28,72,162) is a 5-distinct complete T-orbit classifier, Q-multisets pairwise distinct under both E_diag and E_tri. Source: Wildberger 2020 + cert [244]. Checks E8E_1+CARTAN_LOAD/T_ORBITS/DIAG_FORMULA/DIAG_MIN_Q/DIAG_MULTISET/TRI_PROFILE/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "249_qa_e8_embedding_orbit_classifier_cert",
     "qa_e8_embedding_orbit_classifier_cert_v1", True),
    (250, "QA ADE Mutation Game Cert family",
     _validate_ade_mutation_game_cert_family,
     "Extends [244] to the full simply-laced ADE classification (A_5, D_5, E_6, E_7, E_8). Cartan determinants (6,4,3,2,1), Weyl orbit sizes (30,40,72,126,240) per Humphreys GTM 9 §9.3 Table 1, v^T G v = 2 exhaustive, equal positive/negative split with R-=-R+. Source: Wildberger 2020 + Humphreys 1972 + cert [244]. Checks ADE_1+CARTAN_DETS/BFS_SIZES/ROOT_NORM/SIGN_SPLIT/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "250_qa_ade_mutation_game_cert",
     "qa_ade_mutation_game_cert_v1", True),
    (251, "QA G2 Mutation Game Cert family",
     _validate_g2_mutation_game_cert_family,
     "First non-simply-laced Mutation Game cert, extending [244] and [250] to G_2 via directed edge counts A(0->1)=3, A(1->0)=1 encoding Cartan [[2,-1],[-3,2]]. BFS closes at 12 integer populations; sign split is 6 positive + 6 negative with R-=-R+; Humphreys §12.1 coordinate swap yields three short and three long positive roots under G_sr=[[2,-3],[-3,6]]; s0^2=s1^2=I; strict Coxeter order 6. Source: Wildberger 2020 + Humphreys 1972 §12.1 + theory docs/theory/QA_G2_MUTATION_GAME.md commit b86442f. Checks G2M_1/G2M_2/G2M_3/G2M_4/G2M_5/SRC/WITNESS/F; 1 PASS + 1 FAIL; self-test ok",
     "251_qa_g2_mutation_game_cert",
     "qa_g2_mutation_game_cert_v1", True),
    (225, "QA-KG Consistency Cert v3",
     _validate_kg_consistency_cert_v3,
     "Graph consistency under schema v3: epistemic fields + alias removal. KG1-KG10 gates. Supersedes v2 (frozen). Checks KG1/KG2/KG3/KG4/KG5/KG6/KG7/KG8/KG9/KG10; validator runs against live qa_kg.db",
     "225_qa_kg_consistency_cert_v3",
     "qa_kg_consistency_cert_v3", True),
    (252, "QA-KG Epistemic Fields Cert v1",
     _validate_kg_epistemic_fields_cert,
     "Authority/epistemic_status/method/source_locator/lifecycle_state correctness per Phase 1 QA-MEM. Allowed matrix enforced. Checks EF1/EF2/EF3/EF4/EF5/EF6; validator runs against live qa_kg.db",
     "252_qa_kg_epistemic_fields_cert_v1",
     "qa_kg_epistemic_fields_cert_v1", True),
    (227, "QA-KG Firewall Effective Cert v1",
     _validate_kg_firewall_effective_cert,
     "Phase 2 Theorem NT firewall effectiveness. DB-backed promoted-from, ledger staleness, broadcast provenance. Checks FE1/FE2/FE3/FE4/FE5/FE6; validator runs against live qa_kg.db",
     "227_qa_kg_firewall_effective_cert_v1",
     "qa_kg_firewall_effective_cert_v1", True),
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
    _ledger_results: Dict[str, Dict[str, str]] = {}  # {cert_id: {status, ts}}
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
                _ledger_results[str(fam_id)] = {"status": "PASS", "ts": utc_now_iso()}
            else:
                print(f"[{fam_id}] {label}: SKIPPED ({skip_reason})")
                _ledger_results[str(fam_id)] = {"status": "SKIP", "ts": utc_now_iso()}
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

    # --- Write _meta_ledger.json (Phase 2 — consumed by kg.promote staleness check) ---
    try:
        import subprocess as _ledger_sp
        _ledger_git_head = _ledger_sp.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=_repo_root, timeout=5,
        ).stdout.strip() or "UNKNOWN"
    except Exception:
        _ledger_git_head = "UNKNOWN"
    for _lk in _ledger_results:
        _ledger_results[_lk]["git_head"] = _ledger_git_head
    _ledger_path = os.path.join(base_dir, "_meta_ledger.json")
    with open(_ledger_path, "w", encoding="utf-8") as _lf:
        json.dump(_ledger_results, _lf, indent=2, sort_keys=True)
    print(f"\n[LEDGER] Wrote {len(_ledger_results)} entries to _meta_ledger.json "
          f"(git_head={_ledger_git_head[:10]}...)")

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
            capture_output=True, text=True, timeout=180,
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

    # --- Spine v1 compliance linter ---
    print("\n--- QA DYNAMICS SPINE V1 COMPLIANCE LINTER ---")
    _spine_gate_id = _demo_gate_id + 1
    if test_spine_v1_compliance():
        print(f"[{_spine_gate_id}] QA Dynamics Spine v1 compliance linter: PASS")
    else:
        print(f"[{_spine_gate_id}] QA Dynamics Spine v1 compliance linter: FAIL")
        sys.exit(1)
