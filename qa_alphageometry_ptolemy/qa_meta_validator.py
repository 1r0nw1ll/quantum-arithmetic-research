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


# Populate FAMILY_SWEEPS now that all validator functions are defined.
# To add a new family: add ONE entry here. That's it.
# Format: (id, label, validator_fn, pass_description, doc_slug)
FAMILY_SWEEPS = [
    (18, "QA Datastore family",
     _validate_datastore_family_if_present,
     "semantics + witness + counterexamples", "18_datastore"),
    (19, "QA Topology Resonance bundle",
     _validate_topology_bundle_if_present,
     "bundle manifest verified", "19_topology_resonance"),
    (20, "QA Datastore view family",
     _validate_datastore_view_family_if_present,
     "semantics + witness + counterexamples", "20_datastore_view"),
    (21, "QA A-RAG interface family",
     _validate_arag_family_if_present,
     "semantics + witness + counterexamples", "21_arag_interface"),
    (22, "QA ingest->view bridge family",
     _validate_ingest_view_bridge_family_if_present,
     "semantics + witness + counterexamples", "22_ingest_view_bridge"),
    (23, "QA ingestion family",
     _validate_ingest_family_if_present,
     "semantics + witness + counterexamples", "23_ingestion"),
    (24, "QA SVP-CMC family",
     _validate_svp_cmc_family_if_present,
     "ledger sanity + validator demo", "24_svp_cmc"),
    (26, "QA Competency Detection family",
     _validate_competency_family_if_present,
     "bundle + metrics recompute + fixtures", "26_competency_detection"),
    (27, "QA Elliptic Correspondence bundle",
     _validate_elliptic_bundle_if_present,
     "bundle manifest verified", "27_elliptic_correspondence"),
    (28, "QA Graph Structure bundle",
     _validate_graph_structure_bundle_if_present,
     "bundle manifest verified", "28_graph_structure"),
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
    for fam_id, label, validator_fn, pass_desc, _doc_slug in FAMILY_SWEEPS:
        print(f"\n--- {label.upper()} ---")
        try:
            skip_reason = validator_fn(base_dir)
            if skip_reason is None:
                print(f"[{fam_id}] {label}: PASS ({pass_desc})")
            else:
                print(f"[{fam_id}] {label}: SKIPPED ({skip_reason})")
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

    # --- Doc gate (derived from FAMILY_SWEEPS — no second list to maintain) ---
    print("\n--- HUMAN-TRACT DOC GATE ---")
    _doc_gate_pass = True
    _docs_dir = os.path.normpath(os.path.join(base_dir, "..", "docs", "families"))
    _readme_path = os.path.join(_docs_dir, "README.md")
    if not os.path.isdir(_docs_dir):
        print(f"[{FAMILY_SWEEPS[-1][0] + 2}] Doc gate: FAIL (docs/families/ directory missing)")
        _doc_gate_pass = False
    else:
        for fam_id, _label, _fn, _desc, doc_slug in FAMILY_SWEEPS:
            doc_file = f"{doc_slug}.md"
            if not os.path.exists(os.path.join(_docs_dir, doc_file)):
                print(f"  FAIL: missing docs/families/{doc_file} for family [{fam_id}]")
                _doc_gate_pass = False
        if os.path.exists(_readme_path):
            with open(_readme_path, "r", encoding="utf-8") as _rf:
                _readme_text = _rf.read()
            for fam_id, _label, _fn, _desc, doc_slug in FAMILY_SWEEPS:
                doc_file = f"{doc_slug}.md"
                if doc_file not in _readme_text:
                    print(f"  FAIL: docs/families/README.md missing link to {doc_file}")
                    _doc_gate_pass = False
        else:
            print(f"  FAIL: docs/families/README.md missing")
            _doc_gate_pass = False
    _gate_id = FAMILY_SWEEPS[-1][0] + 2
    if _doc_gate_pass:
        print(f"[{_gate_id}] Human-tract doc gate: PASS ({len(FAMILY_SWEEPS)} families documented)")
    else:
        print(f"[{_gate_id}] Human-tract doc gate: FAIL")
        sys.exit(1)
