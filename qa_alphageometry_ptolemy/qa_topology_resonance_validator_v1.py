#!/usr/bin/env python3
"""
qa_topology_resonance_validator_v1.py

Strict validator for QA topology resonance certificates.

Validation levels:
- Level 1 (Schema): required fields and core shape checks
- Level 2 (Consistency): arithmetic and invariant consistency checks
- Level 3 (Recompute): replay hook placeholder

Usage:
    python qa_topology_resonance_validator_v1.py examples/topology/topology_resonance_success.json
    python qa_topology_resonance_validator_v1.py --demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional


class ValidationLevel(Enum):
    SCHEMA = 1
    CONSISTENCY = 2
    RECOMPUTE = 3


class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    check_name: str
    level: ValidationLevel
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    certificate_id: str
    schema: str
    results: List[ValidationResult]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == ValidationStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == ValidationStatus.FAILED)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == ValidationStatus.WARNING)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        lines = [
            f"Certificate: {self.certificate_id}",
            f"Schema: {self.schema}",
            "",
            f"PASS: {self.passed}",
            f"FAIL: {self.failed}",
            f"WARN: {self.warnings}",
        ]
        if self.all_passed:
            lines.append("")
            lines.append("ALL CHECKS PASSED")
        else:
            lines.append("")
            lines.append("VALIDATION FAILED")
            lines.append("")
            lines.append("Failures:")
            for r in self.results:
                if r.status == ValidationStatus.FAILED:
                    lines.append(f"  - {r.check_name}: {r.message}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "schema": self.schema,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "all_passed": self.all_passed,
            "results": [
                {
                    "check_name": r.check_name,
                    "level": r.level.name.lower(),
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


class TopologyResonanceValidator:
    """Strict validator for QA_TOPOLOGY_RESONANCE_CERT.v1."""

    REQUIRED_FIELDS = [
        "certificate_id",
        "certificate_type",
        "timestamp",
        "version",
        "schema",
        "success",
        "generator_set",
    ]
    SUCCESS_FIELDS = ["topology_witness", "phase_witness", "invariants"]
    FAILURE_FIELDS = ["failure_mode", "failure_witness"]

    VALID_GENERATORS = {"sigma", "mu", "lambda2", "nu"}
    VALID_FAILURE_MODES = {
        "phase_break",
        "scc_drop",
        "resonance_below_threshold",
        "packet_drift",
        "invalid_generator",
    }

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.results: List[ValidationResult] = []

    def _add_result(
        self,
        check_name: str,
        level: ValidationLevel,
        status: ValidationStatus,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.results.append(
            ValidationResult(
                check_name=check_name,
                level=level,
                status=status,
                message=message,
                details=details,
            )
        )

    def _parse_scalar(self, value: Any) -> Fraction:
        if isinstance(value, bool):
            raise ValueError(f"Bool is not a scalar: {value}")
        if isinstance(value, (int, Fraction)):
            return Fraction(value)
        if isinstance(value, str):
            return Fraction(value)
        if isinstance(value, float):
            if self.strict:
                raise ValueError(f"Float not allowed in strict mode: {value}")
            return Fraction(value).limit_denominator(10**9)
        raise ValueError(f"Cannot parse scalar: {value}")

    def validate_schema(self, cert: Dict[str, Any]) -> None:
        for field in self.REQUIRED_FIELDS:
            if field not in cert:
                self._add_result(
                    f"schema.required.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing required field: {field}",
                )
            else:
                self._add_result(
                    f"schema.required.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"Field present: {field}",
                )

        if cert.get("schema") == "QA_TOPOLOGY_RESONANCE_CERT.v1":
            self._add_result(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                "Valid schema: QA_TOPOLOGY_RESONANCE_CERT.v1",
            )
        else:
            self._add_result(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"Unexpected schema: {cert.get('schema')}",
            )

        if cert.get("certificate_type") == "TOPOLOGY_RESONANCE_CERT":
            self._add_result(
                "schema.certificate_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                "Valid certificate_type: TOPOLOGY_RESONANCE_CERT",
            )
        else:
            self._add_result(
                "schema.certificate_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"Unexpected certificate_type: {cert.get('certificate_type')}",
            )

        generators = cert.get("generator_set")
        if not isinstance(generators, list) or len(generators) == 0:
            self._add_result(
                "schema.generator_set",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                "generator_set must be a non-empty list",
            )
        else:
            unknown = [g for g in generators if g not in self.VALID_GENERATORS]
            if unknown:
                self._add_result(
                    "schema.generator_set.unknown",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Unknown generator(s): {unknown}",
                )
            else:
                self._add_result(
                    "schema.generator_set.known",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"All generators recognized: {generators}",
                )

        success = cert.get("success", True)
        if success:
            self._validate_success_schema(cert)
        else:
            self._validate_failure_schema(cert)

    def _validate_success_schema(self, cert: Dict[str, Any]) -> None:
        for field in self.SUCCESS_FIELDS:
            if field not in cert or cert[field] is None:
                self._add_result(
                    f"schema.success.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Success certificate missing: {field}",
                )
            else:
                self._add_result(
                    f"schema.success.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"Success field present: {field}",
                )

        topo = cert.get("topology_witness") or {}
        phase = cert.get("phase_witness") or {}
        inv = cert.get("invariants") or {}

        for field in (
            "scc_count_before",
            "scc_count_after",
            "betti_0_before",
            "betti_0_after",
            "resonance_score",
            "resonance_threshold",
            "resonance_certified",
        ):
            if field not in topo:
                self._add_result(
                    f"schema.topology_witness.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing topology_witness field: {field}",
                )

        for field in (
            "phase_24_before",
            "phase_24_after",
            "phase_9_before",
            "phase_9_after",
            "phase_preserved",
        ):
            if field not in phase:
                self._add_result(
                    f"schema.phase_witness.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing phase_witness field: {field}",
                )

        for field in (
            "scc_monotone_non_decreasing",
            "phase_lock",
            "packet_conservation",
            "no_reduction_axiom",
            "connected_component_first_class",
        ):
            if field not in inv:
                self._add_result(
                    f"schema.invariants.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing invariant field: {field}",
                )

    def _validate_failure_schema(self, cert: Dict[str, Any]) -> None:
        for field in self.FAILURE_FIELDS:
            if field not in cert or cert[field] is None:
                self._add_result(
                    f"schema.failure.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Failure certificate missing: {field}",
                )
            else:
                self._add_result(
                    f"schema.failure.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"Failure field present: {field}",
                )

        failure_mode = cert.get("failure_mode")
        if failure_mode not in self.VALID_FAILURE_MODES:
            self._add_result(
                "schema.failure.mode_valid",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"Unknown failure mode: {failure_mode}",
            )
        else:
            self._add_result(
                "schema.failure.mode_valid",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                f"Known failure mode: {failure_mode}",
            )

    def validate_consistency(self, cert: Dict[str, Any]) -> None:
        if not cert.get("success", True):
            self._add_result(
                "consistency.skip_failure",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.SKIPPED,
                "Skipping consistency checks for failure certificate",
            )
            return

        topo = cert.get("topology_witness", {})
        phase = cert.get("phase_witness", {})
        inv = cert.get("invariants", {})

        try:
            scc_before = int(topo["scc_count_before"])
            scc_after = int(topo["scc_count_after"])
            should_monotone = scc_after >= scc_before
            claimed_monotone = bool(inv["scc_monotone_non_decreasing"])
            if should_monotone == claimed_monotone:
                self._add_result(
                    "consistency.scc.monotonicity",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"SCC monotonicity verified: {scc_before} -> {scc_after}",
                )
            else:
                self._add_result(
                    "consistency.scc.monotonicity",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"SCC monotonicity mismatch: expected {should_monotone}, claimed {claimed_monotone}",
                )
        except Exception as e:
            self._add_result(
                "consistency.scc.monotonicity",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"SCC monotonicity check error: {e}",
            )

        try:
            b0_before = int(topo["betti_0_before"])
            b0_after = int(topo["betti_0_after"])
            if b0_before == int(topo["scc_count_before"]) and b0_after == int(topo["scc_count_after"]):
                self._add_result(
                    "consistency.topology.betti0_scc",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    "betti_0 and SCC counts are aligned",
                )
            else:
                self._add_result(
                    "consistency.topology.betti0_scc",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    "betti_0 and SCC counts diverge",
                )
        except Exception as e:
            self._add_result(
                "consistency.topology.betti0_scc",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Betti/SCC check error: {e}",
            )

        try:
            p24_before = int(phase["phase_24_before"])
            p24_after = int(phase["phase_24_after"])
            p9_before = int(phase["phase_9_before"])
            p9_after = int(phase["phase_9_after"])

            if 0 <= p24_before < 24 and 0 <= p24_after < 24:
                self._add_result(
                    "consistency.phase.mod24_range",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    "phase_24 range check passed",
                )
            else:
                self._add_result(
                    "consistency.phase.mod24_range",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    "phase_24 values must satisfy 0 <= value < 24",
                )

            if 0 <= p9_before < 9 and 0 <= p9_after < 9:
                self._add_result(
                    "consistency.phase.mod9_range",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    "phase_9 range check passed",
                )
            else:
                self._add_result(
                    "consistency.phase.mod9_range",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    "phase_9 values must satisfy 0 <= value < 9",
                )

            should_preserve = (p24_before == p24_after) and (p9_before == p9_after)
            claimed_preserve = bool(phase["phase_preserved"])
            if should_preserve == claimed_preserve:
                self._add_result(
                    "consistency.phase.preservation",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Phase preservation verified: {claimed_preserve}",
                )
            else:
                self._add_result(
                    "consistency.phase.preservation",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Phase preservation mismatch: expected {should_preserve}, claimed {claimed_preserve}",
                )
        except Exception as e:
            self._add_result(
                "consistency.phase",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Phase consistency check error: {e}",
            )

        try:
            score = self._parse_scalar(topo["resonance_score"])
            threshold = self._parse_scalar(topo["resonance_threshold"])
            should_certify = score >= threshold
            claimed_certify = bool(topo["resonance_certified"])
            if should_certify == claimed_certify:
                self._add_result(
                    "consistency.resonance.threshold",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Resonance certification verified: {score} >= {threshold}",
                )
            else:
                self._add_result(
                    "consistency.resonance.threshold",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Resonance mismatch: expected {should_certify}, claimed {claimed_certify}",
                )
        except Exception as e:
            self._add_result(
                "consistency.resonance.threshold",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Resonance check error: {e}",
            )

        for k in ("packet_conservation", "no_reduction_axiom", "connected_component_first_class"):
            if inv.get(k) is True:
                self._add_result(
                    f"consistency.invariants.{k}",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Invariant holds: {k}",
                )
            else:
                self._add_result(
                    f"consistency.invariants.{k}",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Success certificate requires invariant {k}=true",
                )

    def validate_recompute(self, cert: Dict[str, Any], data_path: Optional[Path] = None) -> None:
        if not cert.get("success", True):
            self._add_result(
                "recompute.skip_failure",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.SKIPPED,
                "Skipping recompute checks for failure certificate",
            )
            return

        recompute = cert.get("recompute_inputs")
        if not isinstance(recompute, dict):
            self._add_result(
                "recompute.inputs.present",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                "Success certificate must include recompute_inputs object",
            )
            return

        trace_schema = recompute.get("trace_schema")
        if trace_schema == "QA_TOPOLOGY_TRACE.v1":
            self._add_result(
                "recompute.trace_schema",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED,
                "trace_schema verified: QA_TOPOLOGY_TRACE.v1",
            )
        else:
            self._add_result(
                "recompute.trace_schema",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                f"Unexpected trace_schema: {trace_schema}",
            )
            return

        initial = recompute.get("initial_state")
        trace = recompute.get("transition_trace")
        if not isinstance(initial, dict):
            self._add_result(
                "recompute.initial_state",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                "recompute_inputs.initial_state must be an object",
            )
            return
        if not isinstance(trace, list) or len(trace) == 0:
            self._add_result(
                "recompute.transition_trace",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                "recompute_inputs.transition_trace must be a non-empty list",
            )
            return

        claimed_trace_digest = recompute.get("trace_digest")
        canonical_trace = json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        computed_trace_digest = "sha256:" + hashlib.sha256(canonical_trace.encode("utf-8")).hexdigest()
        self._add_result(
            "recompute.trace_digest",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.PASSED if claimed_trace_digest == computed_trace_digest else ValidationStatus.FAILED,
            f"trace_digest claimed={claimed_trace_digest} computed={computed_trace_digest}",
        )

        try:
            initial_scc = int(initial["scc_count"])
            initial_p24 = int(initial["phase_24"])
            initial_p9 = int(initial["phase_9"])
            resonance = self._parse_scalar(initial["resonance_score"])
            self._add_result(
                "recompute.initial_state.parse",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED,
                "Initial state parsed",
            )
        except Exception as e:
            self._add_result(
                "recompute.initial_state.parse",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                f"Failed to parse initial_state: {e}",
            )
            return

        if not (0 <= initial_p24 < 24 and 0 <= initial_p9 < 9):
            self._add_result(
                "recompute.initial_state.phase_range",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                "initial_state phase out of range",
            )
            return
        self._add_result(
            "recompute.initial_state.phase_range",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.PASSED,
            "Initial phase ranges verified",
        )

        allowed = set(cert.get("generator_set", []))
        previous_step = 0
        previous_scc = initial_scc
        scc_monotone_trace = True
        final_scc = initial_scc
        final_p24 = initial_p24
        final_p9 = initial_p9

        for i, step in enumerate(trace):
            if not isinstance(step, dict):
                self._add_result(
                    f"recompute.trace[{i}].shape",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.FAILED,
                    "Trace step must be an object",
                )
                return

            try:
                step_index = int(step["step_index"])
                generator = step["generator"]
                scc = int(step["scc_count"])
                p24 = int(step["phase_24"])
                p9 = int(step["phase_9"])
                delta = self._parse_scalar(step["resonance_delta"])
            except Exception as e:
                self._add_result(
                    f"recompute.trace[{i}].parse",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.FAILED,
                    f"Trace parse error: {e}",
                )
                return

            if step_index == previous_step + 1:
                self._add_result(
                    f"recompute.trace[{i}].step_index",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.PASSED,
                    f"Step index contiguous: {step_index}",
                )
            else:
                self._add_result(
                    f"recompute.trace[{i}].step_index",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.FAILED,
                    f"Non-contiguous step index: got {step_index}, expected {previous_step + 1}",
                )
                return

            if generator in allowed:
                self._add_result(
                    f"recompute.trace[{i}].generator",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.PASSED,
                    f"Generator allowed: {generator}",
                )
            else:
                self._add_result(
                    f"recompute.trace[{i}].generator",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.FAILED,
                    f"Generator not in generator_set: {generator}",
                )
                return

            if 0 <= p24 < 24 and 0 <= p9 < 9:
                self._add_result(
                    f"recompute.trace[{i}].phase_range",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.PASSED,
                    "Phase values in range",
                )
            else:
                self._add_result(
                    f"recompute.trace[{i}].phase_range",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.FAILED,
                    f"Phase out of range: phase_24={p24}, phase_9={p9}",
                )
                return

            if scc < previous_scc:
                scc_monotone_trace = False

            resonance += delta
            previous_step = step_index
            previous_scc = scc
            final_scc = scc
            final_p24 = p24
            final_p9 = p9

        topo = cert.get("topology_witness", {})
        phase = cert.get("phase_witness", {})
        inv = cert.get("invariants", {})

        try:
            cert_scc_before = int(topo.get("scc_count_before", -1))
            cert_scc_after = int(topo.get("scc_count_after", -1))
            cert_p24_before = int(phase.get("phase_24_before", -1))
            cert_p9_before = int(phase.get("phase_9_before", -1))
            cert_p24_after = int(phase.get("phase_24_after", -1))
            cert_p9_after = int(phase.get("phase_9_after", -1))
        except Exception as e:
            self._add_result(
                "recompute.match.parse_declared_fields",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                f"Failed to parse declared witness fields: {e}",
            )
            return

        checks = [
            ("recompute.match.scc_before", cert_scc_before == initial_scc,
             f"scc_before matches initial trace value {initial_scc}"),
            ("recompute.match.scc_after", cert_scc_after == final_scc,
             f"scc_after matches trace terminal value {final_scc}"),
            ("recompute.match.phase_before",
             cert_p24_before == initial_p24 and cert_p9_before == initial_p9,
             f"phase_before matches initial trace value ({initial_p24}, {initial_p9})"),
            ("recompute.match.phase_after",
             cert_p24_after == final_p24 and cert_p9_after == final_p9,
             f"phase_after matches trace terminal value ({final_p24}, {final_p9})"),
        ]

        for check_name, ok, msg in checks:
            self._add_result(
                check_name,
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
                msg if ok else f"{msg} (mismatch)",
            )

        try:
            cert_resonance = self._parse_scalar(topo["resonance_score"])
            resonance_match = resonance == cert_resonance
            self._add_result(
                "recompute.match.resonance_score",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED if resonance_match else ValidationStatus.FAILED,
                f"Recomputed resonance={resonance}, certificate={cert_resonance}",
            )
        except Exception as e:
            self._add_result(
                "recompute.match.resonance_score",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                f"Failed to parse certificate resonance score: {e}",
            )

        claimed_monotone = bool(inv.get("scc_monotone_non_decreasing"))
        self._add_result(
            "recompute.match.scc_monotone_invariant",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.PASSED if claimed_monotone == scc_monotone_trace else ValidationStatus.FAILED,
            f"Trace monotone={scc_monotone_trace}, claimed={claimed_monotone}",
        )

        trace_phase_preserved = (initial_p24 == final_p24) and (initial_p9 == final_p9)
        claimed_phase_preserved = bool(phase.get("phase_preserved"))
        self._add_result(
            "recompute.match.phase_preserved",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.PASSED if claimed_phase_preserved == trace_phase_preserved else ValidationStatus.FAILED,
            f"Trace phase_preserved={trace_phase_preserved}, claimed={claimed_phase_preserved}",
        )

    def validate(
        self,
        cert: Dict[str, Any],
        level: ValidationLevel = ValidationLevel.CONSISTENCY,
        data_path: Optional[Path] = None,
    ) -> ValidationReport:
        self.results = []
        self.validate_schema(cert)
        if level.value >= ValidationLevel.CONSISTENCY.value:
            self.validate_consistency(cert)
        if level.value >= ValidationLevel.RECOMPUTE.value:
            self.validate_recompute(cert, data_path)
        return ValidationReport(
            certificate_id=cert.get("certificate_id", "unknown"),
            schema=cert.get("schema", "unknown"),
            results=self.results,
        )


def create_demo_success_certificate() -> Dict[str, Any]:
    return {
        "certificate_id": "demo_topology_resonance_001",
        "certificate_type": "TOPOLOGY_RESONANCE_CERT",
        "timestamp": "2026-02-07T00:00:00Z",
        "version": "1.0.0",
        "schema": "QA_TOPOLOGY_RESONANCE_CERT.v1",
        "success": True,
        "generator_set": ["sigma", "mu", "lambda2", "nu"],
        "state_descriptor": {
            "state_family": "caps_n_n",
            "n": 24,
            "seed": 17,
        },
        "topology_witness": {
            "scc_count_before": 3,
            "scc_count_after": 5,
            "betti_0_before": 3,
            "betti_0_after": 5,
            "betti_1_before": 2,
            "betti_1_after": 4,
            "resonance_score": "7/10",
            "resonance_threshold": "3/5",
            "resonance_certified": True,
        },
        "phase_witness": {
            "phase_24_before": 11,
            "phase_24_after": 11,
            "phase_9_before": 4,
            "phase_9_after": 4,
            "phase_preserved": True,
        },
        "invariants": {
            "scc_monotone_non_decreasing": True,
            "phase_lock": True,
            "packet_conservation": True,
            "no_reduction_axiom": True,
            "connected_component_first_class": True,
        },
        "recompute_inputs": {
            "trace_schema": "QA_TOPOLOGY_TRACE.v1",
            "trace_digest": "sha256:9a7814595e9712c6f197bf0d3559db170256b0ea4fee8851ea362b75d589ae3c",
            "scc_algorithm": "tarjan_v1",
            "initial_state": {
                "scc_count": 3,
                "phase_24": 11,
                "phase_9": 4,
                "resonance_score": "1/2",
            },
            "transition_trace": [
                {
                    "step_index": 1,
                    "generator": "sigma",
                    "scc_count": 3,
                    "phase_24": 11,
                    "phase_9": 4,
                    "resonance_delta": "1/20",
                },
                {
                    "step_index": 2,
                    "generator": "mu",
                    "scc_count": 4,
                    "phase_24": 11,
                    "phase_9": 4,
                    "resonance_delta": "1/20",
                },
                {
                    "step_index": 3,
                    "generator": "lambda2",
                    "scc_count": 4,
                    "phase_24": 11,
                    "phase_9": 4,
                    "resonance_delta": "1/20",
                },
                {
                    "step_index": 4,
                    "generator": "nu",
                    "scc_count": 5,
                    "phase_24": 11,
                    "phase_9": 4,
                    "resonance_delta": "1/20",
                },
            ],
        },
        "qa_interpretation": {
            "success_type": "TOPOLOGY_RESONANCE_CERTIFIED",
            "lattice_position": "scc_monotone ∧ phase_lock ∧ resonance_certified",
            "note": "Caps(N,N) transition preserves phase and improves component reachability",
        },
    }


def create_demo_failure_certificate() -> Dict[str, Any]:
    return {
        "certificate_id": "demo_topology_resonance_failure_001",
        "certificate_type": "TOPOLOGY_RESONANCE_CERT",
        "timestamp": "2026-02-07T00:00:00Z",
        "version": "1.0.0",
        "schema": "QA_TOPOLOGY_RESONANCE_CERT.v1",
        "success": False,
        "generator_set": ["sigma", "mu", "lambda2", "nu"],
        "failure_mode": "phase_break",
        "failure_witness": {
            "reason": "phase lock broken after lambda2 transition",
            "expected": {"phase_24": 11, "phase_9": 4},
            "observed": {"phase_24": 12, "phase_9": 4},
            "first_bad_step": 3,
            "remediation": [
                "Constrain lambda2 operator to preserve residue class",
                "Re-run path search with sigma-mu-only prefix",
            ],
        },
        "qa_interpretation": {
            "failure_class": "PHASE_BREAK",
            "obstruction_type": "invariant_violation",
            "note": "Witness demonstrates phase non-preservation",
        },
    }


def _level_from_arg(level_arg: str) -> ValidationLevel:
    level_map = {
        "schema": ValidationLevel.SCHEMA,
        "consistency": ValidationLevel.CONSISTENCY,
        "recompute": ValidationLevel.RECOMPUTE,
    }
    return level_map[level_arg]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA topology resonance certificates")
    parser.add_argument("certificate", nargs="?", help="Path to certificate JSON")
    parser.add_argument("--demo", action="store_true", help="Run demo success certificate")
    parser.add_argument("--demo-failure", action="store_true", help="Run demo failure certificate")
    parser.add_argument(
        "--level",
        default="consistency",
        choices=("schema", "consistency", "recompute"),
        help="Validation depth",
    )
    parser.add_argument("--data", type=Path, help="Optional raw data path for recompute level")
    parser.add_argument("--json", action="store_true", help="Emit report as JSON")
    args = parser.parse_args()

    if args.demo:
        cert = create_demo_success_certificate()
    elif args.demo_failure:
        cert = create_demo_failure_certificate()
    elif args.certificate:
        with open(args.certificate) as f:
            cert = json.load(f)
    else:
        parser.error("Certificate path required (or use --demo / --demo-failure)")

    validator = TopologyResonanceValidator(strict=True)
    report = validator.validate(
        cert,
        level=_level_from_arg(args.level),
        data_path=args.data,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
