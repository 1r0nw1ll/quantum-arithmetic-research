#!/usr/bin/env python3
"""
qa_graph_structure_validator_v1.py

Strict validator for QA graph structure certificates.

Validation levels:
- Level 1 (Schema): required fields and shape checks
- Level 2 (Consistency): metric/invariant consistency checks
- Level 3 (Recompute): replay checks from paired traces

Usage:
    python qa_graph_structure_validator_v1.py examples/graph_structure/graph_structure_success.json
    python qa_graph_structure_validator_v1.py --demo
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
from typing import Any, Dict, List, Optional, Tuple


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


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_trace_digest(baseline_trace: List[Dict[str, Any]], qa_trace: List[Dict[str, Any]]) -> str:
    payload = {"baseline_trace": baseline_trace, "qa_trace": qa_trace}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class GraphStructureValidator:
    """Strict validator for QA_GRAPH_STRUCTURE_CERT.v1."""

    REQUIRED_FIELDS = [
        "certificate_id",
        "certificate_type",
        "timestamp",
        "version",
        "schema",
        "success",
        "generator_set",
    ]
    SUCCESS_FIELDS = ["graph_context", "metric_witness", "phase_witness", "invariants", "recompute_inputs"]
    FAILURE_FIELDS = ["failure_mode", "failure_witness"]
    METRIC_KEYS = ("ari", "nmi", "modularity", "purity")

    VALID_GENERATORS = {
        "sigma_feat_extract",
        "sigma_qa_embed",
        "sigma_cluster",
        "sigma_eval",
        "sigma_phase_analyze",
    }
    VALID_FAILURE_MODES = {"out_of_bounds", "invariant", "phase_violation", "parity", "reduction"}

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
            raise ValueError(f"bool is not a scalar: {value}")
        if isinstance(value, (int, Fraction)):
            return Fraction(value)
        if isinstance(value, str):
            return Fraction(value)
        if isinstance(value, float):
            if self.strict:
                raise ValueError(f"float not allowed in strict mode: {value}")
            return Fraction(value).limit_denominator(10**9)
        raise ValueError(f"cannot parse scalar: {value}")

    def _parse_metric_object(self, metrics: Any, where: str) -> Optional[Dict[str, Fraction]]:
        if not isinstance(metrics, dict):
            self._add_result(
                f"{where}.shape",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                "metrics object must be a dict",
            )
            return None
        parsed: Dict[str, Fraction] = {}
        for k in self.METRIC_KEYS:
            if k not in metrics:
                self._add_result(
                    f"{where}.{k}",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"missing metric: {k}",
                )
                return None
            try:
                parsed[k] = self._parse_scalar(metrics[k])
            except Exception as e:
                self._add_result(
                    f"{where}.{k}",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"invalid scalar for {k}: {e}",
                )
                return None
        self._add_result(
            f"{where}.parse",
            ValidationLevel.CONSISTENCY,
            ValidationStatus.PASSED,
            f"parsed metrics: {', '.join(self.METRIC_KEYS)}",
        )
        return parsed

    def validate_schema(self, cert: Dict[str, Any]) -> None:
        for field in self.REQUIRED_FIELDS:
            if field not in cert:
                self._add_result(
                    f"schema.required.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"missing required field: {field}",
                )
            else:
                self._add_result(
                    f"schema.required.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"field present: {field}",
                )

        if cert.get("schema") == "QA_GRAPH_STRUCTURE_CERT.v1":
            self._add_result(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                "valid schema: QA_GRAPH_STRUCTURE_CERT.v1",
            )
        else:
            self._add_result(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"unexpected schema: {cert.get('schema')}",
            )

        if cert.get("certificate_type") == "GRAPH_STRUCTURE_CERT":
            self._add_result(
                "schema.certificate_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                "valid certificate_type: GRAPH_STRUCTURE_CERT",
            )
        else:
            self._add_result(
                "schema.certificate_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"unexpected certificate_type: {cert.get('certificate_type')}",
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
                    f"unknown generator(s): {unknown}",
                )
            else:
                self._add_result(
                    "schema.generator_set.known",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"all generators recognized: {generators}",
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
                    f"success certificate missing: {field}",
                )
            else:
                self._add_result(
                    f"schema.success.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"success field present: {field}",
                )

    def _validate_failure_schema(self, cert: Dict[str, Any]) -> None:
        for field in self.FAILURE_FIELDS:
            if field not in cert:
                self._add_result(
                    f"schema.failure.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"failure certificate missing: {field}",
                )
            else:
                self._add_result(
                    f"schema.failure.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"failure field present: {field}",
                )

        mode = cert.get("failure_mode")
        if mode in self.VALID_FAILURE_MODES:
            self._add_result(
                "schema.failure.mode",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                f"recognized failure_mode: {mode}",
            )
        else:
            self._add_result(
                "schema.failure.mode",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"unknown failure_mode: {mode}",
            )

    def validate_consistency(self, cert: Dict[str, Any]) -> None:
        success = cert.get("success", True)
        if not success:
            mode = cert.get("failure_mode")
            witness = cert.get("failure_witness")
            mode_ok = mode in self.VALID_FAILURE_MODES
            witness_ok = isinstance(witness, dict)
            self._add_result(
                "consistency.failure_contract",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED if (mode_ok and witness_ok) else ValidationStatus.FAILED,
                "failure contract verified" if (mode_ok and witness_ok) else "failure contract invalid",
            )
            return

        graph_context = cert.get("graph_context")
        metric_witness = cert.get("metric_witness")
        phase = cert.get("phase_witness")
        inv = cert.get("invariants")
        recompute = cert.get("recompute_inputs")
        if not all(isinstance(x, dict) for x in (graph_context, metric_witness, phase, inv, recompute)):
            self._add_result(
                "consistency.required_objects",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                "success certificate requires object fields: graph_context/metric_witness/phase_witness/invariants/recompute_inputs",
            )
            return

        # Graph context bounds
        try:
            node_count = int(graph_context["node_count"])
            edge_count = int(graph_context["edge_count"])
            community_count = int(graph_context["community_count"])
            split_seed = int(graph_context["split_seed"])
            clustering_seed = int(graph_context["clustering_seed"])
            algorithm = str(graph_context["algorithm"])
            dataset_id = str(graph_context["dataset_id"])
        except Exception as e:
            self._add_result(
                "consistency.graph_context.parse",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"failed to parse graph_context: {e}",
            )
            return

        bounds_ok = node_count > 0 and edge_count >= 0 and 1 <= community_count <= node_count and split_seed >= 0 and clustering_seed >= 0
        self._add_result(
            "consistency.graph_context.bounds",
            ValidationLevel.CONSISTENCY,
            ValidationStatus.PASSED if bounds_ok else ValidationStatus.FAILED,
            f"node_count={node_count}, edge_count={edge_count}, community_count={community_count}",
        )

        qa_metrics = self._parse_metric_object(metric_witness.get("qa_metrics"), "consistency.metric_witness.qa_metrics")
        baseline_metrics = self._parse_metric_object(
            metric_witness.get("baseline_metrics"), "consistency.metric_witness.baseline_metrics"
        )
        delta_metrics = self._parse_metric_object(metric_witness.get("delta_metrics"), "consistency.metric_witness.delta_metrics")
        if qa_metrics is None or baseline_metrics is None or delta_metrics is None:
            return

        for metric_key in self.METRIC_KEYS:
            expected_delta = qa_metrics[metric_key] - baseline_metrics[metric_key]
            declared_delta = delta_metrics[metric_key]
            self._add_result(
                f"consistency.metric_delta.{metric_key}",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED if expected_delta == declared_delta else ValidationStatus.FAILED,
                f"delta({metric_key}) expected={expected_delta} declared={declared_delta}",
            )

        delta_non_negative_any = metric_witness.get("delta_non_negative_any")
        if isinstance(delta_non_negative_any, bool):
            computed_any = (delta_metrics["ari"] >= 0) or (delta_metrics["modularity"] >= 0)
            self._add_result(
                "consistency.metric_delta.non_negative_any",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED if computed_any == delta_non_negative_any else ValidationStatus.FAILED,
                f"computed={computed_any}, declared={delta_non_negative_any}",
            )
        else:
            self._add_result(
                "consistency.metric_delta.non_negative_any",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                "metric_witness.delta_non_negative_any must be bool",
            )

        try:
            p24_baseline = int(phase["phase_24_baseline"])
            p24_qa = int(phase["phase_24_qa"])
            p9_baseline = int(phase["phase_9_baseline"])
            p9_qa = int(phase["phase_9_qa"])
            phase_preserved = bool(phase["phase_preserved"])
        except Exception as e:
            self._add_result(
                "consistency.phase.parse",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"phase parse error: {e}",
            )
            return

        range_ok = (0 <= p24_baseline < 24 and 0 <= p24_qa < 24 and 0 <= p9_baseline < 9 and 0 <= p9_qa < 9)
        self._add_result(
            "consistency.phase.range",
            ValidationLevel.CONSISTENCY,
            ValidationStatus.PASSED if range_ok else ValidationStatus.FAILED,
            f"phase_24=({p24_baseline},{p24_qa}), phase_9=({p9_baseline},{p9_qa})",
        )

        computed_preserved = (p24_baseline == p24_qa) and (p9_baseline == p9_qa)
        self._add_result(
            "consistency.phase.preserved",
            ValidationLevel.CONSISTENCY,
            ValidationStatus.PASSED if computed_preserved == phase_preserved else ValidationStatus.FAILED,
            f"computed={computed_preserved}, declared={phase_preserved}",
        )

        for k in ("tuple_consistency", "feature_determinism", "eval_repro", "trace", "baseline_pairing"):
            self._add_result(
                f"consistency.invariants.{k}",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED if inv.get(k) is True else ValidationStatus.FAILED,
                f"invariants.{k} must be true for success certificates",
            )

        paired_cfg = recompute.get("paired_config")
        if not isinstance(paired_cfg, dict):
            self._add_result(
                "consistency.pairing.paired_config",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                "recompute_inputs.paired_config must be object",
            )
            return

        pairing_ok = (
            paired_cfg.get("dataset_id") == dataset_id
            and paired_cfg.get("algorithm") == algorithm
            and int(paired_cfg.get("split_seed", -1)) == split_seed
            and int(paired_cfg.get("clustering_seed", -1)) == clustering_seed
        )
        self._add_result(
            "consistency.pairing.config_match",
            ValidationLevel.CONSISTENCY,
            ValidationStatus.PASSED if pairing_ok else ValidationStatus.FAILED,
            "graph_context and paired_config must match dataset/algorithm/seeds",
        )

    def _validate_trace(
        self,
        trace_name: str,
        trace: Any,
        allowed_generators: set[str],
    ) -> Optional[Tuple[Dict[str, Fraction], int, int]]:
        if not isinstance(trace, list) or len(trace) == 0:
            self._add_result(
                f"recompute.{trace_name}.shape",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                f"{trace_name} must be non-empty list",
            )
            return None

        prev_step = 0
        final_metrics: Optional[Dict[str, Fraction]] = None
        final_p24 = -1
        final_p9 = -1
        for i, step in enumerate(trace):
            if not isinstance(step, dict):
                self._add_result(
                    f"recompute.{trace_name}[{i}].shape",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.FAILED,
                    "trace step must be object",
                )
                return None

            try:
                step_index = int(step["step_index"])
                generator = step["generator"]
                _stage = step["stage"]
                p24 = int(step["phase_24"])
                p9 = int(step["phase_9"])
            except Exception as e:
                self._add_result(
                    f"recompute.{trace_name}[{i}].parse",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.FAILED,
                    f"step parse error: {e}",
                )
                return None

            contiguous = step_index == prev_step + 1
            self._add_result(
                f"recompute.{trace_name}[{i}].step_index",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED if contiguous else ValidationStatus.FAILED,
                f"step_index={step_index}, expected={prev_step + 1}",
            )
            if not contiguous:
                return None

            self._add_result(
                f"recompute.{trace_name}[{i}].generator",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED if generator in allowed_generators else ValidationStatus.FAILED,
                f"generator={generator}",
            )
            if generator not in allowed_generators:
                return None

            in_range = 0 <= p24 < 24 and 0 <= p9 < 9
            self._add_result(
                f"recompute.{trace_name}[{i}].phase_range",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED if in_range else ValidationStatus.FAILED,
                f"phase_24={p24}, phase_9={p9}",
            )
            if not in_range:
                return None

            metrics = self._parse_metric_object(step.get("metrics"), f"recompute.{trace_name}[{i}].metrics")
            if metrics is None:
                return None

            prev_step = step_index
            final_metrics = metrics
            final_p24 = p24
            final_p9 = p9

        if final_metrics is None:
            return None
        return final_metrics, final_p24, final_p9

    def validate_recompute(self, cert: Dict[str, Any], data_path: Optional[Path] = None) -> None:
        if not cert.get("success", True):
            self._add_result(
                "recompute.skip_failure",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.SKIPPED,
                "skipping recompute checks for failure certificate",
            )
            return

        recompute = cert.get("recompute_inputs")
        metric_witness = cert.get("metric_witness", {})
        phase = cert.get("phase_witness", {})
        if not isinstance(recompute, dict):
            self._add_result(
                "recompute.inputs.present",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                "success certificate must include recompute_inputs",
            )
            return

        trace_schema = recompute.get("trace_schema")
        self._add_result(
            "recompute.trace_schema",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.PASSED if trace_schema == "QA_GRAPH_STRUCTURE_TRACE.v1" else ValidationStatus.FAILED,
            f"trace_schema={trace_schema}",
        )
        if trace_schema != "QA_GRAPH_STRUCTURE_TRACE.v1":
            return

        baseline_trace = recompute.get("baseline_trace")
        qa_trace = recompute.get("qa_trace")
        if not isinstance(baseline_trace, list) or not isinstance(qa_trace, list):
            self._add_result(
                "recompute.trace_lists",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                "baseline_trace and qa_trace must be lists",
            )
            return

        claimed_digest = recompute.get("trace_digest")
        computed_digest = _compute_trace_digest(baseline_trace, qa_trace)
        self._add_result(
            "recompute.trace_digest",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.PASSED if claimed_digest == computed_digest else ValidationStatus.FAILED,
            f"claimed={claimed_digest}, computed={computed_digest}",
        )

        allowed_generators = set(cert.get("generator_set", []))
        baseline_terminal = self._validate_trace("baseline_trace", baseline_trace, allowed_generators)
        qa_terminal = self._validate_trace("qa_trace", qa_trace, allowed_generators)
        if baseline_terminal is None or qa_terminal is None:
            return

        baseline_final_metrics, baseline_p24, baseline_p9 = baseline_terminal
        qa_final_metrics, qa_p24, qa_p9 = qa_terminal

        paired_len_ok = len(baseline_trace) == len(qa_trace)
        self._add_result(
            "recompute.trace_pair_length",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.PASSED if paired_len_ok else ValidationStatus.FAILED,
            f"baseline_steps={len(baseline_trace)}, qa_steps={len(qa_trace)}",
        )

        baseline_metrics = self._parse_metric_object(
            metric_witness.get("baseline_metrics"), "recompute.metric_witness.baseline_metrics"
        )
        qa_metrics = self._parse_metric_object(metric_witness.get("qa_metrics"), "recompute.metric_witness.qa_metrics")
        if baseline_metrics is None or qa_metrics is None:
            return

        for metric_key in self.METRIC_KEYS:
            self._add_result(
                f"recompute.match.baseline.{metric_key}",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED if baseline_metrics[metric_key] == baseline_final_metrics[metric_key] else ValidationStatus.FAILED,
                f"terminal={baseline_final_metrics[metric_key]}, declared={baseline_metrics[metric_key]}",
            )
            self._add_result(
                f"recompute.match.qa.{metric_key}",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED if qa_metrics[metric_key] == qa_final_metrics[metric_key] else ValidationStatus.FAILED,
                f"terminal={qa_final_metrics[metric_key]}, declared={qa_metrics[metric_key]}",
            )

        phase_match = (
            int(phase.get("phase_24_baseline", -1)) == baseline_p24
            and int(phase.get("phase_9_baseline", -1)) == baseline_p9
            and int(phase.get("phase_24_qa", -1)) == qa_p24
            and int(phase.get("phase_9_qa", -1)) == qa_p9
        )
        self._add_result(
            "recompute.match.phase_witness",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.PASSED if phase_match else ValidationStatus.FAILED,
            f"baseline=({baseline_p24},{baseline_p9}), qa=({qa_p24},{qa_p9})",
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
    baseline_trace: List[Dict[str, Any]] = [
        {
            "step_index": 1,
            "generator": "sigma_feat_extract",
            "stage": "feature_extract",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "1/2", "nmi": "1/2", "modularity": "9/20", "purity": "3/4"},
        },
        {
            "step_index": 2,
            "generator": "sigma_qa_embed",
            "stage": "baseline_passthrough",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "11/20", "nmi": "13/25", "modularity": "19/40", "purity": "31/40"},
        },
        {
            "step_index": 3,
            "generator": "sigma_cluster",
            "stage": "cluster",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "3/5", "nmi": "27/50", "modularity": "1/2", "purity": "4/5"},
        },
        {
            "step_index": 4,
            "generator": "sigma_eval",
            "stage": "eval",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "3/5", "nmi": "11/20", "modularity": "1/2", "purity": "4/5"},
        },
        {
            "step_index": 5,
            "generator": "sigma_phase_analyze",
            "stage": "phase_analyze",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "3/5", "nmi": "11/20", "modularity": "1/2", "purity": "4/5"},
        },
    ]

    qa_trace: List[Dict[str, Any]] = [
        {
            "step_index": 1,
            "generator": "sigma_feat_extract",
            "stage": "feature_extract",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "1/2", "nmi": "1/2", "modularity": "9/20", "purity": "3/4"},
        },
        {
            "step_index": 2,
            "generator": "sigma_qa_embed",
            "stage": "qa_embed",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "29/50", "nmi": "11/20", "modularity": "13/25", "purity": "79/100"},
        },
        {
            "step_index": 3,
            "generator": "sigma_cluster",
            "stage": "cluster",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "33/50", "nmi": "57/100", "modularity": "11/20", "purity": "81/100"},
        },
        {
            "step_index": 4,
            "generator": "sigma_eval",
            "stage": "eval",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "17/25", "nmi": "29/50", "modularity": "14/25", "purity": "41/50"},
        },
        {
            "step_index": 5,
            "generator": "sigma_phase_analyze",
            "stage": "phase_analyze",
            "phase_24": 7,
            "phase_9": 4,
            "metrics": {"ari": "17/25", "nmi": "29/50", "modularity": "14/25", "purity": "41/50"},
        },
    ]

    return {
        "certificate_id": "demo_graph_structure_karate_001",
        "certificate_type": "GRAPH_STRUCTURE_CERT",
        "timestamp": "2026-02-09T00:00:00Z",
        "version": "1.0.0",
        "schema": "QA_GRAPH_STRUCTURE_CERT.v1",
        "success": True,
        "generator_set": [
            "sigma_feat_extract",
            "sigma_qa_embed",
            "sigma_cluster",
            "sigma_eval",
            "sigma_phase_analyze",
        ],
        "graph_context": {
            "dataset_id": "karate_club",
            "algorithm": "louvain",
            "node_count": 34,
            "edge_count": 78,
            "community_count": 4,
            "split_seed": 17,
            "clustering_seed": 17,
        },
        "metric_witness": {
            "qa_metrics": {"ari": "17/25", "nmi": "29/50", "modularity": "14/25", "purity": "41/50"},
            "baseline_metrics": {"ari": "3/5", "nmi": "11/20", "modularity": "1/2", "purity": "4/5"},
            "delta_metrics": {"ari": "2/25", "nmi": "3/100", "modularity": "3/50", "purity": "1/50"},
            "delta_non_negative_any": True,
        },
        "phase_witness": {
            "phase_24_baseline": 7,
            "phase_24_qa": 7,
            "phase_9_baseline": 4,
            "phase_9_qa": 4,
            "phase_preserved": True,
        },
        "invariants": {
            "tuple_consistency": True,
            "feature_determinism": True,
            "eval_repro": True,
            "trace": True,
            "baseline_pairing": True,
        },
        "recompute_inputs": {
            "trace_schema": "QA_GRAPH_STRUCTURE_TRACE.v1",
            "trace_digest": _compute_trace_digest(baseline_trace, qa_trace),
            "paired_config": {
                "dataset_id": "karate_club",
                "algorithm": "louvain",
                "split_seed": 17,
                "clustering_seed": 17,
                "baseline_mode": "graph_stats_only",
                "qa_mode": "qa_structural_embed",
            },
            "baseline_trace": baseline_trace,
            "qa_trace": qa_trace,
        },
        "qa_interpretation": {
            "success_type": "GRAPH_STRUCTURE_DELTA_CERTIFIED",
            "lattice_position": "paired_baseline ∧ reproducible_metrics ∧ phase_preserved",
            "note": "QA structural packet improves ARI/modularity under matched config",
        },
    }


def create_demo_failure_certificate() -> Dict[str, Any]:
    return {
        "certificate_id": "demo_graph_structure_parity_failure_001",
        "certificate_type": "GRAPH_STRUCTURE_CERT",
        "timestamp": "2026-02-09T00:00:00Z",
        "version": "1.0.0",
        "schema": "QA_GRAPH_STRUCTURE_CERT.v1",
        "success": False,
        "generator_set": [
            "sigma_feat_extract",
            "sigma_qa_embed",
            "sigma_cluster",
            "sigma_eval",
            "sigma_phase_analyze",
        ],
        "failure_mode": "parity",
        "failure_witness": {
            "reason": "paired baseline mismatch: clustering_seed drift",
            "expected": {"split_seed": 17, "clustering_seed": 17},
            "observed": {"split_seed": 17, "clustering_seed": 23},
            "remediation": [
                "Lock baseline and QA runs to identical split/clustering seeds",
                "Re-emit certificate only after paired config parity is restored"
            ]
        },
        "qa_interpretation": {
            "failure_class": "PARITY",
            "obstruction_type": "paired_baseline_mismatch",
            "note": "delta claims are invalid without strict pairing invariants",
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
    parser = argparse.ArgumentParser(description="Validate QA graph structure certificates")
    parser.add_argument("certificate", nargs="?", help="Path to certificate JSON")
    parser.add_argument("--demo", action="store_true", help="Run demo success certificate")
    parser.add_argument("--demo-failure", action="store_true", help="Run demo failure certificate")
    parser.add_argument(
        "--level",
        default="consistency",
        choices=("schema", "consistency", "recompute"),
        help="validation depth",
    )
    parser.add_argument("--data", type=Path, help="Optional raw data path for recompute level")
    parser.add_argument("--json", action="store_true", help="Emit report as JSON")
    args = parser.parse_args()

    if args.demo:
        cert = create_demo_success_certificate()
    elif args.demo_failure:
        cert = create_demo_failure_certificate()
    elif args.certificate:
        with open(args.certificate, "r", encoding="utf-8") as f:
            cert = json.load(f)
    else:
        parser.error("certificate path required (or use --demo / --demo-failure)")

    validator = GraphStructureValidator(strict=True)
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
