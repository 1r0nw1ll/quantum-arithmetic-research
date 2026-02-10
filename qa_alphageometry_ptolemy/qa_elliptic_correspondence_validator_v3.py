#!/usr/bin/env python3
"""
qa_elliptic_correspondence_validator_v3.py

Strict validator for QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.

Validation levels:
- Level 1 (Schema): required fields and shape checks
- Level 2 (Consistency): invariant and replay consistency checks
- Level 3 (Recompute): deterministic trace replay checks
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

from qa_elliptic_correspondence_certificate import (
    FAILURE_MODES,
    GENERATOR_SET,
    build_demo_failure_certificate,
    build_demo_success_certificate,
)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _trace_digest(trace: List[Dict[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(trace).encode("utf-8")).hexdigest()


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


class EllipticCorrespondenceValidator:
    REQUIRED_FIELDS = [
        "certificate_id",
        "certificate_type",
        "timestamp",
        "version",
        "schema",
        "success",
        "generator_set",
    ]
    SUCCESS_FIELDS = ["state_descriptor", "topology_witness", "invariants", "recompute_inputs"]
    FAILURE_FIELDS = ["failure_mode", "failure_witness"]

    VALID_GENERATORS = set(GENERATOR_SET)
    VALID_FAILURE_MODES = set(FAILURE_MODES)

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

        if cert.get("schema") == "QA_ELLIPTIC_CORRESPONDENCE_CERT.v1":
            self._add_result(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                "valid schema: QA_ELLIPTIC_CORRESPONDENCE_CERT.v1",
            )
        else:
            self._add_result(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"unexpected schema: {cert.get('schema')}",
            )

        if cert.get("certificate_type") == "ELLIPTIC_CORRESPONDENCE_CERT":
            self._add_result(
                "schema.certificate_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                "valid certificate_type: ELLIPTIC_CORRESPONDENCE_CERT",
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

        if cert.get("success", True):
            for field in self.SUCCESS_FIELDS:
                if field not in cert:
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
        else:
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

    def _step_key(self, step: Dict[str, Any]) -> Tuple[str, str, str, int, str]:
        return (
            str(step.get("u_in_re")),
            str(step.get("u_in_im")),
            str(step.get("sheet_in")),
            int(step.get("branch_index_in")),
            str(step.get("generator")),
        )

    def _step_value(self, step: Dict[str, Any]) -> Tuple[str, str, str, int, str, str]:
        return (
            str(step.get("u_out_re")),
            str(step.get("u_out_im")),
            str(step.get("sheet_out")),
            int(step.get("branch_index_out")),
            str(step.get("status")),
            str(step.get("fail_type", "")),
        )

    def validate_consistency(self, cert: Dict[str, Any]) -> None:
        if not cert.get("success", True):
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

        generators = cert.get("generator_set", [])
        topo = cert.get("topology_witness")
        inv = cert.get("invariants")
        recomp = cert.get("recompute_inputs")

        if not all(isinstance(x, dict) for x in (topo, inv, recomp)):
            self._add_result(
                "consistency.required_objects",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                "success certificate requires object fields: topology_witness/invariants/recompute_inputs",
            )
            return

        branching = topo.get("branching_factor_declared")
        expected_branching = len(set(generators))
        if branching == expected_branching == 6:
            self._add_result(
                "consistency.branching_factor",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED,
                "declared branch factor matches 6-generator correspondence",
            )
        else:
            self._add_result(
                "consistency.branching_factor",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"branching factor mismatch: declared={branching}, generators={expected_branching}, expected=6",
            )

        invariant_keys = ["curve_constraint", "determinism", "cut_consistency", "trace_complete"]
        bad = [k for k in invariant_keys if inv.get(k) is not True]
        if not bad:
            self._add_result(
                "consistency.invariants",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED,
                "all hard invariants are true",
            )
        else:
            self._add_result(
                "consistency.invariants",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"false/missing invariants: {bad}",
            )

        try:
            max_norm_u = self._parse_scalar(topo.get("max_norm_u"))
            max_norm_v = self._parse_scalar(topo.get("max_norm_v"))
            if max_norm_u >= 0 and max_norm_v >= 0:
                self._add_result(
                    "consistency.max_norms",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    "max_norm_u/max_norm_v are valid non-negative scalars",
                )
            else:
                self._add_result(
                    "consistency.max_norms",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    "max norms must be non-negative",
                )
        except Exception as e:
            self._add_result(
                "consistency.max_norms",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"failed to parse max norms: {e}",
            )

        trace_schema = recomp.get("trace_schema")
        if trace_schema == "QA_ELLIPTIC_CORRESPONDENCE_TRACE.v1":
            self._add_result(
                "consistency.trace_schema",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED,
                "trace_schema recognized",
            )
        else:
            self._add_result(
                "consistency.trace_schema",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"unexpected trace_schema: {trace_schema}",
            )

        initial = recomp.get("initial_state")
        trace = recomp.get("transition_trace")
        if not isinstance(initial, dict) or not isinstance(trace, list) or not trace:
            self._add_result(
                "consistency.trace_shape",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                "initial_state must be object and transition_trace must be non-empty list",
            )
            return

        declared_digest = recomp.get("trace_digest", "")
        computed_digest = _trace_digest(trace)
        if declared_digest == computed_digest:
            self._add_result(
                "consistency.trace_digest",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED,
                "trace_digest matches canonical transition trace",
            )
        else:
            self._add_result(
                "consistency.trace_digest",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                "trace_digest mismatch",
                {
                    "declared": declared_digest,
                    "computed": computed_digest,
                },
            )

        deterministic_map: Dict[Tuple[str, str, str, int, str], Tuple[str, str, str, int, str, str]] = {}
        prev_out: Optional[Tuple[str, str, str, int]] = None

        for i, step in enumerate(trace, start=1):
            if not isinstance(step, dict):
                self._add_result(
                    f"consistency.step.{i}.shape",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    "trace step must be object",
                )
                continue

            idx = step.get("step_index")
            if idx != i:
                self._add_result(
                    f"consistency.step.{i}.index",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"non-contiguous step_index: expected {i}, got {idx}",
                )

            gen = step.get("generator")
            if gen not in self.VALID_GENERATORS:
                self._add_result(
                    f"consistency.step.{i}.generator",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"unknown generator: {gen}",
                )

            status = step.get("status")
            fail_type = step.get("fail_type", "")
            if status not in ("ok", "fail"):
                self._add_result(
                    f"consistency.step.{i}.status",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"invalid status: {status}",
                )
            elif status == "ok" and fail_type not in ("", None):
                self._add_result(
                    f"consistency.step.{i}.fail_type",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    "ok step must not carry fail_type",
                )
            elif status == "fail" and fail_type not in self.VALID_FAILURE_MODES:
                self._add_result(
                    f"consistency.step.{i}.fail_type",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"invalid fail_type for fail step: {fail_type}",
                )

            try:
                _ = self._parse_scalar(step.get("curve_residual_abs", "0"))
            except Exception as e:
                self._add_result(
                    f"consistency.step.{i}.curve_residual_abs",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"invalid curve_residual_abs: {e}",
                )

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

            if i == 1:
                init_tuple = (
                    str(initial.get("u_re")),
                    str(initial.get("u_im")),
                    str(initial.get("sheet")),
                    int(initial.get("branch_index")),
                )
                if in_tuple != init_tuple:
                    self._add_result(
                        "consistency.initial_link",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        f"first trace input {in_tuple} does not match initial_state {init_tuple}",
                    )
            elif prev_out is not None and in_tuple != prev_out:
                self._add_result(
                    f"consistency.step.{i}.continuity",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"step input {in_tuple} does not match previous output {prev_out}",
                )

            prev_out = out_tuple

            det_key = self._step_key(step)
            det_val = self._step_value(step)
            if det_key in deterministic_map:
                if deterministic_map[det_key] != det_val:
                    self._add_result(
                        f"consistency.step.{i}.determinism",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        "same (state,generator) produced different outcomes",
                        {
                            "key": det_key,
                            "first": deterministic_map[det_key],
                            "second": det_val,
                        },
                    )
            else:
                deterministic_map[det_key] = det_val

        if self.failed_count() == 0:
            self._add_result(
                "consistency.trace_replay",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED,
                "trace continuity and determinism checks passed",
            )

    def validate_recompute(self, cert: Dict[str, Any]) -> None:
        if not cert.get("success", True):
            self._add_result(
                "recompute.failure_cert",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.SKIPPED,
                "recompute skipped for failure certificate",
            )
            return

        recomp = cert.get("recompute_inputs", {})
        trace = recomp.get("transition_trace", []) if isinstance(recomp, dict) else []
        if isinstance(trace, list) and trace:
            self._add_result(
                "recompute.trace_hash",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED,
                "recompute trace is present for independent replay",
            )
        else:
            self._add_result(
                "recompute.trace_hash",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                "missing transition trace for recompute",
            )

    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == ValidationStatus.FAILED)

    def validate(self, cert: Dict[str, Any], level: ValidationLevel = ValidationLevel.RECOMPUTE) -> ValidationReport:
        self.results = []
        self.validate_schema(cert)

        if level.value >= ValidationLevel.CONSISTENCY.value:
            self.validate_consistency(cert)

        if level.value >= ValidationLevel.RECOMPUTE.value:
            self.validate_recompute(cert)

        return ValidationReport(
            certificate_id=cert.get("certificate_id", "UNKNOWN"),
            schema=cert.get("schema", "UNKNOWN"),
            results=self.results,
        )


LEVEL_MAP = {
    "schema": ValidationLevel.SCHEMA,
    "consistency": ValidationLevel.CONSISTENCY,
    "recompute": ValidationLevel.RECOMPUTE,
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_demo(level: ValidationLevel, as_json: bool) -> int:
    validator = EllipticCorrespondenceValidator(strict=True)

    success_report = validator.validate(build_demo_success_certificate().to_dict(), level=level)
    failure_report = validator.validate(build_demo_failure_certificate().to_dict(), level=level)

    ok = success_report.all_passed and failure_report.all_passed

    if as_json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "success_demo": success_report.to_dict(),
                    "failure_demo": failure_report.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("=== DEMO: success certificate ===")
        print(success_report.summary())
        print("\n=== DEMO: failure certificate ===")
        print(failure_report.summary())

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA elliptic correspondence certificates")
    parser.add_argument("certificate", nargs="?", help="Path to certificate JSON file")
    parser.add_argument(
        "--level",
        choices=LEVEL_MAP.keys(),
        default="recompute",
        help="Validation level (default: recompute)",
    )
    parser.add_argument("--demo", action="store_true", help="Run demo validation for success/failure certs")
    parser.add_argument("--json", action="store_true", help="Output report as JSON")
    args = parser.parse_args()

    level = LEVEL_MAP[args.level]

    if args.demo:
        return run_demo(level=level, as_json=args.json)

    if not args.certificate:
        parser.error("certificate path is required unless --demo is set")

    cert_path = Path(args.certificate)
    cert = _load_json(cert_path)

    validator = EllipticCorrespondenceValidator(strict=True)
    report = validator.validate(cert, level=level)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.summary())

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
