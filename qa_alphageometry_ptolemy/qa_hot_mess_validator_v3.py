#!/usr/bin/env python3
"""
qa_hot_mess_validator_v3.py

Strict validator for Hot Mess (Incoherence) Certificates.
Based on the certificate schema in qa_hot_mess_certificate.py.

Validation levels:
- Level 1 (Schema): Required fields present, correct types, no floats
- Level 2 (Consistency): Internal arithmetic + invariant threshold checks
- Level 3 (Recompute): Recompute agreement + step-length witnesses from run_outcomes

Usage:
    python qa_hot_mess_validator_v3.py certificate.json
    python qa_hot_mess_validator_v3.py --demo
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from fractions import Fraction
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


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
            f"\u2714 Passed:   {self.passed}",
            f"\u2718 Failed:   {self.failed}",
            f"\u26a0 Warnings: {self.warnings}",
        ]
        if self.all_passed:
            lines.append("")
            lines.append("\u2714 ALL CHECKS PASSED")
        else:
            lines.append("")
            lines.append("\u2718 VALIDATION FAILED")
            lines.append("")
            lines.append("Failures:")
            for r in self.results:
                if r.status == ValidationStatus.FAILED:
                    lines.append(f"  - {r.check_name}: {r.message}")
        return "\n".join(lines)


def _float_paths(obj: Any, path: str = "") -> List[str]:
    paths: List[str] = []
    if isinstance(obj, float):
        paths.append(path or "<root>")
        return paths
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else str(k)
            paths.extend(_float_paths(v, child))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child = f"{path}[{i}]"
            paths.extend(_float_paths(v, child))
    return paths


class HotMessIncoherenceValidator:
    EXPECTED_CERT_TYPE = "HOT_MESS_INCOHERENCE_CERT"
    EXPECTED_SCHEMA = "QA_HOT_MESS_INCOHERENCE_CERT.v1"

    REQUIRED_FIELDS = [
        "certificate_id",
        "certificate_type",
        "timestamp",
        "version",
        "schema",
        "success",
        "model_id",
        "task_family",
        "eval_metric_id",
        "num_runs",
        "run_outcomes",
        "decomposition_witness",
        "coherence_invariant",
    ]

    FAILURE_FIELDS = ["failure_mode", "failure_witness"]

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.results: List[ValidationResult] = []

    def _add(self, name: str, level: ValidationLevel, status: ValidationStatus, msg: str,
             details: Optional[Dict[str, Any]] = None) -> None:
        self.results.append(ValidationResult(name, level, status, msg, details))

    def _parse_scalar(self, value: Any) -> Fraction:
        if isinstance(value, bool):
            raise ValueError("bool is not a scalar")
        if isinstance(value, (int, Fraction)):
            return Fraction(value)
        if isinstance(value, str):
            return Fraction(value)
        if isinstance(value, float):
            if self.strict:
                raise ValueError(f"Float not allowed in strict mode: {value}")
            return Fraction(value).limit_denominator(10**12)
        raise ValueError(f"Cannot parse scalar: {type(value)}")

    def validate_schema(self, cert: Dict[str, Any]) -> None:
        # No floats in strict mode (whole cert)
        float_paths = _float_paths(cert)
        if float_paths and self.strict:
            self._add(
                "schema.no_floats",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                "Found float(s) in certificate (exact scalars required)",
                {"float_paths": float_paths[:25], "count": len(float_paths)},
            )
        else:
            self._add(
                "schema.no_floats",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                "No floats detected" if not float_paths else "Floats allowed (non-strict mode)",
            )

        for field in self.REQUIRED_FIELDS:
            if field not in cert:
                self._add(
                    f"schema.required.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing required field: {field}",
                )
            else:
                self._add(
                    f"schema.required.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"Field present: {field}",
                )

        schema = cert.get("schema", "")
        if schema != self.EXPECTED_SCHEMA:
            self._add(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED if self.strict else ValidationStatus.WARNING,
                f"Unexpected schema: {schema} (expected {self.EXPECTED_SCHEMA})",
            )
        else:
            self._add(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                f"Valid schema: {schema}",
            )

        cert_type = cert.get("certificate_type", "")
        if cert_type != self.EXPECTED_CERT_TYPE:
            self._add(
                "schema.certificate_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED if self.strict else ValidationStatus.WARNING,
                f"Unexpected certificate_type: {cert_type} (expected {self.EXPECTED_CERT_TYPE})",
            )
        else:
            self._add(
                "schema.certificate_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                f"Valid certificate_type: {cert_type}",
            )

        success = cert.get("success")
        if not isinstance(success, bool):
            self._add(
                "schema.success_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                "success must be boolean",
            )
        else:
            self._add(
                "schema.success_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                "success is boolean",
            )
            if success is False:
                for field in self.FAILURE_FIELDS:
                    if field not in cert:
                        self._add(
                            f"schema.failure.{field}",
                            ValidationLevel.SCHEMA,
                            ValidationStatus.FAILED,
                            f"Failure certificate missing: {field}",
                        )
                    else:
                        self._add(
                            f"schema.failure.{field}",
                            ValidationLevel.SCHEMA,
                            ValidationStatus.PASSED,
                            f"Failure field present: {field}",
                        )

        # Basic types
        if "num_runs" in cert:
            ok = isinstance(cert.get("num_runs"), int) and cert.get("num_runs") >= 0
            self._add(
                "schema.num_runs_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
                "num_runs is non-negative int" if ok else "num_runs must be non-negative int",
            )

        outcomes = cert.get("run_outcomes")
        ok_outcomes = isinstance(outcomes, list)
        self._add(
            "schema.run_outcomes_type",
            ValidationLevel.SCHEMA,
            ValidationStatus.PASSED if ok_outcomes else ValidationStatus.FAILED,
            "run_outcomes is list" if ok_outcomes else "run_outcomes must be a list",
        )
        if isinstance(outcomes, list):
            required_fields = {"run_id", "rng_seed", "step_count", "output_hash", "score", "success"}
            for i, o in enumerate(outcomes[:50]):
                if not isinstance(o, dict):
                    self._add(
                        f"schema.run_outcomes[{i}]",
                        ValidationLevel.SCHEMA,
                        ValidationStatus.FAILED,
                        "run_outcomes entries must be dicts",
                    )
                    continue
                missing = [f for f in required_fields if f not in o]
                if missing:
                    self._add(
                        f"schema.run_outcomes[{i}].fields",
                        ValidationLevel.SCHEMA,
                        ValidationStatus.FAILED,
                        f"run_outcomes[{i}] missing fields: {missing}",
                    )
                else:
                    self._add(
                        f"schema.run_outcomes[{i}].fields",
                        ValidationLevel.SCHEMA,
                        ValidationStatus.PASSED,
                        "run_outcomes entry has required fields",
                    )

        # Decomposition witness fields
        dec = cert.get("decomposition_witness")
        if not isinstance(dec, dict):
            self._add(
                "schema.decomposition_witness_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                "decomposition_witness must be dict",
            )
        else:
            for f in ("metric_id", "total_error", "bias_component", "variance_component", "incoherence_ratio"):
                if f not in dec:
                    self._add(
                        f"schema.decomposition_witness.{f}",
                        ValidationLevel.SCHEMA,
                        ValidationStatus.FAILED,
                        f"Missing decomposition_witness field: {f}",
                    )
                else:
                    self._add(
                        f"schema.decomposition_witness.{f}",
                        ValidationLevel.SCHEMA,
                        ValidationStatus.PASSED,
                        f"Field present: {f}",
                    )

        inv = cert.get("coherence_invariant")
        if not isinstance(inv, dict):
            self._add(
                "schema.coherence_invariant_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                "coherence_invariant must be dict",
            )
        else:
            for f in ("metric_id", "max_incoherence_ratio"):
                if f not in inv:
                    self._add(
                        f"schema.coherence_invariant.{f}",
                        ValidationLevel.SCHEMA,
                        ValidationStatus.FAILED,
                        f"Missing coherence_invariant field: {f}",
                    )
                else:
                    self._add(
                        f"schema.coherence_invariant.{f}",
                        ValidationLevel.SCHEMA,
                        ValidationStatus.PASSED,
                        f"Field present: {f}",
                    )

    def _compute_agreement_rate(self, outcomes: List[Dict[str, Any]]) -> Fraction:
        if not outcomes:
            return Fraction(0)
        counts: Dict[str, int] = {}
        for o in outcomes:
            h = o.get("output_hash")
            if isinstance(h, str):
                counts[h] = counts.get(h, 0) + 1
        if not counts:
            return Fraction(0)
        max_count = max(counts.values())
        return Fraction(max_count, len(outcomes))

    def _compute_step_stats(self, outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        steps = sorted(int(o.get("step_count")) for o in outcomes if "step_count" in o)
        if not steps:
            return {"mean_step_count": Fraction(0), "median_step_count": Fraction(0), "p95_step_count": 0}
        n = len(steps)
        mean = Fraction(sum(steps), n)
        if n % 2 == 1:
            median = Fraction(steps[n // 2])
        else:
            median = Fraction(steps[n // 2 - 1] + steps[n // 2], 2)
        idx = (95 * n + 100 - 1) // 100 - 1
        idx = max(0, min(n - 1, idx))
        p95 = steps[idx]
        return {"mean_step_count": mean, "median_step_count": median, "p95_step_count": p95}

    def validate_consistency(self, cert: Dict[str, Any]) -> None:
        # num_runs matches outcomes
        outcomes = cert.get("run_outcomes")
        n = cert.get("num_runs")
        if isinstance(outcomes, list) and isinstance(n, int):
            ok = len(outcomes) == n
            self._add(
                "consistency.num_runs_match",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
                f"len(run_outcomes)={len(outcomes)} matches num_runs={n}" if ok else "num_runs mismatch",
            )

        # Agreement rate consistency
        if isinstance(outcomes, list):
            recomputed = self._compute_agreement_rate(outcomes)
            provided = cert.get("agreement_rate")
            if provided is None:
                self._add(
                    "consistency.agreement_rate_present",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.WARNING,
                    f"agreement_rate missing (recomputed={recomputed})",
                )
            else:
                try:
                    prov = self._parse_scalar(provided)
                    ok = prov == recomputed
                    self._add(
                        "consistency.agreement_rate_match",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
                        "agreement_rate matches recompute" if ok else f"agreement_rate mismatch (provided={prov}, recomputed={recomputed})",
                    )
                except Exception as e:
                    self._add(
                        "consistency.agreement_rate_parse",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        f"agreement_rate not parseable as exact scalar: {e}",
                    )

        # Decomposition arithmetic
        dec = cert.get("decomposition_witness")
        if isinstance(dec, dict):
            try:
                total = self._parse_scalar(dec.get("total_error"))
                bias = self._parse_scalar(dec.get("bias_component"))
                var = self._parse_scalar(dec.get("variance_component"))
                ratio = self._parse_scalar(dec.get("incoherence_ratio"))
                ok_sum = (bias + var) == total
                self._add(
                    "consistency.decomposition.total_equals_sum",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED if ok_sum else ValidationStatus.FAILED,
                    "total_error=bias+variance" if ok_sum else f"total_error mismatch (total={total}, bias+var={bias+var})",
                )
                if total == 0:
                    ok_ratio = ratio == 0
                else:
                    ok_ratio = ratio == (var / total)
                self._add(
                    "consistency.decomposition.ratio",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED if ok_ratio else ValidationStatus.FAILED,
                    "incoherence_ratio=variance/total" if ok_ratio else f"ratio mismatch (ratio={ratio}, expected={var/total if total else 0})",
                )
            except Exception as e:
                self._add(
                    "consistency.decomposition.parse",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Failed to parse decomposition scalars: {e}",
                )

        # Coherence invariant threshold checks (only required for success)
        inv = cert.get("coherence_invariant")
        if isinstance(dec, dict) and isinstance(inv, dict):
            success = cert.get("success", True)
            try:
                incoh = self._parse_scalar(dec.get("incoherence_ratio"))
                max_incoh = self._parse_scalar(inv.get("max_incoherence_ratio"))
                if success is True:
                    ok = incoh <= max_incoh
                    self._add(
                        "consistency.I_coh.incoherence_bound",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
                        f"incoherence_ratio {incoh} <= {max_incoh}" if ok else f"I_coh violated: {incoh} > {max_incoh}",
                    )
                else:
                    self._add(
                        "consistency.I_coh.incoherence_bound",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.SKIPPED,
                        "Skipping success-only coherence bound (success=false)",
                    )

                if "min_agreement_rate" in inv and inv.get("min_agreement_rate") is not None and isinstance(outcomes, list):
                    min_agree = self._parse_scalar(inv.get("min_agreement_rate"))
                    agree = self._compute_agreement_rate(outcomes)
                    if success is True:
                        ok2 = agree >= min_agree
                        self._add(
                            "consistency.I_coh.agreement_bound",
                            ValidationLevel.CONSISTENCY,
                            ValidationStatus.PASSED if ok2 else ValidationStatus.FAILED,
                            f"agreement_rate {agree} >= {min_agree}" if ok2 else f"I_coh violated: {agree} < {min_agree}",
                        )
                    else:
                        self._add(
                            "consistency.I_coh.agreement_bound",
                            ValidationLevel.CONSISTENCY,
                            ValidationStatus.SKIPPED,
                            "Skipping success-only agreement bound (success=false)",
                        )
            except Exception as e:
                self._add(
                    "consistency.I_coh.parse",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Failed to parse I_coh scalars: {e}",
                )

    def validate_recompute(self, cert: Dict[str, Any]) -> None:
        outcomes = cert.get("run_outcomes")
        if not isinstance(outcomes, list):
            self._add(
                "recompute.run_outcomes",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.SKIPPED,
                "run_outcomes missing or not list; cannot recompute witnesses",
            )
            return

        # Recompute agreement rate
        recomputed_agree = self._compute_agreement_rate(outcomes)
        provided = cert.get("agreement_rate")
        if provided is None:
            self._add(
                "recompute.agreement_rate",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.WARNING,
                f"agreement_rate missing (recomputed={recomputed_agree})",
            )
        else:
            try:
                prov = self._parse_scalar(provided)
                ok = prov == recomputed_agree
                self._add(
                    "recompute.agreement_rate",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
                    "agreement_rate matches recompute" if ok else f"agreement_rate mismatch (provided={prov}, recomputed={recomputed_agree})",
                )
            except Exception as e:
                self._add(
                    "recompute.agreement_rate",
                    ValidationLevel.RECOMPUTE,
                    ValidationStatus.FAILED,
                    f"agreement_rate parse error: {e}",
                )

        # Recompute step length witness
        recomputed = self._compute_step_stats(outcomes)
        provided_rlw = cert.get("reasoning_length_witness")
        if provided_rlw is None:
            self._add(
                "recompute.reasoning_length_witness",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.WARNING,
                f"reasoning_length_witness missing (recomputed mean={recomputed['mean_step_count']})",
            )
            return
        if not isinstance(provided_rlw, dict):
            self._add(
                "recompute.reasoning_length_witness",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                "reasoning_length_witness must be dict",
            )
            return
        try:
            mean_p = self._parse_scalar(provided_rlw.get("mean_step_count"))
            median_p = self._parse_scalar(provided_rlw.get("median_step_count"))
            p95_p = int(provided_rlw.get("p95_step_count"))
            ok = (
                mean_p == recomputed["mean_step_count"]
                and median_p == recomputed["median_step_count"]
                and p95_p == recomputed["p95_step_count"]
            )
            self._add(
                "recompute.reasoning_length_witness",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
                "reasoning_length_witness matches recompute" if ok else "reasoning_length_witness mismatch",
                {
                    "provided": {"mean": str(mean_p), "median": str(median_p), "p95": p95_p},
                    "recomputed": {
                        "mean": str(recomputed["mean_step_count"]),
                        "median": str(recomputed["median_step_count"]),
                        "p95": recomputed["p95_step_count"],
                    },
                },
            )
        except Exception as e:
            self._add(
                "recompute.reasoning_length_witness",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.FAILED,
                f"Failed to parse reasoning_length_witness: {e}",
            )


def validate_certificate(cert: Dict[str, Any], strict: bool = True) -> ValidationReport:
    v = HotMessIncoherenceValidator(strict=strict)
    v.validate_schema(cert)
    v.validate_consistency(cert)
    v.validate_recompute(cert)
    return ValidationReport(
        certificate_id=str(cert.get("certificate_id", "UNKNOWN")),
        schema=str(cert.get("schema", "UNKNOWN")),
        results=v.results,
    )


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="QA Hot Mess (Incoherence) Validator v3")
    ap.add_argument("certificate", nargs="?", help="Path to certificate JSON")
    ap.add_argument("--demo", action="store_true", help="Run demo certificates")
    ap.add_argument("--non-strict", action="store_true", help="Allow floats (converted to Fraction)")
    args = ap.parse_args(argv)

    strict = not args.non_strict

    if args.demo:
        from qa_hot_mess_certificate import (
            create_demo_hot_mess_failure_certificate,
            create_demo_hot_mess_success_certificate,
        )
        certs = [
            ("demo_success", create_demo_hot_mess_success_certificate()),
            ("demo_failure", create_demo_hot_mess_failure_certificate()),
        ]
        ok = True
        for label, cert in certs:
            report = validate_certificate(cert, strict=strict)
            print(f"--- {label} ---")
            print(report.summary())
            print()
            ok = ok and report.all_passed
        return 0 if ok else 1

    if not args.certificate:
        ap.print_help()
        return 2

    path = Path(args.certificate)
    cert = json.loads(path.read_text())
    report = validate_certificate(cert, strict=strict)
    print(report.summary())
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

