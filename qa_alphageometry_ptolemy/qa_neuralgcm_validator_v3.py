#!/usr/bin/env python3
"""
qa_neuralgcm_validator_v3.py

Strict validator for NeuralGCM Forecast Certificates.
Based on the certificate schema in qa_neuralgcm_certificate.py.

Validation levels:
- Level 1 (Schema): Required fields present, correct types
- Level 2 (Consistency): Internal arithmetic checks (conservation, skill, CFL)
- Level 3 (Recompute): Recompute witnesses from raw forecast data

Usage:
    python qa_neuralgcm_validator_v3.py certificate.json
    python qa_neuralgcm_validator_v3.py --demo
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from fractions import Fraction
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# VALIDATION RESULT TYPES
# ============================================================================

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
            f"",
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


# ============================================================================
# STRICT VALIDATOR CLASS
# ============================================================================

class NeuralGCMCertificateValidator:
    """Strict validator for NeuralGCM forecast certificates."""

    REQUIRED_FIELDS = ["certificate_id", "schema", "success"]

    SUCCESS_FIELDS = ["conservation"]

    FAILURE_FIELDS = ["failure_mode", "failure_witness"]

    CONSERVATION_TYPES = ["mass", "energy", "momentum", "angular_momentum", "water_vapor"]

    VALID_FAILURE_MODES = [
        "mass_conservation_violation", "energy_conservation_violation",
        "momentum_conservation_violation", "negative_humidity",
        "negative_pressure", "unphysical_temperature", "unphysical_wind",
        "cfl_violation", "numerical_instability", "divergence_detected",
        "skill_below_climatology", "acc_below_threshold", "skill_collapse"
    ]

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.results: List[ValidationResult] = []

    def _add_result(
        self,
        check_name: str,
        level: ValidationLevel,
        status: ValidationStatus,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.results.append(ValidationResult(
            check_name=check_name,
            level=level,
            status=status,
            message=message,
            details=details
        ))

    def _parse_scalar(self, value: Any) -> Fraction:
        """Parse a value as an exact scalar."""
        if isinstance(value, (int, Fraction)):
            return Fraction(value)
        if isinstance(value, str):
            return Fraction(value)
        if isinstance(value, float):
            if self.strict:
                raise ValueError(f"Float not allowed in strict mode: {value}")
            return Fraction(value).limit_denominator(10**12)
        raise ValueError(f"Cannot parse as scalar: {value}")

    # ========================================================================
    # LEVEL 1: SCHEMA VALIDATION
    # ========================================================================

    def validate_schema(self, cert: Dict[str, Any]) -> None:
        """Level 1: Validate required fields and types."""

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in cert:
                self._add_result(
                    f"schema.required.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing required field: {field}"
                )
            else:
                self._add_result(
                    f"schema.required.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"Field present: {field}"
                )

        # Check schema version
        schema = cert.get("schema", "")
        if not schema.startswith("QA_NEURALGCM"):
            self._add_result(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.WARNING,
                f"Unexpected schema: {schema}"
            )
        else:
            self._add_result(
                "schema.version",
                ValidationLevel.SCHEMA,
                ValidationStatus.PASSED,
                f"Valid schema: {schema}"
            )

        # Check success/failure fields
        success = cert.get("success", True)
        if success:
            self._validate_success_schema(cert)
        else:
            self._validate_failure_schema(cert)

    def _validate_success_schema(self, cert: Dict[str, Any]) -> None:
        """Validate success certificate schema."""
        for field in self.SUCCESS_FIELDS:
            if field not in cert or cert[field] is None:
                self._add_result(
                    f"schema.success.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Success certificate missing: {field}"
                )
            else:
                self._add_result(
                    f"schema.success.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"Success field present: {field}"
                )

        # Validate conservation structure
        if "conservation" in cert and cert["conservation"]:
            self._validate_conservation_schema(cert["conservation"])

    def _validate_failure_schema(self, cert: Dict[str, Any]) -> None:
        """Validate failure certificate schema."""
        for field in self.FAILURE_FIELDS:
            if field not in cert or cert[field] is None:
                self._add_result(
                    f"schema.failure.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Failure certificate missing: {field}"
                )
            else:
                self._add_result(
                    f"schema.failure.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.PASSED,
                    f"Failure field present: {field}"
                )

        # Validate failure_mode
        failure_mode = cert.get("failure_mode")
        if failure_mode and failure_mode not in self.VALID_FAILURE_MODES:
            self._add_result(
                "schema.failure.mode_valid",
                ValidationLevel.SCHEMA,
                ValidationStatus.WARNING,
                f"Unknown failure mode: {failure_mode}"
            )

    def _validate_conservation_schema(self, cons: Dict[str, Any]) -> None:
        """Validate conservation bundle structure."""
        required = ["mass", "energy", "momentum"]
        for field in required:
            if field not in cons:
                self._add_result(
                    f"schema.conservation.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing conservation witness: {field}"
                )
            else:
                self._validate_conservation_witness_schema(cons[field], field)

    def _validate_conservation_witness_schema(self, witness: Dict[str, Any], name: str) -> None:
        """Validate individual conservation witness."""
        required = ["initial_value", "final_value", "delta", "tolerance", "conserved"]
        for field in required:
            if field not in witness:
                self._add_result(
                    f"schema.conservation.{name}.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing field in {name} witness: {field}"
                )

    # ========================================================================
    # LEVEL 2: CONSISTENCY VALIDATION
    # ========================================================================

    def validate_consistency(self, cert: Dict[str, Any]) -> None:
        """Level 2: Validate internal arithmetic consistency."""

        if not cert.get("success", True):
            self._add_result(
                "consistency.skip_failure",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.SKIPPED,
                "Skipping consistency checks for failure certificate"
            )
            return

        # Check conservation consistency
        if "conservation" in cert and cert["conservation"]:
            self._check_conservation_consistency(cert["conservation"])

        # Check skill scores
        if "skill_scores" in cert and cert["skill_scores"]:
            for i, skill in enumerate(cert["skill_scores"]):
                self._check_skill_consistency(skill, i)

        # Check physical bounds
        if "physical_bounds" in cert and cert["physical_bounds"]:
            for i, pb in enumerate(cert["physical_bounds"]):
                self._check_physical_bounds_consistency(pb, i)

        # Check numerical stability
        if "numerical_stability" in cert and cert["numerical_stability"]:
            self._check_numerical_stability_consistency(cert["numerical_stability"])

    def _check_conservation_consistency(self, cons: Dict[str, Any]) -> None:
        """Check conservation witness arithmetic."""
        for name in ["mass", "energy", "momentum"]:
            if name not in cons:
                continue
            witness = cons[name]
            self._check_single_conservation(witness, name)

    def _check_single_conservation(self, witness: Dict[str, Any], name: str) -> None:
        """Check a single conservation witness."""
        try:
            initial = self._parse_scalar(witness.get("initial_value", 0))
            final = self._parse_scalar(witness.get("final_value", 0))
            delta = self._parse_scalar(witness.get("delta", 0))
            tolerance = self._parse_scalar(witness.get("tolerance", 0))
            conserved = witness.get("conserved", True)

            # Check delta computation
            computed_delta = final - initial
            if computed_delta == delta:
                self._add_result(
                    f"consistency.conservation.{name}.delta",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"{name} delta verified: {delta}"
                )
            else:
                self._add_result(
                    f"consistency.conservation.{name}.delta",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"{name} delta mismatch: computed {computed_delta} != claimed {delta}"
                )

            # Check conserved flag
            should_be_conserved = abs(delta) <= tolerance
            if should_be_conserved == conserved:
                self._add_result(
                    f"consistency.conservation.{name}.conserved",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"{name} conserved flag verified: {conserved}"
                )
            else:
                self._add_result(
                    f"consistency.conservation.{name}.conserved",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"{name} conserved mismatch: |{delta}| <= {tolerance} is {should_be_conserved}, claimed {conserved}"
                )

        except Exception as e:
            self._add_result(
                f"consistency.conservation.{name}",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"{name} conservation check error: {e}"
            )

    def _check_skill_consistency(self, skill: Dict[str, Any], index: int) -> None:
        """Check skill score consistency."""
        try:
            rmse = self._parse_scalar(skill.get("rmse", 0))
            climatology_rmse = self._parse_scalar(skill.get("climatology_rmse", 1))
            skill_ratio = self._parse_scalar(skill.get("skill_vs_climatology", 0))
            beats_clim = skill.get("beats_climatology", True)

            # Check skill ratio
            if climatology_rmse != 0:
                computed_ratio = rmse / climatology_rmse
                diff = abs(computed_ratio - skill_ratio)
                if diff < Fraction(1, 100):  # 1% tolerance
                    self._add_result(
                        f"consistency.skill[{index}].ratio",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.PASSED,
                        f"Skill ratio verified: {skill_ratio}"
                    )
                else:
                    self._add_result(
                        f"consistency.skill[{index}].ratio",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        f"Skill ratio mismatch: {computed_ratio} != {skill_ratio}"
                    )

            # Check beats_climatology flag
            should_beat = skill_ratio < 1
            if should_beat == beats_clim:
                self._add_result(
                    f"consistency.skill[{index}].beats_climatology",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Beats climatology verified: {beats_clim}"
                )
            else:
                self._add_result(
                    f"consistency.skill[{index}].beats_climatology",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Beats climatology mismatch: ratio {skill_ratio} < 1 is {should_beat}, claimed {beats_clim}"
                )

        except Exception as e:
            self._add_result(
                f"consistency.skill[{index}]",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Skill check error: {e}"
            )

    def _check_physical_bounds_consistency(self, pb: Dict[str, Any], index: int) -> None:
        """Check physical bounds consistency."""
        try:
            min_val = self._parse_scalar(pb.get("min_value", 0))
            max_val = self._parse_scalar(pb.get("max_value", 0))
            phys_min = self._parse_scalar(pb.get("physical_min", 0))
            phys_max = self._parse_scalar(pb.get("physical_max", 0))
            within = pb.get("within_bounds", True)

            should_be_within = min_val >= phys_min and max_val <= phys_max
            if should_be_within == within:
                self._add_result(
                    f"consistency.physical_bounds[{index}]",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Physical bounds verified: {within}"
                )
            else:
                self._add_result(
                    f"consistency.physical_bounds[{index}]",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Physical bounds mismatch: [{min_val}, {max_val}] in [{phys_min}, {phys_max}] is {should_be_within}, claimed {within}"
                )

        except Exception as e:
            self._add_result(
                f"consistency.physical_bounds[{index}]",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Physical bounds check error: {e}"
            )

    def _check_numerical_stability_consistency(self, ns: Dict[str, Any]) -> None:
        """Check numerical stability (CFL) consistency."""
        try:
            dt = self._parse_scalar(ns.get("timestep_seconds", 0))
            dx = self._parse_scalar(ns.get("grid_spacing_meters", 1))
            v_max = self._parse_scalar(ns.get("max_velocity", 0))
            cfl = self._parse_scalar(ns.get("cfl_number", 0))
            cfl_limit = self._parse_scalar(ns.get("cfl_limit", 1))
            stable = ns.get("stable", True)

            # Check CFL computation
            if dx != 0:
                computed_cfl = dt * v_max / dx
                diff = abs(computed_cfl - cfl)
                if diff < Fraction(1, 100):
                    self._add_result(
                        "consistency.cfl.number",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.PASSED,
                        f"CFL number verified: {cfl}"
                    )
                else:
                    self._add_result(
                        "consistency.cfl.number",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        f"CFL mismatch: {computed_cfl} != {cfl}"
                    )

            # Check stability flag
            should_be_stable = cfl < cfl_limit
            if should_be_stable == stable:
                self._add_result(
                    "consistency.cfl.stable",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Stability verified: {stable}"
                )
            else:
                self._add_result(
                    "consistency.cfl.stable",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Stability mismatch: CFL {cfl} < {cfl_limit} is {should_be_stable}, claimed {stable}"
                )

        except Exception as e:
            self._add_result(
                "consistency.cfl",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"CFL check error: {e}"
            )

    # ========================================================================
    # LEVEL 3: RECOMPUTE VALIDATION
    # ========================================================================

    def validate_recompute(self, cert: Dict[str, Any], data_path: Optional[Path] = None) -> None:
        """Level 3: Recompute witnesses from raw data (requires external data)."""
        if data_path is None:
            self._add_result(
                "recompute.skip",
                ValidationLevel.RECOMPUTE,
                ValidationStatus.SKIPPED,
                "Recompute skipped (no data path provided)"
            )
            return

        self._add_result(
            "recompute.not_implemented",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.WARNING,
            "Recompute hooks not yet implemented for NeuralGCM"
        )

    # ========================================================================
    # MAIN VALIDATION ENTRY POINT
    # ========================================================================

    def validate(
        self,
        cert: Dict[str, Any],
        level: ValidationLevel = ValidationLevel.CONSISTENCY,
        data_path: Optional[Path] = None
    ) -> ValidationReport:
        """Run validation up to specified level."""
        self.results = []

        # Level 1: Schema
        self.validate_schema(cert)

        # Level 2: Consistency (if requested)
        if level.value >= ValidationLevel.CONSISTENCY.value:
            self.validate_consistency(cert)

        # Level 3: Recompute (if requested)
        if level.value >= ValidationLevel.RECOMPUTE.value:
            self.validate_recompute(cert, data_path)

        return ValidationReport(
            certificate_id=cert.get("certificate_id", "unknown"),
            schema=cert.get("schema", "unknown"),
            results=self.results
        )


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_demo_certificate() -> Dict[str, Any]:
    """Create a demo success certificate for testing."""
    return {
        "certificate_id": "demo_neuralgcm_10day_001",
        "version": "1.0.0",
        "schema": "QA_NEURALGCM_FORECAST_V1",
        "success": True,
        "forecast_hours": 240,
        "conservation": {
            "mass": {
                "conservation_type": "mass",
                "initial_value": "5150000000000000000",
                "final_value": "5150000000000000000",
                "delta": "0",
                "tolerance": "1000000000000",
                "conserved": True
            },
            "energy": {
                "conservation_type": "energy",
                "initial_value": "1500000000000000000000000",
                "final_value": "1500001000000000000000000",
                "delta": "1000000000000000000",
                "tolerance": "10000000000000000000",
                "conserved": True
            },
            "momentum": {
                "conservation_type": "momentum",
                "initial_value": "14000000000000000000000000000",
                "final_value": "14000000000000000000000000000",
                "delta": "0",
                "tolerance": "10000000000000000000000",
                "conserved": True
            }
        },
        "skill_scores": [
            {
                "lead_time_hours": 240,
                "variable": "z500",
                "rmse": "85",
                "acc": "75/100",
                "climatology_rmse": "120",
                "skill_vs_climatology": "85/120",
                "beats_climatology": True
            }
        ],
        "model_version": "NeuralGCM-1.0",
        "resolution": "0.25deg"
    }


def create_demo_failure_certificate() -> Dict[str, Any]:
    """Create a demo failure certificate for testing."""
    return {
        "certificate_id": "demo_neuralgcm_failure_001",
        "version": "1.0.0",
        "schema": "QA_NEURALGCM_FORECAST_V1",
        "success": False,
        "failure_mode": "mass_conservation_violation",
        "failure_witness": {
            "reason": "Total atmospheric mass leaked during forecast",
            "mass_initial": "5150000000000000000",
            "mass_final": "5140000000000000000",
            "mass_delta": "-10000000000000000",
            "tolerance": "1000000000000",
            "location": "Upper troposphere, tropical Pacific"
        },
        "conservation": {
            "mass": {
                "conservation_type": "mass",
                "initial_value": "5150000000000000000",
                "final_value": "5140000000000000000",
                "delta": "-10000000000000000",
                "tolerance": "1000000000000",
                "conserved": False
            },
            "energy": {
                "conservation_type": "energy",
                "initial_value": "1500000000000000000000000",
                "final_value": "1500000000000000000000000",
                "delta": "0",
                "tolerance": "10000000000000000000",
                "conserved": True
            },
            "momentum": {
                "conservation_type": "momentum",
                "initial_value": "14000000000000000000000000000",
                "final_value": "14000000000000000000000000000",
                "delta": "0",
                "tolerance": "10000000000000000000000",
                "conserved": True
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate NeuralGCM Forecast Certificates"
    )
    parser.add_argument("certificate", nargs="?", help="Path to certificate JSON file")
    parser.add_argument("--demo", action="store_true", help="Run demo validation")
    parser.add_argument("--demo-failure", action="store_true", help="Run demo failure validation")
    parser.add_argument("--strict", action="store_true", default=True, help="Strict mode (no floats)")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3],
                       help="Validation level (1=schema, 2=consistency, 3=recompute)")

    args = parser.parse_args()

    if args.demo:
        print("=== Demo NeuralGCM Success Certificate Validation ===\n")
        cert = create_demo_certificate()
        print("Certificate:")
        print(json.dumps(cert, indent=2))
        print()

        validator = NeuralGCMCertificateValidator(strict=args.strict)
        level = ValidationLevel(args.level)
        report = validator.validate(cert, level=level)

        print(report.summary())
        sys.exit(0 if report.all_passed else 1)

    if args.demo_failure:
        print("=== Demo NeuralGCM Failure Certificate Validation ===\n")
        cert = create_demo_failure_certificate()
        print("Certificate:")
        print(json.dumps(cert, indent=2))
        print()

        validator = NeuralGCMCertificateValidator(strict=args.strict)
        level = ValidationLevel(args.level)
        report = validator.validate(cert, level=level)

        print(report.summary())
        sys.exit(0 if report.all_passed else 1)

    if not args.certificate:
        parser.error("Certificate path required (or use --demo / --demo-failure)")

    path = Path(args.certificate)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    level = ValidationLevel(args.level)
    validator = NeuralGCMCertificateValidator(strict=args.strict)
    report = validator.validate(data, level=level)

    print(report.summary())
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
