#!/usr/bin/env python3
"""
qa_generalization_validator_v3.py

Strict validator for QA Generalization Bound Certificates.
Based on the certificate schema in qa_generalization_certificate.py.

Validation levels:
- Level 1 (Schema): Required fields present, correct types
- Level 2 (Consistency): Internal arithmetic checks
- Level 3 (Recompute): Recompute witnesses from raw data

Usage:
    python qa_generalization_validator_v3.py certificate.json
    python qa_generalization_validator_v3.py --demo
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

class GeneralizationCertificateValidator:
    """Strict validator for generalization bound certificates."""

    REQUIRED_FIELDS = ["certificate_id", "schema", "success"]

    SUCCESS_FIELDS = [
        "metric_geometry",
        "operator_norms",
        "activation_regularity",
        "generalization_bound",
    ]

    FAILURE_FIELDS = ["failure_mode", "failure_witness"]

    METRIC_GEOMETRY_FIELDS = [
        "data_hash", "n_samples", "input_dim",
        "min_distance", "max_distance", "mean_distance",
        "covering_number", "epsilon"
    ]

    OPERATOR_NORM_FIELDS = [
        "layer_count", "spectral_norms", "bias_norms",
        "spectral_product", "bias_sum"
    ]

    VALID_FAILURE_MODES = [
        "insufficient_samples", "data_not_separable", "metric_degeneracy",
        "norm_explosion", "spectral_overflow", "bias_overflow",
        "depth_too_shallow", "width_insufficient",
        "no_zero_loss_solution", "optimization_stuck",
        "bound_vacuous", "bound_not_computable"
    ]

    VALID_ACTIVATION_TYPES = ["relu", "leaky_relu", "gelu", "swish"]

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
            return Fraction(value).limit_denominator(10**9)
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
        if not schema.startswith("QA_GENERALIZATION"):
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

        # Validate metric_geometry structure
        if "metric_geometry" in cert and cert["metric_geometry"]:
            self._validate_metric_geometry_schema(cert["metric_geometry"])

        # Validate operator_norms structure
        if "operator_norms" in cert and cert["operator_norms"]:
            self._validate_operator_norms_schema(cert["operator_norms"])

        # Validate activation_regularity structure
        if "activation_regularity" in cert and cert["activation_regularity"]:
            self._validate_activation_schema(cert["activation_regularity"])

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

    def _validate_metric_geometry_schema(self, mg: Dict[str, Any]) -> None:
        """Validate metric_geometry structure."""
        for field in self.METRIC_GEOMETRY_FIELDS:
            if field not in mg:
                self._add_result(
                    f"schema.metric_geometry.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing metric_geometry field: {field}"
                )

        # Check n_samples > 0
        n = mg.get("n_samples", 0)
        if not isinstance(n, int) or n <= 0:
            self._add_result(
                "schema.metric_geometry.n_samples_positive",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"n_samples must be positive int: {n}"
            )

    def _validate_operator_norms_schema(self, on: Dict[str, Any]) -> None:
        """Validate operator_norms structure."""
        for field in self.OPERATOR_NORM_FIELDS:
            if field not in on:
                self._add_result(
                    f"schema.operator_norms.{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing operator_norms field: {field}"
                )

        # Check layer_count matches array lengths
        layer_count = on.get("layer_count", 0)
        spectral = on.get("spectral_norms", [])
        bias = on.get("bias_norms", [])

        if len(spectral) != layer_count:
            self._add_result(
                "schema.operator_norms.spectral_length",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"spectral_norms length ({len(spectral)}) != layer_count ({layer_count})"
            )

        if len(bias) != layer_count:
            self._add_result(
                "schema.operator_norms.bias_length",
                ValidationLevel.SCHEMA,
                ValidationStatus.FAILED,
                f"bias_norms length ({len(bias)}) != layer_count ({layer_count})"
            )

    def _validate_activation_schema(self, ar: Dict[str, Any]) -> None:
        """Validate activation_regularity structure."""
        act_type = ar.get("activation_type")
        if act_type and act_type not in self.VALID_ACTIVATION_TYPES:
            self._add_result(
                "schema.activation.type_valid",
                ValidationLevel.SCHEMA,
                ValidationStatus.WARNING,
                f"Unknown activation type: {act_type}"
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

        # Check gap computation
        self._check_gap_computation(cert)

        # Check tracking error
        self._check_tracking_error(cert)

        # Check bound validity
        self._check_bound_validity(cert)

        # Check operator norm product
        if "operator_norms" in cert and cert["operator_norms"]:
            self._check_operator_norm_product(cert["operator_norms"])
            self._check_operator_norm_sum(cert["operator_norms"])

        # Check metric geometry consistency
        if "metric_geometry" in cert and cert["metric_geometry"]:
            self._check_metric_geometry_consistency(cert["metric_geometry"])

    def _check_gap_computation(self, cert: Dict[str, Any]) -> None:
        """Verify empirical_gap = test_loss - train_loss."""
        train = cert.get("empirical_train_loss")
        test = cert.get("empirical_test_loss")
        gap = cert.get("empirical_gap")

        if train is None or test is None or gap is None:
            self._add_result(
                "consistency.gap_computation",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.SKIPPED,
                "Gap computation skipped (missing data)"
            )
            return

        try:
            train_f = self._parse_scalar(train)
            test_f = self._parse_scalar(test)
            gap_f = self._parse_scalar(gap)
            computed = test_f - train_f

            if computed == gap_f:
                self._add_result(
                    "consistency.gap_computation",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Gap verified: {test_f} - {train_f} = {gap_f}"
                )
            else:
                self._add_result(
                    "consistency.gap_computation",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Gap mismatch: {test_f} - {train_f} = {computed} != {gap_f}"
                )
        except Exception as e:
            self._add_result(
                "consistency.gap_computation",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Gap computation error: {e}"
            )

    def _check_tracking_error(self, cert: Dict[str, Any]) -> None:
        """Verify tracking_error = bound - empirical_gap."""
        bound = cert.get("generalization_bound")
        gap = cert.get("empirical_gap")
        tracking = cert.get("tracking_error")

        if bound is None or gap is None or tracking is None:
            self._add_result(
                "consistency.tracking_error",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.SKIPPED,
                "Tracking error skipped (missing data)"
            )
            return

        try:
            bound_f = self._parse_scalar(bound)
            gap_f = self._parse_scalar(gap)
            tracking_f = self._parse_scalar(tracking)
            computed = bound_f - gap_f

            if computed == tracking_f:
                self._add_result(
                    "consistency.tracking_error",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Tracking error verified: {bound_f} - {gap_f} = {tracking_f}"
                )
            else:
                self._add_result(
                    "consistency.tracking_error",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Tracking error mismatch: {bound_f} - {gap_f} = {computed} != {tracking_f}"
                )
        except Exception as e:
            self._add_result(
                "consistency.tracking_error",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Tracking error computation error: {e}"
            )

    def _check_bound_validity(self, cert: Dict[str, Any]) -> None:
        """Verify bound >= empirical_gap (non-vacuous)."""
        bound = cert.get("generalization_bound")
        gap = cert.get("empirical_gap")

        if bound is None or gap is None:
            self._add_result(
                "consistency.bound_validity",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.SKIPPED,
                "Bound validity skipped (missing data)"
            )
            return

        try:
            bound_f = self._parse_scalar(bound)
            gap_f = self._parse_scalar(gap)

            if bound_f >= gap_f:
                self._add_result(
                    "consistency.bound_validity",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Bound valid: {bound_f} >= {gap_f}"
                )
            else:
                self._add_result(
                    "consistency.bound_validity",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"VACUOUS BOUND: {bound_f} < {gap_f}"
                )
        except Exception as e:
            self._add_result(
                "consistency.bound_validity",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Bound validity error: {e}"
            )

    def _check_operator_norm_product(self, on: Dict[str, Any]) -> None:
        """Verify spectral_product = prod(spectral_norms)."""
        spectral = on.get("spectral_norms", [])
        product = on.get("spectral_product")

        if not spectral or product is None:
            return

        try:
            computed = Fraction(1)
            for s in spectral:
                computed *= self._parse_scalar(s)
            product_f = self._parse_scalar(product)

            if computed == product_f:
                self._add_result(
                    "consistency.operator_norm_product",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Spectral product verified: {product_f}"
                )
            else:
                self._add_result(
                    "consistency.operator_norm_product",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Spectral product mismatch: computed {computed} != claimed {product_f}"
                )
        except Exception as e:
            self._add_result(
                "consistency.operator_norm_product",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Spectral product error: {e}"
            )

    def _check_operator_norm_sum(self, on: Dict[str, Any]) -> None:
        """Verify bias_sum = sum(bias_norms)."""
        bias = on.get("bias_norms", [])
        bias_sum = on.get("bias_sum")

        if not bias or bias_sum is None:
            return

        try:
            computed = sum(self._parse_scalar(b) for b in bias)
            sum_f = self._parse_scalar(bias_sum)

            if computed == sum_f:
                self._add_result(
                    "consistency.operator_norm_sum",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Bias sum verified: {sum_f}"
                )
            else:
                self._add_result(
                    "consistency.operator_norm_sum",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Bias sum mismatch: computed {computed} != claimed {sum_f}"
                )
        except Exception as e:
            self._add_result(
                "consistency.operator_norm_sum",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Bias sum error: {e}"
            )

    def _check_metric_geometry_consistency(self, mg: Dict[str, Any]) -> None:
        """Check metric geometry constraints."""
        min_d = mg.get("min_distance")
        max_d = mg.get("max_distance")
        mean_d = mg.get("mean_distance")

        if min_d is None or max_d is None:
            return

        try:
            min_f = self._parse_scalar(min_d)
            max_f = self._parse_scalar(max_d)

            if min_f <= max_f:
                self._add_result(
                    "consistency.metric_min_max",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"min_distance <= max_distance: {min_f} <= {max_f}"
                )
            else:
                self._add_result(
                    "consistency.metric_min_max",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"min_distance > max_distance: {min_f} > {max_f}"
                )

            if mean_d is not None:
                mean_f = self._parse_scalar(mean_d)
                if min_f <= mean_f <= max_f:
                    self._add_result(
                        "consistency.metric_mean_bounds",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.PASSED,
                        f"mean_distance in [min, max]: {mean_f}"
                    )
                else:
                    self._add_result(
                        "consistency.metric_mean_bounds",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        f"mean_distance outside [min, max]: {mean_f}"
                    )
        except Exception as e:
            self._add_result(
                "consistency.metric_geometry",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Metric geometry error: {e}"
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

        # TODO: Implement recompute hooks
        self._add_result(
            "recompute.not_implemented",
            ValidationLevel.RECOMPUTE,
            ValidationStatus.WARNING,
            "Recompute hooks not yet implemented"
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
# BUNDLE VALIDATOR
# ============================================================================

class GeneralizationBundleValidator:
    """Validator for certificate bundles."""

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.cert_validator = GeneralizationCertificateValidator(strict=strict)

    def validate_bundle(self, bundle: Dict[str, Any]) -> ValidationReport:
        """Validate a certificate bundle."""
        results: List[ValidationResult] = []

        # Check bundle structure
        if "main_certificate" not in bundle:
            results.append(ValidationResult(
                check_name="bundle.main_certificate",
                level=ValidationLevel.SCHEMA,
                status=ValidationStatus.FAILED,
                message="Bundle missing main_certificate"
            ))
            return ValidationReport(
                certificate_id=bundle.get("bundle_id", "unknown"),
                schema=bundle.get("schema", "unknown"),
                results=results
            )

        # Validate main certificate
        main_report = self.cert_validator.validate(bundle["main_certificate"])
        results.extend(main_report.results)

        # Validate zero_loss_constructor if present
        if "zero_loss_constructor" in bundle and bundle["zero_loss_constructor"]:
            zlc = bundle["zero_loss_constructor"]
            self._validate_zero_loss_constructor(zlc, results)

        # Validate gauge_freedom if present
        if "gauge_freedom" in bundle and bundle["gauge_freedom"]:
            gf = bundle["gauge_freedom"]
            self._validate_gauge_freedom(gf, results)

        # Validate architecture_experiments if present
        if "architecture_experiments" in bundle and bundle["architecture_experiments"]:
            for i, exp in enumerate(bundle["architecture_experiments"]):
                exp_report = self.cert_validator.validate(exp)
                for r in exp_report.results:
                    r.check_name = f"experiment[{i}].{r.check_name}"
                results.extend(exp_report.results)

        return ValidationReport(
            certificate_id=bundle.get("bundle_id", "unknown"),
            schema=bundle.get("schema", "unknown"),
            results=results
        )

    def _validate_zero_loss_constructor(self, zlc: Dict[str, Any], results: List[ValidationResult]) -> None:
        """Validate zero-loss constructor certificate."""
        # Check n <= d
        n = zlc.get("n_samples", 0)
        d = zlc.get("input_dim", 0)

        if n <= d:
            results.append(ValidationResult(
                check_name="zero_loss.n_leq_d",
                level=ValidationLevel.CONSISTENCY,
                status=ValidationStatus.PASSED,
                message=f"n <= d verified: {n} <= {d}"
            ))
        else:
            results.append(ValidationResult(
                check_name="zero_loss.n_leq_d",
                level=ValidationLevel.CONSISTENCY,
                status=ValidationStatus.WARNING,
                message=f"n > d: {n} > {d} (zero-loss may not be guaranteed)"
            ))

        # Check residuals are zero
        residuals = zlc.get("residuals", [])
        if residuals:
            all_zero = all(Fraction(r) == 0 for r in residuals)
            if all_zero:
                results.append(ValidationResult(
                    check_name="zero_loss.residuals_zero",
                    level=ValidationLevel.CONSISTENCY,
                    status=ValidationStatus.PASSED,
                    message="All residuals are exactly zero"
                ))
            else:
                results.append(ValidationResult(
                    check_name="zero_loss.residuals_zero",
                    level=ValidationLevel.CONSISTENCY,
                    status=ValidationStatus.FAILED,
                    message="Non-zero residuals found"
                ))

    def _validate_gauge_freedom(self, gf: Dict[str, Any], results: List[ValidationResult]) -> None:
        """Validate gauge freedom witness."""
        total = gf.get("total_params", 0)
        minimal = gf.get("minimal_params", 0)
        gauge_dim = gf.get("gauge_dim", 0)

        computed = total - minimal
        if computed == gauge_dim:
            results.append(ValidationResult(
                check_name="gauge_freedom.dim",
                level=ValidationLevel.CONSISTENCY,
                status=ValidationStatus.PASSED,
                message=f"Gauge dim verified: {total} - {minimal} = {gauge_dim}"
            ))
        else:
            results.append(ValidationResult(
                check_name="gauge_freedom.dim",
                level=ValidationLevel.CONSISTENCY,
                status=ValidationStatus.FAILED,
                message=f"Gauge dim mismatch: {total} - {minimal} = {computed} != {gauge_dim}"
            ))


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_demo_certificate() -> Dict[str, Any]:
    """Create a demo certificate for testing."""
    return {
        "certificate_id": "demo_mnist_mlp_001",
        "version": "1.0.0",
        "schema": "QA_GENERALIZATION_CERT_V1",
        "success": True,
        "metric_geometry": {
            "data_hash": "mnist_train_sha256_abc123def456",
            "n_samples": 60000,
            "input_dim": 784,
            "min_distance": "1/100",
            "max_distance": "50/1",
            "mean_distance": "15/1",
            "covering_number": 1000,
            "epsilon": "1/10"
        },
        "operator_norms": {
            "layer_count": 3,
            "spectral_norms": ["2/1", "3/2", "1/1"],
            "bias_norms": ["1/10", "1/10", "1/10"],
            "spectral_product": "3/1",
            "bias_sum": "3/10"
        },
        "activation_regularity": {
            "activation_type": "relu",
            "lipschitz_constant": "1"
        },
        "generalization_bound": "22/100",
        "empirical_train_loss": "1/100",
        "empirical_test_loss": "3/100",
        "empirical_gap": "2/100",
        "tracking_error": "20/100"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate QA Generalization Bound Certificates"
    )
    parser.add_argument("certificate", nargs="?", help="Path to certificate JSON file")
    parser.add_argument("--demo", action="store_true", help="Run demo validation")
    parser.add_argument("--strict", action="store_true", default=True, help="Strict mode (no floats)")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3],
                       help="Validation level (1=schema, 2=consistency, 3=recompute)")
    parser.add_argument("--bundle", action="store_true", help="Validate as bundle")

    args = parser.parse_args()

    if args.demo:
        print("=== Demo Certificate Validation ===\n")
        cert = create_demo_certificate()
        print("Certificate:")
        print(json.dumps(cert, indent=2))
        print()

        validator = GeneralizationCertificateValidator(strict=args.strict)
        level = ValidationLevel(args.level)
        report = validator.validate(cert, level=level)

        print(report.summary())
        sys.exit(0 if report.all_passed else 1)

    if not args.certificate:
        parser.error("Certificate path required (or use --demo)")

    path = Path(args.certificate)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    level = ValidationLevel(args.level)

    if args.bundle:
        validator = GeneralizationBundleValidator(strict=args.strict)
        report = validator.validate_bundle(data)
    else:
        validator = GeneralizationCertificateValidator(strict=args.strict)
        report = validator.validate(data, level=level)

    print(report.summary())
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
