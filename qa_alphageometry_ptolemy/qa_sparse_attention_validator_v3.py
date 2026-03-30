#!/usr/bin/env python3
"""
qa_sparse_attention_validator_v3.py

Strict validator for Sparse Attention Certificates.
Based on the certificate schema in qa_sparse_attention_certificate.py.

Validation levels:
- Level 1 (Schema): Required fields present, correct types
- Level 2 (Consistency): Internal arithmetic checks (entropy, rank, sparsity)
- Level 3 (Recompute): Recompute witnesses from raw attention matrices

Usage:
    python qa_sparse_attention_validator_v3.py certificate.json
    python qa_sparse_attention_validator_v3.py --demo
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

class SparseAttentionValidator:
    """Strict validator for sparse attention certificates."""

    REQUIRED_FIELDS = ["certificate_id", "schema", "success"]

    SUCCESS_FIELDS = ["entropy_witnesses", "rank_witnesses"]

    FAILURE_FIELDS = ["failure_mode", "failure_witness"]

    VALID_FAILURE_MODES = [
        "entropy_collapse", "entropy_uniform", "entropy_unstable",
        "rank_collapse", "representation_collapse",
        "disconnected_tokens", "sparsity_too_aggressive",
        "gradient_vanishing", "gradient_exploding", "residual_dominated",
        "linear_approximation_error"
    ]

    VALID_SPARSITY_PATTERNS = ["dense", "local", "strided", "random", "combined", "learned"]

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
        if not schema.startswith("QA_SPARSE_ATTENTION"):
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

        # Validate entropy witnesses
        if "entropy_witnesses" in cert and cert["entropy_witnesses"]:
            for i, ew in enumerate(cert["entropy_witnesses"]):
                self._validate_entropy_witness_schema(ew, i)

        # Validate rank witnesses
        if "rank_witnesses" in cert and cert["rank_witnesses"]:
            for i, rw in enumerate(cert["rank_witnesses"]):
                self._validate_rank_witness_schema(rw, i)

        # Validate sparsity witness
        if "sparsity_witness" in cert and cert["sparsity_witness"]:
            self._validate_sparsity_schema(cert["sparsity_witness"])

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

    def _validate_entropy_witness_schema(self, ew: Dict[str, Any], index: int) -> None:
        """Validate entropy witness schema."""
        required = ["layer", "head", "mean_entropy", "max_possible_entropy",
                   "normalized_entropy", "entropy_healthy"]
        for field in required:
            if field not in ew:
                self._add_result(
                    f"schema.entropy_witness[{index}].{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing field: {field}"
                )

    def _validate_rank_witness_schema(self, rw: Dict[str, Any], index: int) -> None:
        """Validate rank witness schema."""
        required = ["layer", "sequence_length", "effective_rank", "rank_ratio", "rank_healthy"]
        for field in required:
            if field not in rw:
                self._add_result(
                    f"schema.rank_witness[{index}].{field}",
                    ValidationLevel.SCHEMA,
                    ValidationStatus.FAILED,
                    f"Missing field: {field}"
                )

    def _validate_sparsity_schema(self, sp: Dict[str, Any]) -> None:
        """Validate sparsity witness schema."""
        pattern = sp.get("pattern_type")
        if pattern and pattern not in self.VALID_SPARSITY_PATTERNS:
            self._add_result(
                "schema.sparsity.pattern_type",
                ValidationLevel.SCHEMA,
                ValidationStatus.WARNING,
                f"Unknown sparsity pattern: {pattern}"
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

        # Check entropy consistency
        if "entropy_witnesses" in cert and cert["entropy_witnesses"]:
            for i, ew in enumerate(cert["entropy_witnesses"]):
                self._check_entropy_consistency(ew, i)

        # Check rank consistency
        if "rank_witnesses" in cert and cert["rank_witnesses"]:
            for i, rw in enumerate(cert["rank_witnesses"]):
                self._check_rank_consistency(rw, i)

        # Check sparsity consistency
        if "sparsity_witness" in cert and cert["sparsity_witness"]:
            self._check_sparsity_consistency(cert["sparsity_witness"])

        # Check head redundancy consistency
        if "head_redundancy" in cert and cert["head_redundancy"]:
            for i, hr in enumerate(cert["head_redundancy"]):
                self._check_head_redundancy_consistency(hr, i)

    def _check_entropy_consistency(self, ew: Dict[str, Any], index: int) -> None:
        """Check entropy witness arithmetic."""
        try:
            mean = self._parse_scalar(ew.get("mean_entropy", 0))
            max_possible = self._parse_scalar(ew.get("max_possible_entropy", 1))
            normalized = self._parse_scalar(ew.get("normalized_entropy", 0))
            healthy = ew.get("entropy_healthy", True)
            collapse_thresh = self._parse_scalar(ew.get("collapse_threshold", "1/10"))
            uniform_thresh = self._parse_scalar(ew.get("uniform_threshold", "9/10"))

            # Check normalized computation
            if max_possible != 0:
                computed_norm = mean / max_possible
                diff = abs(computed_norm - normalized)
                if diff < Fraction(1, 100):
                    self._add_result(
                        f"consistency.entropy[{index}].normalized",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.PASSED,
                        f"Normalized entropy verified: {normalized}"
                    )
                else:
                    self._add_result(
                        f"consistency.entropy[{index}].normalized",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        f"Normalized mismatch: {computed_norm} != {normalized}"
                    )

            # Check healthy flag
            should_be_healthy = normalized > collapse_thresh and normalized < uniform_thresh
            if should_be_healthy == healthy:
                self._add_result(
                    f"consistency.entropy[{index}].healthy",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Entropy health verified: {healthy}"
                )
            else:
                self._add_result(
                    f"consistency.entropy[{index}].healthy",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Health mismatch: {collapse_thresh} < {normalized} < {uniform_thresh} is {should_be_healthy}, claimed {healthy}"
                )

        except Exception as e:
            self._add_result(
                f"consistency.entropy[{index}]",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Entropy check error: {e}"
            )

    def _check_rank_consistency(self, rw: Dict[str, Any], index: int) -> None:
        """Check rank witness arithmetic."""
        try:
            eff_rank = self._parse_scalar(rw.get("effective_rank", 0))
            seq_len = rw.get("sequence_length", 1)
            rank_ratio = self._parse_scalar(rw.get("rank_ratio", 0))
            healthy = rw.get("rank_healthy", True)
            collapse_thresh = self._parse_scalar(rw.get("collapse_threshold", "1/10"))

            # Check rank ratio
            if seq_len > 0:
                computed_ratio = eff_rank / Fraction(seq_len)
                diff = abs(computed_ratio - rank_ratio)
                if diff < Fraction(1, 100):
                    self._add_result(
                        f"consistency.rank[{index}].ratio",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.PASSED,
                        f"Rank ratio verified: {rank_ratio}"
                    )
                else:
                    self._add_result(
                        f"consistency.rank[{index}].ratio",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        f"Rank ratio mismatch: {computed_ratio} != {rank_ratio}"
                    )

            # Check healthy flag
            should_be_healthy = rank_ratio > collapse_thresh
            if should_be_healthy == healthy:
                self._add_result(
                    f"consistency.rank[{index}].healthy",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Rank health verified: {healthy}"
                )
            else:
                self._add_result(
                    f"consistency.rank[{index}].healthy",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Health mismatch: {rank_ratio} > {collapse_thresh} is {should_be_healthy}, claimed {healthy}"
                )

        except Exception as e:
            self._add_result(
                f"consistency.rank[{index}]",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Rank check error: {e}"
            )

    def _check_sparsity_consistency(self, sp: Dict[str, Any]) -> None:
        """Check sparsity witness consistency."""
        try:
            total = sp.get("total_possible_pairs", 1)
            allowed = sp.get("allowed_pairs", 0)
            ratio = self._parse_scalar(sp.get("sparsity_ratio", 0))

            if total > 0:
                computed_ratio = Fraction(allowed, total)
                diff = abs(computed_ratio - ratio)
                if diff < Fraction(1, 100):
                    self._add_result(
                        "consistency.sparsity.ratio",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.PASSED,
                        f"Sparsity ratio verified: {ratio}"
                    )
                else:
                    self._add_result(
                        "consistency.sparsity.ratio",
                        ValidationLevel.CONSISTENCY,
                        ValidationStatus.FAILED,
                        f"Sparsity ratio mismatch: {computed_ratio} != {ratio}"
                    )

        except Exception as e:
            self._add_result(
                "consistency.sparsity",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Sparsity check error: {e}"
            )

    def _check_head_redundancy_consistency(self, hr: Dict[str, Any], index: int) -> None:
        """Check head redundancy consistency."""
        try:
            total = hr.get("total_heads", 0)
            active = hr.get("active_heads", 0)
            redundant = hr.get("redundant_heads", 0)

            if redundant == total - active:
                self._add_result(
                    f"consistency.head_redundancy[{index}]",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.PASSED,
                    f"Redundant heads verified: {redundant} = {total} - {active}"
                )
            else:
                self._add_result(
                    f"consistency.head_redundancy[{index}]",
                    ValidationLevel.CONSISTENCY,
                    ValidationStatus.FAILED,
                    f"Redundant mismatch: {redundant} != {total} - {active}"
                )

        except Exception as e:
            self._add_result(
                f"consistency.head_redundancy[{index}]",
                ValidationLevel.CONSISTENCY,
                ValidationStatus.FAILED,
                f"Head redundancy error: {e}"
            )

    # ========================================================================
    # LEVEL 3: RECOMPUTE VALIDATION
    # ========================================================================

    def validate_recompute(self, cert: Dict[str, Any], data_path: Optional[Path] = None) -> None:
        """Level 3: Recompute witnesses from raw attention matrices."""
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
            "Recompute hooks not yet implemented for Sparse Attention"
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
        "certificate_id": "demo_bert_attention_001",
        "version": "1.0.0",
        "schema": "QA_SPARSE_ATTENTION_V1",
        "success": True,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "sequence_length": 512,
        "entropy_witnesses": [
            {
                "layer": 0,
                "head": 0,
                "min_entropy": "2/1",
                "max_entropy": "5/1",
                "mean_entropy": "7/2",
                "max_possible_entropy": "7/1",
                "normalized_entropy": "1/2",
                "collapse_threshold": "1/10",
                "uniform_threshold": "9/10",
                "entropy_healthy": True
            }
        ],
        "rank_witnesses": [
            {
                "layer": 0,
                "sequence_length": 512,
                "effective_rank": "180/1",
                "numerical_rank": 200,
                "rank_ratio": "180/512",
                "top_singular_value": "15/1",
                "singular_value_entropy": "4/1",
                "collapse_threshold": "1/10",
                "rank_healthy": True
            }
        ],
        "sparsity_witness": {
            "pattern_type": "local",
            "sequence_length": 512,
            "total_possible_pairs": 262144,
            "allowed_pairs": 65536,
            "sparsity_ratio": "1/4",
            "window_size": 128,
            "all_tokens_reachable": True
        },
        "head_redundancy": [
            {
                "layer": 0,
                "total_heads": 12,
                "active_heads": 8,
                "redundant_heads": 4,
                "gauge_dim": 256,
                "pruning_threshold": "1/100"
            }
        ],
        "mean_entropy": "1/2",
        "mean_rank_ratio": "180/512",
        "total_gauge_dim": 256
    }


def create_demo_failure_certificate() -> Dict[str, Any]:
    """Create a demo failure certificate for testing."""
    return {
        "certificate_id": "demo_rank_collapse_failure_001",
        "version": "1.0.0",
        "schema": "QA_SPARSE_ATTENTION_V1",
        "success": False,
        "failure_mode": "rank_collapse",
        "failure_witness": {
            "reason": "Attention matrix collapsed to near rank-1 at layer 8",
            "layer": 8,
            "effective_rank": "15/1",
            "sequence_length": 512,
            "rank_ratio": "15/512",
            "threshold": "1/10",
            "probable_cause": "Over-regularization or vanishing gradients in deep layers",
            "remediation": [
                "Add residual connections with stronger weight",
                "Use pre-layer normalization",
                "Reduce weight decay",
                "Add attention dropout"
            ]
        },
        "num_layers": 12,
        "num_heads": 12,
        "sequence_length": 512
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate Sparse Attention Certificates"
    )
    parser.add_argument("certificate", nargs="?", help="Path to certificate JSON file")
    parser.add_argument("--demo", action="store_true", help="Run demo validation")
    parser.add_argument("--demo-failure", action="store_true", help="Run demo failure validation")
    parser.add_argument("--strict", action="store_true", default=True, help="Strict mode (no floats)")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3],
                       help="Validation level (1=schema, 2=consistency, 3=recompute)")

    args = parser.parse_args()

    if args.demo:
        print("=== Demo Sparse Attention Success Certificate ===\n")
        cert = create_demo_certificate()
        print("Certificate:")
        print(json.dumps(cert, indent=2))
        print()

        validator = SparseAttentionValidator(strict=args.strict)
        level = ValidationLevel(args.level)
        report = validator.validate(cert, level=level)

        print(report.summary())
        sys.exit(0 if report.all_passed else 1)

    if args.demo_failure:
        print("=== Demo Sparse Attention Failure Certificate ===\n")
        cert = create_demo_failure_certificate()
        print("Certificate:")
        print(json.dumps(cert, indent=2))
        print()

        validator = SparseAttentionValidator(strict=args.strict)
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
    validator = SparseAttentionValidator(strict=args.strict)
    report = validator.validate(data, level=level)

    print(report.summary())
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
