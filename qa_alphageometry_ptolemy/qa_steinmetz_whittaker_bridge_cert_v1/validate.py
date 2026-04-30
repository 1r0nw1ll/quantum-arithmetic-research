#!/usr/bin/env python3
"""Validator for QA_STEINMETZ_WHITTAKER_BRIDGE_CERT.v1."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


CERT_FAMILY = "QA_STEINMETZ_WHITTAKER_BRIDGE_CERT.v1"
CANONICAL_GUARDRAIL = (
    "This cert validates deterministic transform consistency only; it does not "
    "validate or prove a universal physical identity between Steinmetz, "
    "Whittaker, Dollard, Bearden, or QA."
)


class ValidationError(Exception):
    """Raised when a bridge fixture fails validation."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def require_mapping(obj: Any, name: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValidationError(f"{name} must be an object")
    return obj


def require_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValidationError(f"{name} must be finite")
    return float(value)


def require_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer")
    return value


def require_number_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list")
    return [require_number(item, f"{name}[{idx}]") for idx, item in enumerate(value)]


def compute_invariants(tuple_obj: dict[str, Any]) -> dict[str, int]:
    b = require_int(tuple_obj.get("b"), "tuple.b")
    e = require_int(tuple_obj.get("e"), "tuple.e")
    d = require_int(tuple_obj.get("d"), "tuple.d")
    a = require_int(tuple_obj.get("a"), "tuple.a")

    if d != b + e:
        raise ValidationError(f"tuple relation failed: d={d} but b+e={b + e}")
    if a != b + 2 * e:
        raise ValidationError(f"tuple relation failed: a={a} but b+2*e={b + 2 * e}")

    return {
        "J": b * d,
        "X": d * e,
        "K": d * a,
        "F": b * a,
        "C": 2 * e * d,
        "G": e * e + d * d,
    }


def check_declared_invariants(actual: dict[str, int], declared_obj: dict[str, Any]) -> None:
    for key, expected in actual.items():
        declared = require_int(declared_obj.get(key), f"declared_invariants.{key}")
        if declared != expected:
            raise ValidationError(
                f"invariant {key} mismatch: declared {declared}, expected {expected}"
            )


def compare_calibration(calibration: dict[str, Any], evaluation: dict[str, Any]) -> None:
    cal_convention = calibration.get("material_drive_convention")
    eval_convention = evaluation.get("material_drive_convention")
    if cal_convention != eval_convention:
        raise ValidationError(
            "material_drive_convention changed between calibration and evaluation"
        )

    cal_constants = require_mapping(
        calibration.get("calibration_constants"), "calibration.calibration_constants"
    )
    eval_constants = require_mapping(
        evaluation.get("calibration_constants"), "evaluation.calibration_constants"
    )
    if cal_constants != eval_constants:
        raise ValidationError(
            "calibration constants changed between calibration and evaluation"
        )


def loop_integral_h_db(h_values: list[float], b_values: list[float]) -> float:
    if len(h_values) != len(b_values):
        raise ValidationError("H and B must have the same length")
    if len(h_values) < 2:
        raise ValidationError("H and B must contain at least two samples")

    total = 0.0
    for idx in range(len(h_values) - 1):
        total += 0.5 * (h_values[idx] + h_values[idx + 1]) * (
            b_values[idx + 1] - b_values[idx]
        )
    if h_values[0] != h_values[-1] or b_values[0] != b_values[-1]:
        total += 0.5 * (h_values[-1] + h_values[0]) * (b_values[0] - b_values[-1])
    return total


def curvature_proxy(
    theta_values: list[float],
    invariants: dict[str, int],
    constants: dict[str, Any],
    pi_series: list[float] | None = None,
) -> float:
    if len(theta_values) < 2:
        raise ValidationError("theta must contain at least two samples")
    if pi_series is not None:
        if len(pi_series) != len(theta_values):
            raise ValidationError("pi_series and theta must have the same length")
        total = 0.0
        for idx in range(len(theta_values) - 1):
            total += 0.5 * (pi_series[idx] + pi_series[idx + 1]) * (
                theta_values[idx + 1] - theta_values[idx]
            )
        return total

    alpha_x = require_number(constants.get("alpha_X"), "calibration_constants.alpha_X")
    alpha_j = require_number(constants.get("alpha_J"), "calibration_constants.alpha_J")
    alpha_k = require_number(constants.get("alpha_K"), "calibration_constants.alpha_K")
    pi_value = alpha_x * invariants["X"] + alpha_j * invariants["J"] + alpha_k * invariants["K"]

    total = 0.0
    for idx in range(len(theta_values) - 1):
        total += pi_value * (theta_values[idx + 1] - theta_values[idx])
    return total


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise ValidationError(
            f"{label} mismatch: actual {actual}, expected {expected}, tolerance {tolerance}"
        )


def validate_fixture(data: dict[str, Any]) -> str:
    if data.get("cert_family") != CERT_FAMILY:
        raise ValidationError(f"cert_family must be {CERT_FAMILY}")

    tolerance = require_number(data.get("tolerance"), "tolerance")
    if tolerance < 0:
        raise ValidationError("tolerance must be non-negative")

    guardrail = data.get("guardrail")
    if guardrail != CANONICAL_GUARDRAIL:
        raise ValidationError("guardrail text must match the canonical non-claim statement")

    tuple_obj = require_mapping(data.get("tuple"), "tuple")
    invariants = compute_invariants(tuple_obj)
    declared_invariants = require_mapping(
        data.get("declared_invariants"), "declared_invariants"
    )
    check_declared_invariants(invariants, declared_invariants)

    calibration = require_mapping(data.get("calibration"), "calibration")
    evaluation = require_mapping(data.get("evaluation"), "evaluation")
    compare_calibration(calibration, evaluation)

    h_values = require_number_list(evaluation.get("H"), "evaluation.H")
    b_values = require_number_list(evaluation.get("B"), "evaluation.B")
    theta_values = require_number_list(evaluation.get("theta"), "evaluation.theta")
    if len(theta_values) != len(h_values):
        raise ValidationError("theta and H must have the same length")

    area = loop_integral_h_db(h_values, b_values)
    constants = require_mapping(
        calibration.get("calibration_constants"), "calibration.calibration_constants"
    )
    pi_series = None
    if "pi_series" in evaluation:
        pi_series = require_number_list(evaluation.get("pi_series"), "evaluation.pi_series")
    curvature = curvature_proxy(theta_values, invariants, constants, pi_series)

    expected_area = require_number(
        evaluation.get("expected_hysteresis_area"),
        "evaluation.expected_hysteresis_area",
    )
    expected_curvature = require_number(
        evaluation.get("expected_curvature_proxy"),
        "evaluation.expected_curvature_proxy",
    )

    assert_close(area, expected_area, tolerance, "hysteresis area")
    assert_close(curvature, expected_curvature, tolerance, "curvature proxy")
    assert_close(area, curvature, tolerance, "bridge transform")

    return f"transform consistent: area={area}, curvature_proxy={curvature}"


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"could not read fixture: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc
    return require_mapping(data, "fixture")


def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    pass_fixture = load_json(base / "fixtures" / "pass_minimal_loop.json")
    variable_pi_fixture = load_json(base / "fixtures" / "pass_variable_pi_loop.json")
    fail_fixture = load_json(base / "fixtures" / "fail_changed_calibration.json")

    validate_fixture(pass_fixture)
    validate_fixture(variable_pi_fixture)
    try:
        validate_fixture(fail_fixture)
    except ValidationError as exc:
        if "calibration constants changed" not in str(exc):
            raise
    else:
        raise ValidationError("fail_changed_calibration unexpectedly passed")

    print(canonical_json({"ok": True}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate QA_STEINMETZ_WHITTAKER_BRIDGE_CERT.v1 fixtures"
    )
    parser.add_argument("fixture", nargs="?", help="Path to fixture JSON")
    parser.add_argument("--self-test", action="store_true", help="Run bundled self-test")
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            return run_self_test()
        if not args.fixture:
            raise ValidationError("fixture path is required unless --self-test is used")
        reason = validate_fixture(load_json(Path(args.fixture)))
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"PASS: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
