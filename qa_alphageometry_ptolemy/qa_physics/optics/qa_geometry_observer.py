"""
GeometryAngleObserver - First nontrivial observer for reflection laws.

Uses rational geometry (spreads, quadrances) to compute angle-like
observables from exact invariants, then projects to continuous angles.

Key insight: Use spread = (sin θ)² which stays rational for many cases,
delaying float approximation as long as possible.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from qa_physics.projection.qa_observer import QAObserver, Observation
from qa_physics.optics.qa_reflection_problem import QARational


def _rational_to_float(r: tuple[int, int]) -> float:
    """Convert (n,d) tuple to float."""
    return r[0] / r[1]


def _quadrance_to_distance(Q_tuple: tuple[int, int]) -> float:
    """Convert quadrance (squared distance) to distance."""
    Q = _rational_to_float(Q_tuple)
    return math.sqrt(Q)


def _compute_spread(Q1: tuple[int, int], Q2: tuple[int, int], Q3: tuple[int, int]) -> float:
    """
    Compute spread for triangle with quadrances Q1, Q2, Q3.

    Spread s = (spread opposite to Q3) using rational geometry formula:
    s = 1 - ((Q1 + Q2 - Q3)² / (4*Q1*Q2))

    This is exact for rational quadrances, but we convert to float for final output.
    """
    q1 = _rational_to_float(Q1)
    q2 = _rational_to_float(Q2)
    q3 = _rational_to_float(Q3)

    # Avoid division by zero
    if q1 == 0 or q2 == 0:
        return 0.0

    # Spread formula from rational trigonometry
    numerator = (q1 + q2 - q3) ** 2
    denominator = 4 * q1 * q2

    spread = 1.0 - (numerator / denominator)

    # Clamp to [0, 1] to handle numerical errors
    return max(0.0, min(1.0, spread))


def _spread_to_angle_degrees(spread: float) -> float:
    """
    Convert spread (sin²θ) to angle in degrees.

    θ = arcsin(√s)
    """
    sin_theta = math.sqrt(max(0.0, spread))
    theta_rad = math.asin(min(1.0, sin_theta))
    return math.degrees(theta_rad)


class GeometryAngleObserver(QAObserver):
    """
    Observer that computes geometric angles from exact invariants.

    Strategy:
    1. Extract quadrances from invariants (exact)
    2. Compute spreads using rational geometry (exact formula, float arithmetic)
    3. Project spreads to angles (continuous approximation)
    4. Check reflection law: angle_incidence = angle_reflection

    This observer tests: "Does this projection preserve the law of reflection?"
    """

    name = "GeometryAngleObserver"
    version = "0.1"

    def __init__(self, angle_tolerance_deg: float = 0.1):
        """
        Args:
            angle_tolerance_deg: Tolerance for angle equality (degrees)
        """
        self.angle_tolerance_deg = angle_tolerance_deg

    def unit_system(self) -> str:
        return "degrees + dimensionless_spread"

    def time_model(self) -> str:
        return "affine: t = 1.0*k + 0.0"

    def project_time(self, k_edges: int, context: Optional[Dict[str, Any]] = None) -> float:
        return float(k_edges)

    def project_state(self, qa_state: Any) -> Observation:
        """
        Project ReflectionState to geometric observables.

        Computes:
        - Spreads: s_incidence, s_reflection
        - Angles: θ_incidence, θ_reflection (degrees)
        - Reflection law check: |θ_i - θ_r| < tolerance
        """
        inv = qa_state.to_invariants()

        # Check for failures
        if "fail_type" in inv:
            return Observation(
                observables={
                    "fail_type": inv["fail_type"],
                    "fail_detail": inv.get("fail_detail", ""),
                },
                units={},
                metadata={"projection": "geometry_angle_failed"},
            )

        # Extract quadrances
        Q_SM = inv["Q_SM"]  # Source to Mirror
        Q_MT = inv["Q_MT"]  # Mirror to Target
        Q_ST = inv["Q_ST"]  # Source to Target

        # Extract points for normal vector computation
        S = inv["S"]  # {x: (n,d), y: (n,d)}
        M = inv["M"]
        T = inv["T"]
        mirror = inv["mirror"]  # {A, B, C}

        # Compute vectors SM and MT
        # SM = M - S, MT = T - M
        SM_x = _rational_to_float(M["x"]) - _rational_to_float(S["x"])
        SM_y = _rational_to_float(M["y"]) - _rational_to_float(S["y"])

        MT_x = _rational_to_float(T["x"]) - _rational_to_float(M["x"])
        MT_y = _rational_to_float(T["y"]) - _rational_to_float(M["y"])

        # Mirror normal: (A, B) from A*x + B*y + C = 0
        N_x = float(mirror["A"])
        N_y = float(mirror["B"])
        N_mag = math.sqrt(N_x**2 + N_y**2)

        if N_mag == 0:
            return Observation(
                observables={"error": "degenerate_mirror_normal"},
                units={},
                metadata={"projection": "geometry_angle_error"},
            )

        # Normalize
        N_x /= N_mag
        N_y /= N_mag

        # Compute angles with normal using dot product
        # For incident ray (pointing toward mirror): -SM direction
        # For reflected ray (pointing away from mirror): MT direction

        SM_mag = math.sqrt(SM_x**2 + SM_y**2)
        MT_mag = math.sqrt(MT_x**2 + MT_y**2)

        if SM_mag == 0 or MT_mag == 0:
            return Observation(
                observables={"error": "degenerate_ray"},
                units={},
                metadata={"projection": "geometry_angle_error"},
            )

        # Incident ray direction (toward mirror): -SM
        I_x = -SM_x / SM_mag
        I_y = -SM_y / SM_mag

        # Reflected ray direction (away from mirror): MT
        R_x = MT_x / MT_mag
        R_y = MT_y / MT_mag

        # Angle with normal: θ = arccos(|ray · normal|)
        # We use absolute value because we care about angle magnitude
        cos_theta_i = abs(I_x * N_x + I_y * N_y)
        cos_theta_r = abs(R_x * N_x + R_y * N_y)

        # Clamp to valid range
        cos_theta_i = max(-1.0, min(1.0, cos_theta_i))
        cos_theta_r = max(-1.0, min(1.0, cos_theta_r))

        theta_i_rad = math.acos(cos_theta_i)
        theta_r_rad = math.acos(cos_theta_r)

        theta_i_deg = math.degrees(theta_i_rad)
        theta_r_deg = math.degrees(theta_r_rad)

        # Compute spreads (sin²θ)
        spread_i = (math.sin(theta_i_rad)) ** 2
        spread_r = (math.sin(theta_r_rad)) ** 2

        # Check reflection law
        angle_diff = abs(theta_i_deg - theta_r_deg)
        reflection_law_holds = angle_diff < self.angle_tolerance_deg

        observables = {
            "u": inv["u"],
            "theta_incidence_deg": theta_i_deg,
            "theta_reflection_deg": theta_r_deg,
            "spread_incidence": spread_i,
            "spread_reflection": spread_r,
            "angle_difference_deg": angle_diff,
            "reflection_law_holds": reflection_law_holds,
            "Q_SM": _rational_to_float(Q_SM),
            "Q_MT": _rational_to_float(Q_MT),
            "Q_ST": _rational_to_float(Q_ST),
        }

        return Observation(
            observables=observables,
            units={
                "theta_incidence_deg": "degrees",
                "theta_reflection_deg": "degrees",
                "angle_difference_deg": "degrees",
                "spread_incidence": "dimensionless",
                "spread_reflection": "dimensionless",
            },
            metadata={
                "projection": "geometry_angle",
                "tolerance_deg": self.angle_tolerance_deg,
            },
        )

    def project_path(self, qa_path: List[Any]) -> Observation:
        """Project path - for now just return path length and time."""
        k = self.qa_duration(qa_path)
        t = self.project_time(k)

        return Observation(
            observables={
                "k_edges": k,
                "t_obs": t,
                "path_len_nodes": len(qa_path),
            },
            units={"t_obs": "seconds"},
            metadata={"projection": "geometry_angle_path"},
        )

    def preserves_symmetry(self) -> bool:
        # Angles should be scale-invariant under λ
        return True

    def preserves_topology(self) -> bool:
        # Many-to-one projection (angles lose some position info)
        return False

    def preserves_failure_semantics(self) -> bool:
        # Failures map to observational errors
        return True

    def validate(self) -> None:
        super().validate()
        assert self.angle_tolerance_deg > 0, "angle_tolerance_deg must be positive"
