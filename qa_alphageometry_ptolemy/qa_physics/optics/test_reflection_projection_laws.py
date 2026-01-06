"""
Reflection Projection Law Tests

Tests which observer projections satisfy the classical law of reflection:
    angle of incidence = angle of reflection

This is the critical physics test: we're not testing if "QA does optics",
we're testing which PROJECTIONS make optics emerge from QA's exact invariants.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List, Tuple

from qa_physics.optics.qa_reflection_problem import QARational, Point2D, LineABC, ReflectionProblem
from qa_physics.optics.qa_reflection_search import SearchConfig, generate_candidates, ReflectionCandidate
from qa_physics.optics.qa_geometry_observer import GeometryAngleObserver
from qa_physics.projection.qa_observer import NullObserver, AffineTimeGeometricObserver, QAObserver


# ----------------------------
# Test problem fixtures
# ----------------------------

@pytest.fixture
def simple_horizontal_mirror() -> ReflectionProblem:
    """
    Simple test case: horizontal mirror (y=0).

    For symmetric source/target heights, reflection point should be
    at the x-midpoint for perfect reflection.
    """
    mirror = LineABC(A=0, B=1, C=0)  # y = 0
    S = Point2D(QARational(-10, 1), QARational(5, 1))
    T = Point2D(QARational(10, 1), QARational(5, 1))  # Same height as S
    return ReflectionProblem(S=S, T=T, mirror=mirror, u_min=-20, u_max=20)


@pytest.fixture
def asymmetric_mirror() -> ReflectionProblem:
    """
    Asymmetric case: different source/target heights.

    Reflection point should not be at x-midpoint.
    """
    mirror = LineABC(A=0, B=1, C=0)  # y = 0
    S = Point2D(QARational(-10, 1), QARational(5, 1))
    T = Point2D(QARational(10, 1), QARational(3, 1))  # Different height
    return ReflectionProblem(S=S, T=T, mirror=mirror, u_min=-20, u_max=20)


@pytest.fixture
def tilted_mirror() -> ReflectionProblem:
    """
    Tilted mirror: y = x (45 degrees).

    Tests if observer handles non-axis-aligned mirrors.
    """
    mirror = LineABC(A=1, B=-1, C=0)  # x - y = 0 → y = x
    S = Point2D(QARational(0, 1), QARational(10, 1))
    T = Point2D(QARational(10, 1), QARational(0, 1))
    return ReflectionProblem(S=S, T=T, mirror=mirror, u_min=-15, u_max=15)


@pytest.fixture
def all_observers() -> List[QAObserver]:
    """All observers to test."""
    return [
        NullObserver(),
        AffineTimeGeometricObserver(alpha=1.0, beta=0.0),
        GeometryAngleObserver(angle_tolerance_deg=0.1),
        GeometryAngleObserver(angle_tolerance_deg=1.0),  # More tolerant
    ]


# ----------------------------
# Helper functions
# ----------------------------

def find_best_reflection_candidate(
    candidates: List[ReflectionCandidate],
    observer: QAObserver,
) -> Tuple[ReflectionCandidate, Dict[str, Any]]:
    """
    Find the candidate that best satisfies reflection law under this observer.

    For GeometryAngleObserver: use reflection_law_holds flag + minimum angle_difference
    For others: return candidate with minimal path length (u=0 seed)

    Returns: (best_candidate, observation_dict)
    """
    if isinstance(observer, GeometryAngleObserver):
        # Find candidate with reflection_law_holds=True and minimal angle difference
        valid_candidates = []

        for cand in candidates:
            obs = observer.project_state(cand.state)
            obs_dict = obs.observables

            if "error" in obs_dict or "fail_type" in obs_dict:
                continue

            valid_candidates.append((cand, obs_dict))

        if not valid_candidates:
            # No valid candidates - return seed
            seed_cand = next(c for c in candidates if c.u == 0)
            obs_dict = observer.project_state(seed_cand.state).observables
            return seed_cand, obs_dict

        # Sort by: reflection_law_holds (True first), then angle_difference (ascending)
        valid_candidates.sort(
            key=lambda x: (
                not x[1].get("reflection_law_holds", False),
                x[1].get("angle_difference_deg", float("inf")),
            )
        )

        return valid_candidates[0]
    else:
        # For non-geometry observers, just return seed (u=0)
        seed_cand = next(c for c in candidates if c.u == 0)
        obs_dict = observer.project_state(seed_cand.state).observables
        return seed_cand, obs_dict


# ----------------------------
# Reflection law tests
# ----------------------------

def test_geometry_observer_on_symmetric_case(simple_horizontal_mirror: ReflectionProblem):
    """
    GeometryAngleObserver should find perfect reflection on symmetric case.

    Expected: u=0 (midpoint) has θ_i = θ_r = 90° (normal incidence)
    """
    cfg = SearchConfig(u_seed=0, u_min=-20, u_max=20)
    candidates = generate_candidates(simple_horizontal_mirror, cfg)

    observer = GeometryAngleObserver(angle_tolerance_deg=0.1)
    observer.validate()

    best_cand, obs_dict = find_best_reflection_candidate(candidates, observer)

    print(f"\nBest candidate: u={best_cand.u}")
    print(f"θ_incidence: {obs_dict.get('theta_incidence_deg', 'N/A'):.2f}°")
    print(f"θ_reflection: {obs_dict.get('theta_reflection_deg', 'N/A'):.2f}°")
    print(f"Δθ: {obs_dict.get('angle_difference_deg', 'N/A'):.4f}°")
    print(f"Reflection law holds: {obs_dict.get('reflection_law_holds', False)}")

    # For symmetric case, should find a candidate satisfying reflection law
    assert obs_dict.get("reflection_law_holds", False), (
        f"GeometryAngleObserver should satisfy reflection law on symmetric case. "
        f"Best u={best_cand.u}, Δθ={obs_dict.get('angle_difference_deg', 'N/A'):.4f}°"
    )


def test_geometry_observer_on_asymmetric_case(asymmetric_mirror: ReflectionProblem):
    """
    GeometryAngleObserver should handle asymmetric case.

    NOTE: Discrete u sampling means exact reflection point may not be in grid.
    This test measures best-effort approximation rather than requiring perfection.

    Expected: Angle difference should be small (<5°) even if not within tolerance.
    """
    cfg = SearchConfig(u_seed=0, u_min=-20, u_max=20)
    candidates = generate_candidates(asymmetric_mirror, cfg)

    observer = GeometryAngleObserver(angle_tolerance_deg=1.0)  # More tolerant
    observer.validate()

    best_cand, obs_dict = find_best_reflection_candidate(candidates, observer)

    print(f"\nBest candidate: u={best_cand.u}")
    print(f"θ_incidence: {obs_dict.get('theta_incidence_deg', 'N/A'):.2f}°")
    print(f"θ_reflection: {obs_dict.get('theta_reflection_deg', 'N/A'):.2f}°")
    print(f"Δθ: {obs_dict.get('angle_difference_deg', 'N/A'):.4f}°")
    print(f"Reflection law holds: {obs_dict.get('reflection_law_holds', False)}")

    # Discrete sampling limitation: best candidate should be "close" even if not perfect
    angle_diff = obs_dict.get("angle_difference_deg", float("inf"))
    assert angle_diff < 5.0, (
        f"GeometryAngleObserver should find near-reflection on asymmetric case. "
        f"Best u={best_cand.u}, Δθ={angle_diff:.4f}° (expected < 5°)"
    )

    # Document that discrete grid may not contain exact solution
    if not obs_dict.get("reflection_law_holds", False):
        print(f"  ⚠️  Exact reflection point not in discrete grid (Δθ={angle_diff:.2f}°)")
        print(f"      This is expected - continuous solution likely between grid points")


def test_null_observer_does_not_compute_angles(simple_horizontal_mirror: ReflectionProblem):
    """
    NullObserver should NOT compute angles - it's just raw invariants.

    This is the null model baseline.
    """
    cfg = SearchConfig(u_seed=0, u_min=-20, u_max=20)
    candidates = generate_candidates(simple_horizontal_mirror, cfg)

    observer = NullObserver()
    observer.validate()

    # Project a candidate
    cand = candidates[0]
    obs = observer.project_state(cand.state)
    obs_dict = obs.observables

    print(f"\nNullObserver observation keys: {list(obs_dict.keys())}")

    # Should not have angle computations
    assert "theta_incidence_deg" not in obs_dict
    assert "theta_reflection_deg" not in obs_dict
    assert "reflection_law_holds" not in obs_dict


def test_affine_observer_does_not_compute_angles(simple_horizontal_mirror: ReflectionProblem):
    """
    AffineTimeGeometricObserver should NOT compute angles.

    It preserves invariants but doesn't interpret them geometrically.
    """
    cfg = SearchConfig(u_seed=0, u_min=-20, u_max=20)
    candidates = generate_candidates(simple_horizontal_mirror, cfg)

    observer = AffineTimeGeometricObserver(alpha=1.0, beta=0.0)
    observer.validate()

    cand = candidates[0]
    obs = observer.project_state(cand.state)
    obs_dict = obs.observables

    print(f"\nAffineTimeGeometricObserver observation keys: {list(obs_dict.keys())}")

    # Should not have angle computations
    assert "theta_incidence_deg" not in obs_dict
    assert "theta_reflection_deg" not in obs_dict
    assert "reflection_law_holds" not in obs_dict


def test_observer_comparison_on_same_problem(simple_horizontal_mirror: ReflectionProblem):
    """
    Compare all observers on the same problem.

    Expected outcomes:
    - NullObserver: No angles computed
    - AffineTimeGeometric: No angles computed
    - GeometryAngleObserver: Angles computed, law tested
    """
    cfg = SearchConfig(u_seed=0, u_min=-20, u_max=20)
    candidates = generate_candidates(simple_horizontal_mirror, cfg)

    observers = [
        NullObserver(),
        AffineTimeGeometricObserver(alpha=1.0, beta=0.0),
        GeometryAngleObserver(angle_tolerance_deg=0.1),
    ]

    print("\n" + "=" * 70)
    print("OBSERVER COMPARISON")
    print("=" * 70)

    for obs in observers:
        obs.validate()
        best_cand, obs_dict = find_best_reflection_candidate(candidates, obs)

        print(f"\n{obs.name}:")
        print(f"  Best candidate: u={best_cand.u}")

        if "theta_incidence_deg" in obs_dict:
            print(f"  θ_incidence: {obs_dict['theta_incidence_deg']:.2f}°")
            print(f"  θ_reflection: {obs_dict['theta_reflection_deg']:.2f}°")
            print(f"  Δθ: {obs_dict['angle_difference_deg']:.4f}°")
            print(f"  Reflection law: {obs_dict.get('reflection_law_holds', False)}")
        else:
            print(f"  (No angle computation)")

    # Sanity check: only GeometryAngleObserver should have angles
    null_obs = observers[0]
    geom_obs = observers[2]

    null_result = null_obs.project_state(candidates[0].state).observables
    geom_result = geom_obs.project_state(candidates[0].state).observables

    assert "theta_incidence_deg" not in null_result
    assert "theta_incidence_deg" in geom_result


def test_geometry_observer_tolerance_effect(simple_horizontal_mirror: ReflectionProblem):
    """
    Test that angle tolerance affects reflection_law_holds flag.

    Stricter tolerance → fewer candidates satisfy law.
    """
    cfg = SearchConfig(u_seed=0, u_min=-20, u_max=20)
    candidates = generate_candidates(simple_horizontal_mirror, cfg)

    strict_obs = GeometryAngleObserver(angle_tolerance_deg=0.01)
    lenient_obs = GeometryAngleObserver(angle_tolerance_deg=5.0)

    # Count how many candidates satisfy law under each observer
    strict_count = 0
    lenient_count = 0

    for cand in candidates:
        strict_obs_dict = strict_obs.project_state(cand.state).observables
        lenient_obs_dict = lenient_obs.project_state(cand.state).observables

        if strict_obs_dict.get("reflection_law_holds", False):
            strict_count += 1

        if lenient_obs_dict.get("reflection_law_holds", False):
            lenient_count += 1

    print(f"\nStrict (0.01°): {strict_count}/{len(candidates)} satisfy law")
    print(f"Lenient (5.0°): {lenient_count}/{len(candidates)} satisfy law")

    # Lenient should accept more candidates
    assert lenient_count >= strict_count


# ----------------------------
# Run guard
# ----------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
