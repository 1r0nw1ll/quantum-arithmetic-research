"""
Projection validation tests for QA Physics Interface.

Tests follow QA-RML discipline:
- Deterministic (same input → same output)
- Explicit failure modes (topology collapse is measured, not assumed)
- No floating-point equality (use stable JSON comparison)
- Observer-independent validation (tests work for any QAObserver)
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List, Protocol, Set
from dataclasses import dataclass
import json

# FIXED: Use proper package imports instead of sys.path hack
from qa_physics.projection.qa_observer import (
    QAObserver,
    NullObserver,
    AffineTimeGeometricObserver,
    Observation,
    ProjectionReport,
)


# ----------------------------
# Minimal QAState protocol
# ----------------------------

class QAStateProtocol(Protocol):
    """
    Minimal interface for QA states in projection tests.
    Your real QA states should satisfy this.
    """
    def state_id(self) -> str:
        """Unique, stable identifier for this state."""
        ...

    def to_invariants(self) -> Dict[str, Any]:
        """Dictionary of QA invariants (exact arithmetic)."""
        ...


@dataclass(frozen=True)
class MockQAState:
    """
    Mock QA state for testing projections.
    Uses frozen dataclass for determinism.
    """
    b: int
    e: int
    d: int
    a: int
    phi9: int
    phi24: int

    def state_id(self) -> str:
        return f"({self.b},{self.e},{self.d},{self.a})"

    def to_invariants(self) -> Dict[str, Any]:
        return {
            "b": self.b,
            "e": self.e,
            "d": self.d,
            "a": self.a,
            "phi9": self.phi9,
            "phi24": self.phi24,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.to_invariants()


# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture
def sample_states() -> List[MockQAState]:
    """Sample QA states with known structure."""
    return [
        MockQAState(3, 4, 5, 6, 0, 1),
        MockQAState(5, 12, 13, 30, 1, 0),
        MockQAState(8, 15, 17, 60, -1, 1),
        MockQAState(3, 4, 5, 6, 0, 1),  # Duplicate of first (tests collapse)
    ]


@pytest.fixture
def sample_path(sample_states: List[MockQAState]) -> List[MockQAState]:
    """Sample path for path projection tests."""
    return sample_states[:3]


@pytest.fixture
def observers() -> List[QAObserver]:
    """All observers under test."""
    return [
        NullObserver(),
        AffineTimeGeometricObserver(alpha=1.0, beta=0.0),
        AffineTimeGeometricObserver(alpha=0.5, beta=10.0),
    ]


# ----------------------------
# Core determinism tests
# ----------------------------

def test_observer_validation(observers: List[QAObserver]):
    """All observers must pass self-validation."""
    for obs in observers:
        obs.validate()  # Should not raise


def test_observer_reports(observers: List[QAObserver]):
    """All observers must produce valid reports."""
    for obs in observers:
        report = obs.report()
        assert isinstance(report, ProjectionReport)
        assert report.observer_name
        assert report.version
        assert report.time_model
        assert report.unit_system

        # JSON-serializable
        json_str = report.to_json()
        assert json_str
        roundtrip = json.loads(json_str)
        assert roundtrip["observer_name"] == report.observer_name


def test_projection_determinism_state(
    observers: List[QAObserver],
    sample_states: List[MockQAState]
):
    """
    CRITICAL: Same state → same observation (bitwise JSON equality).
    This is the determinism contract.
    """
    for obs in observers:
        for state in sample_states:
            obs1 = obs.project_state(state)
            obs2 = obs.project_state(state)

            # JSON equality (stable serialization)
            assert obs1.to_json() == obs2.to_json(), (
                f"{obs.name} produced different outputs for same state"
            )


def test_projection_determinism_path(
    observers: List[QAObserver],
    sample_path: List[MockQAState]
):
    """Same path → same observation."""
    for obs in observers:
        obs1 = obs.project_path(sample_path)
        obs2 = obs.project_path(sample_path)

        assert obs1.to_json() == obs2.to_json(), (
            f"{obs.name} path projection not deterministic"
        )


# ----------------------------
# Topology collapse tests
# ----------------------------

def test_topology_collapse_measurement(
    observers: List[QAObserver],
    sample_states: List[MockQAState]
):
    """
    Measure topology destruction: distinct states → distinct observations?

    This quantifies what the observer preserves vs destroys.
    """
    for obs in observers:
        # Get unique state IDs
        unique_states = len(set(s.state_id() for s in sample_states))

        # Get unique observations
        observations = [obs.project_state(s).to_json() for s in sample_states]
        unique_observations = len(set(observations))

        collapse_ratio = unique_observations / unique_states

        # Log the result (not an assertion - this is measurement)
        print(f"\n{obs.name}: {unique_observations}/{unique_states} distinct "
              f"(collapse ratio: {collapse_ratio:.2f})")

        # If observer claims preserves_topology, should have high ratio
        if obs.preserves_topology():
            assert collapse_ratio >= 0.9, (
                f"{obs.name} claims topology preservation but "
                f"collapse ratio is {collapse_ratio:.2f}"
            )


def test_duplicate_state_handling(
    observers: List[QAObserver],
    sample_states: List[MockQAState]
):
    """
    Duplicate states (same state_id) MUST produce identical observations.
    """
    # sample_states has a duplicate by construction
    state0 = sample_states[0]
    state3 = sample_states[3]
    assert state0.state_id() == state3.state_id()

    for obs in observers:
        obs0 = obs.project_state(state0)
        obs3 = obs.project_state(state3)

        assert obs0.to_json() == obs3.to_json(), (
            f"{obs.name} produced different observations for duplicate states"
        )


# ----------------------------
# Time projection tests
# ----------------------------

def test_time_projection_monotonicity(observers: List[QAObserver]):
    """
    If observer time model claims to be monotonic, verify it.
    """
    for obs in observers:
        time_model = obs.time_model()

        # Test monotonicity for a range of k values
        k_values = list(range(0, 20))
        t_values = [obs.project_time(k) for k in k_values]

        # Check if monotonic increasing
        is_monotonic = all(t_values[i] <= t_values[i+1]
                          for i in range(len(t_values)-1))

        print(f"\n{obs.name} time monotonicity: {is_monotonic}")
        print(f"  k={k_values[:5]} -> t={t_values[:5]}")

        # FIXED: Detect affine with positive coefficient
        if "affine" in time_model.lower():
            # Extract alpha if possible to check sign
            # For now, just enforce monotonicity for affine models
            assert is_monotonic, (
                f"{obs.name} claims affine time but is not monotonic"
            )


def test_qa_duration_consistency(observers: List[QAObserver]):
    """
    qa_duration should equal len(path)-1 for all observers.
    This is Axiom T1.
    """
    paths = [
        [MockQAState(3, 4, 5, 6, 0, 1)],
        [MockQAState(3, 4, 5, 6, 0, 1), MockQAState(5, 12, 13, 30, 1, 0)],
        [MockQAState(3, 4, 5, 6, 0, 1)] * 10,  # Path of length 9
    ]

    for obs in observers:
        for path in paths:
            k = obs.qa_duration(path)
            expected_k = len(path) - 1

            assert k == expected_k, (
                f"{obs.name}.qa_duration({len(path)} nodes) = {k}, "
                f"expected {expected_k}"
            )


def test_time_projection_zero_length_path(observers: List[QAObserver]):
    """Edge case: empty or single-node paths."""
    # FIXED: Actually test empty path, not just single-node
    empty: List[MockQAState] = []
    single_node = [MockQAState(3, 4, 5, 6, 0, 1)]

    for obs in observers:
        # Empty path
        k_empty = obs.qa_duration(empty)
        assert k_empty == 0, f"{obs.name} empty path duration should be 0"

        # Single node = 0 edges
        k = obs.qa_duration(single_node)
        assert k == 0

        t = obs.project_time(0)
        # Just check it's finite and deterministic
        assert isinstance(t, (int, float))
        assert t == obs.project_time(0)


# ----------------------------
# Symmetry preservation tests
# ----------------------------

def test_scale_symmetry_claim(observers: List[QAObserver]):
    """
    If observer claims preserves_symmetry, test λ-scaling invariance.

    Note: This is a placeholder - real test needs λ operator.
    For now, just verify the claim is explicit.
    """
    for obs in observers:
        symmetry_claim = obs.preserves_symmetry()
        assert isinstance(symmetry_claim, bool), (
            f"{obs.name}.preserves_symmetry() must return bool"
        )

        print(f"\n{obs.name} claims preserves_symmetry: {symmetry_claim}")


# ----------------------------
# Failure semantics tests
# ----------------------------

def test_failure_semantics_claim(observers: List[QAObserver]):
    """
    Verify observers explicitly state failure semantics preservation.
    """
    for obs in observers:
        failure_claim = obs.preserves_failure_semantics()
        assert isinstance(failure_claim, bool)

        print(f"\n{obs.name} claims preserves_failure_semantics: {failure_claim}")


# ----------------------------
# Observer comparison tests
# ----------------------------

def test_multiple_observers_on_same_state(
    observers: List[QAObserver],
    sample_states: List[MockQAState]
):
    """
    Compare different observers on same states.
    This is the "projection robustness" test.

    NOTE: This asserts diversity for v0 exploration.
    Later, when multiple observers may be intentionally equivalent
    on some subdomain, convert to measurement or scope appropriately.
    """
    state = sample_states[0]

    observations = []
    for obs in observers:
        obs_result = obs.project_state(state)
        observations.append((obs.name, obs_result))

    print(f"\n--- Projections of state {state.state_id()} ---")
    for name, obs_result in observations:
        print(f"{name}:")
        print(f"  {obs_result.to_json()[:200]}")

    # Different observers SHOULD produce different observations
    # (if they all produce the same, they're not exploring the space)
    unique_json = len(set(obs.to_json() for _, obs in observations))
    assert unique_json >= 2, (
        "All observers produce identical outputs - insufficient diversity"
    )


def test_observer_distance_contract(
    observers: List[QAObserver],
    sample_states: List[MockQAState]
):
    """
    Test obs_distance optional helper (if implemented).
    """
    state1 = sample_states[0]
    state2 = sample_states[1]

    for obs in observers:
        obs1 = obs.project_state(state1)
        obs2 = obs.project_state(state2)

        # If implemented, should return non-negative float
        d = obs.obs_distance(obs1, obs2)
        assert isinstance(d, (int, float))
        assert d >= 0

        # Self-distance should be zero (if non-trivial implementation)
        d_self = obs.obs_distance(obs1, obs1)
        assert d_self == 0


# ----------------------------
# JSON serialization tests
# ----------------------------

def test_observation_json_roundtrip(
    observers: List[QAObserver],
    sample_states: List[MockQAState]
):
    """Observations must survive JSON roundtrip."""
    for obs in observers:
        for state in sample_states:
            obs_result = obs.project_state(state)

            # Serialize
            json_str = obs_result.to_json()

            # Deserialize
            data = json.loads(json_str)

            # Check structure
            assert "observables" in data
            assert "units" in data
            assert "metadata" in data


# ----------------------------
# Run guard
# ----------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
