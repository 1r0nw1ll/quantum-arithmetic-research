#!/usr/bin/env python3
"""
FilterCertificate Demo: QA-native state estimation
===================================================

Demonstrates certificate-grade state estimation over dynamical systems,
mapping MIT "Algorithms for Decision Making" Ch. 9-11 to QA reachability.

Key concepts:
- State estimation as belief tracking over lattice
- Kalman filter for linear Gaussian systems
- Particle filter for nonlinear/non-Gaussian systems
- Failure certificates for filter degeneracy/divergence

Reference: Kochenderfer et al. "Algorithms for Decision Making" MIT Press
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qa_alphageometry_ptolemy"))

from qa_certificate import (
    FilterCertificate,
    FilterFailType,
    FilterObstructionEvidence,
    FilterMethod,
    FilterMethodProof,
    DerivationWitness,
    validate_filter_certificate,
)


def demo_kalman_filter():
    """
    Demo 1: Kalman Filter for Linear System

    Classic 1D tracking problem:
    - State: [position, velocity]
    - Observations: noisy position measurements
    - Model: constant velocity with process noise

    This is the gold standard for linear Gaussian systems.
    """
    print("\n" + "="*70)
    print("Demo 1: Kalman Filter - Linear Position/Velocity Tracking")
    print("="*70)

    cert = FilterCertificate.from_kalman(
        model_id="constant_velocity_1d",
        state_names=["position", "velocity"],
        observation_dimension=1,
        n_observations=100,
        state_estimate={
            "position": Fraction(1050, 10),  # 105.0
            "velocity": Fraction(21, 10),     # 2.1
        },
        covariance_trace=Fraction(15, 10),  # 1.5
    )

    print(f"\nModel: Constant velocity (1D)")
    print(f"State: [position, velocity]")
    print(f"Observations: {cert.n_observations} noisy position measurements")
    print(f"\nEstimate:")
    print(f"  position = {cert.state_estimate['position']} m")
    print(f"  velocity = {cert.state_estimate['velocity']} m/s")
    print(f"  tr(P) = {cert.covariance_trace} (uncertainty)")
    print(f"\nMethod: {cert.method_proof.method.value}")
    print(f"Certificate valid: {cert.is_valid()}")

    result = validate_filter_certificate(cert)
    print(f"Validation passed: {result.valid}")

    return cert.to_json()


def demo_particle_filter():
    """
    Demo 2: Particle Filter for Nonlinear System

    Robot localization problem:
    - State: [x, y, theta] (pose)
    - Observations: range/bearing to landmarks
    - Model: nonlinear motion + sensor model

    Particle filter handles nonlinearity via sampling.
    """
    print("\n" + "="*70)
    print("Demo 2: Particle Filter - Robot Localization")
    print("="*70)

    cert = FilterCertificate.from_particle_filter(
        model_id="robot_localization",
        state_names=["x", "y", "theta"],
        observation_dimension=2,  # range, bearing
        n_observations=50,
        n_particles=1000,
        state_estimate={
            "x": Fraction(245, 10),      # 24.5
            "y": Fraction(183, 10),      # 18.3
            "theta": Fraction(157, 100), # 1.57 (~pi/2)
        },
        credible_interval_width=Fraction(3, 2),  # 95% CI width
        effective_sample_size=Fraction(750, 1),  # 750 out of 1000
        n_resamples=8,
    )

    print(f"\nModel: Robot pose (x, y, theta)")
    print(f"Observations: {cert.n_observations} range/bearing measurements")
    print(f"Particles: {cert.method_proof.n_particles}")
    print(f"\nEstimate:")
    print(f"  x = {cert.state_estimate['x']} m")
    print(f"  y = {cert.state_estimate['y']} m")
    print(f"  theta = {cert.state_estimate['theta']} rad")
    print(f"  95% CI width = {cert.credible_interval_width}")
    print(f"\nFilter health:")
    print(f"  ESS = {cert.method_proof.effective_sample_size}/{cert.method_proof.n_particles}")
    print(f"  Resamples = {cert.method_proof.n_resamples}")
    print(f"\nMethod: {cert.method_proof.method.value}")
    print(f"Certificate valid: {cert.is_valid()}")

    result = validate_filter_certificate(cert)
    print(f"Validation passed: {result.valid}")

    return cert.to_json()


def demo_particle_degeneracy():
    """
    Demo 3: Particle Degeneracy Failure

    When effective sample size drops too low, the particle
    filter loses diversity and fails to track the true state.

    This demonstrates a certificate-grade failure witness.
    """
    print("\n" + "="*70)
    print("Demo 3: Particle Degeneracy - Filter Failure Certificate")
    print("="*70)

    cert = FilterCertificate(
        model_id="high_noise_tracking",
        model_description="Tracking with severe observation noise",
        state_dimension=3,
        state_names=["x", "y", "z"],
        observation_dimension=1,
        linear_system=False,
        gaussian_noise=False,
        n_observations=30,
        filter_success=False,
        failure_mode=FilterFailType.PARTICLE_DEGENERACY,
        obstruction_if_fail=FilterObstructionEvidence(
            fail_type=FilterFailType.PARTICLE_DEGENERACY,
            effective_sample_size=Fraction(8, 1),   # Only 8!
            n_particles=500,
            ess_threshold=Fraction(50, 1),  # Threshold is 50
            max_weight=Fraction(93, 100),   # One particle has 93% weight
        ),
        method_proof=FilterMethodProof(
            method=FilterMethod.PARTICLE,
            n_particles=500,
            n_timesteps=30,
            resampling_method="multinomial",
            effective_sample_size=Fraction(8, 1),
            n_resamples=25,  # Many resamples but still collapsed
        ),
        strict_mode=True,
    )

    print(f"\nModel: 3D tracking with severe noise")
    print(f"Particles: {cert.method_proof.n_particles}")
    print(f"Timesteps: {cert.method_proof.n_timesteps}")
    print(f"\nFAILURE: {cert.failure_mode.value}")
    obs = cert.obstruction_if_fail
    print(f"Evidence:")
    print(f"  ESS = {obs.effective_sample_size}/{obs.n_particles}")
    print(f"  ESS threshold = {obs.ess_threshold}")
    print(f"  Max particle weight = {obs.max_weight}")
    print(f"  Resamples attempted = {cert.method_proof.n_resamples}")

    result = validate_filter_certificate(cert)
    print(f"\nValidation passed: {result.valid}")

    return cert.to_json()


def demo_state_unobservable():
    """
    Demo 4: Unobservable State Certificate

    Some states cannot be estimated from the available observations.
    This is a structural failure (observability rank deficient).
    """
    print("\n" + "="*70)
    print("Demo 4: Unobservable State - Structural Failure Certificate")
    print("="*70)

    cert = FilterCertificate(
        model_id="partial_observability",
        model_description="4D state with only position measurements",
        state_dimension=4,
        state_names=["x", "vx", "y", "vy"],  # Position + velocity in 2D
        observation_dimension=2,  # Only x, y (no velocities)
        linear_system=True,
        gaussian_noise=True,
        n_observations=50,
        filter_success=False,
        failure_mode=FilterFailType.STATE_UNOBSERVABLE,
        obstruction_if_fail=FilterObstructionEvidence(
            fail_type=FilterFailType.STATE_UNOBSERVABLE,
            observability_rank=2,  # Only rank 2
            state_dimension=4,     # Need rank 4
            unobservable_modes=["vx", "vy"],  # Velocities unobservable
        ),
        strict_mode=True,
    )

    print(f"\nModel: 2D position + velocity")
    print(f"State: [x, vx, y, vy]")
    print(f"Observations: [x, y] only (no velocity)")
    print(f"\nFAILURE: {cert.failure_mode.value}")
    obs = cert.obstruction_if_fail
    print(f"Evidence:")
    print(f"  State dimension = {obs.state_dimension}")
    print(f"  Observability rank = {obs.observability_rank}")
    print(f"  Unobservable modes = {obs.unobservable_modes}")
    print(f"  Reason: Can't estimate velocity from position-only sensors")

    result = validate_filter_certificate(cert)
    print(f"\nValidation passed: {result.valid}")

    return cert.to_json()


def demo_filter_diverged():
    """
    Demo 5: Filter Divergence Failure

    When the filter estimate drifts far from the true state
    (due to model mismatch, numerical issues, etc.).
    """
    print("\n" + "="*70)
    print("Demo 5: Filter Divergence - Estimation Drift Certificate")
    print("="*70)

    cert = FilterCertificate(
        model_id="misspecified_dynamics",
        model_description="Filter with incorrect process model",
        state_dimension=2,
        state_names=["x", "v"],
        observation_dimension=1,
        linear_system=True,
        gaussian_noise=True,
        n_observations=100,
        filter_success=False,
        failure_mode=FilterFailType.FILTER_DIVERGED,
        obstruction_if_fail=FilterObstructionEvidence(
            fail_type=FilterFailType.FILTER_DIVERGED,
            estimation_error=Fraction(150, 1),   # 150 units error
            error_threshold=Fraction(10, 1),     # Threshold is 10
            timestep=85,  # Diverged at timestep 85
        ),
        method_proof=FilterMethodProof(
            method=FilterMethod.KALMAN,
            n_timesteps=85,  # Stopped at divergence
        ),
        strict_mode=True,
    )

    print(f"\nModel: Position + velocity with misspecified dynamics")
    print(f"Timesteps: {cert.method_proof.n_timesteps}")
    print(f"\nFAILURE: {cert.failure_mode.value}")
    obs = cert.obstruction_if_fail
    print(f"Evidence:")
    print(f"  Estimation error = {obs.estimation_error}")
    print(f"  Error threshold = {obs.error_threshold}")
    print(f"  Divergence timestep = {obs.timestep}")
    print(f"  Reason: Model mismatch caused estimate to drift")

    result = validate_filter_certificate(cert)
    print(f"\nValidation passed: {result.valid}")

    return cert.to_json()


def demo_histogram_filter():
    """
    Demo 6: Histogram Filter for Discrete State Space

    Grid-based localization where state space is discretized.
    Exact for discrete domains, no Gaussian assumption.
    """
    print("\n" + "="*70)
    print("Demo 6: Histogram Filter - Discrete Grid Localization")
    print("="*70)

    cert = FilterCertificate(
        model_id="grid_localization",
        model_description="Robot on 10x10 discrete grid",
        state_dimension=1,  # Cell index
        state_names=["cell"],
        observation_dimension=4,  # 4 wall sensors
        linear_system=False,
        gaussian_noise=False,
        n_observations=20,
        filter_success=True,
        state_estimate={"cell": Fraction(47, 1)},  # Cell 47
        entropy=Fraction(2, 1),  # Relatively certain
        method_proof=FilterMethodProof(
            method=FilterMethod.HISTOGRAM,
            n_bins=100,  # 10x10 grid
            n_timesteps=20,
            verifiable=True,
        ),
        estimation_witness=DerivationWitness(
            invariant_name="grid_belief",
            derivation_operator="histogram_filter",
            input_data={
                "grid_size": 100,
                "n_observations": 20,
            },
            output_value=1,
            verifiable=True,
        ),
        strict_mode=True,
    )

    print(f"\nModel: 10x10 discrete grid")
    print(f"Observations: {cert.n_observations} wall sensor readings")
    print(f"\nEstimate:")
    print(f"  Most likely cell = {cert.state_estimate['cell']}")
    print(f"  Entropy = {cert.entropy} (lower = more certain)")
    print(f"\nMethod: {cert.method_proof.method.value}")
    print(f"Bins: {cert.method_proof.n_bins}")
    print(f"Certificate valid: {cert.is_valid()}")

    result = validate_filter_certificate(cert)
    print(f"Validation passed: {result.valid}")

    return cert.to_json()


def main():
    """Run all demos and export certificates."""
    print("="*70)
    print("FILTER CERTIFICATE DEMO")
    print("QA-native state estimation (Ch. 9-11)")
    print("="*70)

    # Run all demos
    demos = {
        "kalman_tracking": demo_kalman_filter(),
        "particle_localization": demo_particle_filter(),
        "particle_degeneracy_failure": demo_particle_degeneracy(),
        "state_unobservable_failure": demo_state_unobservable(),
        "filter_diverged_failure": demo_filter_diverged(),
        "histogram_grid": demo_histogram_filter(),
    }

    # Combine into single certificate
    combined = {
        "demo": "filter_demo",
        "description": "QA-native state estimation certificates",
        "reference": "MIT Algorithms for Decision Making, Chapters 9-11",
        "scenarios": demos,
        "key_insights": [
            "State estimation = tracking belief distribution over state space.",
            "Kalman filter = optimal for linear Gaussian (exact inference).",
            "Particle filter = handles nonlinearity via sampling.",
            "PARTICLE_DEGENERACY = ESS too low (sample impoverishment).",
            "STATE_UNOBSERVABLE = structural failure (observability rank deficient).",
            "FILTER_DIVERGED = estimate drifts from truth (model mismatch).",
            "Histogram filter = exact for discrete state spaces.",
            "All failures are first-class objects with machine-checkable witnesses.",
        ],
    }

    # Export
    output_path = Path(__file__).parent / "filter_cert.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "="*70)
    print(f"Exported combined certificate to: {output_path}")
    print("="*70)

    return combined


if __name__ == "__main__":
    main()
