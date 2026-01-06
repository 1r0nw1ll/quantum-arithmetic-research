from __future__ import annotations

import json
from typing import Any, Dict, List

from qa_physics.optics.qa_reflection_problem import QARational, Point2D, LineABC, ReflectionProblem
from qa_physics.optics.qa_reflection_search import SearchConfig, generate_candidates
from qa_physics.optics.qa_geometry_observer import GeometryAngleObserver
from qa_physics.projection.qa_observer import NullObserver, AffineTimeGeometricObserver, QAObserver


def demo_problem() -> ReflectionProblem:
    # Simple, deterministic problem instance
    # Mirror: y = 0  -> 0*x + 1*y + 0 = 0  (B=1 allows x-parameterization)
    # Symmetric case: S and T at same height → perfect reflection at u=0
    mirror = LineABC(A=0, B=1, C=0)

    S = Point2D(QARational(-10, 1), QARational(5, 1))
    T = Point2D(QARational(10, 1), QARational(5, 1))  # Same height as S

    return ReflectionProblem(S=S, T=T, mirror=mirror, u_min=-20, u_max=20)


def main() -> None:
    problem = demo_problem()

    cfg = SearchConfig(u_seed=0, u_min=problem.u_min, u_max=problem.u_max, max_steps=200)
    candidates = generate_candidates(problem, cfg)

    observers: List[QAObserver] = [
        NullObserver(),
        AffineTimeGeometricObserver(alpha=1.0, beta=0.0),
        GeometryAngleObserver(angle_tolerance_deg=0.1),
    ]

    print("\n=== QA Reflection Demo (Projection Probe) ===")
    print("Problem:", json.dumps(problem.to_jsonable(), sort_keys=True))

    # For v0, just show:
    # - QA time k (edges) and observer time t_obs
    # - invariant packets for a few candidates
    sample_us = [-10, 0, 10]

    for obs in observers:
        obs.validate()
        print(f"\n--- Observer: {obs.name} | {obs.time_model()} | units={obs.unit_system()} ---")
        for u in sample_us:
            cand = next(c for c in candidates if c.u == u)
            k = obs.qa_duration(cand.path_states)
            t = obs.project_time(k)
            observation = obs.project_state(cand.state)
            obs_dict = observation.observables

            print(f"\nCandidate u={u}: k={k}, t_obs={t:.2f}")

            # Show angle info if available
            if "theta_incidence_deg" in obs_dict:
                print(f"  θ_incidence:  {obs_dict['theta_incidence_deg']:.2f}°")
                print(f"  θ_reflection: {obs_dict['theta_reflection_deg']:.2f}°")
                print(f"  Δθ: {obs_dict['angle_difference_deg']:.4f}°")
                print(f"  Reflection law holds: {obs_dict.get('reflection_law_holds', False)}")
            else:
                inv = cand.state.to_invariants()
                print(f"  Invariants (trunc): {json.dumps(inv, sort_keys=True)[:180]}...")

        # Measure topology collapse on this domain (quick sanity)
        unique_states = len(set(c.state.state_id() for c in candidates))
        unique_obs = len(set(obs.project_state(c.state).to_json() for c in candidates))
        print(f"\nTopology (domain): {unique_obs}/{unique_states} distinct observations")

    print("\n" + "=" * 70)
    print("KEY RESULT: Only GeometryAngleObserver computes angles.")
    print("The law of reflection is a PROJECTION property, not a QA property.")
    print("=" * 70)


if __name__ == "__main__":
    main()
