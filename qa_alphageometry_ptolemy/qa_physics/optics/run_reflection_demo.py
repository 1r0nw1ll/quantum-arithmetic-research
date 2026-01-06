from __future__ import annotations

import json
from typing import Any, Dict, List

from qa_physics.optics.qa_reflection_problem import QARational, Point2D, LineABC, ReflectionProblem
from qa_physics.optics.qa_reflection_search import SearchConfig, generate_candidates
from qa_physics.projection.qa_observer import NullObserver, AffineTimeGeometricObserver, QAObserver


def demo_problem() -> ReflectionProblem:
    # Simple, deterministic problem instance
    # Mirror: y = 0  -> 0*x + 1*y + 0 = 0  (B=1 allows x-parameterization)
    mirror = LineABC(A=0, B=1, C=0)

    S = Point2D(QARational(-10, 1), QARational(5, 1))
    T = Point2D(QARational(10, 1), QARational(6, 1))

    return ReflectionProblem(S=S, T=T, mirror=mirror, u_min=-20, u_max=20)


def main() -> None:
    problem = demo_problem()

    cfg = SearchConfig(u_seed=0, u_min=problem.u_min, u_max=problem.u_max, max_steps=200)
    candidates = generate_candidates(problem, cfg)

    observers: List[QAObserver] = [
        NullObserver(),
        AffineTimeGeometricObserver(alpha=1.0, beta=0.0),
        AffineTimeGeometricObserver(alpha=0.5, beta=10.0),
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
            inv = cand.state.to_invariants()

            print(f"\nCandidate u={u}: k={k}, t_obs={t}")
            print("Invariants (trunc):", json.dumps(inv, sort_keys=True)[:240])

        # Measure topology collapse on this domain (quick sanity)
        unique_states = len(set(c.state.state_id() for c in candidates))
        unique_obs = len(set(obs.project_state(c.state).to_json() for c in candidates))
        print(f"\nTopology (domain): {unique_obs}/{unique_states} distinct observations")

    print("\nNext: add a GeometryObserver that computes angles/spreads from invariants,")
    print("then test which projections satisfy angle(incidence)=angle(reflection).")


if __name__ == "__main__":
    main()
