"""
qa_competency_metrics.py

Deterministic competency metrics for QA_COMPETENCY_DETECTION_FRAMEWORK.v1.

Levin-aligned metrics for substrate-independent competency detection:
  - agency_index:      |reachable| / |total|
  - plasticity_index:  delta_reachability / delta_perturbation
  - goal_density:      |attractors| / |total|
  - control_entropy:   -sum p(move) ln p(move)   (natural log)

Standalone: stdlib only, no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else (num / den)


@dataclass(frozen=True)
class CompetencyMetrics:
    agency_index: float
    plasticity_index: float
    goal_density: float
    control_entropy: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "agency_index": self.agency_index,
            "plasticity_index": self.plasticity_index,
            "goal_density": self.goal_density,
            "control_entropy": self.control_entropy,
        }


def agency_index(reachable_states: int, total_states: int) -> float:
    if reachable_states < 0 or total_states < 0:
        raise ValueError("reachable_states/total_states must be non-negative")
    if total_states != 0 and reachable_states > total_states:
        reachable_states = total_states
    return _safe_div(float(reachable_states), float(total_states))


def plasticity_index(delta_reachability: float, delta_perturbation: float) -> float:
    return _safe_div(float(delta_reachability), float(delta_perturbation))


def goal_density(attractor_basins: int, total_states: int) -> float:
    if attractor_basins < 0 or total_states < 0:
        raise ValueError("attractor_basins/total_states must be non-negative")
    if total_states != 0 and attractor_basins > total_states:
        attractor_basins = total_states
    return _safe_div(float(attractor_basins), float(total_states))


def control_entropy(move_probabilities: Dict[str, float]) -> float:
    """Natural-log entropy. Renormalizes if probs don't sum to 1."""
    if not move_probabilities:
        return 0.0
    items = [(k, float(v)) for k, v in move_probabilities.items() if float(v) > 0.0]
    if not items:
        return 0.0
    s = sum(v for _, v in items)
    if s <= 0.0:
        return 0.0
    ent = 0.0
    for _, v in items:
        p = v / s
        ent -= p * math.log(p)
    if ent < 0.0 and ent > -1e-12:
        ent = 0.0
    return ent


def compute_competency_metrics(
    *,
    reachable_states: int,
    total_states: int,
    attractor_basins: int,
    move_probabilities: Optional[Dict[str, float]] = None,
    delta_reachability: float = 0.0,
    delta_perturbation: float = 0.0,
) -> CompetencyMetrics:
    mp = move_probabilities or {}
    return CompetencyMetrics(
        agency_index=agency_index(reachable_states, total_states),
        plasticity_index=plasticity_index(delta_reachability, delta_perturbation),
        goal_density=goal_density(attractor_basins, total_states),
        control_entropy=control_entropy(mp),
    )
