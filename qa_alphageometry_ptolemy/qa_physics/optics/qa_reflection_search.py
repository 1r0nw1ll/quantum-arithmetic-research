from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qa_physics.optics.qa_reflection_state import ReflectionState
from qa_physics.optics.qa_reflection_failures import ReflectionFailure, is_failure, fail, OOB
from qa_physics.projection.qa_observer import TransitionLogEvent


@dataclass(frozen=True)
class SearchConfig:
    """
    Bounded, deterministic search over u-parameterized states.

    This is a QA-time harness:
    - time = path length (edges)
    - moves: u -> u+1 (sigma_plus), u -> u-1 (sigma_minus)
    """
    u_seed: int = 0
    u_min: int = -50
    u_max: int = 50
    max_steps: int = 200  # hard cap to keep runs bounded


def neighbors(u: int) -> List[Tuple[str, int]]:
    return [("sigma_plus", u + 1), ("sigma_minus", u - 1)]


def build_path(seed: int, target: int) -> List[int]:
    """
    Deterministic shortest path in 1D (by construction).
    This is intentionally simple: QA-time scaffolding, not 'optics'.
    """
    if target == seed:
        return [seed]
    step = 1 if target > seed else -1
    path = list(range(seed, target + step, step))
    return path


def enumerate_states(problem, cfg: SearchConfig) -> List[ReflectionState]:
    """
    Deterministically enumerate ReflectionState over configured bounds.
    """
    states: List[ReflectionState] = []
    for u in range(cfg.u_min, cfg.u_max + 1):
        states.append(ReflectionState(problem=problem, u=u))
    return states


def logs_for_path(problem, cfg: SearchConfig, u_target: int) -> List[TransitionLogEvent]:
    """
    Produce deterministic transition logs for the QA path seed->u_target.
    """
    u_path = build_path(cfg.u_seed, u_target)
    events: List[TransitionLogEvent] = []

    for i in range(len(u_path) - 1):
        u0 = u_path[i]
        u1 = u_path[i + 1]
        move = "sigma_plus" if u1 == u0 + 1 else "sigma_minus"

        s0 = ReflectionState(problem=problem, u=u0)
        s1 = ReflectionState(problem=problem, u=u1)

        fail0 = s0.require_in_bounds()
        fail1 = s1.require_in_bounds()
        is_legal = (fail0 is None) and (fail1 is None)

        fail_type = None
        meta: Dict[str, Any] = {"u0": u0, "u1": u1}
        if not is_legal:
            f = fail0 or fail1
            fail_type = f.fail_type if f else OOB

        events.append(
            TransitionLogEvent(
                step=i,
                move=move,
                src_state_id=s0.state_id(),
                dst_state_id=s1.state_id(),
                is_legal=is_legal,
                fail_type=fail_type,
                invariant_diff=None,  # filled later when you introduce invariant web diffs
                obs_diff=None,        # filled later for observer-aware diffs
                meta=meta,
            )
        )
    return events


@dataclass(frozen=True)
class ReflectionCandidate:
    u: int
    state: ReflectionState
    path_u: List[int]
    path_states: List[ReflectionState]
    logs: List[TransitionLogEvent]


def generate_candidates(problem, cfg: SearchConfig) -> List[ReflectionCandidate]:
    """
    Generate candidates u with their QA path (seed->u) and logs.

    This is the core "QA-time harness" needed for the projection study.
    """
    cands: List[ReflectionCandidate] = []
    for u in range(cfg.u_min, cfg.u_max + 1):
        u_path = build_path(cfg.u_seed, u)
        path_states = [ReflectionState(problem=problem, u=uu) for uu in u_path]
        logs = logs_for_path(problem, cfg, u)

        cands.append(
            ReflectionCandidate(
                u=u,
                state=ReflectionState(problem=problem, u=u),
                path_u=u_path,
                path_states=path_states,
                logs=logs,
            )
        )
    return cands
