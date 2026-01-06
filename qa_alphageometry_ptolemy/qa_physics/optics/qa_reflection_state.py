from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

from qa_physics.optics.qa_reflection_problem import Point2D, QARational, ReflectionProblem
from qa_physics.optics.qa_reflection_failures import ReflectionFailure, is_failure, fail, OOB


def _q(x: int) -> QARational:
    return QARational(x, 1)


def quadrance(p: Point2D, q: Point2D) -> QARational:
    """
    Exact squared distance: (dx)^2 + (dy)^2 using non-reducing rationals.
    """
    dx = p.x - q.x
    dy = p.y - q.y
    return (dx * dx) + (dy * dy)


@dataclass(frozen=True)
class ReflectionState:
    """
    A QA state = a candidate mirror-hit point parameterized by integer u.

    This is intentionally *minimal*:
    - state_id: stable identifier
    - to_invariants: observer-facing invariants in exact form
    """
    problem: ReflectionProblem
    u: int

    def state_id(self) -> str:
        return f"ReflectionState(u={self.u})"

    def to_invariants(self) -> Dict[str, Any]:
        """
        Exact invariants exposed to observers.

        Includes:
        - points S, M(u), T (as (n,d) tuples)
        - quadrances Q_SM, Q_MT, Q_ST
        - mirror line coefficients
        - u bounds
        """
        M = self.problem.mirror.point_from_u(self.u)
        if isinstance(M, ReflectionFailure):
            # encode failure as invariant packet (observer can interpret)
            return {
                "fail_type": M.fail_type,
                "fail_detail": M.detail,
                "fail_meta": M.meta,
                "u": self.u,
            }

        Q_SM = quadrance(self.problem.S, M)
        Q_MT = quadrance(M, self.problem.T)
        Q_ST = quadrance(self.problem.S, self.problem.T)

        return {
            "u": self.u,
            "S": self.problem.S.to_jsonable(),
            "M": M.to_jsonable(),
            "T": self.problem.T.to_jsonable(),
            "mirror": {"A": self.problem.mirror.A, "B": self.problem.mirror.B, "C": self.problem.mirror.C},
            "u_min": self.problem.u_min,
            "u_max": self.problem.u_max,
            "Q_SM": Q_SM.to_tuple(),
            "Q_MT": Q_MT.to_tuple(),
            "Q_ST": Q_ST.to_tuple(),
        }

    def is_in_bounds(self) -> bool:
        return self.problem.u_min <= self.u <= self.problem.u_max

    def require_in_bounds(self) -> ReflectionFailure | None:
        if not self.is_in_bounds():
            return fail(OOB, "u out of bounds", u=self.u, u_min=self.problem.u_min, u_max=self.problem.u_max)
        return None
