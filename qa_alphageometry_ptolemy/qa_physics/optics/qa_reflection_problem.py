from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from qa_physics.optics.qa_reflection_failures import ReflectionFailure, fail, GEOM_DEGENERATE


@dataclass(frozen=True)
class QARational:
    """
    Non-reducing rational to respect QA non-reduction discipline.
    Stores numerator/denominator exactly as given (no gcd simplification).
    """
    n: int
    d: int

    def __post_init__(self):
        if self.d == 0:
            raise ZeroDivisionError("QARational denominator cannot be 0")

    def to_float(self) -> float:
        return self.n / self.d

    def to_tuple(self) -> Tuple[int, int]:
        return (self.n, self.d)

    # Basic ops (non-reducing)
    def __add__(self, other: "QARational") -> "QARational":
        return QARational(self.n * other.d + other.n * self.d, self.d * other.d)

    def __sub__(self, other: "QARational") -> "QARational":
        return QARational(self.n * other.d - other.n * self.d, self.d * other.d)

    def __mul__(self, other: "QARational") -> "QARational":
        return QARational(self.n * other.n, self.d * other.d)

    def __truediv__(self, other: "QARational") -> "QARational":
        if other.n == 0:
            raise ZeroDivisionError("division by zero QARational")
        return QARational(self.n * other.d, self.d * other.n)

    def __neg__(self) -> "QARational":
        return QARational(-self.n, self.d)


@dataclass(frozen=True)
class Point2D:
    x: QARational
    y: QARational

    def to_jsonable(self) -> Dict[str, Any]:
        return {"x": self.x.to_tuple(), "y": self.y.to_tuple()}


@dataclass(frozen=True)
class LineABC:
    """
    Line: A*x + B*y + C = 0 with integer coefficients.
    """
    A: int
    B: int
    C: int

    def is_degenerate(self) -> bool:
        return self.A == 0 and self.B == 0

    def solve_y_given_x(self, x: QARational) -> QARational | ReflectionFailure:
        """
        y = -(A*x + C)/B
        Requires B != 0.
        """
        if self.is_degenerate():
            return fail(GEOM_DEGENERATE, "line has A=B=0", A=self.A, B=self.B, C=self.C)
        if self.B == 0:
            return fail(GEOM_DEGENERATE, "cannot parameterize by x when B=0 (vertical line)", A=self.A, B=self.B, C=self.C)

        # -(A*x + C)/B
        Ax = QARational(self.A, 1) * x
        c = QARational(self.C, 1)
        num = -(Ax + c)  # QARational
        den = QARational(self.B, 1)
        return num / den

    def point_from_u(self, u: int) -> Point2D | ReflectionFailure:
        """
        Deterministic parameterization by x=u (integer).
        """
        x = QARational(u, 1)
        y = self.solve_y_given_x(x)
        if isinstance(y, ReflectionFailure):
            return y
        return Point2D(x=x, y=y)


@dataclass(frozen=True)
class ReflectionProblem:
    """
    Minimal reflection problem definition.

    S: source point
    T: target point
    mirror: line
    u_range: integer x-parameter range for candidate mirror hit-points
    """
    S: Point2D
    T: Point2D
    mirror: LineABC
    u_min: int
    u_max: int

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "S": self.S.to_jsonable(),
            "T": self.T.to_jsonable(),
            "mirror": {"A": self.mirror.A, "B": self.mirror.B, "C": self.mirror.C},
            "u_min": self.u_min,
            "u_max": self.u_max,
        }
