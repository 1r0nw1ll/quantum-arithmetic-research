"""
Rational Trigonometry — Wildberger's framework for the QA project.

Replaces classical trig with algebraically exact equivalents:
  distance   → quadrance   Q(A,B) = (Δx)² + (Δy)² + ...
  angle θ    → spread      s      = sin²θ = 1 − cos²θ ∈ [0, 1]
  cos θ      → cross       c      = cos²θ = 1 − s

Three fundamental laws (no approximation):
  Pythagoras:      Q₁ + Q₂ = Q₃          (right-angle case, spread=1)
  Spread law:      s₁/Q₁ = s₂/Q₂ = s₃/Q₃
  Triple spread:   (s₁+s₂+s₃)² = 2(s₁²+s₂²+s₃²) + 4s₁s₂s₃
  Cross law:       (Q₁+Q₂−Q₃)² = 4Q₁Q₂(1−s₃)

Enforcement: RT1 linter rule flags math.sin/cos/tan and np.sin/cos/tan in
QA files. Use these functions instead, or add # noqa: RT1 for legitimate
observer projections (signal synthesis, geodetic coordinates, visualization).
"""

from __future__ import annotations
import math
from fractions import Fraction
from typing import Sequence

import numpy as np


# ── Quadrance ─────────────────────────────────────────────────────────────────

def quadrance(A: Sequence, B: Sequence) -> float:
    """Squared distance between points A and B (replaces distance²)."""
    return sum((float(a) - float(b)) ** 2 for a, b in zip(A, B))


def quadrance_exact(A: Sequence, B: Sequence) -> Fraction:
    """Exact quadrance using Fraction arithmetic (integer/rational inputs only)."""
    return sum((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(A, B))


def quadrance_vec(v: np.ndarray) -> np.ndarray:
    """Row-wise quadrance: Q[i] = ||v[i]||² (replaces np.linalg.norm(v)**2)."""
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return float(np.dot(v, v))
    return np.einsum("ij,ij->i", v, v)


# ── Spread ────────────────────────────────────────────────────────────────────

def spread(v1: Sequence, v2: Sequence) -> float:
    """
    Spread between two direction vectors (replaces angle/cos).

    s = 1 − (v1·v2)² / (|v1|²·|v2|²)

    s = 0 → parallel (same direction)
    s = 1 → perpendicular (right angle)
    s = sin²θ in classical terms, where θ is the angle between v1 and v2.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    dot = float(np.dot(v1, v2))
    q1 = float(np.dot(v1, v1))
    q2 = float(np.dot(v2, v2))
    if q1 < 1e-30 or q2 < 1e-30:
        return 0.0
    return 1.0 - (dot * dot) / (q1 * q2)


def spread_exact(v1: Sequence, v2: Sequence) -> Fraction:
    """Exact spread using Fraction arithmetic."""
    f1 = [Fraction(x) for x in v1]
    f2 = [Fraction(x) for x in v2]
    dot = sum(a * b for a, b in zip(f1, f2))
    q1 = sum(a * a for a in f1)
    q2 = sum(b * b for b in f2)
    if q1 == 0 or q2 == 0:
        return Fraction(0)
    return Fraction(1) - dot * dot / (q1 * q2)


def spread_at(A: Sequence, B: Sequence, C: Sequence) -> float:
    """Spread of angle at vertex B in triangle ABC."""
    return spread(
        np.asarray(A, dtype=float) - np.asarray(B, dtype=float),
        np.asarray(C, dtype=float) - np.asarray(B, dtype=float),
    )


def spread_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise spread matrix: S[i,j] = spread(A[i], B[j]).

    A: (m, d)  B: (n, d)  →  S: (m, n)

    Replaces: 1 − cosine_similarity_matrix(A, B)²
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    dots = A @ B.T                                  # (m, n)
    qa = np.einsum("ij,ij->i", A, A)[:, None]      # (m, 1)
    qb = np.einsum("ij,ij->i", B, B)[None, :]      # (1, n)
    denom = qa * qb                                 # (m, n)
    safe = denom > 1e-30
    return np.where(safe, 1.0 - (dots * dots) / np.where(safe, denom, 1.0), 0.0)


# ── Spread from classical angles (bridge for angle-specified data) ────────────

def spread_from_angle_deg(angle_deg: float) -> float:
    """
    Convert a classical angle (degrees) to rational spread.

    s = sin²(θ)

    Use for angle-specified data (bond angles, survey angles) where the angle
    itself is the known quantity. Do NOT use for arbitrary trig computations —
    use spread(v1, v2) from the vectors instead.
    """
    rad = math.radians(angle_deg)
    sin_val = math.sin(rad)
    return sin_val * sin_val


def spread_from_angle_rad(angle_rad: float) -> float:
    """Convert a classical angle (radians) to rational spread. s = sin²(θ)."""
    sin_val = math.sin(angle_rad)
    return sin_val * sin_val


def cross_from_angle_deg(angle_deg: float) -> float:
    """
    Convert a classical angle (degrees) to rational cross (= cos²θ = 1−spread).

    The cross is the rational analog of cosine-squared.
    """
    return 1.0 - spread_from_angle_deg(angle_deg)


def cross_from_angle_rad(angle_rad: float) -> float:
    """Convert a classical angle (radians) to rational cross (= cos²θ = 1−spread)."""
    return 1.0 - spread_from_angle_rad(angle_rad)


def spread_to_angle_deg(s: float) -> float:
    """
    Convert spread to the classical acute angle in degrees (observer output only).

    θ = arcsin(√s)  →  only use for display/verification, not QA computation.
    Add # noqa: RT1 if this call is inside QA logic.
    """
    return math.degrees(math.asin(math.sqrt(max(0.0, min(1.0, s)))))


# ── Rational analog of cosine similarity ─────────────────────────────────────

def cross_similarity(v1: Sequence, v2: Sequence) -> float:
    """
    Rational analog of cosine similarity: (v1·v2)² / (|v1|²·|v2|²) = 1 − spread.

    Returns cos²θ, not cosθ. Use spread() when you need sin²θ.
    For alignment scoring (higher = more aligned), cross_similarity gives
    the same ordering as cosine similarity but stays rational.
    """
    return 1.0 - spread(v1, v2)


def cross_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise cross matrix: C[i,j] = cross_similarity(A[i], B[j]) = 1 − spread_matrix.

    Replaces cosine_similarity_matrix for QA alignment scoring.
    """
    return 1.0 - spread_matrix(A, B)


# ── Fundamental laws (verification / assertion) ───────────────────────────────

def check_pythagoras(Q1: float, Q2: float, Q3: float, tol: float = 1e-9) -> bool:
    """Pythagoras: Q1 + Q2 = Q3 iff the angle opposite Q3 is perpendicular (s=1)."""
    return abs(Q1 + Q2 - Q3) < tol


def check_triple_spread(s1: float, s2: float, s3: float, tol: float = 1e-9) -> bool:
    """
    Triple spread formula (replaces sine rule): (s1+s2+s3)² = 2(s1²+s2²+s3²) + 4s1s2s3.

    The three spreads of a triangle must satisfy this identity.
    """
    lhs = (s1 + s2 + s3) ** 2
    rhs = 2.0 * (s1 * s1 + s2 * s2 + s3 * s3) + 4.0 * s1 * s2 * s3
    return abs(lhs - rhs) < tol


def check_cross_law(Q1: float, Q2: float, Q3: float, s3: float, tol: float = 1e-9) -> bool:
    """
    Cross law (replaces cosine rule): (Q1+Q2−Q3)² = 4·Q1·Q2·(1−s3).

    s3 is the spread at the vertex opposite Q3.
    """
    lhs = (Q1 + Q2 - Q3) ** 2
    rhs = 4.0 * Q1 * Q2 * (1.0 - s3)
    return abs(lhs - rhs) < tol


def check_spread_law(s1: float, Q1: float, s2: float, Q2: float, tol: float = 1e-6) -> bool:
    """
    Spread law (replaces sine rule): s1/Q1 = s2/Q2.

    The ratio spread/quadrance is constant across all vertices of a triangle.
    """
    if Q1 < 1e-30 or Q2 < 1e-30:
        return True
    return abs(s1 / Q1 - s2 / Q2) < tol


# ── Bragg's law (RT form) ─────────────────────────────────────────────────────

def bragg_rational(n: int, Q_lambda: float, Q_d: float) -> float:
    """
    Bragg's law in rational form: n²·Q_λ = 4·Q_d·s

    Classical: nλ = 2d·sin(θ)  →  n²λ² = 4d²·sin²θ  →  n²Q_λ = 4Q_d·s

    Returns the spread s = sin²(θ_Bragg).
    """
    if Q_d < 1e-30:
        return 0.0
    return (n * n * Q_lambda) / (4.0 * Q_d)


# ── Convenience: triangle from three points ───────────────────────────────────

def triangle_rt(A: Sequence, B: Sequence, C: Sequence) -> dict:
    """
    Compute the rational trigonometry of triangle ABC.

    Returns dict with:
      Q1, Q2, Q3  — quadrances of sides BC, AC, AB
      s1, s2, s3  — spreads at vertices A, B, C
      pythagoras  — True if any spread ≈ 1 (right triangle)
    """
    A, B, C = (np.asarray(x, dtype=float) for x in (A, B, C))
    Q1 = float(np.dot(B - C, B - C))   # side BC opposite A
    Q2 = float(np.dot(A - C, A - C))   # side AC opposite B
    Q3 = float(np.dot(A - B, A - B))   # side AB opposite C

    s1 = spread(B - A, C - A)  # spread at A
    s2 = spread(A - B, C - B)  # spread at B
    s3 = spread(A - C, B - C)  # spread at C

    return {
        "Q1": Q1, "Q2": Q2, "Q3": Q3,
        "s1": s1, "s2": s2, "s3": s3,
        "pythagoras": any(abs(s - 1.0) < 1e-9 for s in (s1, s2, s3)),
        "triple_spread_ok": check_triple_spread(s1, s2, s3),
    }


# ── Known exact spreads (rational constants) ──────────────────────────────────

SPREAD_30  = Fraction(1, 4)    # sin²(30°) = 1/4
SPREAD_45  = Fraction(1, 2)    # sin²(45°) = 1/2
SPREAD_60  = Fraction(3, 4)    # sin²(60°) = 3/4
SPREAD_90  = Fraction(1, 1)    # sin²(90°) = 1
SPREAD_120 = Fraction(3, 4)    # sin²(120°) = 3/4   (same as 60°, spread is symmetric)
SPREAD_TET = Fraction(8, 9)    # sin²(109.47°) ≈ 8/9  tetrahedral bond angle
SPREAD_PASSAGE = Fraction(1, 5) # sin²(26.565°) = 1/5  Great Pyramid passage
