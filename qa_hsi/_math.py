"""Self-contained QA and Koenig arithmetic. No external dependencies."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# QA packet — raw (non-modular) derived coordinates + Pythagorean invariants
# ---------------------------------------------------------------------------

_FAMILY_GRID: Dict[Tuple[int, int], str] = {
    (1,1):"Fibonacci",(1,2):"Fibonacci",(1,3):"Lucas",(1,4):"Phibonacci",(1,5):"Fibonacci",(1,6):"Phibonacci",(1,7):"Lucas",(1,8):"Fibonacci",(1,9):"Fibonacci",
    (2,1):"Lucas",(2,2):"Lucas",(2,3):"Fibonacci",(2,4):"Lucas",(2,5):"Phibonacci",(2,6):"Phibonacci",(2,7):"Lucas",(2,8):"Fibonacci",(2,9):"Lucas",
    (3,1):"Phibonacci",(3,2):"Phibonacci",(3,3):"Tribonacci",(3,4):"Lucas",(3,5):"Fibonacci",(3,6):"Tribonacci",(3,7):"Fibonacci",(3,8):"Lucas",(3,9):"Tribonacci",
    (4,1):"Fibonacci",(4,2):"Phibonacci",(4,3):"Fibonacci",(4,4):"Phibonacci",(4,5):"Phibonacci",(4,6):"Lucas",(4,7):"Lucas",(4,8):"Phibonacci",(4,9):"Phibonacci",
    (5,1):"Phibonacci",(5,2):"Lucas",(5,3):"Lucas",(5,4):"Phibonacci",(5,5):"Phibonacci",(5,6):"Fibonacci",(5,7):"Phibonacci",(5,8):"Fibonacci",(5,9):"Phibonacci",
    (6,1):"Lucas",(6,2):"Fibonacci",(6,3):"Tribonacci",(6,4):"Fibonacci",(6,5):"Lucas",(6,6):"Tribonacci",(6,7):"Phibonacci",(6,8):"Phibonacci",(6,9):"Tribonacci",
    (7,1):"Fibonacci",(7,2):"Lucas",(7,3):"Phibonacci",(7,4):"Phibonacci",(7,5):"Lucas",(7,6):"Fibonacci",(7,7):"Lucas",(7,8):"Lucas",(7,9):"Lucas",
    (8,1):"Fibonacci",(8,2):"Lucas",(8,3):"Phibonacci",(8,4):"Fibonacci",(8,5):"Phibonacci",(8,6):"Lucas",(8,7):"Fibonacci",(8,8):"Fibonacci",(8,9):"Fibonacci",
    (9,1):"Fibonacci",(9,2):"Lucas",(9,3):"Tribonacci",(9,4):"Phibonacci",(9,5):"Phibonacci",(9,6):"Tribonacci",(9,7):"Lucas",(9,8):"Fibonacci",(9,9):"Ninbonacci",
}
_FAMILY_CODE = {"Fibonacci": 0, "Lucas": 1, "Phibonacci": 2, "Tribonacci": 3, "Ninbonacci": 4}
_ORBIT_CODE  = {"Fibonacci": 0, "Lucas": 0, "Phibonacci": 0, "Tribonacci": 1, "Ninbonacci": 2}


def qa_packet(dist_left: int, dist_right: int) -> Dict[str, int]:
    """
    Compute exact QA feature packet from two non-negative graph/Manhattan distances.

    Convention: b = dist_left + 1, e = dist_right + 1 (A1 positive-integer states).
    All derived quantities use RAW (non-modular) arithmetic — d = b+e, a = b+2e.
    """
    b = dist_left + 1
    e = dist_right + 1
    d = b + e          # raw sum
    a = b + 2 * e      # raw second derived

    C = 2 * d * e
    F = a * b
    G = d * d + e * e
    X = C - F          # signed I = 2e²-b²
    I = abs(X)
    H = C + F
    W = d * (e + a)

    # A1 mod-9 residues (1-indexed)
    b9 = ((b - 1) % 9) + 1
    e9 = ((e - 1) % 9) + 1
    family = _FAMILY_GRID.get((b9, e9), "Fibonacci")

    return {
        "dist_left":   dist_left,
        "dist_right":  dist_right,
        "b": b, "e": e, "d": d, "a": a,
        "C": C, "F": F, "G": G, "I": I, "H": H,
        "I_signed": X,
        "W": W,
        "gap_2CF":     2 * C * F,
        "family_code": _FAMILY_CODE[family],
        "orbit_code":  _ORBIT_CODE[family],
    }


def qa_modular_residues(
    packet: Dict[str, int],
    moduli: Tuple[int, ...] = (9, 24),
) -> Dict[str, int]:
    """Return mod-m residues for core QA fields (A1-safe: value % modulus)."""
    out: Dict[str, int] = {}
    for m in moduli:
        for key in ("b", "e", "d", "a", "C", "F", "G", "I", "H"):
            out[f"{key}_mod{m}"] = packet[key] % m
    return out


# ---------------------------------------------------------------------------
# Koenig packet — per-point geometry layer, no dataset dependency
# ---------------------------------------------------------------------------

def koenig_packet(b: int, e: int) -> Dict[str, int]:
    """
    Compute per-point Koenig geometry features from QA (b, e) coordinates.

    These are purely arithmetic and require no fit state.
    """
    d = b + e
    a = b + 2 * e
    C = 2 * d * e
    F = a * b
    G = d * d + e * e
    X = C - F
    H = C + F
    W = d * (e + a)
    lf = Fraction(C * F, 12)
    return {
        "K_I":       abs(X),
        "K_G":       G,
        "K_H":       H,
        "K_I_signed": X,
        "K_W":       W,
        "K_L_num":   lf.numerator,
        "K_L_den":   lf.denominator,
        "K_L_floor": lf.numerator // lf.denominator,
        "K_24L_num": 24 * lf.numerator,
        "K_24L_den": lf.denominator,
        "K_gap_2CF": 2 * C * F,
        "K_conic":   1 if X > 0 else (-1 if X < 0 else 0),
    }


# ---------------------------------------------------------------------------
# Log-scale transform (mirrors benchmark script)
# ---------------------------------------------------------------------------

_LOG_KEYS = frozenset(("C", "F", "G", "I", "H", "gap_2CF",
                        "K_I", "K_G", "K_H", "K_W",
                        "K_L_num", "K_L_floor", "K_24L_num", "K_gap_2CF"))


def log_scale(value: float, key: str) -> float:
    if key in _LOG_KEYS:
        return math.log1p(max(value, 0.0))
    return float(value)


# ---------------------------------------------------------------------------
# Manhattan distance
# ---------------------------------------------------------------------------

def manhattan(row: int, col: int, anchor_row: int, anchor_col: int) -> int:
    return abs(row - anchor_row) + abs(col - anchor_col)
