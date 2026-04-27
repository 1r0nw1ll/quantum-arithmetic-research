"""Classical cyclic coordinate search — Python port of Kochenderfer 2026 Opt Algorithm 7.2.

QA_COMPLIANCE = "classical-baseline port — derivative-free continuous coordinate descent with line search along basis vectors; vocabulary alignment with cert [191] tier hierarchy on QA-discrete side. Continuous line search means firewall-rejected as causal QA dynamics."

Source: (Kochenderfer 2026) Algorithms for Optimization, 2nd ed., MIT Press, §7.1 + Algorithm 7.2.
Transcribed from QA-MEM verbatim excerpt at
docs/theory/kochenderfer_optimization_excerpts.md#opt-7-1-cyclic-coordinate-search
"""

from __future__ import annotations
import math
from typing import Callable


def basis(i: int, n: int) -> list:
    """Return the i-th basis vector of length n (zero-indexed)."""
    return [1.0 if k == i else 0.0 for k in range(n)]


def line_search_brent(
    f: Callable,
    x: list,
    direction: list,
    bracket: tuple = (-10.0, 10.0),
    tol: float = 1e-6,
) -> list:
    """1D minimization of f(x + α*direction) over α in `bracket`.

    Simple golden-section variant — sufficient for the CCS demo. For
    production use, replace with scipy.optimize.minimize_scalar.
    """
    a, b = bracket
    phi = (math.sqrt(5) - 1) / 2
    lo = b - phi * (b - a)
    hi = a + phi * (b - a)
    f_lo = f([xi + lo * di for xi, di in zip(x, direction)])
    f_hi = f([xi + hi * di for xi, di in zip(x, direction)])
    while abs(b - a) > tol:
        if f_lo < f_hi:
            b, hi, f_hi = hi, lo, f_lo
            lo = b - phi * (b - a)
            f_lo = f([xi + lo * di for xi, di in zip(x, direction)])
        else:
            a, lo, f_lo = lo, hi, f_hi
            hi = a + phi * (b - a)
            f_hi = f([xi + hi * di for xi, di in zip(x, direction)])
    alpha = (a + b) / 2
    return [xi + alpha * di for xi, di in zip(x, direction)]


def cyclic_coordinate_descent(
    f: Callable,
    x_init: list,
    eps: float = 1e-4,
    max_cycles: int = 100,
) -> list:
    """Cyclic coordinate descent per Opt Algorithm 7.2.

    Alternates line searches along each of the n basis vectors. Iterates
    full cycles until step over a full cycle drops below eps.
    """
    x = list(x_init)
    n = len(x)
    delta = float("inf")
    cycles = 0
    while abs(delta) > eps and cycles < max_cycles:
        x_orig = list(x)
        for i in range(n):
            direction = basis(i, n)
            x = line_search_brent(f, x, direction)
        delta = math.sqrt(sum((xi - oi) ** 2 for xi, oi in zip(x, x_orig)))
        cycles += 1
    return x


if __name__ == "__main__":
    # Minimize f(x) = (x_0 - 1)² + (x_1 - 2)² over R²; minimum at (1, 2)
    def f(x):
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

    x_opt = cyclic_coordinate_descent(f, [0.0, 0.0], eps=1e-6)
    print(f"Found x* = {x_opt} (true minimum at (1.0, 2.0))")
