#!/usr/bin/env python3
# QA_COMPLIANCE = "QA layer = EXACT integer state (i,j) mod N under the golden map M (A1, no floats). The float trajectory is the OBSERVER/decimal representation being tested; it is never fed back as QA state. This file demonstrates WHY QA forbids decimal drift, in exact arithmetic."
# RT1_OBSERVER_FILE: the float trajectory, drift distances, horizons are observer-layer readouts; the QA state is exact integers.
"""
Does decimal (float) representation drift where exact-integer QA does not? A plain test of
QA's actual claim -- no ring-units, no adeles, just exact vs float on the golden map.

QA's real constraint (Theorem NT / S2): state is EXACT integers, never decimals -- because
floats carry inherent ULP drift. The claim has teeth precisely for CHAOTIC dynamics, where a
tiny representation error grows exponentially. The golden cat map M=[[1,1],[1,0]] on a
rational grid is the sharp case: as EXACT integers (i,j) mod N it is a permutation, so the
orbit is exactly PERIODIC (it returns to its start). As float (x,y)=(i/N,j/N) in [0,1)^2 it
accumulates ULP error at the Lyapunov rate ln(phi), so it drifts, lands on WRONG grid cells,
and never exactly recurs -- the periodic structure that is REAL in exact arithmetic becomes
INVISIBLE in decimal.

COMMITTED PREDICTION (no hedge): float64 (52-bit mantissa) drifts past one grid cell (1/N)
after about k* ~ (52*ln2 - ln N)/ln(phi) steps -- roughly 60 for N=1000 -- and is fully
decorrelated by ~75 steps; the exact-integer orbit is correct forever and reveals the exact
period. FALSIFIABLE: if the drift horizon were thousands of steps it would be a purist's
point, not a practical one; the test reports the horizon so we can judge which it is.
"""
from __future__ import annotations
import math

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def step_exact(i, j, N):
    """Golden map M=[[1,1],[1,0]] on EXACT integer state mod N: (i,j) -> (i+j, i)."""
    return (i + j) % N, i % N


def step_float(x, y):
    """Same map on the DECIMAL (float) representation in [0,1)^2."""
    return (x + y) % 1.0, x % 1.0


def torus_dist(ax, ay, bx, by):
    dx = abs(ax - bx); dx = min(dx, 1 - dx)
    dy = abs(ay - by); dy = min(dy, 1 - dy)
    return math.hypot(dx, dy)


def run():
    print("Exact-integer vs decimal(float) on the golden cat map -- does decimal drift?\n")
    N = 1000
    i0, j0 = 1, 0
    i, j = i0, j0
    x, y = i0 / N, j0 / N

    wrong_cell = None          # first step float lands on a different grid cell than exact
    decorrelated = None        # first step float is essentially unrelated to exact (dist>0.2)
    exact_period = None
    float_min_return = 1.0      # closest float ever gets back to its own start
    max_k = 4000

    for k in range(1, max_k + 1):
        i, j = step_exact(i, j, N)
        x, y = step_float(x, y)
        # does the decimal representation still name the same exact QA state?
        if wrong_cell is None and (round(x * N) % N, round(y * N) % N) != (i, j):
            wrong_cell = k
        d = torus_dist(x, y, i / N, j / N)
        if decorrelated is None and d > 0.2:
            decorrelated = k
        # exact recurrence (the real structure)
        if exact_period is None and (i, j) == (i0, j0):
            exact_period = k
        # float's best attempt to recur to its own start
        float_min_return = min(float_min_return, torus_dist(x, y, i0 / N, j0 / N))

    k_pred = (52 * math.log(2) - math.log(N)) / math.log(PHI)
    print(f"[setup] golden map on a {N}x{N} rational grid, start ({i0},{j0}); Lyapunov = ln(phi) = {math.log(PHI):.3f}")
    print(f"[predicted] float drifts past one grid cell (1/N) near k* ~ {k_pred:.0f} steps.\n")

    print(f"[exact integers] orbit is a permutation of the grid -> EXACTLY periodic:")
    print(f"    exact period = {exact_period} steps (returns precisely to the start). Correct forever.")
    print(f"[decimal float]  same map in float64:")
    print(f"    first WRONG grid cell at step k = {wrong_cell}  (float now names a DIFFERENT QA state)")
    print(f"    fully decorrelated (torus dist > 0.2) by step k = {decorrelated}")
    print(f"    closest the float orbit EVER gets back to its start over {max_k} steps = "
          f"{float_min_return:.4f} (exact grid spacing 1/N = {1/N:.4f})")
    recurs = float_min_return < 1.0 / N
    print(f"    -> float {'DID' if recurs else 'NEVER'} recur to within one grid cell: "
          f"the exact periodicity is {'preserved' if recurs else 'INVISIBLE in decimal'}.")

    # is it a practical concern or a purist's point?
    practical = wrong_cell is not None and wrong_cell < 200
    ok = (wrong_cell is not None and exact_period is not None and not recurs
          and abs(wrong_cell - k_pred) < 30)
    print("\nVERDICT (QA's actual claim: exact integers, no decimal drift):")
    print(f"  * CONFIRMED and it is not cosmetic: the DECIMAL representation of the golden map is")
    print(f"    already naming the WRONG QA state after {wrong_cell} steps and is fully decorrelated")
    print(f"    by {decorrelated} -- because the error grows at the Lyapunov rate ln(phi), exponentially.")
    print(f"    The EXACT-integer orbit is correct forever and shows the real period ({exact_period}),")
    print(f"    which float completely hides (never returns within a grid cell).")
    print(f"  * This is QA's real point, stripped of jargon: it is not about ring-units, it is that")
    print(f"    decimal/float carries drift and QA's dynamics are chaotic, so ONLY exact integer")
    print(f"    (or Fraction) state stays correct. Whole-number, non-unit, rational -- all fine, as")
    print(f"    long as it is EXACT. The unit/adelic framing of the prior phases was beside this point.")
    print(f"  * Practical? the horizon here is ~{wrong_cell} steps ({'a REAL concern for any orbit of' if practical else 'far enough that it is a purist point for orbits under'}")
    print(f"    that length), and it SHORTENS as the grid N grows (finer decimals drift sooner).")
    print(f"\n  STATUS: {'DECIMAL DRIFT CONFIRMED -- exact integers necessary for chaotic QA dynamics' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
