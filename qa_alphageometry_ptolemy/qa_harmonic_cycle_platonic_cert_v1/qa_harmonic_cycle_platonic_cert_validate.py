# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1991) QA Vol II Books 3&4 — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (gcd, lcm); "
    "no QA state evolution; Theorem NT: polygon-vertex counts and "
    "harmonic-cycle periods are observer-layer labels on integer LCM "
    "structure; all arithmetic exact integer, no float"
)

"""
Cert [327] — QA Harmonic Cycle: Platonic Polygon Inscription

Source: Iverson (1991) QA Volume II Books 3 & 4, pp.10-12
"MULTIPLE WAVES", "THE HARMONIC CYCLE", "OTHER CYCLES",
"MOST BASIC MULTIPLE CYCLE"

The 60-unit harmonic cycle is formed by three coprime waves (3, 4, 5).
Their pairwise synchronous points inscribe three Platonic-solid face polygons:
  - (3,4) -> 5 synchronous points -> regular pentagon
  - (3,5) -> 4 synchronous points -> square
  - (4,5) -> 3 synchronous points -> equilateral triangle

These three are the only convex regular polygons that tile the faces of the
5 Platonic solids (icosahedron/dodecahedron use the pentagon; cube the square;
tetrahedron/octahedron the triangle).

Five claims certified exhaustively via integer arithmetic.
"""

from math import gcd


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _lcm3(a: int, b: int, c: int) -> int:
    return _lcm(_lcm(a, b), c)


def _sync_points(p: int, q: int, cycle: int) -> list[int]:
    """Multiples of lcm(p,q) that are <= cycle."""
    L = _lcm(p, q)
    return [k * L for k in range(1, cycle // L + 1)]


def check_c1() -> tuple[bool, str]:
    """60-unit harmonic cycle = lcm(3,4,5); all three waves pairwise coprime."""
    assert _lcm3(3, 4, 5) == 60, "lcm(3,4,5) must be 60"
    assert 3 * 4 * 5 == 60, "coprime product = lcm"
    assert gcd(3, 4) == 1 and gcd(4, 5) == 1 and gcd(3, 5) == 1, "not pairwise coprime"
    return True, "60 = lcm(3,4,5) = 3x4x5; all pairwise coprime"


def check_c2() -> tuple[bool, str]:
    """(3,4) synchronous points in 60-unit cycle: {12,24,36,48,60} = 5 -> pentagon."""
    pts = _sync_points(3, 4, 60)
    assert pts == [12, 24, 36, 48, 60], f"got {pts}"
    assert len(pts) == 5, "expected 5 synchronous points (pentagon)"
    gaps = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    assert all(g == 12 for g in gaps), "points not equally spaced"
    return True, f"(3,4): lcm=12, sync at {pts}; 5 equally spaced points = pentagon"


def check_c3() -> tuple[bool, str]:
    """(3,5) synchronous points in 60-unit cycle: {15,30,45,60} = 4 -> square."""
    pts = _sync_points(3, 5, 60)
    assert pts == [15, 30, 45, 60], f"got {pts}"
    assert len(pts) == 4, "expected 4 synchronous points (square)"
    gaps = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    assert all(g == 15 for g in gaps), "points not equally spaced"
    return True, f"(3,5): lcm=15, sync at {pts}; 4 equally spaced points = square"


def check_c4() -> tuple[bool, str]:
    """(4,5) synchronous points in 60-unit cycle: {20,40,60} = 3 -> equilateral triangle."""
    pts = _sync_points(4, 5, 60)
    assert pts == [20, 40, 60], f"got {pts}"
    assert len(pts) == 3, "expected 3 synchronous points (equilateral triangle)"
    gaps = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    assert all(g == 20 for g in gaps), "points not equally spaced"
    return True, f"(4,5): lcm=20, sync at {pts}; 3 equally spaced points = equilateral triangle"


def check_c5() -> tuple[bool, str]:
    """Fundamental harmonic cycles: 30=lcm(2,3,5), 42=lcm(2,3,7), 60=lcm(3,4,5), 105=lcm(3,5,7)."""
    cases: list[tuple[int, tuple[int, int, int]]] = [
        (30,  (2, 3, 5)),
        (42,  (2, 3, 7)),
        (60,  (3, 4, 5)),
        (105, (3, 5, 7)),
    ]
    for period, waves in cases:
        a, b, c = waves
        computed = _lcm3(a, b, c)
        assert computed == period, f"lcm{waves} = {computed}, expected {period}"
    return True, "Four fundamental harmonic cycles: 30=lcm(2,3,5), 42=lcm(2,3,7), 60=lcm(3,4,5), 105=lcm(3,5,7)"


def main() -> None:
    checks = [check_c1, check_c2, check_c3, check_c4, check_c5]
    passed = 0
    for fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {fn.__name__}: {msg}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise RuntimeError(f"cert [327] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
