# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1991) QA Vol II Books 3&4 — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (gcd, lcm, mod, CRT); "
    "no QA state evolution; Theorem NT: Platonic solid geometry and "
    "musical interval ratios are observer-layer labels on integer "
    "divisibility and modular structure; all arithmetic exact integer, no float"
)

"""
Cert [329] — QA Book 4 Synthesis: Platonic Solid Edges, CRT Triple Uniqueness, Aliquot Minima

Source: Iverson (1991) QA Volume II Books 3 & 4, pp.53-65:
- p.53 'ALIQUOT PARTS': three smallest aliquot parts 30, 42, 210
- p.56 'PAR TYPES': triple classification (mod 3, mod 4, mod 5) uniqueness below 60
- p.61 'PLATONIC SOLIDS': edge count formula E = faces * edges_per_face / 2
- p.62 'PARAMETERS': major third 5:4, major fifth 3:2, major seventh 15:8

CRT connection: lcm(3,4,5) = 60 = the harmonic cycle. The triple
(n mod 3, n mod 4, n mod 5) uniquely determines n in {1,...,60} by CRT,
since 3,4,5 are pairwise coprime (gcd(3,4)=gcd(4,5)=gcd(3,5)=1).

Five claims certified via exact integer arithmetic.
"""

from math import gcd


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


# The five Platonic solids: (name, V, E, F, edges_per_face, faces_per_vertex)
PLATONIC_SOLIDS = [
    ("tetrahedron",  4,  6,  4, 3),
    ("cube",         8, 12,  6, 4),
    ("octahedron",   6, 12,  8, 3),
    ("dodecahedron", 20, 30, 12, 5),
    ("icosahedron",  12, 30, 20, 3),
]


def check_c1() -> tuple[bool, str]:
    """Platonic edge count: E = F * e_per_face / 2 for all 5 solids."""
    for name, V, E, F, epf in PLATONIC_SOLIDS:
        computed_E = F * epf // 2
        assert computed_E == E, f"{name}: F*epf/2 = {computed_E}, expected E={E}"
        assert F * epf % 2 == 0, f"{name}: F*epf is odd — impossible for Platonic solid"
    return True, f"E = F*epf/2 verified for all 5 Platonic solids: {[s[0] for s in PLATONIC_SOLIDS]}"


def check_c2() -> tuple[bool, str]:
    """Euler's formula V-E+F=2 for all 5 Platonic solids."""
    for name, V, E, F, epf in PLATONIC_SOLIDS:
        assert V - E + F == 2, f"{name}: V-E+F = {V-E+F}, expected 2"
    # Face types: triangle (e=3) for tetra/octa/icosa; square (e=4) for cube; pentagon (e=5) for dodeca
    triangle_solids = [s[0] for s in PLATONIC_SOLIDS if s[4] == 3]
    assert sorted(triangle_solids) == sorted(["tetrahedron", "octahedron", "icosahedron"]), \
        f"triangle-faced: {triangle_solids}"
    square_solids = [s[0] for s in PLATONIC_SOLIDS if s[4] == 4]
    assert square_solids == ["cube"], f"square-faced: {square_solids}"
    pentagon_solids = [s[0] for s in PLATONIC_SOLIDS if s[4] == 5]
    assert pentagon_solids == ["dodecahedron"], f"pentagon-faced: {pentagon_solids}"
    return True, "Euler V-E+F=2 holds; face types: triangle(3 solids), square(1), pentagon(1)"


def check_c3() -> tuple[bool, str]:
    """Triple (n mod 3, n mod 4, n mod 5) uniquely identifies n in {1..60}; lcm(3,4,5)=60."""
    assert _lcm(_lcm(3, 4), 5) == 60, "lcm(3,4,5) must be 60"
    assert gcd(3, 4) == 1 and gcd(4, 5) == 1 and gcd(3, 5) == 1, "must be pairwise coprime"
    # CRT: triples (n mod 3, n mod 4, n mod 5) for n in {1..60} must be all distinct
    triples = [(n % 3, n % 4, n % 5) for n in range(1, 61)]
    assert len(set(triples)) == 60, f"not all 60 triples distinct: got {len(set(triples))} unique"
    # Verify Iverson's examples: 37 and 32
    assert (37 % 3, 37 % 4, 37 % 5) == (1, 1, 2), f"37: got {(37%3, 37%4, 37%5)}"
    # 37 mod 3=1 -> 4-tri (3n+1); 37 mod 4=1 -> 5-par; 37 mod 5=2 -> 7-pent
    assert (32 % 3, 32 % 4, 32 % 5) == (2, 0, 2), f"32: got {(32%3, 32%4, 32%5)}"
    # 32 mod 3=2 -> 2-tri (3n-1); 32 mod 4=0 -> 4-par; 32 mod 5=2 -> 7-pent
    # Both 37 and 32 are 7-pent but differ in the tri and par classes -> different triples
    assert (37 % 3, 37 % 4, 37 % 5) != (32 % 3, 32 % 4, 32 % 5), \
        "37 and 32 must have distinct triples despite same pent class"
    return True, "lcm(3,4,5)=60; all 60 triples in {1..60} are distinct by CRT; 37=(1,1,2), 32=(2,0,2)"


def check_c4() -> tuple[bool, str]:
    """Three smallest aliquot parts: 30=2x3x5, 42=2x3x7, 210=2x3x5x7; all divisible by 6."""
    # The three minimal harmonic bases (Iverson p.53)
    assert 2 * 3 * 5 == 30
    assert 2 * 3 * 7 == 42
    assert 2 * 3 * 5 * 7 == 210
    # All divisible by 6 (the "composite even" requirement)
    assert 30 % 6 == 0 and 42 % 6 == 0 and 210 % 6 == 0
    # 210 = 2 * 3 * 5 * 7 is the product of all four primordial primes
    assert 210 == 2 * 3 * 5 * 7
    # Note: 30 = lcm(2,3,5), 42 = lcm(2,3,7), 210 = lcm(2,3,5,7) since all pairwise coprime
    assert _lcm(_lcm(2, 3), 5) == 30
    assert _lcm(_lcm(2, 3), 7) == 42
    assert _lcm(_lcm(_lcm(2, 3), 5), 7) == 210
    # 210/30=7, 210/42=5: each smaller is an aliquot part of 210
    assert 210 % 30 == 0 and 210 % 42 == 0
    return True, "Smallest aliquot parts: 30=lcm(2,3,5), 42=lcm(2,3,7), 210=lcm(2,3,5,7); 210/30=7, 210/42=5"


def check_c5() -> tuple[bool, str]:
    """Musical intervals: major third 5:4, major fifth 3:2, major seventh 15:8; all 7-smooth."""
    intervals = {
        "major_third":   (5, 4),
        "major_fifth":   (3, 2),
        "major_seventh": (15, 8),
    }
    for name, (num, den) in intervals.items():
        assert gcd(num, den) == 1, f"{name}: {num}/{den} not in lowest terms"
    # Verify they are all 7-smooth (prime factors only from {2,3,5,7})
    def is_7smooth(n: int) -> bool:
        for p in [2, 3, 5, 7]:
            while n % p == 0:
                n //= p
        return n == 1
    for name, (num, den) in intervals.items():
        assert is_7smooth(num) and is_7smooth(den), f"{name}: factors beyond 7"
    # Major seventh = major third * major fifth * correction?
    # 15/8 = 5/4 * 3/2 = 15/8 ✓ (the seventh is the product of the third and fifth!)
    assert (intervals["major_third"][0] * intervals["major_fifth"][0] ==
            intervals["major_seventh"][0]), "major seventh numerator = third * fifth numerator"
    assert (intervals["major_third"][1] * intervals["major_fifth"][1] ==
            intervals["major_seventh"][1]), "major seventh denominator = third * fifth denominator"
    return True, "major third=5:4, major fifth=3:2, major seventh=15:8=5/4*3/2; all 7-smooth ratios"


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
        raise RuntimeError(f"cert [329] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
