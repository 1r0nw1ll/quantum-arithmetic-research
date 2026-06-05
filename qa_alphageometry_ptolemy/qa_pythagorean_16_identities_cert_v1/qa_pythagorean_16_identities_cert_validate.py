# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pythagorean identities, "
    "bead number algebra); no QA state evolution; Theorem NT: triangle, "
    "ellipse, and 'identities' are observer-layer labels on bead number "
    "integer relationships; all arithmetic exact integer, no float"
)

"""
Cert [336] — QA Pythagorean 16 Identities: Sum-Difference Squares

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, pp.12-13 (Ch.1 Preview):
  "a base C which equals 2de; an altitude F which equals ab; and a
   hypotenuse G which equals d^2+e^2. Alternatively, the sum of the
   hypotenuse and base is a^2; the difference of the hypotenuse and
   base is b^2; the sum of the hypotenuse and altitude is 2d^2; the
   difference of the hypotenuse and altitude is 2e^2; and the sum and
   difference of the altitude and base are functionally prime numbers,
   H and I."

QA bead numbers: (b, e, d, a) with d=b+e, a=b+2e, gcd(b,e)=1.

Primary identities:
  C = 2de  (base = 2 times product of two middle bead numbers)
  F = ab   (altitude = product of outer bead numbers)
  G = d^2 + e^2  (hypotenuse)

Sum-difference identities:
  G + C = a^2     (hyp + base = square of largest bead number)
  G - C = b^2     (hyp - base = square of smallest bead number)
  G + F = 2*d^2   (hyp + alt = double square of d)
  G - F = 2*e^2   (hyp - alt = double square of e)

Five claims certified via integer algebra.
"""

from math import gcd


def _bead_to_triangle(b: int, e: int) -> tuple[int, int, int, int, int, int]:
    """Compute (d,a,C,F,G) from bead numbers (b,e) with d=b+e, a=b+2e."""
    d = b + e
    a = b + 2 * e
    C = 2 * d * e
    F = a * b
    G = d * d + e * e
    return d, a, C, F, G, b


def check_c1() -> tuple[bool, str]:
    """Primary definitions: C=2de, F=ab, G=d^2+e^2; C^2+F^2=G^2 (Pythagorean)."""
    test_cases = [
        (1, 2),  # d=3, a=5: C=12,F=5,G=13 (5-12-13 triangle)
        (1, 4),  # d=5, a=9: C=40,F=9,G=41 (9-40-41)
        (3, 2),  # d=5, a=7: C=20,F=21,G=29 (20-21-29)
        (1, 6),  # d=7,a=13: C=84,F=13,G=85 (13-84-85)
        (3, 4),  # d=7,a=11: C=56,F=33,G=65 (33-56-65)
        (5, 2),  # d=7, a=9: C=28,F=45,G=53 (28-45-53)
        (5, 4),  # d=9,a=13: C=72,F=65,G=97 (65-72-97)
        (7, 2),  # d=9,a=11: C=36,F=77,G=85 (36-77-85)
    ]
    for b, e in test_cases:
        assert gcd(b, e) == 1 and b % 2 == 1, f"({b},{e}) must have b odd, gcd=1"
        d, a, C, F, G, _ = _bead_to_triangle(b, e)
        assert C * C + F * F == G * G, (
            f"({b},{e}): C^2+F^2!=G^2: {C*C}+{F*F}={C*C+F*F} != {G*G}"
        )
        # Verify definitions
        assert C == 2 * d * e, f"({b},{e}): C={C} != 2de={2*d*e}"
        assert F == a * b, f"({b},{e}): F={F} != ab={a*b}"
        assert G == d * d + e * e, f"({b},{e}): G={G} != d^2+e^2={d*d+e*e}"
    return True, f"C=2de, F=ab, G=d^2+e^2; C^2+F^2=G^2 verified for {len(test_cases)} pairs"


def check_c2() -> tuple[bool, str]:
    """G + C = a^2 and G - C = b^2 for all test cases."""
    test_cases = [
        (1, 2), (1, 4), (3, 2), (1, 6), (3, 4), (5, 2), (5, 4), (7, 2),
        (1, 8), (3, 8), (5, 6), (7, 4), (9, 2),
    ]
    for b, e in test_cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        d, a, C, F, G, _ = _bead_to_triangle(b, e)
        assert G + C == a * a, (
            f"({b},{e}): G+C={G+C} != a^2={a*a} (a={a})"
        )
        assert G - C == b * b, (
            f"({b},{e}): G-C={G-C} != b^2={b*b} (b={b})"
        )
    return True, f"G+C=a^2 and G-C=b^2 verified for {len(test_cases)} bead pairs"


def check_c3() -> tuple[bool, str]:
    """G + F = 2*d^2 and G - F = 2*e^2 for all test cases."""
    test_cases = [
        (1, 2), (1, 4), (3, 2), (1, 6), (3, 4), (5, 2), (5, 4), (7, 2),
        (1, 8), (3, 8), (5, 6), (7, 4), (9, 2),
    ]
    for b, e in test_cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        d, a, C, F, G, _ = _bead_to_triangle(b, e)
        assert G + F == 2 * d * d, (
            f"({b},{e}): G+F={G+F} != 2d^2={2*d*d} (d={d})"
        )
        assert G - F == 2 * e * e, (
            f"({b},{e}): G-F={G-F} != 2e^2={2*e*e} (e={e})"
        )
    return True, f"G+F=2d^2 and G-F=2e^2 verified for {len(test_cases)} bead pairs"


def check_c4() -> tuple[bool, str]:
    """All six identities hold simultaneously; L=abde/6=CF/12."""
    test_cases = [(1, 2), (1, 4), (3, 2), (1, 6), (3, 4), (5, 2), (7, 2), (3, 8)]
    for b, e in test_cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        d, a, C, F, G, _ = _bead_to_triangle(b, e)
        # All six identities simultaneously
        assert G + C == a * a
        assert G - C == b * b
        assert G + F == 2 * d * d
        assert G - F == 2 * e * e
        assert C * C + F * F == G * G
        # L identity (when divisible)
        L_bead = a * b * d * e // 6
        L_CF = C * F // 12
        assert L_bead == L_CF, f"({b},{e}): L_bead={L_bead} != L_CF={L_CF}"
    return True, "All six identities + L=abde/6=CF/12 hold simultaneously for 8 pairs"


def check_c5() -> tuple[bool, str]:
    """Algebraic proof: G+C=a^2 follows from definitions (no case checking required)."""
    # G + C = (d^2+e^2) + 2de = (d+e)^2 = a^2  (since a=d+e=b+2e and d=b+e)
    # Wait: a = b+2e, d=b+e → d+e = b+2e = a ✓
    # G + C = d^2 + e^2 + 2de = (d+e)^2 = a^2 ✓ (algebraic identity, holds for all integers)

    # G - C = (d^2+e^2) - 2de = (d-e)^2 = b^2  (since d-e = (b+e)-e = b)
    # ✓

    # G + F = (d^2+e^2) + ab = d^2+e^2 + (b+2e)*b = d^2+e^2+b^2+2be
    #       = d^2 + (b+e)^2 = d^2 + d^2 = 2d^2  (since b+e=d)
    # Wait: e^2+b^2+2be = (b+e)^2 = d^2, so G+F = d^2+d^2 = 2d^2 ✓

    # G - F = (d^2+e^2) - ab = d^2+e^2 - (b+2e)b = d^2+e^2-b^2-2be
    #       = (d^2-b^2-2be) + e^2 = (b+e)^2-b^2-2be + e^2 = b^2+2be+e^2-b^2-2be+e^2 = 2e^2 ✓

    # Verify algebraically for symbolic substitution with concrete values
    for b in range(1, 10, 2):
        for e in range(1, 10):
            if gcd(b, e) != 1:
                continue
            d, a, C, F, G, _ = _bead_to_triangle(b, e)
            # Direct algebraic verification
            assert (d + e) * (d + e) == a * a, f"(d+e)^2 != a^2"
            assert (d - e) * (d - e) == b * b, f"(d-e)^2 != b^2"
            assert d * d + d * d == G + F, f"2d^2 != G+F"
            assert e * e + e * e == G - F, f"2e^2 != G-F"
    return True, (
        "G+C=(d+e)^2=a^2; G-C=(d-e)^2=b^2; G+F=2d^2; G-F=2e^2 "
        "proven algebraically from definitions d=b+e, a=b+2e; verified 20+ pairs"
    )


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
        raise RuntimeError(f"cert [336] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
