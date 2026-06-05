# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-1 Ch.VIII Ellipse of Archimedes: "
    "2D=J+K=F+G=C+2J; K-J=C; K+J=2D; semiminor^2=DF; eccentricity=2D/C=d/e; J*K=DF); "
    "Theorem NT: 'ellipse', 'orbit', 'apogee', 'perigee', 'eccentricity' are observer projections; "
    "QA layer = bead arithmetic only; no float state, no QA orbit evolution"
)

"""
Cert [365] — QA Pyth-1 Ellipse of Archimedes (Ch.VIII)

Source: Iverson (1993) Pythagorean Arithmetic Vol I, Chapter VIII pp.87-93
  p.90 (Fig.8 caption): 'The major diameter is 2D which is equal to J+K, and equal to
                         C+2J, or to F+G.'
  p.90 (Fig.8 caption): 'J forms the perigee. K forms the apogee. F forms the semi-latus
                         rectum. ... The major diameter is 2D which is equal to J+K.'
  p.91: 'the value of 2D/C = d/e = c, has an additional service to perform in QA'
  p.93: '(df)^2 = D^2 - (C/2)^2 = D(D-E) = DF'
  p.93: 'F can be a square number only when both of its parametric numbers (F=ab) are
         also square numbers.'

Five claims:
  C1: 2D = J+K = F+G = C+2J (three equal representations of the major diameter)
      proofs:
        J+K = bd+ad = d(b+a) = d*2d = 2d² = 2D ✓
        F+G = (d²-e²)+(d²+e²) = 2d² = 2D ✓
        C+2J = 2de+2bd = 2d(e+b) = 2d*d = 2d² = 2D ✓ (since b+e=d)
  C2: K-J = C and K+J = 2D (ellipse focal structure)
      proofs:
        K-J = ad-bd = (a-b)d = 2e*d = 2de = C ✓
        K+J = ad+bd = (a+b)d = 2d*d = 2d² = 2D ✓
  C3: (semiminor diameter)² = DF; semiminor is integer iff F is a perfect square
      proof: Pythagorean right triangle in the ellipse: hypotenuse=D, half-base=C/2=de;
             altitude² = D² - (C/2)² = D² - d²e² = D² - D*E = D(D-E) = D*F ✓
             (since D-E = d²-e² = F)
             Semiminor = df̃ where d̃=√D, f̃=√F; integer iff D (always square) and F are squares;
             D=d² is always a square; semiminor is integer iff F=ab is a perfect square.
  C4: Eccentricity (QA definition) = 2D/C = d/e (bead ratio)
      proof: 2D/C = 2d²/(2de) = d/e ✓
      Note: Iverson defines eccentricity as d/e (the inverse of the conventional definition
      where eccentricity < 1 for an ellipse; QA convention: c=d/e > 1).
  C5: J*K = D*F (product identity relating apogee-perigee to semimajor-semilatus)
      proof: J*K = bd * ad = ab * d² = F * D = DF ✓
"""

from math import gcd


def _prime_pairs(max_b: int, max_e: int):
    """Yield (b, e, d, a) with b odd, gcd(b,e)=1."""
    for b in range(1, max_b + 1, 2):
        for e in range(1, max_e + 1):
            if gcd(b, e) == 1:
                d = b + e
                a = d + e
                yield b, e, d, a


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(n ** 0.5)
    return r * r == n or (r + 1) * (r + 1) == n


def check_c1() -> tuple[bool, str]:
    """2D = J+K = F+G = C+2J for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(35, 35):
        D = d * d
        E = e * e
        F = a * b        # = d*d - e*e = d²-e²
        G = d * d + E    # = d²+e²
        C = 2 * d * e
        J = b * d
        K = a * d
        two_D = 2 * D
        assert J + K == two_D, f"J+K={J+K} != 2D={two_D} at b={b},e={e}"
        assert F + G == two_D, f"F+G={F+G} != 2D={two_D} at b={b},e={e}"
        assert C + 2 * J == two_D, f"C+2J={C+2*J} != 2D={two_D} at b={b},e={e}"
        count += 1
    # proof C+2J: C+2J=2de+2bd=2d(e+b)=2d*d=2D since b+e=d ✓
    return True, (
        f"2D=J+K=F+G=C+2J verified for all {count} prime pairs (b,e)<=35; "
        f"proofs: J+K=d(b+a)=2d²=2D; F+G=2d²=2D; C+2J=2d(e+b)=2d²=2D "
        f"(since b+e=d always) ✓"
    )


def check_c2() -> tuple[bool, str]:
    """K-J = C and K+J = 2D for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(35, 35):
        C = 2 * d * e
        J = b * d
        K = a * d
        two_D = 2 * d * d
        assert K - J == C, f"K-J={K-J} != C={C} at b={b},e={e}"
        assert K + J == two_D, f"K+J={K+J} != 2D={two_D} at b={b},e={e}"
        count += 1
    # proof K-J: K-J=(a-b)d=2e*d=2de=C ✓
    # proof K+J: K+J=(a+b)d=2d*d=2D ✓
    return True, (
        f"K-J=C and K+J=2D verified for all {count} prime pairs (b,e)<=35; "
        f"proof K-J: (a-b)d=2e*d=C; proof K+J: (a+b)d=2d²=2D ✓; "
        f"interpretation: perigee J and apogee K are equidistant from D (major semidiameter) "
        f"by half-the-foci-distance C/2 each ✓"
    )


def check_c3() -> tuple[bool, str]:
    """(semiminor diameter)^2 = DF; semiminor is integer iff F is a perfect square."""
    count = 0
    for b, e, d, a in _prime_pairs(35, 35):
        D = d * d
        E = e * e
        F = a * b
        # (semiminor)^2 = D^2 - (C/2)^2 = D^2 - (de)^2 = D^2 - D*E = D(D-E) = D*F
        # (C/2)^2 = (de)^2 = d^2 * e^2 = D * E
        C_half_sq = (d * e) * (d * e)   # = D * E
        semiminor_sq = D * D - C_half_sq
        assert semiminor_sq == D * F, (
            f"semiminor²={semiminor_sq} != D*F={D*F} at b={b},e={e}"
        )
        # Verify algebraically: D*D - D*E = D*(D-E) = D*F (since D-E=F)
        assert D - E == F, f"D-E={D-E} != F={F} at b={b},e={e}"
        count += 1

    # Find pairs where F is a perfect square (semiminor is integer)
    perfect_sq_cases = []
    for b, e, d, a in _prime_pairs(100, 100):
        F = a * b
        D = d * d
        if _is_perfect_square(F):
            perfect_sq_cases.append((b, e, F, D * F))

    # For a primitive pair, F=ab is a square iff both a and b are squares
    # (since gcd(a,b)=1 for primitive pairs)
    for b, e, d, a in _prime_pairs(50, 50):
        F = a * b
        if _is_perfect_square(F):
            assert _is_perfect_square(a) and _is_perfect_square(b), (
                f"F={F} is a square but a={a} or b={b} is not — coprime product rule violated at b={b},e={e}"
            )

    return True, (
        f"(semiminor)²=DF verified for all {count} prime pairs (b,e)<=35; "
        f"proof: D²-(de)²=D²-DE=D(D-E)=DF since D-E=d²-e²=F; "
        f"found {len(perfect_sq_cases)} pairs with F=perfect square (semiminor integer) up to (b,e)<=100; "
        f"examples: {perfect_sq_cases[:3] if perfect_sq_cases else 'none found in range'}; "
        f"for coprime a,b: ab=□ iff a=□ and b=□ individually ✓"
    )


def check_c4() -> tuple[bool, str]:
    """Eccentricity = 2D/C = d/e as a Fraction (exact, no float)."""
    from fractions import Fraction
    count = 0
    for b, e, d, a in _prime_pairs(35, 35):
        D = d * d
        C = 2 * d * e
        ecc = Fraction(2 * D, C)      # = 2d²/(2de) = d/e
        expected = Fraction(d, e)
        assert ecc == expected, f"2D/C={ecc} != d/e={expected} at b={b},e={e}"
        # For all primitive pairs, eccentricity > 1 (since d = b+e > e always)
        assert ecc > 1, f"eccentricity {ecc} <= 1 at b={b},e={e}"
        count += 1
    # Verify two illustrative examples from Iverson's text
    # (1,2,5,3) = b=1,e=2: d=3,a=5; C=12, D=9; ecc=2*9/12=18/12=3/2 ✓
    b, e, d, a = 1, 2, 3, 5
    C = 2 * d * e   # 12
    D = d * d       # 9
    assert Fraction(2 * D, C) == Fraction(d, e) == Fraction(3, 2)
    return True, (
        f"Eccentricity 2D/C=d/e verified exactly (Fraction arithmetic) for {count} pairs (b,e)<=35; "
        f"proof: 2D/C=2d²/(2de)=d/e; all ecc>1 (d>e always for b≥1); "
        f"example (b=1,e=2): d=3,e=2, C=12, D=9, ecc=18/12=3/2=d/e ✓"
    )


def check_c5() -> tuple[bool, str]:
    """J*K = D*F for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(35, 35):
        D = d * d
        F = a * b
        J = b * d
        K = a * d
        assert J * K == D * F, f"J*K={J*K} != D*F={D*F} at b={b},e={e}"
        count += 1
    # proof: J*K = bd*ad = ab*d² = F*D ✓
    # structural meaning: J*K = DF and C² = 4DE so (J*K)*(C²/4) = DF*DE = D²EF
    # Also: K/J = a/b (ratio of apogee to perigee = ratio of the two bead lengths)
    for b, e, d, a in _prime_pairs(15, 15):
        J = b * d
        K = a * d
        from fractions import Fraction
        assert Fraction(K, J) == Fraction(a, b), f"K/J != a/b at b={b},e={e}"
    return True, (
        f"J*K=DF verified for all {count} prime pairs (b,e)<=35; "
        f"proof: J*K=(bd)(ad)=ab*d²=F*D=DF; "
        f"also K/J=a/b (ratio of apogee to perigee equals ratio of bead lengths a to b); "
        f"DF is the semiminor² from C3, so J*K=semiminor² ✓"
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
        raise RuntimeError(f"cert [365] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
