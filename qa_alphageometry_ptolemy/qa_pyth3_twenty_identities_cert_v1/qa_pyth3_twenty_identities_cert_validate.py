# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.2 Twenty Identities: "
    "W=de+K, Y=C+E, W=F+Y equilateral check; Z=G+de dividing line with Z<W, Z>F, Z>Y; "
    "H^2-I^2=48L; female triangles (b even) C not divisible by 4); "
    "Theorem NT: 'equilateral triangle side', 'ellipse apogee/perigee', 'orbit radius' are "
    "observer projection labels of integer arithmetic identities; no float state, no QA orbit evolution"
)

"""
Cert [357] — QA Pyth-3 Twenty Identities and Equilateral Triangle Parameters

Source: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III, Chapter 2 pp.4-9
  p.7-8: 'W = C/2 + K, or = de + K for the sides of an equilateral triangle.'
         'Y = C + E, for the other division of the base of the triangle.'
         'One can see that Y + F must equal the side of this triangle, or Y + F = W.'
         'Z = G + C/2, or Z = G + de.'
         'H squared minus I squared = 48 L.'
  p.5-6: 'There was a second stage in which only the even numbered integers were used for
         root b... the first was for the conventional male triangles, and the second was for
         unconventional female triangles.'
  p.8:   'The values of J and K will be pellian numbers, being d squared - de = J,
         and d squared + de = K.'

Five claims from Chapter 2 of Pyth-3:
  C1: W=de+K, Y=C+E, W=F+Y for all prime Pythagorean (b odd, gcd=1) pairs;
      algebraic proof: W=d(d+2e), F=d^2-e^2, Y=e(2d+e), Y+F=d^2+2de=W ✓
  C2: Z=G+de=G+C/2; Z<W and Z>F and Z>Y for all valid (b odd) pairs;
      proof: Z-W=-(eb)<0; Z-F=e(2e+d)>0; Z-Y=d(d-e)=db>0
  C3: H^2-I^2=48L for all prime Pythagorean pairs;
      proof: (H+I)(H-I)=2F*2C=4CF=48*(CF/12)=48L
  C4: Female triangles (b even, e odd, gcd=1): d=b+e is odd, a=d+e is even,
      C=2de=2*(odd)^2 ≡ 2 (mod 4), NOT divisible by 4 (contrast with male C always 4-par)
  C5: Female Fibonacci seed (b=2,e=1,d=3,a=4): W=15, Y+F=7+8=15 ✓;
      H^2-I^2=192=48*4=48L ✓; Z=13<W=15 ✓; C=6≡2(mod 4) ✓; J=d^2-de=6=bd ✓
"""

from math import gcd


def _male_beads(max_b: int, max_e: int):
    """Yield (b, e, d, a) with b odd, gcd(b,e)=1 (male triangles)."""
    for b in range(1, max_b + 1, 2):
        for e in range(1, max_e + 1):
            if gcd(b, e) == 1:
                d = b + e
                a = d + e
                yield b, e, d, a


def _female_beads(max_b: int, max_e: int):
    """Yield (b, e, d, a) with b even, e odd, gcd(b,e)=1 (female triangles)."""
    for b in range(2, max_b + 1, 2):
        for e in range(1, max_e + 1, 2):
            if gcd(b, e) == 1:
                d = b + e
                a = d + e
                yield b, e, d, a


def check_c1() -> tuple[bool, str]:
    """W=de+K, Y=C+E; W=F+Y for all male prime Pythagorean pairs."""
    count = 0
    for b, e, d, a in _male_beads(25, 25):
        C = 2 * d * e
        E = e * e
        F = a * b
        K = a * d
        W = d * e + K      # = de + ad = d(e+a) = d(d+2e)
        Y = C + E          # = 2de + e^2 = e(2d+e) = e(d+a)
        # W = F + Y check
        assert W == F + Y, f"W={W} != F+Y={F+Y} at b={b},e={e}"
        # Also verify algebraic form: W = d*(d+2*e)
        W_alg = d * (d + 2 * e)
        assert W == W_alg, f"W={W} != d*(d+2e)={W_alg} at b={b},e={e}"
        # Y = e*(d+a) = e*(2d+e)
        Y_alg = e * (2 * d + e)
        assert Y == Y_alg, f"Y={Y} != e*(2d+e)={Y_alg} at b={b},e={e}"
        count += 1
    # Algebraic proof:
    # W = d(d+2e), F = d^2-e^2, Y = e(2d+e)
    # Y+F = e(2d+e) + d^2-e^2 = 2de+e^2+d^2-e^2 = d^2+2de = d(d+2e) = W ✓
    return True, (
        f"W=de+K, Y=C+E, W=F+Y verified for all {count} male pairs (b,e)<=25; "
        f"proof: Y+F=e(2d+e)+(d^2-e^2)=d^2+2de=d(d+2e)=W ✓"
    )


def check_c2() -> tuple[bool, str]:
    """Z=G+de; Z<W and Z>F and Z>Y for all valid male pairs."""
    count = 0
    for b, e, d, a in _male_beads(25, 25):
        C = 2 * d * e
        E = e * e
        F = a * b
        G = d * d + e * e
        K = a * d
        W = d * e + K
        Y = C + E
        Z = G + d * e      # = G + C/2
        # Z < W
        assert Z < W, f"Z={Z} >= W={W} at b={b},e={e}"
        # Z > F
        assert Z > F, f"Z={Z} <= F={F} at b={b},e={e}"
        # Z > Y
        assert Z > Y, f"Z={Z} <= Y={Y} at b={b},e={e}"
        # Algebraic: Z-W = (G+de) - d(d+2e) = d^2+e^2+de-d^2-2de = e^2-de = e(e-d) = -eb < 0
        diff_ZW = Z - W
        assert diff_ZW == -(e * b), f"Z-W={diff_ZW} != -eb={-(e*b)} at b={b},e={e}"
        count += 1
    # Proofs:
    # Z-W = (G+de)-(d^2+2de) = d^2+e^2+de-d^2-2de = e^2-de = e(e-d) = -e*b < 0 ✓
    # Z-F = d^2+e^2+de-(d^2-e^2) = 2e^2+de = e(2e+d) > 0 ✓
    # Z-Y = d^2+e^2+de-(2de+e^2) = d^2-de = d(d-e) = d*b > 0 ✓
    return True, (
        f"Z=G+de verified: Z<W, Z>F, Z>Y for all {count} male pairs (b,e)<=25; "
        f"proof Z-W=-eb<0, Z-F=e(2e+d)>0, Z-Y=db>0 ✓"
    )


def check_c3() -> tuple[bool, str]:
    """H^2-I^2=48L for all prime Pythagorean pairs."""
    count = 0
    for b, e, d, a in _male_beads(25, 25):
        C = 2 * d * e
        F = a * b
        H = C + F
        I = abs(C - F)
        abde = a * b * d * e
        assert abde % 6 == 0, f"6 does not divide abde at b={b},e={e}"
        L = abde // 6
        # H^2-I^2 = (H+I)(H-I) = 2*max(C,F)*2*min(C,F) = 4CF
        lhs = H * H - I * I
        rhs = 48 * L
        # Also: 4CF
        four_cf = 4 * C * F
        assert lhs == four_cf, f"H^2-I^2={lhs} != 4CF={four_cf} at b={b},e={e}"
        assert lhs == rhs, f"H^2-I^2={lhs} != 48L={rhs} at b={b},e={e}"
        count += 1
    # Proof: H^2-I^2=(H+I)(H-I); H+I=2max(C,F), H-I=2min(C,F), product=4CF=48*(CF/12)=48L ✓
    return True, (
        f"H^2-I^2=4CF=48L verified for all {count} male pairs (b,e)<=25; "
        f"proof: (H+I)(H-I)=2max(C,F)*2min(C,F)=4CF=48*(CF/12)=48L ✓"
    )


def check_c4() -> tuple[bool, str]:
    """Female triangles (b even, e odd, gcd=1): C not divisible by 4 (contrast with male)."""
    count_female_not_div4 = 0
    count_female_total = 0
    count_male_div4 = 0
    count_male_total = 0
    # Male: all have C divisible by 4 (cert [355] C3)
    for b, e, d, a in _male_beads(19, 19):
        C = 2 * d * e
        assert C % 4 == 0, f"Male: C={C} not divisible by 4 at b={b},e={e}"
        count_male_div4 += 1
        count_male_total += 1
    # Female: b even, e odd, gcd=1
    for b, e, d, a in _female_beads(19, 19):
        C = 2 * d * e
        # d=b+e=even+odd=odd, e=odd, so C=2*odd*odd ≡ 2 (mod 4)
        assert b % 2 == 0, f"Female b={b} not even"
        assert e % 2 == 1, f"Female e={e} not odd"
        assert d % 2 == 1, f"Female d={d} not odd (d=b+e=even+odd)"
        assert a % 2 == 0, f"Female a={a} not even (a=d+e=odd+odd)"
        assert C % 4 != 0, f"Female: C={C} unexpectedly divisible by 4 at b={b},e={e}"
        assert C % 2 == 0, f"Female: C={C} not even at b={b},e={e}"
        # Exactly 2 (mod 4) for female:
        assert C % 4 == 2, f"Female: C={C} not ≡2 (mod 4) at b={b},e={e}"
        count_female_not_div4 += 1
        count_female_total += 1
    return True, (
        f"Male ({count_male_div4} pairs): C always divisible by 4 ✓; "
        f"Female ({count_female_total} pairs): C always ≡2 (mod 4), NOT divisible by 4 ✓; "
        f"parity: female has b even, e odd, d odd, a even"
    )


def check_c5() -> tuple[bool, str]:
    """Female Fibonacci seed (b=2,e=1,d=3,a=4): all identities verified."""
    b, e, d, a = 2, 1, 3, 4
    # Basic bead relations
    assert d == b + e and a == d + e
    assert gcd(b, e) == 1
    # Female parity: b even, e odd, d odd, a even
    assert b % 2 == 0 and e % 2 == 1 and d % 2 == 1 and a % 2 == 0
    # Compute all 16+4 identities
    A = a * a            # 16
    B = b * b            # 4
    C = 2 * d * e        # 6
    D = d * d            # 9
    E = e * e            # 1
    F = a * b            # 8
    G = d * d + e * e    # 10
    H = C + F            # 14
    I = abs(C - F)       # |6-8| = 2
    J = b * d            # 6
    K = a * d            # 12
    L = a * b * d * e // 6  # 2*4*3*1//6 = 24//6 = 4
    W = d * e + K        # 3 + 12 = 15
    Y_val = C + E        # 6 + 1 = 7
    Z = G + d * e        # 10 + 3 = 13
    # Verify exact values
    assert A == 16 and B == 4 and C == 6 and D == 9 and E == 1
    assert F == 8 and G == 10 and H == 14 and I == 2 and J == 6
    assert K == 12 and L == 4
    assert W == 15 and Y_val == 7 and Z == 13
    # Check W = F + Y
    assert W == F + Y_val, f"W={W} != F+Y={F+Y_val}"
    # C is 2 (mod 4) — NOT male-like
    assert C % 4 == 2
    # J = d^2 - de = 9 - 3 = 6 = bd ✓ (Pellian form)
    assert d * d - d * e == J
    # K = d^2 + de = 9 + 3 = 12 = ad ✓
    assert d * d + d * e == K
    # H^2-I^2 = 48L
    assert H * H - I * I == 48 * L
    assert 196 - 4 == 192 == 48 * 4
    # Z < W, Z > F, Z > Y
    assert Z < W and Z > F and Z > Y_val
    return True, (
        f"Female seed (2,1,3,4): A=16,B=4,C=6,D=9,E=1,F=8,G=10,H=14,I=2,J=6,K=12,L=4; "
        f"W=15=F+Y=8+7 ✓; Z=13<W=15,Z>F=8,Z>Y=7 ✓; "
        f"H^2-I^2=192=48*4=48L ✓; C=6≡2(mod 4) (female parity) ✓; J=d^2-de=6=bd ✓"
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
        raise RuntimeError(f"cert [357] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
