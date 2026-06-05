# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pythagorean concentric circle area laws: "
    "three arithmetic-mean trios b-d-a/F-D-G/I²-G²-H²; H²+I²=2G²; H²-G²=G²-I²=24L; "
    "2D+2E=A+B; J+K=C+2J=2D); "
    "Theorem NT: 'concentric circle area', 'independent second dimension', 'divisible by 24' "
    "are observer classification labels; causal structure is integer polynomial algebra on "
    "bead numbers (b,e,d,a); no float state, no QA orbit evolution"
)

"""
Cert [352] — QA Pythagorean Concentric Circle Area Divisibility

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, Chapter V pp.53-64
  p.54: "b-d-a, F-D-G, and I²-G²-H², where the center term is the mean of the other two."
  p.54: "H²+I²=2G² ... G² is equal to half the sum of H² and I²."
  p.54-55: "In every case, these remaining areas are divisible by 24. In the case of the
    first triangle, the 3-4-5 triangle, these remaining areas are, in fact, 24."
  p.57: "2D + 2E = A + B"
  p.59: "J+K = C+2J = ... = 2D"

Five claims:
  C1: Three arithmetic-mean trios: (b,d,a) diff=e; (F,D,G) diff=e²; (I²,G²,H²) diff=4deab
      — center term is the arithmetic mean of the outer two
  C2: H²+I²=2G² for all prime Pythagorean triangles (b odd, gcd(b,e)=1)
  C3: H²-G² = G²-I² = 24L for all prime Pythagorean triangles;
      for the 3-4-5 triangle (b=1,e=1): both gaps = 24 exactly
  C4: 2D+2E = A+B (the double-sum identity)
  C5: J+K = C+2J = 2D (the double-square partition identity)
"""

from math import gcd


def _valid_beads(max_b: int, max_e: int):
    """Yield (b, e, d, a) with b odd, gcd(b,e)=1, 1<=b<=max_b, 1<=e<=max_e."""
    for b in range(1, max_b + 1, 2):
        for e in range(1, max_e + 1):
            if gcd(b, e) == 1:
                d = b + e
                a = d + e
                yield b, e, d, a


def check_c1() -> tuple[bool, str]:
    """Three arithmetic-mean trios: b-d-a (diff e), F-D-G (diff e^2), I^2-G^2-H^2 (diff 4deab)."""
    count = 0
    for b, e, d, a in _valid_beads(20, 20):
        # Trio 1: b, d, a -- d is arithmetic mean, common diff = e
        assert d - b == e and a - d == e, f"b-d-a not AP: b={b},d={d},a={a},e={e}"

        # Trio 2: F, D, G -- D is arithmetic mean, common diff = e^2
        F = a * b
        D = d * d
        G = d * d + e * e
        diff2 = e * e
        assert D - F == diff2, f"F-D not e^2: F={F},D={D},diff={D-F},e^2={diff2} b={b},e={e}"
        assert G - D == diff2, f"D-G not e^2: D={D},G={G},diff={G-D},e^2={diff2} b={b},e={e}"

        # Trio 3: I^2, G^2, H^2 -- G^2 is arithmetic mean, diff = 4deab = 24L
        C = 2 * d * e
        H = C + F
        I = abs(C - F)
        area_diff = 4 * d * e * a * b  # 24L = 24 * deab/6 = 4deab
        H2 = H * H
        G2 = G * G
        I2 = I * I
        assert H2 - G2 == area_diff, f"H^2-G^2!=4deab: H={H},G={G},diff={H2-G2},4deab={area_diff} b={b},e={e}"
        assert G2 - I2 == area_diff, f"G^2-I^2!=4deab: G={G},I={I},diff={G2-I2},4deab={area_diff} b={b},e={e}"
        assert I2 + H2 == 2 * G2, f"I^2+H^2!=2G^2: b={b},e={e}"
        count += 1

    return True, (
        f"All three arithmetic-mean trios verified for {count} valid pairs (b,e)<=20: "
        f"b-d-a (diff=e); F-D-G (diff=e^2); I^2-G^2-H^2 (diff=4deab=24L)"
    )


def check_c2() -> tuple[bool, str]:
    """H^2+I^2=2G^2 for all prime Pythagorean triangles."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        F = a * b
        C = 2 * d * e
        G = d * d + e * e
        H = C + F
        I = abs(C - F)
        lhs = H * H + I * I
        rhs = 2 * G * G
        assert lhs == rhs, f"H^2+I^2!=2G^2: H={H},I={I},G={G},b={b},e={e}"
        count += 1
    # 3-4-5 triangle: b=1,e=1,d=2,a=3,H=7,G=5,I=1
    assert 7*7 + 1*1 == 2*5*5
    return True, (
        f"H^2+I^2=2G^2 verified for all {count} valid pairs (b,e)<=25; "
        f"3-4-5 check: 7^2+1^2=50=2x5^2; "
        f"algebraic proof: H^2-I^2=4CF=8deab; each half of split = 4deab = H^2-G^2 = G^2-I^2"
    )


def check_c3() -> tuple[bool, str]:
    """H^2-G^2 = G^2-I^2 = 24L; for 3-4-5 triangle both gaps = 24."""
    # 3-4-5 triangle: b=1,e=1,d=2,a=3,H=7,G=5,I=1,L=1
    assert 7*7 - 5*5 == 24
    assert 5*5 - 1*1 == 24

    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        F = a * b
        C = 2 * d * e
        G = d * d + e * e
        H = C + F
        I = abs(C - F)
        deab = d * e * a * b
        assert deab % 6 == 0, f"deab not divisible by 6: deab={deab} b={b},e={e}"
        L = deab // 6
        twenty_four_L = 24 * L
        assert H*H - G*G == twenty_four_L, f"H^2-G^2!=24L: diff={H*H-G*G}, 24L={twenty_four_L} b={b},e={e}"
        assert G*G - I*I == twenty_four_L, f"G^2-I^2!=24L: diff={G*G-I*I}, 24L={twenty_four_L} b={b},e={e}"
        hg_diff = H * H - G * G
        assert hg_diff % 24 == 0, f"H^2-G^2 not divisible by 24: {hg_diff} b={b},e={e}"
        count += 1

    return True, (
        f"H^2-G^2=G^2-I^2=24L verified for all {count} valid pairs (b,e)<=25; "
        f"3-4-5 triangle: both gaps=24 (L=1); "
        f"algebraic identity: H^2-G^2=4deab=24L since L=deab/6"
    )


def check_c4() -> tuple[bool, str]:
    """2D+2E = A+B (the double-sum identity)."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        A = a * a
        B = b * b
        D = d * d
        E = e * e
        lhs = 2 * D + 2 * E
        rhs = A + B
        assert lhs == rhs, f"2D+2E!=A+B: 2D+2E={lhs}, A+B={rhs} b={b},e={e}"
        count += 1
    # Algebraic: 2(b+e)^2+2e^2 = 2b^2+4be+2e^2+2e^2 = 2b^2+4be+4e^2 = (b+2e)^2+b^2
    return True, (
        f"2D+2E=A+B verified for all {count} valid pairs (b,e)<=25; "
        f"algebraic proof: 2(b+e)^2+2e^2 = 2b^2+4be+4e^2 = (b+2e)^2+b^2"
    )


def check_c5() -> tuple[bool, str]:
    """J+K = C+2J = 2D (the double-square partition identity)."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        J = b * d
        K = a * d
        C = 2 * d * e
        D = d * d
        jk = J + K
        c2j = C + 2 * J
        two_d = 2 * D
        assert jk == two_d, f"J+K!=2D: J+K={jk}, 2D={two_d} b={b},e={e}"
        assert c2j == two_d, f"C+2J!=2D: C+2J={c2j}, 2D={two_d} b={b},e={e}"
        count += 1
    # J+K = bd+ad = d(b+a). Now b+a = b+(b+2e) = 2(b+e) = 2d. So J+K = 2d^2 = 2D.
    # C+2J = 2de+2bd = 2d(e+b) = 2d*d = 2D.
    return True, (
        f"J+K=C+2J=2D verified for all {count} valid pairs (b,e)<=25; "
        f"algebraic proof: J+K=d(b+a)=d*2d=2D; C+2J=2d(e+b)=2D"
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
        raise RuntimeError(f"cert [352] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
