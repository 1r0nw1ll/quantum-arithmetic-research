# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-1 Ch.V Internal Relationships: "
    "d=(b+a)/2; D=(F+G)/2; G^2=(I^2+H^2)/2; 2D+2E=A+B; I,G,H pairwise coprime; "
    "J+K=C+2J=2D; F-C=b^2-2e^2); "
    "Theorem NT: 'circle', 'area', 'triangle', 'orbit' are observer projection labels; "
    "no float state, no QA orbit evolution"
)

"""
Cert [362] — QA Pyth-1 Internal Relationships (Ch.V)

Source: Iverson (1993) Pythagorean Arithmetic Vol I, Chapter V pp.54-65
  p.57: 'Some of them are: 2D+2E=A+B, and the three three-part series of, b-d-a,
         F-D-G, and I^2-G^2-H^2, where the center term is the mean of the other two.'
  p.53: 'H^2+I^2=2G^2'
  p.52: 'J+K = C+2J = ... = 2D'
  p.54: 'I, G, and H are usually prime numbers, are always coprime to each other,
         and are always functionally prime.'
  p.51: 'When b > e: F > C (Table 1a); when e > b: C > F (Table 1b).'
         [algebraic form: F - C = b^2 - 2e^2]

Five claims:
  C1: Three arithmetic mean trios:
      (a) d = (b+a)/2 (b, d, a are in arithmetic progression with step e)
      (b) D = (F+G)/2 (F, D, G are arithmetic; D-F=E, G-D=E)
      (c) G^2 = (I^2+H^2)/2, equivalently H^2+I^2=2G^2
  C2: 2D+2E = A+B (equivalently A+B = a^2+b^2 = 2d^2+2e^2 = 2D+2E)
  C3: I, G, H are always pairwise coprime: gcd(I,G)=gcd(I,H)=gcd(G,H)=1
      for all primitive Pythagorean pairs (b,e) with gcd(b,e)=1
  C4: J+K = C+2J = 2D
      proof: J=bd, K=ad; J+K=d(b+a)=d*2d=2D; C+2J=2de+2bd=2d(e+b)=2d^2=2D
  C5: F - C = b^2 - 2e^2 exactly; F > C iff b^2 > 2e^2
      proof: F-C = d^2-e^2-2de = (d-e)^2-2e^2 = b^2-2e^2
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


def check_c1() -> tuple[bool, str]:
    """Three arithmetic mean trios: b-d-a, F-D-G, I^2-G^2-H^2."""
    count = 0
    for b, e, d, a in _prime_pairs(30, 30):
        C = 2 * d * e
        F = a * b
        G = d * d + e * e
        D = d * d
        E = e * e
        H = C + F
        I = abs(C - F)
        # (a) d = (b+a)/2: arithmetic mean of b and a
        assert 2 * d == b + a, f"2d={2*d} != b+a={b+a} at b={b},e={e}"
        # (b) D = (F+G)/2: arithmetic mean of F and G; equivalently D-F=E and G-D=E
        assert 2 * D == F + G, f"2D={2*D} != F+G={F+G} at b={b},e={e}"
        assert D - F == E and G - D == E, f"D-F={D-F},G-D={G-D},E={E}"
        # (c) G^2 = (I^2+H^2)/2: G^2 is arithmetic mean of I^2 and H^2
        assert 2 * G * G == I * I + H * H, f"H^2+I^2={H*H+I*I} != 2G^2={2*G*G}"
        count += 1
    return True, (
        f"Three mean trios verified for all {count} prime pairs (b,e)<=30: "
        f"2d=b+a; 2D=F+G (D-F=D=G-D=E); H^2+I^2=2G^2 ✓"
    )


def check_c2() -> tuple[bool, str]:
    """2D+2E = A+B for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(30, 30):
        A = a * a
        B = b * b
        D = d * d
        E = e * e
        # 2D+2E = 2d^2+2e^2 = (d+e)^2+(d-e)^2 = a^2+b^2 = A+B
        assert A + B == 2 * D + 2 * E, f"A+B={A+B} != 2D+2E={2*D+2*E}"
        count += 1
    # Algebraic proof: A+B = (d+e)^2+(d-e)^2 = d^2+2de+e^2+d^2-2de+e^2 = 2d^2+2e^2 = 2D+2E
    return True, (
        f"2D+2E=A+B verified for all {count} prime pairs (b,e)<=30; "
        f"proof: A+B=(d+e)^2+(d-e)^2=2d^2+2e^2=2D+2E ✓"
    )


def check_c3() -> tuple[bool, str]:
    """I, G, H are always pairwise coprime for all primitive Pythagorean pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(35, 35):
        C = 2 * d * e
        F = a * b
        G = d * d + e * e
        H = C + F
        I = abs(C - F)
        # All three pairwise coprime
        assert gcd(I, G) == 1, f"gcd(I={I},G={G})={gcd(I,G)} != 1 at b={b},e={e}"
        assert gcd(I, H) == 1, f"gcd(I={I},H={H})={gcd(I,H)} != 1 at b={b},e={e}"
        assert gcd(G, H) == 1, f"gcd(G={G},H={H})={gcd(G,H)} != 1 at b={b},e={e}"
        # Also G,H,I all odd (G=5-par→odd; H=I odd from cert [361] C4)
        assert G % 2 == 1
        assert H % 2 == 1
        assert I % 2 == 1
        count += 1
    return True, (
        f"I,G,H pairwise coprime for all {count} pairs (b,e)<=35: "
        f"gcd(I,G)=gcd(I,H)=gcd(G,H)=1 throughout; all three always odd ✓"
    )


def check_c4() -> tuple[bool, str]:
    """J+K = C+2J = 2D for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(30, 30):
        C = 2 * d * e
        D = d * d
        J = b * d
        K = a * d
        # J+K = d(b+a) = d*2d = 2D
        assert J + K == 2 * D, f"J+K={J+K} != 2D={2*D} at b={b},e={e}"
        # C+2J = 2de+2bd = 2d(e+b) = 2d^2 = 2D
        assert C + 2 * J == 2 * D, f"C+2J={C+2*J} != 2D={2*D} at b={b},e={e}"
        count += 1
    # Algebraic proof:
    # J+K = bd+ad = d(b+a) = d*(b + b+2e) = d*2(b+e) = d*2d = 2d^2 = 2D
    # (since a=b+2e, so b+a=2b+2e=2(b+e)=2d)
    # C+2J = 2de+2bd = 2d(e+b) = 2d^2 = 2D
    return True, (
        f"J+K=C+2J=2D verified for all {count} prime pairs (b,e)<=30; "
        f"proof: J+K=d(b+a)=d*2d=2D; C+2J=2d(e+b)=2d^2=2D ✓"
    )


def check_c5() -> tuple[bool, str]:
    """F - C = b^2 - 2e^2 exactly; F > C iff b^2 > 2e^2."""
    count_f_gt_c = 0
    count_c_gt_f = 0
    count_equal = 0
    for b, e, d, a in _prime_pairs(30, 30):
        C = 2 * d * e
        F = a * b
        # F - C = d^2-e^2-2de = (d-e)^2 - 2e^2 = b^2 - 2e^2
        diff = F - C
        expected = b * b - 2 * e * e
        assert diff == expected, f"F-C={diff} != b^2-2e^2={expected} at b={b},e={e}"
        # dichotomy check
        if b * b > 2 * e * e:
            assert F > C, f"b^2>2e^2 but F<C at b={b},e={e}"
            count_f_gt_c += 1
        elif b * b < 2 * e * e:
            assert C > F, f"b^2<2e^2 but C<F at b={b},e={e}"
            count_c_gt_f += 1
        else:
            count_equal += 1
        count = count_f_gt_c + count_c_gt_f + count_equal
    # Algebraic proof: F-C = d^2-e^2-2de = (d-e)^2-2e^2 = b^2-2e^2
    return True, (
        f"F-C=b^2-2e^2 exact for all pairs (b,e)<=30; "
        f"F>C ({count_f_gt_c} pairs, b^2>2e^2); C>F ({count_c_gt_f} pairs, b^2<2e^2); "
        f"equal ({count_equal} pairs, b^2=2e^2 — impossible over integers); "
        f"proof: F-C=(d-e)^2-2e^2=b^2-2e^2 ✓"
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
        raise RuntimeError(f"cert [362] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
