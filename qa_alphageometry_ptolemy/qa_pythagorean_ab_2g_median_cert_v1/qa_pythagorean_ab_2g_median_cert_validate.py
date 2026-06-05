# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (bead median identities A+B=2G, "
    "a+b=2d); no QA state evolution; Theorem NT: 'sum', 'median', 'average' are "
    "observer-layer labels on integer bead number relationships; all arithmetic "
    "exact integer, no float"
)

"""
Cert [340] — QA Pythagorean A+B=2G and a+b=2d Bead Median Identities

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, Chapter VII pp.62-64
  "(d+e)^2 + (d-e)^2 = 2G. But since d+e=a and d-e=b the formula reduces to
   a^2+b^2=2G, or A+B=2G."
  "the value of G is the median value between A and B."
  "d is the median value of a and b" (i.e., a+b=2d)

Also: D is the median of F and G (F+G=2D) — independently derivable but
  stated by Iverson as the third median identity alongside A+B=2G.

Bead number definitions (A2 compliant, raw — no mod reduction):
  d = b + e
  a = b + 2*e
  A = a*a, B = b*b, G = d*d + e*e

Claims:
  C1: a + b = 2*d   (d is the bead median of a and b)
  C2: A + B = 2*G   (G is the mediant square of A and B: G=(A+B)/2)
  C3: Algebraic proof: A+B = (b+2e)^2 + b^2 = 2b^2+4be+4e^2 = 2(d^2+e^2) = 2G
  C4: F + G = 2*D   (D is the mediant square of F and G)
  C5: All three median identities hold simultaneously for 15 coprime pairs
"""

from math import gcd


def _params(b: int, e: int) -> dict:
    d = b + e
    a = b + 2 * e
    A = a * a
    B = b * b
    C = 2 * d * e
    D = d * d
    E = e * e
    F = a * b
    G = d * d + e * e
    return dict(b=b, e=e, d=d, a=a, A=A, B=B, C=C, D=D, E=E, F=F, G=G)


def check_c1() -> tuple[bool, str]:
    """d is the bead median of a and b: a + b = 2*d."""
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2),(11,2),(1,8),(3,10),(5,6)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        assert p['a'] + b == 2 * p['d'], (
            f"({b},{e}): a+b={p['a']+b} != 2d={2*p['d']}"
        )
    return True, f"a+b=2d (d is bead median of a and b) verified for {len(cases)} pairs"


def check_c2() -> tuple[bool, str]:
    """G is the mediant square of A and B: A + B = 2*G."""
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2),(11,2),(1,8),(3,10),(5,6)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        assert p['A'] + p['B'] == 2 * p['G'], (
            f"({b},{e}): A+B={p['A']+p['B']} != 2G={2*p['G']}"
        )
    return True, f"A+B=2G (G is mediant of A and B) verified for {len(cases)} pairs"


def check_c3() -> tuple[bool, str]:
    """Algebraic proof: A+B=(b+2e)^2+b^2=2b^2+4be+4e^2=2(d^2+e^2)=2G."""
    # Verify for all odd b, coprime to e, b and e < 14
    count = 0
    for b in range(1, 14, 2):
        for e in range(1, 14):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = b + 2 * e
            A = a * a
            B = b * b
            G = d * d + e * e
            # Algebraic expansion: (b+2e)^2+b^2 = 2b^2+4be+4e^2 = 2(b^2+2be+2e^2)
            # And 2G = 2(d^2+e^2) = 2((b+e)^2+e^2) = 2(b^2+2be+e^2+e^2) = 2(b^2+2be+2e^2)
            assert A + B == 2 * G, f"({b},{e}): algebraic proof failed A+B={A+B} 2G={2*G}"
            count += 1
    return True, (
        f"Algebraic proof A+B=(b+2e)^2+b^2=2(d^2+e^2)=2G verified for {count} coprime pairs b,e<14"
    )


def check_c4() -> tuple[bool, str]:
    """D is the mediant of F and G: F + G = 2*D."""
    # F = ab, G = d^2+e^2, D = d^2; F+G=ab+d^2+e^2; 2D=2d^2
    # F = ab = (b+2e)*b = b^2+2be; G = b^2+2be+2e^2; F+G = 2b^2+4be+2e^2 = 2(b+e)^2 = 2d^2 = 2D
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2),(11,2),(1,8),(3,10),(5,6)]
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        assert p['F'] + p['G'] == 2 * p['D'], (
            f"({b},{e}): F+G={p['F']+p['G']} != 2D={2*p['D']}"
        )
    return True, f"F+G=2D (D is mediant of F and G) verified for {len(cases)} pairs"


def check_c5() -> tuple[bool, str]:
    """All three median identities hold simultaneously: a+b=2d, A+B=2G, F+G=2D."""
    cases = [(1,2),(1,4),(3,2),(1,6),(3,4),(5,2),(5,4),(7,2),(3,8),(7,4),(9,2),(11,2),(1,8),(3,10),(5,6)]
    count = 0
    for b, e in cases:
        if gcd(b, e) != 1 or b % 2 == 0:
            continue
        p = _params(b, e)
        ok1 = (p['a'] + b == 2 * p['d'])
        ok2 = (p['A'] + p['B'] == 2 * p['G'])
        ok3 = (p['F'] + p['G'] == 2 * p['D'])
        assert ok1 and ok2 and ok3, (
            f"({b},{e}): simultaneous median check failed: a+b=2d={ok1}, A+B=2G={ok2}, F+G=2D={ok3}"
        )
        count += 1
    return True, f"All three median identities hold simultaneously for {count} coprime pairs"


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
        raise RuntimeError(f"cert [340] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
