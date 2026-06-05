# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (divisibility laws for bead numbers "
    "and triangle identities); no QA state evolution; Theorem NT: 'factor 2/3/4/5', "
    "'parity', 'area divisible by 6' are observer-layer labels on integer bead number "
    "divisibility structure; all arithmetic exact integer modular, no float"
)

"""
Cert [342] — QA Pythagorean Divisibility Laws (Iverson Ch. IX Proofs 1-9,11)

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, Chapter IX pp.95-99
  Statements (1)-(9) and (11) with proofs:
  (1) Either d or e must be even.
  (2) a is always odd.
  (3) Factor 2 in every bead set {b,e,d,a}.
  (4) Factor 3 in every bead set {b,e,d,a}.
  (5)/(6) C, F, or G is divisible by 5.
  (7) C or F contains factor 3.
  (8) C is always 4-par (divisible by 4).
  (9) Area CF/2 is divisible by 6.
  (11) G is always 5-par (G ≡ 1 mod 4).

Bead number definitions (A2 compliant, raw — no mod reduction):
  d = b + e; a = b + 2*e; C = 2*d*e; F = a*b; G = d*d + e*e

Five claims:
  C1: Exactly one of d, e is even; a is always odd (Statements 1 and 2)
  C2: Factor 3 in {b, e, d, a} — at least one of the four bead numbers divisible by 3
  C3: C is always divisible by 4 (4-par) since one of d,e is even → C=2de has factor 4
  C4: G is always 5-par (G ≡ 1 mod 4): G = d^2+e^2 with one even, one odd → 0+1=1 mod 4
  C5: Area CF/2 is divisible by 6 for all prime Pythagorean triangles (Statements 7,8,9)
"""

from math import gcd


def _params(b: int, e: int) -> dict:
    d = b + e
    a = b + 2 * e
    C = 2 * d * e
    F = a * b
    G = d * d + e * e
    return dict(b=b, e=e, d=d, a=a, C=C, F=F, G=G)


def check_c1() -> tuple[bool, str]:
    """Exactly one of d,e is even; a is always odd (Statements 1 and 2)."""
    # Since b is odd: if e is odd then d=b+e=odd+odd=even; if e is even then d=b+e=odd+even=odd
    # Either way exactly one of d,e is even. And a=d+e = odd+even or even+odd = always odd.
    count = 0
    for b in range(1, 20, 2):
        for e in range(1, 20):
            if gcd(b, e) != 1:
                continue
            p = _params(b, e)
            d, a = p['d'], p['a']
            # Exactly one of d, e is even
            assert (d % 2 == 0) != (e % 2 == 0), (
                f"({b},{e}): d={d}, e={e} — both or neither even"
            )
            # a is always odd
            assert a % 2 == 1, f"({b},{e}): a={a} is not odd"
            count += 1
    return True, (
        f"Exactly one of d,e even; a always odd — verified for {count} coprime pairs b,e<20"
    )


def check_c2() -> tuple[bool, str]:
    """Factor 3 in {b,e,d,a}: at least one of the four bead numbers divisible by 3."""
    # Proof (Iverson): All tri-classification cases exhaust to show one of b,e,d,a ≡ 0 mod 3.
    count = 0
    fails = 0
    for b in range(1, 30, 2):
        for e in range(1, 30):
            if gcd(b, e) != 1:
                continue
            p = _params(b, e)
            d, a = p['d'], p['a']
            divisible_by_3 = any(x % 3 == 0 for x in [b, e, d, a])
            assert divisible_by_3, (
                f"({b},{e}): none of b={b}, e={e}, d={d}, a={a} divisible by 3"
            )
            count += 1
    return True, (
        f"At least one of {{b,e,d,a}} divisible by 3 for {count} coprime pairs b,e<30"
    )


def check_c3() -> tuple[bool, str]:
    """C is always divisible by 4 (4-par): C=2de, one of d,e even → C has factor 4."""
    # C = 2*d*e. Since one of d,e is even (C1), say d=2k, then C=2*(2k)*e=4ke divisible by 4.
    count = 0
    for b in range(1, 30, 2):
        for e in range(1, 30):
            if gcd(b, e) != 1:
                continue
            p = _params(b, e)
            C = p['C']
            assert C % 4 == 0, f"({b},{e}): C={C} not divisible by 4"
            count += 1
    return True, f"C divisible by 4 (4-par) for {count} coprime pairs b,e<30"


def check_c4() -> tuple[bool, str]:
    """G is always 5-par (G ≡ 1 mod 4): G=d^2+e^2 with one even,one odd → 0+1=1 mod 4."""
    # Squares: even^2 ≡ 0 mod 4; odd^2 ≡ 1 mod 4
    # G = d^2 + e^2 = 0+1 or 1+0 ≡ 1 mod 4 → G is 5-par (4n+1)
    count = 0
    for b in range(1, 30, 2):
        for e in range(1, 30):
            if gcd(b, e) != 1:
                continue
            p = _params(b, e)
            G = p['G']
            assert G % 4 == 1, f"({b},{e}): G={G} ≡ {G%4} mod 4, not 5-par"
            count += 1
    return True, f"G ≡ 1 (mod 4) (5-par) for {count} coprime pairs b,e<30"


def check_c5() -> tuple[bool, str]:
    """Area CF/2 is divisible by 6 (Statements 7,8,9): C divisible by 4 (→CF/2 by 2); 3 divides C or F."""
    # Area = CF/2. Since C is 4-par, C=4k, so CF/2=2kF which is even (divisible by 2).
    # Factor 3: since 3 divides one of {b,e,d,a}, and F=ab, C=2de, either C or F divisible by 3.
    # Hence area=CF/2 divisible by lcm(2,3)=6.
    count = 0
    for b in range(1, 30, 2):
        for e in range(1, 30):
            if gcd(b, e) != 1:
                continue
            p = _params(b, e)
            C, F = p['C'], p['F']
            area_x2 = C * F  # = 2 * (CF/2); avoid division, check CF divisible by 12
            assert area_x2 % 12 == 0, (
                f"({b},{e}): C*F={area_x2} not divisible by 12 → area=CF/2 not divisible by 6"
            )
            count += 1
    return True, (
        f"C*F divisible by 12 → area=CF/2 divisible by 6 for {count} coprime pairs b,e<30"
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
        raise RuntimeError(f"cert [342] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
