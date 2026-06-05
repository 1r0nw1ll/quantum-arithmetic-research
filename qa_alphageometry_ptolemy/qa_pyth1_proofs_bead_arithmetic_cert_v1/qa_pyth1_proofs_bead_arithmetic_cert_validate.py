# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-1 Ch.IX Proofs: "
    "parity lemma one-of-{d,e}-even; a always odd; factor 3 in every bead set; "
    "area CF/2 divisible by 6; {b,e,d,a} pairwise coprime); "
    "Theorem NT: no float state, no QA orbit evolution"
)

"""
Cert [366] — QA Pyth-1 Proofs: Bead Arithmetic Laws (Ch.IX)

Source: Iverson (1993) Pythagorean Arithmetic Vol I, Chapter IX pp.94-100
  p.95 Statement (1): 'Either d or e must be an even number.'
  p.95 Statement (2): 'The value of a is always an odd number.'
  p.96 Statement (4): 'The factor 3 will be represented in every set of bead numbers.'
  p.98 Statement (9): 'The area of every prime Pythagorean triangle is divisible by 6.'
  p.99 Statement (12): 'd is prime to b and e. And a is prime to b, d, and e.'

Five claims (Statements 1, 2, 4, 9, 12 from Ch.IX):
  C1: Exactly one of {d, e} is even for every primitive pair (b, e)
      proof: b is always odd (given). If e is even: d=b+e=odd+even=odd → e even, d odd.
             If e is odd: d=b+e=odd+odd=even → e odd, d even.
             In both cases exactly one of {d, e} is even. ✓
  C2: a is always odd for every primitive pair (b, e)
      proof: a=d+e; by C1, {d,e} contains one even and one odd;
             a = even + odd = odd ✓
  C3: Factor 3 divides at least one element of {b, e, d, a} for every primitive pair
      proof: complete mod-3 case analysis on (b mod 3, e mod 3):
        (b≡0): 3|b ✓
        (e≡0): 3|e ✓
        (b≡1,e≡1): a=b+2e≡1+2=3≡0 → 3|a ✓
        (b≡1,e≡2): d=b+e≡1+2=3≡0 → 3|d ✓
        (b≡2,e≡1): d=b+e≡2+1=3≡0 → 3|d ✓
        (b≡2,e≡2): a=b+2e≡2+4=6≡0 → 3|a ✓
      All 6 cases covered ✓
  C4: Area = CF/2 ≡ 0 (mod 6) for all prime Pythagorean pairs
      proof: C=2de is divisible by 4 (since exactly one of {d,e} is even, say e=2k;
             C=2d·2k=4dk). F=ab is odd (both a,b odd). Area=CF/2=4dk·ab/2=2dk·ab.
             Need 3|area: by C3, 3 divides some bead; if 3|d or 3|e → 3|C → 3|2dk → 3|dk;
             if 3|b or 3|a → 3|F=ab → 3|ab. Either way 3|area=2dk·ab. So 6|area ✓.
  C5: All four bead numbers {b, e, d, a} are pairwise coprime
      proof (using gcd(b,e)=1 as the primitive pair axiom):
        gcd(b,d)=gcd(b,b+e)=gcd(b,e)=1 ✓
        gcd(e,d)=gcd(e,b+e)=gcd(e,b)=gcd(b,e)=1 ✓
        gcd(b,a)=gcd(b,b+2e)=gcd(b,2e); b odd → gcd(b,2)=1; gcd(b,e)=1 → gcd(b,2e)=1 ✓
        gcd(e,a)=gcd(e,b+2e)=gcd(e,b)=gcd(b,e)=1 ✓
        gcd(d,a)=gcd(b+e,b+2e)=gcd(b+e,e)=gcd(b,e)=1 ✓
      Therefore all six pairwise gcds are 1. ✓
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
    """Exactly one of {d, e} is even for every primitive pair."""
    count = 0
    for b, e, d, a in _prime_pairs(100, 100):
        d_even = (d % 2 == 0)
        e_even = (e % 2 == 0)
        assert d_even != e_even, (
            f"both or neither of {{d,e}} are even at b={b},e={e}: d={d},e={e}"
        )
        # When e even: d is odd; when e odd: d is even
        if e % 2 == 0:
            assert d % 2 == 1, f"e even but d not odd at b={b},e={e}"
        else:
            assert d % 2 == 0, f"e odd but d not even at b={b},e={e}"
        count += 1
    return True, (
        f"Exactly one of {{d,e}} is even verified for all {count} prime pairs (b,e)<=100; "
        f"proof: b odd; if e even: d=b+e=odd+even=odd; if e odd: d=b+e=odd+odd=even; "
        f"exactly one parity flip in either case ✓"
    )


def check_c2() -> tuple[bool, str]:
    """a is always odd for every primitive pair."""
    count = 0
    for b, e, d, a in _prime_pairs(100, 100):
        assert a % 2 == 1, f"a={a} is even at b={b},e={e}"
        count += 1
    # Also b is always odd (given), so {b, a} are both odd; {d, e} split even/odd
    return True, (
        f"a is odd verified for all {count} prime pairs (b,e)<=100; "
        f"proof: a=d+e; by C1 exactly one of {{d,e}} is even; "
        f"a=even+odd=odd or a=odd+even=odd ✓; "
        f"both b and a are always odd; d and e always split (one even, one odd) ✓"
    )


def check_c3() -> tuple[bool, str]:
    """Factor 3 divides at least one of {b, e, d, a} for every primitive pair."""
    count = 0
    for b, e, d, a in _prime_pairs(100, 100):
        divisible_by_3 = [x for x in (b, e, d, a) if x % 3 == 0]
        assert len(divisible_by_3) >= 1, (
            f"None of b={b},e={e},d={d},a={a} divisible by 3"
        )
        count += 1
    # Enumerate which bead is 3-divisible across the range
    b_count = e_count = d_count = a_count = 0
    for b, e, d, a in _prime_pairs(50, 50):
        if b % 3 == 0:
            b_count += 1
        if e % 3 == 0:
            e_count += 1
        if d % 3 == 0:
            d_count += 1
        if a % 3 == 0:
            a_count += 1
    return True, (
        f"Factor 3 in {{b,e,d,a}} verified for all {count} prime pairs (b,e)<=100; "
        f"proof: 6 mod-3 cases all covered; for (b,e)<=50 breakdown: "
        f"b divisible by 3: {b_count}; e: {e_count}; d: {d_count}; a: {a_count} ✓"
    )


def check_c4() -> tuple[bool, str]:
    """Area = CF/2 is divisible by 6 for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(100, 100):
        C = 2 * d * e
        F = a * b
        # Area must be an integer
        assert C % 2 == 0, f"C={C} is odd — impossible since C=2de"
        area_times_2 = C * F       # = 2 * area
        area = area_times_2 // 2   # C is always even so C*F/2 is integer
        assert area * 2 == area_times_2
        assert area % 6 == 0, (
            f"Area={area} not divisible by 6 at b={b},e={e}; "
            f"C={C}, F={F}"
        )
        count += 1
    # Find the minimum area (should be 6, from the 3,4,5 triangle with b=1,e=1)
    min_area = None
    for b, e, d, a in _prime_pairs(10, 10):
        C = 2 * d * e
        F = a * b
        area = C * F // 2
        if min_area is None or area < min_area:
            min_area = area
    return True, (
        f"Area=CF/2≡0(mod 6) verified for all {count} prime pairs (b,e)<=100; "
        f"minimum area={min_area} (the 3,4,5 triangle gives area=6); "
        f"proof: C≡0(mod 4) → area=CF/2 is even; 3 divides some bead → 3|C or 3|F → 3|area; "
        f"lcm(2,3)=6 divides area ✓"
    )


def check_c5() -> tuple[bool, str]:
    """All four bead numbers {b, e, d, a} are pairwise coprime."""
    count = 0
    for b, e, d, a in _prime_pairs(100, 100):
        assert gcd(b, e) == 1, f"gcd(b,e)={gcd(b,e)} != 1 at b={b},e={e}"
        assert gcd(b, d) == 1, f"gcd(b,d)={gcd(b,d)} != 1 at b={b},e={e}"
        assert gcd(b, a) == 1, f"gcd(b,a)={gcd(b,a)} != 1 at b={b},e={e}"
        assert gcd(e, d) == 1, f"gcd(e,d)={gcd(e,d)} != 1 at b={b},e={e}"
        assert gcd(e, a) == 1, f"gcd(e,a)={gcd(e,a)} != 1 at b={b},e={e}"
        assert gcd(d, a) == 1, f"gcd(d,a)={gcd(d,a)} != 1 at b={b},e={e}"
        count += 1
    # Algebraic proofs:
    # gcd(b,d)=gcd(b,b+e)=gcd(b,e)=1
    # gcd(e,d)=gcd(e,b+e)=gcd(e,b)=1
    # gcd(b,a)=gcd(b,b+2e)=gcd(b,2e)=1 (b odd, gcd(b,e)=1)
    # gcd(e,a)=gcd(e,b+2e)=gcd(e,b)=1
    # gcd(d,a)=gcd(b+e,b+2e)=gcd(b+e,e)=gcd(b,e)=1
    return True, (
        f"All 6 pairwise gcds of {{b,e,d,a}} =1 verified for {count} prime pairs (b,e)<=100; "
        f"proof: each pairwise gcd reduces to gcd(b,e)=1 via Euclidean algorithm; "
        f"gcd(d,a)=gcd(b+e,b+2e)=gcd(b+e,e)=gcd(b,e)=1; "
        f"all four beads are mutually coprime ✓"
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
        raise RuntimeError(f"cert [366] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
