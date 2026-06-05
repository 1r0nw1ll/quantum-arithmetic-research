# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pythagorean formal proof statements: "
    "d-e parity; factor-3 in beads; C=2de divisible by 4; area CF/2 divisible by 6; "
    "G-C=b^2); "
    "Theorem NT: 'par', 'tri', 'pent' classification labels are observer projections of "
    "modular arithmetic; causal structure is integer polynomial algebra on bead numbers; "
    "no float state, no QA orbit evolution"
)

"""
Cert [355] — QA Pythagorean Formal Proof Statements

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, Chapter IX pp.94-100
  Statement 1 (p.95): 'Either d or e must be an even number.'
  Statement 4 (p.96): 'The factor 3 will be represented in every set of bead numbers.'
  Statement 8 (p.98): 'Within every triangle, the base, C, will contain the factor 4.'
  Statement 9 (p.98): 'The area of every prime Pythagorean triangle is divisible by 6.'
  Statement 13 (p.99): 'The difference, B, between the hypotenuse, G, and the base, C,
    of a prime Pythagorean triangle is the square of an odd number [B = G-C = b^2].'

Five claims (from Iverson's 13 numbered proof statements):
  C1: Statement 1 — For all prime Pythagorean triangles, exactly one of (d,e) is even
      (since b is odd: e even ↔ d=b+e odd; e odd ↔ d even)
  C2: Statement 4 — Factor 3 always appears in bead set {b,e,d,a}; at least one is divisible by 3
      (proof via tri-classification: all combinations of tri-residues of b,e yield 3|a or 3|d or 3|b or 3|e)
  C3: Statement 8 — C=2de is always divisible by 4 (4-par)
      (proof: exactly one of (d,e) is even; C=2de thus has two factors of 2)
  C4: Statement 9 — CF/2 (area of prime Pythagorean triangle) is always divisible by 6
      (proof: C divisible by 4, either C or F divisible by 3; CF divisible by 12 → CF/2 divisible by 6)
  C5: Statement 13 — G-C = b^2 for all prime Pythagorean triangles
      (proof: G-C = d^2+e^2-2de = (d-e)^2 = b^2 since b=d-e)
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
    """Exactly one of (d,e) is even for all prime Pythagorean triangles."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        d_even = (d % 2 == 0)
        e_even = (e % 2 == 0)
        # Exactly one of (d,e) is even — never both, never neither
        assert d_even != e_even, f"d,e parity fails: d={d},e={e} b={b} (d_even={d_even},e_even={e_even})"
        count += 1
    # Algebraic proof: b odd. e even → d=b+e=odd+even=odd, so e even and d odd ✓.
    #                  e odd → d=b+e=odd+odd=even, so e odd and d even ✓.
    # In both cases exactly one of (d,e) is even.
    return True, (
        f"Exactly one of (d,e) is even for all {count} valid pairs (b,e)<=25; "
        f"algebraic proof: b odd → parity(d)=parity(e+b)=parity(e)^1 ✓"
    )


def check_c2() -> tuple[bool, str]:
    """Factor 3 appears in every bead set {b,e,d,a}."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        has_3 = (b % 3 == 0 or e % 3 == 0 or d % 3 == 0 or a % 3 == 0)
        assert has_3, f"No bead divisible by 3: b={b},e={e},d={d},a={a}"
        count += 1
    # Proof via tri-classification (mod 3 of b,e → check all 4 combinations for mod 3 of b,e):
    # Case (0,*): b≡0 → 3|b ✓
    # Case (*,0): e≡0 → 3|e ✓
    # Case (1,1): a=b+2e≡1+2=3≡0 → 3|a ✓
    # Case (1,2): d=b+e≡1+2=3≡0 → 3|d ✓
    # Case (2,1): d=b+e≡2+1=3≡0 → 3|d ✓
    # Case (2,2): a=b+2e≡2+4=6≡0 → 3|a ✓
    # All 6 combinations yield a bead divisible by 3.
    for b_mod in range(3):
        for e_mod in range(3):
            b_test = b_mod  # representative residue
            e_test = e_mod
            d_test = (b_test + e_test) % 3
            a_test = (b_test + 2 * e_test) % 3
            has_3_test = (b_test == 0 or e_test == 0 or d_test == 0 or a_test == 0)
            assert has_3_test, f"No 3 in residues b_mod={b_mod},e_mod={e_mod}"
    return True, (
        f"Factor 3 present in {{b,e,d,a}} for all {count} valid pairs (b,e)<=25; "
        f"proof: all 9 combinations of (b mod 3, e mod 3) yield at least one 0 residue"
    )


def check_c3() -> tuple[bool, str]:
    """C=2de is always divisible by 4 (C is always 4-par)."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        C = 2 * d * e
        assert C % 4 == 0, f"C={C} not divisible by 4 for b={b},e={e}"
        count += 1
    # Proof: exactly one of (d,e) is even (C1). Say e=2m: C=2d*2m=4dm ✓.
    # Say d=2k: C=2*2k*e=4ke ✓. Either way 4|C.
    return True, (
        f"C=2de divisible by 4 for all {count} valid pairs (b,e)<=25; "
        f"proof: one of (d,e) is even=2k, so C=2de=4k*(other) divisible by 4"
    )


def check_c4() -> tuple[bool, str]:
    """CF/2 (triangle area) always divisible by 6 for all prime Pythagorean triangles."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        C = 2 * d * e
        F = a * b
        area_2 = C * F  # = 2 * area
        area = C * F // 2  # = CF/2 (area of right triangle with legs C and F)
        assert C * F % 2 == 0, f"CF not even for b={b},e={e}"
        assert area % 6 == 0, f"CF/2={area} not divisible by 6 for b={b},e={e}"
        count += 1
    # Proof:
    # C divisible by 4 (C3), and one of {b,e,d,a} divisible by 3 (C2).
    # If 3|a or 3|b then F=ab contains factor 3. If 3|d or 3|e then C=2de contains factor 3.
    # Either way 3|CF. Combined with 4|C: CF divisible by 12. CF/2 divisible by 6.
    return True, (
        f"CF/2 divisible by 6 for all {count} valid pairs (b,e)<=25; "
        f"proof: 4|C (C3) and 3|{{b,e,d,a}} (C2) → 3|(C or F) → 12|CF → 6|(CF/2)"
    )


def check_c5() -> tuple[bool, str]:
    """G-C = b^2 for all prime Pythagorean triangles (Statement 13)."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        G = d * d + e * e
        C = 2 * d * e
        diff = G - C
        expected = b * b
        assert diff == expected, f"G-C={diff}!= b^2={expected} for b={b},e={e}"
        # Also verify B = G-C is odd (5-par minus 4-par = odd)
        assert diff % 2 == 1, f"G-C={diff} not odd for b={b},e={e}"
        count += 1
    # Algebraic proof: G-C = d^2+e^2-2de = (d-e)^2 = b^2 (since b=d-e).
    # b is odd → b^2 is odd → G-C is always an odd perfect square. ✓
    return True, (
        f"G-C=b^2 verified for all {count} valid pairs (b,e)<=25; "
        f"G-C is always an odd perfect square (b is odd); "
        f"proof: G-C=d^2+e^2-2de=(d-e)^2=b^2 ✓"
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
        raise RuntimeError(f"cert [355] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
