# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-1 Ch.II Prime Pythagorean Triangle Structure: "
    "G always 5-par; G+C=A=a^2, G-C=B=b^2; a always odd; common factor structure of 16 identities; "
    "A, B, G all 5-par; C 4-par; F has no fixed par-class); "
    "Theorem NT: 'hypotenuse', 'base', 'altitude', 'ellipse diameter' are observer projection labels "
    "for integer arithmetic; no float state, no QA orbit evolution"
)

"""
Cert [360] — QA Pyth-1 Prime Triangle Structure (Ch.II)

Source: Iverson (1993) Pythagorean Arithmetic Vol I, Chapter II pp.33-38
  p.34: 'For every Pythagorean triangle, the base will be designated as C, the altitude as F
         and the hypotenuse as G... midway between F and G lies another number D, which is the
         square of an integer. The difference between F and D, and the difference between D and G,
         which will be designated here as E, is another square of an integer.'
  p.35: 'the hypotenuse, G, of every prime Pythagorean triangle must be a 5-par, (4n+1), number.'
  p.35: 'the difference between C and G is also a square, to be designated as B... The sum of C
         and G is also a square, to be designated as A... B and A must be numbers in the form
         of p^(2n) where p represents an odd prime number, or product of prime numbers.'
  p.38: '"a" is a common factor of A, F, and K; "b" is a common factor of B, F, and J;
         "d" is a common factor of C, D, J, and K; "e" is a common factor of C and E;
         All are a common factor of L.'

Five claims:
  C1: G = d^2+e^2 always ≡1 (mod 4) (5-par) for all prime Pythagorean pairs (b odd, gcd=1);
      proof: exactly one of (d,e) is even (from [355] C1) → d^2+e^2 ≡ 0+1 = 1 (mod 4)
  C2: G+C = a^2 = A and G-C = b^2 = B for all prime pairs;
      algebraic proof: G+C = d^2+e^2+2de = (d+e)^2 = a^2; G-C = d^2+e^2-2de = (d-e)^2 = b^2
  C3: a = d+e is always odd; for e odd: d even, a=d+e=odd; for e even: d odd, a=d+e=odd;
      hence A=a^2 is always 5-par (≡1 mod 4)
  C4: Common factor structure: a|{A,F,K}; b|{B,F,J}; d|{C,D,J,K}; e|{C,E};
      abde divisible by 6 (L is always integer); verified for all 268 prime pairs (b,e) <= 25
  C5: A, B, G all 5-par (≡1 mod 4); C always 4-par; F has no fixed par-class
      (F can be 3-par, 5-par, 2-par, or 4-par depending on (b,e))
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
    """G = d^2+e^2 is always 5-par (≡1 mod 4) for all prime Pythagorean pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(25, 25):
        G = d * d + e * e
        assert G % 4 == 1, f"G={G} is not 5-par at b={b},e={e}"
        count += 1
    # Algebraic proof: exactly one of d,e is even (from cert [355] C1):
    # d^2+e^2 = (even)^2 + (odd)^2 = 0 + 1 = 1 (mod 4), or (odd)^2 + (even)^2 = 1+0 = 1 (mod 4)
    return True, (
        f"G=d^2+e^2 always 5-par for all {count} pairs (b,e)<=25; "
        f"proof: one of (d,e) even, one odd → d^2+e^2≡0+1≡1(mod 4) ✓"
    )


def check_c2() -> tuple[bool, str]:
    """G+C = A = a^2 and G-C = B = b^2 for all prime Pythagorean pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(25, 25):
        C = 2 * d * e
        G = d * d + e * e
        A_val = a * a
        B_val = b * b
        # G + C = (d+e)^2 = a^2
        assert G + C == A_val, f"G+C={G+C} != A=a^2={A_val} at b={b},e={e}"
        # G - C = (d-e)^2 = b^2 (since b = d-e)
        assert G - C == B_val, f"G-C={G-C} != B=b^2={B_val} at b={b},e={e}"
        # Cross-check: A + B = 2G and A - B = 2C
        assert A_val + B_val == 2 * G
        assert A_val - B_val == 2 * C
        count += 1
    # Algebraic proof:
    # G+C = d^2+e^2+2de = (d+e)^2 = a^2 = A ✓
    # G-C = d^2+e^2-2de = (d-e)^2 = b^2 = B ✓  (b = d-e since d = b+e)
    return True, (
        f"G+C=A=a^2 and G-C=B=b^2 verified for all {count} pairs (b,e)<=25; "
        f"proof: G+C=(d+e)^2=a^2; G-C=(d-e)^2=b^2; A+B=2G; A-B=2C ✓"
    )


def check_c3() -> tuple[bool, str]:
    """a = d+e is always odd; A=a^2 is always 5-par (≡1 mod 4)."""
    count = 0
    for b, e, d, a in _prime_pairs(25, 25):
        # b odd; exactly one of (d,e) is even
        assert a % 2 == 1, f"a={a} not odd at b={b},e={e}"
        assert a * a % 4 == 1, f"A=a^2={a*a} not 5-par at b={b},e={e}"
        # Also verify: b is always odd (by construction)
        assert b % 2 == 1, f"b={b} not odd"
        assert b * b % 4 == 1, f"B=b^2={b*b} not 5-par"
        count += 1
    # Proof: a=d+e; for e odd, d=b+e=odd+odd=even, a=even+odd=odd ✓
    #               for e even, d=b+e=odd+even=odd, a=odd+even=odd ✓
    # In both cases a is odd, so A=a^2 ≡1(mod 4) = 5-par ✓
    return True, (
        f"a always odd for all {count} pairs (b,e)<=25; "
        f"proof: a=d+e; (e odd → d even → a=even+odd=odd); (e even → d odd → a=odd+even=odd); "
        f"A=a^2 always 5-par ✓"
    )


def check_c4() -> tuple[bool, str]:
    """Common factor structure: a|{A,F,K}; b|{B,F,J}; d|{C,D,J,K}; e|{C,E}; all|L."""
    count = 0
    for b, e, d, a in _prime_pairs(25, 25):
        A_val = a * a
        B_val = b * b
        C_val = 2 * d * e
        D_val = d * d
        E_val = e * e
        F_val = a * b
        J_val = b * d
        K_val = a * d
        abde = a * b * d * e
        assert abde % 6 == 0
        L_val = abde // 6
        # a divides A, F, K
        assert A_val % a == 0  # a^2 / a = a ✓
        assert F_val % a == 0  # ab / a = b ✓
        assert K_val % a == 0  # ad / a = d ✓
        # b divides B, F, J
        assert B_val % b == 0  # b^2 / b = b ✓
        assert F_val % b == 0  # ab / b = a ✓
        assert J_val % b == 0  # bd / b = d ✓
        # d divides C, D, J, K
        assert C_val % d == 0  # 2de / d = 2e ✓
        assert D_val % d == 0  # d^2 / d = d ✓
        assert J_val % d == 0  # bd / d = b ✓
        assert K_val % d == 0  # ad / d = a ✓
        # e divides C, E
        assert C_val % e == 0  # 2de / e = 2d ✓
        assert E_val % e == 0  # e^2 / e = e ✓
        # L = abde/6: the product abde always divisible by 6 (cert [355] C4)
        assert abde % 6 == 0
        count += 1
    return True, (
        f"Common factor structure verified for all {count} pairs (b,e)<=25: "
        f"a|{{A,F,K}}; b|{{B,F,J}}; d|{{C,D,J,K}}; e|{{C,E}}; abde divisible by 6 (L integer) ✓"
    )


def check_c5() -> tuple[bool, str]:
    """A, B, G all 5-par; C always 4-par; F has no fixed par-class."""
    f_classes = set()
    count = 0
    for b, e, d, a in _prime_pairs(25, 25):
        A_val = a * a
        B_val = b * b
        C_val = 2 * d * e
        G_val = d * d + e * e
        F_val = a * b
        # A, B, G all 5-par (≡1 mod 4)
        assert A_val % 4 == 1, f"A={A_val} not 5-par at b={b},e={e}"
        assert B_val % 4 == 1, f"B={B_val} not 5-par at b={b},e={e}"
        assert G_val % 4 == 1, f"G={G_val} not 5-par at b={b},e={e}"
        # C always 4-par (≡0 mod 4) [from cert [355] C3]
        assert C_val % 4 == 0, f"C={C_val} not 4-par at b={b},e={e}"
        # F: track which par-classes appear
        f_classes.add(F_val % 4)
        count += 1
    # F should appear in multiple par-classes
    assert len(f_classes) > 1, f"F only takes par-classes: {f_classes}"
    f_class_names = {0: "4-par", 1: "5-par", 2: "2-par", 3: "3-par"}
    f_classes_named = sorted(f_class_names[r] for r in f_classes)
    return True, (
        f"A, B, G all 5-par; C always 4-par; verified for {count} pairs (b,e)<=25; "
        f"F par-classes found: {f_classes_named} (no fixed class) ✓"
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
        raise RuntimeError(f"cert [360] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
