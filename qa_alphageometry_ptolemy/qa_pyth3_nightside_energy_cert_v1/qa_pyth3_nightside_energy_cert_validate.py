# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.3 Nightside Energy: "
    "4-way partition mod 4 (4-par/2-par/3-par/5-par); product and sum rules for "
    "3-par and 5-par classes; nightside correlation b≡2(mod 4) ↔ a≡0(mod 4) for "
    "female tuples; male vs female C parity: 4-par vs 2-par); "
    "Theorem NT: 'nightside energy', 'dayside', 'levitation', 'cooling' are observer "
    "projection labels for integer parity arithmetic; no float state, no QA orbit evolution"
)

"""
Cert [359] — QA Pyth-3 Nightside Energy: 4-Way Integer Partition

Source: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III, Chapter 3 pp.10-12
  p.11: 'These are the even-even, even-odd, odd-even, and odd-odd integers... The even-even
         integers are, of course the 4-n integers. The even-odd integers are even integers,
         which are not evenly divisible by four. The two types of odd integers have, now, been
         named 3-par and 5-par...'
  p.11: 'The product of two 3-par numbers is a 5-par number. The product of two 5-par integers
         is also a 5-par integer. The sum of two 3-par, or two 5-par integers is a 2-par integer.
         The sum of a 3-par and a 5-par integer is a 4-par integer, and the product of a 3-par
         number and a 5-par number is a 3-par number.'
  p.10: 'If one obtains the true quantum number of any Nightside frequency, this quantum number
         will begin with 4 or 2.' (b is even → female triangle)
  p.10: Nightside has b,a both even with b=2n (2-par) and a=4n (4-par);
         Dayside has b,a both odd.

Five claims:
  C1: 4-way partition: n mod 4 = {0→4-par, 2→2-par, 3→3-par, 1→5-par};
      6 of each in Cosmos orbit {1..24}; 12 of each in {1..48}
  C2: Product rules: 3-par*3-par=5-par; 5-par*5-par=5-par; 3-par*5-par=3-par;
      algebraic proofs via mod-4 arithmetic
  C3: Sum rules: 3-par+3-par=2-par; 5-par+5-par=2-par; 3-par+5-par=4-par;
      algebraic proofs via mod-4 arithmetic
  C4: Nightside correlation for female tuples (b even, e odd, gcd=1):
      b≡2(mod 4) iff a≡0(mod 4); b≡0(mod 4) iff a≡2(mod 4);
      proof: a=b+2e, e odd → 2e≡2(mod 4); a≡b+2(mod 4)
  C5: Male (b odd): C=2de always 4-par (not 2-par); Female (b even): C=2de always 2-par (not 4-par);
      parity of C distinguishes male from female uniquely
"""

from math import gcd


def _par_class(n: int) -> str:
    """Return the Iverson par-class of n."""
    r = n % 4
    if r == 0:
        return "4-par"
    elif r == 2:
        return "2-par"
    elif r == 3:
        return "3-par"
    else:  # r == 1
        return "5-par"


def check_c1() -> tuple[bool, str]:
    """4-way partition: 6 of each class in {1..24}; 12 of each in {1..48}."""
    # Every integer belongs to exactly one class
    classes_24 = {c: 0 for c in ["4-par", "2-par", "3-par", "5-par"]}
    for n in range(1, 25):
        classes_24[_par_class(n)] += 1
    assert all(v == 6 for v in classes_24.values()), f"Unequal counts in 1..24: {classes_24}"
    classes_48 = {c: 0 for c in ["4-par", "2-par", "3-par", "5-par"]}
    for n in range(1, 49):
        classes_48[_par_class(n)] += 1
    assert all(v == 12 for v in classes_48.values()), f"Unequal counts in 1..48: {classes_48}"
    # Spot-check the classification
    assert _par_class(4) == "4-par"
    assert _par_class(8) == "4-par"
    assert _par_class(2) == "2-par"
    assert _par_class(6) == "2-par"
    assert _par_class(3) == "3-par"
    assert _par_class(7) == "3-par"
    assert _par_class(1) == "5-par"
    assert _par_class(5) == "5-par"
    return True, (
        f"4-way partition: 4-par(r=0), 2-par(r=2), 3-par(r=3), 5-par(r=1); "
        f"exactly 6 of each in {{1..24}} and 12 of each in {{1..48}} ✓"
    )


def check_c2() -> tuple[bool, str]:
    """Product rules: 3-par*3-par=5-par; 5-par*5-par=5-par; 3-par*5-par=3-par."""
    # Algebraic proof:
    # (4j+3)(4k+3) = 16jk+12j+12k+9 = 4(4jk+3j+3k+2)+1 ≡ 1 (mod 4) → 5-par ✓
    # (4j+1)(4k+1) = 16jk+4j+4k+1 = 4(4jk+j+k)+1 ≡ 1 (mod 4) → 5-par ✓
    # (4j+3)(4k+1) = 16jk+4j+12k+3 = 4(4jk+j+3k)+3 ≡ 3 (mod 4) → 3-par ✓
    count = 0
    for a_val in range(1, 50):
        for b_val in range(1, 50):
            p = a_val * b_val
            pc_a, pc_b, pc_p = _par_class(a_val), _par_class(b_val), _par_class(p)
            if pc_a == "3-par" and pc_b == "3-par":
                assert pc_p == "5-par", f"3-par*3-par: {a_val}*{b_val}={p} is {pc_p}"
                count += 1
            elif pc_a == "5-par" and pc_b == "5-par":
                assert pc_p == "5-par", f"5-par*5-par: {a_val}*{b_val}={p} is {pc_p}"
                count += 1
            elif (pc_a == "3-par" and pc_b == "5-par") or (pc_a == "5-par" and pc_b == "3-par"):
                assert pc_p == "3-par", f"3-par*5-par: {a_val}*{b_val}={p} is {pc_p}"
                count += 1
    return True, (
        f"Product rules verified for all odd pairs in {{1..49}}×{{1..49}} ({count} cases): "
        f"3*3=5-par (proof: (4j+3)(4k+3)≡1 mod 4); "
        f"5*5=5-par (proof: (4j+1)(4k+1)≡1 mod 4); "
        f"3*5=3-par (proof: (4j+3)(4k+1)≡3 mod 4) ✓"
    )


def check_c3() -> tuple[bool, str]:
    """Sum rules: 3-par+3-par=2-par; 5-par+5-par=2-par; 3-par+5-par=4-par."""
    # Algebraic proofs:
    # (4j+3)+(4k+3) = 4(j+k+1)+2 ≡ 2 (mod 4) → 2-par ✓
    # (4j+1)+(4k+1) = 4(j+k)+2 ≡ 2 (mod 4) → 2-par ✓
    # (4j+3)+(4k+1) = 4(j+k+1)+0 ≡ 0 (mod 4) → 4-par ✓
    count = 0
    for a_val in range(1, 50):
        for b_val in range(1, 50):
            s = a_val + b_val
            pc_a, pc_b, pc_s = _par_class(a_val), _par_class(b_val), _par_class(s)
            if pc_a == "3-par" and pc_b == "3-par":
                assert pc_s == "2-par", f"3+3: {a_val}+{b_val}={s} is {pc_s}"
                count += 1
            elif pc_a == "5-par" and pc_b == "5-par":
                assert pc_s == "2-par", f"5+5: {a_val}+{b_val}={s} is {pc_s}"
                count += 1
            elif (pc_a == "3-par" and pc_b == "5-par") or (pc_a == "5-par" and pc_b == "3-par"):
                assert pc_s == "4-par", f"3+5: {a_val}+{b_val}={s} is {pc_s}"
                count += 1
    return True, (
        f"Sum rules verified for all pairs in {{1..49}}×{{1..49}} ({count} cases): "
        f"3+3=2-par (proof: (4j+3)+(4k+3)≡2 mod 4); "
        f"5+5=2-par (proof: (4j+1)+(4k+1)≡2 mod 4); "
        f"3+5=4-par (proof: (4j+3)+(4k+1)≡0 mod 4) ✓"
    )


def check_c4() -> tuple[bool, str]:
    """Nightside correlation: for female (b even, e odd, gcd=1): b≡2(mod 4) ↔ a≡0(mod 4)."""
    # Proof: a = b + 2e; e odd so 2e ≡ 2 (mod 4);
    # If b≡2(mod 4): a ≡ 2+2 = 4 ≡ 0 (mod 4) → a is 4-par ✓
    # If b≡0(mod 4): a ≡ 0+2 = 2 (mod 4) → a is 2-par ✓
    b2_a4 = 0  # b≡2 and a≡0
    b4_a2 = 0  # b≡0 and a≡2
    for b in range(2, 25, 2):  # b even
        for e in range(1, 25, 2):  # e odd
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = d + e
            b_r = b % 4
            a_r = a % 4
            if b_r == 2:
                assert a_r == 0, f"b={b}≡2(mod 4) but a={a}≡{a_r}(mod 4)"
                b2_a4 += 1
            elif b_r == 0:
                assert a_r == 2, f"b={b}≡0(mod 4) but a={a}≡{a_r}(mod 4)"
                b4_a2 += 1
    # Algebraic proof: a = b+2e, e odd → 2e≡2(mod 4), so a≡b+2(mod 4)
    # b≡2 → a≡4≡0; b≡0 → a≡2; bijection ✓
    return True, (
        f"Nightside correlation verified for {b2_a4} pairs with b≡2(mod 4) and {b4_a2} with b≡0(mod 4) "
        f"(female, b,e≤24); b≡2(mod 4) ↔ a≡0(mod 4) always; b≡0(mod 4) ↔ a≡2(mod 4) always; "
        f"proof: a=b+2e, e odd → 2e≡2(mod 4) → a≡b+2(mod 4) ✓"
    )


def check_c5() -> tuple[bool, str]:
    """Male: C=2de always 4-par; Female: C=2de always 2-par; parity of C distinguishes them."""
    count_male_4par = 0
    count_female_2par = 0
    # Male: b odd, gcd(b,e)=1
    for b in range(1, 20, 2):
        for e in range(1, 20):
            if gcd(b, e) != 1:
                continue
            d = b + e
            C_val = 2 * d * e
            assert C_val % 4 == 0, f"Male: C={C_val} not 4-par at b={b},e={e}"
            count_male_4par += 1
    # Female: b even, e odd, gcd(b,e)=1
    for b in range(2, 20, 2):
        for e in range(1, 20, 2):
            if gcd(b, e) != 1:
                continue
            d = b + e
            C_val = 2 * d * e
            assert C_val % 4 == 2, f"Female: C={C_val} not 2-par at b={b},e={e}"
            count_female_2par += 1
    # Boundary: no female C is 4-par, no male C is 2-par → partition is perfect
    return True, (
        f"Male ({count_male_4par} pairs): C=2de always 4-par ✓; "
        f"Female ({count_female_2par} pairs): C=2de always 2-par ✓; "
        f"parity(C) is a complete male/female discriminant: "
        f"C≡0(mod 4) ↔ male; C≡2(mod 4) ↔ female (nightside)"
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
        raise RuntimeError(f"cert [359] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
