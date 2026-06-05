# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-2 Ch.XIII Synchronous Harmonics LCM: "
    "LCM of coprime cycles = product; LCM(2,3,5)=30; LCM(3,5,7)=105; "
    "non-coprime cycle adds nothing; QA sine=F/G; half-cycle symmetry period/2); "
    "Theorem NT: 'sine wave', 'amplitude', 'frequency' are observer projections; "
    "QA layer = LCM arithmetic over bead integers; no float state"
)

"""
Cert [368] — QA Pyth-2 Synchronous Harmonics: LCM and Coincidence Periods (Ch.XIII)

Source: Iverson (1993) Pythagorean Arithmetic Vol II, Chapter XIII pp.38-70+
  p.48 (Fig.14a): 'The cycle of 3 and 5 will coincide every 15 units.'
  p.48: 'The cycle of 2 and 3 will coincide every 6 units. 2 and 5 will coincide
         every 10 units. 3 and 5 will coincide every 15 units. And the cycles of
         2, 3, and 5 will coincide every 30 units.'
  p.49: 'The complete cycle for 3, 5, and 7 would be 105 units before all
         will coincide.'
  p.48: 'The addition of a 6-cycle to the 2-cycle and 3-cycle does not change
         the overall cycle because 6 is not prime to 2 or 3.'
  p.50: 'the sine is represented as F/G'

Five claims:
  C1: LCM of pairwise-coprime integers = their product (Synchronous Harmonics coincidence)
      Verification: lcm(2,3)=6=2*3; lcm(2,5)=10=2*5; lcm(3,5)=15=3*5;
                   lcm(2,3,5)=30=2*3*5; lcm(3,5,7)=105=3*5*7;
                   lcm(2,3,5,7)=210=2*3*5*7.
      proof: if gcd(a,b)=1 then lcm(a,b)=ab/gcd(a,b)=ab ✓
  C2: Non-coprime extension lemma: if c|lcm(a,b) then lcm(a,b,c)=lcm(a,b)
      Specific case: lcm(2,3,6)=6=lcm(2,3); adding the 6-cycle doesn't extend the period.
      proof: c|m=lcm(a,b) → c is already a divisor of m → lcm(m,c)=m ✓
  C3: Pairwise LCM formula: lcm(a,b,c) = lcm(lcm(a,b),c) for any positive integers
      And for pairwise coprime {a,b,c}: lcm(a,b,c)=abc ✓
      Verified for all triples (a,b,c) with a,b,c in [2,20], pairwise coprime.
  C4: QA sine = F/G for all prime Pythagorean pairs (b, e)
      F = ab (altitude); G = d^2+e^2 (hypotenuse); sine = F/G as a rational fraction
      F < G always (since F^2 + C^2 = G^2 and C > 0 → F < G)
      First pair (b=1,e=1): sine=F/G=3/5; (b=1,e=2): F=5,G=13, sine=5/13
      proof: F^2 + C^2 = G^2 (Pythagorean theorem); F/G ∈ (0,1) ✓
  C5: Half-cycle symmetry: the midpoint of the combined period = lcm/2 is an integer
      iff the LCM is even (which holds when any cycle length is even).
      If all cycle lengths are odd, midpoint = lcm/2 is a half-integer (as Iverson notes
      for {3,5,7}: midpoint = 105/2 = 52.5).
      proof: lcm of only-odd numbers is odd → midpoint = odd/2 is not an integer.
             lcm with at least one even number is even → midpoint is an integer. ✓
"""

from math import gcd, lcm
from fractions import Fraction


def _lcm(*args: int) -> int:
    result = args[0]
    for x in args[1:]:
        result = lcm(result, x)
    return result


def _prime_pairs(max_b: int, max_e: int):
    for b in range(1, max_b + 1, 2):
        for e in range(1, max_e + 1):
            if gcd(b, e) == 1:
                d = b + e
                a = d + e
                yield b, e, d, a


def check_c1() -> tuple[bool, str]:
    """LCM of pairwise-coprime integers = product; Iverson's specific examples verified."""
    # Iverson's explicit claims from the text
    examples = [
        (2, 3,    6,    "lcm(2,3)=6"),
        (2, 5,    10,   "lcm(2,5)=10"),
        (3, 5,    15,   "lcm(3,5)=15"),
        (2, 3, 5, 30,   "lcm(2,3,5)=30"),
        (3, 5, 7, 105,  "lcm(3,5,7)=105"),
        (2, 3, 5, 7, 210, "lcm(2,3,5,7)=210"),
    ]
    for args in examples:
        expected = args[-2]
        desc = args[-1]
        nums = args[:-2]
        computed = _lcm(*nums)
        assert computed == expected, f"{desc}: computed {computed} != {expected}"
        # Verify pairwise coprime → lcm = product
        if all(gcd(nums[i], nums[j]) == 1 for i in range(len(nums)) for j in range(i+1, len(nums))):
            product = 1
            for n in nums:
                product *= n
            assert computed == product, f"lcm≠product for coprime set {nums}"

    # General verification: for coprime pairs (a,b): lcm(a,b)=a*b
    count = 0
    for a in range(2, 50):
        for b in range(a+1, 50):
            if gcd(a, b) == 1:
                assert lcm(a, b) == a * b, f"lcm({a},{b})={lcm(a,b)} != {a*b}"
                count += 1
    return True, (
        f"LCM=product for pairwise coprime sets; "
        f"Iverson examples: lcm(2,3)=6; lcm(2,5)=10; lcm(3,5)=15; "
        f"lcm(2,3,5)=30; lcm(3,5,7)=105; lcm(2,3,5,7)=210; "
        f"verified for {count} coprime pairs (a,b)<50; "
        f"proof: gcd(a,b)=1 → lcm(a,b)=ab/gcd(a,b)=ab ✓"
    )


def check_c2() -> tuple[bool, str]:
    """Non-coprime extension: lcm(2,3,6)=lcm(2,3)=6; adding 6-cycle doesn't extend period."""
    # Iverson's claim: 'addition of a 6-cycle... does not change the overall cycle'
    assert lcm(2, 3) == 6
    assert lcm(2, 3, 6) == 6
    assert lcm(2, 3, 6) == lcm(2, 3), f"lcm(2,3,6)={lcm(2,3,6)} != lcm(2,3)={lcm(2,3)}"
    # General rule: if c | lcm(a,b) then lcm(a,b,c) = lcm(a,b)
    count = 0
    for a in range(2, 30):
        for b in range(2, 30):
            m = lcm(a, b)
            for c in range(2, 30):
                if m % c == 0:  # c divides lcm(a,b)
                    assert lcm(a, b, c) == m, (
                        f"lcm({a},{b},{c})={lcm(a,b,c)} != lcm({a},{b})={m}"
                    )
                    count += 1
    return True, (
        f"Non-coprime extension lemma verified: "
        f"lcm(2,3,6)={lcm(2,3,6)}=lcm(2,3)={lcm(2,3)} ✓ (6=2×3 not coprime to 2 or 3); "
        f"general rule c|lcm(a,b) → lcm(a,b,c)=lcm(a,b) verified for {count} triples in [2,29]; "
        f"proof: if c|m then lcm(m,c)=m ✓"
    )


def check_c3() -> tuple[bool, str]:
    """LCM is associative: lcm(a,b,c)=lcm(lcm(a,b),c); pairwise coprime → lcm=product."""
    count_assoc = 0
    count_coprime = 0
    for a in range(2, 20):
        for b in range(2, 20):
            for c in range(2, 20):
                computed = lcm(lcm(a, b), c)
                direct = _lcm(a, b, c)
                assert computed == direct, f"lcm associativity fails at ({a},{b},{c})"
                count_assoc += 1
                # If pairwise coprime, lcm = product
                if gcd(a, b) == 1 and gcd(b, c) == 1 and gcd(a, c) == 1:
                    assert direct == a * b * c, (
                        f"lcm({a},{b},{c})={direct} != {a*b*c}=product for coprime triple"
                    )
                    count_coprime += 1
    return True, (
        f"LCM associativity lcm(a,b,c)=lcm(lcm(a,b),c) verified for {count_assoc} triples [2,19]; "
        f"pairwise-coprime triples (lcm=product) verified for {count_coprime} cases; "
        f"key examples: lcm(2,3,5)=30=2×3×5; lcm(3,5,7)=105=3×5×7 ✓"
    )


def check_c4() -> tuple[bool, str]:
    """QA sine = F/G for all prime pairs; F < G always; first pair sine=3/5."""
    count = 0
    for b, e, d, a in _prime_pairs(30, 30):
        F = a * b        # altitude
        G = d * d + e * e  # hypotenuse
        C = 2 * d * e   # base
        # Pythagorean verification: F^2 + C^2 = G^2 (integer identity, no float)
        assert F * F + C * C == G * G, (
            f"F^2+C^2={F*F+C*C} != G^2={G*G} at b={b},e={e}"
        )
        # F < G always (since C > 0 and F^2 < F^2+C^2 = G^2)
        assert F < G, f"F={F} >= G={G} at b={b},e={e}"
        # Sine = F/G (exact rational in (0,1))
        sine = Fraction(F, G)
        assert 0 < sine < 1, f"sine={sine} not in (0,1) at b={b},e={e}"
        count += 1
    # First pair: (b=1,e=1): d=2,a=3; F=3,G=5; sine=3/5
    b, e, d, a = 1, 1, 2, 3
    F = a * b; G = d * d + e * e
    assert Fraction(F, G) == Fraction(3, 5)
    # Second pair: (b=1,e=2): d=3,a=5; F=5,G=13; sine=5/13
    b, e, d, a = 1, 2, 3, 5
    F = a * b; G = d * d + e * e
    assert Fraction(F, G) == Fraction(5, 13)
    return True, (
        f"QA sine=F/G verified for {count} prime pairs (b,e)<=30; "
        f"F<G always (F^2<F^2+C^2=G^2); F^2+C^2=G^2 integer Pythagorean identity verified; "
        f"first pair (b=1,e=1): sine=3/5; second (b=1,e=2): sine=5/13; "
        f"all sines in (0,1) as exact Fractions ✓"
    )


def check_c5() -> tuple[bool, str]:
    """Half-cycle symmetry: lcm is even iff at least one cycle length is even → integer midpoint."""
    # All-odd cycles: lcm is odd → midpoint is not integer
    assert _lcm(3, 5, 7) == 105
    assert 105 % 2 == 1, "lcm(3,5,7)=105 is odd"
    # Iverson's note: midpoint of {3,5,7} is 52.5 = 105/2 (half-integer)

    # With an even cycle: lcm is even → integer midpoint
    assert _lcm(2, 3, 5) == 30
    assert 30 % 2 == 0, "lcm(2,3,5)=30 is even"
    assert 30 // 2 == 15  # integer midpoint

    # General rule: lcm of all-odd integers is odd
    count_odd = 0
    count_even = 0
    for a in range(3, 30, 2):
        for b in range(3, 30, 2):
            if gcd(a, b) == 1:
                m = lcm(a, b)
                assert m % 2 == 1, f"lcm of odd coprime {a},{b} = {m} is not odd!"
                count_odd += 1

    # lcm with any even number is even
    for a in range(2, 20, 2):
        for b in range(3, 20):
            m = lcm(a, b)
            assert m % 2 == 0, f"lcm({a},{b})={m} is not even but {a} is even"
            count_even += 1

    return True, (
        f"Half-cycle symmetry: "
        f"lcm(3,5,7)=105 is odd → midpoint=52.5 (not integer, as Iverson notes); "
        f"lcm(2,3,5)=30 is even → midpoint=15 (integer); "
        f"all-odd coprime LCMs verified odd for {count_odd} pairs; "
        f"even-containing LCMs verified even for {count_even} pairs; "
        f"proof: product of odd integers is odd; one even factor makes LCM even ✓"
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
        raise RuntimeError(f"cert [368] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
