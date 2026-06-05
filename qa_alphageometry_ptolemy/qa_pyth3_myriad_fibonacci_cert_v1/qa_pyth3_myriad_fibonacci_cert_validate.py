# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.13-14 Myriad structure and Fibonacci strings: "
    "8 creative QN tuples with all bead values <=7; 5041=71^2 boundary; secondary creative groups "
    "k*(1,1,2,3)=(k,k,2k,3k); male/female Fibonacci recurrence strings and prime-power termination); "
    "Theorem NT: 'Myriad', 'creative', 'prime-power', 'nonprime square' are observer classification labels; "
    "causal structure is integer arithmetic on bead tuples (b,e,d,a); no float state, no QA orbit evolution"
)

"""
Cert [354] — QA Pyth-3 Myriad Structure and Fibonacci Strings

Source: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III, Chapters 13-14 pp.74-89
  Ch.13 p.75: "These Quantum Numbers are very limited, being: 1,1,2,3; 1,2,3,5; 1,3,4,7;
    2,1,3,4; [2,3,5,7]; 3,1,4,5; 3,2,5,7; 4,1,5,6; and 5,1,6,7. It is as though these are
    'creative' Quantum numbers."
  Ch.13 p.75: "The value 5040 is derived as the product when the seven prime factors are
    multiplied together. Above that, the value 5041 is 71 squared."
  Ch.14 p.80: "squares which run diagonally... being 2,2,4,6; 3,3,6,9; 4,4,8,12 etc...
    These are the children of 1,1,2,3."
  Ch.14 p.87: "The string in the male gender run 1,1,2,3,5,8,13,21 [canonical: OCR '1,2,2,3'
    is a formatting artifact; male seed is (1,1) per aboriginal bead b=e=1]. The string of female
    gender Fibonacci numbers are 2,1,3,4,7,11,18. The last number in each is the first which
    is not prime or are the power of a single prime."

Five claims:
  C1: Exactly 8 valid (b,e,d,a) tuples exist with gcd(b,e)=1, d=b+e, a=d+e, all values in {1..7}
  C2: 5041 = 71^2; 71 is prime; 5040 = 7! = 1*2*3*4*5*6*7; 5040+1 = 71^2
  C3: Secondary creative groups: (k,k,2k,3k) satisfies d=b+e, a=d+e for all k>=1;
      gcd(k,k)=k (nonprime for k>1); 13 such groups from k=1..13 bracket the Myriad
  C4: Male Fibonacci string 1,2,2,3,5,8,13,21: recurrence F(n+1)=F(n)+F(n-1);
      21=3*7 is the first term that is neither prime nor a prime power
  C5: Female Fibonacci string 2,1,3,4,7,11,18: recurrence F(n+1)=F(n)+F(n-1);
      18=2*3^2 is the first term that is neither prime nor a prime power
"""

from math import gcd


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _is_prime_power(n: int) -> bool:
    """Return True if n is a prime power (p^k for some prime p and k>=1)."""
    if n < 2:
        return False
    if _is_prime(n):
        return True
    for p in range(2, int(n**0.5) + 1):
        if n % p == 0:
            while n % p == 0:
                n //= p
            return n == 1
    return False


def check_c1() -> tuple[bool, str]:
    """Exactly 8 valid creative QN tuples with all bead values in {1..7}."""
    tuples = []
    for b in range(1, 8):
        for e in range(1, 8):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = d + e
            if d <= 7 and a <= 7:
                tuples.append((b, e, d, a))

    # Expected: exactly 8 tuples
    expected = [
        (1, 1, 2, 3),
        (1, 2, 3, 5),
        (1, 3, 4, 7),
        (2, 1, 3, 4),
        (3, 1, 4, 5),
        (3, 2, 5, 7),
        (4, 1, 5, 6),
        (5, 1, 6, 7),
    ]
    assert len(tuples) == 8, f"Expected 8 tuples, found {len(tuples)}: {tuples}"
    assert sorted(tuples) == sorted(expected), f"Tuple mismatch: {tuples} != {expected}"

    # All 8 satisfy d=b+e, a=d+e, gcd(b,e)=1
    for b, e, d, a in tuples:
        assert d == b + e, f"d!=b+e for ({b},{e},{d},{a})"
        assert a == d + e, f"a!=d+e for ({b},{e},{d},{a})"
        assert gcd(b, e) == 1, f"gcd(b,e)!=1 for ({b},{e},{d},{a})"

    return True, (
        f"Exactly 8 valid creative QN tuples with all values in {{1..7}}: {expected}; "
        f"all satisfy d=b+e, a=d+e, gcd(b,e)=1"
    )


def check_c2() -> tuple[bool, str]:
    """5041 = 71^2; 71 is prime; 5040 = 7!; 5040+1 = 71^2."""
    # 5040 = 7!
    factorial_7 = 1 * 2 * 3 * 4 * 5 * 6 * 7
    assert factorial_7 == 5040, f"7! != 5040: got {factorial_7}"
    # 5041 = 71^2
    assert 71 * 71 == 5041, f"71^2 != 5041: got {71*71}"
    assert 5040 + 1 == 5041
    # 71 is prime
    assert _is_prime(71), "71 is not prime"
    # 73 is the next prime after 71
    assert not _is_prime(72)
    assert _is_prime(73)
    # 5040 = 2^4 * 3^2 * 5 * 7 (factorization)
    assert 2**4 * 3**2 * 5 * 7 == 5040, f"2^4*3^2*5*7 != 5040: {2**4*3**2*5*7}"
    return True, (
        "5040=7!=1*2*3*4*5*6*7; 5041=71^2; 71 is prime; 5040+1=71^2; "
        "factorization: 5040=2^4*3^2*5*7; next prime after 71 is 73"
    )


def check_c3() -> tuple[bool, str]:
    """Secondary creative groups (k,k,2k,3k): satisfies d=b+e, a=d+e; nonprime for k>1."""
    # k=1..13 bracket the Myriad (from (1,1,2,3) to (13,13,26,39))
    for k in range(1, 14):
        b, e, d, a = k, k, 2 * k, 3 * k
        assert d == b + e, f"d!=b+e for k={k}: {b},{e},{d}"
        assert a == d + e, f"a!=d+e for k={k}: {d},{e},{a}"
        # gcd(k,k) = k — nonprime for k>1
        assert gcd(b, e) == k, f"gcd({b},{e}) != {k}"
        if k > 1:
            assert gcd(b, e) > 1, f"Expected nonprime for k={k}"

    # The aboriginal (1,1,2,3) is the unique k=1 case with gcd=1
    assert gcd(1, 1) == 1

    # Verify casting-out-nines: scalar of 'a' value = a mod 9 (positive remainder)
    # For 1,1,2,3: a=3, 3 mod 9 = 3
    # For 2,2,4,6: a=6, 6 mod 9 = 6
    # For 3,3,6,9: a=9, 9 mod 9 = 0 → use 9 (the text says "positive remainder when 9 subtracted")
    # For 4,4,8,12: a=12, 12-9=3, scalar=3
    # These match the enneagram scalar pattern 3,6,9,3,6,9,...
    scalars = []
    for k in range(1, 14):
        a_val = 3 * k  # observer label only — a_val is output of (k,k,2k,3k) structure
        remainder = a_val % 9
        scalars.append(9 if remainder == 0 else remainder)

    # Pattern should cycle through 3,6,9 repeating
    expected_scalars = [3, 6, 9] * 4 + [3]  # 13 values
    assert scalars == expected_scalars, f"Scalar pattern mismatch: {scalars}"

    return True, (
        f"Secondary creative groups (k,k,2k,3k) for k=1..13 all satisfy d=b+e, a=d+e; "
        f"gcd(k,k)=k (prime only for k=1); scalar=a mod 9 cycles 3,6,9 repeating ✓"
    )


def check_c4() -> tuple[bool, str]:
    """Male Fibonacci string 1,1,2,3,5,8,13,21: recurrence; 21 is first non-prime-power.
    Source text says '1,2,2,3,...' — OCR artifact; canonical male seed is (1,1) matching
    aboriginal bead (b=1,e=1,d=2,a=3); the string extends as standard Fibonacci from 1,1."""
    male = [1, 1, 2, 3, 5, 8, 13, 21]
    # Verify recurrence: each term = sum of previous two
    for i in range(2, len(male)):
        assert male[i] == male[i-1] + male[i-2], (
            f"Male recurrence fails at index {i}: {male[i]} != {male[i-1]}+{male[i-2]}"
        )
    # Seed: (1, 1) — the aboriginal male bead b=e=1
    assert male[0] == 1 and male[1] == 1

    # Find first term >= 2 that is neither prime nor prime power
    # (1 is a special unit, excluded from prime-power classification per Iverson)
    first_non_pp = None
    for n in male:
        if n < 2:
            continue
        if not _is_prime_power(n):
            first_non_pp = n
            break
    assert first_non_pp == 21, f"First non-prime-power in male string should be 21, got {first_non_pp}"
    assert not _is_prime(21)  # 21 = 3 * 7
    assert not _is_prime_power(21)
    assert 21 == 3 * 7

    # Terms >= 2 preceding 21: 2(prime), 3(prime), 5(prime), 8=2^3(prime power), 13(prime)
    for n in [2, 3, 5, 8, 13]:
        assert _is_prime_power(n), f"{n} should be prime or prime power in male string"

    return True, (
        f"Male string {male}: satisfies Fn+1=Fn+Fn-1 from seed (1,1) (aboriginal male bead b=e=1); "
        f"21=3*7 is first term that is neither prime nor prime power; "
        f"all preceding terms from index 1 onward are prime or prime powers: {{1,2,3,5,8=2^3,13}}"
    )


def check_c5() -> tuple[bool, str]:
    """Female Fibonacci string 2,1,3,4,7,11,18: recurrence; 18 is first non-prime-power."""
    female = [2, 1, 3, 4, 7, 11, 18]
    # Verify recurrence: each term = sum of previous two
    for i in range(2, len(female)):
        assert female[i] == female[i-1] + female[i-2], (
            f"Female recurrence fails at index {i}: {female[i]} != {female[i-1]}+{female[i-2]}"
        )
    # Seed: (2, 1)
    assert female[0] == 2 and female[1] == 1

    # Find first term >= 2 that is neither prime nor prime power
    first_non_pp = None
    for n in female:
        if n < 2:
            continue
        if not _is_prime_power(n):
            first_non_pp = n
            break
    assert first_non_pp == 18, f"First non-prime-power in female string should be 18, got {first_non_pp}"
    assert not _is_prime(18)
    assert not _is_prime_power(18)  # 18 = 2 * 3^2
    assert 18 == 2 * 3 * 3

    # Preceding terms: 2(prime), 1(special), 3(prime), 4=2^2(prime power), 7(prime), 11(prime)
    for n in [2, 3, 4, 7, 11]:
        assert _is_prime_power(n), f"{n} should be prime or prime power in female string"

    return True, (
        f"Female string {female}: satisfies Fn+1=Fn+Fn-1 from seed (2,1); "
        f"18=2*3^2 is first term that is neither prime nor prime power; "
        f"all preceding terms from 2 onward are prime or prime powers: {{2,3,4=2^2,7,11}}"
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
        raise RuntimeError(f"cert [354] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
