# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-1 Ch.VII Koenig Series and Tree of Life: "
    "A+B=2G; I(1,e)=2e^2-1; I(b,1)=b^2-2; primes p≡3,5(mod 8) never appear as H,I; "
    "49=7^2 is functionally prime in QA = power of prime); "
    "Theorem NT: 'concentric circles', 'electron orbit', 'Tree of Life' are observer projections; "
    "no float state, no QA orbit evolution"
)

"""
Cert [364] — QA Pyth-1 Koenig Series and Tree of Life (Ch.VII)

Source: Iverson (1993) Pythagorean Arithmetic Vol I, Chapter VII pp.73-86
  p.67: '(d+e)^2 + (d-e)^2 = 2G. But since d+e=a and d-e=b the formula reduces to
         a^2+b^2=2G, or A+B=2G.'
  p.76-77: 'Two main branches: 1,(5),7,(13),17,(25),31,... and 1,(5),7,(17),23,(37),47,...'
           'the first main branch maintains the value of b at a constant, b=1,
            and the second branch maintains e at a constant, e=1.'
  p.72: '49 is both functionally prime and prime by the definition used in Quantum Arithmetic
         since it is a power of a prime number, and therefore has only one prime factor.'
  p.72: 'the notable exception of 2, 3, 11, 19, and 43' [never appear as H or I values]

Five claims:
  C1: A+B=2G for all prime Pythagorean pairs
      proof: A+B=(d+e)^2+(d-e)^2=2(d^2+e^2)=2G ✓
  C2: I(1,e) = 2e^2 - 1 for all e >= 1 (b=1 Koenig branch I-values)
      proof: b=1, I=|b^2-2e^2|=|1-2e^2|=2e^2-1 (since 2e^2>1 for all e>=1) ✓
      sequence: 1, 7, 17, 31, 49, 71, 97, 127, 161, 199, ...
  C3: I(b,1) = b^2 - 2 for all odd b >= 1 (e=1 Koenig branch I-values)
      proof: e=1, I=|b^2-2*1^2|=b^2-2 (since b odd >= 1: b^2 >= 1 >= 2 only for b=1
             where I=1=|1-2|; for b>=3: b^2>=9>2 always) ✓
      sequence: 1 (b=1), 7 (b=3), 23 (b=5), 47 (b=7), 79 (b=9), 119 (b=11), ...
  C4: Primes p NOT of the form ±(x^2-2y^2) never appear as H or I values.
      Equivalently, prime p appears as H or I value only if p≡1 or p≡7 (mod 8)
      (i.e., 2 is a QR mod p). Primes ≡3 or ≡5 (mod 8) are excluded.
      First excluded primes: 2 (even → excluded by H,I odd), 3 (min factor cert),
      5,13,29,37,53,61,... (≡5 mod 8), 11,19,43,59,67,... (≡3 mod 8)
      Verified numerically for all pairs (b,e)<=50 and all primes p<200.
  C5: 49=7^2 is QA-functionally-prime (only one prime factor); I(1,5)=49 appears
      in the Koenig b=1 branch; this is the first non-prime I-value in that branch.
      General: 'functionally prime' = prime OR prime power (only one prime base factor)
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


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _prime_base(n: int) -> int:
    """If n is a prime power p^k, return p. Else return -1."""
    if n <= 1:
        return -1
    i = 2
    while i * i <= n:
        if n % i == 0:
            p = i
            while n % p == 0:
                n //= p
            return p if n == 1 else -1
        i += 1
    return n  # n itself is prime


def check_c1() -> tuple[bool, str]:
    """A+B=2G for all prime pairs."""
    count = 0
    for b, e, d, a in _prime_pairs(35, 35):
        A = a * a
        B = b * b
        G = d * d + e * e
        assert A + B == 2 * G, f"A+B={A+B} != 2G={2*G} at b={b},e={e}"
        count += 1
    # Proof: A+B=(d+e)^2+(d-e)^2=d^2+2de+e^2+d^2-2de+e^2=2d^2+2e^2=2G ✓
    # This means G is the arithmetic mean of A and B: G=(A+B)/2
    return True, (
        f"A+B=2G verified for all {count} prime pairs (b,e)<=35; "
        f"G is arithmetic mean of A=a^2 and B=b^2; "
        f"proof: A+B=(d+e)^2+(d-e)^2=2(d^2+e^2)=2G ✓"
    )


def check_c2() -> tuple[bool, str]:
    """I(1,e) = 2e^2 - 1 for the b=1 Koenig branch."""
    count = 0
    for e in range(1, 50):
        b = 1
        # gcd(1, e) = 1 always, so all (1,e) are prime pairs
        d = b + e
        a = d + e
        C = 2 * d * e
        F = a * b
        I_val = abs(C - F)
        expected = 2 * e * e - 1
        assert I_val == expected, f"I(1,e)={I_val} != 2e^2-1={expected} at e={e}"
        count += 1
    # Sequence: e=1→1, e=2→7, e=3→17, e=4→31, e=5→49, e=6→71, ...
    # proof: b=1 → I=|b^2-2e^2|=|1-2e^2|=2e^2-1 since 2e^2≥2>1 for all e≥1
    i_vals = [2 * e * e - 1 for e in range(1, 11)]
    return True, (
        f"I(1,e)=2e^2-1 verified for e=1..49 ({count} cases); "
        f"first 10 values: {i_vals}; "
        f"proof: b=1, I=|1-2e^2|=2e^2-1 (since 2e^2≥2>1); "
        f"I(1,5)=49=7^2 is the first non-prime (composite) in this branch ✓"
    )


def check_c3() -> tuple[bool, str]:
    """I(b,1) = b^2 - 2 for the e=1 Koenig branch (b odd, gcd(b,1)=1 always)."""
    count = 0
    for b in range(1, 100, 2):  # odd b from 1 to 99
        e = 1
        d = b + e
        a = d + e
        C = 2 * d * e
        F = a * b
        I_val = abs(C - F)
        expected = abs(b * b - 2)   # = |b^2 - 2*1^2|
        assert I_val == expected, f"I(b,1)={I_val} != b^2-2={expected} at b={b}"
        # For b=1: I=1; for b>=3: I=b^2-2>0
        if b == 1:
            assert I_val == 1
        else:
            assert b * b > 2, f"b^2<=2 impossible for odd b>=3"
            assert I_val == b * b - 2
        count += 1
    i_vals = [abs(b * b - 2) for b in range(1, 22, 2)]  # b=1,3,5,...,21
    return True, (
        f"I(b,1)=b^2-2 verified for odd b=1..99 ({count} cases); "
        f"first 11 values: {i_vals}; "
        f"proof: e=1, I=|b^2-2*1^2|=b^2-2 for b>=3 (and |1-2|=1 for b=1); "
        f"sequence: 1,7,23,47,79,119,... with 119=7x17 first composite ✓"
    )


def check_c4() -> tuple[bool, str]:
    """Primes p≡3 or 5 (mod 8) never appear as H or I values."""
    # Collect all H,I values from pairs (b,e)<=50
    hi_vals: set[int] = set()
    for b, e, d, a in _prime_pairs(50, 50):
        C = 2 * d * e
        F = a * b
        H = C + F
        I = abs(C - F)
        hi_vals.add(H)
        hi_vals.add(I)

    # Check: no prime p≡3(mod 8) or p≡5(mod 8) appears in H,I values
    excluded_primes = []
    included_primes = []
    for p in range(2, 200):
        if not _is_prime(p):
            continue
        if p % 8 in (3, 5):
            # Should NOT appear
            excluded_primes.append(p)
            if p in hi_vals:
                assert False, f"Prime p={p} ≡{p%8}(mod 8) found as H,I value — contradiction!"
        elif p % 8 in (1, 7):
            # Should appear (for large enough pairs)
            included_primes.append(p)
    # Spot-check: confirm some ≡1,7(mod 8) primes DO appear
    known_appear = [7, 17, 23, 31, 41, 47, 71, 73]
    for p in known_appear:
        assert p in hi_vals, f"Expected p={p}≡{p%8}(mod 8) in H,I values but not found"

    excluded_sample = [p for p in excluded_primes[:10]]
    return True, (
        f"No prime p≡3 or 5(mod 8) found in H,I values for (b,e)<=50; "
        f"excluded primes checked: {excluded_sample} (and more); "
        f"all ≡1,7(mod 8) primes verified present: {known_appear}; "
        f"algebraic reason: H=a^2-2e^2, I=|b^2-2e^2|; prime p appears iff 2∈QR(p) iff p≡±1(mod 8) ✓"
    )


def check_c5() -> tuple[bool, str]:
    """49=7^2 is QA-functionally-prime; it is I(1,5) in the Koenig b=1 branch."""
    # Verify I(1,5)=49
    b, e = 1, 5
    d = b + e   # = 6
    a = d + e   # = 11
    C = 2 * d * e  # = 60
    F = a * b      # = 11
    I_val = abs(C - F)  # = |60-11| = 49
    assert I_val == 49, f"I(1,5)={I_val} != 49"
    assert 49 == 7 * 7, "49=7^2 ✓"
    # 49 is QA-functionally-prime: only one prime factor (7), even though composite
    assert _prime_base(49) == 7, "49 has only one prime base (7)"
    assert not _is_prime(49), "49 is not prime by definition"
    # First non-prime I-value in the b=1 branch (e=1..5): check e=1..4 are prime
    for e in range(1, 5):
        I = 2 * e * e - 1
        if I > 1:
            assert _is_prime(I), f"I(1,{e})={I} is not prime (unexpected)"
    # Functionally prime: prime powers p^k (k>=1) are considered prime in QA
    # because they have only one prime "factor" in the prime number system
    fp_examples = [(p, k) for p in [7, 11, 13] for k in [1, 2, 3]
                   if p ** k < 200]
    fp_check = all(_prime_base(p ** k) == p for p, k in fp_examples)
    assert fp_check, "Prime powers should have a single prime base"
    return True, (
        f"I(1,5)=|60-11|=49=7^2 ✓; 49 is QA-functionally-prime (single prime factor 7); "
        f"I(1,e) for e=1..4: {[2*e*e-1 for e in range(1,5)]} are all prime; "
        f"49 is the first composite value in the b=1 branch; "
        f"QA functional primality: prime OR prime power (one prime base) = "
        f"{{p, p^2, p^3,...}} for prime p ✓"
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
        raise RuntimeError(f"cert [364] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
