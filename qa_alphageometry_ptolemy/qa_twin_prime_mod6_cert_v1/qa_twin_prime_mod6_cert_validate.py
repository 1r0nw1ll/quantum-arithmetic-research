# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (twin prime mod-6 structure: "
    "midpoints divisible by 6, all primes >3 ≡ ±1 mod 6, prime-power and "
    "two-prime-product exceptions; Euclid VII.28 coprime sum/difference → QA "
    "bead mutual coprimality); "
    "Theorem NT: mod-6 residues are observer labels; causal structure is "
    "modular arithmetic over integer prime factors; "
    "no float state, no QA orbit evolution"
)

"""
Cert [349] — QA Twin Prime Mod-6 Structure

Source: Iverson, B. & Elkins, C. (2006) Pythagorean Arithmetic Vol III, Chapter 5 pp.24-25
  p.24: "all so-called prime numbers are one of a pair of twin primes. So let us start
    by going through and listing them up to 100. They are: 5-7, 11-13, 17-19, 23-25,
    29-31, 35-37, 41-43, 47-49, 53-55, 59-61, 65-67, 71-73, 77-79, 83-85, 89-91,
    95-97, 101-103. Note the total absence of the prime number 3. This is what
    creates the twin primes. The intervening integer between the twin primes is
    always divisible by 3."
  p.24: "Any power of a prime number, is fundamentally prime and takes the place of
    its root." (25=5², 49=7², 121=11²)
  p.24: "integers which have two smaller prime factors. In the above list, these are:
    35, 55, 65, 77, 85, 91, and 95."
  p.25 (Euclid VII.28): "The sum of two numbers which are prime to each other, are
    also prime to them. The difference of two numbers which are prime to each other
    is also prime to them." — gives Fibonacci configuration (b,e,d,a) with d=b+e, a=d+e,
    all four mutually coprime.

Five claims:
  C1: Iverson twin pairs (6k-1, 6k+1) k=1..17 all have midpoint 6k divisible by 6
  C2: All primes p > 3 satisfy p ≡ ±1 (mod 6); no prime >3 ≡ 0,2,3,4 (mod 6)
  C3: Squares of primes ≥5 are ≡ 1 (mod 6) (same residue class as primes ≡ 1 mod 6)
  C4: Products of two distinct primes ≥5 are ≡ ±1 (mod 6)
  C5: Euclid VII.28 — gcd(b,e)=1 → (b,e,d,a) all mutually coprime (QA bead structure)
"""

from math import gcd


def _sieve(n: int) -> list[int]:
    """Return all primes ≤ n via sieve of Eratosthenes."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]


def check_c1() -> tuple[bool, str]:
    """Iverson's twin pairs (6k-1, 6k+1) have midpoints divisible by 6."""
    # Iverson's explicit list: pairs up to 101-103 → k=1..17
    iverson_pairs = [
        (5, 7), (11, 13), (17, 19), (23, 25), (29, 31),
        (35, 37), (41, 43), (47, 49), (53, 55), (59, 61),
        (65, 67), (71, 73), (77, 79), (83, 85), (89, 91),
        (95, 97), (101, 103),
    ]
    for lo, hi in iverson_pairs:
        mid = lo + 1  # = hi - 1 = 6k
        assert hi - lo == 2, f"({lo},{hi}) not a twin pair (gap != 2)"
        assert mid % 6 == 0, f"midpoint {mid} of ({lo},{hi}) not divisible by 6"
        assert lo % 6 in (1, 5), f"{lo} not ≡ ±1 (mod 6): {lo % 6}"
        assert hi % 6 in (1, 5), f"{hi} not ≡ ±1 (mod 6): {hi % 6}"
    # Also verify the (6k-1, 6k+1) pattern for k=1..20
    for k in range(1, 21):
        lo, hi = 6*k - 1, 6*k + 1
        assert hi - lo == 2
        assert (6*k) % 6 == 0
    return True, (
        f"All {len(iverson_pairs)} Iverson pairs have midpoints divisible by 6; "
        f"all endpoints ≡ ±1 (mod 6); pattern verified k=1..20"
    )


def check_c2() -> tuple[bool, str]:
    """All primes > 3 are ≡ ±1 (mod 6); no prime >3 is ≡ 0,2,3,4 (mod 6)."""
    primes = _sieve(500)
    large_primes = [p for p in primes if p > 3]
    bad = [(p, p % 6) for p in large_primes if p % 6 not in (1, 5)]
    assert not bad, f"Primes >3 not ≡ ±1 (mod 6): {bad}"
    pm1 = [p for p in large_primes if p % 6 == 1]
    pm5 = [p for p in large_primes if p % 6 == 5]
    return True, (
        f"All {len(large_primes)} primes in (3,500] are ≡ ±1 (mod 6); "
        f"{len(pm1)} are ≡ 1, {len(pm5)} are ≡ 5 (mod 6)"
    )


def check_c3() -> tuple[bool, str]:
    """Squares of primes ≥5 are ≡ 1 (mod 6) (Iverson: 25=5², 49=7², 121=11²)."""
    primes = _sieve(200)
    large_primes = [p for p in primes if p >= 5]
    for p in large_primes:
        sq = p * p
        assert sq % 6 == 1, f"{p}² = {sq} ≡ {sq % 6} (mod 6), not 1"
    # Verify Iverson's three specific examples
    assert 25 % 6 == 1
    assert 49 % 6 == 1
    assert 121 % 6 == 1
    return True, (
        f"All squares of {len(large_primes)} primes in [5,200] are ≡ 1 (mod 6); "
        f"25≡{25%6}, 49≡{49%6}, 121≡{121%6} (Iverson examples verified)"
    )


def check_c4() -> tuple[bool, str]:
    """Products of two distinct primes ≥5 are ≡ ±1 (mod 6)."""
    # Iverson's explicit seven: 35,55,65,77,85,91,95
    iverson_semiprimes = [
        (35, 5, 7), (55, 5, 11), (65, 5, 13),
        (77, 7, 11), (85, 5, 17), (91, 7, 13), (95, 5, 19),
    ]
    for n, p, q in iverson_semiprimes:
        assert n == p * q, f"{n} != {p}*{q}"
        assert n % 6 in (1, 5), f"{n}={p}×{q} ≡ {n%6} (mod 6), not ±1"
    # General: any p*q with p,q ≥5 distinct primes → both ≡ ±1 (mod 6) →
    # product ≡ (±1)(±1) = ±1 (mod 6). Verify for all such pairs up to 50.
    primes_ge5 = [p for p in _sieve(100) if p >= 5]
    for i, p in enumerate(primes_ge5):
        for q in primes_ge5[i+1:]:
            prod = p * q
            assert prod % 6 in (1, 5), f"{p}×{q}={prod} ≡ {prod%6} (mod 6)"
    return True, (
        f"All 7 Iverson semiprime exceptions (35,55,65,77,85,91,95) ≡ ±1 (mod 6); "
        f"general property verified for all distinct prime pairs from [5,100]"
    )


def check_c5() -> tuple[bool, str]:
    """Euclid VII.28: gcd(b,e)=1 (b odd) → (b,e,d,a) all mutually coprime (QA beads)."""
    # Euclid VII.28: if gcd(b,e)=1 then gcd(b+e,b)=1 and gcd(b+e,e)=1.
    # With b ODD (Iverson's prime Pythagorean requirement): a=b+2e also satisfies
    # gcd(a,b)=1 because any p|gcd(a,b) → p|(a-b)=2e; p|b odd → p|e, contradiction.
    pairs_checked = 0
    for b in range(1, 20, 2):  # b odd only (Iverson's requirement)
        for e in range(1, 20):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = d + e  # = b + 2*e (always odd since b odd, 2e even)
            # Euclid VII.28 direct: gcd(d,b)=1 and gcd(d,e)=1
            assert gcd(d, b) == 1, f"gcd(b+e,b) != 1 for b={b},e={e}"
            assert gcd(d, e) == 1, f"gcd(b+e,e) != 1 for b={b},e={e}"
            # Apply VII.28 to (d,e): gcd(a,d)=1 and gcd(a,e)=1
            assert gcd(a, d) == 1, f"gcd(a,d) != 1 for b={b},e={e}"
            assert gcd(a, e) == 1, f"gcd(a,e) != 1 for b={b},e={e}"
            # gcd(a,b)=1: b odd, p|gcd(a,b) → p|(a-b)=2e, p odd → p|e → contradiction
            assert gcd(a, b) == 1, f"gcd(a,b) != 1 for b={b},e={e}"
            pairs_checked += 1
    return True, (
        f"Euclid VII.28 verified for {pairs_checked} coprime pairs (b odd, e any) in [1,19]²: "
        f"d=b+e and a=d+e satisfy gcd(d,b)=gcd(d,e)=gcd(a,d)=gcd(a,e)=gcd(a,b)=1; "
        f"all four QA bead numbers (b,e,d,a) mutually coprime when b is odd"
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
        raise RuntimeError(f"cert [349] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
