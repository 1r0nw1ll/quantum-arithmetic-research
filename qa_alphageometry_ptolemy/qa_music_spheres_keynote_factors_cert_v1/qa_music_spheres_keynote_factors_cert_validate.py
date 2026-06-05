# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1991) QA Vol II Books 3&4 — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (prime factoring, gcd); "
    "no QA state evolution; Theorem NT: keynote values and harmonic "
    "quality are observer-layer labels on integer prime factorization; "
    "all arithmetic exact integer, no float"
)

"""
Cert [330] — QA Music of the Spheres Keynote Factoring

Source: Iverson (1991) QA Volume II Books 3 & 4, pp.20-21
"MUSIC OF THE SPHERES", "A MUSICAL SCALE?"

The four male keynotes (891, 1580, 1602, 2226) and the four approximate
female keynotes (756, 1050, 1197, 1548) are given with their prime
factorizations. Iverson: "Their factors are: 2, 3, 5 & 7, along with one
larger prime number between 7 and 100."

Five claims certified via exact integer prime factorization.
"""

from math import gcd, isqrt


def _prime_factors(n: int) -> list[int]:
    """Return sorted list of prime factors (with repetition)."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _distinct_primes(n: int) -> set[int]:
    """Return set of distinct prime factors."""
    return set(_prime_factors(n))


SMALL_PRIMES = {2, 3, 5, 7}


def _large_prime(n: int) -> int | None:
    """Return the unique prime factor > 7, or None if all small."""
    large = _distinct_primes(n) - SMALL_PRIMES
    if len(large) == 1:
        return next(iter(large))
    return None


def check_c1() -> tuple[bool, str]:
    """Male keynote factorizations: 891=3^4*11, 1602=2*3^2*89, 1580=2^2*5*79, 2226=2*3*7*53."""
    cases = {
        891:  [3, 3, 3, 3, 11],      # 3^4 * 11
        1580: [2, 2, 5, 79],          # 2^2 * 5 * 79
        1602: [2, 3, 3, 89],          # 2 * 3^2 * 89
        2226: [2, 3, 7, 53],          # 2 * 3 * 7 * 53
    }
    for val, expected in cases.items():
        got = _prime_factors(val)
        assert got == expected, f"{val}: expected {expected}, got {got}"
    return True, "891=3^4*11, 1580=2^2*5*79, 1602=2*3^2*89, 2226=2*3*7*53"


def check_c2() -> tuple[bool, str]:
    """Each male keynote has exactly one prime factor > 7 and that factor is in (7,100)."""
    male_notes = [891, 1580, 1602, 2226]
    large_primes = {}
    for n in male_notes:
        lp = _large_prime(n)
        assert lp is not None, f"{n} does not have exactly one large prime factor"
        assert 7 < lp < 100, f"{n}: large prime {lp} not in (7,100)"
        large_primes[n] = lp
    # All four large primes must be distinct
    assert len(set(large_primes.values())) == 4, f"large primes not all distinct: {large_primes}"
    return True, f"Large primes: {large_primes}; all distinct, all in (7,100)"


def check_c3() -> tuple[bool, str]:
    """Female keynote integers (756,1050,1197,1548) factorizations verified.

    756 and 1050 are 7-smooth; 1197 and 1548 each have exactly one large prime.
    """
    # Iverson's approximate female values, rounded to nearest integer
    # 754.95383 -> 756=2^2*3^3*7, 1050.7297 -> 1050=2*3*5^2*7
    # 1197.965 -> 1197=3^2*7*19, 1547.4254 -> 1548=2^2*3^2*43
    female_cases = {
        756:  ([2, 2, 3, 3, 3, 7],  False),  # 2^2 * 3^3 * 7  — 7-smooth
        1050: ([2, 3, 5, 5, 7],      False),  # 2 * 3 * 5^2 * 7 — 7-smooth
        1197: ([3, 3, 7, 19],        True),   # 3^2 * 7 * 19
        1548: ([2, 2, 3, 3, 43],     True),   # 2^2 * 3^2 * 43
    }
    for val, (expected, has_large) in female_cases.items():
        got = _prime_factors(val)
        assert got == expected, f"{val}: expected {expected}, got {got}"
        lp = _large_prime(val)
        if has_large:
            assert lp is not None, f"female note {val} expected large prime but got none"
        else:
            assert lp is None, f"female note {val} expected 7-smooth but found large prime {lp}"
    return True, "756=2^2*3^3*7(7-smooth), 1050=2*3*5^2*7(7-smooth), 1197=3^2*7*19, 1548=2^2*3^2*43"


def check_c4() -> tuple[bool, str]:
    """All 8 keynotes: factors ⊆ {2,3,5,7} ∪ {one prime in (7,100)}; at most one large prime."""
    all_notes = [891, 1580, 1602, 2226, 756, 1050, 1197, 1548]
    for n in all_notes:
        factors = _distinct_primes(n)
        large = factors - SMALL_PRIMES
        assert len(large) <= 1, f"{n}: has {len(large)} large prime factors, expected at most 1"
        if large:
            lp = next(iter(large))
            assert 7 < lp < 100, f"{n}: large prime {lp} not in (7,100)"
    return True, "All 8 keynotes: factors ⊆ {2,3,5,7} ∪ {one prime in (7,100)}; at most one large prime each"


def check_c5() -> tuple[bool, str]:
    """Collective factor coverage: {2,3,5,7} all appear across male notes; same for female."""
    male_notes = [891, 1580, 1602, 2226]
    female_notes = [756, 1050, 1197, 1548]
    male_small = set()
    for n in male_notes:
        male_small |= (_distinct_primes(n) & SMALL_PRIMES)
    # Male notes collectively have 3,5,7 but NOT all four — 891 has only {3,11}
    assert 3 in male_small and 5 in male_small and 7 in male_small, \
        f"male notes missing some of {{3,5,7}}: got {male_small}"
    # 2 appears in 1580, 1602, 2226 but not 891
    assert 2 in male_small, f"2 missing from male collective: {male_small}"
    assert male_small == {2, 3, 5, 7}, f"male collective small primes = {male_small}"
    female_small = set()
    for n in female_notes:
        female_small |= (_distinct_primes(n) & SMALL_PRIMES)
    assert female_small == {2, 3, 5, 7}, f"female collective small primes = {female_small}"
    return True, f"Male collective small primes = {male_small}; female = {female_small}; both = {{2,3,5,7}}"


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
        raise RuntimeError(f"cert [330] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
