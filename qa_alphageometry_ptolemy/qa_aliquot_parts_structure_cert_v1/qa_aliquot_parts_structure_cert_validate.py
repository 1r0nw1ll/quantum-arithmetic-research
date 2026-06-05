# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1991) QA Vol II Books 3&4 — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (gcd, prime factoring, product); "
    "no QA state evolution; Theorem NT: aliquot-part validity label and "
    "gear-synchronization count are observer-layer descriptions of integer "
    "divisibility; all arithmetic exact integer, no float"
)

"""
Cert [328] — QA Aliquot Parts Structure

Source: Iverson (1991) QA Volume II Books 3 & 4, pp.13-14
"ALIQUOT PARTS", "EXAMPLE", "QUESTIONS"

An aliquot part of a quantum wave is the product of all prime factors in
its quantum number EXCEPT one.  For two waves X and Y that share aliquot A
with unique primes p_x and p_y, they gear-synchronize after A*p_x*p_y total
units: wave X completes p_y cycles, wave Y completes p_x cycles.

Valid aliquot parts must contain {a power of 2, the prime 3, and at least
one of 5 or 7} -- the harmonic minimum Iverson states explicitly (p.14).

Five claims certified via exact integer arithmetic.
"""

from math import gcd, isqrt


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True


def _aliquot_valid(factors: list[int]) -> bool:
    """Valid if product contains {power of 2, 3, and one of 5 or 7}."""
    p = 1
    for f in factors:
        p *= f
    has_2 = (p % 2 == 0)
    has_3 = (p % 3 == 0)
    has_5_or_7 = (p % 5 == 0) or (p % 7 == 0)
    return has_2 and has_3 and has_5_or_7


def check_c1() -> tuple[bool, str]:
    """Primes coprime-to-30 in (0,30) = {7,11,13,17,19,23,29}; with 1 = 8 values; pairs sum to 30."""
    coprime_to_30 = [n for n in range(1, 30) if gcd(n, 30) == 1]
    # Should be {1,7,11,13,17,19,23,29}
    assert coprime_to_30 == [1, 7, 11, 13, 17, 19, 23, 29], f"got {coprime_to_30}"
    assert len(coprime_to_30) == 8, "expected 8 values"
    # Complement pairs sum to 30
    forward = coprime_to_30[:]
    reversed_ = list(reversed(coprime_to_30))
    for a, b in zip(forward, reversed_):
        assert a + b == 30, f"{a}+{b} != 30"
    # Iverson's special forms: 5^2+5=30, 3^3+3=30, 2^5-2=30
    assert 5 * 5 + 5 == 30
    assert 3 * 3 * 3 + 3 == 30
    assert 2 * 2 * 2 * 2 * 2 - 2 == 30
    return True, f"coprime-to-30 in (0,30) = {coprime_to_30}; all pairs sum to 30; 5^2+5=3^3+3=2^5-2=30"


def check_c2() -> tuple[bool, str]:
    """Aliquot parts of {32,3,5,7,11}: 3 valid {3360,5280,7392}, 2 invalid {1155,12320}."""
    factors = [32, 3, 5, 7, 11]
    candidates = []
    for i in range(len(factors)):
        subset = factors[:i] + factors[i + 1:]
        product = 1
        for f in subset:
            product *= f
        excluded = factors[i]
        valid = _aliquot_valid(subset)
        candidates.append((excluded, product, valid))

    valid_products = sorted(p for _, p, v in candidates if v)
    invalid_products = sorted(p for _, p, v in candidates if not v)
    assert valid_products == [3360, 5280, 7392], f"valid got {valid_products}"
    assert invalid_products == [1155, 12320], f"invalid got {invalid_products}"
    # Verify individual products match Iverson's answer
    assert 32 * 3 * 5 * 7 == 3360  # leave out 11
    assert 32 * 3 * 5 * 11 == 5280  # leave out 7
    assert 32 * 3 * 7 * 11 == 7392  # leave out 5
    assert 3 * 5 * 7 * 11 == 1155   # leave out 32 -- invalid (no power of 2)
    assert 32 * 5 * 7 * 11 == 12320  # leave out 3 -- invalid (no 3)
    return True, f"valid aliquots {valid_products}; invalid {invalid_products}; matches Iverson p.14"


def check_c3() -> tuple[bool, str]:
    """Wave gear synchronization: shared aliquot A with unique primes p, q -> sync at A*p*q."""
    # Iverson's example: aliquot = 2*3*5*13*47, X unique prime = 53, Y unique prime = 7
    A = 2 * 3 * 5 * 13 * 47
    p_x = 53
    p_y = 7
    # Wave X period = A * p_x; wave Y period = A * p_y
    period_x = A * p_x
    period_y = A * p_y
    sync_point = _lcm(period_x, period_y)
    # Since gcd(p_x, p_y) = gcd(53, 7) = 1, lcm = A * p_x * p_y
    assert gcd(p_x, p_y) == 1, "unique primes must be coprime"
    assert sync_point == A * p_x * p_y, "sync point must be A*p_x*p_y for coprime unique primes"
    cycles_x = sync_point // period_x
    cycles_y = sync_point // period_y
    assert cycles_x == p_y, f"X cycles = p_y = {p_y}, got {cycles_x}"
    assert cycles_y == p_x, f"Y cycles = p_x = {p_x}, got {cycles_y}"
    # General verification: 10 random coprime prime pairs sharing aliquot 30
    A2 = 30  # = 2*3*5
    test_pairs = [(7, 11), (11, 13), (7, 13), (11, 17), (13, 17),
                  (7, 17), (11, 19), (7, 19), (13, 19), (17, 19)]
    for px, py in test_pairs:
        assert gcd(px, py) == 1
        sx = _lcm(A2 * px, A2 * py)
        assert sx == A2 * px * py
        assert sx // (A2 * px) == py
        assert sx // (A2 * py) == px
    return True, f"Iverson example: aliquot={A}, X unique=53, Y unique=7; sync at {sync_point}; X completes {cycles_x} cycles, Y completes {cycles_y} cycles"


def check_c4() -> tuple[bool, str]:
    """Three special forms of 30: 5^2+5=30, 3^3+3=30, 2^5-2=30 (Iverson p.14 symmetry note)."""
    assert 5 * 5 + 5 == 30, "5^2+5 must be 30"
    assert 3 * 3 * 3 + 3 == 30, "3^3+3 must be 30"
    assert 2 * 2 * 2 * 2 * 2 - 2 == 30, "2^5-2 must be 30"
    # These reflect that 30 = p*(p^(p-1) + 1) / ... is structurally tied to small primes
    # p=2: 2^5-2 = 2*(2^4-1) = 2*15 = 30; p=3: 3^3+3 = 3*(9+1) = 3*10 = 30; p=5: 5^2+5 = 5*6 = 30
    assert 2 * (2 * 2 * 2 * 2 - 1) == 30  # 2*(16-1)=30
    assert 3 * (3 * 3 + 1) == 30           # 3*10=30
    assert 5 * (5 + 1) == 30               # 5*6=30
    return True, "Three forms: 5^2+5=3^3+3=2^5-2=30; factored: 5*6=3*10=2*15=30"


def check_c5() -> tuple[bool, str]:
    """Aliquot count = n for n prime factors; validity: must retain {2,3,5-or-7}."""
    # For any set of n prime factors, exactly n aliquot candidates (leave out each once)
    for n in range(4, 8):
        primes = [2, 3, 5, 7, 11, 13, 17][:n]
        candidates = [primes[:i] + primes[i + 1:] for i in range(n)]
        assert len(candidates) == n, f"expected {n} candidates"
    # With {2,3,5,7}: leaving out 2 or 3 gives invalid (sole representative of their class)
    factors_4 = [2, 3, 5, 7]
    expected_validity = [False, False, True, True]  # out: 2, 3, 5, 7
    for i in range(4):
        subset = factors_4[:i] + factors_4[i + 1:]
        got = _aliquot_valid(subset)
        assert got == expected_validity[i], (
            f"leaving out {factors_4[i]} from {factors_4}: expected valid={expected_validity[i]}, got {got}"
        )
    # Verify: leaving out 2 gives {3,5,7}=105 (odd, invalid); leaving out 3 gives {2,5,7}=70 (no 3, invalid)
    assert not _aliquot_valid([3, 5, 7]), "no power of 2 -> invalid"
    assert not _aliquot_valid([2, 5, 7]), "no 3 -> invalid"
    assert _aliquot_valid([2, 3, 7]), "has 2, 3, 7 -> valid"
    assert _aliquot_valid([2, 3, 5]), "has 2, 3, 5 -> valid"
    # With {2,3,5,11}: leaving out 5 gives {2,3,11}=66; no 5 or 7 -> invalid
    assert not _aliquot_valid([2, 3, 11]), "no 5 or 7 -> invalid"
    return True, "aliquot count = n; need {2,3,5-or-7}; from {2,3,5,7}: 2 valid (drop 5 or 7), 2 invalid (drop 2 or 3)"


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
        raise RuntimeError(f"cert [328] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
