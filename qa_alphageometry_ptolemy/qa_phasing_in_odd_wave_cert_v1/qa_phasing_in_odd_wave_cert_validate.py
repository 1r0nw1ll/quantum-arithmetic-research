# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1991) QA Vol II Books 3&4 — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (mod, gcd, lcm); "
    "no QA state evolution; Theorem NT: 'phasing in' and 'half-cycle' "
    "are observer-layer labels on integer divisibility; "
    "all arithmetic exact integer, no float"
)

"""
Cert [331] — QA Phasing In: Odd-Wave Half-Integer Half-Cycle Structure

Source: Iverson (1991) QA Volume II Books 3 & 4, pp.8, 11-12
"PHASING IN", "OTHER CYCLES"

Iverson discusses whether waves can synchronize at a half-cycle point.
For a wave of period p, the half-cycle occurs at p/2. If p is odd,
p/2 is not an integer — no discrete half-cycle sync point exists.
If p is even, p/2 is an integer and half-cycle sync is possible.

Five claims certified via integer divisibility arithmetic.
"""

from math import gcd


def _is_even(n: int) -> bool:
    return n % 2 == 0


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _has_integer_half_cycle(p: int) -> bool:
    """Return True iff p has an integer half-cycle (p divisible by 2)."""
    return p % 2 == 0


def check_c1() -> tuple[bool, str]:
    """For all odd p in {3,5,7,9,11,13,15}: p/2 is not an integer (2*floor(p/2) != p)."""
    odd_periods = [3, 5, 7, 9, 11, 13, 15]
    for p in odd_periods:
        assert p % 2 == 1, f"{p} is not odd"
        assert not _has_integer_half_cycle(p), f"odd p={p} unexpectedly has integer half-cycle"
        half_floor = p // 2
        assert 2 * half_floor != p, f"odd p={p}: 2*floor(p/2)={2*half_floor} == p"
    return True, f"All odd periods {odd_periods}: half-cycle p/2 is non-integer"


def check_c2() -> tuple[bool, str]:
    """For all even p in {2,4,6,8,10,12}: p/2 is an exact integer."""
    even_periods = [2, 4, 6, 8, 10, 12]
    for p in even_periods:
        assert p % 2 == 0, f"{p} is not even"
        assert _has_integer_half_cycle(p), f"even p={p} missing integer half-cycle"
        half = p // 2
        assert 2 * half == p, f"even p={p}: 2*(p//2)={2*half} != p"
    return True, f"All even periods {even_periods}: half-cycle p//2 is exact integer"


def check_c3() -> tuple[bool, str]:
    """For two coprime odd periods p,q: lcm(p,q)=p*q; no common factor of 2 to share."""
    coprime_odd_pairs = [(3, 5), (3, 7), (5, 7), (5, 9), (7, 9), (3, 11), (5, 11)]
    for p, q in coprime_odd_pairs:
        assert p % 2 == 1 and q % 2 == 1, f"({p},{q}) not both odd"
        assert gcd(p, q) == 1, f"({p},{q}) not coprime; gcd={gcd(p,q)}"
        lcm_pq = _lcm(p, q)
        assert lcm_pq == p * q, f"({p},{q}): lcm={lcm_pq} != p*q={p*q}"
        # Full cycle lcm is also odd → no integer half-cycle even at full sync
        assert lcm_pq % 2 == 1, f"lcm({p},{q})={lcm_pq} is even (unexpected)"
    return True, f"Coprime odd pairs: lcm=p*q (all odd); verified {coprime_odd_pairs}"


def check_c4() -> tuple[bool, str]:
    """Par-type partition: 2-par (n%4==2) and 4-par (n%4==0) have integer half-cycles; 3-par and 5-par do not."""
    sample = list(range(1, 25))
    for n in sample:
        r = n % 4
        has_half = _has_integer_half_cycle(n)
        if r == 0 or r == 2:
            # 4-par (≡0) and 2-par (≡2) — even, have integer half-cycle
            assert has_half, f"n={n} (r={r}) expected even/integer-half-cycle"
        else:
            # 3-par (≡3) and 5-par (≡1) — odd, no integer half-cycle
            assert not has_half, f"n={n} (r={r}) expected odd/no-integer-half-cycle"
    return True, "Par-types 2-par(≡2 mod 4) and 4-par(≡0) have integer half-cycles; 3-par(≡3) and 5-par(≡1) do not"


def check_c5() -> tuple[bool, str]:
    """Odd period p: first phase-in k with k < p and k ≡ 0 (mod p) does not exist; phase occurs only at k=p."""
    odd_periods = [3, 5, 7, 9, 11]
    for p in odd_periods:
        # No k in {1,...,p-1} divides p (i.e., no sync before full period)
        # More specifically: no half-integer half-cycle means no "phase-in" at p/2
        # The only sync point k with p | k and k >= 1 is k = p, 2p, ...
        sync_in_cycle = [k for k in range(1, p) if k * 2 == p]
        assert sync_in_cycle == [], (
            f"odd p={p}: unexpected half-cycle sync at {sync_in_cycle}"
        )
    # Even periods do have half-cycle sync
    even_periods = [4, 6, 8, 10, 12]
    for p in even_periods:
        half = p // 2
        assert half * 2 == p, f"even p={p}: half-cycle at {half}, 2*{half}={2*half} != {p}"
    return True, "Odd periods: no integer k < p with 2k=p (no half-cycle sync); even periods: k=p//2 exists"


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
        raise RuntimeError(f"cert [331] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
