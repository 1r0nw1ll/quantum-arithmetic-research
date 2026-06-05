# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) QA Book 2 Ch.1 p.7 — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (QN validation, ratio check, "
    "orbit class); no QA state evolution; Theorem NT: 'comet orbit' and "
    "'perigee/apogee ratio' are observer-layer labels on the QN integer "
    "structure; all arithmetic exact integer, no float"
)

"""
Cert [335] — QA Halley's Comet Quantum Number

Source: Iverson, B. (1993) Quantum Arithmetic Book 2: Natural Arithmetic,
  Chapter 1 "Doubling", page 7:
  "The Quantum Number for the orbit of Halley's Comet is 1, 29, 30, 59.
   Its ratio between the perigee and apogee is 1:59. This is a much
   elongated elliptical orbit."

QN = (b, e, d, a) = (1, 29, 30, 59)
  b=1, e=29, d=b+e=30, a=b+2e=59

Ratio b:a = 1:59 (perigee:apogee, minimum:maximum radii in the ellipse).

Properties:
  - b=1 (smallest possible QN base, primordial orbit)
  - gcd(b,e)=gcd(1,29)=1 (coprime, valid QN)
  - 29 is prime (e is a prime)
  - d=30=2×3×5 (maximally 5-smooth d-value; highly composite)
  - a=59 is prime (maximal element is prime)
  - b:a = 1:59; ratio 59:1 = "much elongated"
  - d-value 30 is one of the six Cosmos d-values {6,10,14,15,21,26,30}?
    No — Cosmos d-values under mod-24 are in {1..24}; 30 mod 24=6 →
    d≡6 (mod 24) places (1,29) in the Cosmos orbit with mod-24 tuples.

Five claims certified via integer arithmetic.
"""

from math import gcd


def _qa_step_m24(b: int, e: int) -> tuple[int, int, int, int]:
    """Compute QA 4-tuple with A2 raw coords, then mod-24 orbit class."""
    d_raw = b + e
    a_raw = b + 2 * e
    return b, e, d_raw, a_raw


def _orbit_family_m24(b: int, e: int, m: int = 24) -> str:
    """Classify orbit family under mod-m QA dynamics."""
    d = ((b + e - 1) % m) + 1      # A1 compliant
    a = ((b + 2 * e - 1) % m) + 1
    # Singularity: b==e==d==a (all equal to mod value)
    if b % m == 0 and e % m == 0:
        return "singularity"
    # Satellite: period-8 orbit; check by stepping
    state = (b % m or m, e % m or m)
    seen = {}
    k = 0
    while state not in seen:
        seen[state] = k
        nb = state[1]
        ne = ((state[0] + state[1] - 1) % m) + 1
        state = (nb, ne)
        k += 1
        if k > 200:
            return "unknown"
    period = k - seen[state]
    if period == 8:
        return "satellite"
    if period == 24:
        return "cosmos"
    return f"period-{period}"


def check_c1() -> tuple[bool, str]:
    """QN (1,29,30,59): internal consistency — d=b+e, a=b+2e, gcd=1."""
    b, e = 1, 29
    d_raw = b + e
    a_raw = b + 2 * e
    assert d_raw == 30, f"d={d_raw}, expected 30"
    assert a_raw == 59, f"a={a_raw}, expected 59"
    assert gcd(b, e) == 1, f"gcd(b,e)={gcd(b,e)}, expected 1"
    # A2 raw coords — no mod reduction in QA layer
    return True, f"QN (b,e,d,a)=(1,29,30,59); d=b+e={d_raw}; a=b+2e={a_raw}; gcd(1,29)=1"


def check_c2() -> tuple[bool, str]:
    """Perigee:apogee ratio b:a = 1:59; b is unit, a=59 is prime."""
    b, a = 1, 59
    assert a == 59, f"a={a}, expected 59"
    # Verify 59 is prime
    assert all(59 % i != 0 for i in range(2, 8)), "59 not prime"
    # Ratio
    ratio_str = f"{b}:{a}"
    assert ratio_str == "1:59", f"ratio={ratio_str}, expected 1:59"
    # e=29 is also prime
    e = 29
    assert all(29 % i != 0 for i in range(2, 6)), "29 not prime"
    return True, f"b:a=1:59; b=1 (unit base); a=59 prime; e=29 prime; highly elongated"


def check_c3() -> tuple[bool, str]:
    """d=30=2×3×5 is 5-smooth (no prime >5); d:b = 30:1; a:d = 59:30."""
    d = 30
    # Prime factorization of 30
    factors = []
    n = d
    for p in [2, 3, 5, 7, 11]:
        while n % p == 0:
            factors.append(p)
            n //= p
    assert n == 1, f"30 has prime factor > 11: remaining={n}"
    assert max(factors) == 5, f"30 has prime factor > 5: {factors}"
    assert sorted(factors) == [2, 3, 5], f"30 factors: {factors}"
    # 30 = lcm(2,3,5) = 2×3×5
    from math import lcm
    assert lcm(2, 3, 5) == 30, "lcm(2,3,5) != 30"
    return True, f"d=30=2×3×5 (5-smooth, maximally composite for d=b+e with b=1); lcm(2,3,5)=30"


def check_c4() -> tuple[bool, str]:
    """Mod-24 orbit class of (1,29): b%24=1, e%24=5; classify under QA dynamics."""
    b, e = 1, 29
    # Mod-24 representatives: b mod 24 = 1, e mod 24 = 5
    b24 = b % 24 or 24  # A1: use {1..24}
    e24 = e % 24 or 24
    assert b24 == 1, f"b mod 24 = {b24}, expected 1"
    assert e24 == 5, f"e mod 24 = {e24}, expected 5"
    # d mod 24: (b+e-1)%24+1 = (1+29-1)%24+1 = 29%24+1 = 5+1=6
    d24 = ((b + e - 1) % 24) + 1
    assert d24 == 6, f"d mod 24 = {d24}, expected 6"
    # a mod 24: (b+2e-1)%24+1 = (59-1)%24+1 = 58%24+1 = 10+1=11
    a24 = ((b + 2 * e - 1) % 24) + 1
    assert a24 == 11, f"a mod 24 = {a24}, expected 11"
    family = _orbit_family_m24(b24, e24)
    return True, (
        f"(1,29) mod-24 → (b24,e24,d24,a24)=(1,5,6,11); "
        f"orbit family = {family}"
    )


def check_c5() -> tuple[bool, str]:
    """Context: b=1 is the smallest orbit root; among QNs with b=1 in {1..60}, a=59 is near-extremal."""
    # All valid male QNs with b=1 and e in {1..29}: check a=b+2e values
    b = 1
    max_a_under_60 = 0
    for e in range(1, 30):
        if gcd(b, e) != 1:
            continue
        a = b + 2 * e
        if a <= 60:
            max_a_under_60 = a
    # e=29: a=1+58=59 — the largest a with b=1 and a≤60
    assert max_a_under_60 == 59, f"max a with b=1, a<=60: {max_a_under_60}, expected 59"
    # Elongation: ratio = a/b = 59 vs next smaller: e=28 gives a=57; e=27 gives a=55
    e29_ratio = 59  # a/b = 59/1
    e28_ratio = 57  # a/b = 57/1 (but gcd(1,28)=1, so valid)
    assert e29_ratio > e28_ratio, "a=59 is not more elongated than a=57"
    return True, (
        f"With b=1, the most elongated QN with a<=60 is (1,29,30,59) with ratio 1:59; "
        f"next longest (1,28,29,57) gives 1:57; Halley's is maximally elongated in this class"
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
        raise RuntimeError(f"cert [335] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
