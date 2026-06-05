# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-2 Ch.XII Prime Number Count: "
    "coprime-to-30 symmetry 8 elements 4 pairs; coprime-to-60 symmetry 16 elements 8 pairs (incl 49=7^2); "
    "pairing identity gcd(M-n,M)=gcd(n,M); primes in (5,30) all coprime to 30; "
    "49 legitimately in coprime-to-60 bracket); "
    "Theorem NT: prime sieve illustration is an observer projection; QA layer is gcd arithmetic"
)

"""
Cert [367] — QA Pyth-2 Prime Number Symmetry (Ch.XII)

Source: Iverson (1993) Pythagorean Arithmetic Vol II, Chapter XII pp.28-37
  p.29-30: 'If we start at both ends of 1, 7, 11, 13, 17, 19, 23, and 29
            and add the matched numbers together we come up with 1+29=30,
            7+23=30, 11+19=30, and 13+17=30.'
  p.30: '3 x 4 x 5 = 60 ... 1+59=60, 7+53=60, 11+49=60, 13+47=60,
         17+43=60, 19+41=60, 23+37=60, and 29+31=60.'
  p.30: 'The number 49 is included, but as previously explained, it is a
         prime number because it has a single factor of 7.'

Five claims:
  C1: Coprime-to-30 bracket: {n ∈ [1,29]: gcd(n,30)=1} = {1,7,11,13,17,19,23,29}
      φ(30)=8; these 8 elements form exactly 4 pairs, each summing to 30.
      proof: gcd(n,30)=1 and gcd(30-n,30)=gcd(-n,30)=gcd(n,30)=1 → pairing always works;
             no element equals 15 (gcd(15,30)=15≠1) so n≠30-n → strict pairs ✓
  C2: Coprime-to-60 bracket: {n ∈ [1,59]: gcd(n,60)=1} has φ(60)=16 elements;
      these form 8 pairs each summing to 60; 49=7² is in this bracket.
      proof: gcd(49,60)=gcd(7²,2²·3·5)=1 since 7∤60; same pairing argument ✓
  C3: Pairing identity: for any M≥2 and n with gcd(n,M)=1: gcd(M-n,M)=1
      proof: gcd(M-n,M)=gcd(-n,M)=gcd(n,M)=1 (gcd(−n,M)=gcd(n,M) always) ✓
  C4: All primes p with 5 < p < 30 satisfy gcd(p,30)=1 (so they appear in the bracket)
      proof: any prime p>5 shares no factor with 30=2·3·5 (p≠2,3,5) → gcd(p,30)=1 ✓
  C5: The Koenig value I(1,5)=49=7² is in the coprime-to-60 bracket;
      gcd(49,60)=1; its bracket partner is 60-49=11 (prime), and gcd(11,60)=1 ✓
"""

from math import gcd


def check_c1() -> tuple[bool, str]:
    """Coprime-to-30 bracket: 8 elements in 4 pairs each summing to 30."""
    M = 30
    bracket = [n for n in range(1, M) if gcd(n, M) == 1]
    assert bracket == [1, 7, 11, 13, 17, 19, 23, 29], f"Unexpected bracket: {bracket}"
    assert len(bracket) == 8
    # Euler's totient φ(30) = φ(2)·φ(3)·φ(5) = 1·2·4 = 8
    phi_30 = len(bracket)
    assert phi_30 == 8
    # Form pairs
    pairs = []
    seen = set()
    for n in bracket:
        if n in seen:
            continue
        partner = M - n
        assert partner in bracket, f"{M-n} not in bracket (partner of {n})"
        assert n + partner == M
        pairs.append((n, partner))
        seen.add(n)
        seen.add(partner)
    assert len(pairs) == 4, f"Expected 4 pairs, got {len(pairs)}"
    iverson_pairs = [(1, 29), (7, 23), (11, 19), (13, 17)]
    for p1, p2 in iverson_pairs:
        assert p1 + p2 == M, f"{p1}+{p2} != {M}"
        assert gcd(p1, M) == 1 and gcd(p2, M) == 1
    return True, (
        f"Coprime-to-30 bracket = {bracket}; φ(30)=8; "
        f"4 pairs: {iverson_pairs}; all sum to 30; "
        f"proof: gcd(30-n,30)=gcd(n,30)=1 → each n pairs with 30-n ✓"
    )


def check_c2() -> tuple[bool, str]:
    """Coprime-to-60 bracket: 16 elements in 8 pairs each summing to 60; 49∈bracket."""
    M = 60
    bracket = [n for n in range(1, M) if gcd(n, M) == 1]
    # φ(60) = φ(4)·φ(3)·φ(5) = 2·2·4 = 16
    assert len(bracket) == 16, f"Expected 16 elements, got {len(bracket)}: {bracket}"
    # 49 must be in the bracket
    assert 49 in bracket, f"49 not in coprime-to-60 bracket"
    assert gcd(49, 60) == 1, f"gcd(49,60)={gcd(49,60)} != 1"
    # Form pairs
    pairs = []
    seen = set()
    for n in bracket:
        if n in seen:
            continue
        partner = M - n
        assert partner in bracket, f"{M-n} not in bracket (partner of {n})"
        assert n + partner == M
        pairs.append((n, partner))
        seen.add(n)
        seen.add(partner)
    assert len(pairs) == 8
    iverson_pairs_60 = [(1, 59), (7, 53), (11, 49), (13, 47),
                        (17, 43), (19, 41), (23, 37), (29, 31)]
    for p1, p2 in iverson_pairs_60:
        assert p1 + p2 == M, f"{p1}+{p2} != {M}"
        assert gcd(p1, M) == 1 and gcd(p2, M) == 1
    return True, (
        f"Coprime-to-60 bracket = {bracket}; φ(60)=16; "
        f"8 pairs summing to 60: {iverson_pairs_60}; "
        f"49=7² ∈ bracket (gcd(49,60)={gcd(49,60)}=1); "
        f"49's partner = 60-49=11 (prime, gcd(11,60)={gcd(11,60)}=1) ✓"
    )


def check_c3() -> tuple[bool, str]:
    """Pairing identity: gcd(M-n, M) = gcd(n, M) for all M, n."""
    count = 0
    for M in range(2, 100):
        for n in range(1, M):
            assert gcd(M - n, M) == gcd(n, M), (
                f"gcd(M-n,M)={gcd(M-n,M)} != gcd(n,M)={gcd(n,M)} at M={M},n={n}"
            )
            count += 1
    return True, (
        f"gcd(M-n,M)=gcd(n,M) verified for all M∈[2,99] and n∈[1,M-1] ({count} cases); "
        f"proof: gcd(M-n,M)=gcd(-n,M)=gcd(n,M) (gcd ignores sign); "
        f"corollary: n coprime to M iff M-n coprime to M ✓"
    )


def check_c4() -> tuple[bool, str]:
    """All primes p with 5 < p < 30 satisfy gcd(p, 30) = 1."""
    def is_prime(n):
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

    primes_in_range = [p for p in range(6, 30) if is_prime(p)]
    assert primes_in_range == [7, 11, 13, 17, 19, 23, 29], f"Unexpected: {primes_in_range}"
    for p in primes_in_range:
        g = gcd(p, 30)
        assert g == 1, f"gcd({p},30)={g} != 1"
    # Verify they're all in the coprime-to-30 bracket
    bracket_30 = [n for n in range(1, 30) if gcd(n, 30) == 1]
    for p in primes_in_range:
        assert p in bracket_30, f"prime {p} not in coprime-to-30 bracket"
    return True, (
        f"Primes in (5,30): {primes_in_range}; all have gcd(p,30)=1; "
        f"all appear in coprime-to-30 bracket {bracket_30}; "
        f"proof: prime p>5 → p∉{{2,3,5}} → gcd(p,2)=gcd(p,3)=gcd(p,5)=1 → gcd(p,30)=1 ✓"
    )


def check_c5() -> tuple[bool, str]:
    """49=7²=I(1,5) is legitimately in the coprime-to-60 bracket; its partner is 11."""
    # Verify I(1,5)=49
    b, e = 1, 5
    d = b + e   # 6
    a = d + e   # 11
    C = 2 * d * e  # 60
    F = a * b      # 11
    I_val = abs(C - F)  # 49
    assert I_val == 49
    # Verify gcd(49, 60)=1
    assert gcd(49, 60) == 1
    # Partner: 60-49=11
    partner = 60 - I_val
    assert partner == 11
    assert gcd(11, 60) == 1
    # Verify 11 is prime
    assert all(11 % i != 0 for i in range(2, 11))
    # Verify the full coprime-to-60 bracket contains both 49 and 11
    bracket_60 = [n for n in range(1, 60) if gcd(n, 60) == 1]
    assert 49 in bracket_60 and 11 in bracket_60
    # Verify that 49 is in the (11,49) pair from Iverson's list
    assert 11 + 49 == 60
    # Verify the prime power structure: 49=7², single prime factor
    assert 49 == 7 * 7
    assert gcd(49 // 7, 7) == 7  # confirm 49/7=7 is still 7
    return True, (
        f"I(1,5)=|60-11|=49=7²; gcd(49,60)=1; "
        f"49 ∈ coprime-to-60 bracket (position {bracket_60.index(49)+1} of 16); "
        f"partner: 60-49=11 (prime, gcd(11,60)=1); "
        f"11+49=60 ✓ (Iverson's third pair in the 60-bracket); "
        f"49=7² is QA-functionally-prime with gcd(49,60)=1 ✓"
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
        raise RuntimeError(f"cert [367] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
