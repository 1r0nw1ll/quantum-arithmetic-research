# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (prime residue symmetry mod 30/60; "
    "cycle coincidence LCM for coprime periods); no QA state evolution; "
    "Theorem NT: 'wave', 'cycle', 'symmetry about half-cycle' are observer-layer labels "
    "on integer modular arithmetic structure; all arithmetic exact integer, no float"
)

"""
Cert [344] — QA Prime Residue Symmetry and Cycle Coincidence

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol II, Chapters XII-XIII pp.26-50
  Ch.XII (p.30): "1+29=30, 7+23=30, 11+19=30, and 13+17=30. This set leaves out
   the prime numbers 2, 3, and 5, in that 2x3x5=30, and the remaining prime numbers
   are set symmetrically within the number system."
  Ch.XII (p.30): "1+59=60, 7+53=60, 11+49=60, 13+47=60, 17+43=60, 19+41=60,
   23+37=60, and 29+31=60." [for 3x4x5=60]
  Ch.XIII (p.44): "The cycles of prime numbers, when combined will make a cycle
   which is equal to their product."
  Ch.XIII (p.44): "at those units on the baseline where none of the cycles begin or
   complete, are all prime numbers."
  Ch.XIII (p.44): "The 3 and 5 coincide every 15 units. The 3 and 7 coincide every
   21 units. And the 5 and 7 coincide every 35 units."

Five claims:
  C1: φ(30)=8: exactly 8 residues mod 30 coprime to 30: {1,7,11,13,17,19,23,29}
  C2: The 8 coprime residues mod 30 pair symmetrically about 15: each pair sums to 30
  C3: For coprime primes p,q: lcm(p,q)=p*q (cycle period = product for coprime primes)
  C4: lcm(2,3,5)=30; lcm(3,5,7)=105; these equal their products (Iverson's cycle claim)
  C5: φ(60)=16: 16 residues mod 60 coprime to 60, forming 8 pairs each summing to 60
"""

from math import gcd, lcm


def check_c1() -> tuple[bool, str]:
    """φ(30)=8: exactly 8 residues mod 30 coprime to 30: {1,7,11,13,17,19,23,29}."""
    coprime_30 = [r for r in range(1, 30) if gcd(r, 30) == 1]
    assert len(coprime_30) == 8, f"φ(30)={len(coprime_30)}, expected 8"
    expected = {1, 7, 11, 13, 17, 19, 23, 29}
    assert set(coprime_30) == expected, (
        f"Coprime residues mod 30: {set(coprime_30)} != {expected}"
    )
    # All primes > 5 that are < 30
    primes_lt_30_gt_5 = [n for n in range(7, 30)
                         if all(n % d != 0 for d in range(2, n))]
    for p in primes_lt_30_gt_5:
        assert gcd(p, 30) == 1, f"Prime {p} is not coprime to 30"
    return True, (
        f"φ(30)=8; coprime residues={{1,7,11,13,17,19,23,29}}; "
        f"all primes 7..29 {primes_lt_30_gt_5} are in coprime class"
    )


def check_c2() -> tuple[bool, str]:
    """The 8 coprime residues mod 30 pair symmetrically about 15: each pair sums to 30."""
    # Pairs from Iverson: 1+29=30, 7+23=30, 11+19=30, 13+17=30
    coprime_30 = sorted([r for r in range(1, 30) if gcd(r, 30) == 1])
    pairs = [(coprime_30[i], coprime_30[7 - i]) for i in range(4)]
    iverson_pairs = [(1, 29), (7, 23), (11, 19), (13, 17)]
    assert pairs == iverson_pairs, f"Pairs {pairs} != Iverson's {iverson_pairs}"
    for a, b in pairs:
        assert a + b == 30, f"Pair ({a},{b}): sum={a+b} != 30"
    # Also verify: r and (30-r) both coprime to 30 iff r coprime to 30
    for r in range(1, 30):
        if gcd(r, 30) == 1:
            assert gcd(30 - r, 30) == 1, f"{30-r} not coprime to 30"
    return True, (
        "Coprime residues mod 30 pair symmetrically about 15: "
        "1+29=30, 7+23=30, 11+19=30, 13+17=30"
    )


def check_c3() -> tuple[bool, str]:
    """For coprime primes p,q: lcm(p,q)=p*q (cycle period = product)."""
    # Iverson: lcm(2,3)=6, lcm(2,5)=10, lcm(3,5)=15, lcm(3,7)=21, lcm(5,7)=35
    pairs = [(2, 3, 6), (2, 5, 10), (3, 5, 15), (3, 7, 21), (5, 7, 35), (2, 7, 14)]
    for p, q, expected in pairs:
        assert gcd(p, q) == 1, f"gcd({p},{q}) != 1"
        computed = lcm(p, q)
        assert computed == p * q, f"lcm({p},{q})={computed} != {p}*{q}={p*q}"
        assert computed == expected, f"lcm({p},{q})={computed} != expected {expected}"
    # General: for any two distinct primes p,q: lcm(p,q)=p*q
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    count = 0
    for i, p in enumerate(small_primes):
        for q in small_primes[i+1:]:
            assert lcm(p, q) == p * q, f"lcm({p},{q}) != p*q"
            count += 1
    return True, (
        f"lcm(p,q)=p*q for all coprime prime pairs including Iverson's examples; "
        f"verified {len(pairs)} named pairs and {count} prime pairs"
    )


def check_c4() -> tuple[bool, str]:
    """lcm(2,3,5)=30; lcm(3,5,7)=105; overall cycle = product of coprime prime periods."""
    # Iverson: "Their combined cycle would be the product of the three numbers"
    triples = [(2, 3, 5, 30), (3, 5, 7, 105), (2, 3, 7, 42), (2, 5, 7, 70)]
    for p, q, r, expected in triples:
        assert gcd(p, q) == 1 and gcd(p, r) == 1 and gcd(q, r) == 1, (
            f"Not all coprime: {p},{q},{r}"
        )
        computed = lcm(lcm(p, q), r)
        assert computed == p * q * r, f"lcm({p},{q},{r})={computed} != {p*q*r}"
        assert computed == expected, f"lcm({p},{q},{r})={computed} != expected {expected}"
    return True, (
        "lcm(2,3,5)=30, lcm(3,5,7)=105; overall cycle = product of coprime prime factors; "
        "verified for Iverson's named triples and 2 additional"
    )


def check_c5() -> tuple[bool, str]:
    """φ(60)=16: 16 residues mod 60 coprime to 60, forming 8 pairs summing to 60."""
    # Iverson (Ch.XII): "1+59=60, 7+53=60, 11+49=60, ..."
    coprime_60 = sorted([r for r in range(1, 60) if gcd(r, 60) == 1])
    assert len(coprime_60) == 16, f"φ(60)={len(coprime_60)}, expected 16"
    # Check pairing: r and (60-r) both coprime to 60
    pairs = []
    for r in coprime_60:
        companion = 60 - r
        assert gcd(companion, 60) == 1, f"{companion} not coprime to 60"
        if r < companion:
            pairs.append((r, companion))
    assert len(pairs) == 8, f"Expected 8 pairs, got {len(pairs)}"
    for a, b in pairs:
        assert a + b == 60, f"Pair ({a},{b}): sum={a+b} != 60"
    # Verify Iverson's specific pairs
    iverson_pairs_60 = [(1,59), (7,53), (11,49), (13,47), (17,43), (19,41), (23,37), (29,31)]
    assert set(pairs) == set(iverson_pairs_60), (
        f"Pairs {pairs} != Iverson's {iverson_pairs_60}"
    )
    return True, (
        f"φ(60)=16; 16 coprime residues form 8 pairs each summing to 60: "
        "1+59=60, 7+53=60, 11+49=60, 13+47=60, 17+43=60, 19+41=60, 23+37=60, 29+31=60"
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
        raise RuntimeError(f"cert [344] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
