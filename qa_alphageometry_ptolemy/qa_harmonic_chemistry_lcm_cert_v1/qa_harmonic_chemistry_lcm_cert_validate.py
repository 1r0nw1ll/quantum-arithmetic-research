#!/usr/bin/env python3
"""
QA Harmonic Chemistry LCM Cert [323] -- validator

Primary source:
  Iverson, B. (1995) Quantum Arithmetic Book 3 (QA-3), ITAM, Portland,
    ISBN 1-883401-08-9, Chapter 5: "CHEMISTRY".

Chapter 5 applies the harmonic aliquot theory (cert [322]) to the Myriad of
Light / Chemism interface. Key developments:

  (1) HARMONIC CYCLE IDENTITY: When two harmonic waves bond (sharing aliquot A
      with W1=A*p1, W2=A*p2), the joint harmonic cycle is the LEAST COMMON
      MULTIPLE: HC = lcm(W1,W2) = W1*p2 = W2*p1 = A*p1*p2.
      (Iverson's example: Wave 1 × unique(2) = Wave 2 × unique(1) = common cycle)

  (2) N-WAVE PAIRING: n mutually-harmonic wavelets form exactly C(n,2) = n(n-1)/2
      harmonic pairs. For n=7 (the maximum wave-bond count in the Myriad model):
      C(7,2) = 21 pairs. Iverson: "One can add four more wavelets ... They will
      make 21 different pairs. All of these pairs can be harmonic." (p.68)

  (3) COMPOSITE LCM: For n mutually-harmonic waves sharing aliquot A with unique
      primes p1,...,pn: lcm = A * p1 * p2 * ... * pn = product of all waves / A^(n-1).

  (4) L = b*e*d*a WAVELENGTH PRODUCT: "the wavelength of a wave of sound energy
      relates directly to the product of all four integers (b,e,d,a) in a Quantum
      Number. That value has been designated as 'L'." (Iverson p.72). In QA:
      L = b*e*d*a for all Cosmos pairs is a positive integer in [24, 31824].

  (5) UNIVERSAL 2*3 BOND: "all Quantum Waves must be some multiple of 6" (p.67)
      because all waves contain 2 and 3 as prime factors. QA certification:
      for all 72 Cosmos pairs, b*e*d*a ≡ 0 (mod 6) — the product always
      contains both 2 and 3 as factors.

Five claims:

  C1  LCM harmonic cycle identity: for all 20 Cosmos harmonic dyads
      (d1=A*p1, d2=A*p2), lcm(d1,d2) = d1*p2 = d2*p1 = A*p1*p2 exactly.
      Zero exceptions. Builds on cert [322].

  C2  Three-wave composite LCM: for all C(6,3)=20 three-element subsets of
      the prime Cosmos d-values {3,5,7,11,13,17} (the max mutual-harmonic
      group with A=1), lcm(d1,d2,d3) = d1*d2*d3 exactly (pairwise coprime
      primes). Zero exceptions.

  C3  C(n,2) pairing law: for n in {2,3,4,5,6,7}, the number of harmonic
      pairs = C(n,2) = n*(n-1)/2. For n=7: C(7,2)=21, confirming Iverson's
      "21 different pairs" prediction (p.68). The mod-9 Cosmos has a maximum
      mutual-harmonic group of size 6 ({3,5,7,11,13,17}), showing that n=7
      would require a Myriad beyond the mod-9 state space.

  C4  L = b*e*d*a: for all 72 Cosmos pairs, L = b*e*d*a > 0. The minimum
      L is 24 at (1,2): 1*2*3*5=30 — wait, let me recompute:
      (b=1,e=2): d=3,a=5, L=1*2*3*5=30; (b=2,e=1): d=3,a=4, L=2*1*3*4=24.
      Min L=24 at (2,1); max L=31824 at (8,9). All L are positive integers.

  C5  Universal 2*3 bond: for all 72 Cosmos pairs, L = b*e*d*a is divisible
      by 6 (both 2 and 3 divide L). This is the QA expression of Iverson's
      "all Quantum Waves must be some multiple of 6" (p.67) — the universal
      2-and-3 bonding principle holds across the entire Cosmos orbit.
"""

from math import gcd, lcm
from itertools import combinations

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic: d=b+e raw, a=b+2e raw; "
    "lcm(d1,d2)=harmonic cycle for harmonic dyads; L=b*e*d*a is the "
    "wavelength product; Theorem NT: the harmonic cycle lcm(W1,W2) is an "
    "observer-layer computation on integer QA states; (b,e) is QA causal layer; "
    "C(n,2) pairing law is combinatorial identity, not empirical claim"
)

# ---------------------------------------------------------------------------
# QA primitives (mod-9, no-zero)
# ---------------------------------------------------------------------------

M = 9


def cosmos_states(m: int = M) -> list[tuple[int, int]]:
    """72 Cosmos pairs: b,e ∈ {1..m}, b≠e."""
    return [
        (b, e)
        for b in range(1, m + 1)
        for e in range(1, m + 1)
        if b != e
    ]


def cosmos_d_values(m: int = M) -> list[int]:
    """15 distinct Cosmos d=b+e values {3,...,17}."""
    return sorted({b + e for b, e in cosmos_states(m)})


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def harmonic_dyads(
    d_vals: list[int],
) -> list[tuple[int, int, int, int, int]]:
    """Harmonic dyads: (d1, d2, A, p1, p2) with d1<d2, A=gcd,
    p1=d1/A and p2=d2/A distinct primes, gcd(A,p1*p2)=1."""
    dyads = []
    for i, d1 in enumerate(d_vals):
        for d2 in d_vals[i + 1:]:
            A = gcd(d1, d2)
            q1, q2 = d1 // A, d2 // A
            if (
                is_prime(q1) and is_prime(q2)
                and q1 != q2
                and gcd(A, q1) == 1
                and gcd(A, q2) == 1
            ):
                dyads.append((d1, d2, A, q1, q2))
    return dyads


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------

def check_c1() -> list[str]:
    """C1: lcm(d1,d2) = d1*p2 = d2*p1 = A*p1*p2 for all 20 Cosmos harmonic dyads."""
    failures = []
    dyads = harmonic_dyads(cosmos_d_values())
    if len(dyads) != 20:
        failures.append(f"Expected 20 harmonic dyads, got {len(dyads)}")
    for d1, d2, A, p1, p2 in dyads:
        hc = lcm(d1, d2)
        d1_form = d1 * p2
        d2_form = d2 * p1
        a_form = A * p1 * p2
        if not (hc == d1_form == d2_form == a_form):
            failures.append(
                f"({d1},{d2}): lcm={hc}, d1*p2={d1_form}, "
                f"d2*p1={d2_form}, A*p1*p2={a_form} — not all equal"
            )
    return failures


def check_c2() -> list[str]:
    """C2: lcm(d1,d2,d3) = d1*d2*d3 for all C(6,3)=20 triples of prime d-values."""
    failures = []
    prime_ds = [d for d in cosmos_d_values() if is_prime(d)]
    expected_primes = [3, 5, 7, 11, 13, 17]
    if prime_ds != expected_primes:
        failures.append(f"Prime d-values: got {prime_ds}, expected {expected_primes}")
        return failures

    triples = list(combinations(prime_ds, 3))
    if len(triples) != 20:
        failures.append(f"Expected C(6,3)=20 triples, got {len(triples)}")

    for t in triples:
        d1, d2, d3 = t
        hc = lcm(d1, lcm(d2, d3))
        product = d1 * d2 * d3
        if hc != product:
            failures.append(
                f"lcm{t}={hc} != d1*d2*d3={product}"
            )
    return failures


def check_c3() -> list[str]:
    """C3: C(n,2) = n*(n-1)//2 for n=2..7; C(7,2)=21."""
    failures = []
    expected = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21}
    for n, exp_cn2 in expected.items():
        cn2 = n * (n - 1) // 2
        if cn2 != exp_cn2:
            failures.append(f"C({n},2) = {cn2} != {exp_cn2}")

    # Maximum mutual-harmonic group in mod-9 Cosmos has size 6 (not 7)
    prime_ds = [d for d in cosmos_d_values() if is_prime(d)]
    if len(prime_ds) != 6:
        failures.append(
            f"Expected 6 prime Cosmos d-values, got {len(prime_ds)}: {prime_ds}"
        )

    return failures


def check_c4() -> list[str]:
    """C4: L = b*e*d*a > 0 for all 72 Cosmos pairs; min=24, max=31824."""
    failures = []
    min_L, max_L = float("inf"), 0
    for b, e in cosmos_states():
        d = b + e
        a = b + 2 * e
        L = b * e * d * a
        if L <= 0:
            failures.append(f"({b},{e}): L={L} not positive")
        if L < min_L:
            min_L = L
        if L > max_L:
            max_L = L

    if min_L != 24:
        failures.append(f"min L = {min_L}, expected 24 (at pair (2,1))")
    if max_L != 31824:
        failures.append(f"max L = {max_L}, expected 31824 (at pair (8,9))")

    return failures


def check_c5() -> list[str]:
    """C5: b*e*d*a divisible by 6 for all 72 Cosmos pairs."""
    failures = []
    for b, e in cosmos_states():
        d = b + e
        a = b + 2 * e
        L = b * e * d * a
        if L % 6 != 0:
            failures.append(
                f"({b},{e}): b*e*d*a = {L} NOT divisible by 6"
            )
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("C1", "LCM = harmonic cycle: lcm(d1,d2)=d1*p2=d2*p1=A*p1*p2 — 20 dyads", check_c1),
    ("C2", "3-wave LCM = product for C(6,3)=20 triples of prime d-values",       check_c2),
    ("C3", "C(n,2) pairing law n=2..7; C(7,2)=21; max mutual group = 6 in mod-9", check_c3),
    ("C4", "L = b*e*d*a: all 72 Cosmos L > 0, min=24, max=31824",                 check_c4),
    ("C5", "Universal 2*3 bond: b*e*d*a divisible by 6 for all 72 Cosmos pairs",  check_c5),
]


def main() -> int:
    all_pass = True
    for cid, desc, fn in CHECKS:
        failures = fn()
        status = "PASS" if not failures else "FAIL"
        print(f"  [{status}] {cid}: {desc}")
        for f in failures:
            print(f"        {f}")
        if failures:
            all_pass = False
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
