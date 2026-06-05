#!/usr/bin/env python3
"""
QA Harmonic Aliquot Structure Cert [322] -- validator

Primary source:
  Iverson, B. (1995) Quantum Arithmetic Book 3 (QA-3), ITAM, Portland,
    ISBN 1-883401-08-9, Chapter 4: "HARMONICS".

"HARMONICS comes into play. Harmonics brings the application of Quantum
Arithmetic to Music, and to Chemistry, to further them on the way to
becoming exact sciences." (p.42)

Chapter 4 reverses the conventional understanding: higher harmonics CREATE
the lower tone (not vice versa). Ben's theory of harmony is grounded in the
aliquot-part / unique-prime decomposition of QA states.

Theory of Harmony (Iverson p.52):
  "Harmonics, or harmony occurs between two dissimilar cycles of energy when
   both can be divided into similar aliquot parts having the same magnitude
   but different multitudes."

Formal translation:
  Two waves W1 = A·p1 and W2 = A·p2 are harmonic iff:
    - A is their common aliquot part (gcd(W1,W2) = A)
    - p1 = W1/A and p2 = W2/A are distinct primes
    - gcd(A, p1·p2) = 1  (unique primes not in the aliquot part)

In the QA mod-9 Cosmos, the d-value d = b+e is the "semi-major diameter"
(DO = d² in cert [320]). The 15 distinct Cosmos d-values {3,...,17} are the
natural domain for certifying Ben's harmonic aliquot structure.

Five claims:

  C1  Harmonic dyad count: Among the 15 distinct Cosmos d-values {3,...,17},
      there are exactly 20 harmonic dyads (unordered pairs {d1,d2} satisfying
      Ben's condition). All aliquot parts in these dyads are 7-smooth (prime
      factors ≤ 7), confirming that the shared aliquot structure lives in the
      {2,3,5,7} foundation.

  C2  Direction law — higher harmonics create lower tone: For every harmonic
      dyad {d1, d2} with d1 < d2, the unique prime p2 = d2/gcd > p1 = d1/gcd.
      The larger d (shorter wavelength, higher harmonic) carries the larger
      unique prime. When the higher harmonic cascades to form the lower tone,
      it is the large-unique-prime wave that creates the small-unique-prime
      wave. This is Ben's reversal: "The higher harmonics CREATE the lower
      tone." (p.43) Verified for all 20 dyads.

  C3  Aliquot spectrum: The aliquot parts appearing in Cosmos harmonic dyads
      are exactly {1, 2, 3, 5} — a subset of the 7-smooth primes. No aliquot
      part > 5 appears among the 20 dyads of the mod-9 Cosmos. The aliquot
      parts are the "gear pitch" (Iverson p.50): waves sharing the same
      aliquot mesh harmonically, like gears with the same tooth pitch.

  C4  5040 threshold — quantum flexibility boundary: 5040 = 2^4·3^2·5·7 has
      exactly ω(5040) = 4 distinct prime factors; it is 7! and the reference
      point for Ben's "Quantum Flexibility" threshold ("above 5040 or 10080,
      the correlating... is less definite", p.53). All 15 Cosmos d-values are
      ≤ 17 << 5040, confirming they are within the definite quantum
      configuration range. Cert [318] established 5040 as the synchronous
      harmonics ceiling; this cert establishes it as the aliquot stability
      boundary from the Chapter 4 direction.

  C5  Tonal identity d-values: Among the 15 Cosmos d-values, exactly 3 are
      primes > 7: {11, 13, 17}. These carry "tonal identity" — their unique
      prime exceeds the 7-smooth aliquot core. Iverson: "A wave gains its
      uniqueness through one prime number which is unique to that wave." (p.50)
      Each of {11, 13, 17} forms exactly 5 harmonic dyads (3 with prime
      d-values {3,5,7} + 2 cross-tonal with the other two high primes), giving
      them the richest harmonic connectivity in the mod-9 Cosmos.
"""

from math import gcd

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic: d=b+e raw for all BEDA; "
    "gcd(d1,d2)=aliquot part; unique_prime=d/aliquot observer-layer label; "
    "Theorem NT: the harmonic aliquot decomposition is an observer projection "
    "of the integer d-value; (b,e) remains the QA causal layer; "
    "all arithmetic is exact integer — no float, no mod-reduction"
)

# ---------------------------------------------------------------------------
# QA primitives (mod-9, no-zero)
# ---------------------------------------------------------------------------

M = 9


def cosmos_d_values(m: int = M) -> list[int]:
    """Distinct d=b+e values from the 72 Cosmos pairs."""
    return sorted({
        b + e
        for b in range(1, m + 1)
        for e in range(1, m + 1)
        if b != e
    })


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


def is_7smooth(n: int) -> bool:
    """True if all prime factors of n are ≤ 7."""
    for p in [2, 3, 5, 7]:
        while n % p == 0:
            n //= p
    return n == 1


def harmonic_dyads(d_vals: list[int]) -> list[tuple[int, int, int, int, int]]:
    """Return all harmonic dyads (d1, d2, A, p1, p2) with d1 < d2.

    Condition: A = gcd(d1,d2), p1 = d1/A and p2 = d2/A are distinct primes,
    and gcd(A, p1) = gcd(A, p2) = 1.
    """
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
    """C1: exactly 20 harmonic dyads; all aliquot parts are 7-smooth."""
    failures = []
    d_vals = cosmos_d_values()
    dyads = harmonic_dyads(d_vals)

    if len(dyads) != 20:
        failures.append(
            f"Expected 20 harmonic dyads, got {len(dyads)}: {dyads}"
        )

    for d1, d2, A, p1, p2 in dyads:
        if not is_7smooth(A):
            failures.append(
                f"({d1},{d2}): aliquot A={A} is NOT 7-smooth"
            )

    return failures


def check_c2() -> list[str]:
    """C2: direction law — d1<d2 implies p2>p1 for all harmonic dyads."""
    failures = []
    dyads = harmonic_dyads(cosmos_d_values())

    for d1, d2, A, p1, p2 in dyads:
        if d1 >= d2:
            failures.append(f"Ordering error: d1={d1} >= d2={d2}")
            continue
        if p2 <= p1:
            failures.append(
                f"({d1},{d2}): d1<d2 but p2={p2} <= p1={p1} "
                "— direction law violated"
            )
    return failures


def check_c3() -> list[str]:
    """C3: aliquot spectrum is exactly {1, 2, 3, 5}; no aliquot > 5."""
    failures = []
    dyads = harmonic_dyads(cosmos_d_values())
    aliquots = sorted(set(A for _, _, A, _, _ in dyads))
    expected = [1, 2, 3, 5]

    if aliquots != expected:
        failures.append(
            f"Aliquot spectrum: got {aliquots}, expected {expected}"
        )
    return failures


def check_c4() -> list[str]:
    """C4: ω(5040)=4; 5040=2^4·3^2·5·7; all Cosmos d-values < 5040."""
    failures = []
    THRESHOLD = 5040

    # Factor 5040
    def distinct_prime_factors(n: int) -> list[int]:
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    dpf = distinct_prime_factors(THRESHOLD)
    if dpf != [2, 3, 5, 7]:
        failures.append(f"5040 prime factors: {dpf}, expected [2,3,5,7]")
    if len(dpf) != 4:
        failures.append(f"ω(5040) = {len(dpf)}, expected 4")

    # Verify 5040 = 2^4 * 3^2 * 5 * 7
    expected_val = (2 ** 4) * (3 ** 2) * 5 * 7
    if expected_val != THRESHOLD:
        failures.append(f"2^4·3^2·5·7 = {expected_val} ≠ 5040")

    # All Cosmos d-values < 5040
    d_vals = cosmos_d_values()
    violations = [d for d in d_vals if d >= THRESHOLD]
    if violations:
        failures.append(
            f"Cosmos d-values >= 5040: {violations} — outside definite range"
        )

    return failures


def check_c5() -> list[str]:
    """C5: exactly 3 Cosmos d-values are primes > 7: {11, 13, 17}."""
    failures = []
    d_vals = cosmos_d_values()
    tonal = [d for d in d_vals if is_prime(d) and d > 7]
    expected = [11, 13, 17]

    if tonal != expected:
        failures.append(
            f"Tonal identity d-values: got {tonal}, expected {expected}"
        )

    # Each tonal prime forms exactly 5 dyads:
    # 3 with prime d-values {3,5,7} + 2 cross-tonal with the other two
    dyads = harmonic_dyads(d_vals)
    for t in expected:
        count = sum(
            1 for d1, d2, A, p1, p2 in dyads
            if d1 == t or d2 == t
        )
        if count != 5:
            failures.append(
                f"d={t}: forms {count} harmonic dyads, expected 5"
            )

    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("C1", "20 harmonic dyads; aliquot parts all 7-smooth",                check_c1),
    ("C2", "Direction law: higher d → larger unique prime (higher creates lower)", check_c2),
    ("C3", "Aliquot spectrum = {1,2,3,5} — no aliquot > 5 in mod-9 Cosmos", check_c3),
    ("C4", "5040 = 2^4·3^2·5·7, ω=4; all Cosmos d < 5040",                check_c4),
    ("C5", "Tonal identity d-values {11,13,17} each form 5 harmonic dyads", check_c5),
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
