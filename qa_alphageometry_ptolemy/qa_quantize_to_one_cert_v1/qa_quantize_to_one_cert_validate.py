#!/usr/bin/env python3
"""
QA Quantize-to-ONE Cert [321] -- validator

Primary source:
  Iverson, B. (1995) Quantum Arithmetic Book 3 (QA-3), ITAM, Portland,
    ISBN 1-883401-08-9, Chapter 3: "Quantizing to ONE".

"Quantize to ONE" is the application of the Quantize algorithm where
KO = THE ONE (the Myriad reference unit), so the input is a measurement
ratio JO/KO = b/a strictly less than 1. This chapter establishes the
ratio structure of the Cosmos orbit and the prime factorization bound
for the four BEDA roots.

Key results:
  - b/a = b/(b+2e) ∈ (0,1) for all Cosmos pairs (perigee < apogee)
  - The Satellite orbit has canonical ratio 1/3 exactly, excluded from Cosmos
  - Within a fixed d-class (same d=b+e), ratios are strictly increasing in b
  - The product b*e*d*a has at most 7 distinct prime factors (Iverson p.40);
    empirically at most 5 for mod-9 Cosmos

Five claims:

  C1  Cosmos ratio range: b/a ∈ (0,1) for all 72 Cosmos pairs; equivalently
      b < a = b+2e (since e >= 1). Zero exceptions across all 72 pairs.

  C2  Satellite canonical ratio 1/3: all 8 Satellite pairs (b=e) have
      b/a = b/(3b) = 1/3 exactly; and no Cosmos pair achieves ratio 1/3
      (since b/a = 1/3 implies b = e, which is the Satellite condition).

  C3  Within-d-class monotonicity: for fixed d = b+e, the ratio b/(2d-b)
      is strictly increasing in b; all pairs in the same d-class have
      distinct ratios. Verified for every d-class in the mod-9 Cosmos.

  C4  Seven-prime factorization bound: for all 72 Cosmos pairs, the count
      of distinct prime factors omega(b*e*d*a) <= 5 <= 7. Iverson: "there
      should be no more than seven successful divisions" (p.40). Empirical
      max = 5 (e.g. (5,8): b*e*d*a = 5*8*13*21 = 10920 = 2^3*3*5*7*13).

  C5  Theorem NT: b/a is the observer-layer ratio measurement; (b,e) is
      the QA causal layer. Given any Cosmos pair (b,e), the ratio r=Fraction(b,a)
      combined with the Myriad scale d uniquely reconstructs (b,e) via:
        b_rec = 2*d*r / (1+r)   [exact integer — uses midpoint identity a+b=2d]
        e_rec = d - b_rec
      Proof: 2dr/(1+r) = 2d*(b/a)/((a+b)/a) = 2d*b/(a+b) = 2d*b/(2d) = b.
      Round-trip exact for all 72 Cosmos pairs.
"""

from fractions import Fraction

QA_COMPLIANCE = (
    "cert_validator -- integer/Fraction arithmetic: d=b+e raw, a=b+2e raw; "
    "ratio=Fraction(b,a) observer projection; omega() distinct prime factor count; "
    "Theorem NT: ratio b/a is observer output; (b,e) is QA causal layer; "
    "round-trip (b,e)->ratio->( b,e) via Fraction arithmetic is the QA claim"
)

# ---------------------------------------------------------------------------
# QA primitives (mod-9, no-zero, A1-compliant state space)
# ---------------------------------------------------------------------------

M = 9


def beda(b: int, e: int) -> tuple[int, int, int, int]:
    """BEDA(b,e) = (b, e, d, a) with d=b+e, a=b+2e — both RAW (no mod)."""
    d = b + e
    a = b + 2 * e
    return (b, e, d, a)


def cosmos_states(m: int = M) -> list[tuple[int, int]]:
    """72 Cosmos pairs: b,e ∈ {1..m}, b≠e, excluding Singularity (m,m)."""
    return [
        (b, e)
        for b in range(1, m + 1)
        for e in range(1, m + 1)
        if b != e
    ]


def satellite_states(m: int = M) -> list[tuple[int, int]]:
    """8 Satellite pairs: b=e ∈ {1..m-1}."""
    return [(b, b) for b in range(1, m)]


def omega(n: int) -> int:
    """Number of distinct prime factors of n (omega function)."""
    if n <= 1:
        return 0
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------

def check_c1() -> list[str]:
    """C1: b/a ∈ (0,1) for all 72 Cosmos pairs."""
    failures = []
    for b, e in cosmos_states():
        _, _, d, a = beda(b, e)
        r = Fraction(b, a)
        if not (0 < r < 1):
            failures.append(f"({b},{e}): b/a = {r} not in (0,1)")
    return failures


def check_c2() -> list[str]:
    """C2: all 8 Satellite pairs give b/a = 1/3; no Cosmos pair does."""
    failures = []
    target = Fraction(1, 3)

    for b, e in satellite_states():
        _, _, d, a = beda(b, e)
        r = Fraction(b, a)
        if r != target:
            failures.append(f"Satellite ({b},{e}): b/a = {r} != 1/3")

    for b, e in cosmos_states():
        _, _, d, a = beda(b, e)
        r = Fraction(b, a)
        if r == target:
            failures.append(
                f"Cosmos ({b},{e}): b/a = 1/3 — Satellite ratio leaked into Cosmos"
            )
    return failures


def check_c3() -> list[str]:
    """C3: within each d-class, ratio b/(2d-b) strictly increases with b."""
    failures = []
    # Group Cosmos pairs by d = b+e
    from collections import defaultdict
    d_classes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for b, e in cosmos_states():
        d = b + e
        d_classes[d].append((b, e))

    for d_val, pairs in d_classes.items():
        if len(pairs) < 2:
            continue
        # Sort by b ascending
        pairs_sorted = sorted(pairs, key=lambda be: be[0])
        ratios = [Fraction(b, 2 * d_val - b) for b, _ in pairs_sorted]
        for i in range(len(ratios) - 1):
            if ratios[i] >= ratios[i + 1]:
                b_i = pairs_sorted[i][0]
                b_j = pairs_sorted[i + 1][0]
                failures.append(
                    f"d={d_val}: ratio({b_i})={ratios[i]} >= ratio({b_j})={ratios[i+1]}"
                    " — not strictly increasing"
                )
    return failures


def check_c4() -> list[str]:
    """C4: omega(b*e*d*a) <= 7 for all 72 Cosmos pairs; max is <= 5."""
    failures = []
    max_omega = 0
    max_pair = None
    for b, e in cosmos_states():
        _, _, d, a = beda(b, e)
        prod = b * e * d * a
        w = omega(prod)
        if w > 7:
            failures.append(
                f"({b},{e}): omega({b}*{e}*{d}*{a}={prod}) = {w} > 7"
            )
        if w > max_omega:
            max_omega = w
            max_pair = (b, e, d, a, prod)

    # Also assert the empirical max is <= 5 (stronger bound for documentation)
    if max_omega > 5:
        b, e, d, a, prod = max_pair
        failures.append(
            f"Empirical max omega = {max_omega} > 5 at ({b},{e}): "
            f"omega({prod}) — expected max 5 for mod-9 Cosmos"
        )
    return failures


def check_c5() -> list[str]:
    """C5: round-trip (b,e) -> r=Fraction(b,a) -> (b,e) via b=2dr/(1+r) lossless.

    Uses midpoint identity a+b=2d (cert [320] C1):
      2dr/(1+r) = 2d*(b/a)/((a+b)/a) = 2d*b/(2d) = b  [exact integer always]
    """
    failures = []
    for b, e in cosmos_states():
        _, _, d, a = beda(b, e)
        r = Fraction(b, a)
        # Reconstruct via the midpoint-identity formula
        b_frac = Fraction(2 * d, 1) * r / (1 + r)
        if b_frac.denominator != 1:
            failures.append(
                f"({b},{e}): 2d*r/(1+r) = {b_frac} — not integer"
            )
            continue
        b_rec = b_frac.numerator
        e_rec = d - b_rec
        if (b_rec, e_rec) != (b, e):
            failures.append(
                f"({b},{e}): reconstructed ({b_rec},{e_rec}) — mismatch"
            )
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("C1", "Cosmos ratio b/a ∈ (0,1) — 72 pairs",                  check_c1),
    ("C2", "Satellite canonical ratio 1/3, absent from Cosmos",      check_c2),
    ("C3", "Within-d-class ratio monotonicity",                       check_c3),
    ("C4", "Seven-prime bound omega(b*e*d*a) <= 5 <= 7",             check_c4),
    ("C5", "Round-trip ratio reconstruction lossless — 72 Cosmos",   check_c5),
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
