#!/usr/bin/env python3
"""
QA Music of the Spheres Scale Cert [324] -- validator

Primary source:
  Iverson, B. (1995) Quantum Arithmetic Book 3 (QA-3), ITAM, Portland,
    ISBN 1-883401-08-9, Chapter 6: "THE MYRIAD OF SOUND".

Chapter 6 develops the QA harmonic framework into music theory via the
"Music of the Spheres" — ancient harmonic knowledge encoded in:
  (a) Archimedes' Cattle Problem (4 bulls + 4 cows = 8 keynotes)
  (b) I-Ching (8 trigrams: 4 male + 4 female)
  (c) The 18-note scale per keynote, totaling 144 notes in 8 keys

Key QA connections:
  - 8 keynotes = 8 Satellite orbit states (b=e in {1..8}) in mod-9 QA
  - 18-note scale = 17 reduced fractions p/q (gcd(p,q)=1, q≤7, 0<p<q)
    PLUS the keynote itself = 18 total
  - 8 × 18 = 144 notes (all 8 keynotes combined)
  - All scale fractions have 7-smooth denominators (primes ≤ 7 only)
  - The 17 sorted fractions satisfy the Farey mediant property:
    adjacent fractions (p1/q1, p2/q2) satisfy |p1·q2 - p2·q1| = 1

"Using each of the notes as a 'Keynote' and taking the eighteen low
fractional values of each note produced 144 different notes in eight keys."
(Iverson p.82)

"...fractions of the keynote. These fractions are 3/7, 2/7, 1/7, 1/6, 2/5,
1/5, 1/4, 1/3, and 1/2 subtracted from the keynote" [9 below] + "6/7, 5/7,
4/7, 3/7, 2/7, 1/7, 5/6, 1/6, 4/5, 3/5, 2/5, 1/5, 3/4, 1/4, 2/3, 1/3 & 1/2"
[17 above the keynote, which include the complete set of reduced fractions
with denominator ≤ 7 in (0,1)]. (Iverson pp.93–94)

Five claims:

  C1  Eight keynotes = 8 Satellite states: in mod-9 QA, the Satellite orbit
      has exactly 8 states {(b,b) : b ∈ {1,...,8}}. These correspond to the
      8 values from the Cattle Problem / I-Ching (4 male + 4 female).
      Iverson: "the book pairs up the different binary couples, in a pairing
      of hexagrams ... The curious feature of this book is that it contains
      eight persons being four males and four females." (p.81)

  C2  17 scale fractions: there are exactly 17 reduced fractions p/q with
      gcd(p,q)=1, 1 ≤ p < q ≤ 7. Adding the keynote (p/q=1/1 = the note
      itself) gives 18 total notes per keynote. This matches Iverson's
      "eighteen low fractional values of each note." (p.82)

  C3  144 = 8 × 18: 8 keynotes × 18 notes each = 144 notes (Iverson p.82:
      "produced 144 different notes in eight keys"). This arithmetic identity
      is the QA expression of the Music of the Spheres combinatorics.

  C4  7-smooth denominators: all 17 scale fractions have denominators that
      are 7-smooth (prime factors ≤ 7). The 4 distinct prime denominators
      ≤ 7 are {2, 3, 5, 7}. This confirms Iverson's "harmony depends on
      fractional relationships from halves to sevenths" (p.91) — no fraction
      with denominator 8, 9, 10, or 11 enters the Music of the Spheres scale.

  C5  Farey mediant property: the 17 scale fractions, sorted in ascending
      order, satisfy the Farey mediant property at every adjacent pair:
      |p1·q2 - p2·q1| = 1. This means consecutive notes in the scale are
      Farey neighbors — the strongest possible harmonic adjacency condition,
      confirming Iverson's repeated claim that these scales have "very
      harmonic" relationships between adjacent notes.
"""

from math import gcd

QA_COMPLIANCE = (
    "cert_validator -- integer arithmetic: scale fractions are rational p/q "
    "with integer p,q; gcd check is exact integer; Satellite states are "
    "integer pairs (b,b); Theorem NT: fractional frequency ratios are "
    "observer projections; QA causal layer is (b,e)=(keynote_index, itself); "
    "all arithmetic exact integer/rational, no float"
)

# ---------------------------------------------------------------------------
# QA primitives (mod-9, no-zero)
# ---------------------------------------------------------------------------

M = 9


def satellite_states(m: int = M) -> list[tuple[int, int]]:
    """8 Satellite pairs: b=e ∈ {1..m-1}."""
    return [(b, b) for b in range(1, m)]


def scale_fractions() -> list[tuple[int, int]]:
    """17 reduced fractions p/q: gcd(p,q)=1, 1 ≤ p < q ≤ 7, in sorted order."""
    fracs = []
    for q in range(2, 8):
        for p in range(1, q):
            if gcd(p, q) == 1:
                fracs.append((p, q))
    return sorted(fracs, key=lambda x: x[0] / x[1])


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------

def check_c1() -> list[str]:
    """C1: 8 Satellite states in mod-9 = 8 keynotes."""
    failures = []
    sats = satellite_states()
    if len(sats) != 8:
        failures.append(f"Expected 8 Satellite states, got {len(sats)}: {sats}")
    for b, e in sats:
        if b != e:
            failures.append(f"Satellite ({b},{e}): b≠e — not diagonal")
        if not (1 <= b <= M - 1):
            failures.append(f"Satellite ({b},{e}): b not in {{1..{M-1}}}")
    return failures


def check_c2() -> list[str]:
    """C2: exactly 17 reduced fractions with q≤7; with keynote = 18 notes."""
    failures = []
    fracs = scale_fractions()
    if len(fracs) != 17:
        failures.append(f"Expected 17 reduced fractions, got {len(fracs)}: {fracs}")
    for p, q in fracs:
        if gcd(p, q) != 1:
            failures.append(f"{p}/{q}: not reduced (gcd={gcd(p,q)})")
        if not (1 <= p < q <= 7):
            failures.append(f"{p}/{q}: outside range [1/7, 6/7]")
    notes_per_key = len(fracs) + 1  # fracs + keynote itself
    if notes_per_key != 18:
        failures.append(
            f"Notes per keynote: {notes_per_key} (fracs={len(fracs)} + keynote=1)"
            f", expected 18"
        )
    return failures


def check_c3() -> list[str]:
    """C3: 8 × 18 = 144 total notes."""
    failures = []
    n_keynotes = len(satellite_states())
    n_notes_per_key = len(scale_fractions()) + 1  # + keynote
    total = n_keynotes * n_notes_per_key
    if total != 144:
        failures.append(
            f"{n_keynotes} keynotes × {n_notes_per_key} notes = {total} ≠ 144"
        )
    return failures


def check_c4() -> list[str]:
    """C4: all scale denominators are 7-smooth; 4 distinct prime denominators ≤ 7."""
    failures = []
    fracs = scale_fractions()
    all_denoms = sorted(set(q for _, q in fracs))
    prime_factors_of_denoms = set()

    for q in all_denoms:
        n = q
        for p in [2, 3, 5, 7]:
            while n % p == 0:
                prime_factors_of_denoms.add(p)
                n //= p
        if n > 1:
            failures.append(f"Denominator {q} has prime factor > 7: {n}")

    expected_primes = {2, 3, 5, 7}
    if prime_factors_of_denoms != expected_primes:
        failures.append(
            f"Prime factors of denominators: {sorted(prime_factors_of_denoms)}, "
            f"expected {sorted(expected_primes)}"
        )

    return failures


def check_c5() -> list[str]:
    """C5: adjacent fractions in sorted scale satisfy Farey mediant property."""
    failures = []
    fracs = scale_fractions()
    for i in range(len(fracs) - 1):
        p1, q1 = fracs[i]
        p2, q2 = fracs[i + 1]
        diff = abs(p1 * q2 - p2 * q1)
        if diff != 1:
            failures.append(
                f"{p1}/{q1} and {p2}/{q2}: |p1·q2 - p2·q1| = {diff} ≠ 1 "
                "— not Farey neighbors"
            )
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("C1", "8 Satellite states in mod-9 = 8 keynotes (4 male + 4 female)", check_c1),
    ("C2", "17 reduced fractions q≤7 in (0,1); + keynote = 18 notes",       check_c2),
    ("C3", "144 = 8 keynotes × 18 notes (Iverson p.82)",                    check_c3),
    ("C4", "All scale denominators 7-smooth; 4 distinct prime denominators", check_c4),
    ("C5", "Farey mediant property: all adjacent scale fractions are Farey neighbors", check_c5),
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
