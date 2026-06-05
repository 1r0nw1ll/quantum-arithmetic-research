#!/usr/bin/env python3
"""
QA Euclid Four Par Types Cert [326]

Primary source:
  Iverson, B. (1991) Quantum Arithmetic Volume II — Books 3 & 4:
    New Wave Theory / Synchronous Harmonics. Delta Spectrum Research,
    La Junta CO. p.8, section "EUCLID'S 4 NUMBER TYPES"; also
    pp.5-7 "TIME SYNCHRONIZATION", "EVEN NUMBERS", "GRAPHICS".

Iverson identifies a four-way classification of integers from Euclid Book VII
that contemporary mathematics "completely missed" (p.8):

  "2-par, (even-odd, 4n-2); 3-par, (odd-even, 4n-1);
   4-par, (even-even, 4n); and 5-par, (odd-odd, 4n+1)"

In modern notation: the four classes by residue mod 4:
  4-par → n ≡ 0 (mod 4)   [divisible by 4; "even-even": v2 ≥ 2]
  5-par → n ≡ 1 (mod 4)   ["odd-odd"; Iverson's 4n+1]
  2-par → n ≡ 2 (mod 4)   ["even-odd"; exactly one factor of 2; v2 = 1]
  3-par → n ≡ 3 (mod 4)   ["odd-even"; Iverson's 4n-1]

The naming is non-standard (par names come from Ben's wave-physics labels,
not from the mod-4 residue directly). The key physical consequences Iverson
identifies:
  - Two waves of the SAME par type (both 3-par, or both 5-par) REINFORCE
    each other at the quarter-points of their product (harmonic points).
  - A 3-par and a 5-par wave together CANCEL at quarter-points (null packet).
  - 2-par and 4-par waves are NEVER coprime (share factor of 2); they
    synchronize at lcm < product.

Iverson's time-synchronization table (p.5): for coprime p,q, their first
synchronous return is at time = p×q. For non-coprime p,q, it is lcm(p,q).

Five claims:

  C1  Four-par partition: the four classes {4n}, {4n+1}, {4n+2}, {4n+3}
      partition {1,…,24} into 4 equal groups of exactly 6 elements each.
      4-par = {4,8,12,16,20,24}, 5-par = {1,5,9,13,17,21},
      2-par = {2,6,10,14,18,22}, 3-par = {3,7,11,15,19,23}.

  C2  Synchronous period: for coprime p,q ∈ {2,…,24}, lcm(p,q) = p×q.
      For non-coprime p,q, lcm(p,q) = p×q/gcd(p,q) < p×q.
      Ben's example: (4,6) → gcd=2, lcm=12 = 4×6/2 < 24 (equivalent to
      treating (4,6) as (2,3)×lcm_unit; only the coprime core determines
      the period).

  C3  Quarter-harmonic point: for coprime odd p,q ∈ {3,…,23} with p<q,
      the harmonic surge points occur at ⌊p×q/4⌋ and 3×⌊p×q/4⌋.
      Ben's examples (p.7): (3,7)→21: quarter=5, three-quarter=15 ✓;
      (5,9)→45: quarter=11, three-quarter=33 ✓.
      (Note: 5 and 9 are coprime since gcd(5,9)=1, even though 9=3².)

  C4  Odd par-type multiplication table (mod 4 arithmetic):
      3-par × 3-par = 5-par (3×3≡1 mod 4)
      5-par × 5-par = 5-par (1×1≡1 mod 4)
      3-par × 5-par = 3-par (3×1≡3 mod 4)
      Verified exhaustively: for all (a,b) in {odd integers 1..23}²,
      the product a×b has the par type given by the table.

  C5  2-par vs 4-par non-coprimeness: every 2-par integer n has
      v2(n) = 1 (exactly one factor of 2); every 4-par integer m has
      v2(m) ≥ 2. Therefore gcd(n, m) ≥ 2 — no 2-par and 4-par integer
      can be coprime. Verified for all (2-par, 4-par) pairs in {1..24}.
"""

from math import gcd, log2

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (gcd, lcm, mod, v2); "
    "no QA state evolution; Theorem NT: wave-reinforcement / cancellation "
    "at quarter-harmonic points is an observer-layer label on integer "
    "LCM structure; the par-type classification is exact integer mod 4; "
    "all arithmetic exact integer, no float"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def par_type(n: int) -> int:
    """Return Iverson par type: 4=4-par, 5=5-par, 2=2-par, 3=3-par."""
    r = n % 4
    if r == 0:
        return 4
    if r == 1:
        return 5
    if r == 2:
        return 2
    return 3  # r == 3


def v2(n: int) -> int:
    """2-adic valuation: largest k with 2^k divides n."""
    if n == 0:
        return float('inf')
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------

def check_c1() -> list[str]:
    """C1: Four par types partition {1..24} into 4 equal groups of 6."""
    failures = []
    groups = {2: [], 3: [], 4: [], 5: []}
    for n in range(1, 25):
        groups[par_type(n)].append(n)

    expected = {
        4: [4, 8, 12, 16, 20, 24],
        5: [1, 5, 9, 13, 17, 21],
        2: [2, 6, 10, 14, 18, 22],
        3: [3, 7, 11, 15, 19, 23],
    }
    for pt in [2, 3, 4, 5]:
        if groups[pt] != expected[pt]:
            failures.append(
                f"{pt}-par: got {groups[pt]}, expected {expected[pt]}"
            )
        if len(groups[pt]) != 6:
            failures.append(
                f"{pt}-par: {len(groups[pt])} elements, expected 6"
            )

    # Check union covers {1..24} exactly
    union = sorted(n for g in groups.values() for n in g)
    if union != list(range(1, 25)):
        failures.append(f"Union ≠ {{1..24}}: got {union}")

    return failures


def check_c2() -> list[str]:
    """C2: Coprime p,q → lcm=p×q; non-coprime → lcm < p×q.
    Ben's example: (4,6) → gcd=2, lcm=12 < 24."""
    failures = []
    coprime_checked = 0
    non_coprime_checked = 0

    for p in range(2, 25):
        for q in range(p + 1, 25):
            g = gcd(p, q)
            l = lcm(p, q)
            if g == 1:
                if l != p * q:
                    failures.append(
                        f"lcm({p},{q})={l} ≠ {p}×{q}={p*q} (coprime)"
                    )
                coprime_checked += 1
            else:
                if l >= p * q:
                    failures.append(
                        f"lcm({p},{q})={l} ≥ {p*q} (non-coprime gcd={g})"
                    )
                non_coprime_checked += 1

    # Explicit Ben's example
    if lcm(4, 6) != 12:
        failures.append(f"lcm(4,6) = {lcm(4,6)} ≠ 12 (Ben's example fails)")
    if gcd(4, 6) != 2:
        failures.append(f"gcd(4,6) = {gcd(4,6)} ≠ 2")

    if coprime_checked < 50:
        failures.append(f"Only {coprime_checked} coprime pairs tested")
    if non_coprime_checked < 50:
        failures.append(f"Only {non_coprime_checked} non-coprime pairs tested")

    return failures


def check_c3() -> list[str]:
    """C3: Quarter-harmonic at ⌊p×q/4⌋ and 3×⌊p×q/4⌋ for coprime odd p<q.
    Ben's examples: (3,7)→(5,15); (5,9)→(11,33)."""
    failures = []

    # Ben's explicit examples
    examples = [(3, 7), (5, 9)]
    expected_quarters = {(3, 7): (5, 15), (5, 9): (11, 33)}

    for p, q in examples:
        if gcd(p, q) != 1:
            failures.append(f"({p},{q}): not coprime — gcd={gcd(p,q)}")
            continue
        prod = p * q
        q1 = prod // 4
        q3 = 3 * q1
        exp_q1, exp_q3 = expected_quarters[(p, q)]
        if q1 != exp_q1:
            failures.append(
                f"({p},{q}): ⌊{prod}/4⌋={q1} ≠ {exp_q1} (Ben's example)"
            )
        if q3 != exp_q3:
            failures.append(
                f"({p},{q}): 3×{q1}={q3} ≠ {exp_q3} (Ben's three-quarter)"
            )

    # Exhaustive: check the formula is consistent for all coprime odd pairs {3..23}
    checked = 0
    for p in range(3, 24, 2):
        for q in range(p + 2, 24, 2):
            if gcd(p, q) != 1:
                continue
            prod = p * q
            q1 = prod // 4
            q3 = 3 * q1
            # Both should be within (0, prod) and q1 < q3
            if not (0 < q1 < q3 < prod):
                failures.append(
                    f"({p},{q}): quarter points out of range: q1={q1}, q3={q3}, prod={prod}"
                )
            checked += 1

    if checked < 10:
        failures.append(f"Only {checked} coprime odd pairs verified")

    return failures


def check_c4() -> list[str]:
    """C4: Odd par-type multiplication table mod 4.
    3-par×3-par=5-par; 5-par×5-par=5-par; 3-par×5-par=3-par."""
    failures = []

    table = {(3, 3): 5, (5, 5): 5, (3, 5): 3, (5, 3): 3}

    for a in range(1, 24, 2):  # odd 1..23
        pt_a = par_type(a)
        if pt_a not in (3, 5):
            continue
        for b in range(1, 24, 2):
            pt_b = par_type(b)
            if pt_b not in (3, 5):
                continue
            prod = a * b
            pt_prod = par_type(prod)
            expected = table[(pt_a, pt_b)]
            if pt_prod != expected:
                failures.append(
                    f"{a}({pt_a}-par) × {b}({pt_b}-par) = {prod} "
                    f"({pt_prod}-par) ≠ expected {expected}-par"
                )

    # Verify the mod-4 arithmetic directly
    mod4_table = {(3, 3): 1, (1, 1): 1, (3, 1): 3, (1, 3): 3}
    for (r1, r2), expected_r in mod4_table.items():
        actual = (r1 * r2) % 4
        if actual != expected_r:
            failures.append(
                f"Mod-4 arithmetic: {r1}×{r2} ≡ {actual} ≠ {expected_r} (mod 4)"
            )

    return failures


def check_c5() -> list[str]:
    """C5: 2-par has v2=1; 4-par has v2≥2; → no 2-par/4-par pair is coprime."""
    failures = []

    two_pars = [n for n in range(1, 25) if par_type(n) == 2]
    four_pars = [n for n in range(1, 25) if par_type(n) == 4]

    # Verify v2 values
    for n in two_pars:
        if v2(n) != 1:
            failures.append(f"2-par {n}: v2={v2(n)} ≠ 1")
    for n in four_pars:
        if v2(n) < 2:
            failures.append(f"4-par {n}: v2={v2(n)} < 2")

    # Verify no coprime pair between 2-par and 4-par
    coprime_found = 0
    for a in two_pars:
        for b in four_pars:
            g = gcd(a, b)
            if g == 1:
                coprime_found += 1
                failures.append(f"gcd({a} [2-par], {b} [4-par]) = 1 — unexpected coprime pair")
            elif g < 2:
                failures.append(f"gcd({a},{b}) = {g} < 2 (expected ≥ 2)")

    if coprime_found > 0:
        failures.append(f"{coprime_found} unexpected coprime 2-par/4-par pairs found")

    # Spot-check: gcd between any 2-par and 4-par must have v2 ≥ 1
    for a in two_pars:
        for b in four_pars:
            g = gcd(a, b)
            if v2(g) < 1:
                failures.append(
                    f"v2(gcd({a},{b})) = v2({g}) = {v2(g)} < 1"
                )

    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("C1", "Four par types partition {1..24} into 4 equal groups of 6",                    check_c1),
    ("C2", "Coprime lcm=product; non-coprime lcm<product; Ben's (4,6)→lcm=12 ✓",          check_c2),
    ("C3", "Quarter-harmonic at ⌊p×q/4⌋ and 3×⌊p×q/4⌋; (3,7)→(5,15), (5,9)→(11,33) ✓",check_c3),
    ("C4", "Odd par-type mult table: 3×3=5-par, 5×5=5-par, 3×5=3-par (mod 4)",           check_c4),
    ("C5", "2-par has v2=1; 4-par has v2≥2; → 2-par and 4-par never coprime",             check_c5),
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
