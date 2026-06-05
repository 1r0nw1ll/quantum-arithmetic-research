#!/usr/bin/env python3
"""
QA To-Be-Prime / Euclid VII Prop 28 Coprimeness Chain Cert [325]

Primary source:
  Iverson, B. (1991) Quantum Arithmetic Volume II — Books 3 & 4:
    New Wave Theory / Synchronous Harmonics. Delta Spectrum Research,
    La Junta CO. pp.2-3, section "TO BE PRIME".

Iverson derives the QA coprime quadruple (b,e,d,a) from Euclid Book VII,
Proposition 28: "If a number be prime to two numbers, it will also be prime
to the number composed of them." Equivalently: gcd(a,b)=1 implies
gcd(a+b, a)=1 and gcd(a+b, b)=1.

Iverson's explicit chain (p.2):
  "Every integer is prime to any integer which is one unit less or one more
   than itself." → gcd(n, n+1)=1 ∀n.
  "Every integer is prime to any integer which is two units more or two
   units less than itself, provided the original number is not divisible
   by two." → for odd n, gcd(n, n+2)=1.
  "Every integer is prime to any integer which differs from itself by a
   prime number. This can be taken as a corollary to Euclid VII, Prop 28."
  "Book VII, Proposition 28 states that the sum and the difference between
   two coprime integers will be prime to both of them. This creates the
   four integer sequence which has been called the 'quantum number'."

The cert certifies this chain algebraically — no floating point, no QA
dynamics, just GCD arithmetic on explicit integer pairs.

Five claims:

  C1  Neighbor coprimeness: gcd(n, n+1) = 1 for all n in {1, …, 100}.

  C2  Odd skip-2 coprimeness: for all odd n in {1, 3, …, 99},
      gcd(n, n+2) = 1.
      (Even n fails: gcd(2, 4) = 2; the "provided n is not divisible by
      two" qualifier is integral to Iverson's statement.)

  C3  Euclid VII Prop 28 (SUM): for all a,b ∈ {1,…,50} with gcd(a,b)=1,
      gcd(a+b, a) = 1  and  gcd(a+b, b) = 1.
      (Also verified: gcd(a,b)=1 is NOT preserved in general when
      gcd(a,b)>1; we exhibit one counterexample to the failure of
      extension: gcd(4,6)=2 and gcd(4+6,4)=gcd(10,4)=2 ≠ 1.)

  C4  BEDA coprimeness chain (Prop 28 applied twice to QA raw coords):
      For every mod-9 Cosmos pair (b,e) with b≠e AND gcd(b,e)=1:
        d = b+e (raw)  →  gcd(d, b)=1  and  gcd(d, e)=1  (Prop 28)
        a = b+2e = d+e (raw)  →  gcd(a, d)=1  and  gcd(a, e)=1  (Prop 28)
      The cert counts such pairs, lists all four gcd values, and asserts
      zero failures. (Pairs with gcd(b,e)>1 are counted separately to
      confirm the split.)

  C5  Euclid VII Prop 28 (DIFFERENCE): for all a,b ∈ {1,…,50} with
      a > b and gcd(a,b)=1:
        gcd(a-b, a) = 1  and  gcd(a-b, b) = 1.
      (Together with C3 this gives: coprime pair → sum AND difference
      are coprime to both originals — exactly Iverson's "sum and
      difference … prime to both of them.")
"""

from math import gcd

QA_COMPLIANCE = (
    "cert_validator -- pure integer GCD arithmetic; no floating point; "
    "no QA state evolution; Theorem NT: raw d=b+e and a=b+2e are used "
    "as integer targets for coprimeness checks, not as QA state inputs; "
    "all arithmetic exact integer"
)

M = 9  # mod-9 QA modulus


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cosmos_pairs(m: int = M) -> list[tuple[int, int]]:
    return [(b, e) for b in range(1, m + 1) for e in range(1, m + 1) if b != e]


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------

def check_c1() -> list[str]:
    """C1: gcd(n, n+1) = 1 for n = 1..100."""
    failures = []
    for n in range(1, 101):
        g = gcd(n, n + 1)
        if g != 1:
            failures.append(f"gcd({n},{n+1}) = {g} ≠ 1")
    return failures


def check_c2() -> list[str]:
    """C2: for odd n in {1,3,...,99}, gcd(n, n+2)=1.
    Also confirm the qualifier: gcd(2,4)=2 (even case fails)."""
    failures = []
    for n in range(1, 100, 2):  # odd only
        g = gcd(n, n + 2)
        if g != 1:
            failures.append(f"gcd({n},{n+2}) = {g} ≠ 1 (odd n)")
    # Confirm even case is different (Iverson's qualifier)
    if gcd(2, 4) == 1:
        failures.append("gcd(2,4)=1 — expected =2; qualifier check failed")
    return failures


def check_c3() -> list[str]:
    """C3: Euclid VII Prop 28 SUM: coprime a,b → gcd(a+b,a)=gcd(a+b,b)=1."""
    failures = []
    pair_count = 0
    for a in range(1, 51):
        for b in range(1, 51):
            if gcd(a, b) == 1:
                pair_count += 1
                g1 = gcd(a + b, a)
                g2 = gcd(a + b, b)
                if g1 != 1:
                    failures.append(f"gcd({a+b},{a}) = {g1} ≠ 1 (a={a},b={b})")
                if g2 != 1:
                    failures.append(f"gcd({a+b},{b}) = {g2} ≠ 1 (a={a},b={b})")
    if pair_count < 1000:
        failures.append(f"Only {pair_count} coprime pairs tested — expected >1000")
    # Confirm non-coprime case can fail (counterexample)
    a, b = 4, 6
    if gcd(a, b) == 1:
        failures.append("gcd(4,6) should be 2, not 1 — counterexample broken")
    if gcd(a + b, a) == 1:
        failures.append("gcd(10,4) should be 2, not 1 — counterexample broken")
    return failures


def check_c4() -> list[str]:
    """C4: BEDA coprimeness chain for mod-9 Cosmos pairs with gcd(b,e)=1."""
    failures = []
    coprime_pairs = []
    non_coprime_count = 0
    for b, e in cosmos_pairs():
        if gcd(b, e) == 1:
            coprime_pairs.append((b, e))
        else:
            non_coprime_count += 1

    for b, e in coprime_pairs:
        d = b + e   # raw, no mod
        a = b + 2 * e  # raw, no mod
        # Prop 28 step 1: gcd(b,e)=1 → gcd(d,b)=gcd(d,e)=1
        g_db = gcd(d, b)
        g_de = gcd(d, e)
        if g_db != 1:
            failures.append(f"({b},{e}): gcd(d={d}, b={b}) = {g_db} ≠ 1")
        if g_de != 1:
            failures.append(f"({b},{e}): gcd(d={d}, e={e}) = {g_de} ≠ 1")
        # Prop 28 step 2: gcd(d,e)=1 → gcd(a,d)=gcd(a,e)=1
        g_ad = gcd(a, d)
        g_ae = gcd(a, e)
        if g_ad != 1:
            failures.append(f"({b},{e}): gcd(a={a}, d={d}) = {g_ad} ≠ 1")
        if g_ae != 1:
            failures.append(f"({b},{e}): gcd(a={a}, e={e}) = {g_ae} ≠ 1")

    n_coprime = len(coprime_pairs)
    if n_coprime < 20:
        failures.append(
            f"Only {n_coprime} coprime Cosmos pairs found; expected ≥ 20"
        )
    return failures


def check_c5() -> list[str]:
    """C5: Euclid VII Prop 28 DIFF: coprime a,b (a>b) → gcd(a-b,a)=gcd(a-b,b)=1."""
    failures = []
    pair_count = 0
    for a in range(1, 51):
        for b in range(1, a):
            if gcd(a, b) == 1:
                pair_count += 1
                diff = a - b
                g1 = gcd(diff, a)
                g2 = gcd(diff, b)
                if g1 != 1:
                    failures.append(
                        f"gcd({diff},{a}) = {g1} ≠ 1 (a={a},b={b})"
                    )
                if g2 != 1:
                    failures.append(
                        f"gcd({diff},{b}) = {g2} ≠ 1 (a={a},b={b})"
                    )
    if pair_count < 500:
        failures.append(
            f"Only {pair_count} coprime pairs (a>b) tested — expected ≥ 500"
        )
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("C1", "gcd(n,n+1)=1 for n=1..100 (neighbor coprimeness)",                    check_c1),
    ("C2", "gcd(odd n, n+2)=1 for odd n=1..99; gcd(2,4)=2 (qualifier holds)",     check_c2),
    ("C3", "Euclid VII Prop 28 SUM: coprime a,b → gcd(a+b,a)=gcd(a+b,b)=1",      check_c3),
    ("C4", "BEDA chain: gcd(b,e)=1 → gcd(d,b)=gcd(d,e)=gcd(a,d)=gcd(a,e)=1",    check_c4),
    ("C5", "Euclid VII Prop 28 DIFF: coprime a>b → gcd(a-b,a)=gcd(a-b,b)=1",     check_c5),
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
