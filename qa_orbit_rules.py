#!/usr/bin/env python3
"""
qa_orbit_rules.py — Canonical QA Orbit Classification

Single source of truth for orbit arithmetic. All Track D scripts and
any code that classifies QA orbits must import from here.

Canonical classifier (empirical, based on orbit period under qa_step):

  qa_step(b, e, m)            → compliant step: result in {1,...,m}, never 0
  orbit_period(b, e, m)       → period of the (b,e) orbit under qa_step
  orbit_family(b, e, m)       → singularity / satellite / cosmos by period:
                                  period 1 → singularity
                                  period 8 → satellite
                                  else     → cosmos
  orbit_family_divisor_shortcut(b, e, m)
                              → fast algebraic shortcut, EXACT only when
                                gcd(m, 5) = 1; under-counts period-8 states
                                by 32 when 5 | m. Verified for m ∈ KNOWN_MODULI.

Why orbit_family is empirical:
  The algebraic shortcut "(m//3)|b ∧ (m//3)|e ∧ not singularity" agrees with
  orbit_period == 8 on m ∈ {9, 24} (both coprime to 5) and on every other
  modulus tested with gcd(m, 5) = 1. For m with 5 | m (e.g., 15, 30, 45,
  60, 75) the shortcut under-counts by exactly 32 period-8 pairs that arise
  from the Pisano-period structure of Fibonacci-mod-5 lifted through the
  mod-3 factor. The canonical satellite predicate is therefore orbit_period
  == 8, not the divisor shortcut. The shortcut is preserved as a named
  helper for callers that explicitly want the fast algebraic answer on
  verified moduli.

Why NOT v₃(f(b,e)):
  f(b,e) = b*b + b*e - e*e ≡ 0 (mod 3) requires b≡0 AND e≡0 (mod 3),
  which forces f = 9*(k²+kj-j²), so v₃(f) ∈ {0} ∪ {≥2} — never exactly 1.
  Therefore v₃(f)=1 cannot distinguish satellite from cosmos.

QA_COMPLIANCE = "canonical_orbit_module — not an empirical script"
"""

from functools import lru_cache

__all__ = [
    "qa_step",
    "orbit_family",
    "orbit_family_divisor_shortcut",
    "orbit_period",
    "norm_f",
    "v3",
    "self_test",
    "KNOWN_MODULI",
]

KNOWN_MODULI = frozenset([9, 24])


# ── Core arithmetic ────────────────────────────────────────────────────────────

def norm_f(b: int, e: int) -> int:
    """Integer norm f(b,e) = b*b + b*e - e*e in Z[φ]. S1: b*b not b-squared."""
    return b * b + b * e - e * e


def v3(n: int) -> int:
    """3-adic valuation of n (returns 9999 for n=0)."""
    if n == 0:
        return 9999
    n = abs(n)
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def qa_step(b: int, e: int, m: int) -> tuple[int, int]:
    """
    A1-compliant QA step: result in {1,...,m}, never 0.

    Correct: ((b+e-1) % m) + 1
    NOT:     (b+e) % m  — which produces 0 when (b+e) is a multiple of m.
    """
    assert 1 <= b <= m and 1 <= e <= m, f"qa_step: ({b},{e}) out of {{1,...,{m}}}"
    return e, ((b + e - 1) % m) + 1


# ── Orbit classification ───────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def orbit_period(b: int, e: int, m: int) -> int:
    """
    Compute orbit period by simulation under qa_step. Cached — deterministic.

    Returns the number of distinct states in the orbit starting from (b,e).
    period 1 = fixed point (singularity), period 8 = satellite, else cosmos.
    """
    cb, ce = b, e
    seen: set[tuple[int, int]] = set()
    for _ in range(m * m + 1):
        if (cb, ce) in seen:
            break
        seen.add((cb, ce))
        cb, ce = qa_step(cb, ce, m)
    return len(seen)


@lru_cache(maxsize=None)
def orbit_family(b: int, e: int, m: int = 24) -> str:
    """
    Canonical QA orbit classifier (empirical, by orbit period under qa_step):

      period 1 → singularity   (fixed point)
      period 8 → satellite
      else     → cosmos

    Works for any m ≥ 2. For the fast algebraic shortcut on verified moduli
    (m ∈ KNOWN_MODULI = {9, 24}), see orbit_family_divisor_shortcut.

    S2: b, e must be Python int (not numpy scalar).
    A1: states asserted in {1,...,m}.
    """
    assert isinstance(b, int) and isinstance(e, int), \
        f"S2: b={b!r}, e={e!r} must be Python int, not {type(b).__name__}"
    assert 1 <= b <= m and 1 <= e <= m, \
        f"A1: ({b},{e}) out of {{1,...,{m}}}"

    period = orbit_period(b, e, m)
    if period == 1:
        return "singularity"
    if period == 8:
        return "satellite"
    return "cosmos"


def orbit_family_divisor_shortcut(b: int, e: int, m: int = 24) -> str:
    """
    Fast algebraic shortcut for orbit_family.

    Rule:
      singularity : b == m AND e == m   (unique fixed point of qa_step)
      satellite   : (m//3)|b AND (m//3)|e   (excluding singularity)
      cosmos      : everything else

    Verified to agree with the canonical orbit_family (orbit_period-based)
    for m ∈ KNOWN_MODULI = {9, 24} and, more broadly, for all m with
    gcd(m, 5) = 1 in the {9, 12, 15, ..., 75} sweep. For m with 5 | m
    (e.g., 15, 30, 45, 60, 75) this shortcut UNDER-COUNTS the empirical
    satellite class by exactly 32 pairs per modulus.

    Use this when:
      - m is in KNOWN_MODULI and you want O(1) classification
      - or you explicitly want the algebraic answer (e.g., for studying
        the boundary itself).

    Use orbit_family() instead when:
      - m may have a 5-factor
      - you want the canonical period-based truth
      - performance is acceptable (orbit_period is cached)
    """
    assert isinstance(b, int) and isinstance(e, int), \
        f"S2: b={b!r}, e={e!r} must be Python int, not {type(b).__name__}"
    assert 1 <= b <= m and 1 <= e <= m, \
        f"A1: ({b},{e}) out of {{1,...,{m}}}"

    sat_divisor = m // 3
    if b == m and e == m:
        return "singularity"
    if sat_divisor > 0 and b % sat_divisor == 0 and e % sat_divisor == 0:
        return "satellite"
    return "cosmos"


# ── Self-test ──────────────────────────────────────────────────────────────────

def self_test(verbose: bool = False) -> bool:
    """
    Verify the canonical orbit_family contract and the divisor-shortcut boundary.

    Checks:
      1. orbit_family matches orbit_period classification across m ∈ {9, 24}.
      2. orbit_family_divisor_shortcut matches orbit_family for m ∈ KNOWN_MODULI.
      3. orbit_family_divisor_shortcut under-counts orbit_family by exactly
         32 pairs for m ∈ {15, 30} (5-factor boundary witness).

    Returns True if all checks pass. Raises AssertionError on first failure.
    """
    # Check 1: orbit_family classification matches orbit_period directly.
    for m in [9, 24]:
        errors = 0
        for b in range(1, m + 1):
            for e in range(1, m + 1):
                period = orbit_period(b, e, m)
                if period == 1:
                    expected = "singularity"
                elif period == 8:
                    expected = "satellite"
                else:
                    expected = "cosmos"
                got = orbit_family(b, e, m)
                if got != expected:
                    errors += 1
                    if verbose:
                        print(f"  FAIL canonical mod-{m}: b={b} e={e} period={period} "
                              f"expected={expected} got={got}")
        assert errors == 0, \
            f"orbit_family() has {errors} canonical errors for mod-{m}"
        if verbose:
            print(f"  canonical mod-{m}: 0 errors across {m*m} pairs")

    # Check 2: divisor shortcut matches canonical on KNOWN_MODULI.
    for m in sorted(KNOWN_MODULI):
        diffs = 0
        for b in range(1, m + 1):
            for e in range(1, m + 1):
                if orbit_family(b, e, m) != orbit_family_divisor_shortcut(b, e, m):
                    diffs += 1
        assert diffs == 0, \
            f"divisor shortcut diverges from canonical for mod-{m}: {diffs} cases"
        if verbose:
            print(f"  divisor-shortcut mod-{m}: 0 diffs vs canonical")

    # Check 3: divisor shortcut under-counts by 32 for 5-factor moduli.
    for m in [15, 30]:
        misses = 0
        over_claims = 0
        for b in range(1, m + 1):
            for e in range(1, m + 1):
                canonical = orbit_family(b, e, m)
                shortcut = orbit_family_divisor_shortcut(b, e, m)
                if canonical == "satellite" and shortcut != "satellite":
                    misses += 1
                if shortcut == "satellite" and canonical != "satellite":
                    over_claims += 1
        assert over_claims == 0, \
            f"divisor shortcut over-claims for mod-{m}: {over_claims} cases"
        assert misses == 32, \
            f"divisor shortcut should miss exactly 32 satellites for mod-{m}; got {misses}"
        if verbose:
            print(f"  divisor-shortcut mod-{m}: 32 satellite misses (boundary witness)")

    return True


if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print("qa_orbit_rules.py self-test")
    self_test(verbose=verbose)
    print("PASS — orbit_family canonical (period-based) verified;")
    print("       divisor shortcut verified on KNOWN_MODULI = {9, 24};")
    print("       divisor shortcut under-counts by 32 on m ∈ {15, 30} (5-factor witness).")
