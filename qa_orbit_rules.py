#!/usr/bin/env python3
"""
qa_orbit_rules.py — Canonical QA Orbit Classification

Single source of truth for orbit arithmetic. All Track D scripts and
any code that classifies QA orbits must import from here.

Verified rules (0 errors against simulation for mod-9 and mod-24):

  qa_step(b, e, m)     → compliant step: result in {1,...,m}, never 0
  orbit_family(b, e, m) → singularity / satellite / cosmos

Orbit family rules (exact, algebraically derived):
  singularity: b == m AND e == m  (unique fixed point of compliant qa_step)
  satellite:   (m//3) | b  AND  (m//3) | e  AND  not singularity
  cosmos:      everything else (orbit periods 3, 6, 12, 24 for m=24)

Why NOT v₃(f(b,e)):
  f(b,e) = b*b + b*e - e*e ≡ 0 (mod 3) requires b≡0 AND e≡0 (mod 3),
  which forces f = 9*(k²+kj-j²), so v₃(f) ∈ {0} ∪ {≥2} — never exactly 1.
  Therefore v₃(f)=1 cannot distinguish satellite from cosmos. The correct
  discriminant is divisibility by m//3 (= 3 for mod-9, = 8 for mod-24).

QA_COMPLIANCE = "canonical_orbit_module — not an empirical script"
"""

__all__ = [
    "qa_step",
    "orbit_family",
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

def orbit_family(b: int, e: int, m: int = 24) -> str:
    """
    Classify (b,e) into orbit family for modulus m.

    Exact algebraic rule (verified against simulation, 0 errors):
      singularity : b == m  AND  e == m   (unique fixed point)
      satellite   : (m//3)|b  AND  (m//3)|e   (excludes singularity)
      cosmos      : everything else

    Works for m=9 and m=24. For other moduli, use orbit_period() to verify.

    S2: b, e must be Python int (not numpy scalar).
    """
    assert isinstance(b, int) and isinstance(e, int), \
        f"S2: b={b!r}, e={e!r} must be Python int, not {type(b).__name__}"
    assert 1 <= b <= m and 1 <= e <= m, \
        f"A1: ({b},{e}) out of {{1,...,{m}}}"

    sat_divisor = m // 3   # 3 for mod-9, 8 for mod-24
    if b == m and e == m:
        return "singularity"
    if b % sat_divisor == 0 and e % sat_divisor == 0:
        return "satellite"
    return "cosmos"


def orbit_period(b: int, e: int, m: int) -> int:
    """
    Compute actual orbit period by simulation (verification / non-standard moduli).

    Returns the period (number of distinct states in the orbit starting from (b,e)).
    """
    cb, ce = b, e
    seen: set[tuple[int, int]] = set()
    for _ in range(m * m + 1):
        if (cb, ce) in seen:
            break
        seen.add((cb, ce))
        cb, ce = qa_step(cb, ce, m)
    return len(seen)


# ── Self-test ──────────────────────────────────────────────────────────────────

def self_test(verbose: bool = False) -> bool:
    """
    Verify orbit_family() matches orbit_period() simulation for mod-9 and mod-24.
    Returns True if all tests pass. Raises AssertionError on first failure.
    """
    period_to_family = {1: "singularity", 8: "satellite"}

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
                        print(f"  FAIL mod-{m}: b={b} e={e} period={period} "
                              f"expected={expected} got={got}")
        assert errors == 0, \
            f"orbit_family() has {errors} errors for mod-{m}"
        if verbose:
            print(f"  mod-{m}: 0 errors across {m*m} pairs")

    return True


if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print("qa_orbit_rules.py self-test")
    self_test(verbose=True)
    print("PASS — orbit_family() verified for mod-9 and mod-24")
