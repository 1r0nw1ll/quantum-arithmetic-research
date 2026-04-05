QA_COMPLIANCE = "canonical_orbit_module — not an empirical script"
"""Core QA arithmetic primitives.

Pure Python, zero dependencies. A1-compliant throughout.
"""

__all__ = [
    "qa_step",
    "qa_mod",
    "orbit_family",
    "orbit_period",
    "norm_f",
    "v3",
    "qa_tuple",
    "KNOWN_MODULI",
    "self_test",
]

KNOWN_MODULI = frozenset([9, 24])


def qa_mod(x: int, m: int) -> int:
    """A1-compliant modular reduction: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def norm_f(b: int, e: int) -> int:
    """Integer norm f(b,e) = b*b + b*e - e*e in Z[phi]. S1: use b*b, not the power op."""
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


def qa_step(b: int, e: int, m: int = 24) -> tuple:
    """A1-compliant QA step: (b, e) -> (e, d) where d = b+e in {1,...,m}."""
    d = qa_mod(b + e, m)  # noqa: A2-1
    return e, d


def qa_tuple(b: int, e: int, m: int = 24) -> tuple:
    """Full QA tuple (b, e, d, a) with A2-compliant derived coords."""
    d = qa_mod(b + e, m)  # noqa: A2-1
    a = qa_mod(b + 2 * e, m)  # noqa: A2-2
    return b, e, d, a


def orbit_family(b: int, e: int, m: int = 24) -> str:
    """Classify (b,e) into orbit family.

    Exact algebraic rule (verified against simulation, 0 errors):
      singularity : b == m AND e == m (unique fixed point)
      satellite   : (m//3)|b AND (m//3)|e (excludes singularity)
      cosmos      : everything else
    """
    sat_divisor = m // 3
    if b == m and e == m:
        return "singularity"
    if b % sat_divisor == 0 and e % sat_divisor == 0:
        return "satellite"
    return "cosmos"


def orbit_period(b: int, e: int, m: int = 24) -> int:
    """Compute orbit period by simulation."""
    cb, ce = qa_step(b, e, m)
    steps = 1
    while (cb, ce) != (b, e):
        cb, ce = qa_step(cb, ce, m)
        steps += 1
        if steps > m * m:
            break
    return steps


def self_test(verbose: bool = False) -> bool:
    """Verify orbit_family() matches orbit_period() for mod-9 and mod-24."""
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
        assert errors == 0, f"orbit_family() has {errors} errors for mod-{m}"
        if verbose:
            print(f"  mod-{m}: 0 errors across {m * m} pairs")
    return True
