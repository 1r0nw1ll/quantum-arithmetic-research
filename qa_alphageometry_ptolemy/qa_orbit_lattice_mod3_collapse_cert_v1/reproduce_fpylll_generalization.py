#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=generalization experiment for cert [515]; sources cited in mapping_protocol_ref.json -->
"""
Generalization experiment for cert [515]'s mod-3 collapse finding.

cert [515] proved: whenever 3|m, (x mod m) mod 3 == x mod 3 (reduction mod m
commutes with reduction mod 3), so the ternary NTRU coefficient sequence
(v mod 3 - 1) inherits the short mod-3 Pisano recursion (period <= 8)
regardless of m's own period. Verified 3|9 unsafe, gcd(80,3)=1 safe.

This script checks the two open generalization questions:

1. Does the collapse mechanism actually depend on p=3 specifically, or does
   it hold for ANY prime p|m? (Algebraic check: general, not 3-specific --
   see companion algebra check.) Since NTRU's ternary-coefficient
   construction is fixed at mod 3, only whether 3|m matters for THIS
   construction -- other prime factors of m (e.g. the 2 in m=24 or m=80)
   are irrelevant to the ternary coefficient sequence.

2. m=24 is the QA project's actual "applied" modulus (CLAUDE.md: "mod 9
   (theoretical) or mod 24 (applied)"), and 3|24. Cert [515] never tested
   it -- only m=9 (unsafe) and m=80 (safe) were checked. Since 3|24, the
   theorem predicts m=24 is ALSO unsafe. This is the practically important
   gap: is the QA project's own real default "applied" modulus vulnerable?

3. Robustness of the "safe" claim: is gcd(m,3)=1 enough across several
   different composite structures, or was m=80 a lucky pick? Tests
   m=35=5*7, m=25=5^2, m=49=7^2 in addition to m=80.

4. Does severity change with higher powers of 3 (m=27, m=81) vs m=9?
"""
from __future__ import annotations

import random
import sys
from math import gcd

try:
    from fpylll import IntegerMatrix, LLL, BKZ
except ImportError:
    print("fpylll not installed -- `pip install fpylll cysignals` into a venv first.",
          file=sys.stderr)
    raise

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reproduce_fpylll_experiment import (  # noqa: E402
    qa_poly, seeds_with_period, keygen_random, keygen_qa,
    check_fg_in_lattice, attack_once, summarize, orbit_period,
)


def find_long_period_seed_pool(m: int, trials_scan: int = 400, limit: int = 40):
    """Find a pool of (b0,e0) seeds with long orbit period under mod m,
    so any observed lattice weakness is attributable to the mod-3
    collapse mechanism, not merely a short overall orbit period."""
    periods = {}
    seen = 0
    for b0 in range(1, m + 1):
        for e0 in range(1, m + 1):
            p = orbit_period(b0, e0, m)
            periods.setdefault(p, []).append((b0, e0))
            seen += 1
            if seen >= trials_scan:
                break
        if seen >= trials_scan:
            break
    best_period = max(periods)
    pool = periods[best_period][:limit]
    return pool, best_period


if __name__ == "__main__":
    N, q, trials = 83, 256, 10
    rng = random.Random(42)

    print(f"=== Cert [515] generalization: N={N} q={q}, LLL only ===\n")

    cases = [
        ("m=9   (3|m, tested in [515])", 9),
        ("m=24  (3|m, QA APPLIED modulus, untested)", 24),
        ("m=27  (3^3|m, higher power)", 27),
        ("m=81  (3^4|m, higher power)", 81),
        ("m=80  (gcd=1, tested in [515])", 80),
        ("m=35  (gcd=1, =5*7)", 35),
        ("m=25  (gcd=1, =5^2)", 25),
        ("m=49  (gcd=1, =7^2)", 49),
    ]

    print("--- baseline (random ternary keys) ---")
    ratios_random = []
    for _ in range(trials):
        f, g, h = keygen_random(N, q, rng)
        assert check_fg_in_lattice(f, g, h, N, q)
        best, target = attack_once(f, g, h, N, q)
        ratios_random.append(best / target)
    summarize(ratios_random, "random-key baseline", trials)
    print()

    results = {}
    for label, m in cases:
        pool, best_period = find_long_period_seed_pool(m, trials_scan=min(m * m, 6400), limit=trials * 4)
        ratios = []
        for _ in range(trials):
            f, g, h = keygen_qa(N, q, m, rng, seed_pool=pool)
            assert check_fg_in_lattice(f, g, h, N, q)
            best, target = attack_once(f, g, h, N, q)
            ratios.append(best / target)
        results[m] = ratios
        divides3 = (m % 3 == 0)
        print(f"--- {label}  [longest-period pool: period={best_period}, gcd(m,3)={gcd(m,3)}] ---")
        summarize(ratios, label, trials)
        print()

    print("=== Summary ===")
    print(f"{'case':45s} {'3|m?':>6s} {'broken':>8s} {'avg ratio':>10s}")
    for label, m in cases:
        ratios = results[m]
        broke = sum(1 for r in ratios if r <= 1.5)
        avg = sum(ratios) / len(ratios)
        print(f"{label:45s} {'yes' if m % 3 == 0 else 'no':>6s} {broke:>5d}/{trials:<2d} {avg:10.3f}")
