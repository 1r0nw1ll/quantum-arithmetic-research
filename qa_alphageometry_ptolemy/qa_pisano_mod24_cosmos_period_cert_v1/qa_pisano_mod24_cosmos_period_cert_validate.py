#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — exhaustive integer orbit enumeration on {1,...,24}^2 with A1 no-zero arithmetic; gcd and Fibonacci arithmetic exact; no float QA state"
"""
QA Pisano Mod-24 Applied Cosmos Period Cert [306] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Wall (1960) Fibonacci Primitive Roots, Amer. Math. Monthly 67(6):525-532

This cert closes the arc from [291]–[302] by connecting the Pisano period tower to the
applied QA state space {1,...,24}².

The theoretical system (mod 9) has Cosmos period 24 = π(9) (cert [291]).
The applied system (mod 24) has Pisano period π(24) = lcm(π(3),π(8)) = lcm(8,12) = 24 (cert [302]).
This cert proves these two "24"s name the same phenomenon in the applied dynamics:
the maximum T-orbit period in {1,...,24}² is exactly 24.

Five claims:

  C1  Period ceiling: every orbit in {1,...,24}^2 has period dividing π(24)=24;
      maximum period = 24; period spectrum = {1,3,6,8,12,24} ⊆ divisors(24)

  C2  Applied Singularity: T(24,24) = (24,24); unique period-1 fixed point;
      characterized by 24 | gcd(b,e)

  C3  Applied Satellite analog: the 8 states {8,16,24}² \\ {(24,24)} all have period
      exactly 8; characterized by 8|gcd(b,e) and 3∤gcd(b,e);
      structural parallel: mod-9 Satellite = {3,6,9}² \\ {(9,9)}

  C4  Applied Cosmos: 504 states with period exactly 24 = 21 orbits × 24;
      characterization: 3∤gcd(b,e) AND 8∤gcd(b,e);
      count = 576 − 64 − 9 + 1 = 504 by inclusion-exclusion

  C5  Closing the loop: max orbit period (24) = π(24) (cert [302]) = π(9) = Cosmos period
      (cert [291]); applied modulus 24 is period-self-consistent
"""

import sys
from math import gcd


# ---------------------------------------------------------------------------
# Core machinery
# ---------------------------------------------------------------------------

def t_step(b, e, m=24):
    """QA T-step with A1 no-zero arithmetic: (b,e) -> (e, ((b+e-1) % m) + 1)."""
    return e, ((b + e - 1) % m) + 1


def orbit_period(b0, e0, m=24):
    """Return the orbit period of (b0, e0) under QA T-step mod m."""
    b, e = t_step(b0, e0, m)
    for k in range(1, m * m + 2):
        if (b, e) == (b0, e0):
            return k
        b, e = t_step(b, e, m)
    raise RuntimeError(f"orbit_period: no period found for ({b0},{e0}) mod {m}")


def build_period_map(m=24):
    """Return dict {(b,e): period} for all states in {1,...,m}^2."""
    return {(b, e): orbit_period(b, e, m) for b in range(1, m + 1) for e in range(1, m + 1)}


# ---------------------------------------------------------------------------
# C1 — Period ceiling: all periods divide 24; max = 24
# ---------------------------------------------------------------------------
def check_c1(pm):
    failures = []
    periods = set(pm.values())

    for p in periods:
        if 24 % p != 0:
            failures.append(f"Period {p} does not divide 24")

    expected_spectrum = {1, 3, 6, 8, 12, 24}
    if periods != expected_spectrum:
        failures.append(f"Period spectrum = {sorted(periods)}, expected {sorted(expected_spectrum)}")

    if max(periods) != 24:
        failures.append(f"Max period = {max(periods)}, expected 24")

    if len(pm) != 576:
        failures.append(f"State count = {len(pm)}, expected 576 = 24^2")

    from collections import Counter
    hist = Counter(pm.values())
    expected_hist = {1: 1, 3: 3, 6: 12, 8: 8, 12: 48, 24: 504}
    if dict(hist) != expected_hist:
        failures.append(f"Period histogram = {dict(sorted(hist.items()))}, expected {expected_hist}")

    return failures


# ---------------------------------------------------------------------------
# C2 — Applied Singularity: T(24,24) = (24,24); unique fixed point
# ---------------------------------------------------------------------------
def check_c2(pm):
    failures = []

    b1, e1 = t_step(24, 24)
    if (b1, e1) != (24, 24):
        failures.append(f"T(24,24) = ({b1},{e1}), expected (24,24)")

    period_1_states = [s for s, p in pm.items() if p == 1]
    if period_1_states != [(24, 24)]:
        failures.append(f"Period-1 states = {period_1_states}, expected [(24,24)]")

    # Characterized by 24 | gcd(b,e)
    for b in range(1, 25):
        for e in range(1, 25):
            char = (gcd(b, e) % 24 == 0)
            actual = (pm[(b, e)] == 1)
            if char != actual:
                failures.append(
                    f"({b},{e}): 24|gcd={char} but period-1={actual} — characterization mismatch"
                )

    return failures


# ---------------------------------------------------------------------------
# C3 — Applied Satellite: {8,16,24}² \ {(24,24)} all have period 8
# ---------------------------------------------------------------------------
def check_c3(pm):
    failures = []

    satellite_set = {(8 * k, 8 * l) for k in range(1, 4) for l in range(1, 4)} - {(24, 24)}

    if len(satellite_set) != 8:
        failures.append(f"Satellite candidate count = {len(satellite_set)}, expected 8")

    for s in satellite_set:
        p = pm[s]
        if p != 8:
            failures.append(f"Satellite state {s} has period {p}, expected 8")

    for s, p in pm.items():
        if p == 8 and s not in satellite_set:
            failures.append(f"Non-satellite state {s} has period 8")

    # Characterization: 8|gcd(b,e) and 3∤gcd(b,e)
    period_8_states = {s for s, p in pm.items() if p == 8}
    char_set = {(b, e) for b in range(1, 25) for e in range(1, 25)
                if gcd(b, e) % 8 == 0 and gcd(b, e) % 3 != 0}
    if char_set != period_8_states:
        failures.append(
            f"Characterization mismatch: char_count={len(char_set)}, period_8_count={len(period_8_states)}"
        )

    # Structural parallel with mod-9 Satellite
    sat_9 = {(3 * k, 3 * l) for k in range(1, 4) for l in range(1, 4)} - {(9, 9)}
    if len(sat_9) != 8:
        failures.append(f"Mod-9 Satellite analog count = {len(sat_9)}, expected 8")

    for s in sat_9:
        p9 = orbit_period(s[0], s[1], m=9)
        if p9 != 8:
            failures.append(f"Mod-9 Satellite state {s} has period {p9}, expected 8")

    return failures


# ---------------------------------------------------------------------------
# C4 — Applied Cosmos: 504 period-24 states = 21 orbits × 24
# ---------------------------------------------------------------------------
def check_c4(pm):
    failures = []

    cosmos_states = {s for s, p in pm.items() if p == 24}

    if len(cosmos_states) != 504:
        failures.append(f"Applied Cosmos count = {len(cosmos_states)}, expected 504")

    # Characterization: 3∤gcd(b,e) AND 8∤gcd(b,e)
    char_cosmos = {(b, e) for b in range(1, 25) for e in range(1, 25)
                   if gcd(b, e) % 3 != 0 and gcd(b, e) % 8 != 0}
    if char_cosmos != cosmos_states:
        failures.append(
            f"Characterization mismatch: char={len(char_cosmos)}, cosmos={len(cosmos_states)}"
        )

    # Inclusion-exclusion: 576 - |3|gcd| - |8|gcd| + |24|gcd|
    three_div = sum(1 for b in range(1, 25) for e in range(1, 25) if gcd(b, e) % 3 == 0)
    eight_div = sum(1 for b in range(1, 25) for e in range(1, 25) if gcd(b, e) % 8 == 0)
    tf_div    = sum(1 for b in range(1, 25) for e in range(1, 25) if gcd(b, e) % 24 == 0)

    if three_div != 64:
        failures.append(f"|3|gcd| = {three_div}, expected 64")
    if eight_div != 9:
        failures.append(f"|8|gcd| = {eight_div}, expected 9")
    if tf_div != 1:
        failures.append(f"|24|gcd| = {tf_div}, expected 1")

    ie_count = 576 - three_div - eight_div + tf_div
    if ie_count != 504:
        failures.append(f"Inclusion-exclusion: 576-{three_div}-{eight_div}+{tf_div} = {ie_count}, expected 504")

    # 21 orbits of length exactly 24
    seen = set()
    orbits = []
    for s0 in sorted(cosmos_states):
        if s0 in seen:
            continue
        orbit = [s0]
        b, e = t_step(*s0)
        while (b, e) != s0:
            orbit.append((b, e))
            b, e = t_step(b, e)
        if len(orbit) != 24:
            failures.append(f"Cosmos orbit from {s0} has length {len(orbit)}, expected 24")
        seen.update(orbit)
        orbits.append(orbit)

    if len(orbits) != 21:
        failures.append(f"Cosmos orbit count = {len(orbits)}, expected 21")
    if len(seen) != 504:
        failures.append(f"Cosmos states covered = {len(seen)}, expected 504")

    return failures


# ---------------------------------------------------------------------------
# C5 — Closing the loop: max period = π(24) = π(9) = 24
# ---------------------------------------------------------------------------
def check_c5(pm):
    failures = []
    from math import lcm

    max_period = max(pm.values())
    if max_period != 24:
        failures.append(f"Max orbit period = {max_period}, expected 24")

    # π(24) = lcm(π(3), π(8)) = lcm(8, 12) = 24  (cert [302])
    pi_3  = 8
    pi_8  = 12
    pi_24 = lcm(pi_3, pi_8)
    if pi_24 != 24:
        failures.append(f"π(24) = lcm({pi_3},{pi_8}) = {pi_24}, expected 24")

    # π(9) = 24 (cert [291]): Fibonacci mod-9 has period 24
    def fib_pisano(m):
        a, b = 0, 1
        for k in range(1, m * m + 2):
            a, b = b, (a + b) % m
            if (a, b) == (0, 1):
                return k
        raise RuntimeError(f"Pisano period not found for m={m}")

    pi_9 = fib_pisano(9)
    if pi_9 != 24:
        failures.append(f"π(9) = {pi_9}, expected 24 (cert [291])")

    if not (max_period == pi_24 == pi_9 == 24):
        failures.append(
            f"Closing-loop alignment failed: max_period={max_period}, π(24)={pi_24}, π(9)={pi_9}"
        )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pm = build_period_map()

    checks = [
        ("C1_period_ceiling_24",             check_c1, {"pm": pm}),
        ("C2_applied_singularity",           check_c2, {"pm": pm}),
        ("C3_applied_satellite_analog",      check_c3, {"pm": pm}),
        ("C4_applied_cosmos_504_states",     check_c4, {"pm": pm}),
        ("C5_closing_the_loop_pi24_eq_pi9",  check_c5, {"pm": pm}),
    ]
    all_pass = True
    for label, fn, kwargs in checks:
        failures = fn(**kwargs)
        status = "PASS" if not failures else "FAIL"
        if failures:
            all_pass = False
        suffix = f" — {failures[0]}" if failures else ""
        print(f"  {label}: {status}{suffix}")

    print()
    if all_pass:
        print("CERT [306] PASS — QA Pisano Mod-24 Applied Cosmos Period")
    else:
        print("CERT [306] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
