"""
Preliminary computational check for the cert-D blocker in QA_QFT_ETCR_CROSSMAP.md.

Question: does a non-trivial T-invariant scalar exist on H_raw = {1..m}^2 under
the Fibonacci-like QA step T_F(b, e) = (a1(b+e), b), where a1(x) = ((x-1) mod m) + 1
is the A1-compliant modular step?

If yes, cert-D's orbit-Dirac-bracket construction has a clean level-set formulation
(constraint function = I(b, e) - I_0 for a chosen level value I_0).

RESULT (2026-04-20):
- For m = 9: I(b,e) = (b^2 - be - e^2)^2 mod 9 is T-invariant with 4 distinct values.
  Separates the 3 Cosmos orbits (I = 1, 4, 7); Singularity and Satellite both at I = 0
  (auxiliary constraint, e.g. T-fixed-point, required to separate).
- For m = 24: same I is T-invariant with 4 distinct values across 30 orbits.
  Coarser than orbit separation; cert-D must combine I-level constraints with
  period constraints phi_n(b,e) = T^n(b,e) - (b,e) inside each level set.

Status: preliminary computational support for MC-3 in the cross-map. NOT cert-validated.
Tied to the specific T_F dynamics; other QA step operators may need re-checking.

Run: python etcr_t_invariant_check.py
"""

from collections import Counter


def a1_step(x: int, m: int) -> int:
    """A1-compliant modular step: output in {1..m}, never 0."""
    return ((x - 1) % m) + 1


def T(b: int, e: int, m: int) -> tuple[int, int]:
    """Fibonacci-like QA step T_F(b, e) = (a1(b + e), b)."""
    return (a1_step(b + e, m), b)


def orbit_of(b0: int, e0: int, m: int, max_steps: int = 200) -> tuple[tuple, int]:
    seen: list[tuple[int, int]] = []
    s = (b0, e0)
    while s not in seen:
        seen.append(s)
        s = T(*s, m)
        if len(seen) > max_steps:
            break
    return tuple(sorted(seen)), len(seen)


def all_orbits(m: int) -> list[tuple[int, tuple]]:
    visited: set[tuple[int, int]] = set()
    orbits: list[tuple[int, tuple]] = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in visited:
                continue
            orb, period = orbit_of(b, e, m)
            visited.update(orb)
            orbits.append((period, orb))
    return orbits


def I_cassini_squared(b: int, e: int, m: int) -> int:
    """Cassini/Lucas determinant squared: (b^2 - be - e^2)^2 mod m.

    Under T_F(b, e) = (b + e, b), b^2 - be - e^2 flips sign per step (classical
    Cassini identity F_{n+1} F_{n-1} - F_n^2 = (-1)^n lifted to (b, e) form).
    Squaring preserves invariance.
    """
    val = (b * b - b * e - e * e) % m
    return (val * val) % m


def I_cassini(b: int, e: int, m: int) -> int:
    return (b * b - b * e - e * e) % m


def I_eisenstein(b: int, e: int, m: int) -> int:
    return (b * b + b * e + e * e) % m


def I_pythagorean(b: int, e: int, m: int) -> int:
    d = (b + e) % m
    a = (b + 2 * e) % m
    return (b * b + d * a) % m


def I_sum(b: int, e: int, m: int) -> int:
    return (b + e) % m


def test_invariant(I_func, m: int, orbits: list) -> tuple[bool, int, list]:
    all_invariant = True
    values_seen: set[int] = set()
    per_orbit: list = []
    for p, orb in orbits:
        vals = [I_func(b, e, m) for (b, e) in orb]
        constant = len(set(vals)) == 1
        if not constant:
            all_invariant = False
        per_orbit.append((p, vals[0] if constant else "VAR", len(set(vals))))
        if constant:
            values_seen.add(vals[0])
    return all_invariant, len(values_seen), per_orbit


INVARIANTS = [
    (I_cassini_squared, "I_cassini^2 = (b^2 - be - e^2)^2 mod m"),
    (I_cassini, "I_cassini = (b^2 - be - e^2) mod m"),
    (I_eisenstein, "I_eisenstein = (b^2 + be + e^2) mod m"),
    (I_pythagorean, "I_pyth = (b^2 + d*a) mod m"),
    (I_sum, "I_sum = (b+e) mod m (sanity check; expect FAIL)"),
]


def run_for(m: int) -> None:
    print("=" * 60)
    print(f"m = {m}: orbit structure under T_F(b,e) = (a1(b+e), b)")
    print("=" * 60)
    orbits = all_orbits(m)
    period_counts = Counter(p for p, _ in orbits)
    print(f"Total pairs: {sum(p for p, _ in orbits)} (expect {m * m})")
    print(f"Period histogram: {dict(period_counts)}")
    print(f"Num orbits: {len(orbits)}")

    for I_func, name in INVARIANTS:
        ok, distinct, per = test_invariant(I_func, m, orbits)
        print(f"\n{name}")
        print(f"  T-invariant: {ok}; distinct values across orbits: {distinct}")
        if ok:
            for p, v, _ in per:
                print(f"    orbit period {p} -> I = {v}")


if __name__ == "__main__":
    run_for(9)
    print()
    run_for(24)
