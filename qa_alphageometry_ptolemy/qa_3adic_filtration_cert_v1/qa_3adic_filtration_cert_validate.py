#!/usr/bin/env python3
"""
QA 3-Adic Filtration Cert [301] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Wall (1960) Fibonacci Primitive Roots, Amer. Math. Monthly 67(6):525-532

Working in Z/9Z = {0,...,8} (NOT A1 {1,...,9}) so that multiplication is linear.
T_0(b,e) = (e, (b+e) mod 9)  — the raw (no-offset) QA step.

mu3(b,e) = (3b mod 9, 3e mod 9) — multiplication by 3.

The orbit strata in Z/9Z:
  Singularity: gcd(b,e,9) = 9, i.e. b=e=0   → {(0,0)}                 (1 state)
  Satellite:   3|b and 3|e, not both 0         → 8 states
  Cosmos:      not (3|b and 3|e)               → 72 states

Five claims:
  C1  mu3 maps all 72 Cosmos states to Satellite states
  C2  mu3 maps all 8 Satellite states to (0,0) = Singularity
  C3  mu3 is 9-to-1: each Satellite state has exactly 9 Cosmos preimages
  C4  Intertwining: T_0 o mu3 = mu3 o T_0 on all 81 states (linearity of T_0)
  C5  GF(3) layer: x²-x-1 irreducible over Z/3Z; M mod 3 has order 8;
      Cosmos states reduce 9-to-1 onto (Z/3Z)^2 minus {(0,0)}
"""

import sys
from math import gcd


M = [[0, 1], [1, 1]]


def v3(n):
    if n == 0:
        return float('inf')
    k = 0
    while n % 3 == 0:
        n //= 3
        k += 1
    return k


def orbit_grade(b, e):
    """v3(gcd(b,e)) — 0=Cosmos, 1=Satellite, 2+=Singularity. Works in Z/9Z."""
    g = gcd(b, e)
    if g == 0:
        return 2   # (0,0) is Singularity
    return v3(g)


def t0(b, e, m=9):
    """Raw (no-offset) QA step in Z/mZ."""
    return e % m, (b + e) % m


def mu3(b, e, m=9):
    """Multiplication-by-3 map in Z/mZ."""
    return (3 * b) % m, (3 * e) % m


def all_states(m=9):
    return [(b, e) for b in range(m) for e in range(m)]


def mat_pow_mod(A, n, m):
    result = [[1, 0], [0, 1]]
    base = [[x % m for x in row] for row in A]
    while n:
        if n & 1:
            result = [
                [(result[r][0]*base[0][c] + result[r][1]*base[1][c]) % m for c in range(2)]
                for r in range(2)
            ]
        base = [
            [(base[r][0]*base[0][c] + base[r][1]*base[1][c]) % m for c in range(2)]
            for r in range(2)
        ]
        n >>= 1
    return result


# ---------------------------------------------------------------------------
# C1 — mu3 maps Cosmos → Satellite
# ---------------------------------------------------------------------------
def check_c1(m=9):
    failures = []
    for b, e in all_states(m):
        g = orbit_grade(b, e)
        if g != 0:
            continue   # only check Cosmos states
        mb, me = mu3(b, e, m)
        mg = orbit_grade(mb, me)
        if mg != 1:
            failures.append(f"mu3({b},{e})=({mb},{me}) grade={mg}, expected 1 (Satellite)")
    # Count: exactly 72 Cosmos states should be mapped
    cosmos = [(b, e) for b, e in all_states(m) if orbit_grade(b, e) == 0]
    if len(cosmos) != 72:
        failures.append(f"Cosmos count = {len(cosmos)}, expected 72")
    return failures


# ---------------------------------------------------------------------------
# C2 — mu3 maps Satellite → Singularity
# ---------------------------------------------------------------------------
def check_c2(m=9):
    failures = []
    for b, e in all_states(m):
        if orbit_grade(b, e) != 1:
            continue   # only Satellite states
        mb, me = mu3(b, e, m)
        if (mb, me) != (0, 0):
            failures.append(f"mu3({b},{e})=({mb},{me}), expected (0,0)")
    sat = [(b, e) for b, e in all_states(m) if orbit_grade(b, e) == 1]
    if len(sat) != 8:
        failures.append(f"Satellite count = {len(sat)}, expected 8")
    return failures


# ---------------------------------------------------------------------------
# C3 — mu3 is 9-to-1: each Satellite state has 9 Cosmos preimages
# ---------------------------------------------------------------------------
def check_c3(m=9):
    failures = []
    cosmos = [(b, e) for b, e in all_states(m) if orbit_grade(b, e) == 0]
    satellite = [(b, e) for b, e in all_states(m) if orbit_grade(b, e) == 1]

    from collections import Counter
    preimage_counts = Counter()
    for b, e in cosmos:
        image = mu3(b, e, m)
        preimage_counts[image] += 1

    for s in satellite:
        count = preimage_counts.get(s, 0)
        if count != 9:
            failures.append(f"Satellite state {s} has {count} Cosmos preimages, expected 9")

    # Total preimages should equal 72 = 8 × 9
    total = sum(preimage_counts.values())
    if total != 72:
        failures.append(f"Total Cosmos preimage count = {total}, expected 72")

    return failures


# ---------------------------------------------------------------------------
# C4 — Intertwining: T_0 o mu3 = mu3 o T_0
# ---------------------------------------------------------------------------
def check_c4(m=9):
    """
    Algebraic proof: T_0 is linear (T_0(3v) = 3*T_0(v) mod 9).
    T_0(3b, 3e) = (3e, (3b+3e) mod 9) = 3*(e, (b+e) mod 3) mod 9
               = mu3(T_0(b,e)) since 3*(b+e) mod 9 = 3*((b+e) mod 3) mod 9
    Wait: 3*(b+e) mod 9. If b+e = 9k+r, then 3*(b+e) = 27k+3r ≡ 3r mod 9 = 3*((b+e) mod 3) ...
    Actually 3*(b+e) mod 9 = (3*(b+e)) mod 9, and (b+e) mod 9 maps to 3*(b+e mod 9) mod 9 too.
    The point is T_0 is an affine-linear map and multiplication by 3 is a ring homomorphism mod 9.
    Verified exhaustively here.
    """
    failures = []
    for b, e in all_states(m):
        # Path 1: mu3 then T_0
        mb, me = mu3(b, e, m)
        path1 = t0(mb, me, m)
        # Path 2: T_0 then mu3
        tb, te = t0(b, e, m)
        path2 = mu3(tb, te, m)
        if path1 != path2:
            failures.append(
                f"({b},{e}): T_0(mu3) = {path1}, mu3(T_0) = {path2}"
            )
    return failures


# ---------------------------------------------------------------------------
# C5 — GF(3) layer: irreducibility, order of M mod 3, 9-to-1 reduction
# ---------------------------------------------------------------------------
def check_c5(m=9, p=3):
    failures = []

    # C5a: x²-x-1 irreducible over Z/3Z (no roots)
    for x in range(p):
        val = (x*x - x - 1) % p
        if val == 0:
            failures.append(f"x²-x-1 has root x={x} mod {p}")

    # C5b: M mod 3 has order exactly 8
    I_mod3 = [[1, 0], [0, 1]]
    for k in [1, 2, 4]:   # proper divisors of 8
        Mk = mat_pow_mod(M, k, p)
        if Mk == I_mod3:
            failures.append(f"M^{k} = I mod {p} (order divides {k}, should be 8)")
    M8 = mat_pow_mod(M, 8, p)
    if M8 != I_mod3:
        failures.append(f"M^8 mod {p} = {M8}, expected identity")

    # C5c: Cosmos states reduce 9-to-1 onto (Z/3Z)²\{(0,0)}
    cosmos = [(b, e) for b, e in all_states(m) if orbit_grade(b, e) == 0]
    from collections import Counter
    reduction = Counter((b % p, e % p) for b, e in cosmos)

    # Should hit all 8 nonzero elements of (Z/3Z)², each exactly 9 times
    nonzero_gf3 = [(b, e) for b in range(p) for e in range(p) if (b, e) != (0, 0)]
    if len(nonzero_gf3) != 8:
        failures.append(f"(Z/3Z)²\\{{0}} has {len(nonzero_gf3)} elements, expected 8")

    for point in nonzero_gf3:
        count = reduction.get(point, 0)
        if count != 9:
            failures.append(f"Cosmos reduction: point {point} has {count} preimages, expected 9")

    if reduction.get((0, 0), 0) != 0:
        failures.append(f"Cosmos states reduce to (0,0) mod 3: {reduction.get((0,0),0)} times, expected 0")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_cosmos_to_satellite",     check_c1, {}),
        ("C2_satellite_to_singularity",check_c2, {}),
        ("C3_nine_to_one",             check_c3, {}),
        ("C4_intertwining",            check_c4, {}),
        ("C5_gf3_layer",               check_c5, {}),
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
        print("CERT [301] PASS — QA 3-Adic Filtration")
    else:
        print("CERT [301] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
