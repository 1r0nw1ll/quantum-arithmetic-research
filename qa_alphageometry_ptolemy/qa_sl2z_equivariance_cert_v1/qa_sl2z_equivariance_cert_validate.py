#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — integer matrix arithmetic on SL(2,Z) and Fibonacci sequences; no empirical QA state machine; all state is exact integer"
"""
QA SL(2,Z) Equivariance Cert [300] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Hestenes and Sobczyk (1984) Clifford Algebra to Geometric Calculus, Reidel, ISBN 978-90-277-1673-6

Cert [298] showed T=M preserves v3(gcd). This cert lifts that to ALL of SL(2,Z):
  - L·[b,e]^T preserves gcd (gcd(b,b+e)=gcd(b,e))
  - R·[b,e]^T preserves gcd (gcd(b+e,e)=gcd(b,e))
  - By [294], L and R generate SL(2,Z) → all W∈SL(2,Z) preserve gcd

Then the versor sandwich W·M·W⁻¹ preserves both char poly and orbit partition.

Five claims:
  C1  L preserves gcd: gcd(b, b+e) = gcd(b,e) — algebraic identity + exhaustive on {1,...,9}²
  C2  R preserves gcd: gcd(b+e, e) = gcd(b,e) — algebraic identity + exhaustive on {1,...,9}²
  C3  Full equivariance: all W∈{L,R}*, length 1..8, preserve gcd on all 81 states
  C4  Sandwich preserves char poly: L·M·L⁻¹=[[8,1],[8,2]] mod 9; R·M·R⁻¹=[[1,1],[1,0]] mod 9;
      both have trace=1, det=-1, char poly x²-x-1 (same as M)
  C5  Sandwich preserves orbit partition: L·M·L⁻¹ and R·M·R⁻¹ each produce 1+8+72
"""

import sys
from math import gcd
from itertools import product as iproduct


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------
L = [[1, 0], [1, 1]]
R = [[1, 1], [0, 1]]
M = [[0, 1], [1, 1]]


def mat_mul(A, B):
    return [
        [A[0][0]*B[0][0]+A[0][1]*B[1][0],  A[0][0]*B[0][1]+A[0][1]*B[1][1]],
        [A[1][0]*B[0][0]+A[1][1]*B[1][0],  A[1][0]*B[0][1]+A[1][1]*B[1][1]],
    ]


def mat_inv_int(A):
    """Integer inverse of a 2×2 integer matrix with det ±1."""
    d = det(A)
    assert abs(d) == 1
    return [[d * A[1][1], -d * A[0][1]],
            [-d * A[1][0],  d * A[0][0]]]


def mat_mod(A, m):
    return [[x % m for x in row] for row in A]


def mat_vec(A, v):
    return [A[0][0]*v[0]+A[0][1]*v[1], A[1][0]*v[0]+A[1][1]*v[1]]


def det(A):
    return A[0][0]*A[1][1] - A[0][1]*A[1][0]


def trace(A):
    return A[0][0] + A[1][1]


def mat_pow(A, n):
    result = [[1, 0], [0, 1]]
    base = [row[:] for row in A]
    while n:
        if n & 1:
            result = mat_mul(result, base)
        base = mat_mul(base, base)
        n >>= 1
    return result


def qa_t_op(A, m=9):
    """Apply matrix A as QA step with A1 arithmetic (states {1,...,m})."""
    def step(b, e):
        nb, ne = mat_vec(A, [b - 1, e - 1])  # shift to {0,...,m-1}
        return (nb % m) + 1, (ne % m) + 1
    return step


def orbit_of(step_fn, b, e):
    seen, orbit = set(), []
    cur = (b, e)
    while cur not in seen:
        seen.add(cur)
        orbit.append(cur)
        cur = step_fn(*cur)
    return orbit


# ---------------------------------------------------------------------------
# C1 — L preserves gcd: gcd(b, b+e) = gcd(b,e)
# ---------------------------------------------------------------------------
def check_c1(m=9):
    """
    Algebraic identity: gcd(b, b+e) = gcd(b,e).
    Proof: any common divisor of b and b+e also divides (b+e)-b=e, so
    divides gcd(b,e); conversely gcd(b,e) divides b+e.
    """
    failures = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            lv = mat_vec(L, [b, e])
            if gcd(lv[0], lv[1]) != gcd(b, e):
                failures.append(f"L: gcd({lv[0]},{lv[1]}) != gcd({b},{e})")
    # Also verify the algebraic identity for larger values
    for b in range(1, 50):
        for e in range(1, 50):
            if gcd(b, b + e) != gcd(b, e):
                failures.append(f"algebraic: gcd({b},{b+e}) != gcd({b},{e})")
    return failures


# ---------------------------------------------------------------------------
# C2 — R preserves gcd: gcd(b+e, e) = gcd(b,e)
# ---------------------------------------------------------------------------
def check_c2(m=9):
    failures = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            rv = mat_vec(R, [b, e])
            if gcd(rv[0], rv[1]) != gcd(b, e):
                failures.append(f"R: gcd({rv[0]},{rv[1]}) != gcd({b},{e})")
    for b in range(1, 50):
        for e in range(1, 50):
            if gcd(b + e, e) != gcd(b, e):
                failures.append(f"algebraic: gcd({b+e},{e}) != gcd({b},{e})")
    return failures


# ---------------------------------------------------------------------------
# C3 — Full equivariance: all W∈{L,R}* of length 1..8 preserve gcd
# ---------------------------------------------------------------------------
def check_c3(max_len=8, m=9):
    """
    Enumerate all words of length 1..max_len in {L,R}*.
    For each, verify gcd(W·[b,e]^T) = gcd(b,e) for all 81 states.
    """
    failures = []
    # BFS over words, keyed by (matrix_entry_tuple)
    seen_matrices = set()
    frontier = [([[1,0],[0,1]], [])]   # (matrix, word_chars)

    for step_len in range(1, max_len + 1):
        next_frontier = []
        for W, word in frontier:
            for gen, gname in [(L, 'L'), (R, 'R')]:
                W2 = mat_mul(W, gen)
                key = tuple(tuple(r) for r in W2)
                if key in seen_matrices:
                    continue
                seen_matrices.add(key)
                word2 = word + [gname]
                # Verify gcd preservation on all 81 states
                for b in range(1, m + 1):
                    for e in range(1, m + 1):
                        wv = mat_vec(W2, [b, e])
                        if gcd(wv[0], wv[1]) != gcd(b, e):
                            failures.append(
                                f"W={''.join(word2)}: gcd({wv[0]},{wv[1]}) "
                                f"!= gcd({b},{e})"
                            )
                            if len(failures) >= 3:
                                return failures
                next_frontier.append((W2, word2))
        frontier = next_frontier

    return failures


# ---------------------------------------------------------------------------
# C4 — Sandwich preserves char poly: L·M·L⁻¹ and R·M·R⁻¹
# ---------------------------------------------------------------------------
def check_c4():
    failures = []
    L_inv = mat_inv_int(L)
    R_inv = mat_inv_int(R)

    LML = mat_mul(mat_mul(L, M), L_inv)
    RMR = mat_mul(mat_mul(R, M), R_inv)

    # Expected: L·M·L⁻¹ = [[-1,1],[-1,2]] (exact integer), mod 9 = [[8,1],[8,2]]
    # Expected: R·M·R⁻¹ = [[1,1],[1,0]] (exact integer)
    LML_expected = [[-1, 1], [-1, 2]]
    RMR_expected = [[1, 1], [1, 0]]

    if LML != LML_expected:
        failures.append(f"L·M·L⁻¹ = {LML}, expected {LML_expected}")
    if RMR != RMR_expected:
        failures.append(f"R·M·R⁻¹ = {RMR}, expected {RMR_expected}")

    # mod-9 representations
    LML_mod9 = mat_mod(LML, 9)
    RMR_mod9 = mat_mod(RMR, 9)
    if LML_mod9 != [[8, 1], [8, 2]]:
        failures.append(f"L·M·L⁻¹ mod 9 = {LML_mod9}, expected [[8,1],[8,2]]")
    if RMR_mod9 != [[1, 1], [1, 0]]:
        failures.append(f"R·M·R⁻¹ mod 9 = {RMR_mod9}, expected [[1,1],[1,0]]")

    # Both must have trace=1, det=-1 (same as M)
    for name, mat in [("L·M·L⁻¹", LML), ("R·M·R⁻¹", RMR)]:
        if trace(mat) != 1:
            failures.append(f"trace({name}) = {trace(mat)}, expected 1")
        if det(mat) != -1:
            failures.append(f"det({name}) = {det(mat)}, expected -1")

    # Char poly x²-x-1: verify N²=N+I for each conjugate
    for name, N in [("L·M·L⁻¹", LML), ("R·M·R⁻¹", RMR)]:
        N2 = mat_mul(N, N)
        N_plus_I = [[N[r][c] + (1 if r == c else 0) for c in range(2)] for r in range(2)]
        if N2 != N_plus_I:
            failures.append(f"{name}² != {name}+I: {N2} vs {N_plus_I}")

    return failures


# ---------------------------------------------------------------------------
# C5 — Sandwich preserves orbit partition 1+8+72
# ---------------------------------------------------------------------------
def _orbit_partition(op_matrix, m=9):
    """Compute orbit sizes under the given 2×2 matrix acting on {1,...,m}²."""
    step = qa_t_op(op_matrix, m)
    visited = set()
    sizes = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in visited:
                continue
            orb = orbit_of(step, b, e)
            for s in orb:
                visited.add(s)
            sizes.append(len(orb))
    return sorted(sizes)


def check_c5(m=9):
    failures = []
    L_inv = mat_inv_int(L)
    R_inv = mat_inv_int(R)
    LML = mat_mul(mat_mul(L, M), L_inv)
    RMR = mat_mul(mat_mul(R, M), R_inv)

    # Expected partition: one orbit size 1, one size 8, three size 24 → 1+8+72=81
    from collections import Counter

    for name, N in [("L·M·L⁻¹", LML), ("R·M·R⁻¹", RMR), ("M (reference)", M)]:
        sizes = _orbit_partition(N, m)
        c = Counter(sizes)
        expected = {1: 1, 8: 1, 24: 3}
        if c != expected:
            failures.append(f"{name}: orbit size distribution = {dict(c)}, expected {expected}")
        sing = sum(s for s in sizes if s == 1)
        sat  = sum(s for s in sizes if s == 8)
        cos  = sum(s for s in sizes if s == 24)
        if (sing, sat, cos) != (1, 8, 72):
            failures.append(f"{name}: partition (sing,sat,cos) = ({sing},{sat},{cos}), expected (1,8,72)")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_L_preserves_gcd",        check_c1, {}),
        ("C2_R_preserves_gcd",        check_c2, {}),
        ("C3_full_equivariance",       check_c3, {"max_len": 8}),
        ("C4_sandwich_char_poly",      check_c4, {}),
        ("C5_sandwich_orbit_partition",check_c5, {}),
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
        print("CERT [300] PASS — QA SL(2,Z) Equivariance")
    else:
        print("CERT [300] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
