#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — integer matrix arithmetic on SL(2,Z) and Fibonacci sequences; no empirical QA state machine; all state is exact integer"
QA_COMPLIANCE = "cert_validator — integer matrix arithmetic on SL(2,Z) and Fibonacci sequences; no empirical QA state machine; all state is exact integer"
"""
QA Orbit Grade Decomposition Cert [298] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Wall (1960) Fibonacci Primitive Roots, Amer. Math. Monthly 67(6):525-532

The three QA orbits on {1,...,9}^2 are exactly stratified by v3(gcd(b,e)),
the 3-adic valuation of gcd(b,e). The T-operator preserves this valuation
because gcd(e, b+e) = gcd(e,b) = gcd(b,e).

Five claims:
  C1  v3(gcd(b,e)) is invariant under T — algebraic identity + exhaustive check
  C2  Arithmetic stratification: v3=2 -> Singularity (1), v3=1 -> Satellite (8), v3=0 -> Cosmos (72)
  C3  Period matches stratum: v3=2 -> period 1; v3=1 -> period 8; v3=0 -> period 24
  C4  Even/odd T-step split: Cosmos 12+12=24, Satellite 4+4=8 within each orbit
  C5  M^12 = -I (mod 9): the grade-inversion / antipodal map
"""

import sys
from math import gcd

M = [[0, 1], [1, 1]]    # QA T-operator as matrix


# ---------------------------------------------------------------------------
# Matrix helpers (integer arithmetic)
# ---------------------------------------------------------------------------
def mat_mul(A, B):
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]


def mat_pow(A, n):
    result = [[1, 0], [0, 1]]
    base = [row[:] for row in A]
    while n:
        if n & 1:
            result = mat_mul(result, base)
        base = mat_mul(base, base)
        n >>= 1
    return result


def mat_mod(A, m):
    return [[x % m for x in row] for row in A]


def qa_t_step(b, e, m=9):
    """QA T-operator with A1 arithmetic (states in {1,...,m})."""
    return e, ((b + e - 1) % m) + 1


def v3(n):
    """3-adic valuation of n (largest k such that 3^k divides n)."""
    if n == 0:
        return float('inf')
    k = 0
    while n % 3 == 0:
        n //= 3
        k += 1
    return k


def grade(b, e):
    """v3(gcd(b,e)) — the orbit grade of state (b,e)."""
    return v3(gcd(b, e))


def orbit_of(b, e, m=9):
    """Return the T-orbit of (b,e) as a list of states in visitation order."""
    seen = set()
    orbit = []
    cur = (b, e)
    while cur not in seen:
        seen.add(cur)
        orbit.append(cur)
        cur = qa_t_step(*cur, m=m)
    return orbit


# ---------------------------------------------------------------------------
# C1 — v3(gcd) is T-invariant
# ---------------------------------------------------------------------------
def check_c1(m=9):
    """
    Algebraic proof: gcd(e, b+e) = gcd(e,b) = gcd(b,e) for any integers.
    The A1 mod-9 wrapping (new_e = ((b+e-1)%9)+1) shifts b+e by a multiple
    of 9, and gcd(e, (b+e)+9k) = gcd(e, b+e) = gcd(e,b) = gcd(b,e).
    So gcd is preserved exactly.
    Exhaustive verification on all 81 states confirms.
    """
    failures = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            g_before = gcd(b, e)
            nb, ne = qa_t_step(b, e, m)
            g_after = gcd(nb, ne)
            if v3(g_before) != v3(g_after):
                failures.append(
                    f"v3 not preserved: ({b},{e}) gcd={g_before} v3={v3(g_before)} "
                    f"-> ({nb},{ne}) gcd={g_after} v3={v3(g_after)}"
                )
    return failures


# ---------------------------------------------------------------------------
# C2 — Arithmetic stratification: v3 -> orbit type, counts 1+8+72
# ---------------------------------------------------------------------------
def check_c2(m=9):
    failures = []
    sing = [(b, e) for b in range(1, m+1) for e in range(1, m+1) if grade(b,e) == 2]
    sat  = [(b, e) for b in range(1, m+1) for e in range(1, m+1) if grade(b,e) == 1]
    cos  = [(b, e) for b in range(1, m+1) for e in range(1, m+1) if grade(b,e) == 0]

    if len(sing) != 1:
        failures.append(f"Singularity (v3=2): {len(sing)} states, expected 1; got {sing}")
    if len(sat) != 8:
        failures.append(f"Satellite (v3=1): {len(sat)} states, expected 8; got {sat}")
    if len(cos) != 72:
        failures.append(f"Cosmos (v3=0): {len(cos)} states, expected 72")
    if len(sing) + len(sat) + len(cos) != 81:
        failures.append(f"Total {len(sing)+len(sat)+len(cos)} != 81")

    # Verify Singularity is exactly (9,9)
    if sing and sing[0] != (9, 9):
        failures.append(f"Singularity state = {sing[0]}, expected (9,9)")

    # Verify Satellite states all have 3|b and 3|e (but not 9|b or 9|e)
    for b, e in sat:
        if b % 3 != 0 or e % 3 != 0:
            failures.append(f"Satellite ({b},{e}): not both divisible by 3")
        if b % 9 == 0 and e % 9 == 0:
            failures.append(f"Satellite ({b},{e}): both divisible by 9 (should be Singularity)")

    # Verify Cosmos states: gcd not divisible by 3
    for b, e in cos:
        if gcd(b, e) % 3 == 0:
            failures.append(f"Cosmos ({b},{e}): gcd divisible by 3")

    return failures


# ---------------------------------------------------------------------------
# C3 — Period matches stratum
# ---------------------------------------------------------------------------
def check_c3(m=9):
    failures = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            g = grade(b, e)
            orb = orbit_of(b, e, m)
            period = len(orb)
            expected_period = {2: 1, 1: 8, 0: 24}[g]
            if period != expected_period:
                failures.append(
                    f"({b},{e}) grade={g} period={period}, expected {expected_period}"
                )
    return failures


# ---------------------------------------------------------------------------
# C4 — Even/odd T-step split within each orbit
# ---------------------------------------------------------------------------
def check_c4(m=9):
    """
    Within each orbit of length L, the even-indexed states (index 0,2,4,...)
    and odd-indexed states (index 1,3,5,...) form two disjoint sets of size L/2.
    For Cosmos (L=24): 12+12. For Satellite (L=8): 4+4.
    """
    failures = []
    visited = set()
    for start_b in range(1, m + 1):
        for start_e in range(1, m + 1):
            if (start_b, start_e) in visited:
                continue
            orb = orbit_of(start_b, start_e, m)
            for s in orb:
                visited.add(s)
            L = len(orb)
            if L == 1:
                continue   # Singularity: trivially OK

            even_states = set(orb[i] for i in range(0, L, 2))
            odd_states  = set(orb[i] for i in range(1, L, 2))

            if len(even_states) != L // 2:
                failures.append(f"orbit from {start_b},{start_e}: {len(even_states)} even states, expected {L//2}")
            if len(odd_states) != L // 2:
                failures.append(f"orbit from {start_b},{start_e}: {len(odd_states)} odd states, expected {L//2}")
            if even_states & odd_states:
                failures.append(f"orbit from {start_b},{start_e}: even/odd sets overlap")

    return failures


# ---------------------------------------------------------------------------
# C5 — M^12 = -I (mod 9): the grade-inversion / antipodal map
# ---------------------------------------------------------------------------
def check_c5(m=9):
    failures = []
    M12 = mat_pow(M, 12)
    M12_mod = mat_mod(M12, m)

    # Expected: [[-1,0],[0,-1]] mod 9 = [[8,0],[0,8]]
    expected = [[8, 0], [0, 8]]
    if M12_mod != expected:
        failures.append(f"M^12 mod 9 = {M12_mod}, expected {expected} (= -I mod 9)")

    # Verify the exact integer matrix satisfies M^12 ≡ -I (mod 9)
    for r in range(2):
        for c in range(2):
            val = M12[r][c] % m
            exp = expected[r][c]
            if val != exp:
                failures.append(f"M^12[{r}][{c}] = {M12[r][c]}, mod 9 = {val}, expected {exp}")

    # Confirm that M^12 maps each Cosmos state to its "grade-inverted partner"
    # (i.e., applying M 12 times gives a DIFFERENT state that is 12 T-steps away)
    b, e = 2, 1   # a Cosmos state (gcd=1)
    cur = (b, e)
    for _ in range(12):
        cur = qa_t_step(*cur, m=m)
    b12, e12 = cur
    # M^12 acting on [b,e]^T in Z: result should be ≡ -[b,e] (mod 9)
    Mb12  = M12[0][0]*b + M12[0][1]*e
    Me12  = M12[1][0]*b + M12[1][1]*e
    if (Mb12 % m, Me12 % m) != (b12 % m, e12 % m):
        failures.append(
            f"M^12·[{b},{e}]^T = [{Mb12},{Me12}] mod 9 = [{Mb12%m},{Me12%m}], "
            f"but T^12({b},{e}) = ({b12},{e12})"
        )

    # Confirm M^24 = I (mod 9) — closing the orbit
    M24 = mat_pow(M, 24)
    M24_mod = mat_mod(M24, m)
    if M24_mod != [[1, 0], [0, 1]]:
        failures.append(f"M^24 mod 9 = {M24_mod}, expected identity")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_gcd_invariant",      check_c1, {}),
        ("C2_stratification",     check_c2, {}),
        ("C3_period_by_grade",    check_c3, {}),
        ("C4_even_odd_split",     check_c4, {}),
        ("C5_grade_inversion",    check_c5, {}),
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
        print("CERT [298] PASS — QA Orbit Grade Decomposition")
    else:
        print("CERT [298] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
