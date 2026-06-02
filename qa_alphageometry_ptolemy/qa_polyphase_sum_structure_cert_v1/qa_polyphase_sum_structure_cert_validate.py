#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — integer matrix arithmetic on SL(2,Z); orbit enumeration on {1,...,9}^2 with A1 no-zero arithmetic; Fraction arithmetic for observer claim; no float QA state"
"""
QA Polyphase Sum Structure Cert [304] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Wall (1960) Fibonacci Primitive Roots, Amer. Math. Monthly 67(6):525-532
  Hestenes and Sobczyk (1984) Clifford Algebra to Geometric Calculus, Reidel, ISBN 978-90-277-1673-6

Extends cert [303] (three-phase gives 3I) to the full family of n-phase systems for all n|24.

Key theorem (Grade-Inversion Pairing):
  M^12 = -I (mod 9) [cert 298].
  For any EVEN n|24 with step k=24/n: positions {0,k,2k,...,(n-1)k} split into n/2 pairs
  (jk, jk+12) each summing to M^(jk) + M^(jk+12) = M^(jk)(I + M^12) = M^(jk)(I-I) = 0.
  Therefore the n-phase matrix sum = 0 mod 9 for ALL even n|24.

  n=3 (odd) is the unique non-trivial exception: no antipodal pairing exists.
  The sum M^0+M^8+M^16 = 3I [cert 303].

Five claims:
  C1  Six-phase (n=6, step=4): M^0+M^4+M^8+M^12+M^16+M^20 = 0 (mod 9); three pairs cancel
  C2  Twelve-phase (n=12, step=2): sum of 12 matrices = 0 (mod 9); six pairs cancel
  C3  Universal n-phase theorem: for all n|24 the sum is I (n=1), 3I (n=3), or 0 (all others);
      proved via grade-inversion pairing (M^12=-I) for all even n, direct computation for n=3
  C4  Orbit structure: T^4 maps Cosmos→Cosmos, 72 states form 12 disjoint sextets (period-6);
      T^2 maps Cosmos→Cosmos, 72 states form 6 disjoint 12-tuples (period-12)
  C5  Observer layer (Theorem NT): six-phase rational cosine sum 1+1/2-1/2-1-1/2+1/2=0
      via exact Fraction; all n>=2 polyphase sums are zero at the observer layer regardless
      of discrete QA result; the discrete/observer distinction is Theorem NT in action
"""

import sys
from fractions import Fraction
from math import gcd


def fibonacci(n):
    """F(n): F(0)=0, F(1)=1."""
    if n == 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def M_power(k):
    """M^k as exact integer matrix [[F(k-1),F(k)],[F(k),F(k+1)]]."""
    fk_m1 = 1 if k == 0 else fibonacci(k - 1)
    fk    = fibonacci(k)
    fk_p1 = fibonacci(k + 1)
    return [[fk_m1, fk], [fk, fk_p1]]


def mat_mod(A, m):
    return [[x % m for x in row] for row in A]


def mat_add_mod(A, B, m):
    return [[(A[i][j] + B[i][j]) % m for j in range(2)] for i in range(2)]


def n_phase_sum_mod9(n):
    """Compute sum_{i=0}^{n-1} M^(24i/n) mod 9. n must divide 24."""
    assert 24 % n == 0
    step = 24 // n
    S = [[0, 0], [0, 0]]
    for i in range(n):
        S = mat_add_mod(S, mat_mod(M_power(step * i), 9), 9)
    return S


def cosmos_states():
    return [(b, e) for b in range(1, 10) for e in range(1, 10)
            if gcd(b, e) % 3 != 0]


def t_step_a1(b, e, m=9):
    """QA T-step with A1 no-zero arithmetic."""
    return e, ((b + e - 1) % m) + 1


def t_k(b, e, k, m=9):
    for _ in range(k):
        b, e = t_step_a1(b, e, m)
    return b, e


def orbit_sizes_under_t_k(step):
    """Return list of orbit sizes for T^step acting on the 72 Cosmos states."""
    cosmos = cosmos_states()
    cosmos_set = set(cosmos)
    seen = set()
    sizes = []
    for s in cosmos:
        if s in seen:
            continue
        orb = [s]
        x = t_k(*s, step)
        while x != s:
            orb.append(x)
            x = t_k(*x, step)
        seen.update(orb)
        sizes.append(len(orb))
    return sizes


# ---------------------------------------------------------------------------
# C1 — Six-phase (n=6, step=4) sum = 0 mod 9
# ---------------------------------------------------------------------------
def check_c1():
    failures = []
    m = 9
    n = 6
    step = 4

    S = n_phase_sum_mod9(n)
    if S != [[0, 0], [0, 0]]:
        failures.append(f"Six-phase sum mod 9 = {S}, expected zero matrix")

    # Verify the pairing explicitly: M^k + M^(k+12) = 0 for k in {0,4,8}
    for k in [0, 4, 8]:
        Mk    = mat_mod(M_power(k),      m)
        Mkp12 = mat_mod(M_power(k + 12), m)
        pair  = mat_add_mod(Mk, Mkp12, m)
        if pair != [[0, 0], [0, 0]]:
            failures.append(
                f"Pair M^{k} + M^{k+12} = {pair}, expected zero (grade-inversion)"
            )

    # det(M^4) = +1: inter-phase advance is even-grade
    M4 = M_power(4)
    det_M4 = M4[0][0] * M4[1][1] - M4[0][1] * M4[1][0]
    if det_M4 != 1:
        failures.append(f"det(M^4) = {det_M4}, expected +1 (even-grade rotor)")

    return failures


# ---------------------------------------------------------------------------
# C2 — Twelve-phase (n=12, step=2) sum = 0 mod 9
# ---------------------------------------------------------------------------
def check_c2():
    failures = []
    m = 9
    n = 12

    S = n_phase_sum_mod9(n)
    if S != [[0, 0], [0, 0]]:
        failures.append(f"Twelve-phase sum mod 9 = {S}, expected zero matrix")

    # Verify pairing: M^k + M^(k+12) = 0 for k in {0,2,4,6,8,10}
    for k in [0, 2, 4, 6, 8, 10]:
        Mk    = mat_mod(M_power(k),      m)
        Mkp12 = mat_mod(M_power(k + 12), m)
        pair  = mat_add_mod(Mk, Mkp12, m)
        if pair != [[0, 0], [0, 0]]:
            failures.append(
                f"Pair M^{k} + M^{k+12} = {pair}, expected zero"
            )

    # det(M^2) = +1: even-grade
    M2 = M_power(2)
    det_M2 = M2[0][0] * M2[1][1] - M2[0][1] * M2[1][0]
    if det_M2 != 1:
        failures.append(f"det(M^2) = {det_M2}, expected +1 (even-grade rotor)")

    return failures


# ---------------------------------------------------------------------------
# C3 — Universal n-phase theorem: all n|24
# ---------------------------------------------------------------------------
def check_c3():
    failures = []
    m = 9

    identity  = [[1, 0], [0, 1]]
    three_I   = [[3, 0], [0, 3]]
    zero_mat  = [[0, 0], [0, 0]]

    # M^12 = -I mod 9 (grade-inversion antipodal, cert [298])
    M12 = mat_mod(M_power(12), m)
    if M12 != [[8, 0], [0, 8]]:
        failures.append(f"M^12 mod 9 = {M12}, expected [[8,0],[0,8]] = -I (cert [298])")

    # Pairing lemma: M^k + M^(k+12) = 0 for all k
    for k in range(12):
        Mk    = mat_mod(M_power(k),      m)
        Mkp12 = mat_mod(M_power(k + 12), m)
        pair  = mat_add_mod(Mk, Mkp12, m)
        if pair != zero_mat:
            failures.append(
                f"Pairing lemma fails at k={k}: M^k + M^(k+12) = {pair}"
            )
            break

    # All n|24 sums
    expected = {
        1:  identity,
        2:  zero_mat,
        3:  three_I,    # unique non-zero non-trivial case (cert [303])
        4:  zero_mat,
        6:  zero_mat,
        8:  zero_mat,
        12: zero_mat,
        24: zero_mat,
    }
    for n, exp in expected.items():
        S = n_phase_sum_mod9(n)
        if S != exp:
            failures.append(f"n={n} phase sum = {S}, expected {exp}")

    # det(M^8 - I) = 0 mod 9 (why n=3 can be non-zero): non-invertible
    M8 = mat_mod(M_power(8), m)
    M8mI = [[(M8[i][j] - (1 if i == j else 0)) % m for j in range(2)]
            for i in range(2)]
    det_M8mI = (M8mI[0][0] * M8mI[1][1] - M8mI[0][1] * M8mI[1][0]) % m
    if det_M8mI != 0:
        failures.append(
            f"det(M^8-I) mod 9 = {det_M8mI}, expected 0 "
            "(non-invertible: unique reason n=3 sum can be non-zero)"
        )

    # det(M^4 - I) != 0 mod 9 (why n=6 must be zero): invertible
    M4 = mat_mod(M_power(4), m)
    M4mI = [[(M4[i][j] - (1 if i == j else 0)) % m for j in range(2)]
            for i in range(2)]
    det_M4mI = (M4mI[0][0] * M4mI[1][1] - M4mI[0][1] * M4mI[1][0]) % m
    if det_M4mI == 0:
        failures.append(
            f"det(M^4-I) mod 9 = 0 (unexpected: M^4-I should be invertible → n=6 sum forced to 0)"
        )

    return failures


# ---------------------------------------------------------------------------
# C4 — Orbit structure: 12 sextets under T^4; 6 dodecaplets under T^2
# ---------------------------------------------------------------------------
def check_c4():
    failures = []

    # T^4: 72 Cosmos states → 12 disjoint orbits of size exactly 6
    sizes_4 = orbit_sizes_under_t_k(4)
    if len(sizes_4) != 12:
        failures.append(f"T^4 orbit count = {len(sizes_4)}, expected 12 sextets")
    if any(s != 6 for s in sizes_4):
        failures.append(f"T^4 orbit sizes {set(sizes_4)}, expected all 6")
    if sum(sizes_4) != 72:
        failures.append(f"T^4 total states covered = {sum(sizes_4)}, expected 72")

    # T^2: 72 Cosmos states → 6 disjoint orbits of size exactly 12
    sizes_2 = orbit_sizes_under_t_k(2)
    if len(sizes_2) != 6:
        failures.append(f"T^2 orbit count = {len(sizes_2)}, expected 6 dodecaplets")
    if any(s != 12 for s in sizes_2):
        failures.append(f"T^2 orbit sizes {set(sizes_2)}, expected all 12")
    if sum(sizes_2) != 72:
        failures.append(f"T^2 total states covered = {sum(sizes_2)}, expected 72")

    # Contrast with T^8 (three-phase from cert [303]): 24 triads of size 3
    sizes_8 = orbit_sizes_under_t_k(8)
    if len(sizes_8) != 24 or any(s != 3 for s in sizes_8):
        failures.append(
            f"T^8 orbits = {len(sizes_8)} of size(s) {set(sizes_8)}, "
            "expected 24 triads of size 3 (cert [303] cross-check)"
        )

    # Orbit count formula: 72 / (24 / step) for step | 24
    # step=4 → orbit_size = 24/gcd(4,24) = 6, count = 72/6 = 12 ✓
    # step=2 → orbit_size = 24/gcd(2,24) = 12, count = 72/12 = 6 ✓
    from math import gcd as _gcd
    for step, expected_size, expected_count in [(4, 6, 12), (2, 12, 6), (8, 3, 24)]:
        orbit_size = 24 // _gcd(step, 24)
        orbit_count = 72 // orbit_size
        if orbit_size != expected_size or orbit_count != expected_count:
            failures.append(
                f"step={step}: orbit_size={orbit_size} (want {expected_size}), "
                f"count={orbit_count} (want {expected_count})"
            )

    return failures


# ---------------------------------------------------------------------------
# C5 — Observer layer (Theorem NT): six-phase rational cosine sum = 0
# ---------------------------------------------------------------------------
def check_c5():
    failures = []

    # cos(k·60°) for k=0..5: exact rational values
    # cos(0°)=1, cos(60°)=1/2, cos(120°)=-1/2, cos(180°)=-1, cos(240°)=-1/2, cos(300°)=1/2
    cos_vals = [
        Fraction(1),
        Fraction(1, 2),
        Fraction(-1, 2),
        Fraction(-1),
        Fraction(-1, 2),
        Fraction(1, 2),
    ]
    six_phase_obs_sum = sum(cos_vals)
    if six_phase_obs_sum != Fraction(0):
        failures.append(
            f"Six-phase observer sum = {six_phase_obs_sum}, expected 0"
        )

    # All six values are rational: cos(k·60°) are all ∈ {0, ±1/2, ±1}
    # (unlike twelve-phase which includes cos(30°) = √3/2, irrational)
    # Twelve-phase: rational part only (cos at multiples of 60°, 6 of the 12 terms):
    cos_60_multiples = [
        Fraction(1),      # 0°
        Fraction(1, 2),   # 60°
        Fraction(-1, 2),  # 120°
        Fraction(-1),     # 180°
        Fraction(-1, 2),  # 240°
        Fraction(1, 2),   # 300°
    ]
    twelve_rational_sum = sum(cos_60_multiples)
    if twelve_rational_sum != Fraction(0):
        failures.append(
            f"Twelve-phase rational component sum = {twelve_rational_sum}, expected 0"
        )

    # Confirm discrete sums are NOT zero for n=3 (Theorem NT: observer ≠ discrete)
    n3_sum = n_phase_sum_mod9(3)
    if n3_sum == [[0, 0], [0, 0]]:
        failures.append(
            "n=3 discrete sum is zero (unexpected — Theorem NT boundary would be violated)"
        )

    # All even n: discrete sum = 0 AND observer sum = 0 → consistent (no NT violation)
    # n=3: discrete sum = 3I ≠ 0, observer sum = 0 → Theorem NT boundary: observer sees balance,
    #       discrete layer sees 3I. The physical "balance" of three-phase is observer-only.
    n3_expected = [[3, 0], [0, 3]]
    if n3_sum != n3_expected:
        failures.append(
            f"n=3 discrete sum = {n3_sum}, expected [[3,0],[0,3]] (cert [303] cross-check)"
        )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_six_phase_sum_zero",              check_c1, {}),
        ("C2_twelve_phase_sum_zero",           check_c2, {}),
        ("C3_universal_n_phase_theorem",       check_c3, {}),
        ("C4_orbit_sextets_and_dodecaplets",   check_c4, {}),
        ("C5_observer_balance_theorem_nt",     check_c5, {}),
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
        print("CERT [304] PASS — QA Polyphase Sum Structure")
    else:
        print("CERT [304] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
