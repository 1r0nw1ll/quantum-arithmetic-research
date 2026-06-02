#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — integer matrix arithmetic on SL(2,Z); orbit enumeration on {1,...,9}^2 with A1 no-zero arithmetic; Fraction arithmetic for observer claim; no float QA state"
"""
QA Three-Phase Cosmos Cancellation Cert [303] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Wall (1960) Fibonacci Primitive Roots, Amer. Math. Monthly 67(6):525-532
  Hestenes and Sobczyk (1984) Clifford Algebra to Geometric Calculus, Reidel, ISBN 978-90-277-1673-6
  Dollard (1982) A Common Language for Electrical Engineering, Int'l Tesla Society

On the mod-24 QA clock, three balanced phases at T-steps 0, 8, 16 (= 0°, 120°, 240°)
implement the Dollard three-phase versor structure. Five algebraic claims:

  C1  M^0 + M^8 + M^16 ≡ 3·I (mod 9): three equidistant rotation matrices sum to 3×identity
  C2  det(M^8) = +1: the 8-step inter-phase advance is even-grade (rotor); T^1 is odd-grade (versor)
      Consequence: the relationship between balanced three-phase phases is rotor-type (smooth),
      while single-phase AC oscillation is versor-type (sign-reversing).
  C3  (M^8)^3 = M^24 ≡ I (mod 9): M^8 has order exactly 3; three 8-step advances close the cycle
  C4  T^8 maps all 72 Cosmos states to Cosmos; they partition into exactly 24 disjoint triads
      under T^8 (each triad = {(b,e), T^8(b,e), T^16(b,e)}, closed at T^24=(b,e))
  C5  Observer layer (Theorem NT): exact rational sum 1 + (−1/2) + (−1/2) = 0 confirms the
      continuous balance identity cos(0°)+cos(120°)+cos(240°)=0; this is an observer-layer
      fact only — the discrete QA matrix sum (C1) is 3·I, not 0.
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


def cosmos_states():
    """72 Cosmos states: (b,e) in {1,...,9}^2 with 3 ∤ gcd(b,e)."""
    return [(b, e) for b in range(1, 10) for e in range(1, 10)
            if gcd(b, e) % 3 != 0]


def t_step_a1(b, e, m=9):
    """QA T-step with A1 no-zero arithmetic: (b,e) -> (e, ((b+e-1)%m)+1)."""
    return e, ((b + e - 1) % m) + 1


def t_k(b, e, k, m=9):
    for _ in range(k):
        b, e = t_step_a1(b, e, m)
    return b, e


# ---------------------------------------------------------------------------
# C1 — M^0 + M^8 + M^16 ≡ 3·I (mod 9)
# ---------------------------------------------------------------------------
def check_c1():
    failures = []
    m = 9

    M0  = mat_mod(M_power(0),  m)
    M8  = mat_mod(M_power(8),  m)
    M16 = mat_mod(M_power(16), m)

    total = [[0, 0], [0, 0]]
    for A in [M0, M8, M16]:
        total = mat_add_mod(total, A, m)

    expected = [[3, 0], [0, 3]]
    if total != expected:
        failures.append(
            f"M^0+M^8+M^16 mod 9 = {total}, expected [[3,0],[0,3]] (= 3I)"
        )

    # Confirm individual matrices are non-trivial (not I or 0)
    identity = [[1, 0], [0, 1]]
    if M8 == identity:
        failures.append("M^8 mod 9 = I (unexpected — would mean order divides 8, not 24)")
    if M16 == identity:
        failures.append("M^16 mod 9 = I (unexpected — would mean order divides 16, not 24)")

    return failures


# ---------------------------------------------------------------------------
# C2 — det(M^8) = +1 (even-grade rotor)
# ---------------------------------------------------------------------------
def check_c2():
    failures = []

    # Exact integer determinants via M^k = [[F(k-1),F(k)],[F(k),F(k+1)]]
    # det(M^k) = F(k-1)*F(k+1) - F(k)^2 = (-1)^k  (Cassini/d'Ocagne identity)
    M1 = M_power(1)
    det_M1 = M1[0][0] * M1[1][1] - M1[0][1] * M1[1][0]
    if det_M1 != -1:
        failures.append(f"det(M^1) = {det_M1}, expected -1 (T^1 is odd-grade versor)")

    M8 = M_power(8)
    det_M8 = M8[0][0] * M8[1][1] - M8[0][1] * M8[1][0]
    if det_M8 != 1:
        failures.append(f"det(M^8) = {det_M8}, expected +1 (T^8 is even-grade rotor)")

    # Verify Cassini for the first 25 steps: det(M^k) = (-1)^k
    for k in range(25):
        Mk = M_power(k)
        det_k = Mk[0][0] * Mk[1][1] - Mk[0][1] * Mk[1][0]
        expected = (-1) ** k
        if det_k != expected:
            failures.append(
                f"det(M^{k}) = {det_k}, expected {expected} (Cassini identity)"
            )
            break

    return failures


# ---------------------------------------------------------------------------
# C3 — M^8 has order exactly 3 mod 9; (M^8)^3 = M^24 ≡ I (mod 9)
# ---------------------------------------------------------------------------
def check_c3():
    failures = []
    m = 9

    identity = [[1, 0], [0, 1]]

    # M^24 ≡ I mod 9 (the 24-period, from cert [291])
    M24 = mat_mod(M_power(24), m)
    if M24 != identity:
        failures.append(f"M^24 mod 9 = {M24}, expected identity (24-period cert [291])")

    # M^8 ≢ I mod 9 (order > 1)
    M8 = mat_mod(M_power(8), m)
    if M8 == identity:
        failures.append("M^8 mod 9 = I — order would be 1, contradicting 24-period")

    # M^16 ≢ I mod 9 (order > 2)
    M16 = mat_mod(M_power(16), m)
    if M16 == identity:
        failures.append("M^16 mod 9 = I — order would be 2, contradicting 24-period")

    # Exact values for documentation:
    # M^8  mod 9 = [[4,3],[3,7]]  (F(7)=13≡4, F(8)=21≡3, F(9)=34≡7)
    expected_M8  = [[4, 3], [3, 7]]
    expected_M16 = [[7, 6], [6, 4]]
    if M8 != expected_M8:
        failures.append(f"M^8 mod 9 = {M8}, expected {expected_M8}")
    if M16 != expected_M16:
        failures.append(f"M^16 mod 9 = {M16}, expected {expected_M16}")

    # Divisibility: 3 | 24 (three 8-step advances span the full 24-period)
    if 24 % 8 != 0:
        failures.append("8 does not divide 24")
    if 24 // 8 != 3:
        failures.append("24/8 != 3 (should be exactly 3 phases)")

    return failures


# ---------------------------------------------------------------------------
# C4 — T^8 maps Cosmos→Cosmos; 72 states form 24 disjoint triads
# ---------------------------------------------------------------------------
def check_c4():
    failures = []
    cosmos = cosmos_states()

    if len(cosmos) != 72:
        failures.append(f"Cosmos count = {len(cosmos)}, expected 72")
        return failures

    cosmos_set = set(cosmos)

    # T^8 maps every Cosmos state to a Cosmos state
    for b, e in cosmos:
        b8, e8 = t_k(b, e, 8)
        if (b8, e8) not in cosmos_set:
            failures.append(
                f"T^8({b},{e}) = ({b8},{e8}) is not a Cosmos state"
            )
            break

    # Partition into triads of size exactly 3
    seen = set()
    triads = []
    for state in cosmos:
        if state in seen:
            continue
        s0 = state
        s1 = t_k(*s0, 8)
        s2 = t_k(*s1, 8)
        s3 = t_k(*s2, 8)   # should equal s0

        if s3 != s0:
            failures.append(
                f"T^24({s0[0]},{s0[1]}) = {s3}, expected {s0} (24-period)"
            )
        if len({s0, s1, s2}) != 3:
            failures.append(
                f"Triad ({s0},{s1},{s2}) has repeated states — not size-3"
            )
        # All three must be Cosmos
        for s in (s0, s1, s2):
            if s not in cosmos_set:
                failures.append(f"T^k-image {s} not in Cosmos")

        seen.update([s0, s1, s2])
        triads.append((s0, s1, s2))

    if len(triads) != 24:
        failures.append(f"Triad count = {len(triads)}, expected 24")
    if len(seen) != 72:
        failures.append(f"States covered = {len(seen)}, expected 72 (no gaps)")

    return failures


# ---------------------------------------------------------------------------
# C5 — Observer layer (Theorem NT): rational sum = 0; discrete sum = 3I ≠ 0
# ---------------------------------------------------------------------------
def check_c5():
    failures = []

    # Exact rational values: cos(0°)=1, cos(120°)=−1/2, cos(240°)=−1/2
    cos_0   = Fraction(1)
    cos_120 = Fraction(-1, 2)
    cos_240 = Fraction(-1, 2)

    observer_sum = cos_0 + cos_120 + cos_240
    if observer_sum != Fraction(0):
        failures.append(
            f"Observer sum cos(0°)+cos(120°)+cos(240°) = {observer_sum}, expected 0"
        )

    # Confirm the discrete matrix sum is 3I, NOT 0: the balance is observer-only.
    m = 9
    mat_sum = [[0, 0], [0, 0]]
    for k in [0, 8, 16]:
        mat_sum = mat_add_mod(mat_sum, mat_mod(M_power(k), m), m)

    zero_mat = [[0, 0], [0, 0]]
    if mat_sum == zero_mat:
        failures.append(
            "M^0+M^8+M^16 ≡ 0 mod 9 (unexpected — discrete sum must be 3I, not 0; "
            "if zero, the Theorem NT boundary would be violated)"
        )

    three_I = [[3, 0], [0, 3]]
    if mat_sum != three_I:
        failures.append(
            f"Discrete matrix sum mod 9 = {mat_sum}, expected [[3,0],[0,3]] "
            "(required to confirm Theorem NT: balance is observer-only)"
        )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_three_phase_matrix_sum_3I",   check_c1, {}),
        ("C2_M8_even_grade_rotor",         check_c2, {}),
        ("C3_M8_order_3_mod9",             check_c3, {}),
        ("C4_cosmos_triads_closed",        check_c4, {}),
        ("C5_observer_balance_theorem_nt", check_c5, {}),
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
        print("CERT [303] PASS — QA Three-Phase Cosmos Cancellation")
    else:
        print("CERT [303] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
