#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — integer matrix arithmetic on SL(2,Z) and Fibonacci sequences; no empirical QA state machine; all state is exact integer"
"""
QA Pisano mod-8 CRT Cert [302] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Wall (1960) Fibonacci Primitive Roots, Amer. Math. Monthly 67(6):525-532

The Cosmos period 24 arises from the CRT: Z/24Z = Z/8Z x Z/3Z.
  pi(3) = 8   (3-adic layer, established in [301])
  pi(8) = 12  (2-adic layer, this cert)
  pi(24) = lcm(12, 8) = 24

All computations use M^k = [[F(k-1), F(k)], [F(k), F(k+1)]] (Fibonacci matrix, from [299]).

Five claims:
  C1  pi(8) = 12: F mod 8 has period exactly 12; M^12 equiv I mod 8; proper divisors {1,2,3,4,6} do not
  C2  M^6 equiv 5*I mod 8: the 2-adic half-period scalar; 5^2 equiv 1 mod 8
  C3  CRT: pi(24) = lcm(pi(8), pi(3)) = lcm(12, 8) = 24
  C4  M^12 mod 24 = 17*I: F(11)=89 equiv 17, F(12)=144 equiv 0 mod 24; 17^2 equiv 1 mod 24
      Three grade scalars: mod 9 gives 8 equiv -1 (cert [298]), mod 8 gives 5 (C2),
      mod 24 gives 17 (C4); CRT: 17 equiv -1 mod 9 and 17 equiv 1 mod 8
  C5  Exact order 24 in GL(2,Z/24Z): M^24 equiv I; M^k not I for k in {1,2,3,4,6,8,12}
"""

import sys


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


def is_identity_mod(k, m):
    Mk = mat_mod(M_power(k), m)
    return Mk == [[1, 0], [0, 1]]


def is_scalar_mod(k, m):
    """Return scalar s if M^k = s*I mod m, else None."""
    Mk = mat_mod(M_power(k), m)
    if Mk[0][1] != 0 or Mk[1][0] != 0:
        return None
    if Mk[0][0] != Mk[1][1]:
        return None
    return Mk[0][0]


# ---------------------------------------------------------------------------
# C1 — pi(8) = 12
# ---------------------------------------------------------------------------
def check_c1():
    failures = []
    m = 8

    # Verify period is 12: F(12)=144 equiv 0 mod 8 AND F(13)=233 equiv 1 mod 8
    if fibonacci(12) % m != 0:
        failures.append(f"F(12) mod 8 = {fibonacci(12) % m}, expected 0")
    if fibonacci(13) % m != 1:
        failures.append(f"F(13) mod 8 = {fibonacci(13) % m}, expected 1")

    # M^12 = I mod 8
    if not is_identity_mod(12, m):
        failures.append(f"M^12 mod 8 = {mat_mod(M_power(12), m)}, expected identity")

    # No proper divisor of 12 gives I mod 8
    for d in [1, 2, 3, 4, 6]:
        if is_identity_mod(d, m):
            failures.append(f"M^{d} = I mod 8 — period divides {d}, not 12")

    return failures


# ---------------------------------------------------------------------------
# C2 — M^6 = 5*I mod 8
# ---------------------------------------------------------------------------
def check_c2():
    failures = []
    m = 8

    M6_mod8 = mat_mod(M_power(6), m)
    expected = [[5, 0], [0, 5]]
    if M6_mod8 != expected:
        failures.append(f"M^6 mod 8 = {M6_mod8}, expected [[5,0],[0,5]]")

    # 5 has order 2 in (Z/8Z)^x: 5^2 = 25 equiv 1 mod 8
    if (5 * 5) % 8 != 1:
        failures.append(f"5^2 mod 8 = {(5*5)%8}, expected 1")

    # M^6 not equal to I mod 8 (5 != 1)
    if is_identity_mod(6, m):
        failures.append("M^6 = I mod 8 (unexpected — half-period should be non-trivial)")

    return failures


# ---------------------------------------------------------------------------
# C3 — CRT: pi(24) = lcm(pi(8), pi(3)) = 24
# ---------------------------------------------------------------------------
def check_c3():
    failures = []
    from math import lcm

    pi3  = 8    # pi(3)=8, established in [291],[301]
    pi8  = 12   # pi(8)=12, established in C1
    pi24_expected = lcm(pi3, pi8)   # = 24

    if pi24_expected != 24:
        failures.append(f"lcm({pi3},{pi8}) = {pi24_expected}, expected 24")

    # Verify M^24 = I mod 24
    if not is_identity_mod(24, 24):
        failures.append(f"M^24 mod 24 = {mat_mod(M_power(24), 24)}, expected identity")

    # Confirm via CRT: M^24 = I mod 3 and I mod 8 => I mod 24
    if not is_identity_mod(24, 3):
        failures.append(f"M^24 mod 3 != I")
    if not is_identity_mod(24, 8):
        failures.append(f"M^24 mod 8 != I")

    # The period at each component: pi(8)=12 divides 24, pi(3)=8 divides 24 ✓
    if 24 % pi3 != 0:
        failures.append(f"pi(3)={pi3} does not divide 24")
    if 24 % pi8 != 0:
        failures.append(f"pi(8)={pi8} does not divide 24")

    return failures


# ---------------------------------------------------------------------------
# C4 — M^12 mod 24 = 17*I
# ---------------------------------------------------------------------------
def check_c4():
    failures = []
    m = 24

    # F(11) = 89, F(12) = 144, F(13) = 233
    if fibonacci(11) != 89:
        failures.append(f"F(11) = {fibonacci(11)}, expected 89")
    if fibonacci(12) != 144:
        failures.append(f"F(12) = {fibonacci(12)}, expected 144")
    if fibonacci(13) != 233:
        failures.append(f"F(13) = {fibonacci(13)}, expected 233")

    # mod 24
    if fibonacci(11) % m != 17:
        failures.append(f"F(11) mod 24 = {fibonacci(11)%m}, expected 17")
    if fibonacci(12) % m != 0:
        failures.append(f"F(12) mod 24 = {fibonacci(12)%m}, expected 0")
    if fibonacci(13) % m != 17:
        failures.append(f"F(13) mod 24 = {fibonacci(13)%m}, expected 17")

    M12_mod24 = mat_mod(M_power(12), m)
    expected = [[17, 0], [0, 17]]
    if M12_mod24 != expected:
        failures.append(f"M^12 mod 24 = {M12_mod24}, expected {expected}")

    # 17^2 = 289 equiv 1 mod 24
    if (17 * 17) % m != 1:
        failures.append(f"17^2 mod 24 = {(17*17)%m}, expected 1")

    # CRT consistency: 17 equiv -1 mod 9 and 17 equiv 1 mod 8
    if 17 % 9 != 8:   # 8 = -1 mod 9
        failures.append(f"17 mod 9 = {17%9}, expected 8 (= -1 mod 9)")
    if 17 % 8 != 1:
        failures.append(f"17 mod 8 = {17%8}, expected 1")

    # The three grade scalars form a coherent CRT triple:
    # mod 9: 8 (=-1), mod 8: 5 (half-period at step 6), mod 24: 17 (at step 12)
    # Confirm: scalar_mod9(M^12) = 8
    s9 = is_scalar_mod(12, 9)
    if s9 != 8:
        failures.append(f"M^12 mod 9 scalar = {s9}, expected 8 (confirming cert [298])")

    return failures


# ---------------------------------------------------------------------------
# C5 — Exact order 24 in GL(2,Z/24Z)
# ---------------------------------------------------------------------------
def check_c5():
    failures = []
    m = 24

    # M^24 = I mod 24
    if not is_identity_mod(24, m):
        failures.append(f"M^24 mod 24 != I")

    # None of the proper divisors of 24 give I mod 24
    for d in [1, 2, 3, 4, 6, 8, 12]:
        if is_identity_mod(d, m):
            failures.append(f"M^{d} = I mod 24 — order divides {d}, should be exactly 24")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_pi8_equals_12",         check_c1, {}),
        ("C2_M6_equals_5I_mod8",     check_c2, {}),
        ("C3_crt_pi24_equals_24",    check_c3, {}),
        ("C4_M12_equals_17I_mod24",  check_c4, {}),
        ("C5_exact_order_24_mod24",  check_c5, {}),
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
        print("CERT [302] PASS — QA Pisano mod-8 CRT")
    else:
        print("CERT [302] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
