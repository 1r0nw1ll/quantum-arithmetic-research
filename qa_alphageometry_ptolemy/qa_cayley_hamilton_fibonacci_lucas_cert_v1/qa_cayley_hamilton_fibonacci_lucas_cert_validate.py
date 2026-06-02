#!/usr/bin/env python3
"""
QA Cayley-Hamilton Fibonacci-Lucas Cert [299] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Lucas (1878) Theorie des fonctions numeriques simplement periodiques, Amer. J. Math. 1(2):184-196
  Wall (1960) Fibonacci Primitive Roots, Amer. Math. Monthly 67(6):525-532

M = [[0,1],[1,1]] is the QA T-operator.
Characteristic polynomial: x^2 - x - 1 = 0  =>  M^2 = M + I  (Cayley-Hamilton).
This gives M^k entries in terms of Fibonacci numbers and Tr(M^k) = Lucas numbers.

Five claims:
  C1  M^2 = M + I exactly (Cayley-Hamilton, characteristic polynomial x^2-x-1)
  C2  M^k = [[F(k-1),F(k)],[F(k),F(k+1)]] for k=0..40 (Fibonacci matrix representation)
  C3  Tr(M^k) = L(k) (k-th Lucas number) for k=1..40
  C4  det(M^k) = (-1)^k for all k>=0
  C5  Corollary of [298]: Tr(M^12)=L(12)=322; 322 mod 9 = 7 = -2 mod 9 = Tr(-I mod 9)
      and Tr(M^24) = L(24) mod 9 = 2 = Tr(I mod 9)
"""

import sys


M = [[0, 1], [1, 1]]
I2 = [[1, 0], [0, 1]]   # 2x2 identity


def mat_mul(A, B):
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0],  A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0],  A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]


def mat_add(A, B):
    return [[A[r][c] + B[r][c] for c in range(2)] for r in range(2)]


def mat_scalar(A, s):
    return [[A[r][c] * s for c in range(2)] for r in range(2)]


def mat_pow(A, n):
    result = [[1, 0], [0, 1]]
    base = [row[:] for row in A]
    while n:
        if n & 1:
            result = mat_mul(result, base)
        base = mat_mul(base, base)
        n >>= 1
    return result


def det(A):
    return A[0][0]*A[1][1] - A[0][1]*A[1][0]


def trace(A):
    return A[0][0] + A[1][1]


def fibonacci(n):
    """Return F(n) for n>=0: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)."""
    if n == 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def lucas(n):
    """Return L(n) for n>=0: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2)."""
    if n == 0: return 2
    if n == 1: return 1
    a, b = 2, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


# ---------------------------------------------------------------------------
# C1 — Cayley-Hamilton: M^2 = M + I
# ---------------------------------------------------------------------------
def check_c1():
    M2 = mat_mul(M, M)
    M_plus_I = mat_add(M, I2)
    if M2 != M_plus_I:
        return [f"M^2 = {M2}, M+I = {M_plus_I}"]
    # Also verify characteristic polynomial: det(M - lambda*I) = lambda^2 - lambda - 1
    # At lambda=0: det(M) = -1 = 0^2 - 0 - 1 = -1 ✓
    if det(M) != -1:
        return [f"det(M) = {det(M)}, expected -1"]
    # At lambda=1: det(M-I) = det([[-1,1],[1,0]]) = 0-1 = -1 = 1-1-1 = -1 ✓
    M_minus_I = mat_add(M, mat_scalar(I2, -1))
    if det(M_minus_I) != -1:
        return [f"det(M-I) = {det(M_minus_I)}, expected -1 (= 1-1-1)"]
    return []


# ---------------------------------------------------------------------------
# C2 — Fibonacci matrix entries: M^k = [[F(k-1),F(k)],[F(k),F(k+1)]]
# ---------------------------------------------------------------------------
def check_c2(k_max=40):
    failures = []
    for k in range(k_max + 1):
        Mk = mat_pow(M, k)
        expected = [
            [fibonacci(k - 1) if k >= 1 else 1, fibonacci(k)],
            [fibonacci(k),                        fibonacci(k + 1)],
        ]
        # F(-1) = 1 by convention (F(-1)+F(0)=F(1): 1+0=1 ✓)
        # For k=0: M^0 = I = [[F(-1),F(0)],[F(0),F(1)]] = [[1,0],[0,1]] ✓
        if Mk != expected:
            failures.append(f"k={k}: M^k={Mk}, expected {expected}")
            if len(failures) >= 3:
                break
    return failures


# ---------------------------------------------------------------------------
# C3 — Lucas trace formula: Tr(M^k) = L(k)
# ---------------------------------------------------------------------------
def check_c3(k_max=40):
    """
    Tr(M^k) = F(k-1) + F(k+1).
    Identity: F(k-1) + F(k+1) = L(k) (Lucas numbers).
    Proof: F(k-1)+F(k+1) = F(k-1)+F(k)+F(k-1) ... actually
      F(k+1) = F(k) + F(k-1), so F(k-1)+F(k+1) = F(k-1)+F(k)+F(k-1) = F(k)+2F(k-1).
    Alternatively, the identity F(k-1)+F(k+1) = L(k) is standard.
    """
    failures = []
    for k in range(1, k_max + 1):
        Mk = mat_pow(M, k)
        tr = trace(Mk)
        lk = lucas(k)
        fk_m1 = fibonacci(k - 1) if k >= 1 else 1
        fk_p1 = fibonacci(k + 1)
        if tr != lk:
            failures.append(f"k={k}: Tr(M^k)={tr}, L(k)={lk}")
        if fk_m1 + fk_p1 != lk:
            failures.append(f"k={k}: F(k-1)+F(k+1)={fk_m1+fk_p1} != L(k)={lk}")
        if len(failures) >= 3:
            break
    return failures


# ---------------------------------------------------------------------------
# C4 — Alternating determinant: det(M^k) = (-1)^k
# ---------------------------------------------------------------------------
def check_c4(k_max=40):
    failures = []
    for k in range(k_max + 1):
        Mk = mat_pow(M, k)
        d = det(Mk)
        expected = (-1)**k
        if d != expected:
            failures.append(f"k={k}: det(M^k)={d}, expected {expected}")
    return failures


# ---------------------------------------------------------------------------
# C5 — [298] corollary: Tr(M^12) and Tr(M^24) mod 9
# ---------------------------------------------------------------------------
def check_c5():
    failures = []

    # M^12 corollary: L(12) = 322; 322 mod 9 = 7 = -2 mod 9 = Tr(-I mod 9)
    L12 = lucas(12)
    if L12 != 322:
        failures.append(f"L(12) = {L12}, expected 322")
    if L12 % 9 != 7:
        failures.append(f"L(12) mod 9 = {L12 % 9}, expected 7")
    # -2 mod 9 = 7 = Tr(-I mod 9) = (-1)+(-1) mod 9 = (-2) mod 9 = 7
    if (-2) % 9 != 7:
        failures.append(f"(-2) mod 9 = {(-2)%9}, expected 7")
    # Verify Tr(M^12 mod 9) = 7 directly
    M12 = mat_pow(M, 12)
    if trace(M12) % 9 != 7:
        failures.append(f"Tr(M^12) mod 9 = {trace(M12) % 9}, expected 7")

    # M^24 corollary: L(24) mod 9 = 2 = Tr(I mod 9)
    L24 = lucas(24)
    if L24 % 9 != 2:
        failures.append(f"L(24) mod 9 = {L24 % 9}, expected 2 (= Tr(I mod 9))")
    M24 = mat_pow(M, 24)
    if trace(M24) % 9 != 2:
        failures.append(f"Tr(M^24) mod 9 = {trace(M24) % 9}, expected 2")
    # M^24 entries mod 9 should equal I mod 9
    M24_mod9 = [[x % 9 for x in row] for row in M24]
    if M24_mod9 != [[1, 0], [0, 1]]:
        failures.append(f"M^24 mod 9 = {M24_mod9}, expected identity")

    # Sanity: Tr(M^k) = L(k) at k=12,24 directly
    if trace(mat_pow(M, 12)) != L12:
        failures.append(f"Tr(M^12) != L(12) direct check")
    if trace(mat_pow(M, 24)) != L24:
        failures.append(f"Tr(M^24) != L(24) direct check")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_cayley_hamilton",        check_c1, {}),
        ("C2_fibonacci_entries",      check_c2, {"k_max": 40}),
        ("C3_lucas_trace",            check_c3, {"k_max": 40}),
        ("C4_alternating_det",        check_c4, {"k_max": 40}),
        ("C5_mod9_corollary",         check_c5, {}),
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
        print("CERT [299] PASS — QA Cayley-Hamilton Fibonacci-Lucas")
    else:
        print("CERT [299] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
