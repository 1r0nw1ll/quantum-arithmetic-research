#!/usr/bin/env python3
"""
QA Pell Rotor Eigenspread Cert [297] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Wildberger (2005) Divine Proportions, Wild Egg Books, ISBN 978-0-9757492-0-8

Let A = [[1,2],[1,1]] (det=-1, Pell step matrix).
The Pell chain {(b_n,e_n)} under A converges toward the eigenstate at ratio sqrt(2).
The spread s_n = e_n^2/(b_n^2+e_n^2) converges to 1/3 from alternating sides.

Five claims:
  C1  A=[[1,2],[1,1]], det(A)=-1; maps (b,e)->(b+2e,b+e) exactly
  C2  Pell chain: b_n^2 - 2e_n^2 = (-1)^(n+1) starting from (b_0,e_0)=(1,1); |I|=1 always
  C3  Spread formula: |s_n - 1/3| = 1/(3*G_tilde_n), consistent with [292] I=3G|s-1/3|=1
  C4  No positive integer solution to b^2=2e^2 (sqrt(2) irrational); verified b,e<=10^4
  C5  A^2 = [[3,4],[2,3]] (the RLLR matrix from cert [296]); Mobius fixed point at sqrt(2)
"""

import sys
from fractions import Fraction

# ---------------------------------------------------------------------------
# Matrices (2x2 integer, as nested lists)
# ---------------------------------------------------------------------------
A  = [[1, 2], [1, 1]]   # Pell step: (b,e) -> (b+2e, b+e), det=-1
A2 = [[3, 4], [2, 3]]   # A^2 = RLLR matrix from [296], det=+1


def mat_mul(X, Y):
    return [
        [X[0][0]*Y[0][0] + X[0][1]*Y[1][0], X[0][0]*Y[0][1] + X[0][1]*Y[1][1]],
        [X[1][0]*Y[0][0] + X[1][1]*Y[1][0], X[1][0]*Y[0][1] + X[1][1]*Y[1][1]],
    ]


def mat_vec(X, v):
    return [X[0][0]*v[0] + X[0][1]*v[1],
            X[1][0]*v[0] + X[1][1]*v[1]]


def det(X):
    return X[0][0]*X[1][1] - X[0][1]*X[1][0]


def pell_step(b, e):
    """One application of the Pell map A: (b,e) -> (b+2e, b+e)."""
    return b + 2*e, b + e


# ---------------------------------------------------------------------------
# C1 — Matrix A, det=-1, action correct
# ---------------------------------------------------------------------------
def check_c1():
    failures = []
    if A != [[1, 2], [1, 1]]:
        failures.append(f"A = {A}, expected [[1,2],[1,1]]")
    if det(A) != -1:
        failures.append(f"det(A) = {det(A)}, expected -1")
    for b in range(1, 20):
        for e in range(1, 20):
            result = mat_vec(A, [b, e])
            expected = [b + 2*e, b + e]
            if result != expected:
                failures.append(f"A*[{b},{e}]^T = {result}, expected {expected}")
    return failures


# ---------------------------------------------------------------------------
# C2 — Pell chain: b_n^2 - 2e_n^2 = (-1)^n
# ---------------------------------------------------------------------------
def pell_chain(n_steps=20):
    chain = []
    b, e = 1, 1
    for n in range(n_steps + 1):
        chain.append((b, e, n))
        b, e = pell_step(b, e)
    return chain


def check_c2(n_steps=30):
    """
    (1,1): I=1-2=-1=(-1)^1. So b_n^2-2e_n^2 = (-1)^(n+1).
    Equivalently |I|=1 for all n, with sign alternating starting negative.
    """
    failures = []
    b, e = 1, 1
    for n in range(n_steps + 1):
        I_n = b*b - 2*e*e
        expected_I = (-1)**(n + 1)
        if I_n != expected_I:
            failures.append(f"n={n}: b={b},e={e}, b^2-2e^2={I_n}, expected {expected_I}")
        if abs(I_n) != 1:
            failures.append(f"n={n}: |I|={abs(I_n)}, expected 1")
        b, e = pell_step(b, e)
    return failures


# ---------------------------------------------------------------------------
# C3 — Spread formula: |s_n - 1/3| = 1/(3*G_tilde_n)
# ---------------------------------------------------------------------------
def check_c3(n_steps=30):
    """
    s_n = e_n^2 / (b_n^2 + e_n^2) = e_n^2 / G_tilde_n
    |s_n - 1/3| = |3e_n^2 - G_tilde_n| / (3 G_tilde_n)
                = |3e_n^2 - b_n^2 - e_n^2| / (3 G_tilde_n)
                = |2e_n^2 - b_n^2| / (3 G_tilde_n)
                = |b_n^2 - 2e_n^2| / (3 G_tilde_n)
                = 1 / (3 G_tilde_n)   [since |b_n^2-2e_n^2|=1 for all n]

    Uses exact Fraction arithmetic to avoid float errors.
    """
    failures = []
    b, e = 1, 1
    prev_above = None   # track alternating sides
    for n in range(n_steps + 1):
        G = b*b + e*e
        s = Fraction(e*e, G)
        diff = s - Fraction(1, 3)
        expected_absdiff = Fraction(1, 3 * G)

        if abs(diff) != expected_absdiff:
            failures.append(
                f"n={n}: |s-1/3|={abs(diff)} != 1/(3G̃)={expected_absdiff}"
            )
            if len(failures) >= 3:
                break

        # Verify alternating sides.
        # s = e^2/G, 1/3 = G/(3G) = (b^2+e^2)/(3G).
        # s > 1/3  <=>  3e^2 > b^2+e^2  <=>  2e^2 > b^2  <=>  b^2-2e^2 < 0 (I<0).
        I_n = b*b - 2*e*e
        above_third = diff > 0   # s > 1/3
        if I_n < 0 and not above_third:
            failures.append(f"n={n}: I<0 (b^2<2e^2) but s<=1/3 (expected above)")
        if I_n > 0 and above_third:
            failures.append(f"n={n}: I>0 (b^2>2e^2) but s>=1/3 (expected below)")

        b, e = pell_step(b, e)
    return failures


# ---------------------------------------------------------------------------
# C4 — No positive integer solution to b^2 = 2e^2
# ---------------------------------------------------------------------------
def check_c4(limit=10_000):
    """
    If b^2 = 2e^2, then b = e*sqrt(2), irrational for all integer e>0.
    Exhaustive check: no (b,e) with 1<=b,e<=limit satisfies b^2 = 2*e^2.
    We use the fact that if b^2=2e^2, then (b/gcd,e/gcd) is a primitive
    solution, and primitive solutions must have b,e <= sqrt(2)*limit.
    Search over e from 1 to limit; b = round(e*sqrt(2)).
    """
    failures = []
    import math
    sqrt2_approx = math.isqrt(2 * limit * limit)
    # Search: for each e, check if 2*e^2 is a perfect square
    for e in range(1, limit + 1):
        twice_e2 = 2 * e * e
        b = math.isqrt(twice_e2)
        if b * b == twice_e2:
            failures.append(f"Found b={b}, e={e}: b^2=2e^2 (contradicts irrationality of sqrt(2))")
            break
    # Double-check: also verify the Pell chain never hits b^2=2e^2
    b, e = 1, 1
    for n in range(50):
        if b*b == 2*e*e:
            failures.append(f"Pell chain: n={n}, b={b}, e={e}: b^2=2e^2 (impossible!)")
        b, e = pell_step(b, e)
    return failures


# ---------------------------------------------------------------------------
# C5 — A^2 = [[3,4],[2,3]] and Mobius fixed point at sqrt(2)
# ---------------------------------------------------------------------------
def check_c5():
    failures = []
    # A^2 computation
    A2_computed = mat_mul(A, A)
    if A2_computed != A2:
        failures.append(f"A^2 = {A2_computed}, expected {A2}")
    if det(A2) != 1:
        failures.append(f"det(A^2) = {det(A2)}, expected 1")

    # A^2 Mobius fixed point: (3z+4)/(2z+3) = z  =>  z^2 = 2
    # Verify: no rational z = p/q satisfies (3*(p/q)+4)/(2*(p/q)+3) = p/q
    # i.e. (3p+4q)/(2p+3q) = p/q  =>  q(3p+4q) = p(2p+3q)
    #      3pq + 4q^2 = 2p^2 + 3pq  =>  4q^2 = 2p^2  =>  p^2 = 2q^2
    # So the fixed-point equation reduces to p^2=2q^2 — same as C4.
    # Verify for small fractions:
    for p in range(1, 100):
        for q in range(1, 100):
            lhs = Fraction(3*p + 4*q, 2*p + 3*q)
            rhs = Fraction(p, q)
            if lhs == rhs:
                failures.append(f"Rational fixed point found: p/q={p}/{q}")

    # Verify that A^2 * [b,e]^T maps Pell states two steps forward
    b, e = 1, 1
    for n in range(10):
        b2, e2 = pell_step(*pell_step(b, e))
        result = mat_vec(A2, [b, e])
        if result != [b2, e2]:
            failures.append(f"A^2*[{b},{e}]^T = {result}, expected [{b2},{e2}]")
        b, e = pell_step(b, e)

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_pell_matrix",    check_c1,  {}),
        ("C2_pell_chain",     check_c2,  {"n_steps": 30}),
        ("C3_spread_formula", check_c3,  {"n_steps": 30}),
        ("C4_sqrt2_irrational", check_c4, {"limit": 10_000}),
        ("C5_double_step",    check_c5,  {}),
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
        print("CERT [297] PASS — QA Pell Rotor Eigenspread")
    else:
        print("CERT [297] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
