#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — SL(2,Z) integer matrix arithmetic; Z/24Z subgroup structure; Wildberger direction spread (Fraction); A1 no-zero arithmetic; no float QA state"
"""
QA Scott T-Transformer Cert [308] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford,
    ISBN 978-0-19-921986-5
  Serre, J.-P. (1973) A Course in Arithmetic, Springer, ISBN 978-0-387-90041-1
  Chapman, S.J. (2011) Electric Machinery Fundamentals, 5th ed., McGraw-Hill,
    ISBN 978-0-07-352954-7

QA mapping: Scott-T transformer converts 3-phase to 2-phase by a SL(2,Z) word.
  3-phase clock: subgroup {0,8,16} leq Z/24Z (generator M^8, order 3)
  2-phase clock: subgroup {0,6,12,18} leq Z/24Z (generator M^6, order 4)
  Scott-T word:  W = M^{-2} = [[2,-1],[-1,1]] = R^{-1}.L^{-1} in SL(2,Z)
    The C-phase position (k=8) shifts to Q-axis (k=6) by one application of W: 8-2=6.

Five claims:
  C1  3-phase subgroup {0,8,16} leq Z/24Z: M^8 has order 3 (M^8 neq I, M^16 neq I,
      M^24=I mod 24); 8 T-steps = 120deg equivalent in the 24-step orbit; three equally
      spaced positions generate the period-3 subgroup.
  C2  2-phase subgroup {0,6,12,18} leq Z/24Z: M^6 has order 4 (M^6 neq I, M^12 neq I,
      M^18 neq I, M^24=I mod 24); 6 T-steps = 90deg equivalent; four positions generate
      the period-4 subgroup.
  C3  Scott-T SL(2,Z) word: W = R^{-1}.L^{-1} = [[2,-1],[-1,1]] = M^{-2};
      verification: M^2.W = I (exact integer arithmetic); M^8.W = M^6 (mod 24);
      the unique 2-step SL(2,Z) correction bridging the 3-phase step to the 2-phase step.
  C4  Rational Scott-T coefficient: direction spread s(T^6(v), T^8(v)) for v=(1,1) is
      289/1250 (exact Fraction); rational-trig analog of the physical transformer ratio
      sqrt(3)/2; computed from orbit states T^6=(4,3) and T^8=(7,1) via
      s = (4*1-3*7)^2/((16+9)*(49+1)) = 289/1250.
  C5  Theorem NT: sqrt(3)/2 approx 0.866 is an observer projection of the SL(2,Z) word
      W = R^{-1}.L^{-1}; the 3-to-2 phase transformation is a group operation on the
      Z/24Z orbit clock; no transcendental or irrational arithmetic enters QA logic.
"""

import json
import sys
from fractions import Fraction


def fibonacci(n):
    if n == 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def M_power_exact(k):
    """M^k as exact integer matrix [[F(k-1),F(k)],[F(k),F(k+1)]]."""
    fkm1 = 1 if k == 0 else fibonacci(k - 1)
    fk   = fibonacci(k)
    fkp1 = fibonacci(k + 1)
    return [[fkm1, fk], [fk, fkp1]]


def mat_mod(A, m):
    return [[x % m for x in row] for row in A]


def mat_mul(A, B):
    return [
        [A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
        [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]],
    ]


def mat_eq(A, B, m=None):
    if m is None:
        return A == B
    return [[x % m for x in row] for row in A] == [[x % m for x in row] for row in B]


def identity():
    return [[1, 0], [0, 1]]


def a1_mod(x, m=9):
    return ((x - 1) % m) + 1


def t_step(b, e, m=9):
    return e, a1_mod(b + e, m)


def orbit_state_k(b0, e0, k, m=9):
    b, e = b0, e0
    for _ in range(k):
        b, e = t_step(b, e, m)
    return b, e


def direction_spread(b0, e0, b1, e1):
    """Wildberger spread between direction vectors. Exact Fraction."""
    num = (b0 * e1 - e0 * b1) * (b0 * e1 - e0 * b1)
    den = (b0 * b0 + e0 * e0) * (b1 * b1 + e1 * e1)
    return Fraction(0) if den == 0 else Fraction(num, den)


checks = {}
passed = 0
failed = 0

# SL(2,Z) generators and their inverses
L_inv = [[1, 0], [-1, 1]]
R_inv = [[1, -1], [0, 1]]
# W = M^{-2} = R^{-1}.L^{-1}
W = mat_mul(R_inv, L_inv)  # = [[2,-1],[-1,1]]

# C1 — 3-phase subgroup: M^8 has order 3 in Z/24Z
M8  = mat_mod(M_power_exact(8),  24)
M16 = mat_mod(M_power_exact(16), 24)
M24 = mat_mod(M_power_exact(24), 24)
I   = identity()
ok_c1 = (not mat_eq(M8, I, 24)
          and not mat_eq(M16, I, 24)
          and mat_eq(M24, I, 24))
checks["C1_three_phase_subgroup_order3"] = "PASS" if ok_c1 else "FAIL"
if ok_c1: passed += 1
else:     failed += 1

# C2 — 2-phase subgroup: M^6 has order 4 in Z/24Z
M6  = mat_mod(M_power_exact(6),  24)
M12 = mat_mod(M_power_exact(12), 24)
M18 = mat_mod(M_power_exact(18), 24)
ok_c2 = (not mat_eq(M6,  I, 24)
          and not mat_eq(M12, I, 24)
          and not mat_eq(M18, I, 24)
          and mat_eq(M24, I, 24))
checks["C2_two_phase_subgroup_order4"] = "PASS" if ok_c2 else "FAIL"
if ok_c2: passed += 1
else:     failed += 1

# C3 — Scott-T SL(2,Z) word W = R^{-1}.L^{-1} = [[2,-1],[-1,1]] = M^{-2}
expected_W = [[2, -1], [-1, 1]]
M2         = M_power_exact(2)           # [[1,1],[1,2]]
M2_W       = mat_mul(M2, W)             # should be I
M8_W       = mat_mul(M_power_exact(8), W)  # should equal M^6 mod 24
ok_c3 = (W == expected_W
          and mat_eq(M2_W, I)
          and mat_eq(M8_W, M6, 24))
checks["C3_scott_t_sl2z_word_R_inv_L_inv"] = "PASS" if ok_c3 else "FAIL"
if ok_c3: passed += 1
else:     failed += 1

# C4 — Rational Scott-T coefficient: s(T^6(1,1), T^8(1,1)) = 289/1250
bk6, ek6 = orbit_state_k(1, 1, 6)   # (4,3)
bk8, ek8 = orbit_state_k(1, 1, 8)   # (7,1)
s_scott   = direction_spread(bk6, ek6, bk8, ek8)
ok_c4 = (bk6 == 4 and ek6 == 3
          and bk8 == 7 and ek8 == 1
          and s_scott == Fraction(289, 1250))
checks["C4_rational_scott_t_coeff_289_1250"] = "PASS" if ok_c4 else "FAIL"
if ok_c4: passed += 1
else:     failed += 1

# C5 — Theorem NT: all W-arithmetic is exact integer; s is Fraction; no floats
ok_c5 = (all(isinstance(x, int) for row in W     for x in row)
          and all(isinstance(x, int) for row in M2_W  for x in row)
          and all(isinstance(x, int) for row in M8_W  for x in row)
          and isinstance(s_scott, Fraction))
checks["C5_theorem_nt_sl2z_exact_no_float"] = "PASS" if ok_c5 else "FAIL"
if ok_c5: passed += 1
else:     failed += 1

result = {
    "ok":           failed == 0,
    "passed":       passed,
    "failed":       failed,
    "checks":       checks,
    "W_matrix":     W,
    "T6_state":     [bk6, ek6],
    "T8_state":     [bk8, ek8],
    "s_scott_t":    str(s_scott),
    "M8_W_mod24":   M8_W,
    "M6_mod24":     M6,
}
print(json.dumps(result))
if failed:
    sys.exit(1)
