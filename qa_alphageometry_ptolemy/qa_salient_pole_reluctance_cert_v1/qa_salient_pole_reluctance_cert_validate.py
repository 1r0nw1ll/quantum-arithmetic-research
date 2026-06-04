#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — Wildberger direction spread (Fraction); double-spread defect (Fraction); A1 no-zero arithmetic; no float QA state"
"""
QA Salient-Pole Reluctance Torque Cert [313] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford,
    ISBN 978-0-19-921986-5
  Wildberger, N.J. (2005) Divine Proportions: Rational Trigonometry, Wild Egg Books,
    ISBN 978-0-9757492-0-8
  Chapman, S.J. (2011) Electric Machinery Fundamentals, 5th ed., McGraw-Hill,
    ISBN 978-0-07-352954-7

QA mapping: salient-pole synchronous machine reluctance torque maps to the
double-spread defect of the Fibonacci orbit.

  Saliency: d-axis at k=8 (cert [308] Scott-T d-position), q-axis at k=6.
  Ideal rotation double-spread: 4*sk*(1-sk) (Wildberger rational-trig identity).
  Reluctance defect: Δk = s_{2k mod 24} - 4*sk*(1-sk).
  Positive Δk: orbit "accelerates" past ideal rotation.
  Negative Δk: orbit "brakes" relative to ideal rotation.

Five claims:
  C1  d-q saliency ratio: s8/s6 = (9/25)/(1/50) = 18 (exact Fraction, exact integer);
      d-axis spread 18x larger than q-axis spread; positions from cert [308] Scott-T word.
  C2  Rational double-spread (Wildberger): 4*sk*(1-sk) in (0,1)cap Q for k neq 0,12;
      Δ0=Δ12=0 (synchronous and antipodal are double-spread fixed points by construction);
      this is the Fraction-valued analog of sin^2(2δ) = 4*sin^2(δ)*cos^2(δ).
  C3  Minimum reluctance defect: argmin_k Δk = 9 with Δmin = -479993/515450 (exact
      Fraction); T^9(1,1)=(1,8) is the maximum-braking orbit position; Δ9 is maximally
      negative because s18=1/122 (very small: near-Singularity approach just completed)
      while 4*s9*(1-s9) approx 0.931 (near-maximum ideal spread).
  C4  Defect sign partition: Δk > 0 for exactly k in {4,10,11,13,14} (5 near-Singularity
      acceleration zones); Δk < 0 for 17 positions (braking zones); Δk=0 for {0,12};
      argmax Δk = 11 with Δmax = 324551/862025 (near-Singularity escape: s11=1/290 small,
      s22=16/41 large, making s_{2*11} > 4*s11*(1-s11)).
  C5  Theorem NT: reluctance torque T_r proportional to Δk_observer is observer layer;
      physical saliency X_d - X_q and sin(2δ) are observer projections; QA provides the
      orbit defect Δk (exact Fraction) as the discrete structural pre-image.
"""

import json
import sys
from fractions import Fraction


def a1_mod(x, m=9):
    return ((x - 1) % m) + 1


def t_step(b, e, m=9):
    return e, a1_mod(b + e, m)


def orbit(b0, e0, m=9):
    states, b, e = [], b0, e0
    for _ in range(24):
        states.append((b, e))
        b, e = t_step(b, e, m)
    return states


def direction_spread(b0, e0, b1, e1):
    """Wildberger spread between direction vectors. Exact Fraction."""
    num = (b0 * e1 - e0 * b1) * (b0 * e1 - e0 * b1)
    den = (b0 * b0 + e0 * e0) * (b1 * b1 + e1 * e1)
    return Fraction(0) if den == 0 else Fraction(num, den)


checks = {}
passed = 0
failed = 0

B0, E0 = 1, 1
orb     = orbit(B0, E0)
spreads = [direction_spread(B0, E0, b, e) for (b, e) in orb]
defects = [spreads[(2 * k) % 24] - 4 * spreads[k] * (1 - spreads[k])
           for k in range(24)]

# C1 — d-q saliency ratio s8/s6 = 18
bk6, ek6 = orb[6]   # (4,3)
bk8, ek8 = orb[8]   # (7,1)
s6 = spreads[6]
s8 = spreads[8]
saliency_ratio = s8 / s6
ok_c1 = (bk6 == 4 and ek6 == 3
          and bk8 == 7 and ek8 == 1
          and s6 == Fraction(1, 50)
          and s8 == Fraction(9, 25)
          and saliency_ratio == Fraction(18))
checks["C1_dq_saliency_ratio_18"] = "PASS" if ok_c1 else "FAIL"
if ok_c1: passed += 1
else:     failed += 1

# C2 — Rational double-spread; Δ0=Δ12=0
ok_c2 = (defects[0]  == Fraction(0)
          and defects[12] == Fraction(0))
ok_c2 = ok_c2 and all(
    Fraction(0) < 4 * spreads[k] * (1 - spreads[k]) < Fraction(1)
    for k in range(24) if k not in (0, 12)
)
ok_c2 = ok_c2 and all(isinstance(d, Fraction) for d in defects)
checks["C2_rational_double_spread_delta0_delta12_zero"] = "PASS" if ok_c2 else "FAIL"
if ok_c2: passed += 1
else:     failed += 1

# C3 — Minimum defect at k=9, Δmin=-479993/515450
k_min  = min(range(24), key=lambda k: defects[k])
d_min  = defects[k_min]
bk9, ek9 = orb[9]
ok_c3 = (k_min == 9
          and bk9 == 1 and ek9 == 8
          and d_min == Fraction(-479993, 515450))
checks["C3_min_defect_k9_minus479993_515450"] = "PASS" if ok_c3 else "FAIL"
if ok_c3: passed += 1
else:     failed += 1

# C4 — Sign partition: exactly {4,10,11,13,14} positive, {0,12} zero, rest negative
pos_set  = frozenset(k for k in range(24) if defects[k] > 0)
zero_set = frozenset(k for k in range(24) if defects[k] == 0)
k_max    = max(range(24), key=lambda k: defects[k])
d_max    = defects[k_max]
ok_c4 = (pos_set  == frozenset({4, 10, 11, 13, 14})
          and zero_set == frozenset({0, 12})
          and k_max == 11
          and d_max == Fraction(324551, 862025))
checks["C4_sign_partition_pos4_10_11_13_14"] = "PASS" if ok_c4 else "FAIL"
if ok_c4: passed += 1
else:     failed += 1

# C5 — Theorem NT: defects are Fraction, never float; observer layer is separate
torque_proxy_min = d_min / d_min if d_min != 0 else Fraction(0)  # = 1 (normalised)
torque_proxy_max = d_max / abs(d_min)  # rational rescaling
ok_c5 = (isinstance(d_min, Fraction)
          and isinstance(d_max, Fraction)
          and isinstance(torque_proxy_max, Fraction)
          and torque_proxy_max > 0)
checks["C5_theorem_nt_defect_fraction_observer"] = "PASS" if ok_c5 else "FAIL"
if ok_c5: passed += 1
else:     failed += 1

result = {
    "ok":               failed == 0,
    "passed":           passed,
    "failed":           failed,
    "checks":           checks,
    "saliency_ratio":   str(saliency_ratio),
    "s6":               str(s6),
    "s8":               str(s8),
    "delta_min_k":      k_min,
    "delta_min":        str(d_min),
    "delta_max_k":      k_max,
    "delta_max":        str(d_max),
    "positive_k_set":   sorted(pos_set),
    "zero_k_set":       sorted(zero_set),
}
print(json.dumps(result))
if failed:
    sys.exit(1)
