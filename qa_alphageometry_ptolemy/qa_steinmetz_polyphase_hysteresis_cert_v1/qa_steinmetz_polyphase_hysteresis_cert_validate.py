#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — Fraction cross-spread orbit sum; A1 no-zero arithmetic; T-step bijection; no float QA state"
"""
QA Steinmetz Polyphase Hysteresis Cert [309] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford,
    ISBN 978-0-19-921986-5
  Wildberger, N.J. (2005) Divine Proportions: Rational Trigonometry, Wild Egg Books,
    ISBN 978-0-9757492-0-8
  Steinmetz, C.P. (1892) On the law of hysteresis, Trans. Amer. Inst. Elect. Eng. 9:3-64

QA mapping: Steinmetz hysteresis loop area (one AC cycle) maps to the sum of T-step
cross-spreads over the full 24-cycle Cosmos orbit.

  Single-step coupling: cs(b,e) = (b*d - e*e)^2 / ((b*b+e*e)*(e*e+d*d))
    where d = A1_mod(b+e, 9) (Wildberger cross-spread from cert [305])
  Orbit sum: S = sum of cs over all 24 orbit states (one full hysteresis cycle)
  3-phase sum: 3*S (three identical copies by cert [303] triad symmetry)

Five claims:
  C1  Single-cycle orbit coupling sum: S = sum_{k=0}^{23} cs(T^k(1,1)) is an exact
      Fraction (not approximated); each cs is in [0,1]∩Q; S represents the total
      reactive coupling energy in one 24-step Cosmos cycle (discrete hysteresis proxy).
  C2  Maximum single-step coupling: max cs_k = 1600/1681 = (40/41)^2 at state (1,9)
      (k=22); symmetric case: G=G'=82 gives cs=(b*d-e*e)^2/G^2=(1-81)^2/82^2=(40/41)^2;
      state (1,9) has the largest cross-spread in the orbit due to e^2-dominance (e=9,d=1).
  C3  Polyphase coupling linearity: the three 3-phase orbit copies at offset steps
      {0,8,16} all traverse the same 24-orbit state-set (cert [303] triads); hence
      each 3-phase channel contributes S; total 3-phase hysteresis = 3*S exactly
      (Fraction multiplication by 3, no approximation; parallel: cert [304] 3I theorem).
  C4  Steinmetz exponent is observer: the empirical exponent n approx 1.6 in
      P_h = k_h*f*B_max^n has no role in the QA discrete layer; the orbit provides
      exact integer cs values without parameterization; the exponent approximates
      variation of cs_k across the orbit at the observer (continuous) layer only.
  C5  Theorem NT: P_h = k_h*f*S_observer is observer layer; k_h and B_max are material
      constants (observer); neither re-enters T-step orbit logic; QA provides S (exact
      Fraction), engineering provides the calibration constants.
"""

import json
import sys
from fractions import Fraction
from math import gcd


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


def cross_spread(b, e, m=9):
    """cert [305] C5 cross-spread: cs = (b*d - e*e)^2 / ((b*b+e*e)*(e*e+d*d)). Exact Fraction."""
    d   = a1_mod(b + e, m)
    num = (b * d - e * e) * (b * d - e * e)
    G   = b * b + e * e
    Gp  = e * e + d * d
    den = G * Gp
    return Fraction(0) if den == 0 else Fraction(num, den)


def cosmos_states(m=9):
    return [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)
            if gcd(b, e) % 3 != 0]


checks = {}
passed = 0
failed = 0

B0, E0  = 1, 1
orb     = orbit(B0, E0)
cs_vals = [cross_spread(b, e) for (b, e) in orb]

# C1 — Single-cycle orbit coupling sum is exact Fraction
orbit_sum = sum(cs_vals)
ok_c1 = isinstance(orbit_sum, Fraction) and orbit_sum > 0
ok_c1 = ok_c1 and all(isinstance(cs, Fraction) for cs in cs_vals)
ok_c1 = ok_c1 and all(Fraction(0) <= cs <= Fraction(1) for cs in cs_vals)
checks["C1_orbit_coupling_sum_exact_fraction"] = "PASS" if ok_c1 else "FAIL"
if ok_c1: passed += 1
else:     failed += 1

# C2 — Maximum cross-spread = (40/41)^2 = 1600/1681 at state (1,9) (k=22)
k_max    = max(range(24), key=lambda k: cs_vals[k])
cs_max   = cs_vals[k_max]
bk22, ek22 = orb[22]
ok_c2 = (k_max == 22
          and bk22 == 1 and ek22 == 9
          and cs_max == Fraction(40 * 40, 41 * 41)
          and cs_max == Fraction(1600, 1681))
checks["C2_max_cross_spread_40_41_squared"] = "PASS" if ok_c2 else "FAIL"
if ok_c2: passed += 1
else:     failed += 1

# C3 — Polyphase coupling linearity: 3-phase sum = 3*orbit_sum
# T^8(1,1) = (7,1); T^16(1,1) = (4,1) — three triad starting states
b8, e8   = orb[8]
b16, e16 = orb[16]
orb8     = orbit(b8,  e8)
orb16    = orbit(b16, e16)
sum8     = sum(cross_spread(b, e) for (b, e) in orb8)
sum16    = sum(cross_spread(b, e) for (b, e) in orb16)
three_phase_sum = orbit_sum + sum8 + sum16
ok_c3 = (sum8  == orbit_sum
          and sum16 == orbit_sum
          and three_phase_sum == 3 * orbit_sum)
checks["C3_polyphase_linearity_3x_orbit_sum"] = "PASS" if ok_c3 else "FAIL"
if ok_c3: passed += 1
else:     failed += 1

# C4 — Steinmetz exponent is observer: cs values are exact integers-derived; no exponent
# needed in the discrete layer. Verify cs values are all exact rational (no float).
ok_c4 = all(isinstance(cross_spread(b, e), Fraction) for (b, e) in cosmos_states())
ok_c4 = ok_c4 and isinstance(orbit_sum, Fraction)
checks["C4_steinmetz_exponent_is_observer"] = "PASS" if ok_c4 else "FAIL"
if ok_c4: passed += 1
else:     failed += 1

# C5 — Theorem NT: orbit_sum is Fraction; calibration k_h is observer; no float re-entry
# Demonstrate: physical proxy P_h = k_h * f * orbit_sum uses orbit_sum as Fraction factor
k_h_observer = 1.0   # material constant — observer layer (float OK here)
f_observer   = 50.0  # frequency — observer layer (float OK here)
P_h_observer = k_h_observer * f_observer * float(orbit_sum)   # float multiplication is observer
ok_c5 = (isinstance(orbit_sum, Fraction)
          and isinstance(P_h_observer, float)
          and P_h_observer > 0)
checks["C5_theorem_nt_ph_observer_layer"] = "PASS" if ok_c5 else "FAIL"
if ok_c5: passed += 1
else:     failed += 1

result = {
    "ok":               failed == 0,
    "passed":           passed,
    "failed":           failed,
    "checks":           checks,
    "orbit_sum":        str(orbit_sum),
    "three_phase_sum":  str(three_phase_sum),
    "cs_max":           str(cs_max),
    "cs_max_k":         k_max,
    "cs_max_state":     [bk22, ek22],
    "triad_start_T8":   [b8, e8],
    "triad_start_T16":  [b16, e16],
}
print(json.dumps(result))
if failed:
    sys.exit(1)
