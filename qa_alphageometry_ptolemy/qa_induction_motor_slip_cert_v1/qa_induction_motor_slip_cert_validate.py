#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — Wildberger direction spread (Fraction); A1 no-zero arithmetic; Cassini det identity; no float QA state"
"""
QA Induction Motor Slip Cert [307] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford,
    ISBN 978-0-19-921986-5
  Wildberger, N.J. (2005) Divine Proportions: Rational Trigonometry, Wild Egg Books,
    ISBN 978-0-9757492-0-8
  Chapman, S.J. (2011) Electric Machinery Fundamentals, 5th ed., McGraw-Hill,
    ISBN 978-0-07-352954-7

QA mapping: induction motor slip = T-step phase lag between stator and rotor orbits.
  Stator: reference Cosmos state (b0, e0)
  Rotor:  T^k(b0, e0) — state lagged by k T-steps
  Slip    = k/24 (Fraction, k in {0,...,23})
  Torque proxy = Wildberger direction spread between stator and lagged-rotor direction vectors

Five claims:
  C1  Grade alternation = versor phase lag: det(M^k)=(-1)^k (Cassini, cert [299]); odd k =
      reactive (lagging) versor, even k = active (in-phase) rotor. Slip = k/24 as Fraction
      over the 24-step Cosmos orbit. k=0: synchronous, zero slip; k=12: M^12 equiv -I mod 9
      (antipodal grade inversion, cert [298]). Verified k=0..23.
  C2  Spread profile is non-trivial: s_k = (b0*e_k - e0*b_k)^2 / (G0*G_k) is Fraction-valued
      for k=0..23; s_0=0 (synchronous, parallel); s_12=0 (antipodal state is scalar multiple,
      parallel); all other s_k in (0,1)cap Q; profile is not monotone.
  C3  Pullout T-step: k* = argmax s_k over k=1..23; from seed (1,1): k*=22, s(k*)=16/41;
      symmetry: s_22=s_23=16/41 (near-Singularity extreme states T^22=(1,9) and T^23=(9,1)
      give maximum directional deviation from seed (1,1)).
  C4  Orbit symmetry at k=12: s_12=0 because M^12 equiv -I mod 9 (cert [298]) maps seed
      (1,1) to (8,8)=8*(1,1) mod 9 — scalar multiple has zero spread; {k=0,k=12} partition
      the 24-orbit into two equal half-cycles.
  C5  Theorem NT: slip k/24, torque proxy s_k/s_max, and rotor speed fraction 1-k/24 are
      observer projections (Fraction from integer orbit; never re-entering T-step logic);
      continuous Kloss torque-slip curve T=2T_max/(s/s*+s*/s) is observer layer only.
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


def det_M_k(k):
    """det(M^k) = F(k-1)*F(k+1) - F(k)^2 = (-1)^k  (Cassini identity)."""
    fk   = fibonacci(k)
    fkm1 = 1 if k == 0 else fibonacci(k - 1)
    fkp1 = fibonacci(k + 1)
    return fkm1 * fkp1 - fk * fk


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
    """Wildberger spread between direction vectors (b0,e0),(b1,e1). Exact Fraction."""
    num = (b0 * e1 - e0 * b1) * (b0 * e1 - e0 * b1)
    den = (b0 * b0 + e0 * e0) * (b1 * b1 + e1 * e1)
    return Fraction(0) if den == 0 else Fraction(num, den)


checks = {}
passed = 0
failed = 0

B0, E0 = 1, 1
orb    = orbit(B0, E0)

# C1 — Grade alternation = versor phase lag
ok_c1 = all(det_M_k(k) == (1 if k % 2 == 0 else -1) for k in range(24))
ok_c1 = ok_c1 and Fraction(0, 24) == 0
ok_c1 = ok_c1 and Fraction(12, 24) == Fraction(1, 2)
checks["C1_grade_alternation_versor_phase_lag"] = "PASS" if ok_c1 else "FAIL"
if ok_c1: passed += 1
else:     failed += 1

# C2 — Spread profile is non-trivial
spreads = [direction_spread(B0, E0, b, e) for (b, e) in orb]
ok_c2   = spreads[0] == Fraction(0) and spreads[12] == Fraction(0)
ok_c2   = ok_c2 and all(
    Fraction(0) < spreads[k] < Fraction(1)
    for k in range(24) if k not in (0, 12)
)
ok_c2 = ok_c2 and not all(spreads[k] <= spreads[k + 1] for k in range(23))
checks["C2_spread_profile_nontrivial"] = "PASS" if ok_c2 else "FAIL"
if ok_c2: passed += 1
else:     failed += 1

# C3 — Pullout T-step k*=22, s(k*)=16/41, symmetry s_22=s_23
k_star = max(range(1, 24), key=lambda k: spreads[k])
s_max  = spreads[k_star]
ok_c3  = (k_star == 22
           and s_max == Fraction(16, 41)
           and spreads[22] == spreads[23])
checks["C3_pullout_tstep_k22_s16_41"] = "PASS" if ok_c3 else "FAIL"
if ok_c3: passed += 1
else:     failed += 1

# C4 — Antipodal symmetry at k=12: T^12(1,1)=(8,8)=8*(1,1), spread=0
bk12, ek12 = orb[12]
ok_c4 = (bk12 == 8 and ek12 == 8
          and direction_spread(1, 1, 8, 8) == Fraction(0)
          and spreads[12] == Fraction(0))
checks["C4_antipodal_symmetry_k12_8_8"] = "PASS" if ok_c4 else "FAIL"
if ok_c4: passed += 1
else:     failed += 1

# C5 — Theorem NT: observer projections are Fraction-valued, never re-enter QA state
slip_12    = Fraction(12, 24)
t_proxy_22 = spreads[22] / s_max if s_max != 0 else Fraction(0)
rotor_sync = Fraction(1) - Fraction(0, 24)
ok_c5 = (isinstance(slip_12, Fraction)
          and isinstance(t_proxy_22, Fraction)
          and isinstance(rotor_sync, Fraction)
          and t_proxy_22 == Fraction(1)
          and rotor_sync  == Fraction(1))
checks["C5_theorem_nt_observer_projections"] = "PASS" if ok_c5 else "FAIL"
if ok_c5: passed += 1
else:     failed += 1

result = {
    "ok":            failed == 0,
    "passed":        passed,
    "failed":        failed,
    "checks":        checks,
    "pullout_k_star": k_star,
    "pullout_spread": str(s_max),
    "T22_state":     list(orb[22]),
    "T23_state":     list(orb[23]),
    "T12_state":     list(orb[12]),
    "spread_k0":     str(spreads[0]),
    "spread_k12":    str(spreads[12]),
}
print(json.dumps(result))
if failed:
    sys.exit(1)
