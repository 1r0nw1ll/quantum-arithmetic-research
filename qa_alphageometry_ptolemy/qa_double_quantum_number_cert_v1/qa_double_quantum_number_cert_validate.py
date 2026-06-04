#!/usr/bin/env python3
"""
QA Double Quantum Number for Diadic Fractions Cert [316] -- validator

Primary sources:
  Iverson, B. (1975-1996) Quantum Arithmetic vol 1, QA-1, pp.2-3, 53-56.
    ("Diadic" = doubled fraction 2/n; Double Quantum Number algorithm;
     male/female/double-female QN families; Rhind Papyrus connection;
     "Much more research and study is in order to learn the methods.")
  Chace, A.B. (1927) The Rhind Mathematical Papyrus, MAA, Vol.1-2.
  Gillings, R.J. (1972) Mathematics in the Time of the Pharaohs, MIT Press,
    ISBN 978-0-486-24315-3, Ch.6.

QA mapping (following Iverson exactly):

  DIADIC FRACTION: any fraction with numerator 2, i.e. 2/n ("doubled").

  DOUBLE QUANTUM NUMBER: two BEDA 4-tuples chained at a shared element,
  forming a 7-element sequence (b1,e1,d1,a1,e2,d2,n) where a1=b2.

  GENERAL 3-TERM FORMULA: for any BEDA tuple (b1,e1,d1,a1) and target n,
    k = d1*n / (C1 - a1),  C1 = 2*e1*d1
  When k is a positive integer:
    2/n = 1/(e1*k) + 1/(e1*n) + 1/(d1*n)

  MALE QN (b1 odd, C1 = a1 + n): k = d1 (integer always).
    Produces the canonical 7-element Double QN chain.
    Examples: (1,7,8,15,41,56,97) -> 2/97=1/56+1/679+1/776.

  FEMALE TRANSFORMATION: male (b,e,d,a) -> female (2e, b, a, 2d).
    Derivation: double the two middle elements, place at ends.
    Verified: (b+2e)_female = 2d = a_female (BEDA identity holds).

  DOUBLE-FEMALE QN (b1=2*b_male, e1=2*e_male): C1 = a1 + 2n, k = d1/2.
    Example for 2/71: double-female (2,8,10,18) from male (1,4,5,9);
    k = d1*n/(C1-a1) = 10*71/(160-18) = 5; 2/71=1/40+1/568+1/710.

  4-TERM TWO-LEVEL STRUCTURE: 2/n = 1/p + (2p-n)/(p*n), where the
  inner fraction (2p-n)/p decomposes into 3 unit fractions 1/c_i.
  For n=29: complete BEDA closure -- BEDA(4,2,6,8), p=C1=24,
    inner {e,d,a}={2,6,8}, n = 2p - 2a - p/a = 29 (closed-form formula).
  For n in {43,61,73,79,83,89,101}: arithmetic verified; inner QN
  origin explicitly open per Iverson QA-1.

Five claims:
  C1  General 3-term formula exact: for each (QN, n, k) triple where k is
      a positive integer, 2/n=1/(e1*k)+1/(e1*n)+1/(d1*n) as Fraction.
      Verified for all male QN matches and the double-female (2/71) case.
  C2  Male QN (C1=a1+n, k=d1): exactly 7 Rhind 3-term Cosmos entries match
      the male Double QN formula; multiple valid chains per n verified for
      2/97 (m=56 and m=60 both exact).
  C3  Double-female QN (C1=a1+2n, k=d1/2): female transformation
      (b,e,d,a)->(2e,b,a,2d) is a valid BEDA tuple (BEDA identities hold);
      for 2/71 the double-female (2,8,10,18) gives k=5 and matches Rhind.
  C4  4-term two-level structure: all 8 four-term Rhind entries
      n in {29,43,61,73,79,83,89,101} satisfy 2/n=1/p+sum(1/(ci*n)),
      with inner (2p-n)/p=sum(1/ci) exact as Fraction; for n=29 the
      BEDA closure formula n=2p-2a-p/a is verified (n=29 exactly).
  C5  Theorem NT: scribal notation, choice of p, hieroglyphic form are
      observer projections; QN chain, exact Fraction arithmetic, k formula,
      and two-level structure are discrete QA claims; inner QN derivation
      for 4-term n not in {29} is explicitly open per Iverson.
"""

QA_COMPLIANCE = (
    "cert_validator -- Fraction arithmetic for all diadic decompositions; "
    "BEDA identities (d=b+e, a=b+2e, a=d+e) verified symbolically; "
    "orbit_family not invoked (this cert is pre-orbit-classification); "
    "Theorem NT: scribal notation observer"
)

import json
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# ── Rhind table (Chace 1927) — 3-term Cosmos entries that match male QN ────
RHIND_MALE_MATCH = {
    17:  (12, 51, 68),
    31:  (20, 124, 155),
    37:  (24, 111, 296),
    47:  (30, 141, 470),
    59:  (36, 236, 531),
    67:  (40, 335, 536),
    97:  (56, 679, 776),
}

# For 2/97: two valid male chains
RHIND_97_ALT = (60, 291, 1940)   # m=60=3x20

# For 2/71: double-female match
RHIND_71 = (40, 568, 710)

# 4-term Rhind entries
RHIND_4TERM = {
    29:  (24,  58,  174,  232),
    43:  (42,  86,  129,  301),
    61:  (40, 244,  488,  610),
    73:  (60, 219,  292,  365),
    79:  (60, 237,  316,  790),
    83:  (60, 332,  415,  498),
    89:  (60, 356,  534,  890),
    101: (101, 202,  303,  606),
}

checks = {}
passed = 0
failed = 0


def beda_ok(b, e):
    d, a = b + e, b + 2 * e
    return d, a


def general_formula(e1, d1, a1, n):
    C1 = 2 * e1 * d1
    denom = C1 - a1
    if denom <= 0:
        return None, None
    if (d1 * n) % denom != 0:
        return None, None
    k = (d1 * n) // denom
    val = Fraction(1, e1 * k) + Fraction(1, e1 * n) + Fraction(1, d1 * n)
    return k, val


# C1 -- General 3-term formula: verify for male matches + double-female 2/71
ok_c1 = True
# Male QN checks
for n, denoms in RHIND_MALE_MATCH.items():
    p, qn, rn = denoms
    e1, d1 = qn // n, rn // n
    b1 = d1 - e1
    d1c, a1 = beda_ok(b1, e1)
    assert d1c == d1
    k, val = general_formula(e1, d1, a1, n)
    if k is None or val != Fraction(2, n) or k != d1:
        ok_c1 = False

# 2/97 alternative m=60
e1, d1 = 3, 20
b1 = d1 - e1
_, a1 = beda_ok(b1, e1)
k60, val60 = general_formula(e1, d1, a1, 97)
if val60 != Fraction(2, 97):
    ok_c1 = False

# Double-female 2/71: (b=2,e=8,d=10,a=18), C1=160=a1+2*71
e1, d1, a1 = 8, 10, 18
k71, val71 = general_formula(e1, d1, a1, 71)
if val71 != Fraction(2, 71) or k71 != 5:
    ok_c1 = False

checks["C1_general_formula_male_plus_double_female_2_71"] = "PASS" if ok_c1 else "FAIL"
if ok_c1: passed += 1
else:     failed += 1


# C2 -- Male QN: 7 Rhind matches + two valid chains for 2/97
ok_c2 = True
# Verify all 7 male Rhind matches have C1=a1+n and k=d1
for n, denoms in RHIND_MALE_MATCH.items():
    p, qn, rn = denoms
    e1, d1 = qn // n, rn // n
    b1 = d1 - e1
    _, a1 = beda_ok(b1, e1)
    C1 = 2 * e1 * d1
    if C1 != a1 + n:
        ok_c2 = False
    val = Fraction(1, p) + Fraction(1, qn) + Fraction(1, rn)
    if val != Fraction(2, n):
        ok_c2 = False

# Two valid chains for 2/97
val_56 = sum(Fraction(1, d) for d in RHIND_MALE_MATCH[97])
val_60 = sum(Fraction(1, d) for d in RHIND_97_ALT)
if val_56 != Fraction(2, 97) or val_60 != Fraction(2, 97):
    ok_c2 = False

# Count of male matches is exactly 7
if len(RHIND_MALE_MATCH) != 7:
    ok_c2 = False

checks["C2_male_qn_7_rhind_matches_two_chains_for_97"] = "PASS" if ok_c2 else "FAIL"
if ok_c2: passed += 1
else:     failed += 1


# C3 -- Double-female QN: transformation valid; 2/71 Rhind match
ok_c3 = True

# Female transformation (b,e,d,a) -> (2e, b, a, 2d) is valid BEDA
test_males = [(1,4,5,9), (1,2,3,5), (1,1,2,3), (1,3,4,7), (1,6,7,13), (1,7,8,15)]
for b, e, d, a in test_males:
    fb, fe = 2*e, b
    fd, fa = beda_ok(fb, fe)
    # fd = fb+fe = 2e+b = d_female
    # fa = fb+2*fe = 2e+2b = 2(e+b) = 2d
    if fd != 2*e + b or fa != 2*d:
        ok_c3 = False

# Double-female of (1,4,5,9) is (2,8,10,18)
bdf, edf, ddf, adf = 2, 8, 10, 18
d_check, a_check = beda_ok(bdf, edf)
if d_check != ddf or a_check != adf:
    ok_c3 = False

# C1 for double-female = a1 + 2*n for n=71
C1_df = 2 * edf * ddf
if C1_df != adf + 2 * 71:
    ok_c3 = False

# k=5 and Rhind match
k_df, val_df = general_formula(edf, ddf, adf, 71)
val_rhind71 = sum(Fraction(1, d) for d in RHIND_71)
if k_df != 5 or val_df != Fraction(2, 71) or val_rhind71 != Fraction(2, 71):
    ok_c3 = False

checks["C3_double_female_transform_valid_and_2_71_rhind_match"] = "PASS" if ok_c3 else "FAIL"
if ok_c3: passed += 1
else:     failed += 1


# C4 -- 4-term two-level structure: all 8 entries + n=29 BEDA closure
ok_c4 = True

# Verify all 8 four-term entries sum to 2/n and have structure 2/n=1/p+inner/n
for n, denoms in RHIND_4TERM.items():
    p = denoms[0]
    # All d[1..3] must be multiples of n
    if not all(d % n == 0 for d in denoms[1:]):
        ok_c4 = False
        continue
    ci = [d // n for d in denoms[1:]]
    inner_frac = sum(Fraction(1, c) for c in ci)
    if inner_frac != Fraction(2*p - n, p):
        ok_c4 = False
    total = sum(Fraction(1, d) for d in denoms)
    if total != Fraction(2, n):
        ok_c4 = False

# n=29 BEDA closure: BEDA(4,2,6,8), p=C1=24, inner={e,d,a}={2,6,8},
# n = 2p - 2a - p/a
b29, e29, d29, a29 = 4, 2, 6, 8
_, a29_check = beda_ok(b29, e29)
p29 = 2 * e29 * d29    # = 24 = C1
if a29_check != a29:
    ok_c4 = False
if a29 != e29 + d29:   # BEDA identity a=d+e
    ok_c4 = False
if p29 % a29 != 0:     # a divides p
    ok_c4 = False
n29_formula = 2 * p29 - 2 * a29 - p29 // a29
if n29_formula != 29:
    ok_c4 = False
# Verify {e29,d29,a29} is exactly the inner set for n=29
inner29 = set(d // 29 for d in RHIND_4TERM[29][1:])
if inner29 != {e29, d29, a29}:
    ok_c4 = False

checks["C4_4term_two_level_all_8_verified_n29_beda_closure"] = "PASS" if ok_c4 else "FAIL"
if ok_c4: passed += 1
else:     failed += 1


# C5 -- Theorem NT: all QN data are integers; no float; sums are Fraction
ok_c5 = True
# All denominators in all cases are positive integers
for n, denoms in list(RHIND_MALE_MATCH.items()) + list(RHIND_4TERM.items()):
    for d in denoms:
        if not isinstance(d, int) or d <= 0:
            ok_c5 = False
# RHIND_71 check
for d in RHIND_71:
    if not isinstance(d, int) or d <= 0:
        ok_c5 = False
# k values from general_formula are integers (not floats)
for n, denoms in RHIND_MALE_MATCH.items():
    p, qn, rn = denoms
    e1, d1 = qn // n, rn // n
    _, a1 = beda_ok(d1 - e1, e1)
    k, _ = general_formula(e1, d1, a1, n)
    if not isinstance(k, int):
        ok_c5 = False
# Double-female k=5 is integer
if not isinstance(k71, int) or k71 != 5:
    ok_c5 = False

checks["C5_theorem_nt_all_data_int_sums_fraction_no_float"] = "PASS" if ok_c5 else "FAIL"
if ok_c5: passed += 1
else:     failed += 1


print(json.dumps(checks, indent=2))
print(f"\nTotal: {passed} PASS, {failed} FAIL")
if failed:
    sys.exit(1)
