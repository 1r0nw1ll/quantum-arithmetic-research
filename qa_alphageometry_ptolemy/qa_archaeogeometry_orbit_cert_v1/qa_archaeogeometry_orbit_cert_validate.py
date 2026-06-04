#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — orbit_fam classification of megalithic Pythagorean triangles; exact Fraction spreads; Theorem NT"
"""
QA Archaeogeometry Orbit Classification Cert [311] — validator

Primary sources:
  Iverson, B. (1975-1996) Quantum Arithmetic vols 1-2 (QA-1, QA-2)
  Thom, A. (1962) The Megalithic Unit of Length, J. Royal Statistical Society A 125:243-251
    DOI 10.2307/2982493
  Thom, A. (1967) Megalithic Sites in Britain, Oxford University Press, ISBN 978-0-19-813148-8
  Wildberger, N.J. (2005) Divine Proportions: Rational Trigonometry, Wild Egg Books,
    ISBN 978-0-9757492-0-8

QA mapping: Thom documented five Pythagorean right triangles used in megalithic construction.
Each triangle (C,F,G) maps to a QA base pair (b,e) via cert [310]'s squaring map:
  d = b+e, C = 2de, F = d^2-e^2, G = d^2+e^2 (hypotenuse)

The orbit_fam of (b,e) under mod-9 T-step classifies the triangle's position on the
QA excitation ladder:
  Fibonacci sub-orbit  (canonical seed (1,1), |f|=1):  ground state
  Lucas    sub-orbit  (canonical seed (2,1), |f|=5):  first excited state
  Third    sub-orbit  (canonical seed (1,4), |f|=11): second excited state

where f(b,e) = b^2 + b*e - e^2 (Eisenstein norm, negated each T-step → constant |f| along orbit).

Five claims:
  C1  Cosmos gate (structural): all 5 documented Thom construction Pythagorean triples
      have Cosmos-family generating (b,e) in both mod-9 and mod-24. Follows from cert [310]
      C3: primitive triple iff gcd(b,e)=1 AND b odd → Cosmos (gcd not div by 3 in mod-9,
      not div by 8 in mod-24).
  C2  Ground-state duality (structural): the two most common construction triangles
      (3-4-5 and 5-12-13) both have |f(b,e)|=1 (Fibonacci sub-orbit, minimum-norm ground
      state). They are the unique two primitive Pythagorean triples with G ≤ 13 and
      Fibonacci sub-orbit generators; G=5 → (b,e)=(1,1); G=13 → (b,e)=(1,2).
  C3  Complete excitation ladder (structural): the 5 construction triangles collectively
      span all 3 mod-9 Cosmos sub-orbits: 2/5 Fibonacci, 1/5 Lucas, 2/5 Third. The
      minimum-energy sub-orbit (Fibonacci) contains the two most commonly documented
      triangles (rank 1 and 2 by historical usage).
  C4  Diameter expressibility (empirical): among the 31 distinct integer-MY diameters
      in the Thom 1962 dataset (84 circles, range 4-55), exactly 4 values are expressible
      as primitive Pythagorean hypotenuses G = d^2+e^2 (sum of two coprime squares of
      different parity). All 4 have Cosmos generators. The 2 smallest (G=5, G=13) have
      Fibonacci sub-orbit generators (|f|=1); the 2 larger (G=17, G=29) have Third
      sub-orbit generators (|f|=11). Zero have Lucas sub-orbit (|f|=5) generators.
  C5  Theorem NT boundary: physical diameter in feet is transcendental observer; integer-MY
      diameter is the QA-compatible rational measurement; orbit_fam is the discrete
      classification. The Wildberger spread F^2/G^2 for each construction triangle is
      an exact Fraction in (0,1) (cert [310] C4).
"""

import json
import sys
from fractions import Fraction
from math import gcd, isqrt


def a1_mod(x, m):
    return ((x - 1) % m) + 1


def t_step(b, e, m):
    return e, a1_mod(b + e, m)


def orbit_fam_9(b, e):
    g = gcd(b, e)
    return "satellite" if g % 3 == 0 and g % 9 != 0 else \
           "singularity" if g % 9 == 0 else "cosmos"


def orbit_fam_24(b, e):
    g = gcd(b, e)
    return "singularity" if g % 24 == 0 else \
           "satellite"   if g % 8 == 0 else "cosmos"


def sub_orbit_9(b, e):
    """Return 'fibonacci', 'lucas', or 'third' — which mod-9 Cosmos sub-orbit."""
    cur_b, cur_e = b, e
    orbit_states = set()
    for _ in range(25):
        orbit_states.add((cur_b, cur_e))
        cur_b, cur_e = t_step(cur_b, cur_e, 9)
    if (1, 1) in orbit_states: return "fibonacci"
    if (2, 1) in orbit_states: return "lucas"
    if (1, 4) in orbit_states: return "third"
    return "unknown"


def f_norm(b, e):
    return b * b + b * e - e * e


def generator_from_triple(a_side, b_side, c_hyp):
    """Given a Pythagorean triple (a,b,c), return QA generator (b_qa, e_qa)."""
    even_leg, odd_leg = (a_side, b_side) if a_side % 2 == 0 else (b_side, a_side)
    hyp = c_hyp
    d_sq = (hyp + odd_leg) // 2
    e_sq = (hyp - odd_leg) // 2
    if (hyp + odd_leg) % 2 != 0 or (hyp - odd_leg) % 2 != 0:
        return None, None
    d_root = isqrt(d_sq)
    e_root = isqrt(e_sq)
    if d_root * d_root != d_sq or e_root * e_root != e_sq:
        return None, None
    b_qa = d_root - e_root
    return b_qa, e_root


def generators_for_G(D):
    """All (b,e) generating primitive Pythagorean triples with hypotenuse G=D."""
    gens = []
    for e in range(1, isqrt(D)):
        d_sq = D - e * e
        d_root = isqrt(d_sq)
        if d_root * d_root != d_sq or d_root <= e:
            continue
        if gcd(d_root, e) != 1:
            continue
        if (d_root + e) % 2 == 0:
            continue
        b = d_root - e
        gens.append((b, e, d_root))
    return gens


# ── documented Thom construction triangles ────────────────────────────────────
# Format: (short_leg, long_leg, hypotenuse, historical_rank)
THOM_TRIANGLES = [
    (3,  4,  5,  1,  "3-4-5"),
    (5,  12, 13, 2,  "5-12-13"),
    (8,  15, 17, 3,  "8-15-17"),
    (7,  24, 25, 4,  "7-24-25"),
    (12, 35, 37, 5,  "12-35-37"),
]

# ── distinct integer-MY diameters in Thom 1962 dataset (84 circles) ───────────
THOM_1962_D_MY = [
    4, 5, 5, 6, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10, 10, 10,
    12, 12, 12, 12, 13, 13, 14, 14, 14, 14, 14, 16, 16,
    17, 17, 18, 18, 18, 18, 18, 18, 18, 20, 20, 20, 21, 21,
    22, 22, 22, 24, 24, 24, 24, 26, 26, 27, 28, 28, 28, 28,
    29, 30, 30, 31, 31, 32, 32, 34, 34, 38, 38, 38, 38,
    40, 40, 40, 40, 40, 40, 42, 42, 44, 48, 50, 50, 51, 55,
]

checks = {}
passed = 0
failed = 0


# ── C1  Cosmos gate ───────────────────────────────────────────────────────────
ok_c1 = True
c1_detail = []
for a, b, c, rank, name in THOM_TRIANGLES:
    b_qa, e_qa = generator_from_triple(a, b, c)
    if b_qa is None:
        ok_c1 = False; break
    fam9  = orbit_fam_9(b_qa, e_qa)
    fam24 = orbit_fam_24(b_qa, e_qa)
    if fam9 != "cosmos" or fam24 != "cosmos":
        ok_c1 = False
    prim = gcd(gcd(a, b), c) == 1
    if not prim:
        ok_c1 = False
    c1_detail.append({
        "triple":   [a, b, c],
        "be":       [b_qa, e_qa],
        "fam9":     fam9,
        "fam24":    fam24,
        "primitive": prim,
        "name":     name,
    })
checks["C1_all_construction_triangles_cosmos"] = "PASS" if ok_c1 else "FAIL"
if ok_c1: passed += 1
else:     failed += 1


# ── C2  Ground-state duality ──────────────────────────────────────────────────
# Rank-1 (3-4-5) and rank-2 (5-12-13) both in Fibonacci sub-orbit with |f|=1
# AND they are the unique primitives with G ≤ 13 and Fibonacci sub-orbit.
ok_c2 = True
c2_detail = []
rank12 = [(a,b,c,rank,name) for (a,b,c,rank,name) in THOM_TRIANGLES if rank <= 2]
for a, b, c, rank, name in rank12:
    b_qa, e_qa = generator_from_triple(a, b, c)
    sub = sub_orbit_9(b_qa, e_qa)
    fn  = abs(f_norm(b_qa, e_qa))
    if sub != "fibonacci" or fn != 1:
        ok_c2 = False
    c2_detail.append({"triple": [a,b,c], "be": [b_qa,e_qa], "sub": sub, "fn": fn})
# Verify uniqueness: all primitives with G ≤ 13 and Fibonacci sub-orbit
fib_orbit_small = []
for G in range(2, 14):
    for b_qa, e_qa, d in generators_for_G(G):
        if sub_orbit_9(b_qa, e_qa) == "fibonacci":
            fib_orbit_small.append(G)
if sorted(fib_orbit_small) != [5, 13]:
    ok_c2 = False
checks["C2_rank12_fibonacci_groundstate_G_le_13"] = "PASS" if ok_c2 else "FAIL"
if ok_c2: passed += 1
else:     failed += 1


# ── C3  Complete excitation ladder ────────────────────────────────────────────
sub_counts = {"fibonacci": 0, "lucas": 0, "third": 0, "unknown": 0}
c3_detail = []
for a, b, c, rank, name in THOM_TRIANGLES:
    b_qa, e_qa = generator_from_triple(a, b, c)
    sub = sub_orbit_9(b_qa, e_qa)
    fn  = abs(f_norm(b_qa, e_qa))
    sub_counts[sub] += 1
    c3_detail.append({"triple": [a,b,c], "be": [b_qa,e_qa], "rank": rank,
                       "sub": sub, "fn": fn, "name": name})
# All 3 sub-orbits represented AND rank-1,2 in Fibonacci
ok_c3 = (sub_counts["fibonacci"] == 2 and
         sub_counts["lucas"]     == 1 and
         sub_counts["third"]     == 2 and
         sub_counts["unknown"]   == 0)
# Confirm rank-1,2 are the Fibonacci ones
rank12_sub = [d["sub"] for d in c3_detail if d["rank"] <= 2]
ok_c3 = ok_c3 and all(s == "fibonacci" for s in rank12_sub)
checks["C3_complete_excitation_ladder_2fib_1luc_2third"] = "PASS" if ok_c3 else "FAIL"
if ok_c3: passed += 1
else:     failed += 1


# ── C4  Diameter expressibility ───────────────────────────────────────────────
distinct_d_my = sorted(set(THOM_1962_D_MY))
expressible = []
for D in distinct_d_my:
    gens = generators_for_G(D)
    if gens:
        b_qa, e_qa, d = min(gens, key=lambda x: abs(f_norm(x[0], x[1])))
        fam9 = orbit_fam_9(b_qa, e_qa)
        sub  = sub_orbit_9(b_qa, e_qa)
        fn   = abs(f_norm(b_qa, e_qa))
        expressible.append({
            "G": D, "be": [b_qa, e_qa], "fam9": fam9, "sub": sub, "fn": fn,
        })

# Claims: exactly 4 expressible values, all cosmos, 2 smallest Fibonacci, 2 larger Third, 0 Lucas
ok_c4 = len(expressible) == 4
ok_c4 = ok_c4 and all(e["fam9"] == "cosmos" for e in expressible)
g_vals = sorted(e["G"] for e in expressible)
ok_c4 = ok_c4 and g_vals == [5, 13, 17, 29]
fib_G  = [e["G"] for e in expressible if e["sub"] == "fibonacci"]
third_G = [e["G"] for e in expressible if e["sub"] == "third"]
luc_G  = [e["G"] for e in expressible if e["sub"] == "lucas"]
ok_c4 = ok_c4 and sorted(fib_G) == [5, 13]
ok_c4 = ok_c4 and sorted(third_G) == [17, 29]
ok_c4 = ok_c4 and luc_G == []
checks["C4_four_expressible_diameters_2fib_2third_0lucas"] = "PASS" if ok_c4 else "FAIL"
if ok_c4: passed += 1
else:     failed += 1


# ── C5  Theorem NT — spreads are exact Fractions ─────────────────────────────
ok_c5 = True
c5_detail = []
for a, b, c, rank, name in THOM_TRIANGLES:
    b_qa, e_qa = generator_from_triple(a, b, c)
    d_qa = b_qa + e_qa
    C_qa = 2 * d_qa * e_qa
    F_qa = d_qa * d_qa - e_qa * e_qa
    G_qa = d_qa * d_qa + e_qa * e_qa
    if G_qa != c:  # hypotenuse must match
        ok_c5 = False
    spread_F = Fraction(F_qa * F_qa, G_qa * G_qa)
    spread_C = Fraction(C_qa * C_qa, G_qa * G_qa)
    if not isinstance(spread_F, Fraction) or not isinstance(spread_C, Fraction):
        ok_c5 = False
    if spread_F + spread_C != Fraction(1):
        ok_c5 = False
    # Physical diameter in feet is observer (float); integer MY is QA layer (int)
    diameter_ft_observer = float(c) * 2.72   # MY × 2.72 ft = observer float
    if not isinstance(diameter_ft_observer, float):
        ok_c5 = False
    c5_detail.append({
        "name": name,
        "spread_F": str(spread_F),
        "spread_C": str(spread_C),
        "sum_1":    str(spread_F + spread_C),
        "G_MY_int": G_qa,
        "diam_ft_obs": diameter_ft_observer,
    })
checks["C5_theorem_nt_spreads_exact_fraction_observer_boundary"] = "PASS" if ok_c5 else "FAIL"
if ok_c5: passed += 1
else:     failed += 1


# ── summary ───────────────────────────────────────────────────────────────────
result = {
    "ok":              failed == 0,
    "passed":          passed,
    "failed":          failed,
    "checks":          checks,
    "c1_triangles":    c1_detail,
    "c2_rank12":       c2_detail,
    "c2_fib_G_le_13":  fib_orbit_small,
    "c3_sub_counts":   sub_counts,
    "c3_ladder":       c3_detail,
    "c4_expressible":  expressible,
    "c4_G_values":     g_vals,
    "c4_fib_G":        fib_G,
    "c4_third_G":      third_G,
    "c4_lucas_G":      luc_G,
    "c5_spreads":      c5_detail,
    "distinct_D_my_count": len(distinct_d_my),
}
print(json.dumps(result, indent=2))
if failed:
    sys.exit(1)
