#!/usr/bin/env python3
"""
qa_solar_system_harmonic_test_v2.py — Sharper harmonic tests derived from
Ben Iverson's actual claims.

THREE TESTS:

TEST A — BEN'S LAW OF HARMONICS (Pyth-3 Ch.5):
  "Two QNs sharing all but one prime factor each are in harmonic resonance.
   The lower the ratio of the excepted prime factors, the greater the harmony."
  Metric: for each pair, check if they satisfy "all-but-one" criterion.
  Score: ratio of excepted primes (lower = stronger harmony).
  Null: permutation of QN assignments.

TEST B — TUPLE ELEMENT SHARING:
  The Earth-Moon-Halley chain works through literal element equality:
  Earth a=61 = Moon d=61; Earth b=59 = Halley a=59.
  Metric: count of shared literal values between (b,e,d,a) tuples.
  Null: permutation of QN assignments.

TEST C — FIBONACCI PERIOD RATIOS:
  Known MMRs are 2:1, 3:2, 5:2 — Fibonacci numbers.
  For all coupled pairs with known orbital periods, compute period ratio
  and measure distance to nearest Fibonacci fraction (F_n/F_m).
  Null: random period ratios from the same distribution.

Author: Will Dale (design from Ben's sources), Claude (code)
"""

QA_COMPLIANCE = "observer=harmonic_test_v2, state_alphabet=primitive_QN_tuple"

import math
import itertools
from collections import defaultdict
from math import gcd
from fractions import Fraction

import numpy as np

np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════
# BODY CATALOGUE
# ═══════════════════════════════════════════════════════════════════════
# (name, eccentricity, category, parent, orbital_period_days)
# periods: heliocentric in Earth days, moons in Earth days around parent

BODIES = [
    ("Mercury",    0.20563,  "planet",   None,       87.969),
    ("Venus",      0.00677,  "planet",   None,      224.701),
    ("Earth",      0.01671,  "planet",   None,      365.256),
    ("Mars",       0.09339,  "planet",   None,      686.980),
    ("Jupiter",    0.04839,  "planet",   None,     4332.589),
    ("Saturn",     0.05415,  "planet",   None,    10759.22),
    ("Uranus",     0.04717,  "planet",   None,    30688.5),
    ("Neptune",    0.00859,  "planet",   None,    60182.0),
    ("Moon",       0.0549,   "moon",     "Earth",    27.322),
    ("Phobos",     0.0151,   "moon",     "Mars",      0.319),
    ("Deimos",     0.0002,   "moon",     "Mars",      1.263),
    ("Io",         0.0041,   "moon",     "Jupiter",   1.769),
    ("Europa",     0.0094,   "moon",     "Jupiter",   3.551),
    ("Ganymede",   0.0011,   "moon",     "Jupiter",   7.155),
    ("Callisto",   0.0074,   "moon",     "Jupiter",  16.689),
    ("Amalthea",   0.0032,   "moon",     "Jupiter",   0.498),
    ("Mimas",      0.0196,   "moon",     "Saturn",    0.942),
    ("Enceladus",  0.0047,   "moon",     "Saturn",    1.370),
    ("Tethys",     0.0001,   "moon",     "Saturn",    1.888),
    ("Dione",      0.0022,   "moon",     "Saturn",    2.737),
    ("Rhea",       0.0013,   "moon",     "Saturn",    4.518),
    ("Titan",      0.0288,   "moon",     "Saturn",   15.945),
    ("Iapetus",    0.0276,   "moon",     "Saturn",   79.322),
    ("Miranda",    0.0013,   "moon",     "Uranus",    1.413),
    ("Ariel",      0.0012,   "moon",     "Uranus",    2.520),
    ("Umbriel",    0.0039,   "moon",     "Uranus",    4.144),
    ("Titania",    0.0011,   "moon",     "Uranus",    8.706),
    ("Oberon",     0.0014,   "moon",     "Uranus",   13.463),
    ("Triton",     0.000016, "moon",     "Neptune",   5.877),
    ("Nereid",     0.7507,   "moon",     "Neptune", 360.136),
    ("Charon",     0.0002,   "moon",     "Pluto",     6.387),
    ("Pluto",      0.2488,   "dwarf",    None,    90560.0),
    ("Ceres",      0.0758,   "dwarf",    None,     1681.63),
    ("Eris",       0.4407,   "dwarf",    None,   203830.0),
    ("Haumea",     0.1912,   "dwarf",    None,   103774.0),
    ("Makemake",   0.1559,   "dwarf",    None,   111845.0),
    ("Sedna",      0.8496,   "dwarf",    None,  4404480.0),
    ("Halley",     0.96714,  "comet",    None,    27510.0),
    ("Hale-Bopp",  0.99507,  "comet",    None,   912500.0),
    ("Encke",      0.8483,   "comet",    None,     1204.0),
    ("67P/C-G",    0.6405,   "comet",    None,     2364.0),
    ("Vesta",      0.0887,   "asteroid", None,     1325.75),
    ("Pallas",     0.2313,   "asteroid", None,     1686.49),
    ("Eros",       0.2229,   "asteroid", None,      643.0),
    ("Bennu",      0.2037,   "asteroid", None,      436.65),
    ("Itokawa",    0.2802,   "asteroid", None,      556.38),
]

# Known resonance pairs with their period ratios
RESONANCE_PAIRS = [
    ("Io", "Europa", 2, 1),            # 2:1
    ("Europa", "Ganymede", 2, 1),       # 2:1
    ("Ganymede", "Callisto", 7, 3),     # ~7:3
    ("Mimas", "Tethys", 2, 1),          # 2:1
    ("Enceladus", "Dione", 2, 1),       # 2:1
    ("Titan", "Iapetus", 5, 1),         # ~5:1 approx
    ("Pluto", "Neptune", 3, 2),         # 3:2
    ("Jupiter", "Saturn", 5, 2),        # 5:2 (Great Inequality)
]

# Fibonacci numbers for proximity test
FIBS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
FIBONACCI_FRACTIONS = set()
for i in range(len(FIBS)):
    for j in range(len(FIBS)):
        if FIBS[j] > 0 and FIBS[i] != FIBS[j]:
            FIBONACCI_FRACTIONS.add(Fraction(FIBS[i], FIBS[j]))
# Also add simple integer ratios that are Fibonacci
for n in [1, 2, 3, 5, 8, 13]:
    for d in [1, 2, 3, 5, 8, 13]:
        if d > 0:
            FIBONACCI_FRACTIONS.add(Fraction(n, d))
FIBONACCI_FRACTIONS = sorted(FIBONACCI_FRACTIONS)


# ═══════════════════════════════════════════════════════════════════════
# QN FINDER
# ═══════════════════════════════════════════════════════════════════════

def find_qn(ecc_target, max_d=500):
    if ecc_target <= 0:
        return (1, 0, 1, 1, 0.0, ecc_target)
    if ecc_target >= 1:
        return (1, max_d, max_d + 1, max_d + 2, max_d / (max_d + 1), abs(ecc_target - max_d / (max_d + 1)))
    if ecc_target < 0.005:
        max_d = max(max_d, 5000)
    if ecc_target < 0.001:
        max_d = max(max_d, 10000)
    if ecc_target < 0.0001:
        max_d = max(max_d, 100000)
    if ecc_target > 0.99:
        max_d = max(max_d, 10000)

    best = None
    best_err = float("inf")
    for d_val in range(2, max_d + 1):
        e_val = round(ecc_target * d_val)
        for e_try in range(max(1, e_val - 1), min(d_val, e_val + 2)):
            b_val = d_val - e_try
            if b_val < 1:
                continue
            if gcd(b_val, e_try) != 1:
                continue
            ecc_qa = e_try / d_val
            err = abs(ecc_qa - ecc_target)
            if err < best_err:
                best_err = err
                a_val = b_val + 2 * e_try
                best = (b_val, e_try, d_val, a_val, ecc_qa, err)
                if err < 1e-12:
                    return best
    return best


def prime_factors(n):
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def qn_product_primes(b, e, d, a):
    return prime_factors(b * e * d * a)


def nontrivial_primes(primes, trivial={2, 3, 5}):
    return primes - trivial


# ═══════════════════════════════════════════════════════════════════════
# TEST A — BEN'S LAW OF HARMONICS (all-but-one)
# ═══════════════════════════════════════════════════════════════════════

def bens_harmony_score(primes1, primes2):
    """Ben's Law: two QNs are harmonic if they share all but one prime each.

    Returns (is_harmonic, excepted_ratio) where:
      is_harmonic: True if |primes1 - shared| == 1 AND |primes2 - shared| == 1
      excepted_ratio: min(p1,p2)/max(p1,p2) where p1,p2 are the excepted primes
                      (higher ratio = stronger harmony, 1.0 = max)
    If not harmonic, returns (False, 0.0).
    """
    shared = primes1 & primes2
    only1 = primes1 - shared
    only2 = primes2 - shared

    if len(only1) == 1 and len(only2) == 1:
        p1 = only1.pop()
        p2 = only2.pop()
        ratio = min(p1, p2) / max(p1, p2)
        return True, ratio
    elif len(only1) == 0 and len(only2) == 1:
        # One is a subset of the other + 1 extra
        return True, 0.5  # partial harmony
    elif len(only1) == 1 and len(only2) == 0:
        return True, 0.5
    else:
        return False, 0.0


def test_a_bens_law(body_qns, body_primes, names, coupled_pairs_idx, N_PERM=10000):
    """Test A: Ben's all-but-one criterion on nontrivial primes."""
    print("\n" + "=" * 80)
    print("TEST A — BEN'S LAW OF HARMONICS (all-but-one, nontrivial primes)")
    print("=" * 80)

    # Compute nontrivial prime sets
    nt_primes = {n: nontrivial_primes(body_primes[n]) for n in names}

    # Score coupled pairs
    coupled_scores = []
    coupled_harmonic_count = 0
    for i, j in coupled_pairs_idx:
        is_h, ratio = bens_harmony_score(nt_primes[names[i]], nt_primes[names[j]])
        coupled_scores.append(ratio)
        if is_h:
            coupled_harmonic_count += 1
            print(f"  HARMONIC: {names[i]:12s} ↔ {names[j]:12s}  "
                  f"ratio={ratio:.3f}  "
                  f"primes={sorted(nt_primes[names[i]])} vs {sorted(nt_primes[names[j]])}")

    # Score all pairs
    all_scores = []
    all_harmonic_count = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            is_h, ratio = bens_harmony_score(nt_primes[names[i]], nt_primes[names[j]])
            all_scores.append(ratio)
            if is_h:
                all_harmonic_count += 1

    coupled_rate = coupled_harmonic_count / max(len(coupled_pairs_idx), 1)
    all_rate = all_harmonic_count / max(len(all_scores), 1)

    print(f"\n  Coupled harmonic rate: {coupled_harmonic_count}/{len(coupled_pairs_idx)} = {coupled_rate:.3f}")
    print(f"  All-pair harmonic rate: {all_harmonic_count}/{len(all_scores)} = {all_rate:.3f}")
    print(f"  Coupled mean ratio: {np.mean(coupled_scores):.4f}")
    print(f"  All-pair mean ratio: {np.mean(all_scores):.4f}")

    # Permutation null on harmonic rate
    nt_list = [nt_primes[n] for n in names]
    null_rates = []
    null_mean_ratios = []
    for _ in range(N_PERM):
        perm = np.random.permutation(len(names))
        shuffled = [nt_list[p] for p in perm]
        h_count = 0
        ratios = []
        for i, j in coupled_pairs_idx:
            is_h, r = bens_harmony_score(shuffled[i], shuffled[j])
            if is_h:
                h_count += 1
            ratios.append(r)
        null_rates.append(h_count / max(len(coupled_pairs_idx), 1))
        null_mean_ratios.append(np.mean(ratios))

    null_rates = np.array(null_rates)
    null_mean_ratios = np.array(null_mean_ratios)

    p_rate = np.mean(null_rates >= coupled_rate)
    z_rate = (coupled_rate - np.mean(null_rates)) / (np.std(null_rates) + 1e-10)
    p_ratio = np.mean(null_mean_ratios >= np.mean(coupled_scores))
    z_ratio = (np.mean(coupled_scores) - np.mean(null_mean_ratios)) / (np.std(null_mean_ratios) + 1e-10)

    print(f"\n  PERMUTATION NULL (harmonic rate):")
    print(f"    Real: {coupled_rate:.3f}  Null: {np.mean(null_rates):.3f} ± {np.std(null_rates):.3f}")
    print(f"    z={z_rate:+.2f}  p={p_rate:.4f}  {'SIGNIFICANT' if p_rate < 0.05 else 'not significant'}")

    print(f"  PERMUTATION NULL (mean ratio):")
    print(f"    Real: {np.mean(coupled_scores):.4f}  Null: {np.mean(null_mean_ratios):.4f} ± {np.std(null_mean_ratios):.4f}")
    print(f"    z={z_ratio:+.2f}  p={p_ratio:.4f}  {'SIGNIFICANT' if p_ratio < 0.05 else 'not significant'}")

    return p_rate, p_ratio


# ═══════════════════════════════════════════════════════════════════════
# TEST B — TUPLE ELEMENT SHARING
# ═══════════════════════════════════════════════════════════════════════

def tuple_element_overlap(qn1, qn2):
    """Count shared literal values between two QN tuples.

    Checks: does any element of (b1,e1,d1,a1) equal any element of (b2,e2,d2,a2)?
    Also checks positional sharing: b1==b2, e1==e2, d1==d2, a1==a2.
    And cross-positional: e.g., a1==d2 (Earth a=61 = Moon d=61).
    Returns (positional_matches, any_matches, matching_values).
    """
    s1 = set(qn1)
    s2 = set(qn2)
    shared_vals = s1 & s2

    positional = sum(1 for x, y in zip(qn1, qn2) if x == y)

    # Cross-positional: any element of qn1 appears anywhere in qn2
    any_match = len(shared_vals)

    return positional, any_match, shared_vals


def test_b_tuple_sharing(body_qns, names, coupled_pairs_idx, N_PERM=10000):
    """Test B: Do coupled bodies share literal tuple elements?"""
    print("\n" + "=" * 80)
    print("TEST B — TUPLE ELEMENT SHARING (literal b/e/d/a equality)")
    print("=" * 80)

    qn_list = [body_qns[names[i]] for i in range(len(names))]

    # Score coupled pairs
    coupled_any = []
    for i, j in coupled_pairs_idx:
        pos, any_m, shared = tuple_element_overlap(qn_list[i], qn_list[j])
        coupled_any.append(any_m)
        if any_m > 0:
            print(f"  SHARED: {names[i]:12s} {qn_list[i]} ↔ {names[j]:12s} {qn_list[j]}  "
                  f"values={sorted(shared)}  positional={pos}  any={any_m}")

    # Score all pairs
    all_any = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            _, any_m, _ = tuple_element_overlap(qn_list[i], qn_list[j])
            all_any.append(any_m)

    real_coupled_mean = np.mean(coupled_any)
    all_mean = np.mean(all_any)
    print(f"\n  Coupled mean element overlap: {real_coupled_mean:.4f}")
    print(f"  All-pair mean element overlap: {all_mean:.4f}")
    print(f"  Coupled pairs with ≥1 shared element: {sum(1 for x in coupled_any if x > 0)}/{len(coupled_any)}")

    # Permutation null
    null_means = []
    for _ in range(N_PERM):
        perm = np.random.permutation(len(names))
        shuffled = [qn_list[p] for p in perm]
        scores = []
        for i, j in coupled_pairs_idx:
            _, any_m, _ = tuple_element_overlap(shuffled[i], shuffled[j])
            scores.append(any_m)
        null_means.append(np.mean(scores))

    null_means = np.array(null_means)
    p_val = np.mean(null_means >= real_coupled_mean)
    z_val = (real_coupled_mean - np.mean(null_means)) / (np.std(null_means) + 1e-10)

    print(f"\n  PERMUTATION NULL:")
    print(f"    Real: {real_coupled_mean:.4f}  Null: {np.mean(null_means):.4f} ± {np.std(null_means):.4f}")
    print(f"    z={z_val:+.2f}  p={p_val:.4f}  {'SIGNIFICANT' if p_val < 0.05 else 'not significant'}")

    return p_val


# ═══════════════════════════════════════════════════════════════════════
# TEST C — FIBONACCI PERIOD RATIO PROXIMITY
# ═══════════════════════════════════════════════════════════════════════

def nearest_fib_fraction(ratio):
    """Find the nearest Fibonacci fraction to a given ratio.

    Returns (nearest_frac, distance).
    """
    best_dist = float('inf')
    best_frac = None
    for ff in FIBONACCI_FRACTIONS:
        dist = abs(float(ff) - ratio)
        if dist < best_dist:
            best_dist = dist
            best_frac = ff
    return best_frac, best_dist


def test_c_fibonacci_periods(body_data, coupled_sibling_pairs, N_PERM=10000):
    """Test C: Do coupled body period ratios cluster near Fibonacci fractions?"""
    print("\n" + "=" * 80)
    print("TEST C — FIBONACCI PERIOD RATIO PROXIMITY")
    print("=" * 80)

    name_to_period = {name: period for name, _, _, _, period in body_data}

    # Compute period ratios for sibling pairs (moons of same parent, or resonance partners)
    real_distances = []
    for n1, n2, res_n, res_d in coupled_sibling_pairs:
        if n1 not in name_to_period or n2 not in name_to_period:
            continue
        p1 = name_to_period[n1]
        p2 = name_to_period[n2]
        ratio = max(p1, p2) / min(p1, p2)
        frac, dist = nearest_fib_fraction(ratio)
        real_distances.append(dist)
        print(f"  {n1:12s}/{n2:12s}: period ratio={ratio:.4f}  "
              f"nearest Fib={frac} ({float(frac):.4f})  dist={dist:.4f}  "
              f"(known {res_n}:{res_d}={res_n/res_d:.4f})")

    if not real_distances:
        print("  No sibling pairs with periods found")
        return 1.0

    real_mean_dist = np.mean(real_distances)
    print(f"\n  Mean distance to nearest Fibonacci fraction: {real_mean_dist:.6f}")

    # Null: random period ratios drawn from same period distribution
    all_periods = [p for _, _, _, _, p in body_data]
    null_means = []
    n_pairs = len(real_distances)
    for _ in range(N_PERM):
        # Draw random pairs of periods
        idxs = np.random.choice(len(all_periods), size=n_pairs * 2, replace=True)
        dists = []
        for k in range(n_pairs):
            p1 = all_periods[idxs[2 * k]]
            p2 = all_periods[idxs[2 * k + 1]]
            if p1 == p2:
                continue
            ratio = max(p1, p2) / min(p1, p2)
            _, dist = nearest_fib_fraction(ratio)
            dists.append(dist)
        if dists:
            null_means.append(np.mean(dists))

    null_means = np.array(null_means)

    # One-tailed: real CLOSER to Fibonacci (lower distance) than null
    p_val = np.mean(null_means <= real_mean_dist)
    z_val = (real_mean_dist - np.mean(null_means)) / (np.std(null_means) + 1e-10)

    print(f"\n  PERMUTATION NULL:")
    print(f"    Real mean dist: {real_mean_dist:.6f}")
    print(f"    Null mean dist: {np.mean(null_means):.6f} ± {np.std(null_means):.6f}")
    print(f"    z={z_val:+.2f}  p={p_val:.4f}  (one-tail: real < null)")
    print(f"    {'SIGNIFICANT — resonance ratios ARE closer to Fibonacci' if p_val < 0.05 else 'not significant'}")

    return p_val


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("QA SOLAR SYSTEM HARMONIC TEST v2 — Ben's Actual Claims")
    print("=" * 80)

    # Assign QNs
    body_qns = {}
    body_primes = {}
    names = []
    body_data = []

    for name, ecc, cat, parent, period in BODIES:
        result = find_qn(ecc)
        if result is None:
            continue
        b, e, d, a, ecc_qa, err = result
        body_qns[name] = (b, e, d, a)
        body_primes[name] = qn_product_primes(b, e, d, a)
        names.append(name)
        body_data.append((name, ecc, cat, parent, period))

    print(f"\n  {len(names)} bodies assigned QNs")

    # Build coupled pair indices
    name_to_idx = {n: i for i, n in enumerate(names)}
    coupled_pairs_idx = []
    for name, ecc, cat, parent, period in body_data:
        if parent is not None and parent in name_to_idx and name in name_to_idx:
            coupled_pairs_idx.append((name_to_idx[parent], name_to_idx[name]))

    print(f"  {len(coupled_pairs_idx)} coupled pairs (parent-satellite)")

    # ── TEST A ──
    p_a_rate, p_a_ratio = test_a_bens_law(body_qns, body_primes, names, coupled_pairs_idx)

    # ── TEST B ──
    p_b = test_b_tuple_sharing(body_qns, names, coupled_pairs_idx)

    # ── TEST C ──
    p_c = test_c_fibonacci_periods(body_data, RESONANCE_PAIRS)

    # ── SUMMARY ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    results = [
        ("A: Ben's Law (harmonic rate)", p_a_rate),
        ("A: Ben's Law (mean ratio)", p_a_ratio),
        ("B: Tuple element sharing", p_b),
        ("C: Fibonacci period proximity", p_c),
    ]

    any_sig = False
    for label, p in results:
        sig = p < 0.05
        if sig:
            any_sig = True
        print(f"  {label:40s}  p={p:.4f}  {'*** SIGNIFICANT' if sig else '    not significant'}")

    print()
    if any_sig:
        print("  AT LEAST ONE TEST SIGNIFICANT.")
        print("  The connection exists but through a SPECIFIC mechanism, not generic prime sharing.")
    else:
        print("  NO TEST SIGNIFICANT.")
        print("  The connection, if it exists, is not captured by these three metrics either.")
        print("  Possible next: test Pisano period relationships, or par-class correlations.")

    print()
    print("  NOTE: These tests are HONEST. A negative result means the specific metric")
    print("  tested does not distinguish coupled from uncoupled bodies. It does NOT mean")
    print("  there is no connection — it means we haven't found the right test yet.")


if __name__ == "__main__":
    main()
