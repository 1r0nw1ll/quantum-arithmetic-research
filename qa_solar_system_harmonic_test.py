#!/usr/bin/env python3
"""
qa_solar_system_harmonic_test.py — Quantitative test of QA harmonic structure
in the solar system.

QUESTION: Do gravitationally coupled solar system bodies share more QN prime
factors than random pairs with the same eccentricity distribution?

TEST DESIGN:
1. Find best QN for each of 47 real solar system bodies
2. Compute prime factor sharing (Jaccard similarity) for all pairs
3. Compare COUPLED pairs (parent-satellite, resonance partners) vs ALL pairs
4. NULL MODEL: 10000 random permutations of QN assignments
   (shuffle which QN goes to which body, preserving the eccentricity→QN mapping)
5. If real coupled-pair sharing > 95th percentile of null → Tier 3

ADDITIONAL TEST — PREDICTION:
Withhold 5 bodies. From the remaining network, predict which QN the withheld
body should have (most prime-sharing with its parent/neighbors). Score.

Author: Will Dale (design), Claude (code)
"""

QA_COMPLIANCE = "observer=prime_sharing_test, state_alphabet=primitive_QN_tuple"

import math
import itertools
from collections import defaultdict
from math import gcd

import numpy as np

np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════
# BODY CATALOGUE (from qa_solar_system_deep.py)
# ═══════════════════════════════════════════════════════════════════════

BODIES = [
    ("Mercury",    0.20563,  "planet",   None),
    ("Venus",      0.00677,  "planet",   None),
    ("Earth",      0.01671,  "planet",   None),
    ("Mars",       0.09339,  "planet",   None),
    ("Jupiter",    0.04839,  "planet",   None),
    ("Saturn",     0.05415,  "planet",   None),
    ("Uranus",     0.04717,  "planet",   None),
    ("Neptune",    0.00859,  "planet",   None),
    ("Moon",       0.0549,   "moon",     "Earth"),
    ("Phobos",     0.0151,   "moon",     "Mars"),
    ("Deimos",     0.0002,   "moon",     "Mars"),
    ("Io",         0.0041,   "moon",     "Jupiter"),
    ("Europa",     0.0094,   "moon",     "Jupiter"),
    ("Ganymede",   0.0011,   "moon",     "Jupiter"),
    ("Callisto",   0.0074,   "moon",     "Jupiter"),
    ("Amalthea",   0.0032,   "moon",     "Jupiter"),
    ("Mimas",      0.0196,   "moon",     "Saturn"),
    ("Enceladus",  0.0047,   "moon",     "Saturn"),
    ("Tethys",     0.0001,   "moon",     "Saturn"),
    ("Dione",      0.0022,   "moon",     "Saturn"),
    ("Rhea",       0.0013,   "moon",     "Saturn"),
    ("Titan",      0.0288,   "moon",     "Saturn"),
    ("Iapetus",    0.0276,   "moon",     "Saturn"),
    ("Miranda",    0.0013,   "moon",     "Uranus"),
    ("Ariel",      0.0012,   "moon",     "Uranus"),
    ("Umbriel",    0.0039,   "moon",     "Uranus"),
    ("Titania",    0.0011,   "moon",     "Uranus"),
    ("Oberon",     0.0014,   "moon",     "Uranus"),
    ("Triton",     0.000016, "moon",     "Neptune"),
    ("Nereid",     0.7507,   "moon",     "Neptune"),
    ("Charon",     0.0002,   "moon",     "Pluto"),
    ("Pluto",      0.2488,   "dwarf",    None),
    ("Ceres",      0.0758,   "dwarf",    None),
    ("Eris",       0.4407,   "dwarf",    None),
    ("Haumea",     0.1912,   "dwarf",    None),
    ("Makemake",   0.1559,   "dwarf",    None),
    ("Sedna",      0.8496,   "dwarf",    None),
    ("Halley",     0.96714,  "comet",    None),
    ("Hale-Bopp",  0.99507,  "comet",    None),
    ("Encke",      0.8483,   "comet",    None),
    ("Hyakutake",  0.99990,  "comet",    None),
    ("67P/C-G",    0.6405,   "comet",    None),
    ("Vesta",      0.0887,   "asteroid", None),
    ("Pallas",     0.2313,   "asteroid", None),
    ("Eros",       0.2229,   "asteroid", None),
    ("Bennu",      0.2037,   "asteroid", None),
    ("Itokawa",    0.2802,   "asteroid", None),
]

# Known gravitational couplings (parent-satellite + resonance partners)
COUPLED_PAIRS = []
# Parent-satellite pairs
for name, ecc, cat, parent in BODIES:
    if parent is not None:
        COUPLED_PAIRS.append((parent, name))

# Known mean-motion resonances
RESONANCE_PAIRS = [
    ("Io", "Europa"),           # 2:1
    ("Europa", "Ganymede"),     # 2:1
    ("Mimas", "Tethys"),        # 2:1
    ("Enceladus", "Dione"),     # 2:1
    ("Titan", "Iapetus"),       # 5:1 approx
    ("Pluto", "Neptune"),       # 3:2
    ("Jupiter", "Saturn"),      # 5:2 (Great Inequality)
]


# ═══════════════════════════════════════════════════════════════════════
# QN FINDER
# ═══════════════════════════════════════════════════════════════════════

def find_qn(ecc_target, max_d=500):
    """Find best primitive QN (b,e,d,a) where ecc = e/d ≈ ecc_target."""
    if ecc_target <= 0:
        return (1, 0, 1, 1, 0.0, ecc_target)  # degenerate
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
    """Return set of prime factors of n."""
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


def qn_primes(b, e, d, a):
    """Return set of all primes appearing in QN product b*e*d*a."""
    product = b * e * d * a
    return prime_factors(product)


def jaccard(set1, set2):
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


# ═══════════════════════════════════════════════════════════════════════
# MAIN TEST
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("QA SOLAR SYSTEM HARMONIC TEST — Phase 2 Geodesy Roadmap")
    print("=" * 80)

    # ── Step 1: Find QNs for all bodies ──
    print("\n── STEP 1: QN Assignment ──")
    body_qns = {}
    body_primes = {}
    names = []

    for name, ecc, cat, parent in BODIES:
        result = find_qn(ecc)
        if result is None:
            continue
        b, e, d, a, ecc_qa, err = result
        body_qns[name] = (b, e, d, a)
        body_primes[name] = qn_primes(b, e, d, a)
        names.append(name)

    print(f"  {len(names)} bodies assigned QNs")

    # Print a few examples
    for name in ["Mercury", "Earth", "Moon", "Jupiter", "Io", "Halley"]:
        if name in body_qns:
            b, e, d, a = body_qns[name]
            ecc_target = [ecc for n, ecc, c, p in BODIES if n == name][0]
            print(f"  {name:12s}: ({b},{e},{d},{a}) ecc={e/d:.6f} (target {ecc_target:.6f}) "
                  f"primes={sorted(body_primes[name])}")

    # ── Step 2: Compute prime sharing for coupled vs all pairs ──
    print("\n── STEP 2: Prime Sharing — Coupled vs All Pairs ──")

    # All pairs
    all_jaccards = []
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            j = jaccard(body_primes[n1], body_primes[n2])
            all_jaccards.append(j)

    # Coupled pairs (parent-satellite)
    coupled_jaccards = []
    coupled_labels = []
    for parent, child in COUPLED_PAIRS:
        if parent in body_primes and child in body_primes:
            j = jaccard(body_primes[parent], body_primes[child])
            coupled_jaccards.append(j)
            coupled_labels.append(f"{parent}-{child}")

    # Resonance pairs
    resonance_jaccards = []
    resonance_labels = []
    for n1, n2 in RESONANCE_PAIRS:
        if n1 in body_primes and n2 in body_primes:
            j = jaccard(body_primes[n1], body_primes[n2])
            resonance_jaccards.append(j)
            resonance_labels.append(f"{n1}-{n2}")

    mean_all = np.mean(all_jaccards)
    mean_coupled = np.mean(coupled_jaccards) if coupled_jaccards else 0
    mean_resonance = np.mean(resonance_jaccards) if resonance_jaccards else 0

    print(f"  All pairs:       mean Jaccard = {mean_all:.4f}  (n={len(all_jaccards)})")
    print(f"  Coupled pairs:   mean Jaccard = {mean_coupled:.4f}  (n={len(coupled_jaccards)})")
    print(f"  Resonance pairs: mean Jaccard = {mean_resonance:.4f}  (n={len(resonance_jaccards)})")

    # Show top coupled pairs
    print("\n  Top coupled pairs by prime sharing:")
    sorted_coupled = sorted(zip(coupled_jaccards, coupled_labels), reverse=True)
    for j, label in sorted_coupled[:10]:
        p, c = label.split("-")
        print(f"    {label:25s} J={j:.3f}  primes_shared={sorted(body_primes[p] & body_primes[c])}")

    print("\n  Resonance pairs:")
    for j, label in zip(resonance_jaccards, resonance_labels):
        n1, n2 = label.split("-")
        print(f"    {label:25s} J={j:.3f}  primes_shared={sorted(body_primes[n1] & body_primes[n2])}")

    # ── Step 3: Null model — permutation test ──
    print("\n── STEP 3: Permutation Null Model (10000 shuffles) ──")

    N_PERM = 10000
    qn_list = [body_qns[n] for n in names]
    prime_list = [body_primes[n] for n in names]

    # Index-based coupled pairs
    name_to_idx = {n: i for i, n in enumerate(names)}
    coupled_idx = []
    for parent, child in COUPLED_PAIRS:
        if parent in name_to_idx and child in name_to_idx:
            coupled_idx.append((name_to_idx[parent], name_to_idx[child]))

    resonance_idx = []
    for n1, n2 in RESONANCE_PAIRS:
        if n1 in name_to_idx and n2 in name_to_idx:
            resonance_idx.append((name_to_idx[n1], name_to_idx[n2]))

    def mean_jaccard_for_pairs(prime_assignment, pair_indices):
        if not pair_indices:
            return 0.0
        total = 0.0
        for i, j in pair_indices:
            total += jaccard(prime_assignment[i], prime_assignment[j])
        return total / len(pair_indices)

    real_coupled_mean = mean_jaccard_for_pairs(prime_list, coupled_idx)
    real_resonance_mean = mean_jaccard_for_pairs(prime_list, resonance_idx)

    null_coupled = []
    null_resonance = []

    for _ in range(N_PERM):
        perm = np.random.permutation(len(names))
        shuffled_primes = [prime_list[p] for p in perm]
        null_coupled.append(mean_jaccard_for_pairs(shuffled_primes, coupled_idx))
        null_resonance.append(mean_jaccard_for_pairs(shuffled_primes, resonance_idx))

    null_coupled = np.array(null_coupled)
    null_resonance = np.array(null_resonance)

    # p-values (one-tailed: real > null)
    p_coupled = np.mean(null_coupled >= real_coupled_mean)
    p_resonance = np.mean(null_resonance >= real_resonance_mean)

    # z-scores
    z_coupled = (real_coupled_mean - np.mean(null_coupled)) / (np.std(null_coupled) + 1e-10)
    z_resonance = (real_resonance_mean - np.mean(null_resonance)) / (np.std(null_resonance) + 1e-10)

    print(f"  COUPLED PAIRS:")
    print(f"    Real mean Jaccard:  {real_coupled_mean:.4f}")
    print(f"    Null mean:          {np.mean(null_coupled):.4f} ± {np.std(null_coupled):.4f}")
    print(f"    z-score:            {z_coupled:+.2f}")
    print(f"    p-value (one-tail): {p_coupled:.4f}")
    print(f"    Verdict:            {'SIGNIFICANT (p<0.05)' if p_coupled < 0.05 else 'NOT SIGNIFICANT'}")

    print(f"\n  RESONANCE PAIRS:")
    print(f"    Real mean Jaccard:  {real_resonance_mean:.4f}")
    print(f"    Null mean:          {np.mean(null_resonance):.4f} ± {np.std(null_resonance):.4f}")
    print(f"    z-score:            {z_resonance:+.2f}")
    print(f"    p-value (one-tail): {p_resonance:.4f}")
    print(f"    Verdict:            {'SIGNIFICANT (p<0.05)' if p_resonance < 0.05 else 'NOT SIGNIFICANT'}")

    # ── Step 4: Withheld body prediction test ──
    print("\n── STEP 4: Withheld Body Prediction Test ──")
    print("  (Withhold 5 moons, predict their QN from parent's prime network)")

    withheld = ["Europa", "Titan", "Oberon", "Phobos", "Triton"]
    print(f"  Withheld bodies: {withheld}")

    correct = 0
    total = 0
    for wname in withheld:
        wdata = [(n, e, c, p) for n, e, c, p in BODIES if n == wname]
        if not wdata:
            continue
        _, w_ecc, _, w_parent = wdata[0]
        if w_parent is None or w_parent not in body_primes:
            continue

        parent_primes = body_primes[w_parent]
        real_qn = body_qns[wname]
        real_primes = body_primes[wname]

        # Generate candidate QNs near the true eccentricity
        # (within ±20% of true ecc, to simulate "knowing approximate orbit")
        candidates = []
        for d_val in range(2, 200):
            for e_val in range(1, d_val):
                b_val = d_val - e_val
                if gcd(b_val, e_val) != 1:
                    continue
                ecc = e_val / d_val
                if abs(ecc - w_ecc) / max(w_ecc, 1e-10) < 0.20:
                    a_val = b_val + 2 * e_val
                    cand_primes = qn_primes(b_val, e_val, d_val, a_val)
                    j_parent = jaccard(cand_primes, parent_primes)
                    candidates.append((j_parent, b_val, e_val, d_val, a_val, cand_primes))

        if not candidates:
            print(f"  {wname}: no candidates found")
            continue

        candidates.sort(reverse=True)  # highest Jaccard with parent first
        best_cand = candidates[0]
        predicted_qn = (best_cand[1], best_cand[2], best_cand[3], best_cand[4])

        match = predicted_qn == real_qn
        if match:
            correct += 1
        total += 1

        print(f"  {wname:12s}: real=({real_qn[0]},{real_qn[1]},{real_qn[2]},{real_qn[3]}) "
              f"predicted=({predicted_qn[0]},{predicted_qn[1]},{predicted_qn[2]},{predicted_qn[3]}) "
              f"{'MATCH' if match else 'MISS'} "
              f"(J_parent={best_cand[0]:.3f}, n_candidates={len(candidates)})")

    print(f"\n  Prediction accuracy: {correct}/{total} = {correct/max(total,1)*100:.0f}%")
    print(f"  Chance level: ~1/n_candidates (varies per body)")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Coupled pairs test:   z={z_coupled:+.2f}, p={p_coupled:.4f} "
          f"{'→ Tier 3 CONFIRMED' if p_coupled < 0.05 else '→ Tier 2 (not significant)'}")
    print(f"  Resonance pairs test: z={z_resonance:+.2f}, p={p_resonance:.4f} "
          f"{'→ Tier 3 CONFIRMED' if p_resonance < 0.05 else '→ Tier 2 (not significant)'}")
    print(f"  Prediction test:      {correct}/{total} matches")
    print()

    if p_coupled < 0.05 or p_resonance < 0.05:
        print("  AT LEAST ONE TEST SIGNIFICANT — QA harmonic structure is")
        print("  non-random in gravitationally coupled systems.")
    else:
        print("  NEITHER TEST SIGNIFICANT — QA harmonic structure does NOT")
        print("  exceed chance-level prime sharing for coupled bodies.")
        print("  Solar system QN work remains Tier 2 (structural pattern, not predictive).")

    print()
    print("  HONEST NOTE: The permutation test shuffles QN assignments while")
    print("  preserving the eccentricity→QN mapping. This tests whether the")
    print("  SPECIFIC assignment of bodies to eccentricities creates harmonic")
    print("  structure, not whether QA itself is special. A positive result")
    print("  means coupled orbits share numerical structure; a negative result")
    print("  means prime sharing is explained by the eccentricity distribution alone.")


if __name__ == "__main__":
    main()
