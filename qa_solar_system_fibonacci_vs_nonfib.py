#!/usr/bin/env python3
"""
qa_solar_system_fibonacci_vs_nonfib.py — The discriminating test.

QUESTION: Among all low-order integer ratios, do solar system resonances
preferentially select FIBONACCI ratios over non-Fibonacci ones?

DESIGN:
1. Catalogue ALL known mean-motion resonances in the solar system
   (not just the 8 from the previous test — include Kirkwood gaps,
   Hilda group, Trojans, exoplanet resonances for comparison)
2. Classify each as Fibonacci-ratio or non-Fibonacci-ratio
3. Compare the proportion to what you'd expect if resonances were
   drawn uniformly from all low-order ratios

FIBONACCI NUMBERS: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89...
A ratio p:q is "Fibonacci" if BOTH p and q are Fibonacci numbers.

STANDARD THEORY PREDICTION: resonances should be distributed by ORDER
(|p-q|), with lower orders more common. No preference for Fibonacci
within each order class.

QA PREDICTION: Fibonacci ratios should be overrepresented because the
T-operator is the Fibonacci shift and orbital periods lock to its
eigenstructure (convergents of φ).

Author: Will Dale (question), Claude (code)
"""

QA_COMPLIANCE = "observer=fibonacci_discrimination_test, state_alphabet=resonance_catalogue"

import numpy as np
from fractions import Fraction
from math import gcd

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# COMPREHENSIVE RESONANCE CATALOGUE
# ═══════════════════════════════════════════════════════════════════════
# Sources: Murray & Dermott "Solar System Dynamics" (1999),
#          NASA JPL small body database, Peale (1976), Goldreich (1965)
#
# Each entry: (body1, body2, p, q, type, confidence)
# type: "orbital" (moon-moon or planet-planet), "kirkwood" (asteroid gap),
#       "capture" (captured into resonance), "secular" (long-term)
# confidence: "confirmed" or "approximate"

RESONANCES = [
    # ── Major satellite resonances ──────────────────────────────────
    ("Io",         "Europa",      2, 1, "orbital",  "confirmed"),
    ("Europa",     "Ganymede",    2, 1, "orbital",  "confirmed"),
    ("Mimas",      "Tethys",      2, 1, "orbital",  "confirmed"),
    ("Enceladus",  "Dione",       2, 1, "orbital",  "confirmed"),
    ("Dione",      "Rhea",        5, 3, "orbital",  "approximate"),  # near 5:3
    ("Titan",      "Hyperion",    4, 3, "orbital",  "confirmed"),
    ("Ariel",      "Umbriel",     5, 3, "orbital",  "approximate"),  # near, not locked

    # ── Planet-planet resonances ────────────────────────────────────
    ("Pluto",      "Neptune",     3, 2, "orbital",  "confirmed"),
    ("Jupiter",    "Saturn",      5, 2, "orbital",  "approximate"),  # Great Inequality

    # ── Kirkwood gaps (asteroid belt, with Jupiter) ─────────────────
    ("Kirkwood",   "Jupiter",     4, 1, "kirkwood", "confirmed"),
    ("Kirkwood",   "Jupiter",     3, 1, "kirkwood", "confirmed"),
    ("Kirkwood",   "Jupiter",     5, 2, "kirkwood", "confirmed"),
    ("Kirkwood",   "Jupiter",     7, 3, "kirkwood", "confirmed"),
    ("Kirkwood",   "Jupiter",     2, 1, "kirkwood", "confirmed"),

    # ── Asteroid group resonances (with Jupiter) ────────────────────
    ("Hildas",     "Jupiter",     3, 2, "capture",  "confirmed"),
    ("Trojans",    "Jupiter",     1, 1, "capture",  "confirmed"),
    ("Thule",      "Jupiter",     4, 3, "capture",  "confirmed"),
    ("Zhongguo",   "Jupiter",     2, 1, "capture",  "confirmed"),
    ("Griqua",     "Jupiter",     2, 1, "capture",  "confirmed"),

    # ── Trans-Neptunian resonances (with Neptune) ───────────────────
    ("Plutinos",   "Neptune",     3, 2, "capture",  "confirmed"),
    ("Twotinos",   "Neptune",     2, 1, "capture",  "confirmed"),
    ("5:3 TNOs",   "Neptune",     5, 3, "capture",  "confirmed"),
    ("7:4 TNOs",   "Neptune",     7, 4, "capture",  "confirmed"),
    ("5:2 TNOs",   "Neptune",     5, 2, "capture",  "confirmed"),
    ("4:3 TNOs",   "Neptune",     4, 3, "capture",  "confirmed"),
    ("3:1 TNOs",   "Neptune",     3, 1, "capture",  "confirmed"),
    ("9:4 TNOs",   "Neptune",     9, 4, "capture",  "approximate"),
    ("7:2 TNOs",   "Neptune",     7, 2, "capture",  "approximate"),
    ("5:1 TNOs",   "Neptune",     5, 1, "capture",  "approximate"),
]

# Fibonacci numbers up to 89
FIBS = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}

def is_fibonacci_ratio(p, q):
    """Both p and q must be Fibonacci numbers."""
    return p in FIBS and q in FIBS

def resonance_order(p, q):
    """Order = |p - q|. Lower order = stronger resonance."""
    return abs(p - q)


def main():
    print("=" * 80)
    print("FIBONACCI vs NON-FIBONACCI RESONANCE DISCRIMINATION TEST")
    print("=" * 80)

    # ── Classify all resonances ──
    print("\n── ALL KNOWN SOLAR SYSTEM RESONANCES ──\n")

    fib_count = 0
    nonfib_count = 0
    total = 0

    by_order = {}  # order -> (fib_count, nonfib_count, examples)

    unique_ratios = set()

    for b1, b2, p, q, rtype, conf in RESONANCES:
        # Reduce to lowest terms
        g = gcd(p, q)
        pr, qr = p // g, q // g
        ratio_key = (pr, qr)

        is_fib = is_fibonacci_ratio(pr, qr)
        order = resonance_order(pr, qr)
        tag = "FIB" if is_fib else "   "

        if ratio_key not in unique_ratios:
            unique_ratios.add(ratio_key)

        total += 1
        if is_fib:
            fib_count += 1
        else:
            nonfib_count += 1

        if order not in by_order:
            by_order[order] = {"fib": 0, "nonfib": 0, "fib_ratios": set(), "nonfib_ratios": set()}
        if is_fib:
            by_order[order]["fib"] += 1
            by_order[order]["fib_ratios"].add(f"{pr}:{qr}")
        else:
            by_order[order]["nonfib"] += 1
            by_order[order]["nonfib_ratios"].add(f"{pr}:{qr}")

        print(f"  [{tag}] {pr}:{qr}  order={order}  {b1:12s}↔{b2:12s}  ({rtype}, {conf})")

    # ── Summary by order ──
    print(f"\n── SUMMARY ──\n")
    print(f"  Total resonances catalogued: {total}")
    print(f"  Fibonacci ratios: {fib_count} ({fib_count/total*100:.1f}%)")
    print(f"  Non-Fibonacci:    {nonfib_count} ({nonfib_count/total*100:.1f}%)")

    print(f"\n  Unique ratio types: {len(unique_ratios)}")
    fib_unique = sum(1 for p, q in unique_ratios if is_fibonacci_ratio(p, q))
    nonfib_unique = len(unique_ratios) - fib_unique
    print(f"  Unique Fibonacci:     {fib_unique}")
    print(f"  Unique non-Fibonacci: {nonfib_unique}")

    print(f"\n── BREAKDOWN BY ORDER ──\n")
    print(f"  {'Order':>5s}  {'Fib':>4s}  {'Non':>4s}  {'%Fib':>6s}  Fib ratios              Non-Fib ratios")
    print(f"  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*24}  {'─'*24}")
    for order in sorted(by_order.keys()):
        d = by_order[order]
        ftot = d["fib"] + d["nonfib"]
        pct = d["fib"] / ftot * 100 if ftot > 0 else 0
        fib_r = ", ".join(sorted(d["fib_ratios"]))
        nonfib_r = ", ".join(sorted(d["nonfib_ratios"]))
        print(f"  {order:5d}  {d['fib']:4d}  {d['nonfib']:4d}  {pct:5.1f}%  {fib_r:24s}  {nonfib_r:24s}")

    # ── Expected Fibonacci fraction under null ──
    print(f"\n── NULL MODEL: EXPECTED FIBONACCI FRACTION ──\n")

    # Generate all reduced ratios p:q with p>q, p<=10
    # Count what fraction are Fibonacci
    all_ratios_by_order = {}
    for p in range(1, 11):
        for q in range(1, p):
            if gcd(p, q) != 1:
                continue
            order = p - q
            if order not in all_ratios_by_order:
                all_ratios_by_order[order] = {"fib": 0, "nonfib": 0, "fib_r": [], "nonfib_r": []}
            if is_fibonacci_ratio(p, q):
                all_ratios_by_order[order]["fib"] += 1
                all_ratios_by_order[order]["fib_r"].append(f"{p}:{q}")
            else:
                all_ratios_by_order[order]["nonfib"] += 1
                all_ratios_by_order[order]["nonfib_r"].append(f"{p}:{q}")

    print(f"  All coprime ratios p:q with p≤10, p>q:")
    print(f"  {'Order':>5s}  {'Fib':>4s}  {'Non':>4s}  {'%Fib':>6s}  Fib ratios              Non-Fib ratios")
    print(f"  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*24}  {'─'*24}")

    total_possible_fib = 0
    total_possible_nonfib = 0
    for order in sorted(all_ratios_by_order.keys()):
        d = all_ratios_by_order[order]
        ftot = d["fib"] + d["nonfib"]
        pct = d["fib"] / ftot * 100 if ftot > 0 else 0
        total_possible_fib += d["fib"]
        total_possible_nonfib += d["nonfib"]
        fib_r = ", ".join(d["fib_r"][:5])
        nonfib_r = ", ".join(d["nonfib_r"][:5])
        print(f"  {order:5d}  {d['fib']:4d}  {d['nonfib']:4d}  {pct:5.1f}%  {fib_r:24s}  {nonfib_r:24s}")

    expected_fib_rate = total_possible_fib / (total_possible_fib + total_possible_nonfib)
    print(f"\n  Overall expected Fibonacci rate (uniform over all ratios p≤10): "
          f"{total_possible_fib}/{total_possible_fib+total_possible_nonfib} = {expected_fib_rate:.3f}")

    # ── Statistical test: binomial ──
    print(f"\n── STATISTICAL TEST ──\n")

    # Use unique ratios only (avoid counting 2:1 five times)
    print(f"  Using UNIQUE ratio types (not counting duplicates):")
    print(f"  Observed: {fib_unique}/{fib_unique+nonfib_unique} = {fib_unique/(fib_unique+nonfib_unique):.3f} Fibonacci")
    print(f"  Expected: {expected_fib_rate:.3f} under uniform null")

    # Binomial test
    from scipy import stats
    n_trials = fib_unique + nonfib_unique
    n_success = fib_unique
    p_null = expected_fib_rate

    # One-tailed: observed Fibonacci rate > expected
    p_binom = 1 - stats.binom.cdf(n_success - 1, n_trials, p_null)
    print(f"\n  Binomial test (one-tailed, H1: Fib rate > {p_null:.3f}):")
    print(f"    n={n_trials}, k={n_success}, p_null={p_null:.3f}")
    print(f"    p-value = {p_binom:.4f}")
    print(f"    {'SIGNIFICANT — Fibonacci ratios overrepresented' if p_binom < 0.05 else 'NOT SIGNIFICANT'}")

    # Also test using all instances (with duplicates)
    p_binom_all = 1 - stats.binom.cdf(fib_count - 1, total, p_null)
    print(f"\n  Binomial test (all instances including duplicates):")
    print(f"    n={total}, k={fib_count}, p_null={p_null:.3f}")
    print(f"    p-value = {p_binom_all:.4f}")
    print(f"    {'SIGNIFICANT' if p_binom_all < 0.05 else 'NOT SIGNIFICANT'}")

    # ── ORDER-WEIGHTED test ──
    # Standard theory says lower-order resonances are preferred.
    # Within each order, is the Fibonacci fraction higher than expected?
    print(f"\n── ORDER-STRATIFIED TEST ──\n")
    print(f"  Within each resonance order, is Fibonacci rate > expected?\n")

    order_tests = []
    for order in sorted(by_order.keys()):
        if order not in all_ratios_by_order:
            continue
        obs = by_order[order]
        exp = all_ratios_by_order[order]

        obs_n = obs["fib"] + obs["nonfib"]
        obs_fib = obs["fib"]
        exp_rate = exp["fib"] / (exp["fib"] + exp["nonfib"]) if (exp["fib"] + exp["nonfib"]) > 0 else 0

        if obs_n > 0 and exp_rate > 0 and exp_rate < 1:
            p_ord = 1 - stats.binom.cdf(obs_fib - 1, obs_n, exp_rate)
        else:
            p_ord = 1.0

        order_tests.append((order, obs_fib, obs_n, exp_rate, p_ord))
        print(f"  Order {order}: {obs_fib}/{obs_n} Fibonacci (expected rate {exp_rate:.2f})  p={p_ord:.4f}")

    # Fisher's method to combine
    chi2_stat = -2 * sum(np.log(max(p, 1e-10)) for _, _, _, _, p in order_tests)
    df = 2 * len(order_tests)
    p_fisher = 1 - stats.chi2.cdf(chi2_stat, df)
    print(f"\n  Fisher's combined p-value: χ²={chi2_stat:.2f}, df={df}, p={p_fisher:.4f}")
    print(f"  {'SIGNIFICANT — Fibonacci preference across orders' if p_fisher < 0.05 else 'NOT SIGNIFICANT across orders'}")

    # ── VERDICT ──
    print(f"\n{'='*80}")
    print(f"VERDICT")
    print(f"{'='*80}")
    print()
    print(f"  Observed Fibonacci rate: {fib_unique}/{fib_unique+nonfib_unique} unique ratios "
          f"= {fib_unique/(fib_unique+nonfib_unique)*100:.0f}%")
    print(f"  Expected under uniform:  {expected_fib_rate*100:.0f}%")
    print(f"  Binomial p-value:        {p_binom:.4f}")
    print(f"  Fisher combined p-value: {p_fisher:.4f}")
    print()

    if p_binom < 0.05:
        print(f"  FIBONACCI RATIOS ARE OVERREPRESENTED in solar system resonances.")
        print(f"  This is consistent with QA's prediction that the T-operator")
        print(f"  (Fibonacci shift) biases orbital locking toward Fibonacci fractions.")
        print(f"  Tier 2→3 candidate for the solar system domain.")
    else:
        print(f"  Fibonacci ratios are NOT significantly overrepresented.")
        print(f"  Standard perturbation theory (order-based selection) is sufficient")
        print(f"  to explain the observed resonance distribution.")
        print(f"  Solar system QN work remains Tier 2.")

    print()
    print(f"  CONTEXT: Standard celestial mechanics explains resonance FORMATION")
    print(f"  via tidal dissipation + perturbation theory. The strength of a p:q")
    print(f"  resonance scales as e^|p-q| (eccentricity to the order power).")
    print(f"  This naturally favors low-order ratios. The question here is whether")
    print(f"  Fibonacci ratios are preferred WITHIN each order class — which would")
    print(f"  not be explained by standard theory.")
    print()
    print(f"  THREE-BODY PROBLEM CONNECTION: The three-body problem means we")
    print(f"  cannot analytically predict WHICH resonances will be populated.")
    print(f"  Perturbation theory gives resonance STRENGTHS, but capture depends")
    print(f"  on formation history. If QA's T-operator provides a dynamical")
    print(f"  preference for Fibonacci ratios, it would be a structural explanation")
    print(f"  for a pattern that standard theory treats as contingent.")


if __name__ == "__main__":
    main()
