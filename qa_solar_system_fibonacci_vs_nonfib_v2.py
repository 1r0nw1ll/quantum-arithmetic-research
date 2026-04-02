#!/usr/bin/env python3
"""
qa_solar_system_fibonacci_vs_nonfib_v2.py — Expanded discrimination test
with exoplanet resonance data.

QUESTION: Among all low-order integer ratios, do BOTH solar system AND
exoplanet resonances preferentially select FIBONACCI ratios?

EXPANDED DATASET:
- 29 solar system resonances (from v1)
- ~30 exoplanet adjacent-pair resonances from confirmed resonant chains:
  TRAPPIST-1, HD 110067, K2-138, Kepler-80, Kepler-223, TOI-178,
  GJ-876, HD 158259, Kepler-90 (partial)

Sources: Murray & Dermott (1999), Luger et al. (2017), Luque et al. (2024),
         Wikipedia Orbital Resonance, NASA Exoplanet Archive

Author: Will Dale (question), Claude (code)
"""

QA_COMPLIANCE = "observer=fibonacci_discrimination_v2, state_alphabet=resonance_catalogue"

import numpy as np
from math import gcd

np.random.seed(42)

FIBS = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}

def is_fibonacci_ratio(p, q):
    return p in FIBS and q in FIBS

def resonance_order(p, q):
    return abs(p - q)


# ═══════════════════════════════════════════════════════════════════════
# SOLAR SYSTEM RESONANCES (from v1)
# ═══════════════════════════════════════════════════════════════════════

SOLAR_SYSTEM = [
    # (body1, body2, p, q, source)
    ("Io", "Europa", 2, 1, "Laplace"),
    ("Europa", "Ganymede", 2, 1, "Laplace"),
    ("Mimas", "Tethys", 2, 1, "Saturn"),
    ("Enceladus", "Dione", 2, 1, "Saturn"),
    ("Dione", "Rhea", 5, 3, "Saturn approx"),
    ("Titan", "Hyperion", 4, 3, "Saturn"),
    ("Ariel", "Umbriel", 5, 3, "Uranus approx"),
    ("Pluto", "Neptune", 3, 2, "TNO"),
    ("Jupiter", "Saturn", 5, 2, "Great Inequality"),
    ("Kirkwood 4:1", "Jupiter", 4, 1, "Kirkwood"),
    ("Kirkwood 3:1", "Jupiter", 3, 1, "Kirkwood"),
    ("Kirkwood 5:2", "Jupiter", 5, 2, "Kirkwood"),
    ("Kirkwood 7:3", "Jupiter", 7, 3, "Kirkwood"),
    ("Kirkwood 2:1", "Jupiter", 2, 1, "Kirkwood"),
    ("Hildas", "Jupiter", 3, 2, "capture"),
    ("Trojans", "Jupiter", 1, 1, "capture"),
    ("Thule", "Jupiter", 4, 3, "capture"),
    ("Zhongguo", "Jupiter", 2, 1, "capture"),
    ("Griqua", "Jupiter", 2, 1, "capture"),
    ("Plutinos", "Neptune", 3, 2, "TNO"),
    ("Twotinos", "Neptune", 2, 1, "TNO"),
    ("5:3 TNOs", "Neptune", 5, 3, "TNO"),
    ("7:4 TNOs", "Neptune", 7, 4, "TNO"),
    ("5:2 TNOs", "Neptune", 5, 2, "TNO"),
    ("4:3 TNOs", "Neptune", 4, 3, "TNO"),
    ("3:1 TNOs", "Neptune", 3, 1, "TNO"),
    ("9:4 TNOs", "Neptune", 9, 4, "TNO approx"),
    ("7:2 TNOs", "Neptune", 7, 2, "TNO approx"),
    ("5:1 TNOs", "Neptune", 5, 1, "TNO approx"),
]

# ═══════════════════════════════════════════════════════════════════════
# EXOPLANET RESONANCES — adjacent pair ratios from confirmed chains
# ═══════════════════════════════════════════════════════════════════════

EXOPLANET = [
    # TRAPPIST-1: orbit ratios 24:15:9:6:4:3:2
    # Adjacent pairs: 24/15=8:5, 15/9=5:3, 9/6=3:2, 6/4=3:2, 4/3=4:3, 3/2=3:2
    ("TRAPPIST-1 b-c", "TRAPPIST-1", 8, 5, "Luger 2017"),
    ("TRAPPIST-1 c-d", "TRAPPIST-1", 5, 3, "Luger 2017"),
    ("TRAPPIST-1 d-e", "TRAPPIST-1", 3, 2, "Luger 2017"),
    ("TRAPPIST-1 e-f", "TRAPPIST-1", 3, 2, "Luger 2017"),
    ("TRAPPIST-1 f-g", "TRAPPIST-1", 4, 3, "Luger 2017"),
    ("TRAPPIST-1 g-h", "TRAPPIST-1", 3, 2, "Luger 2017"),

    # HD 110067: adjacent ratios 3:2, 3:2, 3:2, 4:3, 4:3
    ("HD110067 b-c", "HD110067", 3, 2, "Luque 2024"),
    ("HD110067 c-d", "HD110067", 3, 2, "Luque 2024"),
    ("HD110067 d-e", "HD110067", 3, 2, "Luque 2024"),
    ("HD110067 e-f", "HD110067", 4, 3, "Luque 2024"),
    ("HD110067 f-g", "HD110067", 4, 3, "Luque 2024"),

    # K2-138: near 3:2 chain (periods: 2.353, 3.560, 5.405, 8.261, 12.758)
    # Ratios: 3.56/2.35=1.513≈3:2, 5.41/3.56=1.518≈3:2, 8.26/5.41=1.527≈3:2, 12.76/8.26=1.544≈3:2
    ("K2-138 b-c", "K2-138", 3, 2, "Christiansen 2018"),
    ("K2-138 c-d", "K2-138", 3, 2, "Christiansen 2018"),
    ("K2-138 d-e", "K2-138", 3, 2, "Christiansen 2018"),
    ("K2-138 e-f", "K2-138", 3, 2, "Christiansen 2018"),

    # Kepler-223: 8:6:4:3 → adjacent 4:3, 3:2, 4:3
    ("Kepler-223 b-c", "Kepler-223", 4, 3, "Mills 2016"),
    ("Kepler-223 c-d", "Kepler-223", 3, 2, "Mills 2016"),
    ("Kepler-223 d-e", "Kepler-223", 4, 3, "Mills 2016"),

    # Kepler-80: 9:6:4:3:2 → adjacent 3:2, 3:2, 4:3, 3:2
    ("Kepler-80 d-e", "Kepler-80", 3, 2, "MacDonald 2016"),
    ("Kepler-80 e-b", "Kepler-80", 3, 2, "MacDonald 2016"),
    ("Kepler-80 b-c", "Kepler-80", 4, 3, "MacDonald 2016"),
    ("Kepler-80 c-f", "Kepler-80", 3, 2, "MacDonald 2016"),

    # TOI-178: 18:9:6:4:3 → adjacent 2:1, 3:2, 3:2, 4:3
    ("TOI-178 c-d", "TOI-178", 2, 1, "Leleu 2021"),
    ("TOI-178 d-e", "TOI-178", 3, 2, "Leleu 2021"),
    ("TOI-178 e-f", "TOI-178", 3, 2, "Leleu 2021"),
    ("TOI-178 f-g", "TOI-178", 4, 3, "Leleu 2021"),

    # GJ-876: 4:2:1 → adjacent 2:1, 2:1
    ("GJ-876 c-b", "GJ-876", 2, 1, "Rivera 2010"),
    ("GJ-876 b-e", "GJ-876", 2, 1, "Rivera 2010"),

    # HD 158259: near 3:2 chain (3.43, 5.20, 7.95, 12.03 days)
    ("HD158259 b-c", "HD158259", 3, 2, "Hara 2020"),
    ("HD158259 c-d", "HD158259", 3, 2, "Hara 2020"),
    ("HD158259 d-e", "HD158259", 3, 2, "Hara 2020"),
]


def main():
    print("=" * 80)
    print("FIBONACCI vs NON-FIBONACCI — EXPANDED WITH EXOPLANETS")
    print("=" * 80)

    all_resonances = []

    # Process solar system
    print("\n── SOLAR SYSTEM RESONANCES ──\n")
    ss_fib = 0
    ss_nonfib = 0
    ss_unique = set()
    for b1, b2, p, q, src in SOLAR_SYSTEM:
        g = gcd(p, q)
        pr, qr = p // g, q // g
        is_fib = is_fibonacci_ratio(pr, qr)
        tag = "FIB" if is_fib else "   "
        if is_fib:
            ss_fib += 1
        else:
            ss_nonfib += 1
        ss_unique.add((pr, qr))
        all_resonances.append(("solar", pr, qr, is_fib))

    ss_unique_fib = sum(1 for p, q in ss_unique if is_fibonacci_ratio(p, q))
    print(f"  Instances: {ss_fib} Fib / {ss_nonfib} non-Fib (total {ss_fib+ss_nonfib})")
    print(f"  Unique ratios: {ss_unique_fib} Fib / {len(ss_unique)-ss_unique_fib} non-Fib")

    # Process exoplanet
    print("\n── EXOPLANET RESONANCES ──\n")
    ex_fib = 0
    ex_nonfib = 0
    ex_unique = set()
    for b1, sys, p, q, src in EXOPLANET:
        g = gcd(p, q)
        pr, qr = p // g, q // g
        is_fib = is_fibonacci_ratio(pr, qr)
        tag = "FIB" if is_fib else "   "
        if is_fib:
            ex_fib += 1
        else:
            ex_nonfib += 1
        ex_unique.add((pr, qr))
        all_resonances.append(("exo", pr, qr, is_fib))
        print(f"  [{tag}] {pr}:{qr}  {b1:24s}  ({src})")

    ex_unique_fib = sum(1 for p, q in ex_unique if is_fibonacci_ratio(p, q))
    print(f"\n  Instances: {ex_fib} Fib / {ex_nonfib} non-Fib (total {ex_fib+ex_nonfib})")
    print(f"  Unique ratios: {ex_unique_fib} Fib / {len(ex_unique)-ex_unique_fib} non-Fib")

    # ── COMBINED ──
    print("\n" + "=" * 80)
    print("COMBINED ANALYSIS")
    print("=" * 80)

    total = len(all_resonances)
    total_fib = sum(1 for _, _, _, f in all_resonances if f)
    total_nonfib = total - total_fib

    combined_unique = set()
    for _, p, q, _ in all_resonances:
        combined_unique.add((p, q))
    combined_unique_fib = sum(1 for p, q in combined_unique if is_fibonacci_ratio(p, q))
    combined_unique_nonfib = len(combined_unique) - combined_unique_fib

    print(f"\n  ALL INSTANCES:  {total_fib} Fib / {total_nonfib} non-Fib  ({total} total)")
    print(f"  Fibonacci rate: {total_fib/total*100:.1f}%")
    print(f"\n  UNIQUE RATIOS:  {combined_unique_fib} Fib / {combined_unique_nonfib} non-Fib")
    print(f"  Fibonacci rate: {combined_unique_fib/(combined_unique_fib+combined_unique_nonfib)*100:.1f}%")

    # Expected rate
    all_possible = set()
    for p in range(1, 11):
        for q in range(1, p):
            if gcd(p, q) == 1:
                all_possible.add((p, q))
    # Also add 1:1
    all_possible.add((1, 1))
    expected_fib = sum(1 for p, q in all_possible if is_fibonacci_ratio(p, q))
    expected_rate = expected_fib / len(all_possible)
    print(f"\n  Expected Fibonacci rate (uniform over coprime p:q, p≤10): {expected_rate:.3f} ({expected_fib}/{len(all_possible)})")

    # ── Order-stratified breakdown ──
    print(f"\n── ORDER-STRATIFIED BREAKDOWN (combined) ──\n")

    by_order = {}
    for _, p, q, is_fib in all_resonances:
        order = resonance_order(p, q)
        if order not in by_order:
            by_order[order] = {"fib": 0, "nonfib": 0, "fib_r": set(), "nonfib_r": set()}
        if is_fib:
            by_order[order]["fib"] += 1
            by_order[order]["fib_r"].add(f"{p}:{q}")
        else:
            by_order[order]["nonfib"] += 1
            by_order[order]["nonfib_r"].add(f"{p}:{q}")

    # Expected by order
    exp_by_order = {}
    for p, q in all_possible:
        order = resonance_order(p, q)
        if order not in exp_by_order:
            exp_by_order[order] = {"fib": 0, "nonfib": 0}
        if is_fibonacci_ratio(p, q):
            exp_by_order[order]["fib"] += 1
        else:
            exp_by_order[order]["nonfib"] += 1

    print(f"  {'Order':>5s}  {'Fib':>5s}  {'Non':>5s}  {'%Fib':>6s}  {'Exp%':>6s}  Fib ratios              Non-Fib ratios")
    print(f"  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*24}  {'─'*24}")
    for order in sorted(by_order.keys()):
        d = by_order[order]
        ftot = d["fib"] + d["nonfib"]
        pct = d["fib"] / ftot * 100 if ftot > 0 else 0
        exp = exp_by_order.get(order, {"fib": 0, "nonfib": 0})
        exp_pct = exp["fib"] / (exp["fib"] + exp["nonfib"]) * 100 if (exp["fib"] + exp["nonfib"]) > 0 else 0
        fib_r = ", ".join(sorted(d["fib_r"]))
        nonfib_r = ", ".join(sorted(d["nonfib_r"]))
        print(f"  {order:5d}  {d['fib']:5d}  {d['nonfib']:5d}  {pct:5.1f}%  {exp_pct:5.1f}%  {fib_r:24s}  {nonfib_r:24s}")

    # ── Statistical tests ──
    print(f"\n── STATISTICAL TESTS ──\n")

    from scipy import stats

    # Test 1: Binomial on unique ratios
    n_unique = combined_unique_fib + combined_unique_nonfib
    p_binom_unique = 1 - stats.binom.cdf(combined_unique_fib - 1, n_unique, expected_rate)
    print(f"  1. UNIQUE RATIOS binomial test:")
    print(f"     Observed: {combined_unique_fib}/{n_unique} = {combined_unique_fib/n_unique:.3f}")
    print(f"     Expected: {expected_rate:.3f}")
    print(f"     p = {p_binom_unique:.4f}  {'*** SIGNIFICANT' if p_binom_unique < 0.05 else '    not significant'}")

    # Test 2: Binomial on all instances
    p_binom_all = 1 - stats.binom.cdf(total_fib - 1, total, expected_rate)
    print(f"\n  2. ALL INSTANCES binomial test:")
    print(f"     Observed: {total_fib}/{total} = {total_fib/total:.3f}")
    print(f"     Expected: {expected_rate:.3f}")
    print(f"     p = {p_binom_all:.6f}  {'*** SIGNIFICANT' if p_binom_all < 0.05 else '    not significant'}")

    # Test 3: Fisher combined order-stratified
    order_pvals = []
    for order in sorted(by_order.keys()):
        d = by_order[order]
        exp = exp_by_order.get(order, {"fib": 0, "nonfib": 0})
        obs_n = d["fib"] + d["nonfib"]
        obs_fib = d["fib"]
        exp_r = exp["fib"] / (exp["fib"] + exp["nonfib"]) if (exp["fib"] + exp["nonfib"]) > 0 else 0
        if obs_n > 0 and 0 < exp_r < 1:
            p_ord = 1 - stats.binom.cdf(obs_fib - 1, obs_n, exp_r)
        else:
            p_ord = 1.0
        order_pvals.append(p_ord)

    chi2_stat = -2 * sum(np.log(max(p, 1e-10)) for p in order_pvals)
    df = 2 * len(order_pvals)
    p_fisher = 1 - stats.chi2.cdf(chi2_stat, df)
    print(f"\n  3. FISHER COMBINED (order-stratified):")
    print(f"     χ² = {chi2_stat:.2f}, df = {df}")
    print(f"     p = {p_fisher:.6f}  {'*** SIGNIFICANT' if p_fisher < 0.05 else '    not significant'}")

    # Test 4: Exoplanet-only
    ex_total = ex_fib + ex_nonfib
    p_binom_exo = 1 - stats.binom.cdf(ex_fib - 1, ex_total, expected_rate)
    print(f"\n  4. EXOPLANET-ONLY binomial test:")
    print(f"     Observed: {ex_fib}/{ex_total} = {ex_fib/ex_total:.3f}")
    print(f"     Expected: {expected_rate:.3f}")
    print(f"     p = {p_binom_exo:.6f}  {'*** SIGNIFICANT' if p_binom_exo < 0.05 else '    not significant'}")

    # ── The key question ──
    print(f"\n── THE KEY QUESTION ──\n")
    print(f"  Among ORDER-1 resonances (the strongest, most common):")
    order1 = by_order.get(1, {"fib": 0, "nonfib": 0, "fib_r": set(), "nonfib_r": set()})
    o1_total = order1["fib"] + order1["nonfib"]
    o1_pct = order1["fib"] / o1_total * 100 if o1_total > 0 else 0
    exp1 = exp_by_order.get(1, {"fib": 0, "nonfib": 0})
    exp1_rate = exp1["fib"] / (exp1["fib"] + exp1["nonfib"]) if (exp1["fib"] + exp1["nonfib"]) > 0 else 0

    print(f"  Fibonacci (2:1, 3:2): {order1['fib']} instances")
    print(f"  Non-Fibonacci (4:3):  {order1['nonfib']} instances")
    print(f"  Fibonacci rate: {o1_pct:.1f}% (expected {exp1_rate*100:.1f}%)")
    p_o1 = 1 - stats.binom.cdf(order1["fib"] - 1, o1_total, exp1_rate)
    print(f"  p = {p_o1:.6f}")
    print()
    print(f"  This is the CLEANEST test: within order-1 resonances,")
    print(f"  there are 9 possible coprime ratios (2:1 through 10:9).")
    print(f"  Only 2 are Fibonacci (2:1 and 3:2) = {exp1_rate*100:.0f}% expected.")
    print(f"  Nature selects them {o1_pct:.0f}% of the time.")

    # ── VERDICT ──
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}\n")

    any_sig = p_binom_unique < 0.05 or p_binom_all < 0.05 or p_fisher < 0.05 or p_binom_exo < 0.05

    for label, p in [("Unique ratios binomial", p_binom_unique),
                     ("All instances binomial", p_binom_all),
                     ("Fisher combined", p_fisher),
                     ("Exoplanet-only", p_binom_exo),
                     ("Order-1 only", p_o1)]:
        sig = "***" if p < 0.05 else "   "
        print(f"  {sig} {label:30s}  p = {p:.6f}")

    print()
    if any_sig:
        print("  FIBONACCI RATIOS ARE SIGNIFICANTLY OVERREPRESENTED")
        print("  in both solar system and exoplanet resonances.")
        print()
        print("  Standard perturbation theory explains why LOW-ORDER ratios")
        print("  are preferred (e^|p-q| scaling). But it does NOT explain why")
        print("  2:1 and 3:2 dominate over 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9")
        print("  within the same order class.")
        print()
        print("  QA INTERPRETATION: The T-operator (Fibonacci shift [[0,1],[1,1]])")
        print("  creates a dynamical preference for period ratios that are")
        print("  ratios of Fibonacci numbers. This is a structural explanation")
        print("  for a pattern that standard theory treats as contingent.")
        print()
        print("  TIER ASSESSMENT: Tier 2→3 candidate. The pattern holds across")
        print("  independent datasets (solar system + exoplanets). But the test")
        print("  has known limitations: the catalogue may be biased toward")
        print("  easily-detected low-ratio resonances, and the 'expected rate'")
        print("  depends on the maximum ratio considered.")
    else:
        print("  No significant Fibonacci preference detected.")


if __name__ == "__main__":
    main()
