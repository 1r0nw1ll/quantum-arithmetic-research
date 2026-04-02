#!/usr/bin/env python3
"""
qa_stone_circle_analysis.py — QA analysis of stone circle diameters.

PRE-REGISTERED TEST:
If stone circles encode QA structure, their diameters in Megalithic Yards
should be QA-structured numbers (multiples of Fibonacci numbers, or
expressible as QA tuple elements b, e, d, or a).

DATA: Well-measured stone circles from published surveys.
MY = 2.72 feet (Thom's value) = 0.829 m.

DESIGN:
1. Convert each diameter to Megalithic Yards
2. Check if the MY value is close to an integer (Thom's claim)
3. Check if integer MY values are QA-structured:
   - Divisible by Fibonacci numbers?
   - Expressible as QA elements (b, e, d, a for some primitive QN)?
   - Land on specific mod-9 or mod-24 residues?
4. Null model: random diameters in the same range — what fraction
   would land near integers in MY by chance?

Author: Will Dale (question), Claude (analysis)
"""

QA_COMPLIANCE = "observer=stone_circle_analysis, state_alphabet=diameter_measurements"

import numpy as np
from math import gcd
from fractions import Fraction

np.random.seed(42)

MY_FEET = 2.72  # Megalithic Yard in feet
MY_METERS = 0.829  # Megalithic Yard in meters

# ═══════════════════════════════════════════════════════════════════════
# STONE CIRCLE DATA
# Sources: Wikipedia, Thom (1967), various published surveys
# ═══════════════════════════════════════════════════════════════════════

CIRCLES = [
    # (name, diameter_feet, diameter_meters, source, notes)
    ("Ring of Brodgar",     341.0,  104.0,  "Thom/Wikipedia", "Thom: 125 MY; true circle"),
    ("Avebury Outer",      1088.0,  331.6,  "Wikipedia",      "largest in Britain"),
    ("Avebury North Inner", 322.0,   98.0,  "Wikipedia",      "inner circle"),
    ("Avebury South Inner", 354.0,  108.0,  "Wikipedia",      "inner circle"),
    ("Stanton Drew Great",  371.0,  113.0,  "Wikipedia",      "2nd largest"),
    ("Stanton Drew NE",      98.0,   30.0,  "Wikipedia",      ""),
    ("Stanton Drew SW",     141.0,   43.0,  "Wikipedia",      ""),
    ("Stonehenge Sarsen",    98.0,   30.0,  "Wikipedia",      "sarsen circle"),
    ("Stonehenge Bank",     360.0,  110.0,  "Wikipedia",      "outer bank"),
    ("Callanish",            37.0,   11.4,  "Wikipedia",      "flattened E side"),
    # Additional well-known circles with Thom measurements
    ("Castlerigg",           97.5,   29.7,  "Thom",           "Type A flattened"),
    ("Long Meg",            359.0,  109.4,  "Thom",           "true circle or near"),
    ("Swinside",             93.0,   28.3,  "Thom",           "true circle, 55 stones"),
    ("Merry Maidens",        78.0,   23.8,  "Survey",         "true circle, 19 stones"),
    ("Rollright Stones",    104.0,   31.7,  "Survey",         "~77 stones"),
    ("Stenness",            104.0,   31.7,  "Thom",           "Orkney, 12 stones orig"),
]


def main():
    print("=" * 80)
    print("QA ANALYSIS OF STONE CIRCLE DIAMETERS")
    print("=" * 80)

    print(f"\n  Megalithic Yard (MY) = {MY_FEET} feet = {MY_METERS} m")
    print(f"  {len(CIRCLES)} circles analyzed\n")

    # ── Step 1: Convert to MY and check integer proximity ──
    print("── STEP 1: Diameters in Megalithic Yards ──\n")

    FIBS = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377}

    print(f"  {'Circle':30s}  {'Diam(ft)':>8s}  {'Diam(MY)':>8s}  {'Nearest':>7s}  {'Error':>7s}  {'Fib?':>5s}  {'mod9':>4s}  {'mod24':>5s}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*4}  {'─'*5}")

    my_values = []
    nearest_ints = []
    errors = []
    fib_hits = 0
    total = 0

    for name, diam_ft, diam_m, source, notes in CIRCLES:
        my = diam_ft / MY_FEET
        nearest = round(my)
        err = abs(my - nearest)
        err_pct = err / my * 100

        is_fib = nearest in FIBS
        if is_fib:
            fib_hits += 1
        total += 1

        mod9 = nearest % 9
        mod24 = nearest % 24

        my_values.append(my)
        nearest_ints.append(nearest)
        errors.append(err)

        fib_tag = "FIB" if is_fib else ""
        print(f"  {name:30s}  {diam_ft:8.1f}  {my:8.2f}  {nearest:7d}  {err:7.3f}  {fib_tag:>5s}  {mod9:4d}  {mod24:5d}")

    mean_err = np.mean(errors)
    print(f"\n  Mean fractional error: {mean_err:.4f} MY")
    print(f"  Fibonacci integer count: {fib_hits}/{total}")

    # ── Step 2: Null model — random diameters ──
    print(f"\n── STEP 2: Null Model — Random Diameters ──\n")

    # Generate random diameters in the same range
    min_diam = min(d for _, d, _, _, _ in CIRCLES)
    max_diam = max(d for _, d, _, _, _ in CIRCLES)
    N_NULL = 10000
    n_circles = len(CIRCLES)

    null_mean_errs = []
    null_fib_counts = []

    for _ in range(N_NULL):
        rand_diams = np.random.uniform(min_diam, max_diam, n_circles)
        rand_my = rand_diams / MY_FEET
        rand_nearest = np.round(rand_my).astype(int)
        rand_errs = np.abs(rand_my - rand_nearest)
        null_mean_errs.append(np.mean(rand_errs))
        fib_count = sum(1 for n in rand_nearest if n in FIBS)
        null_fib_counts.append(fib_count)

    null_mean_errs = np.array(null_mean_errs)
    null_fib_counts = np.array(null_fib_counts)

    # Test 1: Are real circles closer to integer MY than random?
    p_err = np.mean(null_mean_errs <= mean_err)
    print(f"  Test 1: Integer MY proximity")
    print(f"    Real mean error:  {mean_err:.4f}")
    print(f"    Null mean error:  {np.mean(null_mean_errs):.4f} ± {np.std(null_mean_errs):.4f}")
    print(f"    p-value (real < null): {p_err:.4f}")
    print(f"    {'SIGNIFICANT — diameters ARE closer to integer MY' if p_err < 0.05 else 'not significant'}")

    # Test 2: Do more MY values land on Fibonacci numbers?
    p_fib = np.mean(null_fib_counts >= fib_hits)
    print(f"\n  Test 2: Fibonacci integer count")
    print(f"    Real Fibonacci count: {fib_hits}/{total}")
    print(f"    Null mean: {np.mean(null_fib_counts):.2f} ± {np.std(null_fib_counts):.2f}")
    print(f"    p-value (real > null): {p_fib:.4f}")
    print(f"    {'SIGNIFICANT — more Fibonacci than expected' if p_fib < 0.05 else 'not significant'}")

    # ── Step 3: QA element analysis ──
    print(f"\n── STEP 3: QA Element Analysis ──\n")

    print("  For each integer MY diameter, check if it appears as")
    print("  b, e, d, or a in any primitive QN (b,e,d,a) with small values.\n")

    for name, diam_ft, diam_m, source, notes in CIRCLES:
        my = round(diam_ft / MY_FEET)
        # Check if my = d for some QN, or my = a, etc.
        roles = []
        # As d: d = b+e, need b,e coprime, b>=1, e>=1
        for e in range(1, my):
            b = my - e
            if b >= 1 and gcd(b, e) == 1:
                a = b + 2 * e
                roles.append(f"d of ({b},{e},{my},{a})")
                break  # just show first
        # As a: a = b+2e, need a=my
        for e in range(1, (my - 1) // 2 + 1):
            b = my - 2 * e
            if b >= 1 and gcd(b, e) == 1:
                d = b + e
                roles.append(f"a of ({b},{e},{d},{my})")
                break

        is_fib = my in FIBS
        print(f"  {name:30s}  {my:4d} MY  {'FIB' if is_fib else '   '}  {'; '.join(roles[:2])}")

    # ── Step 4: Thom's specific claim — 125 MY for Brodgar ──
    print(f"\n── STEP 4: Specific Claims ──\n")

    brodgar_my = 341.0 / MY_FEET
    print(f"  Ring of Brodgar: {341.0} ft / {MY_FEET} ft = {brodgar_my:.2f} MY")
    print(f"  Thom claimed: 125 MY (= 5³ = 5×5×5)")
    print(f"  Error: {abs(brodgar_my - 125):.2f} MY ({abs(brodgar_my - 125)/125*100:.2f}%)")
    print(f"  125 in QA: 125 = 5³. As d: (124,1,125,126) or (123,2,125,127) etc.")
    print(f"  125 mod 9 = {125 % 9}, 125 mod 24 = {125 % 24}")
    print(f"  Fibonacci? {'YES' if 125 in FIBS else 'NO (between F_10=55 and F_11=89... actually NO, 125=5³)'}")

    # ── Verdict ──
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}\n")

    print(f"  Integer MY proximity: p = {p_err:.4f} {'← SIGNIFICANT' if p_err < 0.05 else '← not significant'}")
    print(f"  Fibonacci count:      p = {p_fib:.4f} {'← SIGNIFICANT' if p_fib < 0.05 else '← not significant'}")
    print()
    if p_err < 0.05:
        print(f"  Stone circle diameters ARE closer to integer multiples of the")
        print(f"  Megalithic Yard than random diameters in the same range.")
        print(f"  This supports Thom's original claim of a standardized unit.")
    else:
        print(f"  Stone circle diameters are NOT significantly closer to integer MY")
        print(f"  than random. Thom's MY hypothesis may require his full dataset")
        print(f"  (~200 circles) rather than this sample of {total}.")
    print()
    print(f"  NOTE: This analysis uses {total} well-known circles, not Thom's")
    print(f"  full catalogue of ~200. The statistical power is limited.")
    print(f"  Thom's original analysis used the Broadbent method on the full set.")
    print(f"  To properly test the QA connection, we need the complete Table 5.1.")


if __name__ == "__main__":
    main()
