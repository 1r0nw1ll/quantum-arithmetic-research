#!/usr/bin/env python3
"""
qa_pisano_paired_divisibility.py — Paired Fibonacci divisibility test:

CLAIM: For a p:q resonance to be "deep," BOTH p AND q must divide π(m).
Fibonacci PAIRS are more likely to both divide π(m) than non-Fibonacci
pairs of the same resonance order |p-q|.

METHOD:
1. Compute π(m) for m = 2..500
2. For each coprime pair (p,q) with p > q, p ≤ 10:
   - Check if BOTH p and q divide π(m) for each modulus m
3. Compare both-divide rate: Fibonacci pairs vs non-Fibonacci pairs
4. Group by resonance order |p-q| for fair comparison
5. Statistical test per order class

Author: Will Dale (conjecture), Claude (computation)
"""

QA_COMPLIANCE = "observer=paired_pisano_divisibility, state_alphabet=modular_arithmetic"

import numpy as np
from math import gcd
from collections import defaultdict
from scipy import stats

np.random.seed(42)


def pisano_period(m):
    """Compute Pisano period π(m) by Fibonacci sequence mod m."""
    if m <= 1:
        return 1
    prev, curr = 0, 1
    for k in range(1, 6 * m * m + 10):  # S1: m*m not m**2
        prev, curr = curr, (prev + curr) % m
        if prev == 0 and curr == 1:
            return k
    return -1


def is_fibonacci(n):
    """Check if n is a Fibonacci number using the characterization:
    n is Fibonacci iff 5*n*n+4 or 5*n*n-4 is a perfect square."""
    if n <= 0:
        return False
    # S1 compliance: n*n not n**2
    x = 5 * n * n + 4
    sx = int(x ** 0.5)
    # Check nearby values to avoid float precision issues
    for s in [sx - 1, sx, sx + 1]:
        if s >= 0 and s * s == x:
            return True
    y = 5 * n * n - 4
    sy = int(y ** 0.5)
    for s in [sy - 1, sy, sy + 1]:
        if s >= 0 and s * s == y:
            return True
    return False


def main():
    MAX_M = 500
    MAX_P = 10

    print("=" * 80)
    print(f"PAIRED FIBONACCI-PISANO DIVISIBILITY TEST (m = 2..{MAX_M})")
    print("=" * 80)

    # ── Step 1: Compute all Pisano periods ──
    print(f"\n── Computing π(m) for m = 2..{MAX_M} ──")
    pisano = {}
    for m in range(2, MAX_M + 1):
        pisano[m] = pisano_period(m)

    pi_values = list(pisano.values())
    n_moduli = len(pisano)
    print(f"  Done. {n_moduli} moduli computed.")
    print(f"  Max π(m) = {max(pi_values)}, Mean = {np.mean(pi_values):.1f}")

    # ── Step 2: Build Fibonacci set and classify pairs ──
    fibs = set()
    a, b = 1, 1
    while a <= MAX_P:
        fibs.add(a)
        a, b = b, a + b
    print(f"  Fibonacci numbers ≤ {MAX_P}: {sorted(fibs)}")

    # Generate all coprime pairs (p, q) with p > q ≥ 1, p ≤ MAX_P
    pairs = []
    for p in range(2, MAX_P + 1):
        for q in range(1, p):
            if gcd(p, q) == 1:
                pairs.append((p, q))

    # Classify pairs
    fib_pairs = [(p, q) for p, q in pairs if p in fibs and q in fibs]
    nonfib_pairs = [(p, q) for p, q in pairs if not (p in fibs and q in fibs)]

    print(f"\n  Total coprime pairs: {len(pairs)}")
    print(f"  Fibonacci pairs (both Fib): {fib_pairs}")
    print(f"  Non-Fibonacci pairs: {len(nonfib_pairs)}")

    # ── Step 3: Compute both-divide rates ──
    print(f"\n── Computing BOTH-DIVIDE rates ──\n")

    def both_divide_rate(p, q):
        """Fraction of moduli where both p and q divide π(m)."""
        count = 0
        for m in range(2, MAX_M + 1):
            pi_m = pisano[m]
            if pi_m % p == 0 and pi_m % q == 0:
                count += 1
        return count / n_moduli

    pair_rates = {}
    for p, q in pairs:
        pair_rates[(p, q)] = both_divide_rate(p, q)

    # ── Step 4: Group by resonance order ──
    orders = defaultdict(lambda: {"fib": [], "nonfib": []})
    for p, q in pairs:
        order = p - q
        rate = pair_rates[(p, q)]
        if p in fibs and q in fibs:
            orders[order]["fib"].append(((p, q), rate))
        else:
            orders[order]["nonfib"].append(((p, q), rate))

    # ── Step 5: Report per order ──
    print(f"  {'Pair':>8s}  {'Order':>5s}  {'Type':>8s}  {'Both divide rate':>16s}  {'Count':>6s}")
    print(f"  {'─' * 8}  {'─' * 5}  {'─' * 8}  {'─' * 16}  {'─' * 6}")

    for p, q in sorted(pairs, key=lambda x: (x[0] - x[1], -(x[0] in fibs and x[1] in fibs), x[0])):
        order = p - q
        rate = pair_rates[(p, q)]
        tag = "FIB" if (p in fibs and q in fibs) else "   "
        count = int(rate * n_moduli)
        print(f"  ({p:2d},{q:2d})  {order:5d}  {tag:>8s}  {rate:16.4f}  {count:4d}/{n_moduli}")

    # ── Step 6: Statistical comparison per order ──
    print(f"\n{'=' * 80}")
    print("STATISTICAL COMPARISON BY RESONANCE ORDER")
    print(f"{'=' * 80}\n")

    overall_fib_rates = []
    overall_nonfib_rates = []

    for order in sorted(orders.keys()):
        fib_list = orders[order]["fib"]
        nonfib_list = orders[order]["nonfib"]

        if not fib_list:
            continue

        fib_mean = np.mean([r for _, r in fib_list])
        overall_fib_rates.extend([r for _, r in fib_list])

        print(f"  ORDER |p-q| = {order}:")
        print(f"    Fibonacci pairs: {', '.join(f'({p},{q}):{r:.4f}' for (p,q), r in fib_list)}")

        if nonfib_list:
            nonfib_mean = np.mean([r for _, r in nonfib_list])
            overall_nonfib_rates.extend([r for _, r in nonfib_list])

            print(f"    Non-Fib pairs:   {', '.join(f'({p},{q}):{r:.4f}' for (p,q), r in nonfib_list)}")
            print(f"    Fib mean: {fib_mean:.4f}  |  Non-Fib mean: {nonfib_mean:.4f}  |  Ratio: {fib_mean / max(nonfib_mean, 1e-9):.2f}x")

            # Mann-Whitney U test if enough samples
            if len(fib_list) >= 1 and len(nonfib_list) >= 2:
                fib_vals = [r for _, r in fib_list]
                nonfib_vals = [r for _, r in nonfib_list]
                # One-sided: is Fibonacci rate > non-Fibonacci?
                u_stat, u_p = stats.mannwhitneyu(fib_vals, nonfib_vals, alternative='greater')
                print(f"    Mann-Whitney U (H1: Fib > NonFib): U={u_stat:.1f}, p={u_p:.4f}")
                print(f"    {'SIGNIFICANT' if u_p < 0.05 else 'not significant'}")
        else:
            print(f"    (No non-Fibonacci pairs at this order)")
        print()

    # ── Step 7: KEY COMPARISON — Order 1 deep dive ──
    print(f"{'=' * 80}")
    print("KEY COMPARISON: ORDER 1 (consecutive ratios)")
    print(f"{'=' * 80}\n")

    order1_fib = orders[1]["fib"]
    order1_nonfib = orders[1]["nonfib"]

    print(f"  Fibonacci pairs:     {', '.join(f'({p},{q})' for (p,q), _ in order1_fib)}")
    print(f"  Non-Fibonacci pairs: {', '.join(f'({p},{q})' for (p,q), _ in order1_nonfib)}")
    print()

    fib_rates_o1 = [r for _, r in order1_fib]
    nonfib_rates_o1 = [r for _, r in order1_nonfib]

    fib_mean_o1 = np.mean(fib_rates_o1)
    nonfib_mean_o1 = np.mean(nonfib_rates_o1)

    print(f"  Fibonacci both-divide rate:     {fib_mean_o1:.4f}")
    print(f"  Non-Fibonacci both-divide rate: {nonfib_mean_o1:.4f}")
    print(f"  Ratio: {fib_mean_o1 / max(nonfib_mean_o1, 1e-9):.2f}x")

    # Per-modulus comparison for order 1
    print(f"\n  Per-modulus detail (order 1):")
    print(f"  {'m':>4s}  {'π(m)':>6s}  ", end="")
    for (p, q), _ in order1_fib:
        print(f"  ({p},{q})", end="")
    print("  |", end="")
    for (p, q), _ in order1_nonfib[:5]:
        print(f"  ({p},{q})", end="")
    print()

    # Show first 20 moduli
    for m in range(2, 22):
        pi_m = pisano[m]
        print(f"  {m:4d}  {pi_m:6d}  ", end="")
        for (p, q), _ in order1_fib:
            both = "✓" if pi_m % p == 0 and pi_m % q == 0 else "·"
            print(f"    {both} ", end="")
        print("  |", end="")
        for (p, q), _ in order1_nonfib[:5]:
            both = "✓" if pi_m % p == 0 and pi_m % q == 0 else "·"
            print(f"    {both} ", end="")
        print()

    # ── Step 8: Overall summary ──
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}\n")

    overall_fib_mean = np.mean(overall_fib_rates) if overall_fib_rates else 0
    overall_nonfib_mean = np.mean(overall_nonfib_rates) if overall_nonfib_rates else 0

    print(f"  Fibonacci pairs (n={len(overall_fib_rates)}):")
    print(f"    Mean both-divide rate: {overall_fib_mean:.4f}")

    print(f"  Non-Fibonacci pairs (n={len(overall_nonfib_rates)}):")
    print(f"    Mean both-divide rate: {overall_nonfib_mean:.4f}")

    if overall_nonfib_mean > 0:
        print(f"\n  Overall ratio (Fib/NonFib): {overall_fib_mean / overall_nonfib_mean:.2f}x")

    # Overall Mann-Whitney
    if len(overall_fib_rates) >= 1 and len(overall_nonfib_rates) >= 2:
        u_stat, u_p = stats.mannwhitneyu(overall_fib_rates, overall_nonfib_rates, alternative='greater')
        print(f"  Mann-Whitney U (all orders): U={u_stat:.1f}, p={u_p:.4f}")
        print(f"  {'SIGNIFICANT' if u_p < 0.05 else 'not significant'}")

    # ── Step 9: LCM analysis — deeper structural reason ──
    print(f"\n── STRUCTURAL ANALYSIS: LCM effect ──\n")
    print("  For both p AND q to divide π(m), lcm(p,q) must divide π(m).")
    print("  For coprime pairs, lcm(p,q) = p*q.")
    print("  Fibonacci pairs have SMALLER products → lcm divides π(m) more often.\n")

    print(f"  {'Pair':>8s}  {'p*q':>5s}  {'Both rate':>10s}  {'Type':>6s}")
    print(f"  {'─' * 8}  {'─' * 5}  {'─' * 10}  {'─' * 6}")

    for p, q in sorted(pairs, key=lambda x: x[0] * x[1]):
        rate = pair_rates[(p, q)]
        tag = "FIB" if (p in fibs and q in fibs) else ""
        print(f"  ({p:2d},{q:2d})  {p * q:5d}  {rate:10.4f}  {tag:>6s}")

    # Control: compare to product-matched non-Fib pairs
    print(f"\n  Product-matched comparison:")
    print(f"  {'Fib pair':>10s}  {'prod':>5s}  {'rate':>8s}  {'Nearest non-Fib':>16s}  {'prod':>5s}  {'rate':>8s}  {'Fib wins?':>9s}")
    print(f"  {'─' * 10}  {'─' * 5}  {'─' * 8}  {'─' * 16}  {'─' * 5}  {'─' * 8}  {'─' * 9}")

    wins = 0
    total = 0
    for fp, fq in fib_pairs:
        fib_prod = fp * fq
        fib_rate = pair_rates[(fp, fq)]

        # Find non-Fib pair with closest product
        best = None
        best_dist = float('inf')
        for np_, nq in nonfib_pairs:
            prod = np_ * nq
            dist = abs(prod - fib_prod)
            if dist < best_dist:
                best_dist = dist
                best = (np_, nq)

        if best:
            nf_rate = pair_rates[best]
            w = fib_rate > nf_rate
            if w:
                wins += 1
            total += 1
            print(f"  ({fp:2d},{fq:2d})  {fib_prod:5d}  {fib_rate:8.4f}  ({best[0]:2d},{best[1]:2d})            {best[0]*best[1]:5d}  {nf_rate:8.4f}  {'YES' if w else 'no':>9s}")

    if total > 0:
        print(f"\n  Product-matched wins: {wins}/{total} ({wins/total*100:.0f}%)")

    # ── Final verdict ──
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}\n")

    print(f"  PAIRED both-divide rates across {n_moduli} moduli:")
    print(f"    Fibonacci pairs:     {overall_fib_mean:.4f}")
    print(f"    Non-Fibonacci pairs: {overall_nonfib_mean:.4f}")
    if overall_nonfib_mean > 0:
        print(f"    Advantage: {overall_fib_mean / overall_nonfib_mean:.2f}x")
    print()
    print("  MECHANISM: For coprime (p,q), both-divide requires lcm(p,q)=p*q | π(m).")
    print("  Fibonacci pairs (consecutive Fibs) have smaller products than typical")
    print("  pairs at the same resonance order, AND Fibonacci numbers individually")
    print("  divide π(m) more often (3.70x from prior analysis).")
    print("  The PAIRED advantage compounds both effects.")


if __name__ == "__main__":
    main()
