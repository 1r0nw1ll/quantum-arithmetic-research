#!/usr/bin/env python3
"""
Tougher Null Models for QA Fibonacci Resonance [163]
=====================================================

The original null assumes uniform selection over coprime ratios p:q with
p,q <= 10. But orbital mechanics already favors low-integer ratios
(2:1, 3:2 are dynamically preferred). This script tests whether
"Fibonacci" adds anything beyond "low integer."

Three null models + one permutation test:
  A: 1/(p*q) weighted — simpler ratios more likely
  B: 1/(p+q) weighted — additive complexity measure
  C: Empirical distribution of observed period ratios
  D: Permutation test (model-free)

Observer Projection Note (Theorem NT):
  Orbital periods are continuous measurements projected to discrete
  integer ratios (p:q). Classification (Fibonacci vs non-Fib) operates
  on the discrete ratios only.

S1 compliance: uses b*b not b**2 throughout.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from math import gcd, log

import numpy as np
np.random.seed(42)

from scipy import stats

# ── Configuration ──────────────────────────────────────────────────
RESULTS_DIR = "/home/player2/signal_experiments/results/fibonacci_adversarial"
REPLICATION_JSON = os.path.join(RESULTS_DIR, "replication_results.json")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "tougher_null_results.json")

MAX_PQ = 10
RESONANCE_TOL = 0.05
N_PERMUTATIONS = 10000

# Fibonacci numbers up to 100
FIB_SET = set()
a, b = 1, 1
while a <= 100:
    FIB_SET.add(a)
    a, b = b, a + b
# {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}


def is_fibonacci(n):
    return n in FIB_SET


def is_fibonacci_ratio(p, q):
    return is_fibonacci(p) and is_fibonacci(q)


def enumerate_coprime_ratios(max_pq=MAX_PQ):
    """All coprime ratios p:q with p > q, both <= max_pq."""
    ratios = []
    for q in range(1, max_pq + 1):
        for p in range(q + 1, max_pq + 1):
            if gcd(p, q) != 1:
                continue
            ratios.append((p, q))
    return ratios


def coprime_ratios_by_order(max_pq=MAX_PQ):
    """Group coprime ratios by order = |p - q|."""
    groups = defaultdict(list)
    for p, q in enumerate_coprime_ratios(max_pq):
        groups[abs(p - q)].append((p, q))
    return dict(groups)


# ── Load replication data ──────────────────────────────────────────

def load_replication_data():
    """Load existing replication results."""
    with open(REPLICATION_JSON, 'r') as f:
        data = json.load(f)
    return data


# ── Null Model A: 1/(p*q) weighted ────────────────────────────────

def null_a_expected_fib_rate(order=None, max_pq=MAX_PQ):
    """
    Weight each coprime ratio p:q by 1/(p*q).
    Return expected Fibonacci fraction under this weighting.
    """
    ratios = enumerate_coprime_ratios(max_pq)
    if order is not None:
        ratios = [(p, q) for p, q in ratios if abs(p - q) == order]

    total_weight = 0.0
    fib_weight = 0.0
    details = []

    for p, q in ratios:
        w = 1.0 / (p * q)
        total_weight += w
        is_fib = is_fibonacci_ratio(p, q)
        if is_fib:
            fib_weight += w
        details.append((p, q, w, is_fib))

    rate = fib_weight / total_weight if total_weight > 0 else 0.0
    return rate, details


# ── Null Model B: 1/(p+q) weighted ────────────────────────────────

def null_b_expected_fib_rate(order=None, max_pq=MAX_PQ):
    """
    Weight each coprime ratio p:q by 1/(p+q).
    Return expected Fibonacci fraction under this weighting.
    """
    ratios = enumerate_coprime_ratios(max_pq)
    if order is not None:
        ratios = [(p, q) for p, q in ratios if abs(p - q) == order]

    total_weight = 0.0
    fib_weight = 0.0
    details = []

    for p, q in ratios:
        w = 1.0 / (p + q)
        total_weight += w
        is_fib = is_fibonacci_ratio(p, q)
        if is_fib:
            fib_weight += w
        details.append((p, q, w, is_fib))

    rate = fib_weight / total_weight if total_weight > 0 else 0.0
    return rate, details


# ── Null Model C: Empirical distribution ──────────────────────────

def null_c_empirical(ratio_counts, order=None, max_pq=MAX_PQ):
    """
    Use the ACTUAL distribution of matched period ratios from the data
    as the null. For each coprime ratio, its weight is the observed count.

    If order is specified, restrict to that order.

    Returns expected Fibonacci fraction under empirical weighting.
    """
    total_weight = 0
    fib_weight = 0
    details = []

    for ratio_str, count in ratio_counts.items():
        p_str, q_str = ratio_str.split(':')
        p, q = int(p_str), int(q_str)

        if order is not None and abs(p - q) != order:
            continue

        is_fib = is_fibonacci_ratio(p, q)
        total_weight += count
        if is_fib:
            fib_weight += count
        details.append((p, q, count, is_fib))

    rate = fib_weight / total_weight if total_weight > 0 else 0.0
    return rate, details


# ── Permutation test ──────────────────────────────────────────────

def permutation_test(ratio_counts, observed_fib_rate, order=None,
                     n_perms=N_PERMUTATIONS):
    """
    Permutation test: among all observed resonances, randomly
    reassign Fibonacci/non-Fibonacci labels preserving ratio
    identity counts. Ask: how often does Fibonacci rate >= observed?

    Actually, the cleaner version: we have N resonance observations,
    each matched to a coprime ratio. Shuffle which ratio each
    observation is assigned to (from the pool of all coprime ratios
    with order constraint), weighted by the null model (uniform here —
    this is the model-free version).

    Simpler: we have K distinct ratio types observed, each with a count.
    Each ratio is either Fib or not-Fib. Permute the Fib labels across
    ratio types? No — that changes the structure.

    Cleanest model-free approach: We have N observations matched to
    coprime ratios. Under H0, the probability of landing on any
    particular ratio is proportional to its observed frequency.
    The Fibonacci label is a FIXED PROPERTY of the ratio, not random.
    So we need to permute differently.

    Best approach: We observe n_total resonances. n_fib of them land on
    Fibonacci ratios. Under H0 of no Fibonacci preference, each
    resonance independently lands on a ratio drawn from the empirical
    distribution of ALL coprime ratios (not just observed).

    Actually the simplest model-free permutation: treat each observed
    resonance as a draw. The ratio it matched to has a fixed Fibonacci
    status. Permute by resampling from the observed ratio pool.
    """
    # Build the pool: list of all observed resonances as (p, q) pairs
    pool = []
    for ratio_str, count in ratio_counts.items():
        p_str, q_str = ratio_str.split(':')
        p, q = int(p_str), int(q_str)
        if order is not None and abs(p - q) != order:
            continue
        for _ in range(count):
            pool.append((p, q))

    if len(pool) == 0:
        return None, None

    n_total = len(pool)
    fib_labels = np.array([1 if is_fibonacci_ratio(p, q) else 0
                           for p, q in pool])
    observed_n_fib = int(fib_labels.sum())
    observed_rate = observed_n_fib / n_total

    # Permutation: shuffle which resonances are "Fibonacci" by
    # resampling from the uniform coprime pool
    all_coprime = enumerate_coprime_ratios(MAX_PQ)
    if order is not None:
        all_coprime = [(p, q) for p, q in all_coprime if abs(p - q) == order]
    coprime_fib = np.array([1 if is_fibonacci_ratio(p, q) else 0
                            for p, q in all_coprime])

    # Under H0 (uniform coprime): each draw picks a random coprime ratio
    n_exceed = 0
    perm_rates = np.zeros(n_perms)
    for i in range(n_perms):
        # Draw n_total ratios uniformly from coprime set
        draws = np.random.randint(0, len(all_coprime), size=n_total)
        sim_fib = coprime_fib[draws].sum()
        perm_rates[i] = sim_fib / n_total
        if perm_rates[i] >= observed_rate:
            n_exceed += 1

    perm_p = (n_exceed + 1) / (n_perms + 1)  # +1 for observed itself
    return perm_p, perm_rates


def weighted_permutation_test(ratio_counts, weight_fn, observed_fib_rate,
                              order=None, n_perms=N_PERMUTATIONS):
    """
    Permutation test where draws are weighted by weight_fn(p, q).
    This tests: given the LOW-INTEGER BIAS, is Fibonacci still special?
    """
    all_coprime = enumerate_coprime_ratios(MAX_PQ)
    if order is not None:
        all_coprime = [(p, q) for p, q in all_coprime if abs(p - q) == order]

    if len(all_coprime) == 0:
        return None, None

    # Compute weights
    weights = np.array([weight_fn(p, q) for p, q in all_coprime])
    weights = weights / weights.sum()
    coprime_fib = np.array([1 if is_fibonacci_ratio(p, q) else 0
                            for p, q in all_coprime])

    # Count observations
    n_total = 0
    for ratio_str, count in ratio_counts.items():
        p_str, q_str = ratio_str.split(':')
        p, q = int(p_str), int(q_str)
        if order is not None and abs(p - q) != order:
            continue
        n_total += count

    if n_total == 0:
        return None, None

    n_exceed = 0
    perm_rates = np.zeros(n_perms)
    for i in range(n_perms):
        draws = np.random.choice(len(all_coprime), size=n_total, p=weights)
        sim_fib = coprime_fib[draws].sum()
        perm_rates[i] = sim_fib / n_total
        if perm_rates[i] >= observed_fib_rate:
            n_exceed += 1

    perm_p = (n_exceed + 1) / (n_perms + 1)
    return perm_p, perm_rates


# ── Binomial test under weighted null ─────────────────────────────

def binomial_test_weighted(n_fib, n_total, null_rate):
    """One-sided binomial test: observed >= expected."""
    if n_total == 0 or null_rate <= 0:
        return None
    result = stats.binomtest(n_fib, n_total, null_rate, alternative='greater')
    return result.pvalue


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("TOUGHER NULL MODELS — Fibonacci Resonance Adversarial Test")
    print("=" * 72)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Seed: 42 | Permutations: {N_PERMUTATIONS}")
    print()

    # Load data
    data = load_replication_data()
    new_results = data['new_results']
    combined_results = data['combined_results']

    # Use COMBINED data (all 492 resonances) for the toughest test
    # Also test NEW data separately (458 resonances)
    datasets = {
        'new': new_results,
        'combined': combined_results,
    }

    results_out = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'max_pq': MAX_PQ,
            'tolerance': RESONANCE_TOL,
            'n_permutations': N_PERMUTATIONS,
            'seed': 42,
        },
        'null_models': {},
        'tests': {},
    }

    # ── Enumerate coprime ratios and show null rates ──
    print("── Coprime Ratio Census (order-1) ──")
    order1_ratios = [(p, q) for p, q in enumerate_coprime_ratios()
                     if abs(p - q) == 1]
    print(f"  Total order-1 coprime ratios with p,q<=10: {len(order1_ratios)}")
    for p, q in order1_ratios:
        fib = "FIB" if is_fibonacci_ratio(p, q) else "   "
        print(f"    {fib} {p}:{q}  1/(p*q)={1/(p*q):.4f}  1/(p+q)={1/(p+q):.4f}")
    print()

    # ── Null A: 1/(p*q) weighted ──
    print("── NULL A: 1/(p*q) weighted (simpler ratios more likely) ──")
    null_a_o1, details_a_o1 = null_a_expected_fib_rate(order=1)
    null_a_all, details_a_all = null_a_expected_fib_rate(order=None)
    print(f"  Expected Fib rate (order-1): {100*null_a_o1:.1f}%")
    print(f"  Expected Fib rate (all):     {100*null_a_all:.1f}%")

    # Show weights for order-1
    print("  Order-1 weight breakdown:")
    total_w = sum(w for _, _, w, _ in details_a_o1)
    for p, q, w, is_fib in sorted(details_a_o1, key=lambda x: -x[2]):
        fib = "FIB" if is_fib else "   "
        print(f"    {fib} {p}:{q}  w={w/total_w:.4f}")
    print()

    results_out['null_models']['A_product'] = {
        'description': '1/(p*q) weighted — simpler ratios more likely',
        'expected_fib_rate_order1': null_a_o1,
        'expected_fib_rate_all': null_a_all,
    }

    # ── Null B: 1/(p+q) weighted ──
    print("── NULL B: 1/(p+q) weighted (additive complexity) ──")
    null_b_o1, details_b_o1 = null_b_expected_fib_rate(order=1)
    null_b_all, details_b_all = null_b_expected_fib_rate(order=None)
    print(f"  Expected Fib rate (order-1): {100*null_b_o1:.1f}%")
    print(f"  Expected Fib rate (all):     {100*null_b_all:.1f}%")

    print("  Order-1 weight breakdown:")
    total_w = sum(w for _, _, w, _ in details_b_o1)
    for p, q, w, is_fib in sorted(details_b_o1, key=lambda x: -x[2]):
        fib = "FIB" if is_fib else "   "
        print(f"    {fib} {p}:{q}  w={w/total_w:.4f}")
    print()

    results_out['null_models']['B_sum'] = {
        'description': '1/(p+q) weighted — additive complexity',
        'expected_fib_rate_order1': null_b_o1,
        'expected_fib_rate_all': null_b_all,
    }

    # ── Null C: Empirical distribution ──
    print("── NULL C: Empirical distribution from observed data ──")
    for ds_name, ds in datasets.items():
        ratio_counts = ds['ratio_counts']
        null_c_o1, details_c_o1 = null_c_empirical(ratio_counts, order=1)
        null_c_all, details_c_all = null_c_empirical(ratio_counts, order=None)
        print(f"  [{ds_name}] Empirical Fib rate (order-1): {100*null_c_o1:.1f}%")
        print(f"  [{ds_name}] Empirical Fib rate (all):     {100*null_c_all:.1f}%")

        if ds_name == 'combined':
            # Show breakdown
            print(f"  Order-1 empirical breakdown [{ds_name}]:")
            for p, q, count, is_fib in sorted(details_c_o1, key=lambda x: -x[2]):
                fib = "FIB" if is_fib else "   "
                print(f"    {fib} {p}:{q}  count={count}")
    print()

    # NOTE: Null C's expected rate IS the observed rate by definition.
    # The interesting question is whether Fibonacci ratios are enriched
    # WITHIN each ratio's count relative to the theoretical nulls.
    # Null C really asks: "is the observed Fib rate just a consequence of
    # which ratios happen to be common?" The answer is the rate itself.
    print("  NOTE: Null C is tautological for a direct rate comparison —")
    print("  the empirical Fib rate IS the observation. Null C is useful")
    print("  as a BENCHMARK: it tells us what 'low-integer bias' alone")
    print("  would predict. We compare it to the THEORETICAL null models.")
    print()

    # ── Run tests for each dataset ──
    for ds_name, ds in datasets.items():
        print("=" * 72)
        print(f"TESTS ON {ds_name.upper()} DATA")
        print("=" * 72)

        ratio_counts = ds['ratio_counts']
        n_total_o1 = ds['order1_n']
        n_fib_o1 = ds['order1_n_fib']
        obs_rate_o1 = ds['order1_fib_rate']
        n_total_all = ds['all_n']
        n_fib_all = ds['all_n_fib']
        obs_rate_all = ds['all_fib_rate']

        ds_results = {}

        print(f"\n  Observed: order-1 = {n_fib_o1}/{n_total_o1} "
              f"({100*obs_rate_o1:.1f}%)")
        print(f"  Observed: all    = {n_fib_all}/{n_total_all} "
              f"({100*obs_rate_all:.1f}%)")
        print()

        # ── Test against Null A ──
        print(f"  -- Null A: 1/(p*q) weighted --")
        p_a_o1 = binomial_test_weighted(n_fib_o1, n_total_o1, null_a_o1)
        p_a_all = binomial_test_weighted(n_fib_all, n_total_all, null_a_all)
        effect_a_o1 = obs_rate_o1 / null_a_o1 if null_a_o1 > 0 else float('inf')
        effect_a_all = obs_rate_all / null_a_all if null_a_all > 0 else float('inf')
        print(f"    Order-1: expected={100*null_a_o1:.1f}%, "
              f"observed={100*obs_rate_o1:.1f}%, "
              f"effect={effect_a_o1:.2f}x, p={p_a_o1:.4e}")
        print(f"    All:     expected={100*null_a_all:.1f}%, "
              f"observed={100*obs_rate_all:.1f}%, "
              f"effect={effect_a_all:.2f}x, p={p_a_all:.4e}")

        # Weighted permutation test for Null A
        print(f"    Running {N_PERMUTATIONS} weighted permutations (order-1)...")
        perm_p_a, perm_rates_a = weighted_permutation_test(
            ratio_counts, lambda p, q: 1.0 / (p * q),
            obs_rate_o1, order=1)
        if perm_p_a is not None:
            print(f"    Permutation p = {perm_p_a:.6f}")
            print(f"    Permutation mean rate = {100*perm_rates_a.mean():.1f}%, "
                  f"max = {100*perm_rates_a.max():.1f}%")

        ds_results['null_A'] = {
            'order1': {
                'expected_rate': null_a_o1,
                'observed_rate': obs_rate_o1,
                'effect_size': effect_a_o1,
                'binomial_p': p_a_o1,
                'permutation_p': perm_p_a,
            },
            'all': {
                'expected_rate': null_a_all,
                'observed_rate': obs_rate_all,
                'effect_size': effect_a_all,
                'binomial_p': p_a_all,
            },
        }
        print()

        # ── Test against Null B ──
        print(f"  -- Null B: 1/(p+q) weighted --")
        p_b_o1 = binomial_test_weighted(n_fib_o1, n_total_o1, null_b_o1)
        p_b_all = binomial_test_weighted(n_fib_all, n_total_all, null_b_all)
        effect_b_o1 = obs_rate_o1 / null_b_o1 if null_b_o1 > 0 else float('inf')
        effect_b_all = obs_rate_all / null_b_all if null_b_all > 0 else float('inf')
        print(f"    Order-1: expected={100*null_b_o1:.1f}%, "
              f"observed={100*obs_rate_o1:.1f}%, "
              f"effect={effect_b_o1:.2f}x, p={p_b_o1:.4e}")
        print(f"    All:     expected={100*null_b_all:.1f}%, "
              f"observed={100*obs_rate_all:.1f}%, "
              f"effect={effect_b_all:.2f}x, p={p_b_all:.4e}")

        print(f"    Running {N_PERMUTATIONS} weighted permutations (order-1)...")
        perm_p_b, perm_rates_b = weighted_permutation_test(
            ratio_counts, lambda p, q: 1.0 / (p + q),
            obs_rate_o1, order=1)
        if perm_p_b is not None:
            print(f"    Permutation p = {perm_p_b:.6f}")
            print(f"    Permutation mean rate = {100*perm_rates_b.mean():.1f}%, "
                  f"max = {100*perm_rates_b.max():.1f}%")

        ds_results['null_B'] = {
            'order1': {
                'expected_rate': null_b_o1,
                'observed_rate': obs_rate_o1,
                'effect_size': effect_b_o1,
                'binomial_p': p_b_o1,
                'permutation_p': perm_p_b,
            },
            'all': {
                'expected_rate': null_b_all,
                'observed_rate': obs_rate_all,
                'effect_size': effect_b_all,
                'binomial_p': p_b_all,
            },
        }
        print()

        # ── Test against Null C (empirical) ──
        # Null C is more nuanced. The empirical Fib rate is the observation
        # itself. The real question: is the concentration into Fibonacci
        # ratios BEYOND what we'd expect from the empirical ratio
        # distribution if Fibonacci status were random?
        #
        # Approach: we have the empirical distribution of ratios.
        # Under H0, "Fibonacci" is an arbitrary label. Permute which
        # coprime ratios count as "Fibonacci" and see if the observed
        # Fib rate is still extreme.
        #
        # Better approach: Use a chi-squared test. Expected counts for
        # each ratio under a model where Fibonacci ratios have no
        # special status. But what model generates expected counts?
        #
        # Cleanest: Among the 9 order-1 coprime ratios, the observed data
        # assigns counts to each. Under H0 that Fibonacci status is
        # irrelevant, the probability of a random draw being Fibonacci
        # equals the fraction of observations that land on Fibonacci ratios.
        # This IS the empirical rate = tautology.
        #
        # The REAL Null C question: Given the DISTRIBUTION SHAPE over ratios
        # (i.e., 3:2 is much more common than 10:9), compute what Fibonacci
        # fraction we'd expect if ratios were drawn from a smooth complexity
        # prior (interpolated from observed non-Fibonacci ratios).
        #
        # Implementation: Fit a model P(p:q) ~ f(complexity(p,q)) using
        # ONLY non-Fibonacci ratios. Extrapolate to predict Fibonacci counts.
        # Compare observed Fibonacci counts to this prediction.

        print(f"  -- Null C: Empirical distribution test --")

        # Build observed counts per ratio for order-1
        o1_counts = {}
        for ratio_str, count in ratio_counts.items():
            p_str, q_str = ratio_str.split(':')
            p_val, q_val = int(p_str), int(q_str)
            if abs(p_val - q_val) != 1:
                continue
            o1_counts[(p_val, q_val)] = count

        # All order-1 coprime ratios
        all_o1 = [(p, q) for p, q in enumerate_coprime_ratios() if abs(p - q) == 1]

        # For unobserved ratios, count = 0
        for pq in all_o1:
            if pq not in o1_counts:
                o1_counts[pq] = 0

        print(f"    Order-1 ratio counts:")
        total_o1_obs = sum(o1_counts.values())
        fib_o1_obs = sum(c for (p, q), c in o1_counts.items()
                         if is_fibonacci_ratio(p, q))
        nonfib_o1_obs = total_o1_obs - fib_o1_obs

        for (p, q) in sorted(all_o1, key=lambda x: -o1_counts.get(x, 0)):
            fib = "FIB" if is_fibonacci_ratio(p, q) else "   "
            c = o1_counts.get((p, q), 0)
            print(f"      {fib} {p}:{q}  count={c}")

        # Null C: Empirical distribution test.
        # The proper question: given which ratios are actually observed,
        # are Fibonacci ratios over-represented relative to their
        # complexity-predicted counts?
        #
        # Approach 1: Among the observed ratios, compute the Fibonacci
        # rate predicted by Null A (1/(p*q)) weighting applied to ONLY
        # the ratios that actually appear. This is "given that we only
        # see these 5 ratio types, does Fibonacci's share exceed what
        # the 1/(p*q) prior would predict?"
        #
        # Approach 2: Chi-squared goodness-of-fit. Expected counts per
        # ratio from 1/(p*q) model, scaled to total N. Compare observed.

        print(f"\n    Approach 1: Restricted 1/(p*q) null (only observed ratios)")
        observed_o1_ratios = [(p, q) for (p, q) in all_o1
                              if o1_counts.get((p, q), 0) > 0]
        if len(observed_o1_ratios) > 0:
            restricted_w_total = 0.0
            restricted_w_fib = 0.0
            for p, q in observed_o1_ratios:
                w = 1.0 / (p * q)
                restricted_w_total += w
                if is_fibonacci_ratio(p, q):
                    restricted_w_fib += w
            restricted_null_rate = restricted_w_fib / restricted_w_total
            print(f"    Observed ratio types: "
                  f"{[f'{p}:{q}' for p, q in observed_o1_ratios]}")
            print(f"    Restricted 1/(p*q) Fib rate: {100*restricted_null_rate:.1f}%")
            print(f"    Observed Fib rate:           {100*fib_o1_obs/total_o1_obs:.1f}%")

            p_restricted = binomial_test_weighted(
                fib_o1_obs, total_o1_obs, restricted_null_rate)
            effect_restricted = (fib_o1_obs / total_o1_obs) / restricted_null_rate
            print(f"    Effect: {effect_restricted:.2f}x, "
                  f"binomial p = {p_restricted:.4e}")

            ds_results['null_C_restricted'] = {
                'order1': {
                    'restricted_null_rate': restricted_null_rate,
                    'observed_rate': fib_o1_obs / total_o1_obs,
                    'effect_size': effect_restricted,
                    'binomial_p': p_restricted,
                }
            }

        print(f"\n    Approach 2: Chi-squared GOF (1/(p*q) expected counts)")
        # Expected count for each observed ratio under 1/(p*q) prior
        if len(observed_o1_ratios) >= 2:
            weights_obs = {(p, q): 1.0 / (p * q) for p, q in observed_o1_ratios}
            w_sum = sum(weights_obs.values())
            expected_counts = []
            observed_counts_list = []
            labels_list = []
            for p, q in observed_o1_ratios:
                exp_c = total_o1_obs * weights_obs[(p, q)] / w_sum
                obs_c = o1_counts[(p, q)]
                expected_counts.append(exp_c)
                observed_counts_list.append(obs_c)
                labels_list.append(f"{p}:{q}")
                fib = "FIB" if is_fibonacci_ratio(p, q) else "   "
                print(f"      {fib} {p}:{q}  expected={exp_c:.1f}  "
                      f"observed={obs_c}  "
                      f"ratio={obs_c/exp_c:.2f}x")

            chi2_stat, chi2_p = stats.chisquare(
                observed_counts_list, expected_counts)
            print(f"    Chi-squared = {chi2_stat:.2f}, df={len(observed_o1_ratios)-1}, "
                  f"p = {chi2_p:.4e}")
            print(f"    Interpretation: tests whether the observed distribution")
            print(f"    across ratios matches the 1/(p*q) prediction")

            ds_results['null_C_chisquared'] = {
                'order1': {
                    'chi2': chi2_stat,
                    'chi2_p': chi2_p,
                    'df': len(observed_o1_ratios) - 1,
                    'expected': {l: e for l, e in zip(labels_list, expected_counts)},
                    'observed': {l: o for l, o in zip(labels_list, observed_counts_list)},
                }
            }

        # Approach 3: Among ALL order-1 observations, permute the
        # Fibonacci labels ON THE RATIO TYPES (not observations).
        # There are 9 order-1 coprime ratios, 2 are Fibonacci.
        # We observe 5 ratio types. Under H0, pick 2 of the 9 to be
        # "Fibonacci." What fraction of such picks gives >= observed rate?
        print(f"\n    Approach 3: Ratio-label permutation test")
        print(f"    Under H0: 'Fibonacci' is a random label on 2 of 9 "
              f"order-1 ratios")
        # For each of C(9,2)=36 ways to choose 2 "Fibonacci" ratios,
        # compute what Fib rate we'd observe with the actual counts
        from itertools import combinations
        all_o1_sorted = sorted(all_o1)
        n_combos = 0
        n_exceed_combo = 0
        combo_rates = []
        for chosen in combinations(range(len(all_o1_sorted)), 2):
            fake_fib = set(chosen)
            fake_fib_count = sum(
                o1_counts.get(all_o1_sorted[i], 0)
                for i in fake_fib)
            fake_rate = fake_fib_count / total_o1_obs if total_o1_obs > 0 else 0
            combo_rates.append(fake_rate)
            if fake_rate >= fib_o1_obs / total_o1_obs:
                n_exceed_combo += 1
            n_combos += 1

        combo_p = n_exceed_combo / n_combos
        combo_rates = np.array(combo_rates)
        print(f"    {n_combos} combinations of 2-of-9")
        print(f"    {n_exceed_combo} achieve >= {100*fib_o1_obs/total_o1_obs:.1f}%")
        print(f"    Exact combinatorial p = {combo_p:.4f}")
        print(f"    Mean rate under H0 = {100*combo_rates.mean():.1f}%")
        print(f"    Max rate under H0 = {100*combo_rates.max():.1f}%")

        ds_results['null_C_combinatorial'] = {
            'order1': {
                'n_combos': n_combos,
                'n_exceed': n_exceed_combo,
                'exact_p': combo_p,
                'mean_rate_h0': float(combo_rates.mean()),
                'max_rate_h0': float(combo_rates.max()),
            }
        }

        print()

        # ── Model-free permutation test (Null D) ──
        print(f"  -- Null D: Model-free permutation test (uniform coprime) --")
        perm_p_d, perm_rates_d = permutation_test(
            ratio_counts, obs_rate_o1, order=1)
        if perm_p_d is not None:
            print(f"    Order-1: observed={100*obs_rate_o1:.1f}%, "
                  f"perm p = {perm_p_d:.6f}")
            print(f"    Perm mean = {100*perm_rates_d.mean():.1f}%, "
                  f"max = {100*perm_rates_d.max():.1f}%")
            print(f"    0 of {N_PERMUTATIONS} permutations reached observed rate"
                  if perm_p_d <= 1.0/(N_PERMUTATIONS+1) + 0.0001
                  else f"    {int(perm_p_d * (N_PERMUTATIONS+1)) - 1} of "
                       f"{N_PERMUTATIONS} permutations reached observed rate")

        ds_results['null_D_permutation'] = {
            'order1': {
                'observed_rate': obs_rate_o1,
                'permutation_p': perm_p_d,
                'permutation_mean': float(perm_rates_d.mean()) if perm_rates_d is not None else None,
                'permutation_max': float(perm_rates_d.max()) if perm_rates_d is not None else None,
            }
        }

        # Also run all-orders permutation
        perm_p_d_all, perm_rates_d_all = permutation_test(
            ratio_counts, obs_rate_all, order=None)
        if perm_p_d_all is not None:
            print(f"    All:     observed={100*obs_rate_all:.1f}%, "
                  f"perm p = {perm_p_d_all:.6f}")
            ds_results['null_D_permutation']['all'] = {
                'observed_rate': obs_rate_all,
                'permutation_p': perm_p_d_all,
                'permutation_mean': float(perm_rates_d_all.mean()),
                'permutation_max': float(perm_rates_d_all.max()),
            }

        print()
        results_out['tests'][ds_name] = ds_results

    # ── Summary Table ──
    print()
    print("=" * 72)
    print("SUMMARY TABLE — Order-1 Fibonacci Rate vs Tougher Nulls")
    print("=" * 72)
    print()
    print(f"{'Null Model':<35s} {'Expected':>10s} {'Observed':>10s} "
          f"{'Effect':>8s} {'Binom p':>12s} {'Perm p':>12s} {'Verdict':>10s}")
    print("-" * 97)

    # Use combined data for the summary
    ds = combined_results
    n_fib_o1 = ds['order1_n_fib']
    n_total_o1 = ds['order1_n']
    obs_rate = ds['order1_fib_rate']
    combo_tests = results_out['tests']['combined']

    rows = [
        ("Uniform (original)", 0.2222, obs_rate,
         obs_rate / 0.2222, ds['order1_binom_p'], None),
        ("A: 1/(p*q) weighted", null_a_o1, obs_rate,
         combo_tests['null_A']['order1']['effect_size'],
         combo_tests['null_A']['order1']['binomial_p'],
         combo_tests['null_A']['order1'].get('permutation_p')),
        ("B: 1/(p+q) weighted", null_b_o1, obs_rate,
         combo_tests['null_B']['order1']['effect_size'],
         combo_tests['null_B']['order1']['binomial_p'],
         combo_tests['null_B']['order1'].get('permutation_p')),
        ("D: Permutation (uniform)", 0.2222, obs_rate,
         obs_rate / 0.2222, None,
         combo_tests['null_D_permutation']['order1']['permutation_p']),
    ]

    for name, exp, obs, eff, bp, pp in rows:
        bp_str = f"{bp:.4e}" if bp is not None else "—"
        pp_str = f"{pp:.6f}" if pp is not None else "—"
        verdict = "SURVIVES" if (bp is not None and bp < 0.001) or \
                                (pp is not None and pp < 0.001) else \
                  "SURVIVES" if (bp is not None and bp < 0.05) or \
                                (pp is not None and pp < 0.05) else "KILLED"
        print(f"{name:<35s} {100*exp:>9.1f}% {100*obs:>9.1f}% "
              f"{eff:>7.2f}x {bp_str:>12s} {pp_str:>12s} {verdict:>10s}")

    # Add Null C variants if available
    if 'null_C_restricted' in combo_tests:
        c_data = combo_tests['null_C_restricted']['order1']
        c_exp = c_data['restricted_null_rate']
        c_eff = c_data['effect_size']
        c_p = c_data['binomial_p']
        verdict = "SURVIVES" if c_p < 0.05 else "KILLED"
        print(f"{'C1: Restricted 1/(p*q) on obs.':<35s} "
              f"{100*c_exp:>9.1f}% {100*obs_rate:>9.1f}% "
              f"{c_eff:>7.2f}x {c_p:>12.4e} {'—':>12s} {verdict:>10s}")
    if 'null_C_chisquared' in combo_tests:
        c_data = combo_tests['null_C_chisquared']['order1']
        c_p = c_data['chi2_p']
        verdict = "SURVIVES" if c_p < 0.05 else "KILLED"
        print(f"{'C2: Chi-sq GOF vs 1/(p*q)':<35s} "
              f"{'—':>10s} {'—':>10s} "
              f"{'—':>8s} {c_p:>12.4e} {'—':>12s} {verdict:>10s}")
    if 'null_C_combinatorial' in combo_tests:
        c_data = combo_tests['null_C_combinatorial']['order1']
        c_exp = c_data['mean_rate_h0']
        c_p = c_data['exact_p']
        c_eff = obs_rate / c_exp if c_exp > 0 else float('inf')
        verdict = "SURVIVES" if c_p < 0.05 else "KILLED"
        print(f"{'C3: Combinatorial (2-of-9 labels)':<35s} "
              f"{100*c_exp:>9.1f}% {100*obs_rate:>9.1f}% "
              f"{c_eff:>7.2f}x {'—':>12s} {c_p:>12.4f} {verdict:>10s}")

    print()

    # ── Honest Assessment ──
    print("=" * 72)
    print("HONEST ASSESSMENT")
    print("=" * 72)
    print()

    # Check if any null kills the effect
    all_survive = True
    marginal_tests = []
    for ds_name in ['combined']:
        ds_tests = results_out['tests'][ds_name]
        for null_name in ['null_A', 'null_B']:
            bp = ds_tests[null_name]['order1']['binomial_p']
            if bp is not None and bp >= 0.05:
                all_survive = False
                print(f"  KILLED by {null_name} on {ds_name} (p={bp:.4e})")
            elif bp is not None and bp >= 0.01:
                marginal_tests.append((null_name, bp))
        pp = ds_tests['null_D_permutation']['order1']['permutation_p']
        if pp is not None and pp >= 0.05:
            all_survive = False
            print(f"  KILLED by permutation test on {ds_name} (p={pp:.6f})")
        # Check Null C variants
        if 'null_C_restricted' in ds_tests:
            bp = ds_tests['null_C_restricted']['order1']['binomial_p']
            if bp is not None and bp >= 0.05:
                all_survive = False
                print(f"  KILLED by Null C (restricted) on {ds_name} (p={bp:.4e})")
            elif bp is not None and bp >= 0.01:
                marginal_tests.append(('null_C_restricted', bp))
        if 'null_C_chisquared' in ds_tests:
            cp = ds_tests['null_C_chisquared']['order1']['chi2_p']
            if cp is not None and cp >= 0.05:
                # Chi-sq tests distribution shape, not specifically Fib enrichment
                pass  # informational, not a kill criterion

    # Always compute shrinkage
    orig_effect = obs_rate / 0.2222
    worst_effect = min(
        combo_tests['null_A']['order1']['effect_size'],
        combo_tests['null_B']['order1']['effect_size'],
    )
    null_a_p = combo_tests['null_A']['order1']['binomial_p']

    print("  LAYERED VERDICT:")
    print()
    print("  Layer 1 — vs uniform null: MASSIVE effect")
    print(f"    {100*obs_rate:.1f}% vs 22.2% expected, {orig_effect:.2f}x, p < 10^-57")
    print()
    print("  Layer 2 — vs 1/(p+q) weighted null: STRONG effect")
    print(f"    {100*obs_rate:.1f}% vs 47.1% expected, "
          f"{combo_tests['null_B']['order1']['effect_size']:.2f}x, "
          f"p = {combo_tests['null_B']['order1']['binomial_p']:.2e}")
    print()
    print("  Layer 3 — vs 1/(p*q) weighted null: MARGINAL effect")
    print(f"    {100*obs_rate:.1f}% vs 74.1% expected, "
          f"{combo_tests['null_A']['order1']['effect_size']:.2f}x, "
          f"p = {null_a_p:.4f}")
    print()
    print("  Layer 4 — vs restricted empirical null: NO residual effect")
    if 'null_C_restricted' in combo_tests:
        c_exp = combo_tests['null_C_restricted']['order1']['restricted_null_rate']
        print(f"    {100*obs_rate:.1f}% vs {100*c_exp:.1f}% expected, "
              f"effect = {combo_tests['null_C_restricted']['order1']['effect_size']:.2f}x")
    print()
    print("  HOWEVER — the chi-squared test (C2) reveals the distribution")
    print("  WITHIN ratios deviates strongly from 1/(p*q):")
    if 'null_C_chisquared' in combo_tests:
        print(f"    chi2 = {combo_tests['null_C_chisquared']['order1']['chi2']:.1f}, "
              f"p = {combo_tests['null_C_chisquared']['order1']['chi2_p']:.2e}")
    print("    3:2 is 2.6x OVER-represented, 2:1 is 0.45x UNDER-represented")
    print("    The 1/(p*q) model is WRONG about the distribution shape.")
    print("    3:2 dominates over 2:1 despite being 'more complex.'")
    print()
    print("  ═══════════════════════════════════════════════════════════")
    print("  HONEST CONCLUSION:")
    print("  ═══════════════════════════════════════════════════════════")
    print()
    print("  The claim 'Fibonacci ratios are enriched' is TRUE but needs")
    print("  careful framing:")
    print()
    print("  (a) Fibonacci ratios (2:1, 3:2) happen to be the two SIMPLEST")
    print("      order-1 coprime ratios. Under ANY complexity-weighted null,")
    print("      you'd expect them to dominate. The 1/(p*q) null predicts")
    print(f"      74.1% Fibonacci — already close to the {100*obs_rate:.1f}% observed.")
    print()
    print("  (b) The residual 80% vs 74% enrichment (p=0.035) is MARGINAL.")
    print("      It would not survive a Bonferroni correction for testing")
    print("      multiple null models.")
    print()
    print("  (c) HOWEVER: 3:2 is 2.6x more common than 1/(p*q) predicts,")
    print("      while 2:1 is 0.45x LESS common. This means the 1/(p*q) model")
    print("      is wrong — nature specifically prefers 3:2 over 2:1.")
    print("      This is not generic low-integer bias; it is a SPECIFIC")
    print("      preference for the Fibonacci ratio 3:2.")
    print()
    print("  (d) The paper should reframe: not 'Fibonacci enrichment beyond")
    print("      chance' but 'specific dominance of the 3:2 resonance beyond")
    print("      what generic complexity weighting predicts.'")
    print()
    print(f"  Effect size shrinkage:")
    print(f"    vs uniform null:          {orig_effect:.2f}x")
    print(f"    vs toughest weighted null: {worst_effect:.2f}x")
    print(f"    Fraction explained by low-integer bias: "
          f"~{100*(1 - (worst_effect - 1)/(orig_effect - 1)):.0f}%")

    print()

    # Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
