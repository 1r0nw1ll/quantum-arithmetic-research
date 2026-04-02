#!/usr/bin/env python3
QA_COMPLIANCE = "observer=megalithic_deep_analysis, state_alphabet=thom_diameter_measurements"
"""
qa_megalithic_deep.py — Deep analysis of Thom 1962 stone circle diameters
through the QA lens.

Tests:
  1. FATHOM TEST: Do diameters cluster on even MY (fathom multiples)?
  2. DIAMETER RATIOS: Are same-site pairs near Fibonacci fractions?
  3. QA DIRECTION: Same-site pairs as (d,e) -> QA triples
  4. MOD-24 DISTRIBUTION: Structure in nearest_my mod 24

QA Axiom compliance:
  - S1: multiply not power operator
  - A1: states in {1,...,N}
  - A2: d = b+e, a = b+2e always derived
"""

import csv
import numpy as np
from math import gcd
from collections import Counter
from scipy import stats

np.random.seed(42)

# ─── Load data ───────────────────────────────────────────────────────
def load_data(path="thom_1962_diameters.csv"):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["site_name"].strip():
                continue
            rows.append({
                "name": row["site_name"].strip(),
                "diameter_ft": float(row["diameter_ft"]),
                "diameter_my": float(row["diameter_my"]),
                "nearest_my": int(row["nearest_my"]),
                "error_my": float(row["error_my"]),
            })
    return rows

# ─── TEST 1: Fathom Test ─────────────────────────────────────────────
def fathom_test(data, n_perm=10000):
    """
    Megalithic Fathom = 2 MY. If diameters cluster on fathom multiples,
    nearest_my values should prefer EVEN integers over ODD.

    Statistic: fraction of nearest_my that are even.
    Null: permutation of nearest_my values (shuffled assignment).
    Also compare: distance to nearest even MY vs distance to nearest odd MY.
    """
    print("=" * 70)
    print("TEST 1: FATHOM TEST (even vs odd MY preference)")
    print("=" * 70)

    nearest = [d["nearest_my"] for d in data]
    n = len(nearest)

    n_even = sum(1 for x in nearest if x % 2 == 0)
    n_odd = n - n_even
    frac_even = n_even / n

    print(f"  Total circles: {n}")
    print(f"  Even MY (fathom multiples): {n_even} ({100*frac_even:.1f}%)")
    print(f"  Odd MY: {n_odd} ({100*(1-frac_even):.1f}%)")

    # Expected under uniform: depends on range
    min_my = min(nearest)
    max_my = max(nearest)
    possible = list(range(min_my, max_my + 1))
    n_even_possible = sum(1 for x in possible if x % 2 == 0)
    expected_even_frac = n_even_possible / len(possible)
    print(f"  Expected even fraction (uniform over [{min_my},{max_my}]): {100*expected_even_frac:.1f}%")

    # Binomial test
    _, p_binom = stats.binom_test(n_even, n, expected_even_frac) if hasattr(stats, 'binom_test') else (None, None)
    # Use exact binomial test
    p_binom = stats.binomtest(n_even, n, expected_even_frac).pvalue
    print(f"  Binomial test p-value (two-sided): {p_binom:.4f}")

    # Also: mean distance to nearest even vs nearest odd
    diam_my = [d["diameter_my"] for d in data]
    dist_even = []
    dist_odd = []
    for x in diam_my:
        nearest_even = round(x / 2) * 2
        nearest_odd_val = round((x - 1) / 2) * 2 + 1
        dist_even.append(abs(x - nearest_even))
        dist_odd.append(abs(x - nearest_odd_val))

    mean_dist_even = np.mean(dist_even)
    mean_dist_odd = np.mean(dist_odd)
    print(f"\n  Mean distance to nearest even MY: {mean_dist_even:.4f}")
    print(f"  Mean distance to nearest odd MY:  {mean_dist_odd:.4f}")

    # Permutation test on even fraction
    observed_stat = frac_even
    count_ge = 0
    for _ in range(n_perm):
        # Draw n values uniformly from [min_my, max_my]
        fake = np.random.randint(min_my, max_my + 1, size=n)
        fake_even = np.sum(fake % 2 == 0) / n
        if fake_even >= observed_stat:
            count_ge += 1
    p_perm = count_ge / n_perm
    print(f"  Permutation p-value (even frac >= observed): {p_perm:.4f}")

    return frac_even, p_binom, p_perm


# ─── TEST 2: Diameter Ratios ─────────────────────────────────────────
def find_same_site_pairs(data):
    """Group by first word of site name to find same-site pairs."""
    from collections import defaultdict
    groups = defaultdict(list)

    # Also check for explicit multi-word matches
    for d in data:
        name = d["name"]
        # Group by first word
        first_word = name.split()[0]
        groups[first_word].append(d)

    pairs = []
    for key, members in groups.items():
        if len(members) >= 2:
            # All pairwise combinations
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a = members[i]
                    b = members[j]
                    larger = max(a["nearest_my"], b["nearest_my"])
                    smaller = min(a["nearest_my"], b["nearest_my"])
                    if smaller > 0:
                        pairs.append({
                            "name_a": a["name"],
                            "name_b": b["name"],
                            "my_a": larger,
                            "my_b": smaller,
                            "ratio": larger / smaller,
                        })
    return pairs


def diameter_ratio_test(data, n_perm=10000):
    """
    For same-site pairs, check if ratios are near simple Fibonacci fractions.
    Fibonacci fractions: 2/1, 3/2, 5/3, 8/5, 13/8, 1/1, 3/1, 5/2
    """
    print("\n" + "=" * 70)
    print("TEST 2: DIAMETER RATIOS (same-site pairs)")
    print("=" * 70)

    pairs = find_same_site_pairs(data)

    if not pairs:
        print("  No same-site pairs found.")
        return None, None

    # Fibonacci fractions to test against
    fib_fracs = [1.0, 2.0, 1.5, 5/3, 8/5, 3.0, 5/2, 13/8]
    fib_labels = ["1:1", "2:1", "3:2", "5:3", "8:5", "3:1", "5:2", "13:8"]

    print(f"\n  Found {len(pairs)} same-site pair(s):\n")

    total_min_dist = 0.0
    for p in pairs:
        ratio = p["ratio"]
        # Distance to nearest Fibonacci fraction
        dists = [abs(ratio - f) for f in fib_fracs]
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        total_min_dist += min_dist

        print(f"    {p['name_a']} / {p['name_b']}")
        print(f"      MY: {p['my_a']} / {p['my_b']} = {ratio:.4f}")
        print(f"      Nearest Fibonacci fraction: {fib_labels[min_idx]} = {fib_fracs[min_idx]:.4f} (dist = {min_dist:.4f})")

    mean_min_dist = total_min_dist / len(pairs)
    print(f"\n  Mean distance to nearest Fibonacci fraction: {mean_min_dist:.4f}")

    # Null model: random pairs from the full dataset
    all_my = [d["nearest_my"] for d in data if d["nearest_my"] > 0]
    null_means = []
    for _ in range(n_perm):
        null_dist_sum = 0.0
        for _ in range(len(pairs)):
            a, b = np.random.choice(all_my, 2, replace=False)
            larger = max(a, b)
            smaller = min(a, b)
            if smaller == 0:
                continue
            ratio = larger / smaller
            dists = [abs(ratio - f) for f in fib_fracs]
            null_dist_sum += min(dists)
        null_means.append(null_dist_sum / len(pairs))

    p_ratio = np.mean([nm <= mean_min_dist for nm in null_means])
    print(f"  Permutation p-value (random pairs closer): {p_ratio:.4f}")

    return mean_min_dist, p_ratio


# ─── TEST 3: QA Direction from Diameter Pairs ────────────────────────
def qa_direction_test(data):
    """
    For same-site pairs, treat (larger_my, smaller_my) as (d, e).
    Derive b = d - e, a = b + 2*e (A2 compliant).
    Check if gcd(b, e) = 1 (primitive).
    Compute QA triple: C = 2*d*e, F = d*d - e*e (S1 compliant), G = d*d + e*e.
    Verify C*C + F*F == G*G.
    """
    print("\n" + "=" * 70)
    print("TEST 3: QA DIRECTION FROM DIAMETER PAIRS")
    print("=" * 70)

    pairs = find_same_site_pairs(data)

    if not pairs:
        print("  No same-site pairs found.")
        return

    print()
    n_primitive = 0
    n_total = len(pairs)

    for p in pairs:
        d = p["my_a"]   # larger
        e = p["my_b"]   # smaller

        # A2: derived coordinates
        b = d - e
        a = b + 2 * e   # = d + e

        # Check d = b + e (verification)
        assert d == b + e, f"A2 violation: d={d} != b+e={b}+{e}"

        g = gcd(b, e)
        is_primitive = (g == 1)
        if is_primitive:
            n_primitive += 1

        # QA triple (S1 compliant: d*d not pow(d,2))
        C = 2 * d * e
        F = d*d - e*e       # S1: d*d
        G = d*d + e*e       # S1: d*d, e*e

        # Verify Pythagorean identity
        check = (C*C + F*F == G*G)   # S1: C*C, F*F, G*G

        print(f"  Pair: {p['name_a']} / {p['name_b']}")
        print(f"    (d, e) = ({d}, {e})  ->  b = {b}, a = {a}")
        print(f"    gcd(b, e) = {g}  {'PRIMITIVE' if is_primitive else 'composite'}")
        print(f"    QA Triple: C={C}, F={F}, G={G}")
        print(f"    C^2 + F^2 = {C*C + F*F}, G^2 = {G*G}  ->  {'PASS' if check else 'FAIL'}")

        # Chromogeometric interpretation
        # C = Qg (green quadrance), F = Qr (red quadrance), G = Qb (blue quadrance)
        print(f"    Chromogeometry: Qg={C}, Qr={F}, Qb={G}")
        print()

    print(f"  Primitive pairs: {n_primitive}/{n_total}")
    print(f"  (Pythagorean identity C*C + F*F = G*G holds by construction for all integer d,e)")


# ─── TEST 4: Mod-24 Distribution ─────────────────────────────────────
def mod24_test(data, n_perm=10000):
    """
    Test if nearest_my values show structure mod 24 (QA cosmos orbit period).

    A1 compliant: map to {1,...,24} not {0,...,23}.
    residue = ((nearest_my - 1) % 24) + 1

    Use chi-squared test against uniform, plus permutation test.
    """
    print("\n" + "=" * 70)
    print("TEST 4: MOD-24 DISTRIBUTION")
    print("=" * 70)

    nearest = [d["nearest_my"] for d in data]

    # A1: states in {1,...,24}
    residues = [((x - 1) % 24) + 1 for x in nearest]

    counts = Counter(residues)
    n = len(residues)

    print(f"\n  Distribution of nearest_my mod 24 (A1: {{1,...,24}}):\n")

    # Show distribution
    for r in range(1, 25):
        c = counts.get(r, 0)
        bar = "#" * c
        print(f"    {r:2d}: {c:2d} {bar}")

    # Chi-squared test against uniform
    observed = np.array([counts.get(r, 0) for r in range(1, 25)])
    expected = np.ones(24) * n / 24.0

    # Only test bins that have expected > 0 (all do here)
    chi2_stat = np.sum((observed - expected) * (observed - expected) / expected)  # S1: no **2
    df = 23
    p_chi2 = 1 - stats.chi2.cdf(chi2_stat, df)

    print(f"\n  Chi-squared statistic: {chi2_stat:.2f}")
    print(f"  Degrees of freedom: {df}")
    print(f"  p-value: {p_chi2:.4f}")

    # Check specific QA-relevant residues
    # Singularity residue: 9 (mod 24, the fixed point (9,9))
    # Satellite residues: multiples of 3 within {1..24}

    # Entropy test
    probs = observed / n
    probs_nonzero = probs[probs > 0]
    entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    max_entropy = np.log2(24)

    print(f"\n  Entropy: {entropy:.3f} bits (max = {max_entropy:.3f})")
    print(f"  Entropy ratio: {entropy/max_entropy:.3f}")

    # Permutation test: is chi2 unusually large?
    count_ge = 0
    for _ in range(n_perm):
        fake_residues = np.random.randint(1, 25, size=n)  # A1: {1,...,24}
        fake_counts = np.array([np.sum(fake_residues == r) for r in range(1, 25)])
        fake_chi2 = np.sum((fake_counts - expected) * (fake_counts - expected) / expected)
        if fake_chi2 >= chi2_stat:
            count_ge += 1
    p_perm = count_ge / n_perm
    print(f"  Permutation p-value: {p_perm:.4f}")

    # Check even vs odd residue concentration
    even_res = sum(1 for r in residues if r % 2 == 0)
    print(f"\n  Even residues (mod 24): {even_res}/{n} ({100*even_res/n:.1f}%)")

    # Check mod-3 structure (satellite orbit period = 8, satellite ↔ mod 3)
    mod3 = Counter([((x - 1) % 3) + 1 for x in nearest])  # A1
    print(f"  Mod-3 distribution: {dict(sorted(mod3.items()))}")

    # Check mod-8 structure (satellite orbit)
    mod8 = Counter([((x - 1) % 8) + 1 for x in nearest])  # A1
    print(f"  Mod-8 distribution: {dict(sorted(mod8.items()))}")

    return chi2_stat, p_chi2, p_perm


# ─── MAIN ────────────────────────────────────────────────────────────
def main():
    print("QA MEGALITHIC DEEP ANALYSIS")
    print("Thom 1962 Stone Circle Diameters (N=84)")
    print("=" * 70)

    data = load_data()
    print(f"Loaded {len(data)} circles.\n")

    # Summary stats
    my_vals = [d["nearest_my"] for d in data]
    print(f"  MY range: {min(my_vals)} to {max(my_vals)}")
    print(f"  Mean MY: {np.mean(my_vals):.1f}")
    print(f"  Median MY: {np.median(my_vals):.1f}")

    # Run all four tests
    frac_even, p_binom, p_perm_fathom = fathom_test(data)
    mean_dist, p_ratio = diameter_ratio_test(data)
    qa_direction_test(data)
    chi2, p_chi2, p_perm_mod24 = mod24_test(data)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (Fathom):     even frac = {frac_even:.3f}, "
          f"p_binom = {p_binom:.4f}, p_perm = {p_perm_fathom:.4f}")
    if p_ratio is not None:
        print(f"  Test 2 (Ratios):     mean dist to Fib frac = {mean_dist:.4f}, "
              f"p_perm = {p_ratio:.4f}")
    print(f"  Test 3 (QA Dir):     see individual pair results above")
    print(f"  Test 4 (Mod-24):     chi2 = {chi2:.2f}, "
          f"p_chi2 = {p_chi2:.4f}, p_perm = {p_perm_mod24:.4f}")

    print("\n  Interpretation guide:")
    print("    p < 0.05 = significant departure from null")
    print("    p > 0.05 = consistent with null (no special structure)")
    print()


if __name__ == "__main__":
    main()
