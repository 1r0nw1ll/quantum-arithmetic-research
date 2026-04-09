#!/usr/bin/env python3
"""
Adversarial Out-of-Sample Replication of QA Fibonacci Resonance [163]
=====================================================================

Original finding: 77% of order-1 orbital mean-motion resonances are
Fibonacci ratios (p<10^-6) across 60 resonances from 8 systems.

This script fetches FRESH data from the NASA Exoplanet Archive,
computes period ratios for ALL multi-planet systems, separates
original vs new systems, and tests Fibonacci enrichment on NEW data only.

Observer Projection Note (Theorem NT):
  Orbital periods are continuous measurements. We project them to
  discrete integer ratios (p:q) — this is an observer projection,
  not a QA causal input. The QA classification (Fibonacci vs non-Fib)
  operates on the discrete ratios only.

S1 compliance: uses b*b not b**2 throughout.
"""

import csv
import io
import json
import os
import sys
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from math import gcd, log

import numpy as np
np.random.seed(42)

from scipy import stats

# ── Configuration ──────────────────────────────────────────────────

RESULTS_DIR = "/home/player2/signal_experiments/results/fibonacci_adversarial"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Tolerance for resonance identification: |ratio - p/q| < TOL
RESONANCE_TOL = 0.05

# Maximum integers in ratio p:q
MAX_PQ = 10

# Fibonacci numbers up to MAX_PQ (and beyond for classification)
FIB_SET = set()
a, b = 1, 1
while a <= 100:
    FIB_SET.add(a)
    a, b = b, a + b
# FIB_SET = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}

# Original 8 systems used in the paper
ORIGINAL_SYSTEMS = {
    "TRAPPIST-1",
    "HD 110067",
    "K2-138",
    "Kepler-223",
    "Kepler-80",
    "TOI-178",
    "GJ 876",
    "HD 158259",
}

# TAP query URL
TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=SELECT+pl_name,hostname,pl_orbper,pl_orbpererr1,"
    "pl_orbpererr2,sy_pnum+FROM+pscomppars+"
    "WHERE+sy_pnum+>=+3+AND+pl_orbper+IS+NOT+NULL+"
    "ORDER+BY+hostname,pl_orbper&format=csv"
)

# ── Helper Functions ───────────────────────────────────────────────

def is_fibonacci(n):
    """Check if integer n is a Fibonacci number."""
    return n in FIB_SET


def is_fibonacci_ratio(p, q):
    """A ratio p:q is Fibonacci iff BOTH p and q are Fibonacci numbers."""
    return is_fibonacci(p) and is_fibonacci(q)


def find_closest_ratio(value, max_pq=MAX_PQ, tol=RESONANCE_TOL):
    """
    Find the closest simple integer ratio p:q to value,
    where p > q, gcd(p,q)=1, and both <= max_pq.
    Returns (p, q, residual) or None if nothing within tolerance.
    """
    best = None
    best_resid = tol  # must beat tolerance

    for q in range(1, max_pq + 1):
        for p in range(q + 1, max_pq + 1):
            if gcd(p, q) != 1:
                continue
            ratio_pq = p / q
            resid = abs(value - ratio_pq)
            if resid < best_resid:
                best_resid = resid
                best = (p, q, resid)

    return best


def resonance_order(p, q):
    """Order of a resonance = |p - q|. Order-1 = adjacent."""
    return abs(p - q)


def enumerate_coprime_ratios(max_pq=MAX_PQ):
    """Enumerate all coprime ratios p:q with p > q, both <= max_pq."""
    ratios = []
    for q in range(1, max_pq + 1):
        for p in range(q + 1, max_pq + 1):
            if gcd(p, q) != 1:
                continue
            ratios.append((p, q))
    return ratios


def compute_null_fibonacci_rate(order=None, max_pq=MAX_PQ):
    """
    Under uniform distribution over coprime ratios, what fraction are Fibonacci?
    If order is specified, restrict to that order.
    """
    ratios = enumerate_coprime_ratios(max_pq)
    if order is not None:
        ratios = [(p, q) for p, q in ratios if abs(p - q) == order]
    n_total = len(ratios)
    n_fib = sum(1 for p, q in ratios if is_fibonacci_ratio(p, q))
    return n_fib, n_total, n_fib / n_total if n_total > 0 else 0.0


def fetch_exoplanet_data():
    """Fetch data from NASA Exoplanet Archive TAP service."""
    print(f"Fetching data from NASA Exoplanet Archive...")
    print(f"URL: {TAP_URL[:80]}...")

    try:
        req = urllib.request.Request(TAP_URL)
        req.add_header('User-Agent', 'QA-Fibonacci-Replication/1.0')
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode('utf-8')
        print(f"  Received {len(raw)} bytes")
        return raw
    except urllib.error.URLError as e:
        print(f"  ERROR: Network fetch failed: {e}")
        print(f"  Trying with different timeout...")
        try:
            req = urllib.request.Request(TAP_URL)
            req.add_header('User-Agent', 'QA-Fibonacci-Replication/1.0')
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read().decode('utf-8')
            print(f"  Received {len(raw)} bytes on retry")
            return raw
        except Exception as e2:
            print(f"  FATAL: Could not fetch data: {e2}")
            sys.exit(1)


def parse_exoplanet_csv(raw_csv):
    """Parse the CSV into a dict of systems -> list of planets."""
    reader = csv.DictReader(io.StringIO(raw_csv))
    systems = defaultdict(list)

    for row in reader:
        hostname = row['hostname'].strip()
        try:
            period = float(row['pl_orbper'])
        except (ValueError, TypeError):
            continue

        err1 = None
        err2 = None
        try:
            err1 = float(row['pl_orbpererr1']) if row['pl_orbpererr1'] else None
        except (ValueError, TypeError):
            pass
        try:
            err2 = float(row['pl_orbpererr2']) if row['pl_orbpererr2'] else None
        except (ValueError, TypeError):
            pass

        systems[hostname].append({
            'name': row['pl_name'].strip(),
            'hostname': hostname,
            'period': period,
            'period_err_plus': err1,
            'period_err_minus': err2,
            'sy_pnum': int(row['sy_pnum']) if row['sy_pnum'] else None,
        })

    # Sort planets within each system by period
    for hostname in systems:
        systems[hostname].sort(key=lambda x: x['period'])

    return dict(systems)


def compute_period_ratios(systems):
    """
    For each system, compute adjacent-pair period ratios P_{i+1}/P_i.
    Returns list of ratio dicts.
    """
    all_ratios = []

    for hostname, planets in systems.items():
        if len(planets) < 2:
            continue

        for i in range(len(planets) - 1):
            p_inner = planets[i]['period']
            p_outer = planets[i + 1]['period']

            if p_inner <= 0:
                continue

            ratio_val = p_outer / p_inner

            # Skip ratios that are way too large (not meaningful resonances)
            if ratio_val > MAX_PQ + 0.5:
                continue

            match = find_closest_ratio(ratio_val)

            all_ratios.append({
                'hostname': hostname,
                'inner_planet': planets[i]['name'],
                'outer_planet': planets[i + 1]['name'],
                'inner_period': p_inner,
                'outer_period': p_outer,
                'ratio': ratio_val,
                'matched_p': match[0] if match else None,
                'matched_q': match[1] if match else None,
                'residual': match[2] if match else None,
                'is_resonance': match is not None,
                'is_original_system': hostname in ORIGINAL_SYSTEMS,
            })

    return all_ratios


def classify_ratios(ratios):
    """Add Fibonacci and order classification to each ratio."""
    for r in ratios:
        if r['is_resonance']:
            p, q = r['matched_p'], r['matched_q']
            r['order'] = resonance_order(p, q)
            r['is_fibonacci'] = is_fibonacci_ratio(p, q)
            r['ratio_str'] = f"{p}:{q}"
        else:
            r['order'] = None
            r['is_fibonacci'] = None
            r['ratio_str'] = None
    return ratios


def run_statistical_tests(ratios, label=""):
    """Run binomial and Fisher tests on a set of classified resonance ratios."""
    results = {}

    resonances = [r for r in ratios if r['is_resonance']]
    results['n_pairs_total'] = len(ratios)
    results['n_resonances'] = len(resonances)

    if len(resonances) == 0:
        print(f"  [{label}] No resonances found — cannot test.")
        results['verdict'] = 'NO_DATA'
        return results

    # ── Order-1 analysis ──
    order1 = [r for r in resonances if r['order'] == 1]
    n_order1 = len(order1)
    n_fib_order1 = sum(1 for r in order1 if r['is_fibonacci'])
    fib_rate_order1 = n_fib_order1 / n_order1 if n_order1 > 0 else 0.0

    # Null rate for order-1
    n_fib_null, n_total_null, null_rate_order1 = compute_null_fibonacci_rate(order=1)
    print(f"  [{label}] Order-1: {n_fib_order1}/{n_order1} Fibonacci "
          f"({100*fib_rate_order1:.1f}%) vs null {100*null_rate_order1:.1f}%")
    print(f"    Null: {n_fib_null}/{n_total_null} coprime order-1 ratios are Fibonacci")

    results['order1_n'] = n_order1
    results['order1_n_fib'] = n_fib_order1
    results['order1_fib_rate'] = fib_rate_order1
    results['order1_null_rate'] = null_rate_order1

    if n_order1 > 0:
        # One-sided binomial test: observed >= expected under H0
        binom_result = stats.binomtest(n_fib_order1, n_order1, null_rate_order1,
                                       alternative='greater')
        binom_p = binom_result.pvalue
        results['order1_binom_p'] = binom_p
        print(f"    Binomial test p = {binom_p:.6e}")
    else:
        results['order1_binom_p'] = None

    # ── All-orders analysis ──
    n_fib_all = sum(1 for r in resonances if r['is_fibonacci'])
    fib_rate_all = n_fib_all / len(resonances) if len(resonances) > 0 else 0.0
    _, _, null_rate_all = compute_null_fibonacci_rate(order=None)

    results['all_n'] = len(resonances)
    results['all_n_fib'] = n_fib_all
    results['all_fib_rate'] = fib_rate_all
    results['all_null_rate'] = null_rate_all

    print(f"  [{label}] All orders: {n_fib_all}/{len(resonances)} Fibonacci "
          f"({100*fib_rate_all:.1f}%) vs null {100*null_rate_all:.1f}%")

    if len(resonances) > 0:
        binom_p_all = stats.binomtest(n_fib_all, len(resonances), null_rate_all,
                                       alternative='greater').pvalue
        results['all_binom_p'] = binom_p_all
        print(f"    Binomial test p = {binom_p_all:.6e}")
    else:
        results['all_binom_p'] = None

    # ── Per-order stratified analysis + Fisher combined ──
    order_ps = []
    results['per_order'] = {}
    for order in sorted(set(r['order'] for r in resonances)):
        subset = [r for r in resonances if r['order'] == order]
        n_sub = len(subset)
        n_fib_sub = sum(1 for r in subset if r['is_fibonacci'])
        _, _, null_sub = compute_null_fibonacci_rate(order=order)

        if n_sub > 0 and null_sub > 0:
            p_sub = stats.binomtest(n_fib_sub, n_sub, null_sub,
                                     alternative='greater').pvalue
            order_ps.append(p_sub)
            results['per_order'][order] = {
                'n': n_sub, 'n_fib': n_fib_sub,
                'rate': n_fib_sub / n_sub,
                'null_rate': null_sub, 'p': p_sub
            }
            print(f"    Order {order}: {n_fib_sub}/{n_sub} "
                  f"({100*n_fib_sub/n_sub:.1f}%) p={p_sub:.4e}")

    # Fisher combined test across orders
    if len(order_ps) >= 2:
        # Fisher's method: -2 * sum(ln(p)) ~ chi2(2k)
        chi2_stat = -2 * sum(log(max(p, 1e-300)) for p in order_ps)
        fisher_p = stats.chi2.sf(chi2_stat, 2 * len(order_ps))
        results['fisher_chi2'] = chi2_stat
        results['fisher_p'] = fisher_p
        results['fisher_df'] = 2 * len(order_ps)
        print(f"    Fisher combined: chi2={chi2_stat:.2f}, "
              f"df={2*len(order_ps)}, p={fisher_p:.6e}")

    # ── Ratio frequency table ──
    ratio_counts = defaultdict(int)
    for r in resonances:
        ratio_counts[r['ratio_str']] += 1
    results['ratio_counts'] = dict(ratio_counts)

    return results


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("QA Fibonacci Resonance — Adversarial Out-of-Sample Replication")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Tolerance: {RESONANCE_TOL}")
    print(f"Max p,q: {MAX_PQ}")
    print(f"Fibonacci set (up to {MAX_PQ}): {sorted(n for n in FIB_SET if n <= MAX_PQ)}")
    print()

    # ── 0. Enumerate null rates ──
    print("── Null Model (uniform over coprime ratios) ──")
    for order in [1, 2, 3, None]:
        n_f, n_t, rate = compute_null_fibonacci_rate(order=order)
        label = f"Order {order}" if order else "All orders"
        print(f"  {label}: {n_f}/{n_t} = {100*rate:.1f}% Fibonacci")
    print()

    # ── 1. Fetch data ──
    raw_csv = fetch_exoplanet_data()

    # Save raw data
    raw_path = os.path.join(RESULTS_DIR, "raw_nasa_tap.csv")
    with open(raw_path, 'w') as f:
        f.write(raw_csv)
    print(f"  Saved raw data to {raw_path}")
    print()

    # ── 2. Parse ──
    systems = parse_exoplanet_csv(raw_csv)
    print(f"── Parsed Data ──")
    print(f"  Total systems with 3+ planets and known periods: {len(systems)}")
    total_planets = sum(len(v) for v in systems.values())
    print(f"  Total planets: {total_planets}")

    # Identify original vs new
    original_systems = {h: p for h, p in systems.items() if h in ORIGINAL_SYSTEMS}
    new_systems = {h: p for h, p in systems.items() if h not in ORIGINAL_SYSTEMS}
    print(f"  Original systems found: {len(original_systems)} "
          f"(expected 8): {sorted(original_systems.keys())}")
    missing = ORIGINAL_SYSTEMS - set(original_systems.keys())
    if missing:
        print(f"  WARNING: Missing original systems: {missing}")
    print(f"  New systems: {len(new_systems)}")
    print()

    # ── 3. Compute ratios ──
    all_ratios = compute_period_ratios(systems)
    classify_ratios(all_ratios)

    original_ratios = [r for r in all_ratios if r['is_original_system']]
    new_ratios = [r for r in all_ratios if not r['is_original_system']]

    print(f"── Period Ratios ──")
    print(f"  Total adjacent pairs: {len(all_ratios)}")
    print(f"  Near-resonance pairs (within tol={RESONANCE_TOL}): "
          f"{sum(1 for r in all_ratios if r['is_resonance'])}")
    print(f"  Original system pairs: {len(original_ratios)} "
          f"(resonances: {sum(1 for r in original_ratios if r['is_resonance'])})")
    print(f"  New system pairs: {len(new_ratios)} "
          f"(resonances: {sum(1 for r in new_ratios if r['is_resonance'])})")
    print()

    # ── 4. Statistical tests ──
    print("── ORIGINAL DATA (sanity check — should match paper) ──")
    orig_results = run_statistical_tests(original_ratios, label="ORIG")
    print()

    print("── NEW DATA (out-of-sample test) ──")
    new_results = run_statistical_tests(new_ratios, label="NEW")
    print()

    print("── COMBINED (all data) ──")
    combined_results = run_statistical_tests(all_ratios, label="ALL")
    print()

    # ── 5. Show new system resonance details ──
    print("── New System Resonance Details (order-1 only) ──")
    new_order1 = [r for r in new_ratios if r['is_resonance'] and r.get('order') == 1]
    for r in sorted(new_order1, key=lambda x: x['hostname']):
        fib_mark = "FIB" if r['is_fibonacci'] else "   "
        print(f"  {fib_mark} {r['hostname']:25s} "
              f"{r['inner_planet']:30s} → {r['outer_planet']:30s} "
              f"P={r['ratio']:.4f} ≈ {r['ratio_str']} "
              f"(resid={r['residual']:.4f})")
    print()

    # Also show all new resonances
    print("── New System Resonance Details (all orders) ──")
    new_res = [r for r in new_ratios if r['is_resonance']]
    for r in sorted(new_res, key=lambda x: (x['hostname'], x['ratio'])):
        fib_mark = "FIB" if r['is_fibonacci'] else "   "
        print(f"  {fib_mark} {r['hostname']:25s} "
              f"{r['inner_planet']:30s} → {r['outer_planet']:30s} "
              f"P={r['ratio']:.4f} ≈ {r['ratio_str']} "
              f"ord={r['order']} (resid={r['residual']:.4f})")
    print()

    # ── 6. Verdict ──
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if new_results.get('order1_n', 0) == 0:
        print("INSUFFICIENT DATA: No order-1 resonances found in new systems.")
        verdict = "INSUFFICIENT_DATA"
    else:
        obs = new_results['order1_fib_rate']
        p_val = new_results.get('order1_binom_p')
        orig_rate = 0.77  # from paper

        if p_val is not None and p_val < 0.05 and obs > 0.5:
            verdict = "REPLICATES"
            print(f"REPLICATES: Out-of-sample order-1 Fibonacci rate = "
                  f"{100*obs:.1f}% (p={p_val:.4e})")
            print(f"  Original claim: 77%. New data: {100*obs:.1f}%.")
        elif p_val is not None and p_val < 0.05 and obs > new_results['order1_null_rate']:
            verdict = "PARTIAL_REPLICATION"
            print(f"PARTIAL REPLICATION: Fibonacci enrichment is significant "
                  f"(p={p_val:.4e}) but rate {100*obs:.1f}% is below 77%.")
        else:
            verdict = "FAILS"
            if p_val is not None:
                print(f"FAILS TO REPLICATE: Out-of-sample order-1 Fibonacci rate = "
                      f"{100*obs:.1f}% (p={p_val:.4e})")
            else:
                print(f"FAILS TO REPLICATE: Insufficient data for test.")
            print(f"  Null expectation: {100*new_results['order1_null_rate']:.1f}%. "
                  f"Observed: {100*obs:.1f}%.")

    print()
    print(f"Summary:")
    print(f"  New systems tested: {len(new_systems)}")
    print(f"  New adjacent pairs: {len(new_ratios)}")
    new_res_count = sum(1 for r in new_ratios if r['is_resonance'])
    print(f"  New resonance pairs: {new_res_count}")
    new_o1 = new_results.get('order1_n', 0)
    new_o1_fib = new_results.get('order1_n_fib', 0)
    print(f"  New order-1 resonances: {new_o1} ({new_o1_fib} Fibonacci)")
    if new_results.get('order1_binom_p') is not None:
        print(f"  Order-1 binomial p-value: {new_results['order1_binom_p']:.6e}")
    if new_results.get('fisher_p') is not None:
        print(f"  Fisher combined p-value: {new_results['fisher_p']:.6e}")
    print(f"  Verdict: {verdict}")

    # ── 7. Save results ──
    output = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'tolerance': RESONANCE_TOL,
            'max_pq': MAX_PQ,
            'original_systems': sorted(ORIGINAL_SYSTEMS),
            'n_new_systems': len(new_systems),
            'n_new_pairs': len(new_ratios),
            'n_new_resonances': new_res_count,
        },
        'null_model': {
            'order1': compute_null_fibonacci_rate(order=1),
            'all': compute_null_fibonacci_rate(order=None),
        },
        'original_results': {k: v for k, v in orig_results.items()
                             if not isinstance(v, float) or np.isfinite(v)},
        'new_results': {k: v for k, v in new_results.items()
                        if not isinstance(v, float) or np.isfinite(v)},
        'combined_results': {k: v for k, v in combined_results.items()
                             if not isinstance(v, float) or np.isfinite(v)},
        'verdict': verdict,
        'new_resonances': [
            {
                'system': r['hostname'],
                'inner': r['inner_planet'],
                'outer': r['outer_planet'],
                'ratio': r['ratio'],
                'matched': r['ratio_str'],
                'order': r['order'],
                'is_fibonacci': r['is_fibonacci'],
                'residual': r['residual'],
            }
            for r in sorted(new_res, key=lambda x: x['hostname'])
        ],
    }

    results_path = os.path.join(RESULTS_DIR, "replication_results.json")
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save per-system summary
    summary_path = os.path.join(RESULTS_DIR, "system_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hostname', 'n_planets', 'n_pairs', 'n_resonances',
                          'n_order1', 'n_fib_order1', 'is_original'])
        for hostname in sorted(systems.keys()):
            sys_ratios = [r for r in all_ratios if r['hostname'] == hostname]
            sys_res = [r for r in sys_ratios if r['is_resonance']]
            sys_o1 = [r for r in sys_res if r['order'] == 1]
            sys_fib_o1 = [r for r in sys_o1 if r['is_fibonacci']]
            writer.writerow([
                hostname,
                len(systems[hostname]),
                len(sys_ratios),
                len(sys_res),
                len(sys_o1),
                len(sys_fib_o1),
                hostname in ORIGINAL_SYSTEMS,
            ])
    print(f"System summary saved to {summary_path}")

    return verdict


if __name__ == "__main__":
    verdict = main()
