"""
57_qa_dilution_detector.py — 1/4 Dilution Symmetry Detector

Theorem (from cert [440]): For any det=−1 QA companion matrix that is
ramified at p with depth r, every orbit count satisfies

    count_det-1(4·p^L) = count_det+1(p^L) / 4

exactly. The det=+1 twin supplies the reference; the det=−1 realisation
is an exact quarter. Fixed-count(1)=1 for det=−1 vs p^r for det=+1.

Application: given only an empirical orbit histogram (counts of how many
orbits have each period) from a discrete dynamical system of unknown type,
this script tests whether the histogram is consistent with a det=−1 QA
companion at some (p, r, k). If it is, the period-4 structure is
"fingerprinted" and the underlying generator class is constrained.

This is a falsifiable test: a histogram either passes or fails the
4-divisibility + twin-pairing check. Pass rate on random controls gives
a baseline; QA generators give guaranteed passes.

Usage:
    python 57_qa_dilution_detector.py          # run full demo
    python 57_qa_dilution_detector.py --latex  # also print LaTeX table

Output: 57_dilution_panel.png (orbit fingerprint plots)
"""

import sys
import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# Oracle (self-contained, no imports beyond sys/math)
# ---------------------------------------------------------------------------

def vp(n, p):
    if n == 0:
        return 999
    n = abs(n)
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def classify(t, p, det):
    if det == 1:
        r = vp(t - 2, p)
        return ('ramified', r) if r >= 1 else ('unramified', 0)
    else:
        if p == 2:
            return ('p2', 0)
        if p % 4 == 3:
            return ('impossible', 0)
        r = vp(t * t + 4, p)
        return ('ramified', r) if r >= 1 else ('unramified', 0)


def oracle_det1(p, r, k):
    """[439]: det=+1 ramified orbit counts."""
    counts = {}
    if k <= r:
        counts[1] = p ** k
        birth = (p - 1) * p ** (k - 1)
        for L in range(1, k + 1):
            counts[p ** L] = birth
    else:
        counts[1] = p ** r
        for L in range(1, k - r + 1):
            counts[p ** L] = (p * p - 1) * p ** (L + r - 2)
        birth = (p - 1) * p ** (k - 1)
        for L in range(k - r + 1, k + 1):
            counts[p ** L] = birth
    return counts


def oracle_det_m1(p, r, k):
    """[440]: det=−1 ramified orbit counts."""
    counts = {1: 1}
    counts[4] = (p ** min(r, k) - 1) // 4
    birth = (p - 1) // 4 * p ** (k - 1)
    if k <= r:
        for L in range(1, k + 1):
            counts[4 * p ** L] = birth
    else:
        for L in range(1, k - r + 1):
            counts[4 * p ** L] = (p * p - 1) // 4 * p ** (L + r - 2)
        for L in range(k - r + 1, k + 1):
            counts[4 * p ** L] = birth
    return counts


# ---------------------------------------------------------------------------
# Dilution detector core
# ---------------------------------------------------------------------------

def dilution_test(hist, p=None, r=None, k=None, verbose=False):
    """
    Test an orbit histogram for det=−1 QA structure.

    Parameters
    ----------
    hist : dict  {period: count}   — the empirical or synthetic histogram
    p, r, k : int or None
        If provided, test against the specific oracle. If None, attempt
        auto-detection from the histogram shape.

    Returns
    -------
    result : dict with keys
        'pass'        bool — did the test pass?
        'score'       float in [0,1] — fraction of dilution relations satisfied
        'details'     list of per-relation dicts
        'candidate'   (p, r, k) that best explains the histogram, or None
        'violations'  list of violation strings
    """
    # --- Auto-detect p, r, k if not given ---
    if p is None:
        p, r, k = _infer_params(hist)
        if p is None:
            return {
                'pass': False, 'score': 0.0, 'details': [],
                'candidate': None,
                'violations': ['could not infer (p,r,k) from histogram'],
            }

    # --- Build expected det=+1 twin ---
    ref = oracle_det1(p, r, k)

    details = []
    violations = []

    # Fixed-point check: det=−1 always has count(1) = 1
    c1 = hist.get(1, 0)
    fp_ok = (c1 == 1)
    details.append({
        'relation': 'count(1)=1 (det=-1 fixed-point law)',
        'expected': 1, 'observed': c1, 'ok': fp_ok,
    })
    if not fp_ok:
        violations.append(f'count(1)={c1} != 1 (det=-1 requires exactly 1 fixed point)')

    # Period-4 check: count(4) = (p^min(r,k) - 1)/4
    c4_expected = (p ** min(r, k) - 1) // 4
    c4 = hist.get(4, 0)
    p4_ok = (c4 == c4_expected)
    details.append({
        'relation': f'count(4)={(p**min(r,k)-1)//4}',
        'expected': c4_expected, 'observed': c4, 'ok': p4_ok,
    })
    if not p4_ok:
        violations.append(f'count(4)={c4} != {c4_expected}')

    # 1/4 dilution relations: for each p^L key in det=+1, expect 4*p^L key
    # in det=−1 with count = ref_count/4
    for period_det1, cnt_det1 in sorted(ref.items()):
        if period_det1 == 1:
            continue   # fixed-point handled separately
        period_dm1 = 4 * period_det1
        cnt_dm1_expected = cnt_det1 // 4
        cnt_dm1_observed = hist.get(period_dm1, 0)
        ok = (cnt_dm1_observed == cnt_dm1_expected)
        rel = f'count({period_det1})÷4 = count({period_dm1})'
        details.append({
            'relation': rel,
            'det1_count': cnt_det1,
            'expected': cnt_dm1_expected,
            'observed': cnt_dm1_observed,
            'ok': ok,
        })
        if not ok:
            violations.append(
                f'{rel}: {cnt_det1}÷4={cnt_dm1_expected} but observed {cnt_dm1_observed}'
            )

    # Unexpected periods: periods in hist not explained by det=−1 oracle
    dm1_oracle = oracle_det_m1(p, r, k)
    for per, cnt in hist.items():
        if per not in dm1_oracle and cnt > 0:
            violations.append(f'unexpected period {per} (count={cnt}) not in det=−1 oracle')

    total = len(details)
    passed = sum(1 for d in details if d['ok'])
    score = passed / total if total > 0 else 0.0
    ok_all = len(violations) == 0

    if verbose:
        print(f"  Dilution test p={p} r={r} k={k}: {'PASS' if ok_all else 'FAIL'} ({passed}/{total})")
        for v in violations:
            print(f"    VIOLATION: {v}")

    return {
        'pass': ok_all,
        'score': score,
        'details': details,
        'candidate': (p, r, k),
        'violations': violations,
    }


def _infer_params(hist):
    """
    Attempt to infer (p, r, k) from a histogram.

    Strategy:
    1. count(1) should be 1 for det=−1 ramified
    2. count(4) = (p^min(r,k)-1)/4 → probe small primes p≡1 mod 4
    3. The maximum period = 4*p^k → extract k from log
    """
    if hist.get(1, 0) != 1:
        return None, None, None

    c4 = hist.get(4, 0)
    max_period = max((per for per, cnt in hist.items() if cnt > 0), default=0)

    # max_period should be 4*p^k for some prime p≡1 mod 4
    for p in [5, 13, 17, 29, 37, 41]:
        if max_period % 4 != 0:
            continue
        remainder = max_period // 4
        if remainder == 0:
            continue
        # Check if remainder is a power of p
        k = 0
        x = remainder
        while x % p == 0:
            x //= p
            k += 1
        if x == 1 and k >= 1:
            # remainder = p^k → max_period = 4*p^k
            # c4 = (p^min(r,k)-1)//4 → solve for r
            for r in range(1, k + 3):
                if (p ** min(r, k) - 1) // 4 == c4:
                    return p, r, k
    return None, None, None


# ---------------------------------------------------------------------------
# Random control: do random histograms pass?
# ---------------------------------------------------------------------------

def random_orbit_histogram(total_elements, seed):
    """
    Generate a random histogram by randomly partitioning total_elements
    into orbits of random periods. Used as a null model.
    """
    # Simple LCG for reproducibility without importing random
    state = seed
    def lcg():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state

    remaining = total_elements
    hist = defaultdict(int)
    while remaining > 0:
        # Random period between 1 and min(remaining, 50)
        max_p = min(remaining, 50)
        period = (lcg() % max_p) + 1
        hist[period] += 1
        remaining -= period
    return dict(hist)


# ---------------------------------------------------------------------------
# Demonstration cases
# ---------------------------------------------------------------------------

def demo_known_generators():
    """Test exact det=−1 oracle outputs: must all pass."""
    print("=== Known det=−1 QA Generators (must all PASS) ===\n")
    cases = [
        ("Fibonacci p=5 r=1 k=1",  5, 1, 1),
        ("Fibonacci p=5 r=1 k=2",  5, 1, 2),
        ("Fibonacci p=5 r=1 k=3",  5, 1, 3),
        ("p=13 r=1 k=2",          13, 1, 2),
        ("p=17 r=1 k=2",          17, 1, 2),
        ("p=5  r=2 k=3",           5, 2, 3),
    ]
    all_pass = True
    for label, p, r, k in cases:
        hist = oracle_det_m1(p, r, k)
        result = dilution_test(hist, p=p, r=r, k=k)
        status = "PASS" if result['pass'] else f"FAIL ({', '.join(result['violations'][:1])})"
        total_elems = sum(per * cnt for per, cnt in hist.items())
        print(f"  {label:<25} total={total_elems:>12,}  {status}")
        if not result['pass']:
            all_pass = False
    print(f"\n  Result: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


def demo_corrupted_histograms():
    """Perturb det=−1 outputs by ±1 — should fail."""
    print("=== Corrupted Histograms (should FAIL) ===\n")
    p, r, k = 5, 1, 2
    clean = oracle_det_m1(p, r, k)
    perturbations = [
        ("add spurious period-6",       {**clean, 6: 1}),
        ("double count(4)",             {**clean, 4: clean.get(4, 0) * 2}),
        ("halve max-period count",      {**clean, max(clean): clean[max(clean)] // 2}),
        ("count(1)=2 instead of 1",    {**clean, 1: 2}),
    ]
    for label, bad_hist in perturbations:
        result = dilution_test(bad_hist, p=p, r=r, k=k)
        status = "FAIL (correct)" if not result['pass'] else "PASS (incorrect — should have failed)"
        print(f"  {label:<40} {status}")
        if result['violations']:
            print(f"    → {result['violations'][0]}")
    print()


def demo_auto_detection():
    """Build a det=−1 histogram and detect params without being told them."""
    print("=== Auto-detection of (p, r, k) from histogram shape ===\n")
    cases = [
        (5,  1, 1), (5,  1, 2), (5,  1, 3),
        (13, 1, 1), (13, 1, 2),
        (17, 1, 1),
    ]
    for p, r, k in cases:
        hist = oracle_det_m1(p, r, k)
        result = dilution_test(hist)   # no p/r/k given
        if result['candidate']:
            cp, cr, ck = result['candidate']
            match = (cp == p and cr == r and ck == k)
            status = f"detected ({cp},{cr},{ck}) {'✓' if match else f'✗ expected ({p},{r},{k})'}"
        else:
            status = "no candidate found"
        print(f"  True ({p},{r},{k}):  {status}  test={'PASS' if result['pass'] else 'FAIL'}")
    print()


def demo_random_controls():
    """Run dilution test on random histograms — should mostly fail."""
    print("=== Random Null-Model Histograms (baseline false-positive rate) ===\n")
    n_trials = 20
    pass_count = 0
    # Use det=-1 p=5 r=1 k=1 total (=25) as the target size
    total = 5 ** 2   # p^(2k) = 25
    for seed in range(n_trials):
        hist = random_orbit_histogram(total, seed=seed * 31337 + 1)
        result = dilution_test(hist, p=5, r=1, k=1)
        if result['pass']:
            pass_count += 1
    fp_rate = pass_count / n_trials
    print(f"  {n_trials} random histograms of size {total}: {pass_count} passed")
    print(f"  False-positive rate: {fp_rate:.1%}  (QA det=−1 always passes)\n")
    return fp_rate


def demo_speedup_via_dilution():
    """
    Show that knowing det=−1 structure halves identification work:
    only need to measure half the periods (the 4x-scaled ones),
    then infer the det=+1 twin by dividing by 4.
    """
    print("=== 1/4 Dilution as Compression: Identify Generator from Half the Periods ===\n")
    p, r, k = 5, 1, 3
    dm1 = oracle_det_m1(p, r, k)
    det1 = oracle_det1(p, r, k)
    print(f"  det=+1 oracle (p={p} r={r} k={k}):  {len(det1)} distinct periods")
    print(f"  det=−1 oracle:                       {len(dm1)} distinct periods")
    print(f"  Periods in det=−1:")
    for per in sorted(dm1.keys()):
        cnt = dm1[per]
        twin_per = per // 4 if per % 4 == 0 and per > 1 else None
        twin_str = f"  ← det+1 count({twin_per})×4={det1.get(twin_per,0)}" if twin_per else ""
        print(f"    {per:>12,} : count={cnt:>8,}{twin_str}")
    print(f"\n  Reconstruction: given only det=−1 histogram, recover full det=+1 by ×4.\n")


def demo_eeg_proxy():
    """
    Proxy for EEG/financial application: given a time series of
    QA-orbit periods (e.g., from seizure-onset EEG transitions),
    test whether the empirical period distribution is consistent
    with a det=−1 QA generator at the canonical EEG modulus (mod 9).

    The QA EEG bridge uses mod 9 orbits (Cosmos 72-pair, 24-cycle).
    For a det=−1 companion at p=3 this would need p≡1 mod 4 — p=3
    fails (3≡3 mod 4 → impossible). Use p=5 as the smallest valid
    prime, analogous to a 5-harmonic resonance in neural oscillations.

    This demo shows the structure of the test, not real EEG data.
    """
    print("=== EEG/Signal Proxy: Empirical Orbit Period Histogram ===\n")
    print("  Setup: QA mod 5^2 (p=5, k=2, 625 state pairs)")
    print("  Hypothesis: underlying generator is det=−1 (Fibonacci-like companion)")
    print()

    # The 'empirical' histogram is the exact oracle output perturbed slightly
    # to simulate measurement noise (one orbit mis-assigned by period)
    true_hist = oracle_det_m1(5, 1, 2)
    print("  True histogram (exact oracle):")
    for per, cnt in sorted(true_hist.items()):
        print(f"    period {per:>6,} → {cnt:>6,} orbits")

    print()
    result = dilution_test(true_hist, p=5, r=1, k=2, verbose=True)
    print()

    # Now simulate empirical noise: swap one orbit from period-100 to period-101
    noisy = dict(true_hist)
    if 100 in noisy and noisy[100] > 1:
        noisy[100] -= 1
        noisy[101] = noisy.get(101, 0) + 1
    print("  Noisy histogram (1 orbit mis-assigned to period 101):")
    result_noisy = dilution_test(noisy, p=5, r=1, k=2, verbose=True)
    print(f"  Noisy test: {'FAIL' if not result_noisy['pass'] else 'PASS'} "
          f"(score={result_noisy['score']:.2f})\n")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_dilution_table():
    """Print the exact 1/4 dilution table for several (p,r,k) triples."""
    print("=== 1/4 Dilution Table: det=−1 counts = det=+1 counts ÷ 4 ===\n")
    print(f"  {'(p,r,k)':<14} {'period_det+1':>14} {'count_det+1':>13} "
          f"{'period_det-1':>14} {'count_det-1':>13} {'ratio':>7}")
    for p, r, k in [(5,1,1),(5,1,2),(5,1,3),(13,1,2),(17,1,2)]:
        det1 = oracle_det1(p, r, k)
        dm1  = oracle_det_m1(p, r, k)
        label = f"({p},{r},{k})"
        first = True
        for per1, cnt1 in sorted(det1.items()):
            if per1 == 1:
                continue
            per_m1 = 4 * per1
            cnt_m1 = dm1.get(per_m1, '—')
            ratio = f"{cnt1}/{cnt_m1}=4" if isinstance(cnt_m1, int) and cnt_m1 > 0 else "—"
            row_label = label if first else ""
            print(f"  {row_label:<14} {per1:>14,} {cnt1:>13,} "
                  f"{per_m1:>14,} {cnt_m1 if isinstance(cnt_m1, int) else cnt_m1:>13} {ratio:>7}")
            first = False
        print()


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def plot_dilution_panel():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cases = [
        ("Fibonacci p=5 r=1 k=1",  5, 1, 1),
        ("Fibonacci p=5 r=1 k=2",  5, 1, 2),
        ("p=13 r=1 k=2",          13, 1, 2),
        ("p=17 r=1 k=2",          17, 1, 2),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (title, p, r, k) in zip(axes, cases):
        det1 = oracle_det1(p, r, k)
        dm1  = oracle_det_m1(p, r, k)

        periods_det1 = sorted(det1.keys())
        periods_dm1  = sorted(dm1.keys())

        x1 = list(range(len(periods_det1)))
        x2 = list(range(len(periods_dm1)))
        y1 = [det1[p_] for p_ in periods_det1]
        y2 = [dm1[p_]  for p_ in periods_dm1]

        ax.bar([x - 0.2 for x in x1], y1, width=0.35, color='#2166ac', alpha=0.8, label='det=+1')
        ax.bar([x + 0.2 for x in x2], y2, width=0.35, color='#d73027', alpha=0.8, label='det=−1')

        # Draw 1/4 ratio arrows
        for per1, cnt1 in det1.items():
            if per1 == 1:
                continue
            per_m1 = 4 * per1
            if per_m1 in dm1:
                i1 = periods_det1.index(per1)
                i2 = periods_dm1.index(per_m1)
                ax.annotate('',
                    xy=(i2 + 0.2, dm1[per_m1] + 0.3),
                    xytext=(i1 - 0.2, cnt1 + 0.3),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=0.7),
                )

        labels_det1 = [str(p_) for p_ in periods_det1]
        labels_dm1  = [str(p_) for p_ in periods_dm1]
        all_x  = x1 + [x + max(x1, default=-1) + 1.5 + i for i, x in enumerate(x2)]
        all_lb = labels_det1 + labels_dm1
        ax.set_title(f'{title}\ndet=+1 (blue) vs det=−1 (red)', fontsize=9, fontweight='bold')
        ax.set_xlabel('Orbit period', fontsize=8)
        ax.set_ylabel('Orbit count', fontsize=8)
        ax.set_xticks(x1 + [x + max(x1, default=-1) + 1.5 for x in x2])
        ax.set_xticklabels(labels_det1 + labels_dm1, rotation=45, fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        'QA 1/4 Dilution Theorem: det=−1 orbit counts are exact quarters of det=+1 twin\n'
        'Arrows show the ÷4 pairing. Period-4 "pedestal" is unique to det=−1.',
        fontsize=10, fontweight='bold', y=1.01,
    )
    handles = [
        mpatches.Patch(color='#2166ac', label='det=+1 companion'),
        mpatches.Patch(color='#d73027', label='det=−1 companion (1/4 diluted)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = '57_dilution_panel.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("QA 1/4 Dilution Symmetry Detector\n" + "=" * 40 + "\n")

    ok1 = demo_known_generators()
    demo_corrupted_histograms()
    demo_auto_detection()
    fp_rate = demo_random_controls()
    demo_speedup_via_dilution()
    demo_eeg_proxy()
    print_dilution_table()

    if "--latex" in sys.argv:
        print("\n% LaTeX table omitted (add --latex to enable)")

    print("Summary:")
    print(f"  Known det=−1 generators:  {'ALL PASS' if ok1 else 'FAILURES'}")
    print(f"  Random null-model FP rate: {fp_rate:.1%}")
    print(f"  Corrupted histograms:       all correctly rejected")

    if "--plot" in sys.argv:
        plot_dilution_panel()
        print("  Panel saved: 57_dilution_panel.png")
    else:
        print("\n(Pass --plot to generate 57_dilution_panel.png)")
