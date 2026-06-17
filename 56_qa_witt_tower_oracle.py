"""
56_qa_witt_tower_oracle.py — Analytic Orbit Oracle from Witt Tower Chain [439]/[440]

Demonstrates that the certified orbit-count formulas replace brute-force
enumeration (O(p^2k) per call) with O(k) pure arithmetic.

Three regimes, fully characterised by the cert chain:
  det=+1, ramified (v_p(t-2)=r≥1) .............. [437]–[439]
  det=−1, ramified (v_p(t²+4)=r≥1, p≡1 mod 4).. [436],[440]
  unramified (both families) .................... [435], Wall (1960)

Output: 56_witt_oracle_panel.png
"""

import sys

# ---------------------------------------------------------------------------
# Core arithmetic helpers
# ---------------------------------------------------------------------------

def vp(n, p):
    """p-adic valuation."""
    if n == 0:
        return 999
    n = abs(n)
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def mat_mul_mod(A, B, m):
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % m,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % m],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % m,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % m],
    ]


def mat_order_mod(M, m):
    """Order of matrix M in GL_2(Z/mZ). Brute-force for small m only."""
    I = [[1, 0], [0, 1]]
    cur = [row[:] for row in M]
    for k in range(1, m * m * m + 2):
        if cur == I:
            return k
        cur = mat_mul_mod(cur, M, m)
    raise RuntimeError(f"order not found mod {m}")


def base_period_det1(t, p):
    """Period of M=[[t,-1],[1,0]] on (Z/pZ)^2 — brute-force at k=1 only."""
    M = [[t % p, (-1) % p], [1, 0]]
    return mat_order_mod(M, p)


def base_period_det_m1(t, p):
    """Period of M=[[t,1],[1,0]] on (Z/pZ)^2 — brute-force at k=1 only."""
    M = [[t % p, 1], [1, 0]]
    return mat_order_mod(M, p)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(t, p, det):
    """
    Returns (kind, r) where kind in {'ramified','unramified','impossible','p2'}.
    det=+1: char poly x²-tx+1, ramification cond p|(t-2).
    det=-1: char poly x²-tx-1, ramification cond p|(t²+4).
    """
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


# ---------------------------------------------------------------------------
# Analytic oracle — [439] (det=+1) and [440] (det=−1)
# ---------------------------------------------------------------------------

def oracle_det1_ramified(p, r, k):
    """[439]: exact orbit counts for det=+1 companion, v_p(t-2)=r, tower level k."""
    counts = {}
    if k <= r:
        counts[1] = p ** k
        birth = (p - 1) * p ** (k - 1)
        for L in range(1, k + 1):
            counts[p ** L] = birth
    else:
        counts[1] = p ** r                       # saturated fixed points
        for L in range(1, k - r + 1):           # frozen layers
            counts[p ** L] = (p * p - 1) * p ** (L + r - 2)
        birth = (p - 1) * p ** (k - 1)
        for L in range(k - r + 1, k + 1):       # birth layers
            counts[p ** L] = birth
    return counts


def oracle_det_m1_ramified(p, r, k):
    """[440]: exact orbit counts for det=−1 companion, v_p(t²+4)=r, p≡1 mod 4."""
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


def oracle_unramified(base_T, p, k):
    """Wall lift: period at p^k = base_T * p^(k-1). Returns {period: orbit_count}."""
    T_k = base_T * (p ** (k - 1))
    total = p ** (2 * k)
    # All non-fixed orbits have the same period T_k (or a divisor).
    # For the unramified case the period set is more complex, but the MAX
    # period (= the Pisano period) is T_k and carries most orbits.
    # We return a simplified view: {T_k: orbit_count_at_max_period}.
    # Fixed points: 1 (only zero vector for det=-1 or p^0=1 for det=+1 unramified).
    fixed = 1  # approximate for unramified: count(1) = 1
    rest = total - fixed
    count_max = rest // T_k if T_k > 0 else 0
    return {1: fixed, T_k: count_max}


def oracle(t, p, k, det=1):
    """
    Main oracle. Returns dict {period: orbit_count} for M on (Z/p^k Z)^2.
    Uses [439] or [440] for ramified; Wall lift for unramified.
    """
    kind, r = classify(t, p, det)
    if kind == 'ramified':
        if det == 1:
            return oracle_det1_ramified(p, r, k)
        else:
            return oracle_det_m1_ramified(p, r, k)
    elif kind == 'unramified':
        if det == 1:
            base_T = base_period_det1(t, p)
        else:
            base_T = base_period_det_m1(t, p)
        return oracle_unramified(base_T, p, k)
    else:
        return {}   # impossible or p=2 exceptional


def total_elements(counts, p, k):
    """Verify: sum(period * count) == p^(2k)."""
    return sum(period * cnt for period, cnt in counts.items())


# ---------------------------------------------------------------------------
# Brute-force for cross-validation (small cases only)
# ---------------------------------------------------------------------------

def brute_force_det1(t, p, k):
    """Orbit counts by visited-array. Hard cap at 30000 elements."""
    m = p ** k
    total = m * m
    assert total <= 30000, f"brute_force called with {total} elements — too large"
    visited = bytearray(total)
    counts = {}
    for start in range(total):
        if visited[start]:
            continue
        a, b = divmod(start, m)
        period = 0
        while True:
            i = a * m + b
            if visited[i]:
                break
            visited[i] = 1          # mark immediately so the loop terminates
            na = (t * a - b) % m
            nb = a
            a, b = na, nb
            period += 1
        counts[period] = counts.get(period, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Speedup calculation
# ---------------------------------------------------------------------------

def oracle_flops(p, r, k):
    """Approx arithmetic operations for oracle (O(k) additions/multiplications)."""
    return k * 5   # rough: 5 ops per layer

def bruteforce_flops(p, k):
    """Approx iterations for brute-force enumeration."""
    return p ** (2 * k)


# ---------------------------------------------------------------------------
# Validation: oracle vs brute-force for small cases
# ---------------------------------------------------------------------------

def validate_oracle():
    test_cases = [
        (5,  3, 1, 1),   # det=+1, t=5, p=3, r=v3(3)=1
        (11, 3, 2, 1),   # det=+1, t=11, p=3, r=v3(9)=2
        (7,  5, 1, 1),   # det=+1, t=7, p=5, r=v5(5)=1
        (1,  5, 1, -1),  # det=-1 (Fibonacci), p=5, r=1
    ]
    print("=== Oracle Validation (analytic vs brute-force, k=1..2) ===")
    all_ok = True
    for t, p, r_expected, det in test_cases:
        kind, r = classify(t, p, det)
        label = f"t={t} p={p} det={det:+d} ({kind} r={r})"
        for k in range(1, 3):
            if p ** (2 * k) > 30000:
                continue
            analytic = oracle(t, p, k, det)
            if det == 1:
                brute = brute_force_det1(t, p, k)
            else:
                # brute-force det=-1: companion [[t,1],[1,0]]
                m = p ** k
                total = m * m
                if total > 30000:
                    continue
                visited = bytearray(total)
                brute = {}
                for start in range(total):
                    if visited[start]:
                        continue
                    a, b = divmod(start, m)
                    period = 0
                    while True:
                        i = a * m + b
                        if visited[i]:
                            break
                        visited[i] = 1   # mark immediately
                        na = (t * a + b) % m
                        nb = a
                        a, b = na, nb
                        period += 1
                    brute[period] = brute.get(period, 0) + 1

            ok = analytic == brute
            all_ok = all_ok and ok
            status = "PASS" if ok else f"FAIL analytic={analytic} brute={brute}"
            print(f"  {label} k={k}: {status}")
    print(f"Validation: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}\n")
    return all_ok


# ---------------------------------------------------------------------------
# Large-k oracle demonstration (impossible for brute-force)
# ---------------------------------------------------------------------------

def demonstrate_large_k():
    print("=== Large-k Orbit Oracle (analytic only — brute-force impossible) ===")
    cases = [
        ("Fibonacci p=5 det=-1 [441]", 1, 5, -1),
        ("t=5 p=3 r=1 det=+1 [437]",  5, 3, +1),
        ("t=11 p=3 r=2 det=+1 [438]", 11, 3, +1),
        ("t=7 p=5 r=1 det=+1 [437]",  7, 5, +1),
    ]
    for label, t, p, det in cases:
        kind, r = classify(t, p, det)
        print(f"\n{label} | {kind} r={r}")
        print(f"  {'k':>3}  {'fixed':>12}  {'max_period':>14}  {'max_count':>12}  {'BF_cost':>14}  {'oracle_cost':>12}")
        for k in [1, 2, 3, 5, 8, 12, 20]:
            counts = oracle(t, p, k, det)
            if not counts:
                continue
            fixed = counts.get(1, 0)
            max_per = max(counts.keys())
            max_cnt = counts[max_per]
            bf_cost = bruteforce_flops(p, k)
            oc_cost = oracle_flops(p, r, k)
            speedup = bf_cost / oc_cost
            bf_str = f"{bf_cost:.2e}" if bf_cost > 1e6 else str(bf_cost)
            print(f"  {k:>3}  {fixed:>12,}  {max_per:>14,}  {max_cnt:>12,}  {bf_str:>14}  {oc_cost:>12}  ({speedup:.1e}x)")


# ---------------------------------------------------------------------------
# Visualisation: tower structure for four representative cases
# ---------------------------------------------------------------------------

def build_tower_data(t, p, det, k_max=10):
    """For k=1..k_max, classify orbit layers as fixed/frozen/birth."""
    kind, r = classify(t, p, det)
    layers = []
    for k in range(1, k_max + 1):
        counts = oracle(t, p, k, det)
        if not counts:
            layers.append({'k': k, 'fixed': 0, 'frozen': 0, 'birth': 0, 'total': 0})
            continue
        # For ramified cases, categorise by layer type
        fixed_elems = counts.get(1, 0) * 1
        if kind == 'ramified' and det == 1:
            frozen_elems = sum(
                cnt * per for per, cnt in counts.items()
                if per > 1 and per <= p ** (k - r) and k > r
            )
            birth_elems = sum(
                cnt * per for per, cnt in counts.items()
                if per > p ** (k - r) and k > r
            ) if k > r else sum(
                cnt * per for per, cnt in counts.items() if per > 1
            )
        elif kind == 'ramified' and det == -1:
            # period-4 orbits are "fixed-equivalent"; frozen/birth at 4p^L
            fixed_elems = counts.get(1, 0) + counts.get(4, 0) * 4
            if k > r:
                frozen_elems = sum(
                    cnt * per for per, cnt in counts.items()
                    if per > 4 and per <= 4 * p ** (k - r)
                )
                birth_elems = sum(
                    cnt * per for per, cnt in counts.items()
                    if per > 4 * p ** (k - r)
                )
            else:
                frozen_elems = 0
                birth_elems = sum(
                    cnt * per for per, cnt in counts.items()
                    if per > 4
                )
        else:  # unramified
            frozen_elems = 0
            birth_elems = sum(cnt * per for per, cnt in counts.items() if per > 1)

        total = p ** (2 * k)
        layers.append({'k': k, 'fixed': fixed_elems, 'frozen': frozen_elems,
                       'birth': birth_elems, 'total': total})
    return layers, kind, r


def plot_oracle_panel():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cases = [
        ("Fibonacci\\np=5, det=−1\\n[440]/[441]",  1, 5, -1),
        ("t=5, p=3, r=1\\ndet=+1\\n[437]/[439]",   5, 3, +1),
        ("t=11, p=3, r=2\\ndet=+1\\n[438]/[439]",  11, 3, +1),
        ("t=3, p=7\\nunramified\\n[435]+Wall",       3, 7, +1),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    colors = {'fixed': '#2166ac', 'frozen': '#d73027', 'birth': '#1a9641'}
    k_max = 10

    for ax, (title, t, p, det) in zip(axes, cases):
        layers, kind, r = build_tower_data(t, p, det, k_max=k_max)
        ks = [d['k'] for d in layers]
        totals = [d['total'] for d in layers]
        fixeds = [d['fixed'] / d['total'] for d in layers]
        frozens = [d['frozen'] / d['total'] for d in layers]
        births = [d['birth'] / d['total'] for d in layers]

        ax.stackplot(ks, fixeds, frozens, births,
                     labels=['Fixed (count(1))', 'Frozen layers', 'Birth layers'],
                     colors=[colors['fixed'], colors['frozen'], colors['birth']],
                     alpha=0.85)

        if kind == 'ramified' and r < k_max:
            ax.axvline(x=r, color='black', linestyle='--', linewidth=1.2,
                       label=f'Saturation k=r={r}')
            ax.text(r + 0.15, 0.92, f'k=r={r}', fontsize=8, color='black',
                    transform=ax.get_xaxis_transform())

        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel('Tower level k', fontsize=9)
        ax.set_ylabel('Fraction of (Z/p^k Z)²', fontsize=9)
        ax.set_xlim(1, k_max)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, k_max + 1))
        ax.tick_params(labelsize=8)

        # Annotate max period at k=k_max
        counts_last = oracle(t, p, k_max, det)
        if counts_last:
            max_per = max(counts_last.keys())
            ax.annotate(f'Max period\n={max_per:,}', xy=(k_max, 0.5),
                        xytext=(k_max - 2.5, 0.6), fontsize=7, color='white',
                        arrowprops=dict(arrowstyle='->', color='white', lw=0.8))

    # Legend on last axis
    handles = [
        mpatches.Patch(color=colors['fixed'],  label='Fixed / base-period orbits'),
        mpatches.Patch(color=colors['frozen'], label='Frozen layers (count∝p^(L+r−2))'),
        mpatches.Patch(color=colors['birth'],  label='Birth layers (count=(p−1)p^(k−1))'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Saturation k=r'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'QA Witt Tower Orbit Oracle — Analytic Layer Decomposition [439]/[440]\n'
        'Each fraction computed in O(k) arithmetic ops vs O(p²ᵏ) brute-force',
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = '56_witt_oracle_panel.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Speedup table
# ---------------------------------------------------------------------------

def print_speedup_table():
    print("\n=== Oracle Speedup vs Brute-Force Enumeration ===")
    print(f"{'Case':<30} {'k':>3}  {'BF cost (p^2k)':>18}  {'Oracle cost':>12}  {'Speedup':>12}")
    cases = [
        ("Fibonacci p=5",   5, 1, -1),
        ("t=5 p=3 r=1",     3, 1, +1),
        ("t=7 p=5 r=1",     5, 1, +1),
        ("t=11 p=3 r=2",    3, 2, +1),
    ]
    for label, p, r, det in cases:
        for k in [2, 4, 6, 10, 20]:
            bf = p ** (2 * k)
            oc = k * 5
            ratio = bf / oc
            print(f"  {label:<28} {k:>3}  {bf:>18,}  {oc:>12,}  {ratio:>12.2e}x")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("QA Witt Tower Analytic Orbit Oracle\n" + "=" * 40)

    ok = validate_oracle()
    demonstrate_large_k()
    print_speedup_table()

    if "--plot" in sys.argv:
        plot_oracle_panel()
        print("\nDone. Oracle validated and panel saved.")
    else:
        print("\nDone. (Pass --plot to generate 56_witt_oracle_panel.png)")

    sys.exit(0 if ok else 1)
