"""
59_qa_filter_bank.py — Multi-Resolution QA Filter Bank

Design
------
Modulus sequence : p=3, levels k=1..6 (moduli 3, 9, 27, 81, 243, 729)
Signal           : orbit trajectory of det=+1 companion M=[[5,-1],[1,0]] mod 3^6,
                   projected down to each coarser level
Companion        : t=5, p=3, r=v_3(5-2)=v_3(3)=1  →  [439] applies, r=1
Output           : layer energy profile + birth coefficient spectrum +
                   multi-resolution orbit-label matrix

Filter bank layer structure at level k (from cert [439], p=3, r=1):
  Fixed layer   : 3  single-element orbits (period 1)   — DC component
  Frozen L      : 8·3^{L-1} orbits of period 3^L, for L=1..k-1  — low/mid freq
  Birth layer   : 2·3^{k-1} orbits of period 3^k   — finest detail at level k

Key result: birth fraction = 2/3 at EVERY k (= (p-1)/p for p=3).
This makes the QA filter bank "equal-detail": each new level adds the same
fraction of new information, unlike classical wavelets where detail decays.

Birth-layer test (no period enumeration needed):
  is_birth_at_k(b,e,k)  ↔  M^{3^{k-1}}(b,e) ≠ (b,e)  mod 3^k
  Computed via fast matrix power: O(k·log 3) matrix multiplications per query.

p-adic wavelet analogy
  Classical wavelet:   signal = approx_k + Σ detail_j  (j=1..k)
  QA filter bank:      orbit_k = lift(orbit_{k-1}) + birth_layer_k
  The "detail" coefficient at level k = probability of being born at k.

Primary sources: cert [439] doi:QA-Witt-Tower-det+1, Wall (1960), Serre (1979).
"""

import sys
import math


# ---------------------------------------------------------------------------
# Arithmetic: fast matrix power mod m
# ---------------------------------------------------------------------------

def mat_mul_mod(A, B, m):
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % m,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % m],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % m,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % m],
    ]


def mat_pow_mod(M, n, m):
    """M^n mod m via binary exponentiation. O(log n) matrix multiplications."""
    if n == 0:
        return [[1, 0], [0, 1]]
    result = [[1, 0], [0, 1]]
    base = [row[:] for row in M]
    while n > 0:
        if n & 1:
            result = mat_mul_mod(result, base, m)
        base = mat_mul_mod(base, base, m)
        n >>= 1
    return result


def apply_mat(M, b, e, m):
    """Apply matrix M to vector (b, e) mod m. Returns (b', e')."""
    return (M[0][0]*b + M[0][1]*e) % m, (M[1][0]*b + M[1][1]*e) % m


# ---------------------------------------------------------------------------
# Companion matrix for t=5, det=+1: M = [[5, -1], [1, 0]]
# Canonical QA det=+1 companion at p=3, r=1.
# ---------------------------------------------------------------------------

T = 5   # trace
P = 3   # prime
K_MAX = 6   # finest level (mod 3^6 = 729)
COMPANION = [[T, -1], [1, 0]]   # det=T*0 - (-1)*1 = 1 ✓


# ---------------------------------------------------------------------------
# Analytic oracle (cert [439], p=3, r=1)
# ---------------------------------------------------------------------------

def oracle_counts(k):
    """
    Exact orbit counts at level k for COMPANION at p=3, r=1.
    Returns dict: layer_label → (orbit_count, period, element_count).
    """
    p, r = P, 1
    layers = {}
    # Fixed: p^r = 3 orbits of period 1
    layers['fixed'] = (p ** r, 1, p ** r)

    if k <= r:
        # Only birth layer: (p-1)*p^{k-1} orbits of period p^k
        birth_cnt = (p - 1) * p ** (k - 1)
        layers['birth_k'] = (birth_cnt, p ** k, birth_cnt * p ** k)
    else:
        # Frozen layers L=1..k-r=k-1
        for L in range(1, k):
            cnt = (p * p - 1) * p ** (L + r - 2)   # = 8 * 3^{L-1}
            layers[f'frozen_L{L}'] = (cnt, p ** L, cnt * p ** L)
        # Birth layer L=k
        birth_cnt = (p - 1) * p ** (k - 1)          # = 2 * 3^{k-1}
        layers['birth_k'] = (birth_cnt, p ** k, birth_cnt * p ** k)

    return layers


def layer_fractions(k):
    """Fraction of total elements in each layer at level k."""
    total = P ** (2 * k)
    counts = oracle_counts(k)
    return {name: elems / total for name, (_, _, elems) in counts.items()}


# ---------------------------------------------------------------------------
# Birth-layer test for a single state (b, e)
# ---------------------------------------------------------------------------

def period_val(b, e, k):
    """
    Return v_p(period of (b,e) mod P^k).

    The period at level k is always P^j for some j in {0,...,k} (since the
    group exponent at level k divides P^k for this companion). We find the
    minimum j such that M^{P^j} fixes (b mod P^k, e mod P^k).

    j=0 → period=1  (fixed layer)
    j=L, 1≤L<k → period=P^L (frozen at layer L)
    j=k → period=P^k (birth layer at level k)
    """
    M = COMPANION
    mod = P ** k
    b_k = b % mod
    e_k = e % mod
    for j in range(k + 1):
        pj = P ** j
        Mj = mat_pow_mod(M, pj, mod)
        b2, e2 = apply_mat(Mj, b_k, e_k, mod)
        if b2 == b_k and e2 == e_k:
            return j   # period = P^j at level k
    return k   # fallback (shouldn't reach for prime-power companion)


# ---------------------------------------------------------------------------
# Random state sample (for birth-level distribution demo)
# ---------------------------------------------------------------------------

def sample_states(n, mod, seed=12345):
    """
    Sample n random states from (Z/mod Z)^2 using an LCG.
    Random sample across ALL orbits gives birth-level histogram ≈ oracle fractions.
    Single orbit trajectories give a delta function (all states share one birth level).
    """
    s = seed
    def lcg():
        nonlocal s
        s = (6364136223846793005 * s + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return s
    return [(lcg() % mod, lcg() % mod) for _ in range(n)]


def orbit_label_matrix(states, k_max=K_MAX):
    """
    Classify each state in the multi-resolution tower.

    Returns:
        labels_kmax : list of period_val(b,e,k_max) for each state — the
                      layer index at the finest level (0=fixed, L=frozen-L,
                      k_max=birth)
        fracs       : dict k → (fixed_frac, frozen_frac, birth_frac) computed
                      independently at each level k via period_val(b,e,k)
    """
    labels_kmax = [period_val(b, e, k_max) for b, e in states]

    fracs = {}
    N = len(states)
    for k in range(1, k_max + 1):
        labels_k = [period_val(b, e, k) for b, e in states]
        n_fixed  = sum(1 for pv in labels_k if pv == 0)
        n_birth  = sum(1 for pv in labels_k if pv == k)
        n_frozen = N - n_fixed - n_birth
        fracs[k] = (n_fixed / N, n_frozen / N, n_birth / N)

    return labels_kmax, fracs


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def print_theoretical_profile():
    print("=== Theoretical Layer Energy Profile (from cert [439]) ===\n")
    print(f"  Companion M=[[{T},{-1}],[1,0]], p={P}, r=1")
    print(f"  Total elements at level k: p^(2k) = 3^(2k)\n")

    print(f"  {'k':>3}  {'total':>10}  {'fixed%':>8}  {'frozen%':>8}  {'birth%':>8}  "
          f"{'birth_cnt':>10}  {'birth_per':>10}")
    print("  " + "-" * 70)

    for k in range(1, K_MAX + 1):
        total = P ** (2 * k)
        counts = oracle_counts(k)
        fixed_c, _, fixed_e   = counts['fixed']
        birth_c, birth_p, birth_e = counts['birth_k']

        frozen_e = sum(elems for name, (_, _, elems) in counts.items()
                       if name.startswith('frozen'))

        fp = fixed_e  / total * 100
        frp = frozen_e / total * 100
        bp = birth_e  / total * 100

        print(f"  {k:>3}  {total:>10,}  {fp:>7.3f}%  {frp:>7.3f}%  {bp:>7.3f}%  "
              f"{birth_c:>10,}  {birth_p:>10,}")

    print()
    print(f"  Key result: birth fraction = (p-1)/p = 2/3 = 66.667% at every k.")
    print(f"  The QA filter bank is equal-detail: each level adds the same")
    print(f"  fraction of new frequency content. Fixed layer → 0 as k → ∞.\n")


def print_frozen_layer_spacing():
    print("=== Frozen Layer Spectrum (frequency bin sizes) ===\n")
    print(f"  At each Witt level k, frozen layers form a geometric frequency grid.")
    print(f"  Bin L has period 3^L and count (p²-1)·p^{{L+r-2}} = 8·3^{{L-1}} orbits.\n")

    print(f"  Level k=6 frozen spectrum (periods 3^1 through 3^5):")
    k = 6
    total = P ** (2 * k)
    print(f"  {'Layer L':>8}  {'period':>8}  {'orbit_count':>12}  {'element_count':>14}  {'frac%':>8}")
    print("  " + "-" * 58)
    for L in range(1, k):
        cnt = 8 * (P ** (L - 1))
        period = P ** L
        elems = cnt * period
        frac = elems / total * 100
        print(f"  {'L='+str(L):>8}  {period:>8,}  {cnt:>12,}  {elems:>14,}  {frac:>7.4f}%")

    birth_c = 2 * P ** (k - 1)
    birth_p = P ** k
    birth_e = birth_c * birth_p
    print(f"  {'birth':>8}  {birth_p:>8,}  {birth_c:>12,}  {birth_e:>14,}  "
          f"{birth_e/total*100:>7.4f}%")
    print()
    print(f"  Bin widths grow geometrically (×3 each level).")
    print(f"  This is the p-adic logarithmic frequency axis.\n")


def print_trajectory_analysis(traj, labels, fracs):
    n = len(traj)
    print(f"=== Sample Analysis: {n} random states from (Z/3^{K_MAX}Z)^2 ===\n")

    # Layer histogram at finest level (labels = period_val at K_MAX)
    # j=0 → fixed; j=1..K_MAX-1 → frozen at layer j; j=K_MAX → birth layer
    from_level = [0] * (K_MAX + 1)
    for lvl in labels:
        from_level[min(lvl, K_MAX)] += 1

    print(f"  Layer distribution at finest level k={K_MAX} (mod 3^{K_MAX}={P**K_MAX}):")
    layer_names = ['fixed (period 1)'] + \
                  [f'frozen L={j} (period {P**j})' for j in range(1, K_MAX)] + \
                  [f'birth   (period {P**K_MAX})']
    for lvl in range(K_MAX + 1):
        cnt = from_level[lvl]
        bar = '█' * (cnt * 40 // n) if n > 0 else ''
        th = (P**(K_MAX - 1) * (P - 1) / P**(2*K_MAX)) if lvl == K_MAX else 0
        print(f"    {layer_names[lvl]:<28}: {cnt:>5} ({cnt/n*100:>5.1f}%)  {bar}")
    print()

    print("  Multi-resolution layer fractions along trajectory:")
    print(f"  {'k':>3}  {'fixed%':>8}  {'frozen%':>8}  {'birth_k%':>9}  note")
    print("  " + "-" * 55)
    for k in range(1, K_MAX + 1):
        fp, frp, bp = fracs[k]
        # Compare to theoretical
        th_fracs = layer_fractions(k)
        th_birth = th_fracs['birth_k']
        note = f"theory birth={th_birth*100:.1f}%" if abs(bp - th_birth) > 0.01 else "≈ theory"
        print(f"  {k:>3}  {fp*100:>7.3f}%  {frp*100:>7.3f}%  {bp*100:>8.3f}%  {note}")
    print()


def print_reconstruction_sketch():
    print("=== Reconstruction: Coarse-to-Fine Orbit Lifting ===\n")
    print("  Classical wavelet: x = Σ_{k=0}^{K} d_k  where d_k = detail at level k")
    print("  QA filter bank:    orbit_k = lift(orbit_{k-1}) ⊕ birth_k")
    print()
    print("  The lift map: given orbit O at level k-1, the orbits at level k that")
    print("  project down to O are:")
    print("    - exactly 1 'frozen copy' of O (same period at level k)")
    print("    - p new 'birth orbits' (period p^k, subdivide O at finest scale)")
    print()
    print("  This is the QA analogue of the two-channel filter bank split:")
    print("    lowpass (frozen copy) + highpass (p new birth orbits)")
    print()
    print("  At each level the signal energy splits as:")
    print(f"    1/(p+1) into the frozen channel (kept from level k-1)")
    print(f"    p/(p+1) into p birth channels (new at level k)")
    print(f"  For p=3: 1/4 frozen, 3/4 into 3 new birth channels.")
    print()
    # Show the actual numbers for k=2 → k=3
    k = 3
    counts = oracle_counts(k)
    print(f"  Concrete: level k={k-1} → k={k} (mod 3^{k-1}={P**(k-1)} → mod 3^{k}={P**k})")
    pk1 = P ** (k - 1)
    pk  = P ** k
    birth_k1 = 2 * P ** (k - 2)
    birth_k  = 2 * P ** (k - 1)
    frozen_k = 8 * P ** (k - 2)
    print(f"    Birth orbits at k-1: {birth_k1} (period {pk1})")
    print(f"    Each birth orbit at k-1 splits into:")
    print(f"      1 frozen copy (period {pk1}, period unchanged)")
    print(f"      3 new birth orbits (period {pk})")
    print(f"    Total birth at k: {birth_k} new orbits  (= 3 × {birth_k1} = {3*birth_k1})")
    print(f"    Total frozen at k, L={k-1}: {frozen_k} orbits  (includes all prior births)")
    print()


def print_speed_comparison():
    print("=== Computational Cost: Filter Bank vs Classical Enumeration ===\n")
    print(f"  {'Level k':>8}  {'modulus':>8}  {'states':>12}  "
          f"{'enum_cost':>14}  {'oracle_cost':>13}  {'birth_test':>12}  speedup")
    print("  " + "-" * 85)
    for k in range(1, K_MAX + 1):
        pk = P ** k
        states = pk * pk
        enum_cost = states               # O(p^{2k}) for full enumeration
        oracle_cost = k * 5             # O(k) arithmetic ops
        birth_test = k * 2 * (2 * k)   # O(k log p) matrix mults, ~2k^2 for p=3
        speedup = enum_cost / oracle_cost
        print(f"  {k:>8}  {pk:>8,}  {states:>12,}  "
              f"{enum_cost:>14,}  {oracle_cost:>13,}  {birth_test:>12,}  {speedup:.1e}x")
    print()


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def plot_filter_bank_panel():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_energy, ax_spectrum, ax_traj, ax_lift = axes.flatten()

    ks = list(range(1, K_MAX + 1))
    colors = {'fixed': '#2166ac', 'frozen': '#d73027', 'birth': '#1a9641'}

    # --- Panel 1: Layer energy profile ---
    fixed_fracs  = []
    frozen_fracs = []
    birth_fracs  = []
    for k in ks:
        f = layer_fractions(k)
        total_frozen = sum(v for name, v in f.items() if name.startswith('frozen'))
        fixed_fracs.append(f['fixed'])
        frozen_fracs.append(total_frozen)
        birth_fracs.append(f['birth_k'])

    ax_energy.stackplot(ks, fixed_fracs, frozen_fracs, birth_fracs,
                        labels=['Fixed (DC)', 'Frozen (low/mid freq)', 'Birth (high freq)'],
                        colors=[colors['fixed'], colors['frozen'], colors['birth']],
                        alpha=0.85)
    ax_energy.axhline(y=2/3, color='black', linestyle='--', linewidth=1,
                      label='Birth = 2/3 (constant)')
    ax_energy.set_title('Layer Energy vs Tower Level k\n(p=3, t=5, r=1)', fontsize=10, fontweight='bold')
    ax_energy.set_xlabel('Tower level k', fontsize=9)
    ax_energy.set_ylabel('Fraction of total elements', fontsize=9)
    ax_energy.set_xlim(1, K_MAX)
    ax_energy.set_ylim(0, 1.02)
    ax_energy.set_xticks(ks)
    ax_energy.legend(fontsize=8, loc='center right')

    # --- Panel 2: Frozen layer spectrum at k=K_MAX ---
    k = K_MAX
    total = P ** (2 * k)
    frozen_periods = [P ** L for L in range(1, k)]
    frozen_counts  = [8 * P ** (L - 1) for L in range(1, k)]
    frozen_fracs_k = [cnt * P ** L / total for cnt, L in zip(frozen_counts, range(1, k))]
    birth_p = P ** k
    birth_f = 2 * P ** (k-1) * P ** k / total

    all_periods = frozen_periods + [birth_p]
    all_fracs   = frozen_fracs_k  + [birth_f]
    ax_spectrum.bar(range(len(all_periods)), [f*100 for f in all_fracs],
                    color=[colors['frozen']] * len(frozen_periods) + [colors['birth']],
                    alpha=0.85, edgecolor='white')
    ax_spectrum.set_xticks(range(len(all_periods)))
    ax_spectrum.set_xticklabels([f'3^{L}' for L in range(1, k)] + [f'3^{k}(birth)'],
                                 fontsize=8, rotation=30)
    ax_spectrum.set_title(f'Frozen Layer Frequency Spectrum (k={k})\nElement fraction per period bin',
                          fontsize=10, fontweight='bold')
    ax_spectrum.set_ylabel('% of elements', fontsize=9)

    # --- Panel 3: Birth-level histogram from random state sample ---
    traj = sample_states(500, P ** K_MAX)
    labels, fracs_traj = orbit_label_matrix(traj, K_MAX)
    birth_hist = [0] * (K_MAX + 1)
    for lvl in labels:
        birth_hist[min(lvl, K_MAX)] += 1
    n = len(traj)
    ax_traj.bar(range(K_MAX + 1),
                [h / n * 100 for h in birth_hist],
                color=[colors['fixed']] + [colors['birth']] * K_MAX,
                alpha=0.85, edgecolor='white')
    ax_traj.set_xticks(range(K_MAX + 1))
    ax_traj.set_xticklabels(['fixed'] + [f'born k={k}' for k in range(1, K_MAX + 1)],
                             fontsize=8, rotation=30)
    ax_traj.set_title(f'Trajectory Birth-Level Distribution\n({n} steps from (1,1))',
                      fontsize=10, fontweight='bold')
    ax_traj.set_ylabel('% of trajectory states', fontsize=9)

    # --- Panel 4: Multi-resolution layer fractions along trajectory ---
    k_list = list(fracs_traj.keys())
    fix_t  = [fracs_traj[k][0] * 100 for k in k_list]
    frz_t  = [fracs_traj[k][1] * 100 for k in k_list]
    bth_t  = [fracs_traj[k][2] * 100 for k in k_list]
    th_bth = [layer_fractions(k)['birth_k'] * 100 for k in k_list]

    ax_lift.plot(k_list, fix_t, 'o-', color=colors['fixed'],  label='Fixed', linewidth=2)
    ax_lift.plot(k_list, frz_t, 's-', color=colors['frozen'], label='Frozen', linewidth=2)
    ax_lift.plot(k_list, bth_t, '^-', color=colors['birth'],  label='Birth k', linewidth=2)
    ax_lift.plot(k_list, th_bth, '--', color='grey', linewidth=1.2, label='Theory birth')
    ax_lift.set_title('Trajectory vs Theory: Layer Fractions\nMulti-resolution filter output',
                      fontsize=10, fontweight='bold')
    ax_lift.set_xlabel('Tower level k', fontsize=9)
    ax_lift.set_ylabel('% in layer', fontsize=9)
    ax_lift.set_xticks(k_list)
    ax_lift.legend(fontsize=8)

    fig.suptitle(
        'QA Multi-Resolution Filter Bank — p=3, k=1..6, det=+1 companion (t=5, r=1)\n'
        'Frozen layer = lowpass; Birth layer = highpass (66.7% of energy, constant)',
        fontsize=11, fontweight='bold', y=1.01,
    )
    handles = [
        mpatches.Patch(color=colors['fixed'],  label='Fixed (DC)'),
        mpatches.Patch(color=colors['frozen'], label='Frozen layers (low/mid freq)'),
        mpatches.Patch(color=colors['birth'],  label='Birth layer (high freq, 2/3 const)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = '59_filter_bank_panel.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("QA Multi-Resolution Filter Bank\n" + "=" * 40 + "\n")
    print(f"Companion: M=[[{T},{-1}],[1,0]], p={P}, r=1  →  [439] orbit law\n")

    print_theoretical_profile()
    print_frozen_layer_spacing()

    # Sample random states across all orbits and analyse
    mod_K = P ** K_MAX
    N_TRAJ = 500
    traj = sample_states(N_TRAJ, mod_K)
    labels, fracs_traj = orbit_label_matrix(traj, K_MAX)

    print_trajectory_analysis(traj, labels, fracs_traj)
    print_reconstruction_sketch()
    print_speed_comparison()

    if "--plot" in sys.argv:
        plot_filter_bank_panel()
        print("Panel saved: 59_filter_bank_panel.png")
    else:
        print("(Pass --plot to generate 59_filter_bank_panel.png)")
