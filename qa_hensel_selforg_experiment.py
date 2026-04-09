"""
QA Hensel Lift Self-Organization Experiment
============================================
Tests whether QA-native resonance dynamics discover the hierarchical
orbit structure predicted by Pudelko (2510.24882) and the Hensel lift
from mod-3 → mod-9.

BACKGROUND (OB 2026-03-10):
  Step 1: 3 is inert in Z[phi]. Z[phi]/3Z[phi] = GF(9).
          F on (Z/3Z)^2 has exactly 2 orbits: {size 8, size 1}.
  Step 2: Hensel lift mod-3 → mod-9 splits the single mod-3 cosmos
          into 3 cosmos orbits of size 24, introduces Tribonacci.
          The 5 families are a mod-9=3^2 phenomenon.

EXPERIMENT:
  Run QA resonance self-organization at mod-3, mod-9, and mod-27=3^3.
  Measure orbit family count at convergence. QA predicts:
    mod-3:  2 orbit families (1 cosmos size 8 + 1 singularity)
    mod-9:  5 orbit families (3 cosmos + 1 satellite + 1 singularity)
    mod-27: should show further splitting per Pudelko fractal self-similarity

  Also extract the resonance coupling matrix eigenvalues and test
  whether they match QA orbit periods (Schiffman prediction applied
  to QA-native dynamics instead of standard transformer).

This is QA-native: states in {1,...,m}, QA step function, resonance
coupling, orbit classification — no standard ML architecture.

Author: Will Dale
"""
QA_COMPLIANCE = {
    'observer': 'orbit_census, resonance_eigenvalues, entropy -> float (observer projections)',
    'state_alphabet': '{1,...,m} for m in {3, 9, 27}',
    'discrete_layer': '(b,e) pairs on grid, int; resonance-weighted contagion',
    'observer_layer': 'all continuous outputs are Theorem NT compliant',
    'cert_family': '[198]+[199]+[200] — QA-native validation',
}

import numpy as np
from collections import Counter
from scipy import stats

# ─── Parameters ───────────────────────────────────────────────────────

GRID_SIZE = 16
N_CELLS = GRID_SIZE * GRID_SIZE
N_STEPS = 100
N_TRIALS = 30
ADOPT_PROB = 0.25
MODULI = [3, 9, 27]  # Hensel tower: 3, 3^2, 3^3

np.random.seed(42)


# ─── QA Core (A1-compliant) ──────────────────────────────────────────

def qa_step(bi, ei, m):
    """A1-compliant QA step: states in {1,...,m}."""
    b_new = ((bi + ei - 1) % m) + 1
    e_new = ((ei + b_new - 1) % m) + 1
    return b_new, e_new


def qa_tuple(bi, ei, m):
    """Full (b, e, d, a) tuple. A2: d=b+e, a=b+2e (A1-compliant)."""
    d_val = ((bi + ei - 1) % m) + 1
    a_val = ((bi + 2 * ei - 1) % m) + 1
    return (bi, ei, d_val, a_val)


def classify_orbit(bi, ei, m):
    """Return (orbit_type, cycle_length). Orbit by cycle length."""
    seen = []
    state = (bi, ei)
    for _ in range(m * m + 1):
        if state in seen:
            clen = len(seen) - seen.index(state)
            return clen
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    return -1


def orbit_family_id(bi, ei, m):
    """
    Return a canonical orbit family identifier.
    Two (b,e) pairs are in the same family if they lie on the same cycle.
    """
    visited = set()
    state = (bi, ei)
    for _ in range(m * m + 1):
        if state in visited:
            # Return the lexicographically smallest state in the cycle
            cycle_start = state
            cycle = [cycle_start]
            s = qa_step(cycle_start[0], cycle_start[1], m)
            while s != cycle_start:
                cycle.append(s)
                s = qa_step(s[0], s[1], m)
            return min(cycle)  # canonical representative
        visited.add(state)
        state = qa_step(state[0], state[1], m)
    return (bi, ei)


def resonance(bi1, ei1, bi2, ei2, m):
    """Resonance = tuple inner product (observer projection)."""
    t1 = qa_tuple(bi1, ei1, m)
    t2 = qa_tuple(bi2, ei2, m)
    return float(sum(t1[k] * t2[k] for k in range(4)))


# ─── Full Orbit Census ──────────────────────────────────────────────

def enumerate_orbits(m):
    """
    Exhaustively enumerate all orbit families for modulus m.
    States in {1,...,m} x {1,...,m} (A1).
    Returns dict: family_id -> {cycle_length, member_count, members}
    """
    families = {}
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            fid = orbit_family_id(b, e, m)
            if fid not in families:
                clen = classify_orbit(b, e, m)
                families[fid] = {'cycle_length': clen, 'members': [], 'count': 0}
            families[fid]['members'].append((b, e))
            families[fid]['count'] += 1

    return families


def orbit_type_label(clen):
    """Classify by cycle length into QA orbit types."""
    if clen == 1:
        return 'singularity'
    elif clen <= 8:
        return 'satellite'
    else:
        return 'cosmos'


# ─── Neighborhood & Contagion ────────────────────────────────────────

def build_neighbors(size, ntype='moore8'):
    """Build neighbor index lists."""
    if ntype == 'moore8':
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    elif ntype == 'moore24':
        offsets = [(dr, dc) for dr in range(-2, 3)
                   for dc in range(-2, 3) if dr != 0 or dc != 0]
    else:
        raise ValueError(f"Unknown neighborhood: {ntype}")

    nbrs = [[] for _ in range(size * size)]
    for row in range(size):
        for col in range(size):
            idx = row * size + col
            for dr, dc in offsets:
                nr = (row + dr) % size
                nc = (col + dc) % size
                nbrs[idx].append(nr * size + nc)
    return nbrs


def resonance_step(cell_b, cell_e, neighbors, adopt_prob, m):
    """One step of resonance-weighted contagion. All state is int (S2)."""
    n_cells = len(cell_b)
    new_b = list(cell_b)
    new_e = list(cell_e)

    for i in range(n_cells):
        best_res = -1e30
        best_j = -1
        for j in neighbors[i]:
            res = resonance(cell_b[i], cell_e[i], cell_b[j], cell_e[j], m)
            if res > best_res:
                best_res = res
                best_j = j

        if best_j >= 0 and np.random.random() < adopt_prob:  # noqa: T2-D-5
            new_b[i] = cell_b[best_j]
            new_e[i] = cell_e[best_j]

    return new_b, new_e


# ─── Resonance Matrix Eigenvalue Extraction ──────────────────────────

def extract_resonance_eigenvalues(cell_b, cell_e, m):
    """
    Build the resonance coupling matrix R[i,j] between all unique (b,e)
    states present on the grid, then extract eigenvalues.

    Schiffman prediction (QA-native): after self-organization converges,
    the resonance matrix should have eigenvalues reflecting the orbit
    period structure of the modulus.
    """
    # Find unique states
    unique_states = list(set(zip(cell_b, cell_e)))
    n = len(unique_states)
    if n < 2:
        return np.array([1.0]), unique_states

    # Build resonance matrix
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = resonance(
                unique_states[i][0], unique_states[i][1],
                unique_states[j][0], unique_states[j][1], m
            )

    # Normalize
    r_max = np.abs(R).max()
    if r_max > 0:
        R_norm = R / r_max
    else:
        R_norm = R

    eigenvalues = np.linalg.eigvals(R_norm)
    return eigenvalues, unique_states


# ─── Hensel Lift Analysis ────────────────────────────────────────────

def analyze_mod3_reduction(cell_b, cell_e, m):
    """
    For m=9 or m=27, reduce each cell's (b,e) to mod-3 and check
    whether the mod-3 orbit structure is preserved.

    This tests the Hensel lift: mod-9 families should project down
    to mod-3 families cleanly.
    """
    if m <= 3:
        return None

    m_reduced = 3
    reduced_orbits = Counter()
    full_orbits = Counter()

    for b, e in zip(cell_b, cell_e):
        # Full orbit
        full_fid = orbit_family_id(b, e, m)
        full_clen = classify_orbit(b, e, m)
        full_orbits[orbit_type_label(full_clen)] += 1

        # Reduced mod-3
        b3 = ((b - 1) % m_reduced) + 1  # A1-compliant reduction
        e3 = ((e - 1) % m_reduced) + 1
        red_clen = classify_orbit(b3, e3, m_reduced)
        reduced_orbits[orbit_type_label(red_clen)] += 1

    return {
        'full_orbit_census': dict(full_orbits),
        'reduced_mod3_census': dict(reduced_orbits),
    }


# ─── EXPERIMENT ───────────────────────────────────────────────────────

def run_experiment(m, n_steps, n_trials, grid_size, adopt_prob, ntype='moore8'):
    """Run self-org experiment for one modulus."""
    n_cells = grid_size * grid_size
    nbrs = build_neighbors(grid_size, ntype)
    fair_nbrs = build_neighbors(grid_size, 'moore8')

    trial_results = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)

        # Random init: each cell gets random (b,e) in {1,...,m}
        cell_b = [int(np.random.randint(1, m + 1)) for _ in range(n_cells)]
        cell_e = [int(np.random.randint(1, m + 1)) for _ in range(n_cells)]

        entropy_curve = []
        n_families_curve = []

        for step in range(n_steps):
            cell_b, cell_e = resonance_step(
                cell_b, cell_e, nbrs, adopt_prob, m)

            # Orbit family census
            family_ids = Counter(
                orbit_family_id(cell_b[i], cell_e[i], m)
                for i in range(n_cells)
            )
            n_families = len(family_ids)
            n_families_curve.append(n_families)

            # Entropy of orbit type distribution
            type_counts = Counter(
                orbit_type_label(classify_orbit(cell_b[i], cell_e[i], m))
                for i in range(n_cells)
            )
            h_val = 0.0
            for cnt in type_counts.values():
                if cnt > 0:
                    p_val = cnt / n_cells
                    h_val -= p_val * np.log2(p_val)
            entropy_curve.append(h_val)

        # Final state analysis
        eigenvalues, unique_states = extract_resonance_eigenvalues(
            cell_b, cell_e, m)

        hensel = analyze_mod3_reduction(cell_b, cell_e, m)

        final_families = Counter(
            orbit_family_id(cell_b[i], cell_e[i], m) for i in range(n_cells)
        )
        final_types = Counter(
            orbit_type_label(classify_orbit(cell_b[i], cell_e[i], m))
            for i in range(n_cells)
        )

        trial_results.append({
            'entropy_curve': entropy_curve,
            'n_families_curve': n_families_curve,
            'final_n_families': len(final_families),
            'final_n_unique_states': len(unique_states),
            'final_type_census': dict(final_types),
            'eigenvalue_magnitudes': np.sort(np.abs(eigenvalues))[::-1].tolist(),
            'n_eigenvalues_near_1': int(np.sum(np.abs(np.abs(eigenvalues) - 1.0) < 0.1)),
            'hensel': hensel,
        })

    return trial_results


# ─── Ground Truth ────────────────────────────────────────────────────

print("=" * 70)
print("QA HENSEL LIFT SELF-ORGANIZATION EXPERIMENT")
print("=" * 70)

print("\n--- Ground Truth: Orbit Family Enumeration ---")
for m in MODULI:
    families = enumerate_orbits(m)
    n_fam = len(families)
    by_type = Counter()
    for fid, info in families.items():
        label = orbit_type_label(info['cycle_length'])
        by_type[label] += 1

    print(f"\n  mod-{m}: {n_fam} orbit families, {m*m} total states")
    print(f"    Types: {dict(by_type)}")
    for fid, info in sorted(families.items()):
        label = orbit_type_label(info['cycle_length'])
        print(f"    Family {fid}: cycle_len={info['cycle_length']}, "
              f"members={info['count']}, type={label}")


# ─── Self-Organization Runs ──────────────────────────────────────────

for ntype in ['moore8', 'moore24']:
    print(f"\n{'='*70}")
    print(f"NEIGHBORHOOD: {ntype}")
    print(f"{'='*70}")

    for m in MODULI:
        print(f"\n--- mod-{m} ({ntype}) ---")

        # Scale steps: larger moduli need more time
        steps = N_STEPS * (m // 3)
        results = run_experiment(m, steps, N_TRIALS, GRID_SIZE, ADOPT_PROB, ntype)

        # Aggregate
        final_fam_counts = [r['final_n_families'] for r in results]
        final_unique = [r['final_n_unique_states'] for r in results]
        final_eig_uc = [r['n_eigenvalues_near_1'] for r in results]

        print(f"  Final orbit families: {np.mean(final_fam_counts):.1f} "
              f"+/- {np.std(final_fam_counts):.1f} "
              f"(range {min(final_fam_counts)}-{max(final_fam_counts)})")
        print(f"  Final unique states:  {np.mean(final_unique):.1f} "
              f"+/- {np.std(final_unique):.1f}")
        print(f"  Eigenvalues near |lambda|=1: {np.mean(final_eig_uc):.1f} "
              f"+/- {np.std(final_eig_uc):.1f}")

        # Type census
        all_types = Counter()
        for r in results:
            for t, c in r['final_type_census'].items():
                all_types[t] += c
        total = sum(all_types.values())
        print(f"  Orbit type distribution (pooled):")
        for t in ['singularity', 'satellite', 'cosmos']:
            frac = all_types.get(t, 0) / total if total > 0 else 0
            print(f"    {t:15s}: {all_types.get(t,0):6d} ({frac:.3f})")

        # Entropy convergence
        ent_final = [r['entropy_curve'][-1] for r in results]
        ent_init = [r['entropy_curve'][0] for r in results]
        print(f"  Entropy: {np.mean(ent_init):.3f} -> {np.mean(ent_final):.3f}")

        # Hensel lift analysis (for m > 3)
        if m > 3:
            print(f"  Hensel lift (mod-{m} -> mod-3):")
            for r in results[:3]:  # Show first 3 trials
                h = r['hensel']
                if h:
                    print(f"    Full: {h['full_orbit_census']}")
                    print(f"    Mod-3: {h['reduced_mod3_census']}")


# ─── Predictions vs Results ──────────────────────────────────────────

print(f"\n{'='*70}")
print("QA PREDICTIONS vs RESULTS")
print(f"{'='*70}")
print("""
QA Hensel Lift Predictions:
  mod-3:  2 orbit families (1 cosmos[8] + 1 singularity[1])
  mod-9:  5 orbit families (3 cosmos[24] + 1 satellite[8] + 1 singularity[1])
  mod-27: further splitting per Pudelko fractal self-similarity

Pudelko [198]: fractal self-similarity at prime powers —
  orbit count at m=p^k should reflect p^(k-1) * orbit_count(p)

Schiffman [199]: eigenvalues on unit circle count = active modes —
  applied to QA-native resonance matrix, not standard transformer

Key question: Does resonance-weighted self-organization CONVERGE
to the ground-truth orbit family count, or to a subset?
If it converges to the full count: QA dynamics are self-discovering.
If it converges to a subset: tells us which orbits are attractors.
""")

print("DONE")
