"""
QA Hensel Orbit-Cycling Self-Organization Experiment
=====================================================
Variant of qa_hensel_selforg_experiment.py that interleaves QA orbit
stepping with resonance contagion. Each round:

  1. EVOLVE: every cell advances one QA step along its orbit
  2. COUPLE: resonance-weighted contagion (copy best-resonance neighbor)

This prevents singularity collapse because cells are constantly cycling
through their orbits. The question becomes: does the interplay of
deterministic orbit dynamics + stochastic resonance coupling
self-organize into the ground-truth family structure?

Ground truth (A1-compliant):
  mod-3:  3 families (2 satellite[4] + 1 singularity[1])
  mod-9:  9 families (6 cosmos[12] + 2 satellite[4] + 1 singularity[1])
  mod-27: 27 families (24 cosmos + 2 satellite + 1 singularity)
  Pattern: 3^k families for mod-3^k (Pudelko fractal self-similarity)

Coupling variants tested:
  A. EVOLVE only (no coupling — baseline: should preserve random init distribution)
  B. COUPLE only (no evolution — previous experiment: collapses to singularity)
  C. EVOLVE then COUPLE (orbit cycling + resonance contagion)
  D. COUPLE then EVOLVE (reversed order — tests whether order matters)

Author: Will Dale
"""
QA_COMPLIANCE = {
    'observer': 'orbit_census, family_count, entropy -> float (observer projections)',
    'state_alphabet': '{1,...,m} for m in {3, 9, 27}',
    'discrete_layer': '(b,e) pairs, int; QA step + resonance contagion',
    'observer_layer': 'all continuous outputs are Theorem NT compliant',
    'cert_family': '[198]+[199]+[200] — QA-native orbit-cycling validation',
}

import numpy as np
from collections import Counter

# ─── Parameters ───────────────────────────────────────────────────────

GRID_SIZE = 16
N_CELLS = GRID_SIZE * GRID_SIZE
N_STEPS = 200
N_TRIALS = 30
ADOPT_PROB = 0.15       # Lower than v2b — orbit cycling adds its own mixing
MODULI = [3, 9, 27]

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
    """Return cycle length for (b,e) under QA step."""
    seen = []
    state = (bi, ei)
    for _ in range(m * m + 1):
        if state in seen:
            return len(seen) - seen.index(state)
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    return -1


def orbit_family_id(bi, ei, m):
    """Canonical orbit family: lexicographic minimum state in the cycle."""
    visited = set()
    state = (bi, ei)
    for _ in range(m * m + 1):
        if state in visited:
            cycle_start = state
            cycle = [cycle_start]
            s = qa_step(cycle_start[0], cycle_start[1], m)
            while s != cycle_start:
                cycle.append(s)
                s = qa_step(s[0], s[1], m)
            return min(cycle)
        visited.add(state)
        state = qa_step(state[0], state[1], m)
    return (bi, ei)


def orbit_type_label(clen):
    if clen == 1:
        return 'singularity'
    elif clen <= 8:
        return 'satellite'
    else:
        return 'cosmos'


def resonance(bi1, ei1, bi2, ei2, m):
    """Resonance = tuple inner product (observer projection)."""
    t1 = qa_tuple(bi1, ei1, m)
    t2 = qa_tuple(bi2, ei2, m)
    return float(sum(t1[k] * t2[k] for k in range(4)))


# ─── Neighborhood ────────────────────────────────────────────────────

def build_neighbors(size, ntype='moore8'):
    if ntype == 'moore8':
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    elif ntype == 'moore24':
        offsets = [(dr, dc) for dr in range(-2, 3)
                   for dc in range(-2, 3) if dr != 0 or dc != 0]
    else:
        raise ValueError(f"Unknown: {ntype}")

    nbrs = [[] for _ in range(size * size)]
    for row in range(size):
        for col in range(size):
            idx = row * size + col
            for dr, dc in offsets:
                nr = (row + dr) % size
                nc = (col + dc) % size
                nbrs[idx].append(nr * size + nc)
    return nbrs


# ─── Step Variants ───────────────────────────────────────────────────

def evolve_all(cell_b, cell_e, m):
    """Advance every cell one QA step along its orbit. Deterministic."""
    new_b = []
    new_e = []
    for b, e in zip(cell_b, cell_e):
        bn, en = qa_step(b, e, m)
        new_b.append(bn)
        new_e.append(en)
    return new_b, new_e


def couple_all(cell_b, cell_e, neighbors, adopt_prob, m):
    """Resonance-weighted contagion. Stochastic."""
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


def step_evolve_only(cell_b, cell_e, neighbors, adopt_prob, m):
    """Variant A: evolve only, no coupling."""
    return evolve_all(cell_b, cell_e, m)


def step_couple_only(cell_b, cell_e, neighbors, adopt_prob, m):
    """Variant B: couple only, no evolution."""
    return couple_all(cell_b, cell_e, neighbors, adopt_prob, m)


def step_evolve_then_couple(cell_b, cell_e, neighbors, adopt_prob, m):
    """Variant C: evolve then couple."""
    b1, e1 = evolve_all(cell_b, cell_e, m)
    return couple_all(b1, e1, neighbors, adopt_prob, m)


def step_couple_then_evolve(cell_b, cell_e, neighbors, adopt_prob, m):
    """Variant D: couple then evolve."""
    b1, e1 = couple_all(cell_b, cell_e, neighbors, adopt_prob, m)
    return evolve_all(b1, e1, m)


VARIANTS = {
    'A_evolve_only': step_evolve_only,
    'B_couple_only': step_couple_only,
    'C_evolve_couple': step_evolve_then_couple,
    'D_couple_evolve': step_couple_then_evolve,
}


# ─── Ground Truth ────────────────────────────────────────────────────

def enumerate_orbits(m):
    families = {}
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            fid = orbit_family_id(b, e, m)
            if fid not in families:
                clen = classify_orbit(b, e, m)
                families[fid] = {'cycle_length': clen, 'count': 0}
            families[fid]['count'] += 1
    return families


# ─── Measurement (observer projections) ──────────────────────────────

def measure_state(cell_b, cell_e, m):
    """All measurements are observer projections (Theorem NT)."""
    n = len(cell_b)

    # Family census
    fam_counter = Counter(
        orbit_family_id(cell_b[i], cell_e[i], m) for i in range(n))
    n_families = len(fam_counter)

    # Type census
    type_counter = Counter(
        orbit_type_label(classify_orbit(cell_b[i], cell_e[i], m))
        for i in range(n))

    # Entropy of type distribution
    h_val = 0.0
    for cnt in type_counter.values():
        if cnt > 0:
            p = cnt / n
            h_val -= p * np.log2(p)

    # Unique (b,e) states
    n_unique = len(set(zip(cell_b, cell_e)))

    return {
        'n_families': n_families,
        'n_unique': n_unique,
        'type_census': dict(type_counter),
        'entropy': h_val,
    }


# ─── Experiment Runner ───────────────────────────────────────────────

def run_variant(m, variant_name, step_fn, ntype, n_steps, n_trials):
    """Run one variant for one modulus."""
    nbrs = build_neighbors(GRID_SIZE, ntype)
    trial_results = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)

        cell_b = [int(np.random.randint(1, m + 1)) for _ in range(N_CELLS)]
        cell_e = [int(np.random.randint(1, m + 1)) for _ in range(N_CELLS)]

        fam_curve = []
        ent_curve = []
        type_snapshots = []

        for step in range(n_steps):
            cell_b, cell_e = step_fn(
                cell_b, cell_e, nbrs, ADOPT_PROB, m)

            if step % 10 == 0 or step == n_steps - 1:
                meas = measure_state(cell_b, cell_e, m)
                fam_curve.append((step, meas['n_families']))
                ent_curve.append((step, meas['entropy']))
                if step == n_steps - 1:
                    type_snapshots.append(meas['type_census'])

        final = measure_state(cell_b, cell_e, m)
        trial_results.append({
            'final': final,
            'fam_curve': fam_curve,
            'ent_curve': ent_curve,
        })

    return trial_results


# ─── MAIN ────────────────────────────────────────────────────────────

print("=" * 70)
print("QA HENSEL ORBIT-CYCLING SELF-ORGANIZATION")
print("=" * 70)

# Ground truth
print("\n--- Ground Truth ---")
for m in MODULI:
    families = enumerate_orbits(m)
    by_type = Counter()
    for fid, info in families.items():
        by_type[orbit_type_label(info['cycle_length'])] += 1
    print(f"  mod-{m}: {len(families)} families — {dict(by_type)}")

# Run experiments
for ntype in ['moore8', 'moore24']:
    print(f"\n{'#'*70}")
    print(f"  NEIGHBORHOOD: {ntype}")
    print(f"{'#'*70}")

    for m in MODULI:
        steps = N_STEPS * max(1, m // 3)  # scale with modulus

        print(f"\n{'='*50}")
        print(f"  mod-{m} | {ntype} | {steps} steps | {N_TRIALS} trials")
        print(f"{'='*50}")

        gt = enumerate_orbits(m)
        gt_n = len(gt)

        for vname, vfn in VARIANTS.items():
            results = run_variant(m, vname, vfn, ntype, steps, N_TRIALS)

            final_fams = [r['final']['n_families'] for r in results]
            final_uniq = [r['final']['n_unique'] for r in results]
            final_ent = [r['final']['entropy'] for r in results]

            # Pooled type census
            pooled = Counter()
            for r in results:
                for t, c in r['final']['type_census'].items():
                    pooled[t] += c
            total = sum(pooled.values())

            # Family count trajectory (median across trials)
            fam_traj = {}
            for r in results:
                for step, nf in r['fam_curve']:
                    fam_traj.setdefault(step, []).append(nf)

            print(f"\n  {vname}:")
            print(f"    Families: {np.mean(final_fams):.1f} +/- {np.std(final_fams):.1f} "
                  f"(range {min(final_fams)}-{max(final_fams)}) "
                  f"[ground truth: {gt_n}]")
            print(f"    Unique states: {np.mean(final_uniq):.1f} +/- {np.std(final_uniq):.1f}")
            print(f"    Entropy: {np.mean(final_ent):.3f} +/- {np.std(final_ent):.3f}")
            type_str = ', '.join(
                f"{t}={pooled.get(t,0)/total:.3f}" for t in ['singularity', 'satellite', 'cosmos'])
            print(f"    Types: {type_str}")

            # Show trajectory at key steps
            key_steps = sorted(fam_traj.keys())
            if len(key_steps) > 5:
                show = [key_steps[0], key_steps[len(key_steps)//4],
                        key_steps[len(key_steps)//2],
                        key_steps[3*len(key_steps)//4], key_steps[-1]]
            else:
                show = key_steps
            traj_str = ' -> '.join(
                f"s{s}:{np.median(fam_traj[s]):.0f}" for s in show)
            print(f"    Family trajectory: {traj_str}")

            # Match to ground truth?
            median_final = np.median(final_fams)
            if abs(median_final - gt_n) < 1:
                print(f"    *** MATCHES ground truth ({gt_n}) ***")
            elif median_final > 1:
                print(f"    Partial structure: {median_final:.0f}/{gt_n} families")


# ─── Summary ─────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("""
Variant A (evolve only): Baseline — QA dynamics alone, no coupling.
  Cells cycle through their orbits. Family count should stay at init level
  (random init samples many families) but the DISTRIBUTION shifts as
  cells traverse their orbits.

Variant B (couple only): Previous experiment — collapses to singularity.
  Singularity has max self-resonance (4m^2), dominates all coupling.

Variant C (evolve then couple): The key test.
  Orbit cycling changes each cell's resonance profile every step.
  A cell in a cosmos orbit will resonance-match DIFFERENT neighbors
  at different points in its cycle. Does this prevent singularity
  collapse and stabilize orbit diversity?

Variant D (couple then evolve): Order-reversed control.
  Tests whether coupling before evolution changes the outcome.

Key question: Does variant C converge to a family count BETWEEN
the extremes of A (many, unstable) and B (1, singularity)?
If C stabilizes near the ground-truth count, QA orbit-cycling
+ resonance coupling is self-discovering.
""")

print("DONE")
