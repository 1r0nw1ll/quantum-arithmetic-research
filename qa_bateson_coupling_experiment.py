"""
QA Bateson-Level Coupling Experiment
=====================================
Tests orbit-aware coupling rules derived from Bateson Learning Levels [191]
and the Tiered Reachability Theorem.

The problem: pure resonance contagion collapses to singularity because
singularity has maximal self-resonance (4m^2). Orbit cycling alone
doesn't prevent this.

The fix: coupling rules that respect the Bateson invariant filtration:
  L0 = within-orbit (same cycle position)
  L1 = within-family (same orbit, different position)
  L2 = cross-family (different orbit families)

Per [191], only 26% of S_9 is L1 reachable. Unstructured L2 operators
DEGRADE performance (BraiNCA confirmed). So coupling should be:

Variant E: FAMILY-PRESERVING RESONANCE (L1 only)
  Copy only neighbors in the SAME orbit family. Families never mix.
  Resonance coupling aligns cycle positions within families.
  Prediction: exact ground-truth family count preserved.

Variant F: LOCAL MAJORITY COMPETITION
  Each cell counts same-family neighbors vs other-family neighbors.
  If outnumbered, probabilistically adopt the majority family.
  Prediction: dominant families expand, rare families contract.
  Ecological competitive exclusion — tests which families are "fittest."

Variant G: ANTI-RESONANCE DIVERSITY SEEKING
  Copy the LOWEST resonance neighbor (seek maximum dissimilarity).
  Prediction: chaotic but diversity-preserving.

Variant H: RESONANCE-MODULATED ORBIT SPEED
  No state copying at all. Instead, resonance with neighbors determines
  how many QA steps a cell takes per round. High local resonance = fast
  cycling. Low resonance = slow cycling. Families never change, but
  cycle speeds differentiate by local context.
  Prediction: spatial frequency patterns emerge.

Author: Will Dale
"""
QA_COMPLIANCE = {
    'observer': 'orbit_census, family_count, entropy, speed_map -> float (observer projections)',
    'state_alphabet': '{1,...,m} for m in {3, 9, 27}',
    'discrete_layer': '(b,e) pairs, int; Bateson-level coupling',
    'observer_layer': 'Theorem NT: all continuous outputs are measurement only',
    'cert_family': '[191] Bateson + [198]+[199]+[200]',
}

import numpy as np
from collections import Counter

# ─── Parameters ───────────────────────────────────────────────────────

GRID_SIZE = 16
N_CELLS = GRID_SIZE * GRID_SIZE
N_STEPS = 200
N_TRIALS = 30
ADOPT_PROB = 0.15
MODULI = [3, 9, 27]

np.random.seed(42)


# ─── QA Core (A1-compliant) ──────────────────────────────────────────

def qa_step(bi, ei, m):
    b_new = ((bi + ei - 1) % m) + 1
    e_new = ((ei + b_new - 1) % m) + 1
    return b_new, e_new


def qa_tuple(bi, ei, m):
    d_val = ((bi + ei - 1) % m) + 1
    a_val = ((bi + 2 * ei - 1) % m) + 1
    return (bi, ei, d_val, a_val)


def classify_orbit(bi, ei, m):
    seen = []
    state = (bi, ei)
    for _ in range(m * m + 1):
        if state in seen:
            return len(seen) - seen.index(state)
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    return -1


def orbit_family_id(bi, ei, m):
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
    t1 = qa_tuple(bi1, ei1, m)
    t2 = qa_tuple(bi2, ei2, m)
    return float(sum(t1[k] * t2[k] for k in range(4)))


# ─── Precompute family membership ───────────────────────────────────

def precompute_families(m):
    """Build lookup: (b,e) -> family_id for all states."""
    lookup = {}
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            lookup[(b, e)] = orbit_family_id(b, e, m)
    return lookup


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
        raise ValueError(ntype)

    nbrs = [[] for _ in range(size * size)]
    for row in range(size):
        for col in range(size):
            idx = row * size + col
            for dr, dc in offsets:
                nr = (row + dr) % size
                nc = (col + dc) % size
                nbrs[idx].append(nr * size + nc)
    return nbrs


# ─── Coupling Variants ───────────────────────────────────────────────

def step_E_family_preserving(cell_b, cell_e, neighbors, adopt_prob, m, fam_lookup):
    """
    Variant E: L1-only coupling. Copy best-resonance neighbor ONLY if
    they are in the SAME orbit family. Cross-family contagion blocked.
    Evolve one QA step first.
    """
    n = len(cell_b)
    # Evolve
    new_b, new_e = [], []
    for b, e in zip(cell_b, cell_e):
        bn, en = qa_step(b, e, m)
        new_b.append(bn)
        new_e.append(en)

    # Couple (L1 only)
    out_b, out_e = list(new_b), list(new_e)
    for i in range(n):
        my_fam = fam_lookup[(new_b[i], new_e[i])]
        best_res = -1e30
        best_j = -1
        for j in neighbors[i]:
            if fam_lookup[(new_b[j], new_e[j])] == my_fam:  # same family only
                res = resonance(new_b[i], new_e[i], new_b[j], new_e[j], m)
                if res > best_res:
                    best_res = res
                    best_j = j
        if best_j >= 0 and np.random.random() < adopt_prob:  # noqa: T2-D-5
            out_b[i] = new_b[best_j]
            out_e[i] = new_e[best_j]

    return out_b, out_e


def step_F_local_majority(cell_b, cell_e, neighbors, adopt_prob, m, fam_lookup):
    """
    Variant F: Local majority competition. Each cell counts how many
    neighbors share its family. If outnumbered by another single family,
    adopt a random member of the majority family with probability.
    Evolve first.
    """
    n = len(cell_b)
    # Evolve
    new_b, new_e = [], []
    for b, e in zip(cell_b, cell_e):
        bn, en = qa_step(b, e, m)
        new_b.append(bn)
        new_e.append(en)

    # Compete
    out_b, out_e = list(new_b), list(new_e)
    for i in range(n):
        my_fam = fam_lookup[(new_b[i], new_e[i])]
        nbr_fams = Counter()
        nbr_states_by_fam = {}
        for j in neighbors[i]:
            f = fam_lookup[(new_b[j], new_e[j])]
            nbr_fams[f] += 1
            if f not in nbr_states_by_fam:
                nbr_states_by_fam[f] = []
            nbr_states_by_fam[f].append(j)

        # Find majority family among neighbors
        if nbr_fams:
            majority_fam, majority_count = nbr_fams.most_common(1)[0]
            my_count = nbr_fams.get(my_fam, 0)

            # If outnumbered, probabilistically adopt majority
            if majority_fam != my_fam and majority_count > my_count:
                if np.random.random() < adopt_prob:  # noqa: T2-D-5
                    donor = np.random.choice(nbr_states_by_fam[majority_fam])  # noqa: T2-D-5
                    out_b[i] = new_b[donor]
                    out_e[i] = new_e[donor]

    return out_b, out_e


def step_G_anti_resonance(cell_b, cell_e, neighbors, adopt_prob, m, fam_lookup):
    """
    Variant G: Anti-resonance diversity seeking. Copy the LOWEST
    resonance neighbor. Seeks maximum dissimilarity. Evolve first.
    """
    n = len(cell_b)
    # Evolve
    new_b, new_e = [], []
    for b, e in zip(cell_b, cell_e):
        bn, en = qa_step(b, e, m)
        new_b.append(bn)
        new_e.append(en)

    # Anti-couple
    out_b, out_e = list(new_b), list(new_e)
    for i in range(n):
        worst_res = 1e30
        worst_j = -1
        for j in neighbors[i]:
            res = resonance(new_b[i], new_e[i], new_b[j], new_e[j], m)
            if res < worst_res:
                worst_res = res
                worst_j = j
        if worst_j >= 0 and np.random.random() < adopt_prob:  # noqa: T2-D-5
            out_b[i] = new_b[worst_j]
            out_e[i] = new_e[worst_j]

    return out_b, out_e


def step_H_speed_modulated(cell_b, cell_e, neighbors, adopt_prob, m, fam_lookup):
    """
    Variant H: Resonance-modulated orbit speed. No state copying.
    Mean resonance with neighbors determines how many QA steps
    the cell takes this round: 1 (low resonance) to 3 (high).
    Families NEVER change. Spatial speed patterns emerge.
    """
    n = len(cell_b)
    out_b, out_e = list(cell_b), list(cell_e)

    for i in range(n):
        # Compute mean resonance with neighbors
        total_res = 0.0
        for j in neighbors[i]:
            total_res += resonance(cell_b[i], cell_e[i], cell_b[j], cell_e[j], m)
        mean_res = total_res / len(neighbors[i]) if neighbors[i] else 0

        # Map resonance to step count: higher resonance = more steps
        # Use integer thresholds to avoid float->int cast (T2-b compliance)
        # max possible resonance = 4*m*m (singularity self-resonance)
        max_res_int = 4 * m * m
        # Integer thresholds: low < 1/3 max, mid < 2/3 max, high >= 2/3
        if mean_res * 3 < max_res_int:
            n_steps = 1
        elif mean_res * 3 < 2 * max_res_int:
            n_steps = 2
        else:
            n_steps = 3

        b, e = cell_b[i], cell_e[i]
        for _ in range(n_steps):
            b, e = qa_step(b, e, m)
        out_b[i] = b
        out_e[i] = e

    return out_b, out_e


VARIANTS = {
    'E_family_L1': step_E_family_preserving,
    'F_majority': step_F_local_majority,
    'G_anti_res': step_G_anti_resonance,
    'H_speed_mod': step_H_speed_modulated,
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


# ─── Measurement ─────────────────────────────────────────────────────

def measure_state(cell_b, cell_e, m, fam_lookup):
    n = len(cell_b)
    fam_counter = Counter(fam_lookup[(cell_b[i], cell_e[i])] for i in range(n))
    type_counter = Counter(
        orbit_type_label(classify_orbit(cell_b[i], cell_e[i], m))
        for i in range(n))

    h_val = 0.0
    for cnt in type_counter.values():
        if cnt > 0:
            p = cnt / n
            h_val -= p * np.log2(p)

    # Family entropy (richer than type entropy)
    h_fam = 0.0
    for cnt in fam_counter.values():
        if cnt > 0:
            p = cnt / n
            h_fam -= p * np.log2(p)

    return {
        'n_families': len(fam_counter),
        'n_unique': len(set(zip(cell_b, cell_e))),
        'type_census': dict(type_counter),
        'type_entropy': h_val,
        'family_entropy': h_fam,
        'family_sizes': sorted(fam_counter.values(), reverse=True),
    }


# ─── Experiment Runner ───────────────────────────────────────────────

def run_variant(m, vname, step_fn, ntype, n_steps, n_trials):
    nbrs = build_neighbors(GRID_SIZE, ntype)
    fam_lookup = precompute_families(m)
    results = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        cell_b = [int(np.random.randint(1, m + 1)) for _ in range(N_CELLS)]
        cell_e = [int(np.random.randint(1, m + 1)) for _ in range(N_CELLS)]

        fam_curve = []
        for step in range(n_steps):
            cell_b, cell_e = step_fn(
                cell_b, cell_e, nbrs, ADOPT_PROB, m, fam_lookup)

            if step % 10 == 0 or step == n_steps - 1:
                meas = measure_state(cell_b, cell_e, m, fam_lookup)
                fam_curve.append((step, meas['n_families']))

        final = measure_state(cell_b, cell_e, m, fam_lookup)
        results.append({'final': final, 'fam_curve': fam_curve})

    return results


# ─── MAIN ────────────────────────────────────────────────────────────

print("=" * 70)
print("QA BATESON-LEVEL COUPLING EXPERIMENT")
print("=" * 70)

print("\n--- Ground Truth ---")
for m in MODULI:
    fams = enumerate_orbits(m)
    by_type = Counter(orbit_type_label(f['cycle_length']) for f in fams.values())
    print(f"  mod-{m}: {len(fams)} families — {dict(by_type)}")

for ntype in ['moore8', 'moore24']:
    print(f"\n{'#'*70}")
    print(f"  NEIGHBORHOOD: {ntype}")
    print(f"{'#'*70}")

    for m in MODULI:
        steps = N_STEPS * max(1, m // 3)
        gt_n = len(enumerate_orbits(m))

        print(f"\n{'='*50}")
        print(f"  mod-{m} | {ntype} | {steps} steps | ground truth: {gt_n} families")
        print(f"{'='*50}")

        for vname, vfn in VARIANTS.items():
            results = run_variant(m, vname, vfn, ntype, steps, N_TRIALS)

            final_fams = [r['final']['n_families'] for r in results]
            final_ent = [r['final']['family_entropy'] for r in results]

            pooled = Counter()
            for r in results:
                for t, c in r['final']['type_census'].items():
                    pooled[t] += c
            total = sum(pooled.values())

            # Trajectory
            fam_traj = {}
            for r in results:
                for step, nf in r['fam_curve']:
                    fam_traj.setdefault(step, []).append(nf)
            key_steps = sorted(fam_traj.keys())
            show = [key_steps[0]] + [key_steps[max(1, len(key_steps)*i//4)]
                    for i in range(1, 4)] + [key_steps[-1]]
            show = sorted(set(show))
            traj_str = ' -> '.join(f"s{s}:{np.median(fam_traj[s]):.0f}" for s in show)

            # Dominant family sizes
            all_sizes = []
            for r in results:
                all_sizes.append(r['final']['family_sizes'][:3])

            print(f"\n  {vname}:")
            print(f"    Families: {np.mean(final_fams):.1f} +/- {np.std(final_fams):.1f} "
                  f"(range {min(final_fams)}-{max(final_fams)}) [GT: {gt_n}]")
            print(f"    Family entropy: {np.mean(final_ent):.3f} +/- {np.std(final_ent):.3f}")
            type_str = ', '.join(
                f"{t}={pooled.get(t,0)/total:.3f}" for t in ['sing', 'sat', 'cos']
            ) if total > 0 else 'empty'
            # Fix labels
            type_str = ', '.join(
                f"{t[:4]}={pooled.get(t,0)/total:.3f}"
                for t in ['singularity', 'satellite', 'cosmos'])
            print(f"    Types: {type_str}")
            print(f"    Trajectory: {traj_str}")

            if abs(np.median(final_fams) - gt_n) < 1.5:
                print(f"    *** MATCHES ground truth ({gt_n}) ***")
            elif np.median(final_fams) > 1.5:
                print(f"    Partial: {np.median(final_fams):.0f}/{gt_n} families")

print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
print("""
E (Family-preserving L1): Coupling within families only.
  Families can't mix → count preserved by construction.
  Tests: does L1 coupling create spatial organization WITHIN families?

F (Local majority): Ecological competition.
  Dominant families expand, rare ones contract.
  Tests: which orbit types are competitively "fittest"?
  QA prediction: cosmos (largest families, most members) should dominate.

G (Anti-resonance): Diversity-seeking coupling.
  Copy most DISSIMILAR neighbor.
  Tests: does actively seeking diversity maintain family count?

H (Speed-modulated): No state copying, only orbit speed varies.
  Families perfectly preserved. Resonance context modulates cycling rate.
  Tests: do spatial speed patterns emerge from resonance topology?
""")
print("DONE")
