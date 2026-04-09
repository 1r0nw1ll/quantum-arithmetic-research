"""
QA BraiNCA Self-Organization Speed v2b — Resonance-Weighted Contagion
=====================================================================

Maps to: Pio-Lopez, Hartl, Levin (2026) arXiv:2604.01932 (BraiNCA)

Purely emergent self-organization: NO organizer cells.  Random
initialization, then resonance-weighted contagion drives convergence.
Each step: find the best-resonance neighbor, copy their (b,e) with
probability adopt_prob.

Measures:
  1. Shannon entropy of orbit census (observer projection)
  2. Spatial clustering: fraction of cells whose 8 nearest neighbors
     share the same majority orbit.  Always measured on 8 nearest
     neighbors regardless of contagion neighborhood (fairness).

Compares four neighborhood types:
  - moore8:            standard 3x3
  - moore8_random_lr:  3x3 + 3 random long-range edges
  - moore8_struct_lr:  3x3 + 3 structured long-range (half-grid hops)
  - moore24:           5x5 (24 neighbors)

Result: moore24 converges dramatically faster — entropy at step 5 is
0.946 vs 1.168 for moore8 (p < 0.0001), clustering at step 10 is
0.936 vs 0.810 (p < 0.0001).

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'entropy, spatial_clustering -> float (observer projection)',
    'state_alphabet': '{1,...,9} (mod-9 integer QA states)',
    'discrete_layer': '(b,e) pairs on 16x16 grid, int; resonance-weighted contagion',
    'observer_layer': 'entropy, clustering -> float (Theorem NT, measurement only)',
    'signal_injection': 'none (random initialization, purely emergent self-organization)',
    'coupling': 'resonance = tuple inner product; best-resonance neighbor copied with prob adopt_prob',
}

import numpy as np
from collections import Counter
from scipy import stats

# ─── Parameters ───────────────────────────────────────────────────────

M = 9
GRID_SIZE = 16
N_CELLS = GRID_SIZE * GRID_SIZE
N_STEPS = 50
N_TRIALS = 25
ADOPT_PROB = 0.25

np.random.seed(42)


# ─── QA Core (A1-compliant) ──────────────────────────────────────────

def qa_step(bi, ei, m=M):
    """A1-compliant QA step: states in {1,...,m}."""
    b_new = ((bi + ei - 1) % m) + 1
    e_new = ((ei + b_new - 1) % m) + 1
    return b_new, e_new


def qa_tuple(bi, ei, m=M):
    """Derive full (b, e, d_val, a_val) tuple. A2: d_val = b+e, a_val = b+2e."""
    d_val = ((bi + ei - 1) % m) + 1      # A1-compliant derived coord
    a_val = ((bi + 2 * ei - 1) % m) + 1  # A1-compliant derived coord
    return (bi, ei, d_val, a_val)


def classify_orbit(bi, ei, m=M):
    """Classify orbit by cycle length: 0=singularity, 1=satellite, 2=cosmos."""
    seen = []
    state = (bi, ei)
    for _ in range(m * m + 1):
        if state in seen:
            clen = len(seen) - seen.index(state)
            if clen == 1:
                return 0  # singularity
            elif clen <= 8:
                return 1  # satellite
            else:
                return 2  # cosmos
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    return -1


def resonance(bi1, ei1, bi2, ei2, m=M):
    """
    Resonance = tuple inner product (observer projection — float).
    sum(t_i[k] * t_j[k] for k in 0..3) where t = (b, e, d, a).
    """
    t1 = qa_tuple(bi1, ei1, m)
    t2 = qa_tuple(bi2, ei2, m)
    return float(sum(t1[k] * t2[k] for k in range(4)))


# ─── Observer Projections ─────────────────────────────────────────────

def shannon_entropy(counts_dict, total):
    """Shannon entropy of orbit distribution (observer projection — float)."""
    h_val = 0.0
    for cnt in counts_dict.values():
        if cnt > 0:
            p_val = cnt / total
            h_val -= p_val * np.log2(p_val)
    return h_val


def spatial_clustering_fair(cell_b, cell_e, fair_nbrs, m=M):
    """
    Fraction of cells whose 8 nearest neighbors share the same orbit.
    Always measured on the FAIR (moore8) neighborhood list regardless
    of which contagion neighborhood is used (observer projection — float).
    """
    n_cells = len(cell_b)
    orbits = [classify_orbit(cell_b[i], cell_e[i], m) for i in range(n_cells)]
    match_count = 0
    total_pairs = 0
    for i in range(n_cells):
        for j in fair_nbrs[i]:
            total_pairs += 1
            if orbits[i] == orbits[j]:
                match_count += 1
    if total_pairs == 0:
        return 0.0
    return match_count / total_pairs


# ─── Neighborhood Construction ───────────────────────────────────────

def build_neighbors(size, ntype):
    """Build neighbor index lists for each cell."""
    nbrs = [[] for _ in range(size * size)]

    if ntype in ('moore8', 'moore8_random_lr', 'moore8_struct_lr'):
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    elif ntype == 'moore24':
        offsets = [(dr, dc) for dr in range(-2, 3)
                   for dc in range(-2, 3) if dr != 0 or dc != 0]
    else:
        raise ValueError(f"Unknown neighborhood: {ntype}")

    for row in range(size):
        for col in range(size):
            idx = row * size + col
            for dr, dc in offsets:
                nr = (row + dr) % size
                nc = (col + dc) % size
                nbrs[idx].append(nr * size + nc)

    # Add long-range edges
    if ntype == 'moore8_random_lr':
        for i in range(size * size):
            for _ in range(3):
                j = np.random.randint(0, size * size)  # noqa: T2-D-5
                if j != i and j not in nbrs[i]:
                    nbrs[i].append(j)
    elif ntype == 'moore8_struct_lr':
        for i in range(size * size):
            ri = i // size
            ci = i % size
            targets = [
                ((ri + size // 2) % size) * size + (ci % size),
                (ri % size) * size + ((ci + size // 2) % size),
                ((ri + size // 4) % size) * size + ((ci + size // 4) % size),
            ]
            for j in targets:
                if j not in nbrs[i]:
                    nbrs[i].append(j)

    return nbrs


# ─── Resonance-Weighted Contagion Step (No Organizers) ───────────────

def resonance_step(cell_b, cell_e, neighbors, adopt_prob=ADOPT_PROB, m=M):
    """
    One step of resonance-weighted contagion — purely emergent.
    Each cell finds its best-resonance neighbor and copies with probability.
    All state variables are Python int (S2 compliant).
    """
    n_cells = len(cell_b)
    new_b = list(cell_b)
    new_e = list(cell_e)

    for i in range(n_cells):
        # Find best-resonance neighbor
        best_res = -1e30
        best_j = -1
        for j in neighbors[i]:
            res = resonance(cell_b[i], cell_e[i], cell_b[j], cell_e[j], m)
            if res > best_res:
                best_res = res
                best_j = j

        # Probabilistically copy best neighbor's state
        if best_j >= 0 and np.random.random() < adopt_prob:  # noqa: T2-D-5
            new_b[i] = cell_b[best_j]
            new_e[i] = cell_e[best_j]

    return new_b, new_e


# ─── EXPERIMENT ───────────────────────────────────────────────────────

print("=" * 70)
print("QA BraiNCA SELF-ORGANIZATION v2b — Resonance-Weighted Contagion")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, mod-{M}, {N_STEPS} steps, {N_TRIALS} trials")
print(f"adopt_prob={ADOPT_PROB}, NO organizers (purely emergent)")
print("=" * 70)

conditions = ['moore8', 'moore8_random_lr', 'moore8_struct_lr', 'moore24']

# Fair neighbors for clustering measurement (always moore8)
fair_nbrs = build_neighbors(GRID_SIZE, 'moore8')

# Storage: per-condition, per-trial curves
all_entropy_curves = {cond: [] for cond in conditions}
all_cluster_curves = {cond: [] for cond in conditions}

for cond in conditions:
    print(f"\n--- {cond} ---")

    # Need fresh random state for LR construction per condition
    # but each trial reseeds, so build neighbors once
    np.random.seed(100)  # noqa: T2-D-5
    nbrs = build_neighbors(GRID_SIZE, cond)

    for trial in range(N_TRIALS):
        np.random.seed(42 + trial)

        # Random initialization: each cell gets random (b,e) in {1,...,M}
        cell_b = [int(np.random.randint(1, M + 1)) for _ in range(N_CELLS)]
        cell_e = [int(np.random.randint(1, M + 1)) for _ in range(N_CELLS)]

        ent_curve = []
        clust_curve = []

        for step in range(N_STEPS):
            cell_b, cell_e = resonance_step(
                cell_b, cell_e, nbrs, adopt_prob=ADOPT_PROB, m=M
            )

            # Entropy (observer projection — float)
            orbit_counts = Counter(
                classify_orbit(cell_b[i], cell_e[i], M) for i in range(N_CELLS)
            )
            ent = shannon_entropy(orbit_counts, N_CELLS)
            ent_curve.append(ent)

            # Spatial clustering (observer projection — float)
            clust = spatial_clustering_fair(cell_b, cell_e, fair_nbrs, M)
            clust_curve.append(clust)

        all_entropy_curves[cond].append(ent_curve)
        all_cluster_curves[cond].append(clust_curve)

    # Summary
    ent5 = [curve[4] for curve in all_entropy_curves[cond]]   # step 5
    ent10 = [curve[9] for curve in all_entropy_curves[cond]]  # step 10
    clust10 = [curve[9] for curve in all_cluster_curves[cond]]
    print(f"  Entropy step 5:    {np.mean(ent5):.3f} +/- {np.std(ent5):.3f}")
    print(f"  Entropy step 10:   {np.mean(ent10):.3f} +/- {np.std(ent10):.3f}")
    print(f"  Clustering step 10: {np.mean(clust10):.3f} +/- {np.std(clust10):.3f}")


# ─── Statistical Comparisons ─────────────────────────────────────────

print("\n" + "=" * 70)
print("STATISTICAL COMPARISONS (observer projections)")
print("=" * 70)

# Entropy at step 5
ent5_8 = [curve[4] for curve in all_entropy_curves['moore8']]
ent5_24 = [curve[4] for curve in all_entropy_curves['moore24']]
t_ent5, p_ent5 = stats.ttest_ind(ent5_24, ent5_8)
print(f"\nEntropy step 5:")
print(f"  moore24: {np.mean(ent5_24):.3f}  moore8: {np.mean(ent5_8):.3f}")
print(f"  t={t_ent5:.3f}, p={p_ent5:.2e}")
print(f"  Expected: 24=0.946, 8=1.168 (p<0.0001)")

# Entropy at step 10
ent10_8 = [curve[9] for curve in all_entropy_curves['moore8']]
ent10_24 = [curve[9] for curve in all_entropy_curves['moore24']]
t_ent10, p_ent10 = stats.ttest_ind(ent10_24, ent10_8)
print(f"\nEntropy step 10:")
print(f"  moore24: {np.mean(ent10_24):.3f}  moore8: {np.mean(ent10_8):.3f}")
print(f"  t={t_ent10:.3f}, p={p_ent10:.2e}")
print(f"  Expected: 24=0.357, 8=0.805 (p<0.0001)")

# Clustering at step 10
clust10_8 = [curve[9] for curve in all_cluster_curves['moore8']]
clust10_24 = [curve[9] for curve in all_cluster_curves['moore24']]
t_clust, p_clust = stats.ttest_ind(clust10_24, clust10_8)
print(f"\nClustering step 10:")
print(f"  moore24: {np.mean(clust10_24):.3f}  moore8: {np.mean(clust10_8):.3f}")
print(f"  t={t_clust:.3f}, p={p_clust:.2e}")
print(f"  Expected: 24=0.936, 8=0.810 (p<0.0001)")

# Entropy decrease rate (steps 0-10)
print(f"\nEntropy decrease rate (step 0 -> step 10):")
for cond in conditions:
    ent0 = [curve[0] for curve in all_entropy_curves[cond]]
    ent10_c = [curve[9] for curve in all_entropy_curves[cond]]
    rate = np.mean(ent0) - np.mean(ent10_c)
    print(f"  {cond:20s}: rate = +{rate:.3f} (higher = faster convergence)")

# Long-range comparison
print(f"\nLong-range augmentation effects (entropy step 10):")
ent10_8r = [curve[9] for curve in all_entropy_curves['moore8_random_lr']]
ent10_8s = [curve[9] for curve in all_entropy_curves['moore8_struct_lr']]
t_lr_r, p_lr_r = stats.ttest_ind(ent10_8r, ent10_8)
t_lr_s, p_lr_s = stats.ttest_ind(ent10_8s, ent10_8)
print(f"  random_lr vs moore8: t={t_lr_r:.3f}, p={p_lr_r:.2e}")
print(f"  struct_lr vs moore8: t={t_lr_s:.3f}, p={p_lr_s:.2e}")

# ─── BraiNCA Comparison Summary ──────────────────────────────────────

print("\n" + "=" * 70)
print("BRAINCA COMPARISON SUMMARY")
print("=" * 70)

ent_rate_24 = np.mean([curve[0] for curve in all_entropy_curves['moore24']]) - \
              np.mean([curve[9] for curve in all_entropy_curves['moore24']])
ent_rate_8 = np.mean([curve[0] for curve in all_entropy_curves['moore8']]) - \
             np.mean([curve[9] for curve in all_entropy_curves['moore8']])

print(f"""
BraiNCA (Pio-Lopez et al. 2026):
  - Gradient-trained NCA with learned self-organization
  - Larger neighborhoods => faster convergence to target pattern

QA Self-Organization v2b (this experiment):
  - NO organizers, NO gradient training — purely emergent
  - Resonance-weighted contagion only (QA-native attention)
  - Entropy step 5:    24={np.mean(ent5_24):.3f} vs 8={np.mean(ent5_8):.3f}
  - Entropy step 10:   24={np.mean(ent10_24):.3f} vs 8={np.mean(ent10_8):.3f}
  - Clustering step 10: 24={np.mean(clust10_24):.3f} vs 8={np.mean(clust10_8):.3f}
  - Entropy decrease rate (0-10): 24=+{ent_rate_24:.3f}, 8=+{ent_rate_8:.3f}

Expected results:
  - Entropy step 5:  24=0.946 vs 8=1.168 (p<0.0001)
  - Entropy step 10: 24=0.357 vs 8=0.805 (p<0.0001)
  - Clustering step 10: 24=0.936 vs 8=0.810 (p<0.0001)
  - Entropy decrease rate: 24=+0.436, 8=-0.354
""")

print("DONE")
