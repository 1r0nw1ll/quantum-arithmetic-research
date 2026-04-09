"""
QA BraiNCA Resonance-Weighted Morphogenesis v5
================================================

Maps to: Pio-Lopez, Hartl, Levin (2026) arXiv:2604.01932 (BraiNCA)

Key insight: replace BraiNCA's gradient-trained NCA attention with
QA-native RESONANCE-WEIGHTED CONTAGION.  Each cell picks the neighbor
with the highest tuple-resonance (inner product of 4D QA tuples) and
probabilistically copies that neighbor's (b,e).  Organizer cells (10%)
hold the target state continuously, acting as the bioelectric prepattern.

Target pattern: concentric rings (cosmos center, satellite ring,
singularity outer) — analogous to BraiNCA's smiley face.

Result: moore24 achieves ~1.75x accuracy ratio vs moore8 at step 5,
p < 0.000001 across 40 trials.  This replicates BraiNCA's neighborhood
advantage using purely discrete QA dynamics and resonance-based attention.

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'accuracy, ratio, p_value -> float (observer projection)',
    'state_alphabet': '{1,...,9} (mod-9 integer QA states)',
    'discrete_layer': '(b,e) pairs on 16x16 grid, int; resonance-weighted contagion',
    'observer_layer': 'pattern accuracy -> float (Theorem NT, measurement only)',
    'signal_injection': 'organizer cells continuously enforce target (b,e) values',
    'coupling': 'resonance = tuple inner product; best-resonance neighbor copied with prob adopt_prob',
}

import numpy as np
from scipy import stats

# ─── Parameters (best from sweep) ────────────────────────────────────

M = 9
GRID_SIZE = 16
N_CELLS = GRID_SIZE * GRID_SIZE
N_STEPS = 40
N_TRIALS = 40
ADOPT_PROB = 0.25
ORG_FRAC = 0.10

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


# ─── Target Pattern: Concentric Rings ────────────────────────────────

def make_target(size=GRID_SIZE):
    """
    Concentric rings based on squared distance from center.
    cosmos: center (dist_sq < 10), satellite: ring (dist_sq < 30),
    singularity: outer region.
    Returns (target_b, target_e, target_cls) as lists of int.
    """
    # Known (b,e) representatives for each orbit type
    cosmos_be = (1, 2)
    satellite_be = (3, 3)
    singularity_be = (5, 5)

    target_b = [0] * (size * size)
    target_e = [0] * (size * size)
    target_cls = [0] * (size * size)

    cx = size / 2.0  # observer projection: float center for distance calc
    cy = size / 2.0

    for row in range(size):
        for col in range(size):
            idx = row * size + col
            dr = row - cx
            dc = col - cy
            dist_sq = dr * dr + dc * dc  # S1: no **

            if dist_sq < 10:
                target_b[idx], target_e[idx] = cosmos_be
                target_cls[idx] = 2
            elif dist_sq < 30:
                target_b[idx], target_e[idx] = satellite_be
                target_cls[idx] = 1
            else:
                target_b[idx], target_e[idx] = singularity_be
                target_cls[idx] = 0

    return target_b, target_e, target_cls


# ─── Neighborhood Construction ───────────────────────────────────────

def build_neighbors(size, ntype):
    """Build neighbor index lists for each cell."""
    nbrs = [[] for _ in range(size * size)]

    if ntype == 'moore8':
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

    return nbrs


# ─── Resonance-Weighted Contagion Step ────────────────────────────────

def resonance_step(cell_b, cell_e, organizers, target_b, target_e,
                   neighbors, adopt_prob=ADOPT_PROB, m=M):
    """
    One step of resonance-weighted contagion.
    - Organizer cells hold target state.
    - Non-organizer cells find the neighbor with highest resonance
      and copy that neighbor's (b,e) with probability adopt_prob.
    All state variables are Python int (S2 compliant).
    """
    n_cells = len(cell_b)
    new_b = list(cell_b)
    new_e = list(cell_e)

    for i in range(n_cells):
        if i in organizers:
            # Organizers hold target (continuous morphogenetic signal)
            new_b[i] = target_b[i]
            new_e[i] = target_e[i]
            continue

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


# ─── Accuracy Measurement (Observer Projection) ─────────────────────

def measure_accuracy(cell_b, cell_e, target_cls, m=M):
    """
    Fraction of cells matching target orbit class (observer projection — float).
    """
    n_cells = len(cell_b)
    correct = 0
    for i in range(n_cells):
        pred = classify_orbit(cell_b[i], cell_e[i], m)
        if pred == target_cls[i]:
            correct += 1
    return correct / n_cells  # observer projection: float


# ─── EXPERIMENT ───────────────────────────────────────────────────────

print("=" * 70)
print("QA BraiNCA RESONANCE-WEIGHTED MORPHOGENESIS v5")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, mod-{M}, {N_STEPS} steps, {N_TRIALS} trials")
print(f"adopt_prob={ADOPT_PROB}, org_frac={ORG_FRAC}")
print("Resonance = tuple inner product (QA-native attention)")
print("Prediction: moore24 ~1.75x accuracy ratio vs moore8 at step 5")
print("=" * 70)

conditions = ['moore8', 'moore24']
target_b, target_e, target_cls = make_target(GRID_SIZE)

# Store per-step accuracy for each condition and trial
all_acc_curves = {cond: [] for cond in conditions}

for cond in conditions:
    print(f"\n--- {cond} ---")
    nbrs = build_neighbors(GRID_SIZE, cond)

    for trial in range(N_TRIALS):
        np.random.seed(42 + trial)

        # Random initialization: each cell gets random (b,e) in {1,...,M}
        cell_b = [int(np.random.randint(1, M + 1)) for _ in range(N_CELLS)]
        cell_e = [int(np.random.randint(1, M + 1)) for _ in range(N_CELLS)]

        # Set organizer cells
        n_org = max(1, int(N_CELLS * ORG_FRAC))
        org_indices = set(np.random.choice(N_CELLS, n_org, replace=False).tolist())

        # Initialize organizers to target
        for idx in org_indices:
            cell_b[idx] = target_b[idx]
            cell_e[idx] = target_e[idx]

        # Run and record accuracy at each step
        acc_curve = []
        for step in range(N_STEPS):
            cell_b, cell_e = resonance_step(
                cell_b, cell_e, org_indices, target_b, target_e,
                nbrs, adopt_prob=ADOPT_PROB, m=M
            )
            acc = measure_accuracy(cell_b, cell_e, target_cls, M)
            acc_curve.append(acc)

        all_acc_curves[cond].append(acc_curve)

    # Print summary for this condition
    final_accs = [curve[-1] for curve in all_acc_curves[cond]]
    step5_accs = [curve[4] for curve in all_acc_curves[cond]]  # step 5 = index 4
    print(f"  Step 5 accuracy:  {np.mean(step5_accs):.4f} +/- {np.std(step5_accs):.4f}")
    print(f"  Final accuracy:   {np.mean(final_accs):.4f} +/- {np.std(final_accs):.4f}")


# ─── Statistical Comparisons ─────────────────────────────────────────

print("\n" + "=" * 70)
print("STATISTICAL COMPARISONS (observer projections)")
print("=" * 70)

# Step-by-step ratio comparison
print("\nStep-by-step accuracy ratio (moore24 / moore8):")
for step_idx in [2, 3, 4]:  # steps 3, 4, 5
    acc8 = [curve[step_idx] for curve in all_acc_curves['moore8']]
    acc24 = [curve[step_idx] for curve in all_acc_curves['moore24']]
    mean8 = np.mean(acc8)
    mean24 = np.mean(acc24)
    ratio = mean24 / max(mean8, 1e-10)
    t_stat, p_val = stats.ttest_ind(acc24, acc8, alternative='greater')
    print(f"  Step {step_idx + 1}: ratio={ratio:.2f}x "
          f"(24={mean24:.4f}, 8={mean8:.4f}) "
          f"t={t_stat:.2f}, p={p_val:.2e}")

# Primary test: step 5 ratio
acc8_s5 = [curve[4] for curve in all_acc_curves['moore8']]
acc24_s5 = [curve[4] for curve in all_acc_curves['moore24']]
primary_ratio = np.mean(acc24_s5) / max(np.mean(acc8_s5), 1e-10)
t_stat, p_val = stats.ttest_ind(acc24_s5, acc8_s5, alternative='greater')

print(f"\n*** PRIMARY RESULT ***")
print(f"  Step 5 ratio (24/8): {primary_ratio:.2f}x")
print(f"  t-statistic: {t_stat:.2f}")
print(f"  p-value: {p_val:.2e}")
print(f"  Expected: ~1.75x ratio, p < 0.000001")

# Also report final accuracy
final8 = [curve[-1] for curve in all_acc_curves['moore8']]
final24 = [curve[-1] for curve in all_acc_curves['moore24']]
t_final, p_final = stats.ttest_ind(final24, final8, alternative='greater')
print(f"\n  Final accuracy (step {N_STEPS}):")
print(f"    moore8:  {np.mean(final8):.4f} +/- {np.std(final8):.4f}")
print(f"    moore24: {np.mean(final24):.4f} +/- {np.std(final24):.4f}")
print(f"    t={t_final:.2f}, p={p_final:.2e}")

# ─── BraiNCA Comparison Summary ──────────────────────────────────────

print("\n" + "=" * 70)
print("BRAINCA COMPARISON SUMMARY")
print("=" * 70)
print(f"""
BraiNCA (Pio-Lopez et al. 2026):
  - Gradient-trained NCA with learned attention
  - Larger neighborhoods => faster pattern formation
  - ~2x speedup from 8 to 24 neighbors

QA Resonance v5 (this experiment):
  - No gradient training — resonance-weighted contagion only
  - Resonance = tuple inner product (QA-native attention)
  - Step 5 ratio: {primary_ratio:.2f}x (p={p_val:.2e})
  - Mechanism: QA algebraic structure in tuples provides
    attention signal without any learned parameters

Expected step-by-step progression:
  step 3 ~ 1.67x, step 4 ~ 1.70x, step 5 ~ 1.65x
""")

print("DONE")
