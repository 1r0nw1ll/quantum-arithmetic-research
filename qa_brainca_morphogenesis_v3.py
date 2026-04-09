"""
QA Morphogenesis v3 — Continuous Morphogenetic Field
=====================================================

Maps to: Pio-Lopez, Hartl, Levin (2026) BraiNCA

v1 failed: one-shot injection, pure coupling. Wrong.
v2 failed: self-org speed, coupling too homogenizing. Wrong metric.
v3 insight: biology uses CONTINUOUS morphogenetic signaling, not one-shot.
  The bioelectric field IS the continuous signal. QA can do this:
  a subset of "organizer" cells continuously broadcast target state,
  and the rest self-organize toward it through coupling.

This is how real morphogenesis works (Levin's own model):
  - Bioelectric prepattern = persistent target signal
  - Gap junctions = coupling between cells
  - Cells read local signals and adjust toward the pattern
  - Larger neighborhood = faster propagation of pattern info

QA prediction: 24-neighbor propagates the morphogenetic signal
faster than 8-neighbor. Convergence ratio should approach ~2x.

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'accuracy, convergence_step -> float (observer projection)',
    'state_alphabet': '{1,...,9} (mod-9 integer QA states)',
    'discrete_layer': '(b,e) pairs on grid, int; organizer cells broadcast target',
    'observer_layer': 'pattern accuracy -> float (Theorem NT, measurement only)',
    'signal_injection': 'organizer cells continuously enforce target (b,e) values',
    'coupling': 'neighborhood averaging pulls toward local consensus',
}

import numpy as np
from collections import Counter
from scipy import stats

np.random.seed(42)

M = 9

def qa_step(b, e, m=M):
    """A1-compliant QA step."""
    return ((b + e - 1) % m) + 1, ((e + ((b + e - 1) % m) + 1 - 1) % m) + 1

def classify_orbit(b, e, m=M):
    """Classify orbit: 0=singularity, 1=satellite, 2=cosmos."""
    seen = []
    state = (b, e)
    for _ in range(m * m + 1):
        if state in seen:
            clen = len(seen) - seen.index(state)
            if clen == 1: return 0
            elif clen <= 8: return 1
            else: return 2
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    return -1


# ─── Target Pattern ──────────────────────────────────────────────────

def make_target(size=16):
    """
    Target pattern: 3 orbit types arranged spatially.
    Like BraiNCA's smiley — background/boundary/feature.
    Returns target (b,e) values for each cell.
    """
    target_b = [0] * (size * size)
    target_e = [0] * (size * size)
    target_class = [0] * (size * size)

    # Known (b,e) representatives for each orbit type
    cosmos_be = (1, 2)    # cosmos orbit
    satellite_be = (3, 3)  # satellite orbit
    singularity_be = (5, 5)  # near-singularity (center of mod-9)

    for r in range(size):
        for c in range(size):
            idx = r * size + c
            cr = r - size / 2
            cc = c - size / 2
            dist = (cr * cr + cc * cc)  # S1 compliant: no **

            if dist < (size * 0.2) * (size * 0.2):
                # Inner circle: cosmos (features)
                target_b[idx], target_e[idx] = cosmos_be
                target_class[idx] = 2
            elif dist < (size * 0.4) * (size * 0.4):
                # Ring: satellite (boundary)
                target_b[idx], target_e[idx] = satellite_be
                target_class[idx] = 1
            else:
                # Outer: singularity-adjacent (background)
                target_b[idx], target_e[idx] = singularity_be
                target_class[idx] = 0

    return target_b, target_e, target_class


# ─── QA Morphogenetic Grid ───────────────────────────────────────────

class QAMorphogeneticGrid:
    """
    Grid with organizer cells that continuously broadcast target state.
    Non-organizer cells self-organize through neighborhood coupling.
    """

    def __init__(self, size=16, m=M, neighborhood='moore8',
                 organizer_fraction=0.1):
        self.size = size
        self.m = m
        self.n = size * size

        # Random initialization
        self.b = [np.random.randint(1, m + 1) for _ in range(self.n)]
        self.e = [np.random.randint(1, m + 1) for _ in range(self.n)]

        # Target pattern
        self.target_b, self.target_e, self.target_class = make_target(size)

        # Organizer cells: randomly chosen subset that KNOWS the target
        # (like bioelectric prepattern cells that maintain the "memory")
        n_organizers = max(1, int(self.n * organizer_fraction))
        self.organizers = set(np.random.choice(self.n, n_organizers, replace=False))

        # Initialize organizers to target state
        for i in self.organizers:
            self.b[i] = self.target_b[i]
            self.e[i] = self.target_e[i]

        # Build neighborhoods
        self.neighbors = self._build_neighbors(neighborhood)

    def _idx(self, r, c):
        return (r % self.size) * self.size + (c % self.size)

    def _build_neighbors(self, ntype):
        nbrs = [[] for _ in range(self.n)]
        if ntype == 'moore8':
            offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        elif ntype == 'moore24':
            offsets = [(dr, dc) for dr in range(-2, 3)
                       for dc in range(-2, 3) if dr != 0 or dc != 0]
        elif ntype == 'moore8_random_lr':
            offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        elif ntype == 'moore8_struct_lr':
            offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        else:
            offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        for r in range(self.size):
            for c in range(self.size):
                idx = self._idx(r, c)
                for dr, dc in offsets:
                    nbrs[idx].append(self._idx(r + dr, c + dc))

        if ntype == 'moore8_random_lr':
            for i in range(self.n):
                for _ in range(3):
                    j = np.random.randint(0, self.n)
                    if j != i and j not in nbrs[i]:
                        nbrs[i].append(j)
        elif ntype == 'moore8_struct_lr':
            for i in range(self.n):
                ri, ci = i // self.size, i % self.size
                for j in [self._idx(ri + self.size//2, ci),
                          self._idx(ri, ci + self.size//2),
                          self._idx(ri + self.size//4, ci + self.size//4)]:
                    if j not in nbrs[i]:
                        nbrs[i].append(j)

        return nbrs

    def step(self, coupling=0.5):
        """One step: organizers hold, others couple to neighbors."""
        new_b = list(self.b)
        new_e = list(self.e)

        for i in range(self.n):
            if i in self.organizers:
                # Organizers HOLD target state (continuous morphogenetic signal)
                new_b[i] = self.target_b[i]
                new_e[i] = self.target_e[i]
                continue

            # Non-organizer: QA step + neighborhood coupling
            bi, ei = qa_step(self.b[i], self.e[i], self.m)

            # Pull toward neighborhood average
            n_nbrs = len(self.neighbors[i])
            if n_nbrs > 0:
                avg_b = sum(self.b[j] for j in self.neighbors[i]) / n_nbrs
                avg_e = sum(self.e[j] for j in self.neighbors[i]) / n_nbrs
                bi = int(round(bi * (1 - coupling) + avg_b * coupling))
                ei = int(round(ei * (1 - coupling) + avg_e * coupling))

            new_b[i] = max(1, min(self.m, bi))
            new_e[i] = max(1, min(self.m, ei))

        self.b = new_b
        self.e = new_e

    def accuracy(self):
        """Fraction of cells matching target orbit class (observer projection)."""
        correct = 0
        for i in range(self.n):
            pred = classify_orbit(self.b[i], self.e[i], self.m)
            if pred == self.target_class[i]:
                correct += 1
        return correct / self.n


# ─── EXPERIMENT ───────────────────────────────────────────────────────

print("=" * 70)
print("QA MORPHOGENESIS v3 — Continuous Morphogenetic Field")
print("Prediction: 24-neighbor propagates pattern faster than 8-neighbor")
print("=" * 70)

N_STEPS = 50
N_TRIALS = 30
THRESHOLD = 0.85  # convergence = 85% orbit-class accuracy

conditions = {
    'moore8': 'Moore-8 (3x3 vanilla)',
    'moore8_random_lr': 'Moore-8 + random LR',
    'moore8_struct_lr': 'Moore-8 + structured LR',
    'moore24': 'Moore-24 (5x5 vanilla)',
}

results = {}

for cond, label in conditions.items():
    print(f"\n--- {label} ---")
    conv_steps = []
    final_accs = []

    for trial in range(N_TRIALS):
        np.random.seed(42 + trial)
        grid = QAMorphogeneticGrid(size=16, m=M, neighborhood=cond,
                                    organizer_fraction=0.10)

        converged_at = N_STEPS
        accs = []
        for step in range(N_STEPS):
            grid.step(coupling=0.5)
            acc = grid.accuracy()
            accs.append(acc)
            if acc >= THRESHOLD and converged_at == N_STEPS:
                converged_at = step + 1

        conv_steps.append(converged_at)
        final_accs.append(accs[-1])

    results[cond] = {'conv': conv_steps, 'acc': final_accs}

    mean_c = np.mean(conv_steps)
    mean_a = np.mean(final_accs)
    n_conv = sum(1 for s in conv_steps if s < N_STEPS)
    print(f"  Convergence: {mean_c:.1f} ± {np.std(conv_steps):.1f} steps")
    print(f"  Final accuracy: {mean_a:.3f} ± {np.std(final_accs):.3f}")
    print(f"  Converged ({THRESHOLD:.0%}): {n_conv}/{N_TRIALS}")


# ─── Comparisons ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STATISTICAL COMPARISONS")
print("=" * 70)

s8 = results['moore8']['conv']
s24 = results['moore24']['conv']
s8r = results['moore8_random_lr']['conv']
s8s = results['moore8_struct_lr']['conv']

# 1. Main prediction: 24 faster than 8
ratio = np.mean(s8) / max(np.mean(s24), 0.01)
t1, p1 = stats.mannwhitneyu(s8, s24, alternative='greater')
print(f"\n1. Moore-8 vs Moore-24:")
print(f"   8-nbr:  {np.mean(s8):.1f} ± {np.std(s8):.1f} steps")
print(f"   24-nbr: {np.mean(s24):.1f} ± {np.std(s24):.1f} steps")
print(f"   Ratio: {ratio:.2f}x  (BraiNCA: 1.72x)")
print(f"   p={p1:.4f}")
print(f"   24 faster: {'CONFIRMED' if np.mean(s24) < np.mean(s8) else 'REJECTED'}")

# 2. Random LR effect
t2, p2 = stats.mannwhitneyu(s8r, s8, alternative='greater')
print(f"\n2. Random LR effect:")
print(f"   Vanilla: {np.mean(s8):.1f}, Random: {np.mean(s8r):.1f}")
print(f"   Random LR {'HURTS' if np.mean(s8r) > np.mean(s8) else 'HELPS'} (p={p2:.4f})")
print(f"   [191] predicts: HURTS")

# 3. Structured LR effect
t3, p3 = stats.mannwhitneyu(s8, s8s, alternative='greater')
print(f"\n3. Structured LR effect:")
print(f"   Vanilla: {np.mean(s8):.1f}, Structured: {np.mean(s8s):.1f}")
print(f"   Structured LR {'HELPS' if np.mean(s8s) < np.mean(s8) else 'HURTS'} (p={p3:.4f})")
print(f"   [191] predicts: HELPS")

# 4. Best vs worst
s_best = min([s24, s8s], key=lambda x: np.mean(x))
best_name = '24-nbr' if np.mean(s24) <= np.mean(s8s) else '8+struct'
ratio_bw = np.mean(s8) / max(np.mean(s_best), 0.01)
print(f"\n4. Best ({best_name}) vs Worst (8 vanilla):")
print(f"   Ratio: {ratio_bw:.2f}x  (BraiNCA: 2.19x)")

# 5. Final accuracy comparison
a8 = results['moore8']['acc']
a24 = results['moore24']['acc']
t5, p5 = stats.mannwhitneyu(a24, a8, alternative='greater')
print(f"\n5. Final Accuracy:")
print(f"   8-nbr:  {np.mean(a8):.3f}")
print(f"   24-nbr: {np.mean(a24):.3f}")
print(f"   p={p5:.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
v3 uses CONTINUOUS morphogenetic signaling: 10% of cells are "organizers"
that persistently broadcast the target (b,e) state. The remaining 90%
self-organize through neighborhood coupling. This matches Levin's actual
model: bioelectric prepattern = persistent signal, gap junctions = coupling.

The question: does 24-neighbor propagate the pattern faster than 8-neighbor?
If yes, this confirms pi(9)=24 as the information-propagation bridge.
""")

print("Script complete.")
