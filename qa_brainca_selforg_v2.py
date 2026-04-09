"""
QA Self-Organization Speed Test v2
===================================

Maps to: Pio-Lopez, Hartl, Levin (2026) arXiv:2604.01932 (BraiNCA)

v1 FAILED: pure QA coupling cannot replicate gradient-trained convergence.
v2 REDESIGN: instead of pattern accuracy, measure SELF-ORGANIZATION SPEED.

Question: How quickly does an initially random grid develop structured
orbit patterns under different neighborhood sizes?

QA Prediction: 24-neighbor grids self-organize faster than 8-neighbor
grids because pi(9)=24 means larger neighborhoods capture more of the
algebraic structure. The convergence ratio should approximate 24/8 = 3x
or the empirical BraiNCA ratio of ~2x.

Metrics:
  1. Time to stable orbit census (orbit counts stop changing)
  2. Orbit entropy decrease rate (Shannon entropy of orbit distribution)
  3. Spatial clustering (Moran's I of orbit types on grid)

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'orbit_entropy, morans_I, stability_time -> float (observer projection)',
    'state_alphabet': '{1,...,9} (mod-9 integer QA states)',
    'discrete_layer': '(b,e) pairs on 16x16 grid, int; self-organizing under coupling',
    'observer_layer': 'entropy, spatial stats -> float (measurement only, Theorem NT)',
    'signal_injection': 'none (random initialization, emergent self-organization)',
    'coupling': 'neighborhood-based averaging (8 or 24 neighbors)',
}

import numpy as np
from collections import Counter
from scipy import stats

np.random.seed(42)

M = 9

def qa_step(b, e, m=M):
    """A1-compliant QA step."""
    b_new = ((b + e - 1) % m) + 1
    e_new = ((e + b_new - 1) % m) + 1
    return b_new, e_new

def classify_orbit(b, e, m=M):
    """Classify orbit type by cycle length."""
    seen = []
    state = (b, e)
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


def shannon_entropy(counts_dict, total):
    """Shannon entropy of a distribution (observer projection — float)."""
    H = 0.0
    for c in counts_dict.values():
        if c > 0:
            p = c / total
            H -= p * np.log2(p)  # noqa: T2-D-1 observer measurement
    return H


def morans_I(grid, size, neighbors_list):
    """
    Moran's I spatial autocorrelation (observer projection — float).
    Measures spatial clustering of orbit types on the grid.
    +1 = perfect clustering, 0 = random, -1 = dispersed.
    """
    n = size * size
    x = grid.astype(float)
    x_bar = np.mean(x)
    if np.var(x) == 0:
        return 0.0

    numerator = 0.0
    W = 0.0
    for i in range(n):
        for j in neighbors_list[i]:
            w_ij = 1.0
            numerator += w_ij * (x[i] - x_bar) * (x[j] - x_bar)
            W += w_ij

    denominator = np.sum((x - x_bar) * (x - x_bar))  # noqa: S1
    if denominator == 0 or W == 0:
        return 0.0

    I = (n / W) * (numerator / denominator)
    return I


class QASelfOrgGrid:
    """QA grid that self-organizes under neighborhood coupling."""

    def __init__(self, size=16, m=M, neighborhood='moore8'):
        self.size = size
        self.m = m
        self.n = size * size

        # Random initialization — FULLY random (b,e) in {1,...,m}
        self.b = [np.random.randint(1, m + 1) for _ in range(self.n)]
        self.e = [np.random.randint(1, m + 1) for _ in range(self.n)]

        # Build neighborhoods
        self.nbr_type = neighborhood
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

        # Add long-range for specific types
        if ntype == 'moore8_random_lr':
            for i in range(self.n):
                for _ in range(3):
                    j = np.random.randint(0, self.n)
                    if j != i and j not in nbrs[i]:
                        nbrs[i].append(j)
        elif ntype == 'moore8_struct_lr':
            for i in range(self.n):
                ri, ci = i // self.size, i % self.size
                targets = [
                    self._idx(ri + self.size // 2, ci),
                    self._idx(ri, ci + self.size // 2),
                    self._idx(ri + self.size // 4, ci + self.size // 4),
                ]
                for j in targets:
                    if j not in nbrs[i]:
                        nbrs[i].append(j)

        return nbrs

    def step(self, coupling=0.4):
        """One QA timestep with neighborhood coupling."""
        new_b = [0] * self.n
        new_e = [0] * self.n

        for i in range(self.n):
            bi, ei = qa_step(self.b[i], self.e[i], self.m)

            if self.neighbors[i]:
                n_nbrs = len(self.neighbors[i])
                avg_b = sum(self.b[j] for j in self.neighbors[i]) / n_nbrs
                avg_e = sum(self.e[j] for j in self.neighbors[i]) / n_nbrs
                bi = int(round(bi + coupling * (avg_b - bi)))
                ei = int(round(ei + coupling * (avg_e - ei)))

            new_b[i] = max(1, min(self.m, bi))
            new_e[i] = max(1, min(self.m, ei))

        self.b = new_b
        self.e = new_e

    def get_orbit_grid(self):
        """Classify each cell's orbit type."""
        return np.array([classify_orbit(self.b[i], self.e[i], self.m)
                        for i in range(self.n)])

    def orbit_census(self):
        """Count cells by orbit type."""
        grid = self.get_orbit_grid()
        return Counter(grid.tolist())

    def orbit_entropy(self):
        """Shannon entropy of orbit distribution (observer projection)."""
        census = self.orbit_census()
        return shannon_entropy(census, self.n)

    def spatial_clustering(self):
        """Moran's I of orbit types (observer projection)."""
        grid = self.get_orbit_grid()
        return morans_I(grid, self.size, self.neighbors)


# ─── EXPERIMENT: Self-Organization Speed ──────────────────────────────

print("=" * 70)
print("QA SELF-ORGANIZATION SPEED TEST v2")
print("Prediction: 24-neighbor grids self-organize faster than 8-neighbor")
print("           (pi(9)=24 bridge; BraiNCA reported ~2x speedup)")
print("=" * 70)

N_STEPS = 60
N_TRIALS = 25
STABILITY_WINDOW = 5  # census must be stable for this many steps

conditions = {
    'moore8': 'Moore-8 (3x3 vanilla)',
    'moore8_random_lr': 'Moore-8 + random LR',
    'moore8_struct_lr': 'Moore-8 + structured LR',
    'moore24': 'Moore-24 (5x5 vanilla)',
}

results = {}

for cond, label in conditions.items():
    print(f"\n--- {label} ---")

    stability_times = []
    final_entropies = []
    final_morans = []
    entropy_curves = []

    for trial in range(N_TRIALS):
        np.random.seed(42 + trial)
        grid = QASelfOrgGrid(size=16, m=M, neighborhood=cond)

        # Track orbit census over time
        history = []
        for step in range(N_STEPS):
            grid.step(coupling=0.4)
            census = grid.orbit_census()
            entropy = grid.orbit_entropy()
            history.append({'census': dict(census), 'entropy': entropy, 'step': step})

        # Measure stability time: first step where census stays constant for STABILITY_WINDOW
        stability_time = N_STEPS
        for t in range(len(history) - STABILITY_WINDOW):
            window = history[t:t + STABILITY_WINDOW]
            if all(w['census'] == window[0]['census'] for w in window):
                stability_time = t
                break

        stability_times.append(stability_time)
        final_entropies.append(history[-1]['entropy'])
        final_morans.append(grid.spatial_clustering())
        entropy_curves.append([h['entropy'] for h in history])

    results[cond] = {
        'stability': stability_times,
        'entropy': final_entropies,
        'morans': final_morans,
        'entropy_curves': entropy_curves,
    }

    mean_stab = np.mean(stability_times)
    mean_ent = np.mean(final_entropies)
    mean_mor = np.mean(final_morans)
    print(f"  Stability time: {mean_stab:.1f} ± {np.std(stability_times):.1f} steps")
    print(f"  Final entropy:  {mean_ent:.3f} ± {np.std(final_entropies):.3f}")
    print(f"  Moran's I:      {mean_mor:.3f} ± {np.std(final_morans):.3f}")
    print(f"  Final census example: {history[-1]['census']}")


# ─── Statistical Comparisons ─────────────────────────────────────────

print("\n" + "=" * 70)
print("STATISTICAL COMPARISONS")
print("=" * 70)

s8 = results['moore8']['stability']
s24 = results['moore24']['stability']

# 1. Stability time: 24 should be FASTER (lower)
t1, p1 = stats.mannwhitneyu(s8, s24, alternative='greater')
ratio = np.mean(s8) / max(np.mean(s24), 0.01)
print(f"\n1. Stability Time — Moore-8 vs Moore-24:")
print(f"   8-nbr: {np.mean(s8):.1f} ± {np.std(s8):.1f}")
print(f"   24-nbr: {np.mean(s24):.1f} ± {np.std(s24):.1f}")
print(f"   Ratio: {ratio:.2f}x  (BraiNCA: ~2x)")
print(f"   Mann-Whitney U p={p1:.4f}")
print(f"   Prediction (24 faster): {'CONFIRMED' if np.mean(s24) < np.mean(s8) else 'REJECTED'}")

# 2. Entropy: 24 should have LOWER final entropy (more organized)
e8 = results['moore8']['entropy']
e24 = results['moore24']['entropy']
t2, p2 = stats.mannwhitneyu(e8, e24, alternative='greater')
print(f"\n2. Final Entropy — Moore-8 vs Moore-24:")
print(f"   8-nbr: {np.mean(e8):.3f}")
print(f"   24-nbr: {np.mean(e24):.3f}")
print(f"   Mann-Whitney U p={p2:.4f}")
print(f"   Prediction (24 lower entropy): {'CONFIRMED' if np.mean(e24) < np.mean(e8) else 'REJECTED'}")

# 3. Moran's I: 24 should have HIGHER spatial clustering
m8 = results['moore8']['morans']
m24 = results['moore24']['morans']
t3, p3 = stats.mannwhitneyu(m24, m8, alternative='greater')
print(f"\n3. Spatial Clustering (Moran's I) — Moore-8 vs Moore-24:")
print(f"   8-nbr: {np.mean(m8):.3f}")
print(f"   24-nbr: {np.mean(m24):.3f}")
print(f"   Mann-Whitney U p={p3:.4f}")
print(f"   Prediction (24 higher clustering): {'CONFIRMED' if np.mean(m24) > np.mean(m8) else 'REJECTED'}")

# 4. Random LR effect
s8r = results['moore8_random_lr']['stability']
t4, p4 = stats.mannwhitneyu(s8r, s8, alternative='greater')
print(f"\n4. Random LR Effect on Self-Organization:")
print(f"   Vanilla: {np.mean(s8):.1f}, Random LR: {np.mean(s8r):.1f}")
print(f"   Random LR {'SLOWS' if np.mean(s8r) > np.mean(s8) else 'SPEEDS'} self-org (p={p4:.4f})")
print(f"   [191] predicts: SLOWS (unstructured L2)")

# 5. Structured LR effect
s8s = results['moore8_struct_lr']['stability']
t5, p5 = stats.mannwhitneyu(s8, s8s, alternative='greater')
print(f"\n5. Structured LR Effect on Self-Organization:")
print(f"   Vanilla: {np.mean(s8):.1f}, Structured LR: {np.mean(s8s):.1f}")
print(f"   Structured LR {'SPEEDS' if np.mean(s8s) < np.mean(s8) else 'SLOWS'} self-org (p={p5:.4f})")
print(f"   [191] predicts: SPEEDS (structured L2)")

# 6. Entropy decrease rate (first 10 steps)
print(f"\n6. Entropy Decrease Rate (first 10 steps):")
for cond, label in conditions.items():
    curves = results[cond]['entropy_curves']
    rates = []
    for curve in curves:
        if len(curve) >= 10 and curve[0] > 0:
            rate = (curve[0] - curve[9]) / curve[0]
            rates.append(rate)
    if rates:
        print(f"   {label:30s}: {np.mean(rates):.3f} ± {np.std(rates):.3f}")


# ─── SUMMARY ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
This v2 test measures SELF-ORGANIZATION SPEED instead of pattern accuracy.
An initially random grid evolves under QA dynamics with neighborhood coupling.

Key question: Does larger neighborhood (24 vs 8) accelerate the emergence
of structured orbit patterns?

QA prediction: YES, because pi(9)=24 means the 24-neighbor system captures
the full applied modulus structure, while 8-neighbor captures only the
satellite orbit substructure.

BraiNCA parallel: Their 5x5 (24 neighbors) converged 1.72x faster than
3x3 (8 neighbors) on morphogenesis. With structured long-range connections,
2.19x faster. Our self-organization metric should show a similar ratio.
""")

print("Script complete.")
