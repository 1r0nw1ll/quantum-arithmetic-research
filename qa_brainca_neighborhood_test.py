"""
QA-BraiNCA Neighborhood Convergence Test
=========================================

Maps to: Pio-Lopez, Hartl, Levin (2026) arXiv:2604.01932
"BraiNCA: brain-inspired neural cellular automata"

QA Prediction: 24-neighbor (mod-24) outperforms 8-neighbor (mod-9/Satellite)
  by a factor related to pi(9)=24 (Pisano period bridge).
  BraiNCA reported: 2.19x faster convergence for 5x5+LR vs 3x3 vanilla.

Tests:
  1. 8 vs 24 neighbor convergence speed on morphogenesis-like task
  2. Random long-range links HURT (Tiered Reachability [191])
  3. Structured long-range links HELP

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'convergence_speed, accuracy -> float (observer projection)',
    'state_alphabet': '{1,...,9} and {1,...,24} (mod-9 and mod-24 QA states)',
    'discrete_layer': '(b,e) pairs on grid, int; orbit classification by neighborhood',
    'observer_layer': 'loss, accuracy -> float (measurement only, Theorem NT)',
    'signal_injection': 'target pattern enters via initial b configuration',
    'coupling': 'neighborhood-based resonance (8 or 24 neighbors)',
}

import numpy as np
from collections import Counter
from scipy import stats

np.random.seed(42)

# ─── QA Core ──────────────────────────────────────────────────────────

def qa_step(b, e, m):
    """A1-compliant QA step."""
    b_new = ((b + e - 1) % m) + 1
    e_new = ((e + b_new - 1) % m) + 1
    return b_new, e_new

def classify_orbit(b, e, m):
    """Classify by cycle length."""
    seen = []
    state = (b, e)
    for _ in range(m * m + 1):
        if state in seen:
            cycle_len = len(seen) - seen.index(state)
            if cycle_len == 1:
                return 'singularity'
            elif cycle_len <= 8:
                return 'satellite'
            else:
                return 'cosmos'
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    return 'unknown'


# ─── Grid-Based QA Cellular Automaton ─────────────────────────────────

class QACellularAutomaton:
    """
    QA-based cellular automaton on a 2D grid.
    Each cell has state (b, e) evolving under QA dynamics.
    Neighbors influence via resonance coupling.
    """

    def __init__(self, grid_size=16, m=9, neighborhood='moore8',
                 long_range=None, long_range_type='random'):
        self.size = grid_size
        self.m = m
        self.n_cells = grid_size * grid_size

        # Initialize all cells near center of state space
        center = m // 2 + 1
        self.b = np.array([max(1, min(m, center + np.random.randint(-2, 3)))
                           for _ in range(self.n_cells)], dtype=int)
        self.e = np.array([max(1, min(m, center + np.random.randint(-2, 3)))
                           for _ in range(self.n_cells)], dtype=int)

        # Build neighborhood structure
        self.neighbors = self._build_neighbors(neighborhood)

        # Add long-range connections if specified
        if long_range is not None:
            self._add_long_range(long_range, long_range_type)

    def _idx(self, r, c):
        """2D → 1D index with wrapping."""
        return (r % self.size) * self.size + (c % self.size)

    def _build_neighbors(self, ntype):
        """Build neighbor lists for each cell."""
        nbrs = [[] for _ in range(self.n_cells)]

        if ntype == 'moore8':
            # 3x3 Moore neighborhood = 8 neighbors
            offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        elif ntype == 'moore24':
            # 5x5 Moore neighborhood = 24 neighbors
            offsets = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr != 0 or dc != 0:
                        offsets.append((dr, dc))
        else:
            raise ValueError(f"Unknown neighborhood: {ntype}")

        for r in range(self.size):
            for c in range(self.size):
                idx = self._idx(r, c)
                for dr, dc in offsets:
                    nbrs[idx].append(self._idx(r + dr, c + dc))

        return nbrs

    def _add_long_range(self, n_connections, lr_type):
        """Add long-range connections."""
        if lr_type == 'random':
            # Random long-range = unstructured L2 (should HURT per [191])
            for i in range(self.n_cells):
                for _ in range(n_connections):
                    j = np.random.randint(0, self.n_cells)
                    if j != i and j not in self.neighbors[i]:
                        self.neighbors[i].append(j)

        elif lr_type == 'structured':
            # Structured long-range = somatotopic (should HELP)
            # Connect cells to their orbit-matched partners across the grid
            for i in range(self.n_cells):
                ri, ci = i // self.size, i % self.size
                # Connect to same-position cells in other quadrants
                targets = [
                    self._idx(ri + self.size // 2, ci),
                    self._idx(ri, ci + self.size // 2),
                    self._idx(ri + self.size // 2, ci + self.size // 2),
                ]
                for j in targets[:n_connections]:
                    if j not in self.neighbors[i]:
                        self.neighbors[i].append(j)

    def set_target(self, target_pattern):
        """Set target pattern (3 classes: 0=background, 1=boundary, 2=feature)."""
        self.target = target_pattern

    def step(self, coupling=0.3):
        """One QA timestep with neighborhood coupling."""
        new_b = list(self.b)
        new_e = list(self.e)

        for i in range(self.n_cells):
            # Local QA step
            bi, ei = qa_step(int(self.b[i]), int(self.e[i]), self.m)

            # Neighborhood coupling
            if self.neighbors[i]:
                avg_b = sum(int(self.b[j]) for j in self.neighbors[i]) / len(self.neighbors[i])
                avg_e = sum(int(self.e[j]) for j in self.neighbors[i]) / len(self.neighbors[i])

                # Coupling pulls toward neighborhood average
                bi = int(round(bi + coupling * (avg_b - bi)))
                ei = int(round(ei + coupling * (avg_e - ei)))

            new_b[i] = max(1, min(self.m, bi))
            new_e[i] = max(1, min(self.m, ei))

        self.b = np.array(new_b, dtype=int)
        self.e = np.array(new_e, dtype=int)

    def get_pattern(self):
        """Classify each cell into pattern class by orbit type."""
        pattern = []
        for i in range(self.n_cells):
            otype = classify_orbit(int(self.b[i]), int(self.e[i]), self.m)
            if otype == 'singularity':
                pattern.append(0)  # background
            elif otype == 'satellite':
                pattern.append(1)  # boundary
            else:
                pattern.append(2)  # feature (cosmos)
        return np.array(pattern)

    def accuracy(self):
        """Pixel-wise accuracy vs target (observer projection — float)."""
        pred = self.get_pattern()
        return np.mean(pred == self.target)

    def inject_target_signal(self, strength=2):
        """Inject target pattern as signal (via b state)."""
        for i in range(self.n_cells):
            if self.target[i] == 0:
                # Background → push toward singularity region
                center = self.m // 2 + 1
                self.b[i] = max(1, min(self.m, center))
                self.e[i] = max(1, min(self.m, center))
            elif self.target[i] == 1:
                # Boundary → push toward satellite
                self.b[i] = max(1, min(self.m, 3))
                self.e[i] = max(1, min(self.m, 3 if self.m == 9 else 6))
            else:
                # Feature → push toward cosmos
                self.b[i] = max(1, min(self.m, 1 + np.random.randint(0, 3)))
                self.e[i] = max(1, min(self.m, 1 + np.random.randint(0, 3)))


# ─── Target Pattern (Simple Smiley — like BraiNCA) ───────────────────

def make_smiley(size=16):
    """Create a simple smiley face target (3 classes)."""
    target = np.zeros(size * size, dtype=int)  # 0 = background

    for r in range(size):
        for c in range(size):
            idx = r * size + c
            # Circular face boundary
            dist = np.sqrt((r - size/2)**2 + (c - size/2)**2)  # noqa: T2-D-1 observer projection geometry
            if abs(dist - size * 0.35) < 1.5:
                target[idx] = 1  # boundary

            # Eyes (features)
            if (r - size//3)**2 + (c - size//3)**2 < 4:  # noqa: T2-D-1
                target[idx] = 2
            if (r - size//3)**2 + (c - 2*size//3)**2 < 4:  # noqa: T2-D-1
                target[idx] = 2

            # Mouth (feature)
            if size//2 < r < size//2 + 3 and size//4 < c < 3*size//4:
                target[idx] = 2

    return target


# ─── EXPERIMENT 1: 8 vs 24 Neighbor Convergence ──────────────────────

print("=" * 70)
print("QA-BraiNCA NEIGHBORHOOD CONVERGENCE TEST")
print("Prediction: 24-neighbor converges faster than 8-neighbor")
print("           (BraiNCA reported 2.19x for 5x5+LR vs 3x3)")
print("=" * 70)

target = make_smiley(16)
n_classes = len(set(target))
print(f"\nTarget pattern: {len(target)} cells, {n_classes} classes")
print(f"  Class distribution: {Counter(target)}")

N_TRIALS = 20
N_STEPS = 80
THRESHOLD = 0.6  # convergence threshold (lower than BraiNCA's 98% — our system is simpler)

conditions = {
    'moore8_vanilla': {'m': 9, 'neighborhood': 'moore8', 'long_range': None, 'lr_type': 'random'},
    'moore8_random_lr': {'m': 9, 'neighborhood': 'moore8', 'long_range': 3, 'lr_type': 'random'},
    'moore8_struct_lr': {'m': 9, 'neighborhood': 'moore8', 'long_range': 3, 'lr_type': 'structured'},
    'moore24_vanilla': {'m': 9, 'neighborhood': 'moore24', 'long_range': None, 'lr_type': 'random'},
    'moore24_random_lr': {'m': 9, 'neighborhood': 'moore24', 'long_range': 3, 'lr_type': 'random'},
    'moore24_struct_lr': {'m': 9, 'neighborhood': 'moore24', 'long_range': 3, 'lr_type': 'structured'},
}

convergence_results = {}

for cond_name, params in conditions.items():
    print(f"\n--- {cond_name} ---")
    convergence_steps = []
    final_accuracies = []

    for trial in range(N_TRIALS):
        np.random.seed(42 + trial)

        ca = QACellularAutomaton(
            grid_size=16, m=params['m'],
            neighborhood=params['neighborhood'],
            long_range=params['long_range'],
            long_range_type=params['lr_type']
        )
        ca.set_target(target)
        ca.inject_target_signal(strength=2)

        converged_at = N_STEPS  # default: didn't converge
        for step in range(N_STEPS):
            ca.step(coupling=0.3)
            acc = ca.accuracy()
            if acc >= THRESHOLD and converged_at == N_STEPS:
                converged_at = step + 1

        final_acc = ca.accuracy()
        convergence_steps.append(converged_at)
        final_accuracies.append(final_acc)

    mean_conv = np.mean(convergence_steps)
    std_conv = np.std(convergence_steps)
    mean_acc = np.mean(final_accuracies)
    n_converged = sum(1 for s in convergence_steps if s < N_STEPS)

    convergence_results[cond_name] = {
        'steps': convergence_steps,
        'mean': mean_conv,
        'std': std_conv,
        'final_acc': mean_acc,
        'n_converged': n_converged,
    }

    print(f"  Convergence: {mean_conv:.1f} ± {std_conv:.1f} steps")
    print(f"  Final accuracy: {mean_acc:.3f}")
    print(f"  Converged: {n_converged}/{N_TRIALS}")

# ─── Statistical Comparisons ─────────────────────────────────────────

print("\n" + "=" * 70)
print("STATISTICAL COMPARISONS")
print("=" * 70)

# Key comparison 1: 8 vs 24 vanilla
s8 = convergence_results['moore8_vanilla']['steps']
s24 = convergence_results['moore24_vanilla']['steps']
ratio_8_24 = np.mean(s8) / max(np.mean(s24), 0.01)
t, p = stats.mannwhitneyu(s8, s24, alternative='greater')
print(f"\n1. Moore-8 vs Moore-24 (vanilla):")
print(f"   8-nbr: {np.mean(s8):.1f} steps, 24-nbr: {np.mean(s24):.1f} steps")
print(f"   Ratio: {ratio_8_24:.2f}x  (BraiNCA reported: 1.72x)")
print(f"   Mann-Whitney U p={p:.4f}")

# Key comparison 2: Random LR hurts? (Tiered Reachability)
s8v = convergence_results['moore8_vanilla']['steps']
s8r = convergence_results['moore8_random_lr']['steps']
t2, p2 = stats.mannwhitneyu(s8r, s8v, alternative='greater')
print(f"\n2. Random LR effect on Moore-8:")
print(f"   Vanilla: {np.mean(s8v):.1f}, Random LR: {np.mean(s8r):.1f}")
print(f"   Random LR {'HURTS' if np.mean(s8r) > np.mean(s8v) else 'HELPS'} (p={p2:.4f})")
print(f"   [191] predicts: HURTS (unstructured L2 degrades performance)")

# Key comparison 3: Structured LR helps?
s8s = convergence_results['moore8_struct_lr']['steps']
t3, p3 = stats.mannwhitneyu(s8v, s8s, alternative='greater')
print(f"\n3. Structured LR effect on Moore-8:")
print(f"   Vanilla: {np.mean(s8v):.1f}, Structured LR: {np.mean(s8s):.1f}")
print(f"   Structured LR {'HELPS' if np.mean(s8s) < np.mean(s8v) else 'HURTS'} (p={p3:.4f})")
print(f"   [191] predicts: HELPS (structured L2 aligned to task)")

# Key comparison 4: Best config (24 + structured LR) vs worst (8 vanilla)
s_best = convergence_results['moore24_struct_lr']['steps']
ratio_best = np.mean(s8v) / max(np.mean(s_best), 0.01)
t4, p4 = stats.mannwhitneyu(s8v, s_best, alternative='greater')
print(f"\n4. Best (24+struct) vs Worst (8 vanilla):")
print(f"   8-vanilla: {np.mean(s8v):.1f}, 24+struct: {np.mean(s_best):.1f}")
print(f"   Ratio: {ratio_best:.2f}x  (BraiNCA reported: 2.19x)")
print(f"   Mann-Whitney U p={p4:.4f}")


# ─── SUMMARY ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: QA Predictions vs BraiNCA Results")
print("=" * 70)

print(f"""
BraiNCA (Pio-Lopez et al. 2026):
  3x3 vanilla:     658 episodes (baseline)
  3x3 + LR:        454 episodes (1.45x faster)
  5x5 vanilla:     384 episodes (1.72x faster)
  5x5 + LR:        301 episodes (2.19x faster)
  Random LR alone:  HURTS motor control (77% vs 84%)

QA Simulation (this script):
  8-nbr vanilla:    {np.mean(s8):.1f} steps (baseline)
  8-nbr + random:   {np.mean(s8r):.1f} steps
  8-nbr + struct:   {np.mean(s8s):.1f} steps
  24-nbr vanilla:   {np.mean(s24):.1f} steps
  24-nbr + struct:  {np.mean(s_best):.1f} steps

QA Structural Predictions:
  pi(9) = 24: mod-9 (8 neighbors) → mod-24 (24 neighbors) is the
  canonical theoretical→applied bridge [192]. Larger neighborhood
  captures more coupling structure.

  Tiered Reachability [191]: Only 26% of S_9 is L1-reachable.
  Unstructured L2 (random long-range) adds NOISE to the reachability
  structure. Structured L2 (orbit-aligned) expands USEFUL reachability.
""")

print("Script complete.")
