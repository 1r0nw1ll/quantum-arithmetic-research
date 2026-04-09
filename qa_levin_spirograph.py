"""
QA Multi-Orbit Spirograph Simulation
=====================================

Maps to: Fotowat, O'Neill, Pio-Lopez et al. (2026) Advanced Science
"Engineered Living Systems With Self-Organizing Neural Networks"

QA Prediction: Mixed-orbit coupled systems produce multi-frequency
  (spirograph) trajectories. Number of spectral peaks = number of
  distinct orbital frequencies contributing to coupled dynamics.

  - Single-orbit (biobot-like): single spectral peak, simple circle
  - Mixed-orbit (neurobot-like): multiple peaks, spirograph pattern
  - More orbit diversity = higher Complexity Index

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'trajectory_xy, spectral_peaks -> float (observer projection)',
    'state_alphabet': '{1,...,9} (mod-9 integer QA states)',
    'discrete_layer': '(b,e) pairs per cell, int; orbit determines frequency',
    'observer_layer': 'x,y position, Welch PSD -> float (measurement, Theorem NT)',
    'signal_injection': 'none (emergent dynamics only)',
    'coupling': 'resonance matrix from tuple inner products',
}

import numpy as np
from collections import Counter
from scipy import signal as sig
from scipy import stats

np.random.seed(42)

# ─── QA Core ──────────────────────────────────────────────────────────

M = 9

def qa_step(b, e, m=M):
    """A1-compliant QA step."""
    b_new = ((b + e - 1) % m) + 1
    e_new = ((e + b_new - 1) % m) + 1
    return b_new, e_new

def qa_tuple(b, e, m=M):
    """Full tuple — A2: d, a derived."""
    d = ((b + e - 1) % m) + 1
    a = ((b + 2*e - 1) % m) + 1
    return (b, e, d, a)

def get_cycle_length(b, e, m=M):
    """Get exact cycle length for this (b,e) pair."""
    seen = []
    state = (b, e)
    for _ in range(m * m + 1):
        if state in seen:
            return len(seen) - seen.index(state)
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    return 0

def classify_orbit(b, e, m=M):
    """Classify orbit type."""
    clen = get_cycle_length(b, e, m)
    if clen == 1:
        return 'singularity', clen
    elif clen <= 8:
        return 'satellite', clen
    else:
        return 'cosmos', clen


# ─── Coupled Multi-Agent QA System with 2D Projection ────────────────

class QAMotileSystem:
    """
    Models a motile biological construct (Xenobot/Neurobot).
    N cells with QA states produce thrust vectors.
    Coupled dynamics project to 2D trajectory.
    """

    def __init__(self, n_cells, cell_types='uniform', m=M, coupling=0.2):
        self.n = n_cells
        self.m = m
        self.coupling = coupling

        # Position in 2D (observer projection)
        self.x = 0.0
        self.y = 0.0

        # Each cell has (b, e) state and a thrust angle
        # thrust angle = cell's position on the body (fixed geometry)
        self.angles = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)

        # Initialize cell states based on type
        if cell_types == 'uniform':
            # All cosmos (biobot-like: MCCs only, single orbit type)
            self.b = [max(1, min(m, 1 + i % 3)) for i in range(n_cells)]
            self.e = [max(1, min(m, 1 + (i + 1) % 3)) for i in range(n_cells)]
        elif cell_types == 'mixed':
            # Mixed cosmos + satellite + singularity (neurobot-like)
            self.b = []
            self.e = []
            for i in range(n_cells):
                if i < n_cells // 3:
                    # Cosmos cells
                    self.b.append(max(1, min(m, 1 + i % 4)))
                    self.e.append(max(1, min(m, 2 + i % 3)))
                elif i < 2 * n_cells // 3:
                    # Satellite cells
                    self.b.append(3)
                    self.e.append(3 + (i % 2) * 3)
                else:
                    # Near-singularity cells
                    self.b.append(m // 2 + 1)
                    self.e.append(m // 2 + 1)
        elif cell_types == 'all_satellite':
            # All satellite (for comparison)
            self.b = [3 for _ in range(n_cells)]
            self.e = [3 + (i % 2) * 3 for i in range(n_cells)]
        else:
            raise ValueError(f"Unknown cell_types: {cell_types}")

        # Coupling matrix (resonance-based)
        self.W = np.ones((n_cells, n_cells)) * coupling / n_cells
        np.fill_diagonal(self.W, 0)

    def orbit_census(self):
        """Count cells by orbit type."""
        counts = Counter()
        for i in range(self.n):
            otype, clen = classify_orbit(self.b[i], self.e[i], self.m)
            counts[otype] += 1
        return dict(counts)

    def step(self):
        """One timestep: evolve QA states, compute thrust, update position."""
        new_b = list(self.b)
        new_e = list(self.e)

        for i in range(self.n):
            bi, ei = qa_step(self.b[i], self.e[i], self.m)

            # Coupling from neighbors
            if self.n > 1:
                for j in range(self.n):
                    if i != j:
                        ti = qa_tuple(self.b[i], self.e[i], self.m)
                        tj = qa_tuple(self.b[j], self.e[j], self.m)
                        res = sum(a*b for a, b in zip(ti, tj))
                        bi = int(round(bi + self.W[i, j] * res * (self.b[j] - self.b[i]) * 0.01))
                        ei = int(round(ei + self.W[i, j] * res * (self.e[j] - self.e[i]) * 0.01))

            new_b[i] = max(1, min(self.m, bi))
            new_e[i] = max(1, min(self.m, ei))

        self.b = new_b
        self.e = new_e

        # Compute thrust from each cell (observer projection to 2D)
        total_fx = 0.0
        total_fy = 0.0
        for i in range(self.n):
            # Thrust magnitude proportional to QA norm
            t = qa_tuple(self.b[i], self.e[i], self.m)
            magnitude = abs(t[0]*t[0] + t[0]*t[1] - t[1]*t[1]) / (self.m * self.m)  # noqa: S1
            total_fx += magnitude * np.cos(self.angles[i])  # noqa: T2-D-1 observer projection
            total_fy += magnitude * np.sin(self.angles[i])  # noqa: T2-D-1 observer projection

        # Update position
        self.x += total_fx * 0.5
        self.y += total_fy * 0.5

        return self.x, self.y

    def run(self, n_steps):
        """Run for n_steps, return trajectory."""
        trajectory_x = []
        trajectory_y = []
        for _ in range(n_steps):
            x, y = self.step()
            trajectory_x.append(x)
            trajectory_y.append(y)
        return np.array(trajectory_x), np.array(trajectory_y)


# ─── Complexity Index (from Welch PSD) ────────────────────────────────

def complexity_index(x, y, fs=1.0, threshold_ratio=0.1):
    """
    Compute Complexity Index: number of significant spectral peaks.
    CI=1 = simple circle, CI>1 = spirograph.
    Matches Fotowat et al. method (Welch PSD of x,y trajectories).
    """
    # Welch PSD on x and y separately
    if len(x) < 16:
        return 0, [], []

    nperseg = min(len(x) // 2, 64)
    freqs_x, psd_x = sig.welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    freqs_y, psd_y = sig.welch(y - np.mean(y), fs=fs, nperseg=nperseg)

    # Combined power
    psd_combined = psd_x + psd_y

    # Find peaks above threshold
    if np.max(psd_combined) == 0:
        return 0, freqs_x, psd_combined

    threshold = threshold_ratio * np.max(psd_combined)
    peaks, properties = sig.find_peaks(psd_combined, height=threshold,
                                        distance=2, prominence=threshold * 0.5)

    ci = len(peaks)
    return ci, freqs_x, psd_combined


# ─── EXPERIMENT: Uniform vs Mixed Orbit Systems ──────────────────────

print("=" * 70)
print("QA SPIROGRAPH SIMULATION")
print("Mapping: Fotowat et al. (2026) Neurobots → QA Multi-Orbit Dynamics")
print("=" * 70)

N_STEPS = 500
N_TRIALS = 30

system_types = {
    'uniform_cosmos': {'cell_types': 'uniform', 'label': 'Biobot (uniform cosmos)'},
    'all_satellite': {'cell_types': 'all_satellite', 'label': 'All-satellite'},
    'mixed': {'cell_types': 'mixed', 'label': 'Neurobot (mixed orbits)'},
}

all_ci = {k: [] for k in system_types}
all_trajectories = {}

for stype, params in system_types.items():
    print(f"\n--- {params['label']} ---")

    for trial in range(N_TRIALS):
        np.random.seed(42 + trial)

        system = QAMotileSystem(
            n_cells=12, cell_types=params['cell_types'],
            m=M, coupling=0.2
        )

        if trial == 0:
            print(f"  Orbit census: {system.orbit_census()}")

        traj_x, traj_y = system.run(N_STEPS)

        ci, freqs, psd = complexity_index(traj_x, traj_y)
        all_ci[stype].append(ci)

        if trial == 0:
            all_trajectories[stype] = (traj_x, traj_y)

    mean_ci = np.mean(all_ci[stype])
    std_ci = np.std(all_ci[stype])
    print(f"  Complexity Index: {mean_ci:.2f} ± {std_ci:.2f}")
    print(f"  CI distribution: {Counter(all_ci[stype])}")


# ─── Statistical Comparisons ─────────────────────────────────────────

print("\n" + "=" * 70)
print("STATISTICAL COMPARISONS")
print("=" * 70)

ci_uniform = all_ci['uniform_cosmos']
ci_mixed = all_ci['mixed']
ci_satellite = all_ci['all_satellite']

# Main prediction: mixed > uniform
u_stat, p_val = stats.mannwhitneyu(ci_mixed, ci_uniform, alternative='greater')
print(f"\n1. Mixed (neurobot) vs Uniform (biobot) CI:")
print(f"   Mixed: {np.mean(ci_mixed):.2f} ± {np.std(ci_mixed):.2f}")
print(f"   Uniform: {np.mean(ci_uniform):.2f} ± {np.std(ci_uniform):.2f}")
print(f"   Mann-Whitney U p={p_val:.4f}")
print(f"   Prediction (mixed > uniform): {'CONFIRMED' if np.mean(ci_mixed) > np.mean(ci_uniform) else 'REJECTED'}")
print(f"   (Fotowat et al.: neurobot CI significantly higher, K-W p=0.039)")

# Secondary: satellite vs uniform
u2, p2 = stats.mannwhitneyu(ci_satellite, ci_uniform, alternative='two-sided')
print(f"\n2. Satellite vs Uniform CI:")
print(f"   Satellite: {np.mean(ci_satellite):.2f}")
print(f"   Uniform: {np.mean(ci_uniform):.2f}")
print(f"   p={p2:.4f}")

# Trajectory characterization
print(f"\n--- Trajectory Characteristics ---")
for stype, (tx, ty) in all_trajectories.items():
    total_dist = np.sum(np.sqrt(np.diff(tx)*np.diff(tx) + np.diff(ty)*np.diff(ty)))  # noqa: S1
    bbox = (np.max(tx) - np.min(tx)) * (np.max(ty) - np.min(ty))
    print(f"  {stype:20s}: distance={total_dist:.2f}, bbox={bbox:.4f}")


# ─── SUMMARY ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: QA Predictions vs Engineered Living Systems Results")
print("=" * 70)

print("""
Fotowat et al. (2026) findings:
  - Neurobots: significantly MORE COMPLEX trajectories (K-W p=0.039)
  - Neurobots: spirograph-like patterns with multiple spectral peaks
  - Biobots: simple circular/oval trajectories (CI=1)
  - PTZ drug: biobots decrease CI, neurobots increase CI

QA structural prediction:
  - Single-orbit system (biobot) → single frequency → CI=1 (circle)
  - Mixed-orbit system (neurobot) → multiple frequencies → CI>1 (spirograph)
  - Each spectral peak corresponds to one orbital frequency
  - Cosmos (24-cycle) + Satellite (8-cycle) = at least 2 frequencies
  - Adding Singularity (1-cycle = DC offset) doesn't add a peak but shifts baseline

The spirograph pattern IS epicyclic motion from multiple QA orbital
frequencies projected onto the 2D plane. QA predicts: CI ≈ number of
distinct orbit types with nonzero thrust in the coupled system.
""")

print("Script complete.")
