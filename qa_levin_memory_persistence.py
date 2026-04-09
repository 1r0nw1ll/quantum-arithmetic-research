"""
QA Memory Persistence Simulation — Levin Xenobot Prediction Generator
=====================================================================

Maps to: Pai, Traer, Sperry, Zeng, Levin (2026) bioRxiv
"Behavioral, Physiological, and Transcriptional Mechanisms of Memory
 in a Synthetic Living Construct"

QA Prediction: Memory persistence scales with orbit cycle length.
  - Cosmos-bound memory (extract-like): persists ~24 steps (24-cycle)
  - Satellite-bound memory (intermediate): persists ~8 steps (8-cycle)
  - Singularity-bound memory (ATP-like): collapses to fixed point

Tests:
  1. Signal injection shifts orbit membership
  2. Persistence after signal removal depends on target orbit
  3. Only 3 qualitatively distinct memory types exist (mod-9)
  4. Coupling strength modulates memory robustness

Will Dale, 2026-04-06
"""

QA_COMPLIANCE = {
    'observer': 'calcium_variance_crosscorr -> float (observer projection of discrete orbit state)',
    'state_alphabet': '{1,...,9} (mod-9 integer QA states)',
    'discrete_layer': '(b,e) pairs, int, {1,...,9}; orbit classification by v3(f)',
    'observer_layer': 'variance, cross_corr -> float (measurement only, never fed back to QA)',
    'signal_injection': 'external stimulus enters via b shift (standard QA protocol)',
    'coupling': 'resonance matrix from tuple inner products (Markovian, self-organizing)',
}

import numpy as np
from fractions import Fraction
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── QA Core (A1-compliant) ───────────────────────────────────────────

M = 9  # theoretical modulus

def qa_step(b, e, m=M):
    """A1-compliant QA step: states in {1,...,m}, never {0,...,m-1}."""
    b_new = ((b + e - 1) % m) + 1
    e_new = ((e + b_new - 1) % m) + 1
    return b_new, e_new

def qa_tuple(b, e, m=M):
    """Compute full tuple (b, e, d, a) — A2: d=b+e, a=b+2e derived."""
    d = ((b + e - 1) % m) + 1
    a = ((b + 2*e - 1) % m) + 1
    return (b, e, d, a)

def classify_orbit(b, e, m=M):
    """Classify (b,e) into orbit type by cycle length."""
    seen = []
    state = (b, e)
    for step in range(m * m + 1):
        if state in seen:
            cycle_len = len(seen) - seen.index(state)
            if cycle_len == 1:
                return 'singularity', cycle_len
            elif cycle_len <= 8:
                return 'satellite', cycle_len
            else:
                return 'cosmos', cycle_len
        seen.append(state)
        state = qa_step(state[0], state[1], m)
    return 'unknown', 0

def f_norm(b, e):
    """QA norm: f(b,e) = b*b + b*e - e*e (S1: no **2)."""
    return b*b + b*e - e*e


# ─── Orbit Census ─────────────────────────────────────────────────────

print("=" * 70)
print("QA MEMORY PERSISTENCE SIMULATION")
print("Mapping: Pai et al. (2026) Xenobot Memory → QA Orbit Dynamics")
print("=" * 70)

orbit_census = Counter()
orbit_examples = {'cosmos': [], 'satellite': [], 'singularity': []}

for b in range(1, M + 1):
    for e in range(1, M + 1):
        otype, clen = classify_orbit(b, e, M)
        orbit_census[otype] += 1
        if len(orbit_examples[otype]) < 3:
            orbit_examples[otype].append((b, e, clen))

print(f"\n--- Orbit Census (mod {M}) ---")
for otype in ['cosmos', 'satellite', 'singularity']:
    count = orbit_census[otype]
    examples = orbit_examples[otype]
    print(f"  {otype:12s}: {count:3d} pairs  (examples: {examples})")
print(f"  Total: {sum(orbit_census.values())} pairs")


# ─── Multi-Agent Coupled QA System ────────────────────────────────────

class QACellCollective:
    """
    A collective of N coupled (b,e) pairs — models a Xenobot.

    Each cell has state (b, e).
    Coupling matrix W determines inter-cell influence.
    Signal injection perturbs b values.
    """

    def __init__(self, n_cells=20, m=M, coupling_strength=0.3):
        self.n = n_cells
        self.m = m
        self.coupling = coupling_strength

        # Initialize cells near singularity (undifferentiated, like Xenobot start)
        # Small random perturbation from (m//2+1, m//2+1)
        center = m // 2 + 1  # = 5 for mod 9
        # S2-compliant: b, e are int arrays, never float
        self.b = np.array([max(1, min(m, center + np.random.randint(-1, 2)))
                           for _ in range(n_cells)], dtype=int)
        self.e = np.array([max(1, min(m, center + np.random.randint(-1, 2)))
                           for _ in range(n_cells)], dtype=int)

        # Coupling matrix (resonance-based, will self-organize)
        self.W = np.ones((n_cells, n_cells)) * coupling_strength / n_cells
        np.fill_diagonal(self.W, 0)

    def get_state(self):
        """Return current collective state — observer projection (float, read-only)."""
        # Observer projection: int QA states → float measurements (Theorem NT)
        b_arr = np.array(self.b, dtype=float)
        e_arr = np.array(self.e, dtype=float)
        var_b = np.var(b_arr)
        var_e = np.var(e_arr)
        variance = var_b + var_e

        if np.std(b_arr) > 0 and np.std(e_arr) > 0:
            cross_corr = np.corrcoef(b_arr, e_arr)[0, 1]
        else:
            cross_corr = 0.0

        return {
            'mean_b': np.mean(b_arr),
            'mean_e': np.mean(e_arr),
            'variance': variance,
            'cross_corr': cross_corr if not np.isnan(cross_corr) else 0.0,
            'orbit_counts': self._orbit_census()
        }

    def _orbit_census(self):
        """Count cells in each orbit type."""
        counts = Counter()
        for i in range(self.n):
            otype, _ = classify_orbit(int(self.b[i]), int(self.e[i]), self.m)
            counts[otype] += 1
        return dict(counts)

    def step(self):
        """One QA timestep with coupling."""
        new_b = [0] * self.n  # S2: plain int list for QA state
        new_e = [0] * self.n  # S2: plain int list for QA state

        for i in range(self.n):
            # Local QA step (A1-compliant)
            bi, ei = qa_step(int(self.b[i]), int(self.e[i]), self.m)

            # Coupling influence from neighbors
            coupling_b = 0
            coupling_e = 0
            for j in range(self.n):
                if i != j:
                    # Resonance coupling: influence proportional to tuple similarity
                    ti = qa_tuple(int(self.b[i]), int(self.e[i]), self.m)
                    tj = qa_tuple(int(self.b[j]), int(self.e[j]), self.m)
                    resonance = sum(a*b for a, b in zip(ti, tj))
                    coupling_b += self.W[i, j] * resonance * (self.b[j] - self.b[i])
                    coupling_e += self.W[i, j] * resonance * (self.e[j] - self.e[i])

            # Apply coupling (discretized — S2 compliant)
            bi = int(round(bi + self.coupling * coupling_b))
            ei = int(round(ei + self.coupling * coupling_e))

            # Clamp to {1,...,m} (A1)
            new_b[i] = max(1, min(self.m, bi))
            new_e[i] = max(1, min(self.m, ei))

        self.b = new_b
        self.e = new_e

        # Update coupling matrix from resonance (Markovian)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                ti = qa_tuple(int(self.b[i]), int(self.e[i]), self.m)
                tj = qa_tuple(int(self.b[j]), int(self.e[j]), self.m)
                res = sum(a*b for a, b in zip(ti, tj)) / (self.m * self.m * 4)
                self.W[i, j] = 0.9 * self.W[i, j] + 0.1 * res * self.coupling
                self.W[j, i] = self.W[i, j]

    def inject_signal(self, signal_type='cosmos', strength=3):
        """
        Inject signal into the collective (models chemical stimulus).

        signal_type:
          'cosmos'     — shift toward Cosmos orbit (like embryo extract → increased cohesion)
          'singularity' — shift toward Singularity (like ATP → decreased cohesion)
          'satellite'   — shift toward Satellite (PREDICTED 3rd memory type)
        """
        if signal_type == 'cosmos':
            # Push b values apart (increase variance), strengthen coupling
            for i in range(self.n):
                shift = strength * (1 if i % 2 == 0 else -1)
                self.b[i] = max(1, min(self.m, int(self.b[i]) + shift))
            self.coupling *= 1.5  # increase coupling (extract → more cohesion)

        elif signal_type == 'singularity':
            # Push everything toward (m//2+1, m//2+1) = (5,5) ≈ singularity
            center = self.m // 2 + 1
            for i in range(self.n):
                self.b[i] = max(1, min(self.m, int(round(
                    self.b[i] + strength * (center - self.b[i]) / max(1, abs(center - self.b[i]))
                ))))
                self.e[i] = max(1, min(self.m, int(round(
                    self.e[i] + strength * (center - self.e[i]) / max(1, abs(center - self.e[i]))
                ))))
            self.coupling *= 0.5  # decrease coupling (ATP → less cohesion)

        elif signal_type == 'satellite':
            # Push into intermediate cycling regime
            # Satellite pairs have specific structure — shift toward known satellite states
            satellite_targets = [(3, 6), (6, 3), (3, 3), (6, 6)]
            for i in range(self.n):
                target = satellite_targets[i % len(satellite_targets)]
                self.b[i] = max(1, min(self.m, int(round(
                    self.b[i] + strength * np.sign(target[0] - self.b[i])
                ))))
                self.e[i] = max(1, min(self.m, int(round(
                    self.e[i] + strength * np.sign(target[1] - self.e[i])
                ))))
            # Coupling stays moderate (satellite = intermediate)


# ─── EXPERIMENT 1: Three Memory Types ─────────────────────────────────

print("\n" + "=" * 70)
print("EXPERIMENT 1: Three Qualitatively Distinct Memory Types")
print("Prediction: mod-9 QA has exactly 3 orbit types → 3 memory types")
print("=" * 70)

N_STEPS_PRE = 15     # baseline period
N_STEPS_STIM = 5     # stimulus duration (models 15-min exposure)
N_STEPS_POST = 50    # post-stimulus monitoring

signal_types = ['cosmos', 'singularity', 'satellite']
signal_labels = {
    'cosmos': 'Embryo Extract (→ Cosmos)',
    'singularity': 'ATP (→ Singularity)',
    'satellite': 'PREDICTED 3rd Type (→ Satellite)'
}

results = {}

for stype in signal_types:
    print(f"\n--- {signal_labels[stype]} ---")

    # Fresh collective for each trial
    collective = QACellCollective(n_cells=20, m=M, coupling_strength=0.3)

    trajectory = []

    # Phase 1: Baseline
    for t in range(N_STEPS_PRE):
        collective.step()
        state = collective.get_state()
        state['phase'] = 'baseline'
        state['t'] = t
        trajectory.append(state)

    # Phase 2: Stimulus
    collective.inject_signal(signal_type=stype, strength=3)
    for t in range(N_STEPS_STIM):
        collective.step()
        state = collective.get_state()
        state['phase'] = 'stimulus'
        state['t'] = N_STEPS_PRE + t
        trajectory.append(state)

    # Phase 3: Post-stimulus (signal removed — does memory persist?)
    # Reset coupling toward baseline (signal washout)
    collective.coupling = 0.3  # restore baseline coupling rate
    for t in range(N_STEPS_POST):
        collective.step()
        state = collective.get_state()
        state['phase'] = 'post'
        state['t'] = N_STEPS_PRE + N_STEPS_STIM + t
        trajectory.append(state)

    results[stype] = trajectory

    # Report
    baseline_var = np.mean([s['variance'] for s in trajectory if s['phase'] == 'baseline'][-5:])
    baseline_cc = np.mean([s['cross_corr'] for s in trajectory if s['phase'] == 'baseline'][-5:])

    stim_var = np.mean([s['variance'] for s in trajectory if s['phase'] == 'stimulus'])
    stim_cc = np.mean([s['cross_corr'] for s in trajectory if s['phase'] == 'stimulus'])

    post_early = [s for s in trajectory if s['phase'] == 'post'][:10]
    post_late = [s for s in trajectory if s['phase'] == 'post'][-10:]

    post_early_var = np.mean([s['variance'] for s in post_early])
    post_early_cc = np.mean([s['cross_corr'] for s in post_early])
    post_late_var = np.mean([s['variance'] for s in post_late])
    post_late_cc = np.mean([s['cross_corr'] for s in post_late])

    print(f"  Baseline:    var={baseline_var:.3f}  cross_corr={baseline_cc:.3f}")
    print(f"  Stimulus:    var={stim_var:.3f}  cross_corr={stim_cc:.3f}")
    print(f"  Post-early:  var={post_early_var:.3f}  cross_corr={post_early_cc:.3f}")
    print(f"  Post-late:   var={post_late_var:.3f}  cross_corr={post_late_cc:.3f}")

    # Orbit census at end
    final_orbits = trajectory[-1]['orbit_counts']
    print(f"  Final orbits: {final_orbits}")

    # Memory persistence: how many steps until state returns to within 20% of baseline?
    persistence = 0
    for s in [s for s in trajectory if s['phase'] == 'post']:
        if abs(s['variance'] - baseline_var) > 0.2 * max(baseline_var, 0.01):
            persistence += 1
        else:
            break
    print(f"  Memory persistence: {persistence} steps (of {N_STEPS_POST})")


# ─── EXPERIMENT 2: Memory Duration vs Orbit Cycle Length ──────────────

print("\n" + "=" * 70)
print("EXPERIMENT 2: Memory Duration Scales with Orbit Cycle Length")
print("Prediction: Cosmos (24-cycle) > Satellite (8-cycle) > Singularity (1-cycle)")
print("=" * 70)

N_TRIALS = 10
duration_by_type = {stype: [] for stype in signal_types}

for trial in range(N_TRIALS):
    np.random.seed(42 + trial)

    for stype in signal_types:
        collective = QACellCollective(n_cells=20, m=M, coupling_strength=0.3)

        # Baseline
        for t in range(15):
            collective.step()
        baseline_state = collective.get_state()
        baseline_var = baseline_state['variance']

        # Inject
        collective.inject_signal(signal_type=stype, strength=3)
        for t in range(5):
            collective.step()

        # Washout — measure persistence
        collective.coupling = 0.3
        persistence = 0
        for t in range(100):
            collective.step()
            state = collective.get_state()
            if abs(state['variance'] - baseline_var) > 0.15 * max(baseline_var, 0.01):
                persistence += 1
            else:
                break

        duration_by_type[stype].append(persistence)

print("\n--- Memory Duration (steps until return to baseline, 10 trials) ---")
for stype in signal_types:
    durations = duration_by_type[stype]
    mean_d = np.mean(durations)
    std_d = np.std(durations)
    print(f"  {signal_labels[stype]:40s}: mean={mean_d:.1f} ± {std_d:.1f}  (raw: {durations})")

# Statistical comparison
from scipy import stats

cosmos_d = duration_by_type['cosmos']
satellite_d = duration_by_type['satellite']
singularity_d = duration_by_type['singularity']

if np.std(cosmos_d) > 0 and np.std(singularity_d) > 0:
    t_cs, p_cs = stats.ttest_ind(cosmos_d, singularity_d)
    print(f"\n  Cosmos vs Singularity: t={t_cs:.3f}, p={p_cs:.4f}")

if np.std(cosmos_d) > 0 and np.std(satellite_d) > 0:
    t_csat, p_csat = stats.ttest_ind(cosmos_d, satellite_d)
    print(f"  Cosmos vs Satellite:   t={t_csat:.3f}, p={p_csat:.4f}")

if np.std(satellite_d) > 0 and np.std(singularity_d) > 0:
    t_ss, p_ss = stats.ttest_ind(satellite_d, singularity_d)
    print(f"  Satellite vs Singularity: t={t_ss:.3f}, p={p_ss:.4f}")

# Check ordering prediction
mean_c = np.mean(cosmos_d)
mean_sat = np.mean(satellite_d)
mean_s = np.mean(singularity_d)
ordering_correct = mean_c >= mean_sat >= mean_s
print(f"\n  Predicted ordering (Cosmos >= Satellite >= Singularity): {ordering_correct}")
print(f"  Means: Cosmos={mean_c:.1f}, Satellite={mean_sat:.1f}, Singularity={mean_s:.1f}")


# ─── EXPERIMENT 3: Coupling Strength Modulates Memory ─────────────────

print("\n" + "=" * 70)
print("EXPERIMENT 3: Coupling Strength Modulates Memory Robustness")
print("Prediction: Stronger coupling → more persistent Cosmos memory")
print("           (mirrors extract → increased cohesion finding)")
print("=" * 70)

coupling_strengths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
coupling_persistence = []

for cs in coupling_strengths:
    trial_durations = []
    for trial in range(10):
        np.random.seed(42 + trial)
        collective = QACellCollective(n_cells=20, m=M, coupling_strength=cs)

        for t in range(15):
            collective.step()
        baseline_var = collective.get_state()['variance']

        collective.inject_signal(signal_type='cosmos', strength=3)
        for t in range(5):
            collective.step()

        collective.coupling = cs  # restore original coupling
        persistence = 0
        for t in range(100):
            collective.step()
            state = collective.get_state()
            if abs(state['variance'] - baseline_var) > 0.15 * max(baseline_var, 0.01):
                persistence += 1
            else:
                break
        trial_durations.append(persistence)

    mean_p = np.mean(trial_durations)
    coupling_persistence.append(mean_p)
    print(f"  Coupling={cs:.2f}: mean persistence={mean_p:.1f} steps")

# Correlation
if np.std(coupling_persistence) > 0:
    r, p = stats.pearsonr(coupling_strengths, coupling_persistence)
    print(f"\n  Coupling-persistence correlation: r={r:.3f}, p={p:.4f}")
    print(f"  Prediction (positive r): {'CONFIRMED' if r > 0 else 'REJECTED'}")


# ─── SUMMARY ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: QA Predictions for Levin Lab (Xenobot Memory)")
print("=" * 70)

print("""
PREDICTION 1: Exactly 3 qualitatively distinct memory types exist (mod-9).
  - Cosmos-type (extract): increased variance + increased cohesion, PERSISTENT
  - Singularity-type (ATP): collapsed variance + decreased cohesion, TRANSIENT
  - Satellite-type (???): moderate variance + OSCILLATING cohesion, INTERMEDIATE
  → Test: find a 3rd stimulus that produces cycling cross-correlation

PREDICTION 2: Memory duration scales with orbit cycle length.
  - Cosmos memory (24-cycle) outlasts Satellite (8-cycle) outlasts Singularity (1-cycle)
  → Test: measure extract vs ATP memory at 48hr, 72hr, 96hr

PREDICTION 3: Coupling strength (cell integration) modulates memory robustness.
  - Stronger coupling → more persistent Cosmos-type memory
  - Weaker coupling → faster decay to baseline
  → Test: gap junction blockers should reduce memory persistence

PREDICTION 4: The (variance, cross-correlation) trajectory traces a QA orbit.
  - Each Xenobot's 24hr calcium trajectory should show discrete orbit structure
  - Spinner=Satellite, Rotator=Cosmos, Arcer=Singularity
  → Test: classify motion types by orbit dynamics on (var, CC) pairs
""")

print("Script complete. Results above are QA-generated predictions")
print("ready for comparison with Pai et al. (2026) experimental data.")
