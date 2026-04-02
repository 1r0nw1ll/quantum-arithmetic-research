#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_algorithm_competency_batch4.py
====================================
Fourth batch: 20 algorithms → 71 total.
New families: number_theory, signal_processing
Fills out: sort, search, graph, optimize, learn, time_series

Usage:
  python qa_algorithm_competency_batch4.py
  python qa_algorithm_competency_batch4.py --dry-run
"""

from __future__ import annotations
import json, sys, argparse
from pathlib import Path

MODULUS  = 9
REG_PATH = Path("qa_algorithm_competency_registry.json")

QA1  = "qa-1__qa_1_all_pages__docx.md"
QA2  = "qa-2__001_qa_2_all_pages__docx.md"
QA3  = "qa_3__ocr__qa3.md"
QA4  = "qa-4__00_qa_books_3_&_4_all_pages__pdf.md"
QUAD = "quadrature__00_quadratureprint__pdf.md"
P1   = "pyth_1__ocr__pyth1.md"
P2   = "pyth_2__ocr__pyth2.md"
P3   = "pyth-3__pythagoras_vol3_enneagram__docx.md"
WB   = "qa_workbook__ocr__workbook.md"

def qa_step(b, e, m=MODULUS): return e % m, (b + e) % m
def qa_orbit_family(b, e, m=MODULUS, max_steps=500):
    seen, state = {}, (b % m, e % m)
    for t in range(max_steps):
        if state in seen:
            p = t - seen[state]
            if p == 1:   return "singularity"
            if p == 8:   return "satellite"
            if p == 24:  return "cosmos"
            return f"period_{p}"
        seen[state] = t
        state = qa_step(*state, m)
    return "unknown"

def orbit_follow_rate(b, e, m=MODULUS, steps=48):
    traj, state = [], (b % m, e % m)
    for _ in range(steps):
        traj.append(state)
        state = qa_step(*state, m)
    if len(traj) < 3:
        return 0.0
    hits = sum(
        1 for i in range(len(traj) - 2)
        if traj[i+1][0] == traj[i][1] % m
        and traj[i+1][1] == (traj[i][0] + traj[i][1]) % m
    )
    return round(hits / (len(traj) - 2), 4)

BATCH4 = [

    # ── sort (2) ──────────────────────────────────────────────────────────────

    {
        "name": "timsort", "family": "sort",
        "goal": "Hybrid merge+insertion sort exploiting natural runs; O(N log N) worst, O(N) best on nearly-sorted input",
        "orbit_seed": [3, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["adaptive_sorter", "run_detector", "merge_coordinator"],
        "differentiation_profile": {
            "dediff_conditions":   ["array fully random (no natural runs)", "pathological comparator"],
            "recommit_conditions": ["input exhibits partial order", "nearly-sorted subsequences detected"],
            "max_satellite_cycles": 3,
            "drift_threshold": 0.15,
        },
        "source_corpus_refs": [QA1, QA2, WB],
        "corpus_concepts":    ["arithmetic", "integer", "proportion", "series", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Timsort's run-detection is analogous to orbit recognition in QA: "
            "it identifies already-ordered subsequences (cosmos-like segments) and "
            "merges them efficiently. Mixed orbit because behaviour adapts to input structure."
        ),
    },

    {
        "name": "shell_sort", "family": "sort",
        "goal": "Sort by comparing elements at decreasing gap intervals; bridges insertion sort and merge sort",
        "orbit_seed": [2, 7],
        "levin_cell_type": "differentiated",
        "organ_roles": ["gap_scheduler", "insertion_sorter", "coarse_to_fine_organizer"],
        "differentiation_profile": {
            "dediff_conditions":   ["gap sequence degenerates to 1 (= insertion sort)", "adversarial input for chosen gaps"],
            "recommit_conditions": ["structured partially-sorted input", "gap > 1 phase active"],
            "max_satellite_cycles": 4,
            "drift_threshold": 0.20,
        },
        "source_corpus_refs": [QA1, QA2, P2],
        "corpus_concepts":    ["arithmetic", "integer", "proportion", "interval", "period"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Shell sort's diminishing gap schedule mirrors orbit contraction in QA: "
            "large gaps = coarse satellite-like sweeps; gap=1 = final cosmos convergence pass."
        ),
    },

    # ── search (3) ────────────────────────────────────────────────────────────

    {
        "name": "fibonacci_search", "family": "search",
        "goal": "Search sorted array using Fibonacci number intervals; avoids division, cache-friendly",
        "orbit_seed": [1, 1],   # Fibonacci seed — also QA canonical start
        "levin_cell_type": "differentiated",
        "organ_roles": ["fibonacci_partitioner", "sorted_array_searcher", "division_free_reducer"],
        "differentiation_profile": {
            "dediff_conditions":   ["unsorted input", "Fibonacci table overflow for N"],
            "recommit_conditions": ["sorted array", "division-cost dominates (embedded)"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.10,
        },
        "source_corpus_refs": [QA1, QA2, P1, P2, WB],
        "corpus_concepts":    ["fibonacci", "phi", "golden ratio", "proportion", "interval", "harmonic"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "DIRECT QA ALIGNMENT: orbit_seed (1,1) is both the Fibonacci seed and the "
            "canonical QA starting pair. Fibonacci numbers are the denominators of the "
            "convergents of φ — QA's algebraic core is Q(√5) = Q(φ). "
            "This algorithm is the most φ-native search in the registry."
        ),
    },

    {
        "name": "ternary_search", "family": "search",
        "goal": "Find extremum of unimodal function by recursive trisection; O(log₃ N)",
        "orbit_seed": [3, 6],
        "levin_cell_type": "differentiated",
        "organ_roles": ["unimodal_optimizer", "trisection_reducer", "extremum_finder"],
        "differentiation_profile": {
            "dediff_conditions":   ["non-unimodal function", "discrete input with ties"],
            "recommit_conditions": ["continuous unimodal function", "golden-section context"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.12,
        },
        "source_corpus_refs": [QA1, QA3, P2],
        "corpus_concepts":    ["proportion", "interval", "convergence", "rational"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "exponential_search", "family": "search",
        "goal": "Find range via doubling then binary search; O(log i) where i = target index; ideal for unbounded arrays",
        "orbit_seed": [1, 2],
        "levin_cell_type": "progenitor",
        "organ_roles": ["range_expander", "unbounded_searcher", "doubling_probe"],
        "differentiation_profile": {
            "dediff_conditions":   ["bounded array known", "target near beginning"],
            "recommit_conditions": ["unbounded or unknown-size stream", "target index >> 1"],
            "max_satellite_cycles": 5,
            "drift_threshold": 0.25,
        },
        "source_corpus_refs": [QA1, QA2],
        "corpus_concepts":    ["proportion", "interval", "series", "period"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
        "qa_research_note": (
            "Doubling phase maps to QA orbit expansion before period lock; "
            "binary search refinement = cosmos convergence once range is bounded."
        ),
    },

    # ── graph (2) ─────────────────────────────────────────────────────────────

    {
        "name": "floyd_warshall", "family": "graph",
        "goal": "All-pairs shortest paths via dynamic programming; O(V³); handles negative weights",
        "orbit_seed": [4, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["all_pairs_pathfinder", "distance_matrix_builder", "negative_cycle_detector"],
        "differentiation_profile": {
            "dediff_conditions":   ["negative cycle present", "sparse graph (use Johnson's)"],
            "recommit_conditions": ["dense graph", "all-pairs query pattern"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.08,
        },
        "source_corpus_refs": [QA4, QA1, WB],
        "corpus_concepts":    ["invariant", "transformation", "measure", "dimension", "period"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "The DP relaxation matrix is an orbit-distance table: "
            "d[i][j] after k iterations = shortest path using only nodes 0..k as intermediates. "
            "Maps to QA reachability under k orbit steps."
        ),
    },

    {
        "name": "prim_mst", "family": "graph",
        "goal": "Grow minimum spanning tree greedily from a root by always adding cheapest outgoing edge",
        "orbit_seed": [2, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["mst_grower", "priority_queue_consumer", "connectivity_expander"],
        "differentiation_profile": {
            "dediff_conditions":   ["disconnected graph", "directed graph without MST semantics"],
            "recommit_conditions": ["dense graph (O(V²) faster than Kruskal)", "incremental edge additions"],
            "max_satellite_cycles": 3,
            "drift_threshold": 0.12,
        },
        "source_corpus_refs": [QA4, P1, QA2],
        "corpus_concepts":    ["orbit", "cycle", "proportion", "measure", "invariant"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    # ── optimize (3) ──────────────────────────────────────────────────────────

    {
        "name": "conjugate_gradient", "family": "optimize",
        "goal": "Solve Ax=b for symmetric positive-definite A; converges in at most N steps; memory-efficient",
        "orbit_seed": [1, 4],
        "levin_cell_type": "differentiated",
        "organ_roles": ["linear_solver", "krylov_explorer", "residual_minimizer"],
        "differentiation_profile": {
            "dediff_conditions":   ["A not symmetric positive definite", "ill-conditioned system"],
            "recommit_conditions": ["sparse SPD system", "matrix-vector product cheaper than factorization"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.07,
        },
        "source_corpus_refs": [QA1, QA3, P2, WB],
        "corpus_concepts":    ["convergence", "harmonic", "linear", "measure", "proportion", "orthogonal"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "CG search directions are A-conjugate (orthogonal under A-inner-product). "
            "QA analog: successive orbit directions are spread-orthogonal in Z[φ]/mZ[φ]. "
            "Cosmos orbit: convergence in finite steps guaranteed for SPD case."
        ),
    },

    {
        "name": "differential_evolution", "family": "optimize",
        "goal": "Population-based global optimizer using mutation and crossover on real-valued vectors; derivative-free",
        "orbit_seed": [5, 7],
        "levin_cell_type": "progenitor",
        "organ_roles": ["global_searcher", "population_mutator", "crossover_selector"],
        "differentiation_profile": {
            "dediff_conditions":   ["population collapses (diversity lost)", "premature convergence"],
            "recommit_conditions": ["multimodal landscape", "black-box objective", "population diverse"],
            "max_satellite_cycles": 10,
            "drift_threshold": 0.30,
        },
        "source_corpus_refs": [QA3, QA4, P1],
        "corpus_concepts":    ["orbit", "cycle", "resonance", "harmonic", "period", "convergence"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "DE mutation (v = a + F*(b-c)) mirrors QA perturbation: "
            "difference vector b-c probes the orbit neighborhood. "
            "Satellite orbit: population cycles; cosmos exit = convergence to global optimum."
        ),
    },

    {
        "name": "nesterov_momentum", "family": "optimize",
        "goal": "Accelerated gradient descent with look-ahead momentum; O(1/t²) convergence vs O(1/t) for GD",
        "orbit_seed": [3, 4],
        "levin_cell_type": "differentiated",
        "organ_roles": ["accelerated_descender", "momentum_accumulator", "look_ahead_corrector"],
        "differentiation_profile": {
            "dediff_conditions":   ["non-convex landscape with many saddles", "momentum overshoots minimum"],
            "recommit_conditions": ["strongly convex objective", "large-scale smooth optimization"],
            "max_satellite_cycles": 3,
            "drift_threshold": 0.12,
        },
        "source_corpus_refs": [QA1, QA2, QA3, WB],
        "corpus_concepts":    ["convergence", "harmonic", "gain", "phase", "series", "proportion"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Nesterov's look-ahead step (evaluate gradient at x + momentum) "
            "maps to QA orbit anticipation: the next state is predicted one step ahead "
            "before computing the coupling update. Mixed orbit: acceleration is convex-specific."
        ),
    },

    # ── learn (3) ─────────────────────────────────────────────────────────────

    {
        "name": "decision_tree", "family": "learn",
        "goal": "Recursively partition feature space by information gain / Gini impurity; interpretable classifier/regressor",
        "orbit_seed": [2, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["feature_partitioner", "entropy_minimizer", "rule_extractor"],
        "differentiation_profile": {
            "dediff_conditions":   ["overfitting (tree too deep)", "continuous features without binning"],
            "recommit_conditions": ["interpretability required", "categorical features dominant"],
            "max_satellite_cycles": 3,
            "drift_threshold": 0.15,
        },
        "source_corpus_refs": [QA1, QA2, WB],
        "corpus_concepts":    ["measure", "dimension", "proportion", "harmonic", "threshold"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Each split = orbit boundary: left child and right child inhabit different "
            "QA reachability regions. Leaf nodes = differentiated cells (committed to a class). "
            "Pruning = dedifferentiation recovery."
        ),
    },

    {
        "name": "q_learning", "family": "learn",
        "goal": "Model-free RL via Bellman backup on Q-table; converges to optimal policy for finite MDPs",
        "orbit_seed": [6, 3],
        "levin_cell_type": "progenitor",
        "organ_roles": ["value_estimator", "policy_improver", "exploration_agent"],
        "differentiation_profile": {
            "dediff_conditions":   ["Q-table converged, ε→0", "deterministic greedy policy locked in"],
            "recommit_conditions": ["non-stationary reward", "ε > threshold (exploration active)"],
            "max_satellite_cycles": 8,
            "drift_threshold": 0.25,
        },
        "source_corpus_refs": [QA3, QA4, P1],
        "corpus_concepts":    ["orbit", "cycle", "convergence", "resonance", "period", "fixed point"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Q-learning orbit: satellite during exploration (value estimates cycling), "
            "cosmos convergence when Bellman residuals shrink below threshold. "
            "The Q-table = orbit reachability map: Q(s,a) ≈ expected orbit descent from state s."
        ),
    },

    {
        "name": "variational_autoencoder", "family": "learn",
        "goal": "Learn continuous latent representation by maximising ELBO; enables generative sampling via reparameterisation",
        "orbit_seed": [5, 2],
        "levin_cell_type": "progenitor",
        "organ_roles": ["latent_encoder", "generative_decoder", "kl_regularizer", "manifold_learner"],
        "differentiation_profile": {
            "dediff_conditions":   ["posterior collapse (decoder ignores latent)", "KL vanishing"],
            "recommit_conditions": ["reconstruction loss stabilised", "latent space meaningful (disentangled)"],
            "max_satellite_cycles": 6,
            "drift_threshold": 0.28,
        },
        "source_corpus_refs": [QA3, QA4, P3, WB],
        "corpus_concepts":    ["orbit", "dimension", "measure", "distribution", "harmonic", "spiral"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "VAE latent space = QA orbit manifold projection: the reparameterisation trick "
            "(z = μ + σ·ε) is a stochastic orbit perturbation. KL term = orbit spread penalty. "
            "Posterior collapse maps to singularity: all inputs map to same orbit point."
        ),
    },

    # ── time_series (2) ───────────────────────────────────────────────────────

    {
        "name": "wavelet_transform", "family": "time_series",
        "goal": "Multi-resolution signal decomposition via scaled/translated wavelets; simultaneous time-frequency localisation",
        "orbit_seed": [1, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["multi_scale_decomposer", "time_frequency_localiser", "singularity_detector", "denoiser"],
        "differentiation_profile": {
            "dediff_conditions":   ["signal non-stationary at all scales", "boundary effects dominate"],
            "recommit_conditions": ["transient detection required", "multi-scale structure present"],
            "max_satellite_cycles": 3,
            "drift_threshold": 0.10,
        },
        "source_corpus_refs": [QA3, QA4, P2, WB],
        "corpus_concepts":    ["wave", "harmonic", "series", "period", "frequency", "oscillation", "measure", "phase"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Wavelet decomposition mirrors QA orbit decomposition: coarse scales = "
            "long-period cosmos orbit; fine scales = satellite oscillations. "
            "Wavelet singularity detection (Lipschitz exponent) = QA orbit collapse detector."
        ),
    },

    {
        "name": "spectral_analysis", "family": "time_series",
        "goal": "Estimate power spectral density of a signal; identify dominant frequencies and periodicities",
        "orbit_seed": [2, 4],
        "levin_cell_type": "differentiated",
        "organ_roles": ["frequency_decomposer", "periodicity_detector", "power_estimator"],
        "differentiation_profile": {
            "dediff_conditions":   ["signal non-stationary", "spectral leakage from non-integer periods"],
            "recommit_conditions": ["stationary periodic signal", "frequency identification task"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.08,
        },
        "source_corpus_refs": [QA1, QA3, QA4, P2, WB],
        "corpus_concepts":    ["wave", "harmonic", "series", "period", "frequency", "oscillation", "resonance", "phase"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "QA orbits have exact periods (1, 8, 24). Spectral analysis applied to "
            "QA orbit sequences recovers these as sharp peaks. The Harmonic Index (HI) "
            "is a spectral coherence measure in disguise — peak sharpness = orbit purity."
        ),
    },

    # ── signal_processing (NEW family, 3) ─────────────────────────────────────

    {
        "name": "fft_cooley_tukey", "family": "signal_processing",
        "goal": "Compute Discrete Fourier Transform in O(N log N) via divide-and-conquer butterfly operations",
        "orbit_seed": [1, 8],
        "levin_cell_type": "differentiated",
        "organ_roles": ["frequency_transformer", "butterfly_executor", "spectrum_builder"],
        "differentiation_profile": {
            "dediff_conditions":   ["N not a power of 2 (use Bluestein)", "very small N (direct DFT faster)"],
            "recommit_conditions": ["N = power of 2", "repeated frequency-domain operations"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.05,
        },
        "source_corpus_refs": [QA1, QA3, QA4, P2, WB],
        "corpus_concepts":    ["wave", "harmonic", "frequency", "period", "oscillation", "resonance", "phase", "series"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "FFT butterfly stages = QA orbit recursion levels. The twiddle factors "
            "e^{-2πi k/N} are roots of unity — QA works in Z[φ]/mZ[φ] (another cyclotomic extension). "
            "orbit_seed (1,8): 8 = satellite period; N=8 FFT is the minimal nontrivial butterfly."
        ),
    },

    {
        "name": "viterbi", "family": "signal_processing",
        "goal": "Find most probable hidden state sequence in HMM via dynamic programming; O(T·S²)",
        "orbit_seed": [3, 7],
        "levin_cell_type": "differentiated",
        "organ_roles": ["sequence_decoder", "path_tracker", "hmm_inference_engine"],
        "differentiation_profile": {
            "dediff_conditions":   ["model mismatch (wrong HMM)", "emission probs near-uniform"],
            "recommit_conditions": ["HMM well-fitted", "sequence decoding with known model"],
            "max_satellite_cycles": 3,
            "drift_threshold": 0.12,
        },
        "source_corpus_refs": [QA2, QA4, P1],
        "corpus_concepts":    ["orbit", "period", "transition", "convergence", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Viterbi trellis = QA orbit state diagram: each column is a time slice of "
            "possible orbit states; backpointers = orbit trajectory reconstruction. "
            "Most probable path = highest-resonance orbit sequence."
        ),
    },

    {
        "name": "phase_locked_loop", "family": "signal_processing",
        "goal": "Track and synchronise to phase of input signal via feedback; used in clocks, demodulation, motor control",
        "orbit_seed": [1, 1],
        "levin_cell_type": "progenitor",
        "organ_roles": ["phase_tracker", "frequency_synchroniser", "lock_detector", "demodulator"],
        "differentiation_profile": {
            "dediff_conditions":   ["input frequency outside lock range", "phase error diverging"],
            "recommit_conditions": ["locked: phase error < threshold", "frequency stable"],
            "max_satellite_cycles": 7,
            "drift_threshold": 0.20,
        },
        "source_corpus_refs": [QA1, QA3, QA4, WB],
        "corpus_concepts":    ["phase", "frequency", "oscillation", "resonance", "gain", "harmonic", "feedback", "cycle"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "PLL lifecycle = QA orbit Levin lifecycle: acquisition = satellite (cycling, seeking lock); "
            "locked = cosmos (stable, differentiated); loss-of-lock = dedifferentiation back to satellite. "
            "The phase detector is a QA spread comparator between reference and VCO orbits."
        ),
    },

    # ── number_theory (NEW family, 3) ─────────────────────────────────────────

    {
        "name": "euclidean_gcd", "family": "number_theory",
        "goal": "Compute GCD via repeated remainder; foundational to all modular arithmetic including QA",
        "orbit_seed": [1, 0],
        "levin_cell_type": "differentiated",
        "organ_roles": ["gcd_computer", "modular_reducer", "coprimality_tester", "bezout_solver"],
        "differentiation_profile": {
            "dediff_conditions":   ["inputs not integers", "one input is zero (trivial)"],
            "recommit_conditions": ["always converges for positive integers", "modular inverse needed"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.03,
        },
        "source_corpus_refs": [QA1, QA2, QA3, P1, P2, QUAD, WB],
        "corpus_concepts":    ["arithmetic", "modular", "integer", "congruence", "proportion", "rational", "measure", "period"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "THE foundational algorithm for QA: gcd(b, e) determines the period of the "
            "QA orbit starting at (b, e). gcd=1 → long cosmos orbits; gcd=m → singularity. "
            "Every QA orbit classification reduces to a GCD computation. "
            "Euclidean algorithm convergence rate ~ φ^{-steps} (Fibonacci bound — direct φ connection)."
        ),
    },

    {
        "name": "sieve_eratosthenes", "family": "number_theory",
        "goal": "Find all primes up to N in O(N log log N) via iterative composite elimination",
        "orbit_seed": [2, 1],
        "levin_cell_type": "differentiated",
        "organ_roles": ["prime_generator", "composite_eliminator", "number_theoretic_filter"],
        "differentiation_profile": {
            "dediff_conditions":   ["N too large for memory (use segmented sieve)", "primality of single number needed"],
            "recommit_conditions": ["all primes below N needed", "N fits in memory"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.04,
        },
        "source_corpus_refs": [QA1, QA2, P1, P2, QUAD],
        "corpus_concepts":    ["arithmetic", "integer", "modular", "period", "congruence", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Primes are the inert elements of QA orbit classification: "
            "p inert in Z[φ] ↔ p ≡ ±2 (mod 5) ↔ full-period orbit (cosmos). "
            "The sieve can be repurposed as a QA orbit period classifier: "
            "mark orbit_period[n] for each n in the modular range."
        ),
    },

    {
        "name": "miller_rabin", "family": "number_theory",
        "goal": "Probabilistic primality test; O(k log²N log log N); used in cryptography for large prime generation",
        "orbit_seed": [2, 5],
        "levin_cell_type": "progenitor",
        "organ_roles": ["primality_tester", "witness_checker", "cryptographic_prime_finder"],
        "differentiation_profile": {
            "dediff_conditions":   ["N is composite (test returns false)", "deterministic proof needed (use BPSW)"],
            "recommit_conditions": ["N passes all k witnesses (probable prime)", "cryptographic key generation"],
            "max_satellite_cycles": 6,
            "drift_threshold": 0.20,
        },
        "source_corpus_refs": [QA2, QA3, P1, P2, QUAD],
        "corpus_concepts":    ["arithmetic", "modular", "integer", "congruence", "period", "orbit", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Miller-Rabin witnesses probe modular orbits: a^{(N-1)/2^s} mod N must land on "
            "±1 or cycle through quadratic residues. This is a QA orbit membership test: "
            "does a follow the expected cosmos trajectory for prime moduli? "
            "Composite N = QA orbit anomaly (wrong period detected)."
        ),
    },

]


def make_entry(d: dict, modulus: int = MODULUS) -> dict:
    b, e = d["orbit_seed"]
    family = qa_orbit_family(b, e, modulus)
    ofr    = orbit_follow_rate(b, e, modulus)
    return {
        "name":                   d["name"],
        "family":                 d["family"],
        "goal":                   d["goal"],
        "orbit_seed":             [b, e],
        "orbit_signature":        d.get("orbit_signature", family),
        "orbit_follow_rate":      ofr,
        "levin_cell_type":        d["levin_cell_type"],
        "organ_roles":            d["organ_roles"],
        "differentiation_profile": d["differentiation_profile"],
        "source_corpus_refs":     d.get("source_corpus_refs", []),
        "corpus_concepts":        d.get("corpus_concepts", []),
        "needs_ocr_backfill":     d.get("needs_ocr_backfill", False),
        "confidence":             d.get("confidence", "medium"),
        "qa_research_note":       d.get("qa_research_note", ""),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    reg = json.loads(REG_PATH.read_text()) if REG_PATH.exists() else {"algorithms": [], "corpus_status": {}}
    existing = {a["name"] for a in reg["algorithms"]}

    new_entries = []
    for d in BATCH4:
        if d["name"] in existing:
            print(f"  ! skip duplicate: {d['name']}")
            continue
        entry = make_entry(d)
        sig   = entry["orbit_signature"]
        fam   = entry["family"]
        ofr   = entry["orbit_follow_rate"]
        print(f"  + {entry['name']:35s} {sig:12s} {fam}  ofr={ofr}")
        new_entries.append(entry)

    if not args.dry_run:
        reg["algorithms"].extend(new_entries)
        REG_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
        total = len(reg["algorithms"])
        from collections import Counter
        fams   = Counter(a["family"]           for a in reg["algorithms"])
        orbits = Counter(a["orbit_signature"]  for a in reg["algorithms"])
        cells  = Counter(a["levin_cell_type"]  for a in reg["algorithms"])
        print(f"\nMerged {len(new_entries)} → {REG_PATH.name}  (total: {total})")
        print(f"Families: {dict(fams)}")
        print(f"Orbits:   {dict(orbits)}")
        print(f"Cells:    {dict(cells)}")
    else:
        print(f"\n[dry-run] would add {len(new_entries)} algorithms")


if __name__ == "__main__":
    main()
