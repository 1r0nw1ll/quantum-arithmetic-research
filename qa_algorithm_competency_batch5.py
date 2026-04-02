#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_algorithm_competency_batch5.py
====================================
Fifth and final batch: 28 algorithms → 100 total.

New families:  planning, geometry
Expanded:      signal_processing (3→8), number_theory (3→10), learn (13→18)

New schema fields added to all entries here, and backfilled onto all 72
existing entries at the end:
  cognitive_horizon  : "local" | "regional" | "global"
  convergence        : "guaranteed" | "asymptotic" | "probabilistic" | "heuristic"
  failure_modes      : list[str]
  composition_rules  : list[str]

Usage:
  python qa_algorithm_competency_batch5.py
  python qa_algorithm_competency_batch5.py --dry-run
"""

from __future__ import annotations
import json, sys, argparse
from pathlib import Path
from collections import Counter

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
            if p == 1:  return "singularity"
            if p == 8:  return "satellite"
            if p == 24: return "cosmos"
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

# ── Default field values used when backfilling existing entries ───────────────

FAMILY_DEFAULTS = {
    "sort": {
        "cognitive_horizon": "local",
        "convergence":       "guaranteed",
        "failure_modes":     ["pathological comparator", "stability violation (unstable sorts)"],
        "composition_rules": ["pipeline after data normalisation", "stable sort needed before secondary key"],
    },
    "search": {
        "cognitive_horizon": "regional",
        "convergence":       "guaranteed",
        "failure_modes":     ["unsorted input (for binary variants)", "target absent → exhaustive scan"],
        "composition_rules": ["precondition: sorted / indexed structure", "compose with sort as preprocessing"],
    },
    "graph": {
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["negative cycles (shortest-path algorithms)", "disconnected graph"],
        "composition_rules": ["BFS/DFS as subroutines", "Dijkstra → Bellman-Ford fallback on negative weights"],
    },
    "optimize": {
        "cognitive_horizon": "regional",
        "convergence":       "asymptotic",
        "failure_modes":     ["non-convex landscape (local minima)", "ill-conditioned Hessian", "vanishing/exploding gradients"],
        "composition_rules": ["wrap with learning rate scheduler", "combine with momentum for acceleration"],
    },
    "learn": {
        "cognitive_horizon": "global",
        "convergence":       "probabilistic",
        "failure_modes":     ["overfitting", "distribution shift", "label noise"],
        "composition_rules": ["pipeline with feature engineering", "ensemble with orthogonal learners"],
    },
    "control": {
        "cognitive_horizon": "local",
        "convergence":       "guaranteed",
        "failure_modes":     ["plant model mismatch", "actuator saturation", "unstable open-loop plant"],
        "composition_rules": ["outer-loop / inner-loop cascade", "state estimator (Kalman) feeds controller"],
    },
    "distributed": {
        "cognitive_horizon": "global",
        "convergence":       "probabilistic",
        "failure_modes":     ["network partition", "Byzantine nodes", "message reordering"],
        "composition_rules": ["consensus layer under application layer", "gossip feeds DHT routing"],
    },
    "time_series": {
        "cognitive_horizon": "regional",
        "convergence":       "asymptotic",
        "failure_modes":     ["non-stationarity", "structural break", "seasonality misspecification"],
        "composition_rules": ["decompose trend+season before modelling", "ensemble with spectral methods"],
    },
    "signal_processing": {
        "cognitive_horizon": "regional",
        "convergence":       "guaranteed",
        "failure_modes":     ["aliasing (Nyquist violation)", "spectral leakage", "phase distortion"],
        "composition_rules": ["filter → transform → detect pipeline", "PLL locks phase before coherent detection"],
    },
    "number_theory": {
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["integer overflow for large N", "Carmichael numbers (Miller-Rabin false positive)"],
        "composition_rules": ["GCD → extended GCD → modular inverse → CRT", "prime sieve feeds factorisation"],
    },
    "planning": {
        "cognitive_horizon": "global",
        "convergence":       "heuristic",
        "failure_modes":     ["inadmissible heuristic (A*)", "exponential branching factor", "constraint unsatisfiability"],
        "composition_rules": ["heuristic feeds beam/A* search", "SAT solver as backbone for CSP"],
    },
    "geometry": {
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["degenerate inputs (collinear points, duplicates)", "floating-point precision"],
        "composition_rules": ["convex hull → Voronoi → Delaunay pipeline", "triangulation feeds mesh algorithms"],
    },
}

# ── Batch 5 entries ───────────────────────────────────────────────────────────

BATCH5 = [

    # ── signal_processing (5 more → 8 total) ─────────────────────────────────

    {
        "name": "hilbert_transform", "family": "signal_processing",
        "goal": "Produce analytic signal with instantaneous amplitude/phase/frequency; 90° phase shift across all frequencies",
        "orbit_seed": [4, 1],
        "levin_cell_type": "differentiated",
        "organ_roles": ["analytic_signal_producer", "instantaneous_phase_extractor", "envelope_detector"],
        "cognitive_horizon": "regional",
        "convergence":       "guaranteed",
        "failure_modes":     ["edge effects (finite-length signal)", "non-causal in real-time", "DC offset distortion"],
        "composition_rules": ["apply after bandpass filter", "pair with Hilbert→instantaneous freq for demodulation"],
        "differentiation_profile": {
            "dediff_conditions":   ["broadband non-stationary signal", "DC component present"],
            "recommit_conditions": ["narrowband analytic signal needed", "AM/FM demodulation task"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.08,
        },
        "source_corpus_refs": [QA3, QA4, P2, WB],
        "corpus_concepts":    ["phase", "frequency", "wave", "harmonic", "oscillation", "resonance"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "The Hilbert transform extracts the quadrature component — the 90°-shifted "
            "partner of a real signal. In QA orbit terms, if (b,e) is the in-phase state, "
            "the quadrature is (e, b+e): the next orbit step. Instantaneous phase = "
            "atan2(e, b), a direct read of the QA orbit angle."
        ),
    },

    {
        "name": "matched_filter", "family": "signal_processing",
        "goal": "Maximise SNR for known-waveform detection in additive white Gaussian noise; correlates signal with template",
        "orbit_seed": [3, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["template_correlator", "snr_maximiser", "detection_oracle"],
        "cognitive_horizon": "local",
        "convergence":       "guaranteed",
        "failure_modes":     ["template mismatch (Doppler, multipath)", "non-white noise", "unknown signal shape"],
        "composition_rules": ["follows signal acquisition", "pair with Viterbi for sequence detection"],
        "differentiation_profile": {
            "dediff_conditions":   ["template unknown or changing", "non-stationary noise"],
            "recommit_conditions": ["known waveform and noise statistics", "radar/sonar/comms detection"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.05,
        },
        "source_corpus_refs": [QA3, QA4, WB],
        "corpus_concepts":    ["resonance", "harmonic", "correlation", "measure", "phase"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Matched filter output = inner product of signal with orbit template. "
            "In QA: if an input sequence follows a cosmos orbit, matched filter peaks "
            "at the orbit's period. Used as an orbit detector: correlate input against "
            "each of the three canonical orbit templates (1, 8, 24 period)."
        ),
    },

    {
        "name": "wiener_filter", "family": "signal_processing",
        "goal": "Optimal linear filter minimising MSE between estimate and desired signal; requires known power spectra",
        "orbit_seed": [2, 6],
        "levin_cell_type": "differentiated",
        "organ_roles": ["optimal_denoiser", "mse_minimiser", "spectral_estimator"],
        "cognitive_horizon": "regional",
        "convergence":       "guaranteed",
        "failure_modes":     ["non-stationary signal (use adaptive Wiener)", "cross-spectrum estimation error"],
        "composition_rules": ["pair with spectral_analysis to estimate PSD", "degrade to matched_filter when noise is white"],
        "differentiation_profile": {
            "dediff_conditions":   ["spectra unknown or non-stationary", "computational cost of matrix inversion"],
            "recommit_conditions": ["stationary signal with known/estimated spectra", "deconvolution task"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.10,
        },
        "source_corpus_refs": [QA1, QA3, QA4, WB],
        "corpus_concepts":    ["harmonic", "resonance", "measure", "proportion", "frequency", "wave"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "adaptive_lms", "family": "signal_processing",
        "goal": "Adaptive FIR filter using Least Mean Squares weight update; tracks non-stationary signals online",
        "orbit_seed": [5, 3],
        "levin_cell_type": "progenitor",
        "organ_roles": ["online_filter_adapter", "noise_canceller", "echo_canceller", "predictor"],
        "cognitive_horizon": "local",
        "convergence":       "asymptotic",
        "failure_modes":     ["step-size too large (diverge)", "correlated input (slow convergence)", "eigenvalue spread"],
        "composition_rules": ["pair with Wiener for initialisation", "use RLS variant for faster convergence"],
        "differentiation_profile": {
            "dediff_conditions":   ["step-size μ exceeds stability bound", "input power varies widely"],
            "recommit_conditions": ["filter weights stabilised (small update norm)", "error variance converged"],
            "max_satellite_cycles": 8,
            "drift_threshold": 0.30,
        },
        "source_corpus_refs": [QA1, QA3, QA4, WB],
        "corpus_concepts":    ["harmonic", "gain", "resonance", "convergence", "feedback", "wave", "phase"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "LMS update (w ← w + 2μe·x) is a stochastic orbit step: "
            "the filter weight vector walks through coefficient space guided by error signal. "
            "Satellite orbit during adaptation (weights cycling); cosmos convergence when locked. "
            "Maps directly to QA Lab SelfImprovementAgent weight updates."
        ),
    },

    {
        "name": "autocorrelation", "family": "signal_processing",
        "goal": "Measure self-similarity of a signal at time lag τ; detects periodicity, reverb, and hidden structure",
        "orbit_seed": [1, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["periodicity_detector", "lag_structure_measurer", "orbit_follow_rate_baseline"],
        "cognitive_horizon": "regional",
        "convergence":       "guaranteed",
        "failure_modes":     ["short signal (bias in estimator)", "non-stationary signal"],
        "composition_rules": ["precedes spectral_analysis (Wiener-Khinchin)", "baseline for orbit_follow_rate significance testing"],
        "differentiation_profile": {
            "dediff_conditions":   ["signal too short for reliable estimation", "non-stationarity invalidates lag assumption"],
            "recommit_conditions": ["signal stationary and long enough", "periodicity hypothesis to test"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.05,
        },
        "source_corpus_refs": [QA1, QA3, QA4, P2, WB],
        "corpus_concepts":    ["period", "resonance", "harmonic", "wave", "cycle", "correlation", "phase"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "DIRECT QA RELEVANCE: autocorrelation at lag 1 is the null baseline for "
            "orbit_follow_rate. The open empirical question (from Next Priorities) is: "
            "'is sine_880Hz orbit_follow_rate=0.1715 > chance, or just lag-1 AC?' "
            "This algorithm IS the baseline. R(1) of a sine = cos(2πf/fs) — "
            "if orbit_follow_rate tracks R(1), QA adds nothing. If it diverges, QA is real."
        ),
    },

    # ── number_theory (7 more → 10 total) ────────────────────────────────────

    {
        "name": "extended_euclidean", "family": "number_theory",
        "goal": "Compute GCD(a,b) and Bézout coefficients x,y such that ax+by=gcd; foundation for modular inverses",
        "orbit_seed": [1, 0],
        "levin_cell_type": "differentiated",
        "organ_roles": ["bezout_solver", "modular_inverse_computer", "linear_diophantine_solver"],
        "cognitive_horizon": "local",
        "convergence":       "guaranteed",
        "failure_modes":     ["gcd ≠ 1 (inverse does not exist)", "signed integer overflow for large coefficients"],
        "composition_rules": ["GCD → extended_GCD → modular_inverse → CRT", "prerequisite for RSA key generation"],
        "differentiation_profile": {
            "dediff_conditions":   ["inputs not coprime (inverse undefined)", "modulus is composite"],
            "recommit_conditions": ["modular inverse required", "Bézout identity needed"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.03,
        },
        "source_corpus_refs": [QA1, QA2, QA3, P1, P2, QUAD, WB],
        "corpus_concepts":    ["arithmetic", "modular", "integer", "congruence", "proportion", "rational"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Extended GCD computes the inverse of b (mod m) when gcd(b,m)=1 — "
            "the key operation for QA state decoding. Every QA orbit reversal "
            "(running the map backwards) requires a modular inverse. "
            "Bézout coefficients = QA orbit backward step weights."
        ),
    },

    {
        "name": "chinese_remainder_theorem", "family": "number_theory",
        "goal": "Reconstruct integer from its residues mod pairwise coprime moduli; O(n log n) via Garner's algorithm",
        "orbit_seed": [3, 7],
        "levin_cell_type": "differentiated",
        "organ_roles": ["multi_modulus_reconstructor", "parallel_residue_combiner", "orbit_lifter"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["moduli not pairwise coprime", "integer range exceeds product of moduli"],
        "composition_rules": ["pair with Miller-Rabin to select prime moduli", "used in RSA decryption speedup"],
        "differentiation_profile": {
            "dediff_conditions":   ["moduli share common factor", "residues inconsistent"],
            "recommit_conditions": ["pairwise coprime moduli", "parallel computation across moduli"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.03,
        },
        "source_corpus_refs": [QA1, QA2, QA3, P1, P2, QUAD, WB],
        "corpus_concepts":    ["modular", "arithmetic", "congruence", "integer", "period", "orbit"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "CRT is the multi-scale QA orbit reconstruction theorem. "
            "QA operates mod m; CRT reconstructs the full integer from mod-p₁, mod-p₂, … "
            "components — the 'full orbit' from its prime-period projections. "
            "Garner's algorithm = QA orbit lift from Z/pZ to Z/p₁p₂…Z."
        ),
    },

    {
        "name": "pollard_rho", "family": "number_theory",
        "goal": "Probabilistic integer factorisation in O(N^{1/4}) expected time via cycle detection on pseudorandom sequences",
        "orbit_seed": [2, 1],
        "levin_cell_type": "progenitor",
        "organ_roles": ["factor_finder", "cycle_detector", "pseudorandom_orbit_walker"],
        "cognitive_horizon": "global",
        "convergence":       "probabilistic",
        "failure_modes":     ["N is prime (no factor found)", "cycle detected at p+q boundary", "requires restart with new seed"],
        "composition_rules": ["screen with Miller-Rabin first", "combine with trial division for small factors"],
        "differentiation_profile": {
            "dediff_conditions":   ["N prime (restart)", "cycle detected but gcd=N (bad seed)"],
            "recommit_conditions": ["non-trivial factor found (gcd > 1)", "factor confirmed prime by Miller-Rabin"],
            "max_satellite_cycles": 12,
            "drift_threshold": 0.35,
        },
        "source_corpus_refs": [QA2, QA3, P1, P2, QUAD],
        "corpus_concepts":    ["orbit", "cycle", "period", "modular", "arithmetic", "integer"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Pollard ρ IS a QA orbit algorithm: the pseudorandom walk x_{n+1} = x_n² + c (mod N) "
            "is a nonlinear orbit; Floyd's cycle detection finds the period. "
            "The ρ-shape (tail + cycle) = QA transient + satellite pattern. "
            "The cycle period length reveals a factor of N — direct orbit-to-factor correspondence."
        ),
    },

    {
        "name": "tonelli_shanks", "family": "number_theory",
        "goal": "Compute modular square root √a (mod p) for odd prime p; prerequisite for elliptic curve arithmetic",
        "orbit_seed": [4, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["modular_sqrt_solver", "quadratic_residue_tester", "elliptic_curve_prereq"],
        "cognitive_horizon": "local",
        "convergence":       "guaranteed",
        "failure_modes":     ["a is quadratic non-residue mod p (no solution)", "p = 2 (use direct formula)"],
        "composition_rules": ["prerequisite for point decompression in ECC", "pair with Euler criterion for QR test"],
        "differentiation_profile": {
            "dediff_conditions":   ["a is quadratic non-residue (no sqrt)", "p ≡ 3 (mod 4) (use simpler formula)"],
            "recommit_conditions": ["a is QR mod p", "ECC or number-theoretic application requiring sqrt"],
            "max_satellite_cycles": 3,
            "drift_threshold": 0.12,
        },
        "source_corpus_refs": [QA2, QA3, P2, QUAD],
        "corpus_concepts":    ["modular", "arithmetic", "quadrance", "spread", "congruence", "orbit"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Modular sqrt is the inverse of squaring in Z/pZ — the inverse of the QA "
            "quadrance function Q(b,e) = b²+be-e² (mod p). "
            "Tonelli-Shanks finds which orbit state (b,e) has a given norm value Q, "
            "enabling QA state reconstruction from a single observable (Q)."
        ),
    },

    {
        "name": "pisano_period", "family": "number_theory",
        "goal": "Compute period of Fibonacci sequence mod m (π(m)); reveals orbit structure of φ-based recursions",
        "orbit_seed": [1, 1],   # Fibonacci seed
        "levin_cell_type": "differentiated",
        "organ_roles": ["fibonacci_orbit_classifier", "period_finder", "modular_structure_analyst"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["large m requires factorisation of m first", "non-Fibonacci recursions"],
        "composition_rules": ["use prime factorisation of m → π(m) = lcm(π(pᵢ^kᵢ))", "pair with sieve for small primes"],
        "differentiation_profile": {
            "dediff_conditions":   ["m has large prime factors (slow)", "non-linear recurrence"],
            "recommit_conditions": ["m = QA modulus (9 or 24)", "Fibonacci-based cryptography"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.03,
        },
        "source_corpus_refs": [QA1, QA2, QA3, P1, P2, QUAD, WB],
        "corpus_concepts":    ["fibonacci", "phi", "golden ratio", "period", "orbit", "modular", "cycle", "arithmetic"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "CRITICAL QA CONNECTION: the Pisano period π(m) is the period of Fibonacci mod m — "
            "which is the same as the orbit period of the linear map T=(0,1;1,1) in Z/mZ. "
            "QA's T = ×φ² map has the same structure. π(9)=24, π(24)=24 — "
            "these ARE the QA cosmos orbit periods. Pisano period computation IS QA orbit classification."
        ),
    },

    {
        "name": "modular_exponentiation", "family": "number_theory",
        "goal": "Compute a^b mod m in O(log b) via repeated squaring; backbone of RSA, Diffie-Hellman, primality tests",
        "orbit_seed": [2, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["fast_power_computer", "cryptographic_primitive", "fermat_witness"],
        "cognitive_horizon": "local",
        "convergence":       "guaranteed",
        "failure_modes":     ["integer overflow without big-int library", "b < 0 (use modular inverse first)"],
        "composition_rules": ["core of Miller-Rabin witness check", "base operation for Diffie-Hellman / RSA"],
        "differentiation_profile": {
            "dediff_conditions":   ["b = 0 (trivial)", "m = 1 (result always 0)"],
            "recommit_conditions": ["large exponent (b >> 1)", "cryptographic or number-theoretic context"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.02,
        },
        "source_corpus_refs": [QA1, QA2, QA3, P1, P2, QUAD],
        "corpus_concepts":    ["arithmetic", "modular", "integer", "period", "congruence", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "a^b mod m traces the orbit of a under repeated multiplication in Z/mZ. "
            "Fermat's little theorem: a^{p-1} ≡ 1 (mod p) — the orbit period divides p-1. "
            "QA orbit periods (1, 8, 24) divide φ(m) for valid QA moduli. "
            "Repeated squaring mirrors QA double-step: T² state reached in log steps."
        ),
    },

    {
        "name": "continued_fractions", "family": "number_theory",
        "goal": "Represent real number as sequence of integer partial quotients [a₀;a₁,a₂,…]; best rational approximations",
        "orbit_seed": [1, 1],   # φ = [1;1,1,1,...] — all ones
        "levin_cell_type": "differentiated",
        "organ_roles": ["rational_approximator", "diophantine_solver", "phi_expander", "pell_solver"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["irrational with no pattern (random partial quotients)", "finite precision truncation"],
        "composition_rules": ["pair with Euclidean GCD (same recursion)", "convergents feed Pell equation solver"],
        "differentiation_profile": {
            "dediff_conditions":   ["target is rational (finite CF)", "partial quotients grow unboundedly"],
            "recommit_conditions": ["φ approximation needed", "best rational approx for Diophantine task"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.03,
        },
        "source_corpus_refs": [QA1, QA2, QA3, P1, P2, QUAD, WB],
        "corpus_concepts":    ["fibonacci", "phi", "golden ratio", "proportion", "rational", "period", "orbit", "convergence"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "DEEPEST QA CONNECTION in the registry: φ = [1;1,1,1,...] — "
            "the continued fraction of φ is all-ones, the simplest possible. "
            "The convergents are Fibonacci ratios F_{n+1}/F_n → φ. "
            "QA arithmetic is arithmetic in Z[φ] = Z[1;1,1,...]. "
            "Every CF convergent is a QA orbit state (Fibonacci pair). "
            "orbit_seed (1,1): the Fibonacci / φ seed."
        ),
    },

    # ── planning (NEW family, 8) ───────────────────────────────────────────────

    {
        "name": "ida_star", "family": "planning",
        "goal": "Memory-efficient optimal search via iterative deepening with A* heuristic; O(bd) space vs O(b^d) for A*",
        "orbit_seed": [2, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["optimal_path_finder", "memory_efficient_searcher", "heuristic_pruner"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["inadmissible heuristic (loses optimality)", "large branching factor with poor heuristic"],
        "composition_rules": ["drop-in replacement for A* when memory-constrained", "pair with pattern databases for heuristic"],
        "differentiation_profile": {
            "dediff_conditions":   ["heuristic too weak (exponential nodes)", "infinite branching factor"],
            "recommit_conditions": ["admissible heuristic available", "memory << O(b^d)"],
            "max_satellite_cycles": 4,
            "drift_threshold": 0.15,
        },
        "source_corpus_refs": [QA3, QA4, WB],
        "corpus_concepts":    ["orbit", "convergence", "measure", "threshold", "proportion"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "IDA* threshold = QA orbit depth limit: expand nodes within f-cost ≤ threshold, "
            "then increase threshold by minimum excess. Maps to QA orbit step budget: "
            "deepen exploration until orbit period bound is reached, then extend."
        ),
    },

    {
        "name": "dynamic_programming", "family": "planning",
        "goal": "Solve optimisation via overlapping subproblems and optimal substructure; memoisation or tabulation",
        "orbit_seed": [1, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["subproblem_memoiser", "optimal_combiner", "value_table_builder"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["no optimal substructure", "state space exponential", "continuous state (use ADP)"],
        "composition_rules": ["Viterbi is DP on HMM", "Bellman equations = DP for MDPs", "matrix-chain / sequence alignment"],
        "differentiation_profile": {
            "dediff_conditions":   ["subproblems not independent (overlapping side effects)", "state space infinite"],
            "recommit_conditions": ["optimal substructure proven", "overlapping subproblems identified"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.08,
        },
        "source_corpus_refs": [QA1, QA2, QA3, QA4, WB],
        "corpus_concepts":    ["orbit", "period", "convergence", "measure", "proportion", "invariant"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "DP table over QA orbit states: dp[t][(b,e)] = optimal value reaching orbit "
            "state (b,e) at step t. Bellman backup = QA orbit step with value propagation. "
            "The orbit reachability table IS a DP table — QA orbit classification is DP."
        ),
    },

    {
        "name": "branch_and_bound", "family": "planning",
        "goal": "Exact combinatorial optimisation via tree search with upper/lower bound pruning; backbone of MIP solvers",
        "orbit_seed": [3, 6],
        "levin_cell_type": "progenitor",
        "organ_roles": ["combinatorial_optimizer", "bound_pruner", "subtree_explorer"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["weak bounds (no pruning)", "exponential worst-case with poor branching order"],
        "composition_rules": ["LP relaxation provides lower bound", "pair with greedy heuristic for upper bound"],
        "differentiation_profile": {
            "dediff_conditions":   ["optimal subtree found and verified", "all branches pruned"],
            "recommit_conditions": ["better bound found (incumbent updated)", "unpruned subtrees remain"],
            "max_satellite_cycles": 10,
            "drift_threshold": 0.30,
        },
        "source_corpus_refs": [QA3, QA4, P1, WB],
        "corpus_concepts":    ["orbit", "convergence", "measure", "threshold", "boundary"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "beam_search", "family": "planning",
        "goal": "Breadth-first search retaining only top-k candidates per level; trades completeness for tractability",
        "orbit_seed": [4, 6],
        "levin_cell_type": "progenitor",
        "organ_roles": ["top_k_explorer", "hypothesis_pruner", "sequence_decoder"],
        "cognitive_horizon": "regional",
        "convergence":       "heuristic",
        "failure_modes":     ["beam too narrow (misses optimal)", "all beams collapse to same candidate"],
        "composition_rules": ["used in MT / ASR / LLM decoding", "pair with length normalisation for sequence tasks"],
        "differentiation_profile": {
            "dediff_conditions":   ["beam width → 1 (greedy)", "beam width → ∞ (BFS)"],
            "recommit_conditions": ["beam diversity maintained", "scoring function discriminative"],
            "max_satellite_cycles": 6,
            "drift_threshold": 0.25,
        },
        "source_corpus_refs": [QA3, QA4, WB],
        "corpus_concepts":    ["orbit", "convergence", "period", "measure", "proportion"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Beam width = QA orbit ensemble size: keep top-k orbit trajectories, "
            "prune the rest. Beam collapse = orbit singularity: all trajectories converge "
            "to the same fixed point. Satellite beam: top-k cycle without converging."
        ),
    },

    {
        "name": "cdcl_sat", "family": "planning",
        "goal": "DPLL with Conflict-Driven Clause Learning; industrial SAT backbone; solves millions of variables in practice",
        "orbit_seed": [5, 4],
        "levin_cell_type": "progenitor",
        "organ_roles": ["constraint_satisfier", "conflict_learner", "unit_propagator", "backjumper"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["intractable instance (random 3-SAT near phase transition)", "UNSAT detection expensive"],
        "composition_rules": ["backbone of MIP solvers", "pair with constraint_propagation as preprocessing"],
        "differentiation_profile": {
            "dediff_conditions":   ["satisfying assignment found", "UNSAT proven via resolution"],
            "recommit_conditions": ["conflict encountered (new clause learned)", "restart triggered"],
            "max_satellite_cycles": 15,
            "drift_threshold": 0.40,
        },
        "source_corpus_refs": [QA3, QA4, WB],
        "corpus_concepts":    ["orbit", "invariant", "convergence", "measure", "threshold", "boundary"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "CDCL conflict = QA orbit obstruction: the system reaches a state from which "
            "no valid assignment (orbit) exists. Clause learning = recording the obstruction "
            "in the failure algebra. Restarts = dedifferentiation events. "
            "Unit propagation = QA forced-step: only one valid orbit extension."
        ),
    },

    {
        "name": "constraint_propagation", "family": "planning",
        "goal": "Reduce search space by enforcing local consistency (arc, path, bound); removes impossible values before search",
        "orbit_seed": [1, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["domain_reducer", "arc_enforcer", "constraint_pusher"],
        "cognitive_horizon": "local",
        "convergence":       "guaranteed",
        "failure_modes":     ["constraint too weak (no pruning)", "global constraints intractable to propagate"],
        "composition_rules": ["always precedes backtracking search", "pair with CDCL for SAT problems"],
        "differentiation_profile": {
            "dediff_conditions":   ["empty domain detected (infeasible)", "no further pruning possible"],
            "recommit_conditions": ["new assignment reduces domain", "propagation triggered"],
            "max_satellite_cycles": 3,
            "drift_threshold": 0.10,
        },
        "source_corpus_refs": [QA1, QA3, QA4, WB],
        "corpus_concepts":    ["orbit", "boundary", "invariant", "convergence", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Arc consistency = QA reachability pruning: for each domain value, "
            "check whether a supporting orbit extension exists. "
            "AC-3 fixpoint = QA orbit invariant: the set of reachable states cannot shrink further. "
            "Failure algebra: domain wipeout = structural unreachability proof."
        ),
    },

    {
        "name": "mcts", "family": "planning",
        "goal": "Monte Carlo Tree Search: UCT-guided tree expansion with random rollouts; combines exploration and exploitation",
        "orbit_seed": [6, 5],
        "levin_cell_type": "progenitor",
        "organ_roles": ["tree_builder", "rollout_simulator", "value_backpropagator", "uct_selector"],
        "cognitive_horizon": "global",
        "convergence":       "asymptotic",
        "failure_modes":     ["insufficient simulations (poor value estimates)", "adversarial opponent models wrong"],
        "composition_rules": ["replace minimax in high-branching games", "pair with neural value/policy net (AlphaZero)"],
        "differentiation_profile": {
            "dediff_conditions":   ["budget exhausted", "terminal node reached"],
            "recommit_conditions": ["new tree node expanded", "backpropagation updates ancestors"],
            "max_satellite_cycles": 8,
            "drift_threshold": 0.28,
        },
        "source_corpus_refs": [QA3, QA4, P1, WB],
        "corpus_concepts":    ["orbit", "convergence", "resonance", "cycle", "proportion", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "MCTS tree = QA orbit tree: each node is an orbit state; edges are orbit steps. "
            "UCT exploration bonus = QA orbit novelty reward (visit uncharted orbit regions). "
            "Value backpropagation = QA backward orbit labelling with reachability scores. "
            "AlphaZero = MCTS + neural orbit evaluator."
        ),
    },

    {
        "name": "minimax_alphabeta", "family": "planning",
        "goal": "Optimal adversarial search with α-β pruning; eliminates branches that cannot affect optimal play",
        "orbit_seed": [3, 4],
        "levin_cell_type": "differentiated",
        "organ_roles": ["adversarial_planner", "game_tree_searcher", "alphabeta_pruner"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["branching factor too high (impractical depth)", "non-zero-sum game (wrong model)"],
        "composition_rules": ["pair with transposition table for repeated positions", "iterative deepening for time control"],
        "differentiation_profile": {
            "dediff_conditions":   ["terminal node or depth limit reached", "α ≥ β (pruning)"],
            "recommit_conditions": ["new incumbent found (α updated)", "deeper search warranted"],
            "max_satellite_cycles": 4,
            "drift_threshold": 0.12,
        },
        "source_corpus_refs": [QA3, QA4, WB],
        "corpus_concepts":    ["orbit", "boundary", "invariant", "measure", "convergence"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    # ── learn (5 more → 18 total) ─────────────────────────────────────────────

    {
        "name": "hopfield_network", "family": "learn",
        "goal": "Recurrent associative memory converging to stored patterns; energy minimisation via Lyapunov function",
        "orbit_seed": [7, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["pattern_retriever", "associative_memory", "energy_minimiser"],
        "cognitive_horizon": "local",
        "convergence":       "guaranteed",
        "failure_modes":     ["spurious attractors", "capacity limit (0.138N patterns)", "saturation with correlated patterns"],
        "composition_rules": ["use as error-correcting memory for noisy QA states", "energy function ~ QA Lyapunov function"],
        "differentiation_profile": {
            "dediff_conditions":   ["spurious attractor reached", "energy landscape flat"],
            "recommit_conditions": ["stored pattern retrieved (energy minimum)", "Hamming distance converging"],
            "max_satellite_cycles": 4,
            "drift_threshold": 0.15,
        },
        "source_corpus_refs": [QA3, QA4, P1, WB],
        "corpus_concepts":    ["orbit", "convergence", "resonance", "fixed point", "harmonic", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Hopfield dynamics ARE a nonlinear QA orbit: each neuron update is "
            "sign(∑ w_{ij} s_j) — a threshold-gated orbit step. "
            "Attractors = QA singularities (period-1 orbits). "
            "Energy function E = -½ s·W·s maps to QA norm function N(b+eφ)."
        ),
    },

    {
        "name": "self_organizing_map", "family": "learn",
        "goal": "Unsupervised topology-preserving dimensionality reduction; learns 2D map of high-dim data manifold",
        "orbit_seed": [4, 7],
        "levin_cell_type": "progenitor",
        "organ_roles": ["topology_mapper", "manifold_projector", "competitive_learner"],
        "cognitive_horizon": "global",
        "convergence":       "asymptotic",
        "failure_modes":     ["folded map (topology not preserved)", "learning rate too high (oscillation)"],
        "composition_rules": ["use as orbit manifold visualiser", "feeds clustering or classification post-hoc"],
        "differentiation_profile": {
            "dediff_conditions":   ["neighbourhood radius shrinks to 0", "map topology stabilised"],
            "recommit_conditions": ["topology still being shaped", "neighbourhood radius > 0"],
            "max_satellite_cycles": 7,
            "drift_threshold": 0.25,
        },
        "source_corpus_refs": [QA3, QA4, P3, WB],
        "corpus_concepts":    ["orbit", "dimension", "measure", "resonance", "cluster", "period", "harmonic"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "SOM maps high-dim data to 2D grid preserving topology — "
            "same goal as projecting QA 4-tuples (b,e,d,a) to E8 (8D) or 2D visualisation. "
            "Winner-takes-all = orbit attractor pull. Neighbourhood function = orbit coupling radius."
        ),
    },

    {
        "name": "echo_state_network", "family": "learn",
        "goal": "Reservoir computing: fixed random recurrent network with trained readout; efficient temporal learning",
        "orbit_seed": [6, 1],
        "levin_cell_type": "progenitor",
        "organ_roles": ["temporal_pattern_learner", "reservoir_projector", "readout_trainer"],
        "cognitive_horizon": "regional",
        "convergence":       "asymptotic",
        "failure_modes":     ["reservoir not at 'edge of chaos' (spectral radius wrong)", "input washout (memory too short)"],
        "composition_rules": ["reservoir fixed; only train readout (fast)", "pair with Wiener filter for linear readout"],
        "differentiation_profile": {
            "dediff_conditions":   ["readout weights converged", "echo state property satisfied"],
            "recommit_conditions": ["non-stationary input shifts reservoir dynamics", "spectral radius needs tuning"],
            "max_satellite_cycles": 8,
            "drift_threshold": 0.30,
        },
        "source_corpus_refs": [QA3, QA4, P1, WB],
        "corpus_concepts":    ["orbit", "resonance", "cycle", "harmonic", "convergence", "period", "wave"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Echo state network reservoir IS a random QA-like orbit system: "
            "the reservoir state x_{t+1} = tanh(W·x_t + W_in·u_t) is a driven orbit. "
            "Echo state property = orbit stability (perturbations wash out). "
            "Spectral radius ≈ 1 = edge between cosmos and satellite behaviour."
        ),
    },

    {
        "name": "belief_propagation", "family": "learn",
        "goal": "Exact/approximate marginal inference on graphical models via message-passing between variable and factor nodes",
        "orbit_seed": [3, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["marginal_computer", "message_passer", "graphical_model_inferencer"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["loopy graph (loopy BP may not converge)", "messages double-counted in cycles"],
        "composition_rules": ["exact on trees; loopy approximation on general graphs", "Viterbi = max-product BP variant"],
        "differentiation_profile": {
            "dediff_conditions":   ["all messages converged (fixed point)", "tree graph (exact result)"],
            "recommit_conditions": ["message update changes > threshold", "new observation arrives"],
            "max_satellite_cycles": 5,
            "drift_threshold": 0.15,
        },
        "source_corpus_refs": [QA2, QA3, QA4, WB],
        "corpus_concepts":    ["orbit", "period", "invariant", "convergence", "measure", "proportion"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "BP messages are orbit probability distributions: μ_{x→f}(x) = probability "
            "of orbit state x given upstream evidence. Fixed-point messages = QA orbit invariant. "
            "Loopy BP on QA orbit graph = approximate inference over orbit history."
        ),
    },

    {
        "name": "ucb_bandit", "family": "learn",
        "goal": "Multi-armed bandit with Upper Confidence Bound exploration; O(log T) regret via optimism under uncertainty",
        "orbit_seed": [5, 6],
        "levin_cell_type": "progenitor",
        "organ_roles": ["arm_selector", "exploration_balancer", "regret_minimiser"],
        "cognitive_horizon": "local",
        "convergence":       "asymptotic",
        "failure_modes":     ["non-stationary rewards (use sliding window UCB)", "adversarial bandit"],
        "composition_rules": ["MCTS uses UCB1 for node selection", "feeds online RL as subroutine"],
        "differentiation_profile": {
            "dediff_conditions":   ["one arm clearly dominant (exploit forever)", "exploration budget exhausted"],
            "recommit_conditions": ["confidence intervals still wide", "new arm added"],
            "max_satellite_cycles": 9,
            "drift_threshold": 0.28,
        },
        "source_corpus_refs": [QA3, QA4, WB],
        "corpus_concepts":    ["orbit", "convergence", "resonance", "measure", "proportion"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "UCB arm selection = QA orbit exploration policy: "
            "UCB(a) = Q̂(a) + √(2 log t / n_a) balances exploitation (Q̂) and exploration (1/n_a). "
            "QA analog: prefer orbit states with high estimated value OR low visit count. "
            "The progenitor cell: still exploring orbit space, not yet committed."
        ),
    },

    # ── geometry (NEW family, 3) ───────────────────────────────────────────────

    {
        "name": "convex_hull", "family": "geometry",
        "goal": "Find minimal convex polygon enclosing a point set; Graham scan / Jarvis march in O(N log N)",
        "orbit_seed": [2, 7],
        "levin_cell_type": "differentiated",
        "organ_roles": ["boundary_finder", "convexity_enforcer", "extreme_point_selector"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["collinear points (degenerate hull)", "floating-point precision failures"],
        "composition_rules": ["precedes Delaunay triangulation", "used in collision detection, gift-wrapping"],
        "differentiation_profile": {
            "dediff_conditions":   ["all points collinear (degenerate)", "single point"],
            "recommit_conditions": ["points in general position", "enclosure query needed"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.08,
        },
        "source_corpus_refs": [QA3, QA4, QUAD, P2],
        "corpus_concepts":    ["quadrance", "spread", "area", "proportion", "boundary", "measure", "invariant"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Convex hull = QA reachability boundary in state space: "
            "the set of all orbit states reachable from initial conditions forms a convex region "
            "in the quadrance metric. Hull vertices = extreme orbit states. "
            "QA spread (angle) between consecutive hull vertices = angular orbit step."
        ),
    },

    {
        "name": "delaunay_triangulation", "family": "geometry",
        "goal": "Triangulate point set maximising minimum angle; circumcircle of each triangle contains no other points",
        "orbit_seed": [1, 6],
        "levin_cell_type": "differentiated",
        "organ_roles": ["optimal_triangulator", "voronoi_dual_builder", "mesh_generator"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["degenerate configuration (co-circular points)", "floating-point incircle test failures"],
        "composition_rules": ["dual of Voronoi diagram", "prerequisite for finite element mesh generation"],
        "differentiation_profile": {
            "dediff_conditions":   ["co-circular degenerate case", "infinite Voronoi edges"],
            "recommit_conditions": ["points in general position", "mesh quality optimisation needed"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.08,
        },
        "source_corpus_refs": [QA3, QA4, QUAD, P2],
        "corpus_concepts":    ["quadrance", "spread", "area", "proportion", "measure", "triple"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Delaunay circumcircle condition = QA spread condition: no point inside the "
            "circumcircle means no orbit state inside the quadrance sphere of the triangle. "
            "The triangulation maximises minimum angle — equivalent to maximising the minimum "
            "spread (QA angle measure) across all triangle edges."
        ),
    },

    {
        "name": "voronoi_diagram", "family": "geometry",
        "goal": "Partition plane into regions closest to each seed point; dual of Delaunay triangulation; O(N log N)",
        "orbit_seed": [5, 1],
        "levin_cell_type": "differentiated",
        "organ_roles": ["nearest_neighbour_partitioner", "proximity_mapper", "territory_assigner"],
        "cognitive_horizon": "global",
        "convergence":       "guaranteed",
        "failure_modes":     ["degenerate co-circular points (ill-defined boundaries)", "unbounded regions at convex hull"],
        "composition_rules": ["dual of Delaunay; compute together", "feeds k-means (Lloyd's algorithm uses Voronoi)"],
        "differentiation_profile": {
            "dediff_conditions":   ["degenerate configuration", "unbounded diagram"],
            "recommit_conditions": ["points in general position", "proximity / territory partition needed"],
            "max_satellite_cycles": 2,
            "drift_threshold": 0.08,
        },
        "source_corpus_refs": [QA3, QA4, QUAD, P2],
        "corpus_concepts":    ["quadrance", "spread", "area", "proportion", "boundary", "measure"],
        "needs_ocr_backfill": False,
        "confidence": "high",
        "qa_research_note": (
            "Voronoi cells = QA orbit basins of attraction: each cell is the set of "
            "initial states that converge to the same orbit attractor. "
            "Cell boundaries = QA obstruction surfaces (unreachable transition lines). "
            "Lloyd's k-means iteration = Voronoi → centroid → repeat: orbit centroid update."
        ),
    },

]


# ── Schema enrichment defaults ─────────────────────────────────────────────────

FAMILY_DEFAULTS = {
    "sort": {
        "cognitive_horizon": "local", "convergence": "guaranteed",
        "failure_modes":     ["pathological comparator", "stability violation"],
        "composition_rules": ["pipeline after data normalisation", "stable sort before secondary key"],
    },
    "search": {
        "cognitive_horizon": "regional", "convergence": "guaranteed",
        "failure_modes":     ["unsorted input (binary variants)", "target absent"],
        "composition_rules": ["precondition: sorted/indexed structure", "compose with sort as preprocessing"],
    },
    "graph": {
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "failure_modes":     ["negative cycles", "disconnected graph"],
        "composition_rules": ["BFS/DFS as subroutines", "Dijkstra → Bellman-Ford fallback"],
    },
    "optimize": {
        "cognitive_horizon": "regional", "convergence": "asymptotic",
        "failure_modes":     ["non-convex local minima", "ill-conditioned Hessian", "vanishing gradients"],
        "composition_rules": ["wrap with lr scheduler", "combine with momentum"],
    },
    "learn": {
        "cognitive_horizon": "global", "convergence": "probabilistic",
        "failure_modes":     ["overfitting", "distribution shift", "label noise"],
        "composition_rules": ["pipeline with feature engineering", "ensemble with orthogonal learners"],
    },
    "control": {
        "cognitive_horizon": "local", "convergence": "guaranteed",
        "failure_modes":     ["plant model mismatch", "actuator saturation"],
        "composition_rules": ["outer/inner-loop cascade", "state estimator feeds controller"],
    },
    "distributed": {
        "cognitive_horizon": "global", "convergence": "probabilistic",
        "failure_modes":     ["network partition", "Byzantine nodes"],
        "composition_rules": ["consensus under application layer", "gossip feeds DHT routing"],
    },
    "time_series": {
        "cognitive_horizon": "regional", "convergence": "asymptotic",
        "failure_modes":     ["non-stationarity", "structural break"],
        "composition_rules": ["decompose trend+season first", "ensemble with spectral methods"],
    },
    "signal_processing": {
        "cognitive_horizon": "regional", "convergence": "guaranteed",
        "failure_modes":     ["aliasing", "spectral leakage", "phase distortion"],
        "composition_rules": ["filter → transform → detect pipeline", "PLL before coherent detection"],
    },
    "number_theory": {
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "failure_modes":     ["integer overflow for large N", "Carmichael false positives"],
        "composition_rules": ["GCD → extended GCD → modular inverse → CRT", "sieve feeds factorisation"],
    },
    "planning": {
        "cognitive_horizon": "global", "convergence": "heuristic",
        "failure_modes":     ["inadmissible heuristic", "exponential branching", "unsatisfiability"],
        "composition_rules": ["heuristic feeds beam/A* search", "SAT as CSP backbone"],
    },
    "geometry": {
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "failure_modes":     ["degenerate inputs (collinear/co-circular)", "floating-point precision"],
        "composition_rules": ["convex hull → Voronoi → Delaunay pipeline", "triangulation feeds mesh"],
    },
}

NEW_SCHEMA_FIELDS = ["cognitive_horizon", "convergence", "failure_modes", "composition_rules"]


def make_entry(d: dict, modulus: int = MODULUS) -> dict:
    b, e = d["orbit_seed"]
    sig  = qa_orbit_family(b, e, modulus)
    ofr  = orbit_follow_rate(b, e, modulus)
    return {
        "name":                    d["name"],
        "family":                  d["family"],
        "goal":                    d["goal"],
        "orbit_seed":              [b, e],
        "orbit_signature":         d.get("orbit_signature", sig),
        "orbit_follow_rate":       ofr,
        "cognitive_horizon":       d["cognitive_horizon"],
        "convergence":             d["convergence"],
        "levin_cell_type":         d["levin_cell_type"],
        "organ_roles":             d["organ_roles"],
        "failure_modes":           d["failure_modes"],
        "composition_rules":       d["composition_rules"],
        "differentiation_profile": d["differentiation_profile"],
        "source_corpus_refs":      d.get("source_corpus_refs", []),
        "corpus_concepts":         d.get("corpus_concepts", []),
        "needs_ocr_backfill":      d.get("needs_ocr_backfill", False),
        "confidence":              d.get("confidence", "medium"),
        "qa_research_note":        d.get("qa_research_note", ""),
    }


def backfill_schema(reg: dict) -> int:
    """Add missing new schema fields to all existing entries using family defaults."""
    updated = 0
    for alg in reg["algorithms"]:
        if all(f in alg for f in NEW_SCHEMA_FIELDS):
            continue
        fam      = alg.get("family", "learn")
        defaults = FAMILY_DEFAULTS.get(fam, FAMILY_DEFAULTS["learn"])
        changed  = False
        for field in NEW_SCHEMA_FIELDS:
            if field not in alg:
                alg[field] = defaults[field]
                changed = True
        if changed:
            updated += 1
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    reg      = json.loads(REG_PATH.read_text()) if REG_PATH.exists() else {"algorithms": [], "corpus_status": {}}
    existing = {a["name"] for a in reg["algorithms"]}

    # New entries
    new_entries = []
    for d in BATCH5:
        if d["name"] in existing:
            print(f"  ! skip duplicate: {d['name']}")
            continue
        entry = make_entry(d)
        sig   = entry["orbit_signature"]
        fam   = entry["family"]
        print(f"  + {entry['name']:38s} {sig:12s} {fam}")
        new_entries.append(entry)

    # Schema backfill for existing 72 entries
    if not args.dry_run:
        reg["algorithms"].extend(new_entries)
        backfilled = backfill_schema(reg)
        REG_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False))

        total  = len(reg["algorithms"])
        fams   = Counter(a["family"]          for a in reg["algorithms"])
        orbits = Counter(a["orbit_signature"] for a in reg["algorithms"])
        cells  = Counter(a["levin_cell_type"] for a in reg["algorithms"])

        print(f"\nMerged {len(new_entries)} new + backfilled schema on {backfilled} existing")
        print(f"Total: {total} algorithms")
        print(f"\nFamilies:")
        for fam, cnt in sorted(fams.items(), key=lambda x: -x[1]):
            print(f"  {fam:22s} {cnt:3d}")
        print(f"\nOrbits:  {dict(orbits)}")
        print(f"Cells:   {dict(cells)}")
    else:
        print(f"\n[dry-run] would add {len(new_entries)} algorithms + backfill schema on existing")


if __name__ == "__main__":
    main()
