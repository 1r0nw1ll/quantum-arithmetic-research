#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_algorithm_competency_batch2.py
====================================
Second batch: 15 more algorithms bringing total to 36.
Families: optimize (3 more), learn (3 more), distributed (5), control (2 more), graph (2 more)

Merges into qa_algorithm_competency_registry.json.

Usage:
  python qa_algorithm_competency_batch2.py          # merge into registry
  python qa_algorithm_competency_batch2.py --dry-run  # print without writing
"""

from __future__ import annotations
import json, sys, argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List

MODULUS = 9
REG_PATH = Path("qa_algorithm_competency_registry.json")

QA1  = "qa-1__qa_1_all_pages__docx.md"
QA2  = "qa-2__001_qa_2_all_pages__docx.md"
QA3  = "qa_3__ocr__qa3.md"
QA4  = "qa-4__00_qa_books_3_&_4_all_pages__pdf.md"
QUAD = "quadrature__00_quadratureprint__pdf.md"
P3   = "pyth-3__pythagoras_vol3_enneagram__docx.md"

def qa_step(b,e,m=MODULUS): return e%m,(b+e)%m
def qa_orbit_family(b,e,m=MODULUS,max_steps=500):
    seen,state={},((b%m),(e%m))
    for t in range(max_steps):
        if state in seen:
            p=t-seen[state]
            return "singularity" if p==1 else "satellite" if p==8 else "cosmos" if p==24 else f"period_{p}"
        seen[state]=t; state=qa_step(*state,m)
    return "unknown"
def ofr(b,e,m=MODULUS,steps=48):
    traj,state=[],((b%m),(e%m))
    for _ in range(steps): traj.append(state); state=qa_step(*state,m)
    return round(sum(1 for i in range(len(traj)-2) if traj[i+1][0]==traj[i][1]%m and traj[i+1][1]==(traj[i][0]+traj[i][1])%m)/(len(traj)-2),4) if len(traj)>=3 else 0.0

# ── Batch 2 algorithms ────────────────────────────────────────────────────────

BATCH2 = [

    # ── optimize (3) ──────────────────────────────────────────────────────────

    {
        "name": "newtons_method", "family": "optimize",
        "goal": "Find root of f(x)=0 via x_{n+1} = x_n - f(x_n)/f'(x_n); quadratic convergence near root",
        "cognitive_horizon": "local", "convergence": "conditional",
        "time_complexity": "O(D²·T) D=params, T=steps (Hessian cost)", "space_complexity": "O(D²)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Near the root, Newton's method converges quadratically — doubling the correct digits per step. "
            "This maps to the cosmos orbit's 24-step cycle folding in on itself. The Hessian inverse "
            "is the curvature tensor of the QA state space; each step is a maximum-curvature orbit jump. "
            "QA-1 Law 12: 'twice the product of the intermediate roots forms the base' — Newton's "
            "update is geometrically the altitude of the QA right triangle formed by the current gradient. "
            "Far from root (ill-conditioned): satellite orbit — Newton can overshoot indefinitely."
        ),
        "orbit_seed": [1, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["quadratic_optimizer", "hessian_inverter", "second_order_spine"],
        "failure_modes": [
            "Far from root: overshoot → satellite/divergence",
            "Singular Hessian: undefined step → singularity",
            "Non-convex: converges to saddle point, not minimum",
            "O(D²) cost per step — infeasible for large neural nets",
        ],
        "composition_rules": [
            "adaptive:quasi_newton_BFGS (approx Hessian to avoid O(D²))",
            "sequential:gradient_descent (GD as fallback far from root)",
            "hybrid:gauss_newton (for least-squares problems)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Hessian singular or near-singular", "Overshoot detected (loss increasing)"],
            "recommit_conditions": ["Near root with well-conditioned Hessian"],
            "max_satellite_cycles": 2, "drift_threshold": 0.08, "partial_fail_threshold": 3,
        },
        "source_corpus_refs": [QA1, QA4],
        "corpus_concepts": ["hessian_as_qa_curvature", "quadratic_convergence_cosmos", "law_12_altitude"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "genetic_algorithm", "family": "optimize",
        "goal": "Population-based optimization via selection, crossover, mutation across generations",
        "cognitive_horizon": "adaptive", "convergence": "probabilistic",
        "time_complexity": "O(G·P·D) G=generations, P=pop size, D=dim", "space_complexity": "O(P·D)",
        "orbit_signature": "satellite",
        "orbit_rationale": (
            "A population of candidate solutions evolves over generations — a cycling orbit "
            "where the 'fitness landscape' is the QA orbit attractor. Selection pressure = "
            "orbit damper: without it, random mutation = satellite random walk. "
            "QA-1 Law 10: 'male/female root structure' maps to parent pair crossover — "
            "the two-parent crossover produces offspring matching the QA (b,e)→(d,a) recurrence. "
            "Elitism (keep best individual) = orbit anchoring: prevents satellite from drifting "
            "away from cosmos. Schema theorem = QA invariant: short, high-fitness schemata "
            "survive across generations like QA quantum number invariants."
        ),
        "orbit_seed": [2, 3],
        "levin_cell_type": "progenitor",
        "organ_roles": ["population_evolver", "global_explorer", "schema_discoverer"],
        "failure_modes": [
            "Premature convergence: population collapses to local optimum (satellite trap)",
            "Schema disruption: high mutation rate destroys good building blocks",
            "Fitness landscape deception: GA driven away from global optimum",
            "No convergence guarantee: may cycle indefinitely",
        ],
        "composition_rules": [
            "adaptive:differential_evolution (GA variant, better continuous domains)",
            "hierarchical:memetic_algorithm (GA + local search)",
            "parallel:island_model (multiple sub-populations, periodic migration)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Fitness diversity < 5% (premature convergence)", "Fitness not improving for 100 generations"],
            "recommit_conditions": ["Diversity restored via mutation injection, fitness increasing"],
            "max_satellite_cycles": 8, "drift_threshold": 0.12, "partial_fail_threshold": 10,
        },
        "source_corpus_refs": [QA1, QA3],
        "corpus_concepts": ["male_female_root_crossover", "schema_as_quantum_invariant", "selection_orbit_damper", "law_10_parent_structure"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
    },

    {
        "name": "bayesian_optimization", "family": "optimize",
        "goal": "Global optimization of expensive black-box f via surrogate model + acquisition function",
        "cognitive_horizon": "global", "convergence": "probabilistic",
        "time_complexity": "O(N³) per step (GP kernel inversion)", "space_complexity": "O(N²)",
        "orbit_signature": "mixed",
        "orbit_rationale": (
            "Bayesian optimization alternates between exploitation (sample near current best = cosmos) "
            "and exploration (sample in uncertain region = satellite). The Gaussian Process surrogate "
            "is the QA orbit model: it predicts the orbit trajectory of the objective function. "
            "Expected Improvement (EI) acquisition = QA orbit selector: balances cosmos (exploit) "
            "and satellite (explore). With sufficient samples, converges to global optimum = cosmos. "
            "QA-3 Quantizing: BO is the probabilistic quantizer — it finds the quantum number "
            "(global optimum) by measuring at strategically chosen states."
        ),
        "orbit_seed": [3, 6],
        "levin_cell_type": "progenitor",
        "organ_roles": ["surrogate_modeler", "acquisition_oracle", "hyperparameter_tuner"],
        "failure_modes": [
            "O(N³) GP cost: infeasible beyond ~1000 observations",
            "Misspecified kernel: surrogate fails to capture true landscape",
            "High dimensions (> 20): curse of dimensionality degrades GP",
            "Non-stationary landscape: GP assumptions violated",
        ],
        "composition_rules": [
            "sequential:gaussian_process (BO uses GP as surrogate)",
            "adaptive:TPE (Tree Parzen Estimator — BO without GP for high-dim)",
            "hierarchical:multi_fidelity_BO (cheap approximations at low fidelity)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["N > 1000 (GP cost prohibitive)", "Dim > 20 (GP degrades)"],
            "recommit_conditions": ["Low-dimensional, expensive evaluation budget < 200"],
            "max_satellite_cycles": 5, "drift_threshold": 0.15, "partial_fail_threshold": 5,
        },
        "source_corpus_refs": [QA3, QA4],
        "corpus_concepts": ["probabilistic_quantizer", "surrogate_orbit_model", "exploit_explore_balance"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
    },

    # ── learn (3) ─────────────────────────────────────────────────────────────

    {
        "name": "random_forest", "family": "learn",
        "goal": "Ensemble of decision trees via bootstrap sampling + random feature subsets",
        "cognitive_horizon": "adaptive", "convergence": "guaranteed",
        "time_complexity": "O(T·N·log N·√D) T=trees", "space_complexity": "O(T·N)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Random forest combines T independent decision trees — each tree is a cosmos orbit "
            "(guaranteed to fit training data). The ensemble average is a convex combination "
            "of cosmos orbits = still cosmos. Bagging breaks correlation between trees, "
            "preventing satellite (correlated error cycling). "
            "QA-4: random forest is a synchronous harmonic ensemble — T trees = T harmonic modes, "
            "each capturing a different frequency of the data distribution. "
            "The majority vote / average = quantum mean of the ensemble."
        ),
        "orbit_seed": [1, 4],
        "levin_cell_type": "differentiated",
        "organ_roles": ["ensemble_backbone", "feature_importance_oracle", "robust_classifier"],
        "failure_modes": [
            "Extrapolation: all trees vote the same for out-of-distribution inputs",
            "Memory: O(T·N) can be large for deep trees",
            "Interpretability: ensemble hides individual tree logic",
            "Correlated features: random subset selection less effective",
        ],
        "composition_rules": [
            "hierarchical:gradient_boosting (sequential vs RF's parallel ensemble)",
            "sequential:PCA (feature reduction before RF)",
            "parallel:distributed_RF (trees trained on partitioned data)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Out-of-distribution detected (all trees agree but wrong)"],
            "recommit_conditions": ["In-distribution confirmed, T ≥ 100 trees"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 3,
        },
        "source_corpus_refs": [QA4, QA1],
        "corpus_concepts": ["synchronous_harmonic_ensemble", "quantum_mean_vote", "independent_orbit_combination"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "svm", "family": "learn",
        "goal": "Find maximum-margin hyperplane separating classes via quadratic programming",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(N²D) to O(N³)", "space_complexity": "O(N) support vectors",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "SVM optimizes a convex QP — guaranteed unique global optimum. "
            "The support vectors are the QA 'quantum roots' of the decision boundary: "
            "the minimum set of points that determines the entire orbit structure. "
            "QA-1 Law 9 ('each set of roots is unique'): the support vector set is unique. "
            "The kernel trick maps to QA extended orbit: transforming feature space = "
            "extending the QA modulus to a higher-dimensional space where the orbit "
            "becomes linearly separable (cosmos in the lifted space). "
            "Soft margin (slack variable) = QA orbit tolerance: allows bounded violations."
        ),
        "orbit_seed": [2, 6],
        "levin_cell_type": "differentiated",
        "organ_roles": ["margin_maximizer", "support_vector_oracle", "kernel_transformer"],
        "failure_modes": [
            "O(N²) to O(N³): infeasible for large N",
            "Kernel choice: wrong kernel = no separability in lifted space",
            "Soft margin C: too small = underfitting; too large = overfitting",
            "Multi-class: requires one-vs-one or one-vs-rest decomposition",
        ],
        "composition_rules": [
            "sequential:PCA (dimensionality reduction before SVM)",
            "adaptive:kernel_SVM (RBF/poly/sigmoid kernels for nonlinear)",
            "hierarchical:SVM_in_cascade (SVM as final classifier in pipeline)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["N > 10^5 (QP infeasible)", "Kernel not positive definite"],
            "recommit_conditions": ["N < 10^4, appropriate kernel validated on validation set"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA1, QA2],
        "corpus_concepts": ["support_vectors_as_quantum_roots", "law_9_unique_roots", "kernel_orbit_extension", "convex_cosmos_guarantee"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "lstm", "family": "learn",
        "goal": "Sequential model with gated memory cell; learns long-range dependencies",
        "cognitive_horizon": "adaptive", "convergence": "conditional",
        "time_complexity": "O(T·D²) T=seq_len, D=hidden_dim", "space_complexity": "O(D²)",
        "orbit_signature": "mixed",
        "orbit_rationale": (
            "LSTM's forget gate (f_t) controls orbit memory: f_t=1 → full cosmos (remembers everything), "
            "f_t=0 → singularity (forgets = no orbit history). "
            "The three gates (input, forget, output) are QA orbit selectors: each gate chooses "
            "which part of the orbit state to preserve. Vanishing gradient: "
            "deep temporal unrolling → satellite orbit (gradient shrinks at each step). "
            "LSTM's constant error carousel (CEC) is the cosmos orbit anchor: "
            "the cell state C_t flows unchanged unless actively modified by gates. "
            "QA-4 Synchronous Harmonics: LSTM memory cell = harmonic resonator that sustains "
            "a frequency pattern across multiple time steps."
        ),
        "orbit_seed": [4, 7],
        "levin_cell_type": "progenitor",
        "organ_roles": ["sequence_memory", "temporal_pattern_holder", "recurrent_backbone"],
        "failure_modes": [
            "Vanishing gradient: very long sequences → satellite orbit in backprop",
            "Exploding gradient: unstable update → escape from orbit",
            "Mode collapse: all gates open/closed → singularity (trivial output)",
            "O(D²) per step: slow for large hidden dimensions",
        ],
        "composition_rules": [
            "hierarchical:stacked_LSTM (depth = multiple harmonic frequencies)",
            "adaptive:attention_over_LSTM (attention selects which memory step to read)",
            "hybrid:transformer (LSTM replacement for long sequences)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Sequence length > 1000 (gradient too diluted)", "Gate saturation detected"],
            "recommit_conditions": ["Sequence length < 500, appropriate lr and gradient clipping"],
            "max_satellite_cycles": 3, "drift_threshold": 0.15, "partial_fail_threshold": 5,
        },
        "source_corpus_refs": [QA4, QA3],
        "corpus_concepts": ["constant_error_carousel_cosmos", "gate_as_orbit_selector", "harmonic_resonator_cell", "forget_gate_singularity"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
    },

    # ── graph (2 more) ────────────────────────────────────────────────────────

    {
        "name": "pagerank", "family": "graph",
        "goal": "Rank nodes in directed graph by stationary distribution of random walk",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(I·(V+E)) I=power iterations", "space_complexity": "O(V)",
        "orbit_signature": "satellite",
        "orbit_rationale": (
            "PageRank iterates the power method on the link matrix — a satellite orbit: "
            "the rank vector cycles until convergence to the stationary distribution. "
            "The damping factor d=0.85 is the orbit damper: prevents rank sinks (singularity "
            "from dangling nodes) and ensures the Markov chain is ergodic (satellite→cosmos). "
            "QA-4 Wave Theory: PageRank is the harmonic resonance of the web graph — "
            "high-rank nodes are the fundamental frequencies; low-rank nodes are harmonics. "
            "Each power iteration = one wave period; convergence = standing wave."
        ),
        "orbit_seed": [3, 4],
        "levin_cell_type": "progenitor",
        "organ_roles": ["node_ranker", "authority_estimator", "graph_harmonic_finder"],
        "failure_modes": [
            "Dangling nodes: no outlinks → rank sink → singularity without damping",
            "Spider traps: strongly connected subgraph absorbs all rank → satellite",
            "Slow convergence on near-disconnected graphs",
            "Sensitive to link spam (adversarial orbit injection)",
        ],
        "composition_rules": [
            "sequential:HITS (hub/authority alternative to PageRank)",
            "parallel:distributed_PageRank (partition graph across machines)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["No damping factor (dangling node singularity)", "Graph changes invalidate stationary distribution"],
            "recommit_conditions": ["Damping d=0.85, graph static for iteration window"],
            "max_satellite_cycles": 100, "drift_threshold": 0.01, "partial_fail_threshold": 10,
        },
        "source_corpus_refs": [QA4, QA3],
        "corpus_concepts": ["stationary_wave_orbit", "harmonic_resonance_ranking", "damping_as_orbit_damper", "power_iteration_satellite"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "topological_sort", "family": "graph",
        "goal": "Linear ordering of DAG vertices such that all edges go forward",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(V+E)", "space_complexity": "O(V)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Topological sort produces a unique linearization of a DAG — a cosmos orbit "
            "that processes all vertices in exactly one forward pass. No backtracking, "
            "no cycles (by DAG definition). Kahn's algorithm: in-degree queue is the "
            "QA frontier expander — each dequeue is one cosmos step. "
            "QA-1 root ordering: topological sort is the canonical root sequence of the DAG "
            "where each root (source node) precedes its derived nodes. "
            "Cycle detection (failure case) is the orbit singularity: a cycle means "
            "the DAG assumption is violated, and the 'sort' loops indefinitely."
        ),
        "orbit_seed": [1, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["dependency_resolver", "build_order_oracle", "dag_linearizer"],
        "failure_modes": [
            "Cycle in graph: topological sort undefined → singularity",
            "Multiple valid orderings: result is not unique (non-deterministic)",
            "Incremental updates: full recompute needed on DAG change",
        ],
        "composition_rules": [
            "sequential:DFS (DFS-based topological sort for cycle detection)",
            "hierarchical:build_system (makefile dependency resolution)",
            "sequential:critical_path_method (topological sort + longest path)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Cycle detected (graph is not a DAG)"],
            "recommit_conditions": ["DAG structure confirmed"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA1, QA2],
        "corpus_concepts": ["root_ordering_sequence", "dag_as_quantum_derivation", "forward_orbit_linearization"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    # ── control (2 more) ──────────────────────────────────────────────────────

    {
        "name": "mpc", "family": "control",
        "goal": "Optimize control sequence over finite horizon H by solving constrained OCP at each step",
        "cognitive_horizon": "bounded", "convergence": "conditional",
        "time_complexity": "O(H·D³) per step (QP solve)", "space_complexity": "O(H·D)",
        "orbit_signature": "mixed",
        "orbit_rationale": (
            "Linear MPC on convex OCP: cosmos orbit — QP solve gives globally optimal control. "
            "Nonlinear MPC: mixed orbit — NLP solver may find local optimum (satellite). "
            "Receding horizon: at each step, the cosmos window slides forward by 1 — "
            "a periodic re-optimization that is itself a satellite-like cycle, but each "
            "cycle step is cosmos-optimal. "
            "QA-4 Wave Theory: MPC is the predictive harmonic controller — the H-step "
            "horizon predicts H future wave periods and selects the control that keeps "
            "the system in the desired harmonic. Terminal constraint = orbit endpoint lock."
        ),
        "orbit_seed": [3, 5],
        "levin_cell_type": "progenitor",
        "organ_roles": ["predictive_controller", "constraint_handler", "horizon_planner"],
        "failure_modes": [
            "Infeasible QP: constraints too tight → singularity (no control found)",
            "Model mismatch: plant differs from model → satellite tracking error",
            "Computation time: QP solve > sample period → real-time violation",
            "Nonlinear: local NLP optimum → suboptimal control",
        ],
        "composition_rules": [
            "sequential:kalman_filter (state estimation feeds MPC)",
            "hierarchical:hierarchical_MPC (coarse + fine horizon levels)",
            "adaptive:NMPC (nonlinear MPC for nonlinear plants)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["QP infeasible for 3 consecutive steps", "Computation exceeds sample period"],
            "recommit_conditions": ["Linear plant, constraints relaxed, H tuned"],
            "max_satellite_cycles": 3, "drift_threshold": 0.15, "partial_fail_threshold": 4,
        },
        "source_corpus_refs": [QA4, QA3],
        "corpus_concepts": ["predictive_harmonic_controller", "receding_horizon_satellite_cycle", "terminal_orbit_lock"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
    },

    {
        "name": "lqr", "family": "control",
        "goal": "Optimal linear state-feedback u=-Kx minimizing ∫(x'Qx + u'Ru)dt",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(D³) Riccati solve (offline)", "space_complexity": "O(D²)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "LQR solves the algebraic Riccati equation — a fixed-point equation whose solution "
            "is the unique positive-definite P matrix. This is the canonical cosmos orbit: "
            "guaranteed convergence to the globally optimal feedback gain K. "
            "The cost function J = x'Px is the QA orbit energy: minimizing J = finding "
            "the lowest-energy orbit. The Q matrix weights state deviation; R weights control effort — "
            "together they define the QA orbit metric. "
            "QA-4: LQR is the linear harmonic controller — it finds the gain that makes "
            "the closed-loop system poles (eigenvalues) land at the desired harmonic frequencies."
        ),
        "orbit_seed": [1, 6],
        "levin_cell_type": "differentiated",
        "organ_roles": ["optimal_linear_controller", "riccati_solver", "pole_placer"],
        "failure_modes": [
            "Nonlinear plant: LQR valid only near linearization point",
            "Uncontrollable system: Riccati has no solution",
            "Wrong Q/R tuning: suboptimal performance (but still stable)",
            "Model uncertainty: robustness margin may be insufficient",
        ],
        "composition_rules": [
            "sequential:kalman_filter (LQG = LQR + Kalman)",
            "adaptive:ILQR (iterative LQR for nonlinear systems)",
            "hierarchical:hierarchical_LQR (coarse + fine control levels)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["System nonlinear beyond linearization validity", "Riccati solver fails (uncontrollable)"],
            "recommit_conditions": ["Linear dynamics confirmed, Q/R tuned via simulation"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA4, QA1],
        "corpus_concepts": ["riccati_fixed_point_cosmos", "orbit_energy_minimization", "closed_loop_harmonic_poles", "lqr_as_quantum_controller"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    # ── distributed (5) ───────────────────────────────────────────────────────

    {
        "name": "raft_consensus", "family": "distributed",
        "goal": "Replicated log consensus via leader election + log replication across N nodes",
        "cognitive_horizon": "bounded", "convergence": "guaranteed",
        "time_complexity": "O(log N) round trips (normal operation)", "space_complexity": "O(log size)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Raft guarantees safety (no two leaders) and liveness (progress if majority alive). "
            "Leader election = cosmos orbit: deterministic winner after randomized timeout. "
            "Log replication = cosmos forward orbit: entries committed in order, never retracted. "
            "QA-1 Law 8 ('the first set of roots 1,1,2,3 represents the ONE'): "
            "the leader is the 'ONE' of Raft — all operations flow through it. "
            "Network partition = satellite trap: minority partition cannot commit (blocked orbit). "
            "But majority partition continues in cosmos."
        ),
        "orbit_seed": [1, 1],
        "levin_cell_type": "differentiated",
        "organ_roles": ["consensus_spine", "log_replicator", "leader_oracle"],
        "failure_modes": [
            "Network partition: minority cannot commit (liveness blocked)",
            "Follower lag: slow follower delays commit quorum",
            "Split vote: simultaneous elections → no leader → satellite retry",
            "Disk failure: log corruption → undefined state",
        ],
        "composition_rules": [
            "hierarchical:multi_raft (multiple Raft groups, each owns a shard)",
            "sequential:log_compaction (snapshot to bound log size)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Majority of nodes unreachable (partition > N/2)"],
            "recommit_conditions": ["Majority connected, leader stable"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 3,
        },
        "source_corpus_refs": [QA1, QA4],
        "corpus_concepts": ["law_8_leader_as_one", "log_as_forward_orbit", "majority_quorum_cosmos"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "map_reduce", "family": "distributed",
        "goal": "Distributed computation via Map (parallelize) → shuffle → Reduce (aggregate)",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(N/W + sort cost) W=workers", "space_complexity": "O(N)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Map phase: independent cosmos orbits on each data partition — all parallel, "
            "all guaranteed to terminate. Reduce phase: aggregation is a deterministic "
            "fold — cosmos forward orbit. The entire computation is a DAG: Map → Shuffle → Reduce, "
            "topologically sorted (cosmos). "
            "QA-4: MapReduce is a synchronous harmonic computation — Map extracts "
            "individual harmonics from data partitions; Reduce combines them into the "
            "fundamental frequency (final result). The shuffle phase = harmonic alignment "
            "that groups same-key results before reduction."
        ),
        "orbit_seed": [2, 7],
        "levin_cell_type": "differentiated",
        "organ_roles": ["parallel_spine", "data_partitioner", "distributed_aggregator"],
        "failure_modes": [
            "Skewed keys: one reducer overwhelmed (satellite bottleneck)",
            "Network bandwidth: shuffle phase dominates on large data",
            "Stragglers: one slow worker blocks all reducers",
            "Iterative algorithms: poor fit (use Spark/BSP instead)",
        ],
        "composition_rules": [
            "hierarchical:iterative_map_reduce (chain for multi-pass algorithms)",
            "parallel:combiner (local pre-aggregation reduces shuffle volume)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Key skew detected (single reducer > 10x average load)"],
            "recommit_conditions": ["Keys uniformly distributed, no straggler workers"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA4, QA3],
        "corpus_concepts": ["harmonic_map_extract", "shuffle_as_harmonic_alignment", "reduce_as_fundamental_frequency", "dag_cosmos_computation"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "gossip_protocol", "family": "distributed",
        "goal": "Probabilistic information dissemination: each node periodically shares state with random peer",
        "cognitive_horizon": "local", "convergence": "probabilistic",
        "time_complexity": "O(log N) rounds to reach all nodes (expected)", "space_complexity": "O(N) state",
        "orbit_signature": "satellite",
        "orbit_rationale": (
            "Gossip cycles: each node repeatedly selects random peer and exchanges state — "
            "a satellite orbit (fixed-period, probabilistic). Convergence to global state "
            "is probabilistic (epidemiological): like a wave propagating through a medium. "
            "QA-4 Wave Theory: gossip propagation = wave packet spreading — each gossip round "
            "is one wave period. Convergence rate = QA harmonic decay constant: "
            "information spreads in O(log N) periods, matching the harmonic series 1/1, 1/2, 1/4... "
            "Anti-entropy mode = QA orbit correction: nodes reconcile differences, "
            "damping the satellite oscillation toward consistent state."
        ),
        "orbit_seed": [2, 3],
        "levin_cell_type": "progenitor",
        "organ_roles": ["state_propagator", "eventual_consistency_agent", "epidemic_spreader"],
        "failure_modes": [
            "Permanent partition: disconnected component never receives updates",
            "Churn: rapid node join/leave disrupts convergence",
            "False rumor: incorrect state propagates — no built-in correction",
            "Bandwidth: O(N log N) total messages per round for full state",
        ],
        "composition_rules": [
            "adaptive:CRDT (conflict-free replicated data type for gossip state)",
            "hierarchical:hierarchical_gossip (tree structure reduces rounds)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Convergence time > 5x expected O(log N) rounds"],
            "recommit_conditions": ["Connected graph, bounded churn rate"],
            "max_satellite_cycles": 20, "drift_threshold": 0.08, "partial_fail_threshold": 10,
        },
        "source_corpus_refs": [QA4, QA3],
        "corpus_concepts": ["wave_packet_propagation", "harmonic_decay_convergence", "epidemic_orbit", "anti_entropy_damping"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
    },

    {
        "name": "chord_dht", "family": "distributed",
        "goal": "Distributed hash table with O(log N) lookup via consistent hashing on a ring",
        "cognitive_horizon": "bounded", "convergence": "guaranteed",
        "time_complexity": "O(log N) hops per lookup", "space_complexity": "O(log N) routing table",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Chord routes queries around a ring in O(log N) hops — a bounded cosmos orbit. "
            "Each hop halves the remaining arc distance (like binary search on a ring) — "
            "the same QA cosmos halving property. The finger table = QA orbit accelerator: "
            "instead of linear scan (satellite), it jumps in powers of 2 (cosmos). "
            "QA-2 prime number structure: node IDs are distributed via SHA-1 modulo 2^m — "
            "the ring modulus maps to QA arithmetic mod m. "
            "Node join/leave = QA orbit perturbation: requires finger table update "
            "but ring structure (cosmos orbit) is preserved."
        ),
        "orbit_seed": [1, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["distributed_lookup", "ring_navigator", "consistent_hasher"],
        "failure_modes": [
            "Churn: rapid join/leave invalidates finger tables",
            "Routing loop: inconsistent tables during stabilization",
            "Hot spots: popular keys overload successor nodes",
            "Long chains: O(log N) hops may be slow for latency-sensitive apps",
        ],
        "composition_rules": [
            "hierarchical:Kademlia (XOR metric DHT, better locality)",
            "sequential:replication (store k copies for fault tolerance)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Finger tables stale (churn rate exceeds stabilization)"],
            "recommit_conditions": ["Stable membership, finger tables converged"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 3,
        },
        "source_corpus_refs": [QA2, QA1],
        "corpus_concepts": ["ring_modulus_arithmetic", "binary_halving_cosmos", "finger_table_orbit_accelerator", "prime_id_distribution"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
    },

    {
        "name": "byzantine_fault_tolerance", "family": "distributed",
        "goal": "Reach consensus despite up to f Byzantine (arbitrary/malicious) faulty nodes; N ≥ 3f+1",
        "cognitive_horizon": "global", "convergence": "conditional",
        "time_complexity": "O(N²) messages per consensus round", "space_complexity": "O(N²)",
        "orbit_signature": "mixed",
        "orbit_rationale": (
            "With f < N/3 Byzantine nodes: cosmos orbit — PBFT guarantees safety and liveness. "
            "With f ≥ N/3: satellite/singularity — no algorithm can guarantee consensus. "
            "The 3f+1 threshold is a QA obstruction certificate: it is the minimum N such "
            "that the honest majority can always outvote the adversary. "
            "QA-1 Law 1 ('every quantum number contains not less than three prime numbers'): "
            "the 3f+1 requirement mirrors the minimum three-prime-factor rule for quantum numbers. "
            "BFT view change = QA dedifferentiation: the protocol resets to singularity "
            "(new view) when the current view is stuck (satellite timeout)."
        ),
        "orbit_seed": [4, 6],
        "levin_cell_type": "progenitor",
        "organ_roles": ["byzantine_tolerant_consensus", "adversary_resistant_organ", "safety_guarantor"],
        "failure_modes": [
            "f ≥ N/3: safety violation possible (undecidable)",
            "O(N²) messages: does not scale beyond ~100 nodes",
            "View change storms: repeated timeouts → satellite cycling",
            "Sybil attack: adversary creates many fake identities to exceed f",
        ],
        "composition_rules": [
            "hierarchical:HotStuff (linear BFT — O(N) messages via chaining)",
            "adaptive:Tendermint (BFT + PoS for blockchain)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["f ≥ N/3 suspected (safety threshold breached)", "View change storm > 5 consecutive views"],
            "recommit_conditions": ["N ≥ 3f+1 verified, network synchrony assumed"],
            "max_satellite_cycles": 5, "drift_threshold": 0.15, "partial_fail_threshold": 5,
        },
        "source_corpus_refs": [QA1, QA2],
        "corpus_concepts": ["law_1_three_prime_minimum", "3f_plus_1_obstruction_cert", "view_change_dedifferentiation", "adversary_orbit_injection"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
    },
]


# ── Merge into registry ───────────────────────────────────────────────────────

def merge(dry_run: bool = False) -> None:
    if not REG_PATH.exists():
        print(f"ERROR: {REG_PATH} not found — run qa_algorithm_competency_expand.py first")
        sys.exit(1)

    reg = json.loads(REG_PATH.read_text())
    existing_names = {a["name"] for a in reg["algorithms"]}

    added = 0
    for entry in BATCH2:
        if entry["name"] in existing_names:
            print(f"  skip (exists): {entry['name']}")
            continue

        b, e = entry["orbit_seed"]
        entry["simulated_orbit"] = qa_orbit_family(b, e)
        entry["simulated_ofr"]   = ofr(b, e)
        entry["orbit_seed"]      = list(entry["orbit_seed"])

        if not dry_run:
            reg["algorithms"].append(entry)
        print(f"  {'[dry]' if dry_run else '+'} {entry['name']:30s}  {entry['orbit_signature']:10s}  {entry['family']}")
        added += 1

    if not dry_run:
        reg["total_algorithms"] = len(reg["algorithms"])
        # Rebuild summary fields
        for fam in ("cosmos", "satellite", "mixed", "singularity"):
            reg["orbit_distribution"][fam] = [
                a["name"] for a in reg["algorithms"] if a.get("orbit_signature") == fam
            ]
        for ct in ("differentiated", "progenitor", "stem"):
            reg["cell_type_distribution"][ct] = [
                a["name"] for a in reg["algorithms"] if a.get("levin_cell_type") == ct
            ]
        reg["families"] = sorted(set(a["family"] for a in reg["algorithms"]))
        REG_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        print(f"\nMerged {added} algorithms → {REG_PATH}  (total: {reg['total_algorithms']})")
    else:
        print(f"\n[dry-run] Would add {added} algorithms (total would be: {len(existing_names)+added})")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    merge(args.dry_run)


if __name__ == "__main__":
    main()
