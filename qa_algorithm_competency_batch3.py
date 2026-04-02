#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_algorithm_competency_batch3.py
====================================
Third batch: 14 algorithms bringing total to 50.
New family: time_series (ARIMA, Holt-Winters, Sornette log-periodic, Kalman smoother)
Fills out: sort, search, graph, optimize, learn

Usage:
  python qa_algorithm_competency_batch3.py
  python qa_algorithm_competency_batch3.py --dry-run
"""

from __future__ import annotations
import json, sys, argparse
from pathlib import Path

MODULUS = 9
REG_PATH = Path("qa_algorithm_competency_registry.json")

QA1  = "qa-1__qa_1_all_pages__docx.md"
QA2  = "qa-2__001_qa_2_all_pages__docx.md"
QA3  = "qa_3__ocr__qa3.md"
QA4  = "qa-4__00_qa_books_3_&_4_all_pages__pdf.md"
QUAD = "quadrature__00_quadratureprint__pdf.md"
P1   = "pyth_1__ocr__pyth1.md"
P3   = "pyth-3__pythagoras_vol3_enneagram__docx.md"
# P2 = pyth_2__ocr__pyth2.md — pending, flagged with needs_ocr_backfill=True

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

BATCH3 = [

    # ── sort (2) ──────────────────────────────────────────────────────────────

    {
        "name": "radix_sort", "family": "sort",
        "goal": "Sort integers by processing digits from least to most significant; O(Nk) guaranteed",
        "cognitive_horizon": "bounded", "convergence": "guaranteed",
        "time_complexity": "O(N·k) k=digits/key_length", "space_complexity": "O(N+b) b=base",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Radix sort processes k digit positions in sequence — exactly k cosmos passes, "
            "each guaranteed to produce a stable partial ordering. The digit decomposition "
            "maps directly to QA-2's prime factorization: each digit position is one prime "
            "factor level in the quantum number. LSD radix sort: cosmos orbit at each level, "
            "orbit chain length = k. QA-1 Law 5 (divisibility by powers of base): "
            "radix sort is the algorithmic implementation of this law — checking divisibility "
            "at each digit = extracting the QA root at each factorization level. "
            "The base b = QA modulus choice: b=10 (decimal), b=2 (binary), b=24 (QA natural)."
        ),
        "orbit_seed": [1, 4],
        "levin_cell_type": "differentiated",
        "organ_roles": ["integer_sorter", "digit_decomposer", "modular_sort_spine"],
        "failure_modes": [
            "Floating point: digit decomposition undefined for non-integers",
            "Variable-length keys: padding required, wastes passes",
            "Large k (many digits): O(Nk) degrades to O(N²) when k ~ N",
            "Memory: O(N) auxiliary per pass for counting sort buckets",
        ],
        "composition_rules": [
            "sequential:counting_sort (radix uses counting_sort as subroutine per digit)",
            "hybrid:hybrid_radix (radix for large N, insertion_sort for small buckets)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Non-integer keys (strings need different radix)", "k > log2(N) (insertion_sort faster)"],
            "recommit_conditions": ["Integer keys, bounded digit count, N > 10^4"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA1, QA2],
        "corpus_concepts": ["prime_factorization_digit_levels", "law_5_divisibility_base", "modular_sort", "k_pass_cosmos_chain"],
        "needs_ocr_backfill": True,
        "confidence": "high",
    },

    {
        "name": "counting_sort", "family": "sort",
        "goal": "Sort integers in range [0,k] by counting occurrences; O(N+k) guaranteed",
        "cognitive_horizon": "bounded", "convergence": "guaranteed",
        "time_complexity": "O(N+k)", "space_complexity": "O(k)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Counting sort makes exactly three linear passes: count, prefix-sum, place. "
            "Three passes = the QA triple of (b,e,d): count=b, prefix=e, place=d. "
            "Each pass is a deterministic cosmos orbit over N or k elements. "
            "The key range k maps to the QA modulus: this is literally modular arithmetic "
            "sort — keys live in Z_k. When k = 9 or 24, counting sort is the natural "
            "QA sorter. QA-1 parity laws: the prefix-sum step computes cumulative "
            "parity distribution across the key range."
        ),
        "orbit_seed": [2, 7],
        "levin_cell_type": "differentiated",
        "organ_roles": ["integer_range_sorter", "frequency_counter", "radix_subroutine"],
        "failure_modes": [
            "Large k: O(k) space dominates when k >> N",
            "Non-integer keys: inapplicable",
            "Negative keys: offset required",
        ],
        "composition_rules": [
            "sequential:radix_sort (counting_sort as subroutine)",
            "sequential:bucket_sort (counting_sort for integer buckets)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["k > 10*N (space cost too high)"],
            "recommit_conditions": ["Integer keys in bounded range k ≤ 10*N"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA1, QA2],
        "corpus_concepts": ["z_k_modular_sort", "three_pass_b_e_d", "parity_prefix_sum", "qa_natural_sorter"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    # ── search (2) ────────────────────────────────────────────────────────────

    {
        "name": "interpolation_search", "family": "search",
        "goal": "Search sorted array by estimating target position via linear interpolation",
        "cognitive_horizon": "adaptive", "convergence": "conditional",
        "time_complexity": "O(log log N) expected uniform, O(N) worst", "space_complexity": "O(1)",
        "orbit_signature": "mixed",
        "orbit_rationale": (
            "Uniformly distributed keys: cosmos orbit — interpolation probe lands near target "
            "in O(log log N) steps, better than binary search's O(log N). "
            "Non-uniform / clustered keys: satellite orbit — probe repeatedly overshoots, "
            "degrades to O(N) linear scan. The interpolation formula is a QA ratio estimate: "
            "pos = lo + (target-arr[lo])/(arr[hi]-arr[lo]) * (hi-lo) maps the value range "
            "to an index range, exactly the QA 'quantizing' operation of Chapter 2 in QA-3. "
            "Worst case (adversarial distribution) = satellite: the orbit estimate is wrong "
            "and the search oscillates."
        ),
        "orbit_seed": [3, 7],
        "levin_cell_type": "progenitor",
        "organ_roles": ["distribution_aware_searcher", "quantize_estimator"],
        "failure_modes": [
            "Non-uniform distribution: O(N) worst case",
            "Integer arithmetic overflow in probe calculation",
            "Adversarial input: attacker crafts distribution to maximise probes",
        ],
        "composition_rules": [
            "adaptive:binary_search (fallback when distribution unknown)",
            "adaptive:exponential_search (combine with interpolation for unbounded arrays)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Non-uniform distribution detected (probe error > 2x expected)", "Overflow risk in probe calc"],
            "recommit_conditions": ["Uniform distribution confirmed by sampling"],
            "max_satellite_cycles": 4, "drift_threshold": 0.15, "partial_fail_threshold": 4,
        },
        "source_corpus_refs": [QA3, QA2],
        "corpus_concepts": ["quantize_ratio_probe", "value_to_index_mapping", "distribution_orbit_selector"],
        "needs_ocr_backfill": True,
        "confidence": "medium",
    },

    {
        "name": "jump_search", "family": "search",
        "goal": "Search sorted array by jumping √N steps then linear scan in found block",
        "cognitive_horizon": "bounded", "convergence": "guaranteed",
        "time_complexity": "O(√N)", "space_complexity": "O(1)",
        "orbit_signature": "satellite",
        "orbit_rationale": (
            "Jump search makes √N jumps then up to √N linear steps — a two-phase satellite "
            "orbit: coarse jump phase + fine scan phase. Each phase is bounded (√N steps) "
            "but the algorithm loops through the array in fixed increments without adaptation. "
            "QA-1 Law 12(a): '2·d·e = base of Pythagorean triangle' — the jump size √N "
            "is the geometric mean of 1 and N, the QA intermediate root between extremes. "
            "Unlike binary search (cosmos halving), jump search uses fixed arithmetic steps "
            "matching the satellite fixed-period structure."
        ),
        "orbit_seed": [2, 5],
        "levin_cell_type": "progenitor",
        "organ_roles": ["block_scanner", "sqrt_N_navigator"],
        "failure_modes": [
            "Requires sorted array",
            "Suboptimal vs binary search: O(√N) > O(log N) for large N",
            "Fixed jump size: no adaptation to data distribution",
        ],
        "composition_rules": [
            "sequential:binary_search (upgrade for better complexity)",
            "sequential:linear_search (jump_search uses linear_search within block)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["N > 10^6 (binary search strictly better)", "Unsorted array"],
            "recommit_conditions": ["Sorted array, N moderate (10^3–10^5), backward jump prohibited"],
            "max_satellite_cycles": 6, "drift_threshold": 0.10, "partial_fail_threshold": 5,
        },
        "source_corpus_refs": [QA1],
        "corpus_concepts": ["sqrt_N_intermediate_root", "law_12_geometric_mean", "fixed_step_satellite"],
        "needs_ocr_backfill": False,
        "confidence": "medium",
    },

    # ── graph (2) ─────────────────────────────────────────────────────────────

    {
        "name": "tarjan_scc", "family": "graph",
        "goal": "Find all strongly connected components of directed graph in O(V+E)",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(V+E)", "space_complexity": "O(V)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Tarjan's algorithm is DFS augmented with a low-link value tracker — "
            "a cosmos orbit that collapses each SCC into a single node on the stack. "
            "The discovery time and low-link values form a QA (b,e,d,a) tuple per node: "
            "disc=b, low=e, their difference=d tracks the SCC boundary. "
            "Stack-based SCC extraction = QA orbit closure: when low[v]==disc[v], "
            "the stack segment above v is a completed cosmos orbit (the SCC). "
            "QA-1 Law 7 (roots are unique): each SCC's node set is unique — "
            "no node appears in two SCCs, matching QA's unique root decomposition."
        ),
        "orbit_seed": [1, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["scc_finder", "cycle_decomposer", "graph_condensation_oracle"],
        "failure_modes": [
            "Undirected graphs: use union-find instead (SCC undefined)",
            "Stack overflow: very deep DFS on large graphs",
            "Multiple valid orderings: SCC condensation is unique but topological order is not",
        ],
        "composition_rules": [
            "sequential:topological_sort (condensation DAG is topologically sorted)",
            "sequential:2_SAT (SCC-based 2-SAT solver)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Graph undirected (wrong algorithm)", "Stack overflow on deep DFS"],
            "recommit_conditions": ["Directed graph confirmed"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA1, QA2],
        "corpus_concepts": ["disc_low_b_e_d_tuple", "scc_as_cosmos_closure", "law_7_unique_root_decomposition", "stack_orbit_extraction"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    {
        "name": "kruskal_mst", "family": "graph",
        "goal": "Find minimum spanning tree by greedily adding cheapest edge that doesn't form cycle",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(E log E)", "space_complexity": "O(V)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Kruskal sorts all edges then uses union-find to greedily build the MST — "
            "a two-phase cosmos orbit. Sort phase: cosmos (merge_sort). "
            "Union-find phase: near-O(E·α(V)) — effectively cosmos. "
            "The MST is unique (for distinct edge weights) — QA Law 9 uniqueness of roots. "
            "QA-4 Wave Theory: MST is the minimal spanning harmonic — the tree that "
            "connects all nodes at minimum total 'harmonic cost'. Cycle detection = "
            "obstruction certificate: adding a cycle creates a satellite (redundant path). "
            "Pyth-1 compatible pairs: MST edges are the compatible pairs of the graph — "
            "minimal relationships that span the entire structure."
        ),
        "orbit_seed": [1, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["mst_builder", "spanning_tree_oracle", "network_backbone_finder"],
        "failure_modes": [
            "Equal edge weights: MST not unique (multiple valid trees)",
            "Dense graphs: Prim's O(V²) may be faster than Kruskal's O(E log E)",
            "Dynamic edges: full recompute needed on edge change",
        ],
        "composition_rules": [
            "sequential:union_find (Kruskal uses union-find for cycle detection)",
            "sequential:prim_mst (alternative MST algorithm, better for dense graphs)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Dense graph where Prim's preferred", "Edge weights not distinct"],
            "recommit_conditions": ["Sparse graph, distinct edge weights"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA4, P1],
        "corpus_concepts": ["minimal_spanning_harmonic", "compatible_pair_edges", "law_9_unique_mst", "cycle_obstruction_cert"],
        "needs_ocr_backfill": True,
        "confidence": "high",
    },

    # ── optimize (2) ──────────────────────────────────────────────────────────

    {
        "name": "pso", "family": "optimize",
        "goal": "Swarm optimization: particles explore search space guided by personal + global best",
        "cognitive_horizon": "adaptive", "convergence": "probabilistic",
        "time_complexity": "O(I·P·D) I=iterations, P=swarm_size, D=dim", "space_complexity": "O(P·D)",
        "orbit_signature": "satellite",
        "orbit_rationale": (
            "Each particle moves via: v = w·v + c1·(pbest-x) + c2·(gbest-x). "
            "The inertia w controls orbit damping: w>1 = satellite (exploration), w<1 = cosmos decay. "
            "The swarm collectively orbits around the global best — a satellite pattern "
            "where particles cycle around the attractor without guaranteed convergence. "
            "QA-4 Synchronous Harmonics: the swarm is a set of N harmonic oscillators — "
            "each particle is a wave with its own phase, all attracted to the global resonance (gbest). "
            "Cognitive component (c1·pbest) = individual orbit memory; "
            "social component (c2·gbest) = collective orbit synchronization."
        ),
        "orbit_seed": [3, 4],
        "levin_cell_type": "progenitor",
        "organ_roles": ["swarm_explorer", "global_attractor_seeker", "collective_optimizer"],
        "failure_modes": [
            "Premature convergence: swarm collapses to local optimum",
            "Velocity explosion: w too large → particles escape search space",
            "Stagnation: all particles at gbest → no exploration (singularity)",
            "Discontinuous landscapes: velocity model breaks",
        ],
        "composition_rules": [
            "adaptive:SPSO (standard PSO with constriction factor for convergence guarantee)",
            "hierarchical:multi_swarm_PSO (multiple swarms with migration)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Swarm diversity < 5% of search space (premature convergence)", "Velocity norm > search_space_diameter"],
            "recommit_conditions": ["w tuned to 0.4–0.9, swarm re-diversified"],
            "max_satellite_cycles": 10, "drift_threshold": 0.12, "partial_fail_threshold": 8,
        },
        "source_corpus_refs": [QA4, QA3],
        "corpus_concepts": ["harmonic_oscillator_swarm", "gbest_as_fundamental_frequency", "inertia_as_orbit_damper", "synchronous_swarm_orbit"],
        "needs_ocr_backfill": True,
        "confidence": "medium",
    },

    {
        "name": "lbfgs", "family": "optimize",
        "goal": "Quasi-Newton optimization using limited-memory approximation of inverse Hessian",
        "cognitive_horizon": "adaptive", "convergence": "conditional",
        "time_complexity": "O(m·D·T) m=memory_steps, D=params, T=iters", "space_complexity": "O(m·D)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "L-BFGS maintains a rank-2m approximation of the inverse Hessian via the last m "
            "gradient differences — a bounded cosmos orbit that uses QA-1's Fibonacci-like "
            "recursion: each H_k update uses only the immediately preceding m steps (bounded memory). "
            "For convex problems: cosmos guaranteed — superlinear convergence. "
            "For non-convex: mixed — Hessian approximation may be wrong, but line search "
            "ensures sufficient decrease (Wolf conditions = orbit descent certificate). "
            "QA orbit connection: the L-BFGS two-loop recursion for computing H·g has "
            "exactly 2m steps — matching the satellite (8) and cosmos (24) orbit periods "
            "when m=4 or m=12 respectively."
        ),
        "orbit_seed": [2, 6],
        "levin_cell_type": "differentiated",
        "organ_roles": ["quasi_newton_optimizer", "bounded_memory_hessian", "superlinear_convergence_spine"],
        "failure_modes": [
            "Non-convex: Hessian approximation may be indefinite",
            "Very noisy gradients: curvature estimate unreliable",
            "Ill-conditioned: large condition number slows convergence",
            "Batch size: stochastic gradients corrupt Hessian approximation (use Adam instead)",
        ],
        "composition_rules": [
            "sequential:gradient_descent (L-BFGS as accelerated GD)",
            "sequential:line_search (Wolfe conditions guard each step)",
            "hierarchical:natural_gradient (information-geometry generalization)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Stochastic gradient (noise corrupts Hessian)", "Hessian becomes indefinite"],
            "recommit_conditions": ["Full-batch or low-noise gradients, convex or mildly non-convex loss"],
            "max_satellite_cycles": 1, "drift_threshold": 0.08, "partial_fail_threshold": 3,
        },
        "source_corpus_refs": [QA1, QA4],
        "corpus_concepts": ["fibonacci_bounded_memory", "two_loop_2m_recursion", "wolfe_orbit_descent_cert", "superlinear_cosmos"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },

    # ── learn (3) ─────────────────────────────────────────────────────────────

    {
        "name": "linear_regression", "family": "learn",
        "goal": "Fit y = Xβ + ε via closed-form OLS: β = (X'X)^{-1}X'y",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(ND²+D³) N=samples, D=features", "space_complexity": "O(D²)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "OLS has a unique closed-form solution when X'X is invertible — a single-step "
            "cosmos orbit. No iteration needed; the solution is the unique minimum of a "
            "convex quadratic. QA-1: the normal equations X'Xβ = X'y are the QA parity "
            "equations written in matrix form — each row is a quantum number constraint. "
            "Pyth-1 triangle geometry: OLS regression is the projection of y onto the "
            "column space of X — geometrically identical to finding the altitude of a "
            "Pythagorean triangle (Law 12c). The residual ε is the perpendicular orbit "
            "component that cannot be explained by the model."
        ),
        "orbit_seed": [1, 1],
        "levin_cell_type": "differentiated",
        "organ_roles": ["linear_model_spine", "projection_oracle", "baseline_predictor"],
        "failure_modes": [
            "Multicollinearity: X'X singular or near-singular → no unique solution",
            "N < D: underdetermined system (infinite solutions)",
            "Nonlinear relationship: model misspecified",
            "Outliers: OLS is non-robust (use LAD or Huber loss instead)",
        ],
        "composition_rules": [
            "sequential:gradient_descent (iterative alternative when N too large for matrix inversion)",
            "adaptive:ridge_regression (add L2 regularization to handle multicollinearity)",
            "hierarchical:GLM (linear regression with link function for non-Gaussian targets)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["X'X singular (multicollinearity)", "N < D (underdetermined)"],
            "recommit_conditions": ["Full rank X'X confirmed, N >> D"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA1, P1],
        "corpus_concepts": ["normal_equations_as_parity", "projection_as_pythagorean_altitude", "law_12c_residual", "closed_form_cosmos"],
        "needs_ocr_backfill": True,
        "confidence": "high",
    },

    {
        "name": "em_gmm", "family": "learn",
        "goal": "Fit Gaussian mixture model via Expectation-Maximization; converges to local MLE",
        "cognitive_horizon": "adaptive", "convergence": "conditional",
        "time_complexity": "O(T·N·K·D²) T=iters, K=components", "space_complexity": "O(N·K)",
        "orbit_signature": "mixed",
        "orbit_rationale": (
            "EM alternates E-step (soft assignment) and M-step (parameter update) — "
            "a two-phase orbit: E is the satellite pass (assigns responsibilities), "
            "M is the cosmos step (re-centers Gaussians toward maximum likelihood). "
            "The log-likelihood is monotonically non-decreasing per EM iteration: "
            "guaranteed not to diverge but may converge to saddle/local max (mixed orbit). "
            "QA-3 Myriads: each Gaussian component is one myriad — a classification bucket "
            "operating in its own frequency range. EM finds the K myriad centers "
            "that best explain the data. "
            "Pyth-3 Enneagram: K=9 GMM components map to the 9 Enneagram types — "
            "personality clustering as QA mod-9 arithmetic."
        ),
        "orbit_seed": [4, 5],
        "levin_cell_type": "progenitor",
        "organ_roles": ["soft_cluster_modeler", "density_estimator", "latent_variable_learner"],
        "failure_modes": [
            "Local maxima: initialization-dependent",
            "Degenerate components: Gaussian collapses to single point (singularity)",
            "Wrong K: over/underfitting of mixture",
            "Slow convergence: many iterations needed near saddle points",
        ],
        "composition_rules": [
            "sequential:k_means (k_means++ init for EM starting point)",
            "adaptive:variational_EM (Bayesian EM avoids degenerate components)",
            "hierarchical:hierarchical_EM (nested mixture models)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Degenerate component detected (covariance singular)", "Likelihood not increasing for 5 steps"],
            "recommit_conditions": ["K validated, initialization re-run with k-means++"],
            "max_satellite_cycles": 4, "drift_threshold": 0.15, "partial_fail_threshold": 5,
        },
        "source_corpus_refs": [QA3, P3],
        "corpus_concepts": ["myriad_gaussian_component", "enneagram_mod9_clustering", "em_two_phase_orbit", "log_likelihood_monotone"],
        "needs_ocr_backfill": True,
        "confidence": "medium",
    },

    {
        "name": "dbscan", "family": "learn",
        "goal": "Density-based clustering: finds arbitrary-shape clusters via ε-neighborhood reachability",
        "cognitive_horizon": "bounded", "convergence": "guaranteed",
        "time_complexity": "O(N log N) with spatial index, O(N²) naive", "space_complexity": "O(N)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "DBSCAN processes each point exactly once, expanding dense regions exhaustively — "
            "a cosmos orbit with no revisits. The ε-neighborhood is the QA orbit radius: "
            "points within ε are orbit-reachable. MinPts is the QA quantum number threshold: "
            "a core point requires ≥ MinPts orbit-neighbors. "
            "QA-3 prime clusters: dense clusters are the 'prime regions' of the data space — "
            "regions where quantum numbers concentrate. Noise points are non-quantum: "
            "they don't belong to any dense orbit. "
            "Unlike k-means (satellite cycling), DBSCAN makes one deterministic pass: "
            "true cosmos orbit with no iteration."
        ),
        "orbit_seed": [1, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["density_explorer", "arbitrary_shape_clusterer", "noise_detector"],
        "failure_modes": [
            "Varying density: single ε fails (use HDBSCAN)",
            "High dimensions: ε-neighborhood becomes meaningless (curse of dimensionality)",
            "Border point assignment: non-deterministic when point is on border of two clusters",
            "Memory: O(N²) distance matrix without spatial index",
        ],
        "composition_rules": [
            "sequential:PCA (dimensionality reduction before DBSCAN)",
            "adaptive:HDBSCAN (hierarchical DBSCAN for varying density)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Varying density (use HDBSCAN)", "Dimensionality > 20"],
            "recommit_conditions": ["Low-dim data, density roughly uniform, ε validated on sample"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA3, QA1],
        "corpus_concepts": ["epsilon_orbit_radius", "minpts_quantum_threshold", "prime_density_region", "noise_as_non_quantum"],
        "needs_ocr_backfill": True,
        "confidence": "high",
    },

    # ── time_series (NEW family, 4 algorithms) ────────────────────────────────

    {
        "name": "arima", "family": "time_series",
        "goal": "Model and forecast time series via autoregression + differencing + moving average",
        "cognitive_horizon": "bounded", "convergence": "conditional",
        "time_complexity": "O(T·(p+q)²) T=series length", "space_complexity": "O(p+q)",
        "orbit_signature": "mixed",
        "orbit_rationale": (
            "ARIMA(p,d,q): AR component (p lags) = bounded cosmos orbit over recent history. "
            "MA component (q lags) = satellite smoothing: weighted average of past errors. "
            "Differencing (d) = orbit stabilizer: removes trend/non-stationarity → satellite→cosmos. "
            "QA-4 Synchronous Harmonics: ARIMA decomposes the time series into its "
            "harmonic components — AR extracts periodic patterns (harmonics), "
            "MA dampens residual noise. The ARIMA characteristic polynomial roots "
            "determine orbit stability: roots inside unit circle = cosmos; outside = divergence. "
            "QA-1 Fibonacci: ARIMA(p=2, d=0, q=0) with coefficients φ₁=1,φ₂=1 is "
            "exactly the Fibonacci recurrence — the QA cosmos orbit."
        ),
        "orbit_seed": [3, 5],
        "levin_cell_type": "progenitor",
        "organ_roles": ["time_series_modeler", "forecast_generator", "trend_decomposer"],
        "failure_modes": [
            "Non-stationary without sufficient differencing: divergent orbit",
            "Wrong p,q order: overfitting or underfitting",
            "Structural breaks: model assumes stationary after differencing",
            "Long-range dependence: ARIMA cannot capture (use ARFIMA)",
        ],
        "composition_rules": [
            "sequential:adf_test (stationarity test to determine d)",
            "adaptive:SARIMA (seasonal extension for periodic series)",
            "hierarchical:ARIMAX (ARIMA + exogenous variables)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Characteristic roots outside unit circle (divergent)", "AIC/BIC not improving with order increase"],
            "recommit_conditions": ["Stationarity confirmed (ADF test), residuals white noise"],
            "max_satellite_cycles": 3, "drift_threshold": 0.15, "partial_fail_threshold": 5,
        },
        "source_corpus_refs": [QA4, QA1],
        "corpus_concepts": ["harmonic_decomposition_AR", "fibonacci_ar2_recurrence", "differencing_orbit_stabilizer", "unit_circle_cosmos_boundary"],
        "needs_ocr_backfill": True,
        "confidence": "medium",
    },

    {
        "name": "holt_winters", "family": "time_series",
        "goal": "Exponential smoothing with trend + seasonal components for forecasting",
        "cognitive_horizon": "adaptive", "convergence": "guaranteed",
        "time_complexity": "O(T) per update", "space_complexity": "O(s) s=season_length",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Holt-Winters updates three exponentially weighted states: level (α), trend (β), "
            "seasonal (γ) — mapping directly to QA's (b,e,d,a) tuple: "
            "level=b, trend=e, seasonal=d, forecast=a. "
            "The exponential weights (α,β,γ ∈ (0,1)) are orbit dampers: "
            "they ensure each component decays toward zero if not updated → stable cosmos. "
            "QA-4 Wave Theory: the seasonal component is the fundamental harmonic frequency; "
            "the trend is the carrier wave; the level is the DC offset. "
            "Together they form the full QA wave decomposition: "
            "level (singularity seed) + trend (satellite growth) + seasonal (cosmos period)."
        ),
        "orbit_seed": [1, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["seasonal_forecaster", "exponential_smoother", "trend_seasonal_decomposer"],
        "failure_modes": [
            "Wrong season length s: seasonal component misaligned",
            "High α: too reactive to noise (satellite oscillation)",
            "Low β: trend doesn't adapt to structural change",
            "Multiplicative model instability when level near zero",
        ],
        "composition_rules": [
            "sequential:STL (seasonal decomposition before HW for complex seasonality)",
            "adaptive:ETS (state space formulation of Holt-Winters)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Season length unknown or changing", "Level crosses zero with multiplicative model"],
            "recommit_conditions": ["Stable seasonality confirmed, α/β/γ tuned via cross-validation"],
            "max_satellite_cycles": 0, "drift_threshold": 0.08, "partial_fail_threshold": 3,
        },
        "source_corpus_refs": [QA4, QA1],
        "corpus_concepts": ["b_e_d_a_level_trend_seasonal", "exponential_orbit_damper", "wave_dc_carrier_harmonic", "cosmos_periodic_forecast"],
        "needs_ocr_backfill": True,
        "confidence": "high",
    },

    {
        "name": "sornette_log_periodic", "family": "time_series",
        "goal": "Detect log-periodic power law (LPPL) signatures preceding critical transitions (crashes/singularities)",
        "cognitive_horizon": "global", "convergence": "conditional",
        "time_complexity": "O(N·F) N=series length, F=fitting iterations", "space_complexity": "O(N)",
        "orbit_signature": "mixed",
        "orbit_rationale": (
            "LPPL model: p(t) = A + B(t_c-t)^m · (1 + C·cos(ω·log(t_c-t) + φ)). "
            "The log-periodic term cos(ω·log(t_c-t)) is a QA satellite orbit: "
            "it cycles in log-time with angular frequency ω, accelerating as t→t_c. "
            "The power law (t_c-t)^m is the orbit descent toward singularity at t_c. "
            "t_c is the QA singularity point: the state (0,0) that cannot be escaped. "
            "QA-3 Quantizing: LPPL fitting quantizes the price series to find the "
            "critical time t_c — the quantum number of the crash. "
            "High ω = tight satellite orbit (many oscillations before crash). "
            "Successful fit (low residual) = cosmos confirmation: the system is on the "
            "predicted orbit toward singularity. Failed fit = mixed (no singularity detected)."
        ),
        "orbit_seed": [0, 0],   # singularity seed — t_c is the singularity
        "levin_cell_type": "progenitor",
        "organ_roles": ["crash_predictor", "singularity_detector", "log_periodic_fitter", "regime_change_oracle"],
        "failure_modes": [
            "False positive: log-periodic pattern fit but no crash (overfitting)",
            "t_c estimation error: small changes in fit → large t_c shift",
            "Non-unique fit: multiple local minima in LPPL parameter space",
            "Post-crash: LPPL valid only pre-crash, not during recovery",
            "Regime change: market structure change invalidates fit",
        ],
        "composition_rules": [
            "sequential:arima (ARIMA residuals → LPPL fit on detrended series)",
            "sequential:kalman_filter (state-space LPPL for online detection)",
            "adaptive:ensemble_LPPL (multiple fits, vote on t_c)",
            "hierarchical:multi_scale_LPPL (fit at multiple time scales simultaneously)",
        ],
        "differentiation_profile": {
            "dediff_conditions": [
                "Fit residual > 3x baseline noise (no LPPL pattern)",
                "t_c in the past (crash already occurred)",
                "ω < 4 or ω > 25 (outside Sornette's empirical bounds)",
            ],
            "recommit_conditions": [
                "Residual < baseline, m ∈ (0.1, 0.9), ω ∈ (4, 25), t_c in credible future window",
                "Ensemble of fits agrees on t_c within ±10% window",
            ],
            "max_satellite_cycles": 3, "drift_threshold": 0.20, "partial_fail_threshold": 4,
        },
        "source_corpus_refs": [QA3, QA4],
        "corpus_concepts": [
            "log_periodic_satellite_orbit", "singularity_tc_crash_point", "lppl_quantize_critical_time",
            "orbit_descent_power_law", "omega_satellite_frequency", "qa_singularity_as_crash",
        ],
        "needs_ocr_backfill": True,  # Pyth-2 may have wave/cycle vocabulary
        "confidence": "medium",
        "qa_research_note": (
            "This is the Sornette singularity exit test from ChatGPT's priority list. "
            "Maps directly to QA Lab: VIX/price enters satellite orbit (log-periodic oscillation) "
            "→ orbit descent toward singularity (crash) → dedifferentiation event → "
            "new regime. The SelfImprovementAgent should monitor for LPPL signatures "
            "in financial time series as a real-world orbit singularity detector."
        ),
    },

    {
        "name": "kalman_smoother", "family": "time_series",
        "goal": "Optimal smoothed state estimate using all observations (forward + backward Kalman pass)",
        "cognitive_horizon": "global", "convergence": "guaranteed",
        "time_complexity": "O(T·D³) T=series length, D=state dim", "space_complexity": "O(T·D²)",
        "orbit_signature": "cosmos",
        "orbit_rationale": (
            "Kalman smoother runs the Kalman filter forward then backward — a two-pass "
            "cosmos orbit that uses all T observations. The backward pass (RTS smoother) "
            "corrects the forward orbit estimates using future information. "
            "QA-4: the smoother is the full harmonic reconstruction — forward pass extracts "
            "causal harmonics, backward pass extracts anti-causal harmonics, their combination "
            "gives the globally optimal estimate. "
            "QA-1 Law 7 (Fibonacci roots): the smoother gain J_t = P_t·F'·P_{t+1|t}^{-1} "
            "has the same Fibonacci-like recursion as the QA root sequence — "
            "each smoother gain depends on the next gain in the sequence."
        ),
        "orbit_seed": [2, 4],
        "levin_cell_type": "differentiated",
        "organ_roles": ["offline_state_estimator", "full_trajectory_smoother", "retrospective_filter"],
        "failure_modes": [
            "Online use: requires all T observations — not causal (use Kalman filter instead)",
            "Memory: O(T·D²) stores all predicted covariances",
            "Model mismatch: same issues as Kalman filter",
        ],
        "composition_rules": [
            "sequential:kalman_filter (smoother uses filter forward pass)",
            "sequential:EM (EM for unknown model parameters uses Kalman smoother in E-step)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["Online use required (switch to Kalman filter)", "T·D² exceeds memory"],
            "recommit_conditions": ["Offline batch processing, all data available"],
            "max_satellite_cycles": 0, "drift_threshold": 0.05, "partial_fail_threshold": 2,
        },
        "source_corpus_refs": [QA4, QA1],
        "corpus_concepts": ["forward_backward_cosmos_pass", "full_harmonic_reconstruction", "rts_smoother_fibonacci_gain", "global_optimal_orbit_estimate"],
        "needs_ocr_backfill": False,
        "confidence": "high",
    },
]


def merge(dry_run: bool = False) -> None:
    if not REG_PATH.exists():
        print(f"ERROR: {REG_PATH} not found")
        sys.exit(1)

    reg = json.loads(REG_PATH.read_text())
    existing = {a["name"] for a in reg["algorithms"]}

    added = 0
    for entry in BATCH3:
        if entry["name"] in existing:
            print(f"  skip: {entry['name']}")
            continue
        b, e = entry["orbit_seed"]
        entry["simulated_orbit"] = qa_orbit_family(b, e)
        entry["simulated_ofr"]   = ofr(b, e)
        entry["orbit_seed"]      = list(entry["orbit_seed"])
        if not dry_run:
            reg["algorithms"].append(entry)
        print(f"  {'[dry]' if dry_run else '+'} {entry['name']:35s} {entry['orbit_signature']:10s}  {entry['family']}")
        added += 1

    if not dry_run:
        reg["total_algorithms"] = len(reg["algorithms"])
        for fam in ("cosmos","satellite","mixed","singularity"):
            reg["orbit_distribution"][fam] = [a["name"] for a in reg["algorithms"] if a.get("orbit_signature")==fam]
        for ct in ("differentiated","progenitor","stem"):
            reg["cell_type_distribution"][ct] = [a["name"] for a in reg["algorithms"] if a.get("levin_cell_type")==ct]
        reg["families"] = sorted(set(a["family"] for a in reg["algorithms"]))
        REG_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        print(f"\nMerged {added} → {REG_PATH}  (total: {reg['total_algorithms']})")
        print(f"Families: {reg['families']}")
    else:
        print(f"\n[dry-run] Would add {added} (total: {len(existing)+added})")


if __name__ == "__main__":
    p = __import__('argparse').ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    merge(p.parse_args().dry_run)
