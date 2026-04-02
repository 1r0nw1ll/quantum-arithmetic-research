---
generated: 2026-03-27
total_algorithms: 100
total_families: 12
corpus_chars: 2816328
---

# QA Algorithm Competency Registry — Consolidation Summary

## Registry Statistics

| Metric | Value |
|---|---|
| Total algorithms | 100 |
| Families | 12 |
| Corpus chars | 2,816,328 |
| OCR sources | QA-1, QA-2, QA-3 (OCR), QA-4, Quadrature, Pyth-1 (OCR), Pyth-2 (OCR), QA-Workbook (OCR) |

## Family Breakdown

| Family | N | Cosmos | Mixed | Satellite | Differentiated | Progenitor |
|---|---|---|---|---|---|---|
| learn | 18 | 12 | 3 | 3 | 8 | 10 |
| optimize | 11 | 5 | 4 | 2 | 4 | 7 |
| graph | 10 | 9 | 0 | 1 | 9 | 1 |
| number_theory | 10 | 10 | 0 | 0 | 8 | 2 |
| sort | 9 | 6 | 1 | 2 | 6 | 3 |
| search | 8 | 3 | 2 | 3 | 3 | 5 |
| signal_processing | 8 | 8 | 0 | 0 | 6 | 2 |
| planning | 8 | 7 | 0 | 1 | 4 | 4 |
| time_series | 6 | 4 | 2 | 0 | 4 | 2 |
| distributed | 5 | 3 | 1 | 1 | 3 | 2 |
| control | 4 | 2 | 2 | 0 | 2 | 2 |
| geometry | 3 | 3 | 0 | 0 | 3 | 0 |
| **TOTAL** | **100** | **72** | **15** | **13** | **60** | **40** |

## Orbit Distribution

```
cosmos    72 ████████████████████████████████████████████████████████████████████░ 72%
mixed     15 ███████████████░                                                       15%
satellite 13 █████████████░                                                         13%
```

## Cell Type Distribution

```
differentiated 60 ████████████████████████████████████████████████████████████░ 60%
progenitor     40 ████████████████████████████████████████░                      40%
```

## Organ Templates

### Spine Agents (51)
*Differentiated + cosmos + guaranteed convergence — stable, committed, core infrastructure*

- **number_theory**: euclidean_gcd, extended_euclidean, chinese_remainder_theorem, pisano_period, modular_exponentiation, tonelli_shanks, sieve_eratosthenes, continued_fractions
- **graph**: bfs, dfs, dijkstra, floyd_warshall, kruskal_mst, prim_mst, tarjan_scc, topological_sort, bellman_ford
- **signal_processing**: fft_cooley_tukey, hilbert_transform, matched_filter, wiener_filter, autocorrelation, viterbi
- **sort**: merge_sort, heap_sort, radix_sort, counting_sort, timsort, shell_sort
- **learn**: hopfield_network, random_forest, linear_regression, dbscan, svm, belief_propagation
- **planning**: dynamic_programming, ida_star, constraint_propagation, minimax_alphabeta
- **control**: kalman_filter, lqr
- **time_series**: holt_winters, kalman_smoother
- **geometry**: convex_hull, delaunay_triangulation, voronoi_diagram
- **distributed**: map_reduce, raft_consensus, chord_dht
- **search**: fibonacci_search, binary_search
- **other**: fibonacci_search, holt_winters, kalman_smoother, kalman_filter

### Ring / Adaptive Agents (27)
*Progenitor + satellite or mixed — cycling, adaptive, still differentiating*

adaptive_lms, adaptive search variants, backpropagation, bayesian_optimization, branch_and_bound, bubble_sort, byzantine_fault_tolerance, em_gmm, genetic_algorithm, gossip_protocol, gradient_descent, insertion_sort, interpolation_search, jump_search, k_means, linear_search, lstm, mpc, pagerank, pid_controller, policy_gradient_rl, pso, q_learning, quicksort, simulated_annealing, sornette_log_periodic, ucb_bandit

### Stem Differentiation Targets (8)
*Progenitor + satellite + ≥3 organ roles — most tractable to push from progenitor → differentiated*

| Algorithm | Current orbit | Organ roles | Differentiation trigger |
|---|---|---|---|
| q_learning | satellite | value_estimator, policy_improver, exploration_agent | ε → 0, Q-table converged |
| policy_gradient_rl | satellite | policy_optimizer, reward_maximiser, trajectory_sampler | policy entropy → 0 |
| k_means | satellite | cluster_assigner, centroid_updater, convergence_tester | centroid drift < threshold |
| pagerank | satellite | authority_scorer, link_follower, convergence_monitor | rank vector converged |
| pso | satellite | global_searcher, particle_mover, velocity_updater | swarm collapsed to best |
| genetic_algorithm | satellite | population_mutator, crossover_operator, fitness_selector | population converged |
| gossip_protocol | satellite | rumor_spreader, state_synchroniser, epidemic_controller | all nodes consistent |
| branch_and_bound | satellite | combinatorial_optimizer, bound_pruner, subtree_explorer | optimal subtree found |

### Adaptive Mixed-Orbit Agents (15)
*Mixed orbit — behaviour depends on landscape; best candidates for regime-switching organs*

a_star, adam, arima, backpropagation, bayesian_optimization, byzantine_fault_tolerance, em_gmm, gradient_descent, interpolation_search, lstm, mpc, pid_controller, quicksort, simulated_annealing, **sornette_log_periodic**

Note: `sornette_log_periodic` is the only mixed-orbit algorithm with `orbit_seed=(0,0)` (the QA singularity) — it IS the crash/singularity detector.

## Strongest QA Affinity (top 10 by research note depth)

1. **continued_fractions** — φ = [1;1,1,...], every convergent is a QA orbit state
2. **pisano_period** — π(9)=24, π(24)=24: these ARE the QA cosmos orbit periods
3. **euclidean_gcd** — every QA orbit classification reduces to GCD; Fibonacci convergence ~ φ⁻ˢᵗᵉᵖˢ
4. **fibonacci_search** — orbit_seed (1,1), most φ-native search; convergents of φ
5. **pollard_rho** — pseudorandom orbit walk; ρ-shape = QA transient + satellite
6. **autocorrelation** — null baseline for orbit_follow_rate significance (open empirical question)
7. **sornette_log_periodic** — LPPL crash → QA singularity; SelfImprovementAgent monitor target
8. **hopfield_network** — energy function ~ QA norm N(b+eφ); attractors = singularities
9. **phase_locked_loop** — satellite (seeking) → cosmos (locked): textbook Levin lifecycle
10. **echo_state_network** — reservoir orbit; spectral radius ≈ 1 = cosmos/satellite edge

## Next Steps (post-consolidation)

1. **Autocorrelation baseline experiment** — is orbit_follow_rate (sine_880Hz=0.1715) > lag-1 AC?
   Use `autocorrelation` entry as the explicit baseline algorithm
2. **Sornette singularity exit implementation** — `qa_sornette_singularity_exit.py`
3. **Stem differentiation experiments** — push q_learning, k_means, pagerank through Levin lifecycle
4. **Organ assembly** — combine spine agents into multi-algorithm organs:
   - Number theory spine: GCD → extended_GCD → CRT → pisano_period (φ-orbit classifier)
   - Signal spine: autocorrelation → spectral_analysis → FFT → wavelet_transform
   - Planning spine: constraint_propagation → CDCL → beam_search → MCTS
5. **Batch 6** (when ready): probability, information theory, geometry (manifold learning), quantum algorithms
