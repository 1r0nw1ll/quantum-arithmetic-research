<!-- PRIMARY-SOURCE-EXEMPT: reason="Inventory of github.com/algorithmsbooks/algforopt-notebooks (Kochenderfer + Wheeler 2026 Algorithms for Optimization companion notebooks). Repo cloned into algorithm_database/external_sources/algforopt-notebooks/ (gitignored). This file is the inventory metadata; raw cloned content stays local-only per the database rule that external_sources/ is not vendored. (Kochenderfer, 2026; algorithmsbooks-notebooks, 2024)." -->

# `algforopt-notebooks` — Notebook Inventory

**Status**: FETCHED (2026-04-27 via `git clone --depth 1`) + INVENTORIED. Cloned content lives at `algorithm_database/external_sources/algforopt-notebooks/` and is gitignored — only this metadata file enters git.

**Source**: <https://github.com/algorithmsbooks/algforopt-notebooks>
**Per the repo's own README**: *"These notebooks were generated from the Algorithms for Optimization source code. We provide these notebooks to aid with the development of lectures and understanding the material."* — i.e., the notebooks are **supplemental visualization** for the textbook, not the canonical source for algorithm bodies. The book PDF (`Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_optimization_2e.pdf`) remains the canonical source.

**Repo size**: 836 KB; 24 .ipynb notebooks; Julia 1.0.1 kernel; PGFPlots.jl-driven figure rendering.

## Inventory (all 24 notebooks)

| Notebook | Book chapter | Type | Algorithm/code cells extracted | Candidate db entry slug | Already covered? |
|---|---|---|---|---|---|
| `introduction.ipynb` | Ch 1 Introduction | viz-only | (none — figure rendering) | — | n/a |
| `derivatives.ipynb` | Ch 2 Derivatives & Gradients | viz-only | `get_axis()` (helper) | — | n/a |
| `bracketing.ipynb` | Ch 3 Bracketing | viz-only | (none) | `fibonacci_search`, `golden_section_search`, `quadratic_fit_search`, `bisection_method` | NOT covered (4 candidates for v1.2) |
| `descent.ipynb` | Ch 4 Local Descent | viz-only | `solve_trust_region_subproblem()` | `line_search`, `trust_region`, `backtracking_line_search` | NOT covered (3 candidates for v1.2) |
| `first-order.ipynb` | Ch 5 First-Order Methods | algorithm structs | `abstract type DescentMethod`, `struct GradientDescent`, `init!()`, `step!()` | `gradient_descent` | **YES — v1 entry uses book Algorithm 5.1; notebook has compatible struct definition** |
| `second-order.ipynb` | Ch 6 Second-Order Methods | algorithm hooks | `abstract type DescentMethod`, `_line_search()`, `init!()`, `step!()` | `newton_method`, `secant_method`, `levenberg_marquardt`, `bfgs`, `lbfgs` | NOT covered (5 candidates for v1.2) |
| `direct.ipynb` | Ch 7 Direct Methods | data structs | `struct Cell`, `struct Cell2D`, `struct FuncEval`, `struct Interval` (Divided Rectangles support) | `cyclic_coordinate_search` (covered) + `powell`, `hooke_jeeves`, `nelder_mead`, `divided_rectangles` | PARTIAL — `cyclic_coordinate_search` covered; 4 v1.2 candidates remain |
| `stochastic.ipynb` | Ch 8 Stochastic Methods | helper | `struct OnlineMeanAndVariance`, `struct Pattern`, `update()` | `simulated_annealing` (covered) + `cross_entropy_method`, `cma_es`, `noisy_descent` | PARTIAL — `simulated_annealing` covered; 3 v1.2 candidates remain |
| `population.ipynb` | Ch 9 Population Methods | viz-only | (none) | `genetic_algorithm`, `differential_evolution`, `particle_swarm`, `firefly`, `cuckoo_search` | NOT covered (5 candidates for v1.2) |
| `penalty.ipynb` | Ch 10 Constraints | viz-only | (none) | `penalty_method`, `lagrange_multipliers`, `interior_point` | NOT covered (3 candidates for v1.2) |
| `linear.ipynb` | Ch 12 Linear Programming | viz-only | (none) | `simplex_algorithm`, `linear_program` | NOT covered (2 candidates for v1.2) |
| `multiobjective.ipynb` | Ch 15 Multiobjective | algorithm code | `dominates()`, `naive_pareto()`, `mutate()`, `rand_population_uniform()`, GA structs (`TruncationSelection`, `GaussianMutation`, `InterpolationCrossover`) | `pareto_optimality`, `dominates`, `naive_pareto`, `weight_method`, `multiobjective_population` | NOT covered (5 candidates for v1.2) |
| `sampling-plans.ipynb` | Ch 16 Sampling Plans | algorithm code | `uniform_projection_plan()`, `pairwise_distances()`, `mutate()`, `exchange_algorithm()`, `greedy_local_search()`, `multistart_local_search()`, `halton()`, `get_filling_set_*()` (4 variants) | `uniform_projection_plan`, `halton_sampling`, `space_filling_subset` | NOT covered (3 candidates for v1.2) |
| `surrogate-models.ipynb` | Ch 17 Surrogate Models | algorithm code | `linear_regression()`, `regression()`, `design_matrix()`, `polynomial_bases()`, `sinusoidal_bases()`, `train_and_validate()`, `cross_validation_estimate()`, `bootstrap_*()` (3 variants), `holdout_partition()`, `k_fold_cross_validation_sets()`, `leave_one_out_bootstrap_estimate()`, `print_coordinate()` | `linear_regression`, `polynomial_basis`, `cross_validation`, `bootstrap` | NOT covered (4 candidates for v1.2) |
| `prob-surrogate-models.ipynb` | Ch 18 Probabilistic Surrogates (Gaussian Processes) | viz-only | (none) | `gaussian_process_fit`, `gaussian_process_predict` | NOT covered (2 candidates for v1.2) |
| `surrogate-optimization.ipynb` | Ch 19 Surrogate Optimization | algorithm code | `expected_improvement()`, `flower()` (test fn) | `expected_improvement`, `prob_of_improvement`, `lower_confidence_bound`, `safe_optimization` | NOT covered (4 candidates for v1.2) |
| `design-under-uncertainty.ipynb` | Ch 20 Optimization Under Uncertainty | viz-only | (none) | `minimax`, `set_based_uncertainty` | NOT covered (2 candidates for v1.2) |
| `uncertaintyprop.ipynb` | Ch 21 Uncertainty Propagation | algorithm code | `hermite()`, `laguerre()`, `legendre()` (orthogonal polynomial bases) | `monte_carlo_estimation`, `polynomial_chaos`, `taylor_approximation` | NOT covered (3 candidates for v1.2) |
| `discrete.ipynb` | Ch 22 Discrete Optimization | viz-only | (none) | `branch_and_bound` (covered) + `cutting_plane`, `dynamic_programming`, `ant_colony` | PARTIAL — `branch_and_bound` covered; 3 v1.2 candidates remain |
| `expr.ipynb` | Ch 23 Expression Optimization | viz-only | (none) | `genetic_programming`, `grammatical_evolution` | NOT covered (2 candidates for v1.2) |
| `mdo.ipynb` | Ch 24 Multidisciplinary | viz-only | (none) | `multidisciplinary_feasible`, `sequential_optimization` | NOT covered (2 candidates for v1.2) |
| `julia.ipynb` | Appendix D Julia | language tutorial | (none — Julia syntax intro) | — | n/a |
| `math-concepts.ipynb` | Appendix B Math Concepts | reference | (none — math reference cards) | — | n/a |
| `test-functions.ipynb` | Appendix B Test Functions | reference | (none — Ackley, Branin, Booth, Rosenbrock, etc. visualizations) | — | n/a |

## Summary

- **24 notebooks total** — 8 contain algorithm code (functions/structs); 13 are figure-rendering only; 3 are appendix/reference material.
- **Coverage of current 7 db entries**:
  - `gradient_descent` ✓ — backed by `first-order.ipynb` `GradientDescent` struct
  - `cyclic_coordinate_search` ✓ — `direct.ipynb` has supporting Divided Rectangles structs (not the CCS algorithm itself, but same chapter)
  - `simulated_annealing` ✓ — `stochastic.ipynb` chapter; SA algorithm body is in book Algorithm 8.5 (notebook has only OnlineMeanAndVariance helper)
  - `branch_and_bound` ✓ — `discrete.ipynb` chapter; B&B body is in book Algorithm 22.5 (notebook is figure-only)
  - `iterative_policy_evaluation`, `value_iteration`, `forward_search` — these are **DM book entries; NOT in this Optimization-book repo**. They would need fetching from `algorithmsbooks/decisionmaking-code` (not yet fetched).
- **No book-canonical algorithm body lost**: notebooks are visualizations; book PDF remains the source for all 7 v1 entries. The earlier database scaffold decision to anchor at book sections (not notebooks) was correct.

## Candidate v1.2 entries from this inventory

The notebooks suggest these algorithms as natural next-batch entries (each maps to a non-empty algorithm cell with QA-relevant content):

| Slug | Notebook | Bridge spec status | Notes |
|---|---|---|---|
| `expected_improvement` | `surrogate-optimization.ipynb` | rejected (continuous Bayesian opt; bridge §8.5) | Add as catalog row with `rejected (continuous)` status; demonstrates the firewall-rejected lane is still inventoried. |
| `pareto_optimality` (or `dominates`) | `multiobjective.ipynb` | candidate (bridge §8.6 — cert ecosystem composition admits Pareto re-description) | Real notebook code (`dominates()`, `naive_pareto()`); could be a runnable Python port. |
| `linear_regression` (surrogate) | `surrogate-models.ipynb` | rejected (continuous) / candidate (cert [259]-style observer projection at input boundary) | Real notebook code; useful for documenting the firewall-rejected-but-observer-projection lane. |
| `newton_method` | `second-order.ipynb` | rejected (continuous second-order) | Same pattern as gradient_descent: classical port + firewall-rejection note. |
| `divided_rectangles` | `direct.ipynb` | candidate (extends cyclic_coordinate_search; tier hierarchy analog?) | Has multiple supporting structs in the notebook; real candidate for v1.2. |
| `genetic_algorithm` | `multiobjective.ipynb` (population context) + `population.ipynb` | rejected (continuous + stochastic) | Off-QA baseline; document for completeness. |
| `simplex_algorithm` | `linear.ipynb` | candidate / rejected (continuous LP) | Standard LP method; bridge §8.1 row points at LP formulation. |

These are **not added in v1.1** — v1.1 is inventory-only. Per Will + ChatGPT 2026-04-27: "inventory before expansion" prevents another drift cycle.

## What this inventory does NOT do

- Does **not** copy notebook content into QA-MEM. The cloned repo is a fetched reference, not a SourceWork. If any notebook gets ingested into qa_kg.db as a primary source, that's a separate Phase 4.x ingestion decision (not v1.1 scope).
- Does **not** retrofit the existing 7 entries. None of the 7 `qa_mapping.md` files need updating because the book PDF anchor is still canonical.
- Does **not** create new sharp-claim certs. v1.2 expansion may add database rows but not certs (separate lane per `INDEX.md` standing rule).
