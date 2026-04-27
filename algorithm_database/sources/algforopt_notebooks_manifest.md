<!-- PRIMARY-SOURCE-EXEMPT: reason="Manifest for github.com/algorithmsbooks/algforopt-notebooks (Jupyter notebooks for Algorithms for Optimization). Repo NOT YET FETCHED in this pass — manifest documents the structure for future passes. Source: (Kochenderfer, 2026) trilogy + (algorithmsbooks-notebooks, 2024) repo." -->

# `algforopt-notebooks` — Source Manifest

GitHub: <https://github.com/algorithmsbooks/algforopt-notebooks>

Companion Jupyter notebooks for *Algorithms for Optimization* (Kochenderfer + Wheeler). The repo provides Julia 1.0.1 + IJulia + PGFPlots.jl notebooks reproducing the figures and algorithms in the textbook.

## Status: FETCHED + INVENTORIED (2026-04-27, v1.1)

The algforopt-notebooks repo was fetched in algorithm-database v1.1 via `git clone --depth 1` into `algorithm_database/external_sources/algforopt-notebooks/` (gitignored — cloned content stays local-only; only inventory metadata enters git).

**Full inventory**: see `algforopt_notebooks_inventory.md` (this directory). 24 .ipynb notebooks parsed; 8 contain algorithm code, 13 are figure-rendering, 3 are appendix/reference. Per the upstream README: *"These notebooks were generated from the Algorithms for Optimization source code... to aid with the development of lectures."* — i.e., the notebooks are **supplemental visualization** for the textbook, not the canonical source for algorithm bodies. The book PDF stays canonical.

**Coverage of v1's 7 entries**: 
- `gradient_descent` directly backed by `first-order.ipynb` `GradientDescent` struct
- `cyclic_coordinate_search`, `simulated_annealing`, `branch_and_bound` are in chapters whose notebooks contain only viz/helpers (book PDF remains canonical)
- `iterative_policy_evaluation`, `value_iteration`, `forward_search` are **DM-book entries; not in this Optimization-book repo** (would need `algorithmsbooks/decisionmaking-code` fetch — see `algorithmsbooks_org.md`)

## Notebook inventory (per ChatGPT's analysis 2026-04-26 in the original ingestion request)

| Notebook | Why it matters for QA |
|---|---|
| `discrete.ipynb` | Best immediate bridge to integer/discrete optimization, QA state-space search, generator moves, reachability, and failure algebra |
| `descent.ipynb` | Classical baseline for QA HGD comparison; gradient descent / line search / step factor |
| `first-order.ipynb` | First-order methods (gradient, momentum, Adam, Hypergradient) — note Kochenderfer's HGD is mathematically distinct from QA HGD per bridge §8.2 |
| `second-order.ipynb` | Newton's method, Levenberg-Marquardt, quasi-Newton |
| `stochastic.ipynb` | Simulated annealing, cross-entropy, CMA-ES — observer-projection candidates only |
| `surrogate-models.ipynb` | Surrogate model fitting — observer-projection candidates only |
| `surrogate-optimization.ipynb` | Bayesian optimization / expected improvement — observer-projection candidates only |
| `multiobjective.ipynb` | Pareto optimality, dominance, weight methods |
| `design-under-uncertainty.ipynb` | Set-based + probabilistic uncertainty; minimax |
| `uncertaintyprop.ipynb` | Monte Carlo, Taylor approximation, polynomial chaos |
| `probabilistic-surrogate-models.ipynb` | Gaussian processes — firewall-rejected as causal QA |
| `expression-optimization.ipynb` | Genetic programming, grammars |

(Source: ChatGPT 2026-04-26 analysis pasted in the algorithm-database ingestion request.)

## Future-pass strategy (v1.2+)

Now that the repo is fetched and inventoried, the next steps when database expansion is greenlit:

1. Add v1.2 entries from inventory candidates listed in `algforopt_notebooks_inventory.md` "Candidate v1.2 entries" section. Each new entry: copy `TEMPLATE/`, fill in source reference (now pointing at notebook OR book OR both), classical port (where notebook has runnable code), QA mapping (cite bridge spec row + cert evidence as available).
2. Decide whether to ingest specific notebook content into QA-MEM as additional SourceWorks (per Phase 4.x discipline). Most notebooks are figure-rendering and shouldn't be ingested; the few with algorithm code (`first-order`, `multiobjective`, `sampling-plans`, `surrogate-models`, `surrogate-optimization`) are candidates if a future cert needs them as primary source.
3. Fetch `algorithmsbooks/decisionmaking-code` to back the DM-book entries (`iterative_policy_evaluation`, `value_iteration`, `forward_search`) with notebook/code refs.

## Reproducing the fetch

```bash
mkdir -p algorithm_database/external_sources
cd algorithm_database/external_sources
git clone --depth 1 https://github.com/algorithmsbooks/algforopt-notebooks.git
# .gitignore at external_sources/.gitignore excludes the clone from git
```

Repo size after clone: ~836 KB. Julia 1.0.1 kernels; PGFPlots.jl-driven figures.
