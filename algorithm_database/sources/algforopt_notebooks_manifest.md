<!-- PRIMARY-SOURCE-EXEMPT: reason="Manifest for github.com/algorithmsbooks/algforopt-notebooks (Jupyter notebooks for Algorithms for Optimization). Repo NOT YET FETCHED in this pass — manifest documents the structure for future passes. Source: (Kochenderfer, 2026) trilogy + (algorithmsbooks-notebooks, 2024) repo." -->

# `algforopt-notebooks` — Source Manifest

GitHub: <https://github.com/algorithmsbooks/algforopt-notebooks>

Companion Jupyter notebooks for *Algorithms for Optimization* (Kochenderfer + Wheeler). The repo provides Julia 1.0.1 + IJulia + PGFPlots.jl notebooks reproducing the figures and algorithms in the textbook.

## Status: NOT YET FETCHED in this pass

The algorithm-database v1 (2026-04-27) uses the **textbook PDFs** as the primary source; the Julia notebooks repo has not been cloned. This manifest documents the notebook inventory for future passes when fetching is greenlit.

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

## Future-pass instructions (when fetching is greenlit)

1. Clone the repo into a controlled local location (e.g., `~/Downloads/algforopt-notebooks/` or under `Documents/kochenderfer_corpus/`).
2. Decide whether to ingest the notebooks as additional QA-MEM SourceWorks (one SourceWork per notebook? one per repo? per Phase 4.x discipline).
3. For each notebook with QA-relevant algorithms (discrete/descent/stochastic), extract the algorithm bodies and create algorithm-database entries that link the notebook code to the existing book-level entries (the database row would have BOTH a book pseudocode AND a notebook code reference).
4. Update `INDEX.md` with notebook-source rows or extend existing book-source rows with a "notebook code" column.

## Why we didn't fetch in this pass

Per Will + ChatGPT scoping discipline 2026-04-27: build the catalog scaffold first using the QA-MEM-anchored book pseudocode (which has proper attribution and is already on disk), then layer on notebook code as a second pass. The catalog structure does not depend on the notebooks being fetched; the entries point at the book sections via the QA-MEM excerpts file.

This is consistent with the bridge spec's standing rule that the database is a "front-door / query layer over QA-MEM + bridge + certs, not a replacement for them" — the notebook fetch is a corpus-extension move, deferred to a focused session.
