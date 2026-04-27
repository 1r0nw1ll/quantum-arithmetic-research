<!-- PRIMARY-SOURCE-EXEMPT: reason="Manifest for github.com/algorithmsbooks GitHub organization (broader catalog of repos backing the Kochenderfer trilogy). Org-level manifest, not a primary-source ingestion. (Kochenderfer, 2026; algorithmsbooks-org, 2024)." -->

# `algorithmsbooks` GitHub Organization — Source Manifest

GitHub: <https://github.com/algorithmsbooks>

Hosts the source code, notebooks, and ancillary materials for the three textbooks in the Kochenderfer/Wheeler/Wray/Katz/Corso/Moss series.

## Repository inventory

| Repo | Purpose | QA relevance | Fetch status |
|---|---|---|---|
| `algforopt-notebooks` | Jupyter notebooks for *Algorithms for Optimization* | Classical baseline for HGD, QA optimizer, discrete search, surrogate optimization | NOT YET FETCHED — see `algforopt_notebooks_manifest.md` |
| `optimization` | Errata for *Algorithms for Optimization* | Useful for citation hygiene / source correction | NOT YET FETCHED |
| `optimization-ancillaries` | Supporting materials for the optimization book | Likely figures, exercises, supplemental assets | NOT YET FETCHED |
| `decisionmaking` | *Algorithms for Decision Making* textbook source | Very relevant to QA as control / reachability theory | NOT YET FETCHED |
| `decisionmaking-code` | Typeset code blocks from *Algorithms for Decision Making* | Easier to extract algorithms than from notebooks | **FETCHED + INVENTORIED 2026-04-27 (v1.2)** — see [`decisionmaking_code_inventory.md`](decisionmaking_code_inventory.md). Single Julia file (4593 lines), 195 chapter-sections, 253 functions. Backs v1's 4 DM-context entries (`iterative_policy_evaluation`, `value_iteration`, `forward_search`, `branch_and_bound`). |
| `decisionmaking-ancillaries` | Ancillaries for the decision-making book | Supporting reference set | NOT YET FETCHED |
| `DecisionMakingProblems.jl` | Julia package of decision-making problems | Strong candidate for benchmark harnesses | NOT YET FETCHED |
| `validation` | *Algorithms for Validation* textbook source | Relevant to certs, empirical validation, benchmark design | NOT YET FETCHED |
| `validation-code` | Code for validation book | Useful for QA empirical observation certs | NOT YET FETCHED |
| `validation-ancillaries` | Supporting validation materials | Good fit for validator/cert ecosystem | NOT YET FETCHED |
| `validation-figures` | Figures for validation book | Useful if explanatory diagrams are needed later | NOT YET FETCHED |

(Source: ChatGPT 2026-04-26 analysis pasted in the algorithm-database ingestion request.)

## Coverage status (v1 algorithm-database, 2026-04-27)

In v1, **none of the org repos are fetched**. The PDFs of the three textbooks (Validation, Decision Making, Optimization 2e + 1e) are on disk under `Documents/kochenderfer_corpus/` and serve as the primary source for the v1 algorithm-database entries. **v1.1 (2026-04-27)** added `algforopt-notebooks` (24 .ipynb), and **v1.2 (2026-04-27)** added `decisionmaking-code` (single 4593-line Julia file). Combined, v1.1 + v1.2 give code-repo backing for all 7 v1 entries. Algorithm pseudocode is transcribed from the QA-MEM verbatim excerpts (`docs/theory/kochenderfer_*_excerpts.md`), which were extracted with proper attribution during the QA-MEM Phase 4.x ingestion.

## Future-pass strategy

Per Will + ChatGPT scoping 2026-04-27, the database is a **front-door / query layer**, not a replacement for the QA-MEM corpus. Org-level fetch decisions belong to a focused later session when there's a concrete need:

1. **`decisionmaking-code` next** — typeset code blocks would be the easiest source for clean Python ports of MDP / POMDP / Markov-game algorithms.
2. **`DecisionMakingProblems.jl`** — would unlock real benchmarks for the candidate sharp-claim certs in bridge §7 (e.g., `qa_dec_pomdp_orbit_coordination_cert_v1`).
3. **`algforopt-notebooks`** — see `algforopt_notebooks_manifest.md`; lowest-priority of the three because the textbook PDFs already cover most of the algorithm content.
4. **`validation-code`** — useful for empirical-observation certs once we want to expand beyond the v1 synthetic benchmarks in [263][264][265].

Each fetch is a corpus-extension move per QA-MEM Phase 4.x discipline (don't ingest into a side scaffold; SourceWorks go through QA-MEM).
