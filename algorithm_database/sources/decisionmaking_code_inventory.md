<!-- PRIMARY-SOURCE-EXEMPT: reason="Inventory of github.com/algorithmsbooks/decisionmaking-code (companion code for Kochenderfer/Wheeler/Wray 2022 Algorithms for Decision Making). Repo cloned into algorithm_database/external_sources/decisionmaking-code/ (gitignored). This file is the inventory metadata; raw clone stays local-only. (Kochenderfer, 2022; algorithmsbooks-org, 2024)." -->

# `decisionmaking-code` — Code Inventory

**Status**: FETCHED (2026-04-27 via `git clone --depth 1`) + INVENTORIED. Cloned content lives at `algorithm_database/external_sources/decisionmaking-code/` and is gitignored.

**Source**: <https://github.com/algorithmsbooks/decisionmaking-code>
**Repo size**: 360 KB; **single Julia file** (`decision_making_code.jl`, 4593 lines) containing all algorithm code from the DM book. **No notebooks** — this is the typeset Julia source code, organized by chapter.

**Per the DM-book README**: this file is the canonical Julia source for every Algorithm box in (Kochenderfer 2022) *Algorithms for Decision Making*, MIT Press. Easier to extract algorithms from than the algforopt-notebooks repo (which mixes algorithm code with figure-rendering).

## Structure

The single file uses comment-section markers `#################### <chapter-name> <subsection-number>` to demarcate each chapter's algorithms. **195 sections** across **25 chapters**:

| Section name | Book chapter | Line range | Subsections |
|---|---|---|---|
| `representation` | Ch 2 (Representation) | L13-L63 | 4 |
| `inference` | Ch 3 (Inference) | L71-L298 | 12 |
| `parameter-learning` | Ch 4 (Parameter Learning) | L311-L346 | 3 |
| `structure-learning` | Ch 5 (Structure Learning) | L354-L456 | 5 |
| `simple-decisions` | Ch 6 (Simple Decisions) | L502-L527 | 2 |
| `exact-solutions` | Ch 7 (Exact Solution Methods, MDP) + Ch 20 (Exact Belief State Planning, POMDP) | L541-L2821 | 11 |
| `value-function-approximations` | Ch 8 (Approximate Value Functions) | L700-L877 | 7 |
| `online-approximations` | Ch 9 (Online Planning, MDP) + Ch 22 (Online Belief State Planning) | L894-L3458 | 13 |
| `policy-search` | Ch 10 (Policy Search) | L1176-L1305 | 6 |
| `policy-gradient-estimation` | Ch 11 (Policy Gradient Estimation) | L1330-L1417 | 6 |
| `policy-gradient-optimization` | Ch 12 (Policy Gradient Optimization) | L1441-L1560 | 6 |
| `actor-critic` | Ch 13 (Actor-Critic) | L1601-L1655 | 3 |
| `validation` | Ch 14 (Policy Validation) | L1687 | 1 |
| `exploration-and-exploitation` | Ch 15 (Exploration and Exploitation) | L1703-L1812 | 9 |
| `model-based-methods` | Ch 16 (Model-Based Methods) | L1823-L2039 | 9 |
| `model-free-methods` | Ch 17 (Model-Free Methods) | L2060-L2170 | 6 |
| `imitation-learning` | Ch 18 (Imitation Learning) | L2205-L2434 | 8 |
| `beliefs` | Ch 19 (Beliefs) | L2479-L2647 | 9 |
| `offline-approximations` | Ch 21 (Offline Belief State Planning) | L2855-L3284 | 18 |
| `controller-abstractions` | Ch 23 (Controller Abstractions) | L3503-L3733 | 7 |
| `multiagent_reasoning` | Ch 24 (Multiagent Reasoning) | L3760-L3963 | 12 |
| `sequential_problems` | Ch 25 (Sequential Problems) | L3991-L4190 | 9 |
| `state_uncertainty` | Ch 26 (State Uncertainty) | L4230-L4293 | 4 |
| `collaborative_agents` | Ch 27 (Collaborative Agents) | L4346-L4447 | 5 |
| `search` | Appendix (Search Algorithms) | L4498-L4568 | 5 |

253 function definitions + 131 struct/abstract-type definitions, organized inside these sections.

## Coverage of v1's 7 algorithm-database entries

This repo backs **4 of v1's 7 entries** with code-repo references:

| v1 entry | Match | Lines | Section | Notes |
|---|---|---|---|---|
| `iterative_policy_evaluation` | ✓ direct | L564-571 | exact-solutions 3 (MDP) | DM Algorithm 7.3; Julia: `function iterative_policy_evaluation(𝒫::MDP, π, k_max)` |
| `value_iteration` | ✓ direct | L617-635 | exact-solutions 7 + 8 (MDP) | DM Algorithms 7.7-7.8; Julia: `function backup(𝒫::MDP, U, s)` + `struct ValueIteration` + `solve(M::ValueIteration, 𝒫::MDP)` |
| `forward_search` | ✓ direct | L926-944 | online-approximations (MDP) | DM Algorithm 9.2; Julia: `function forward_search(𝒫, s, d, U)` |
| `branch_and_bound` | ✓ direct | L952-973 | online-approximations (MDP) | DM Algorithm 9.3; Julia: `function branch_and_bound(𝒫, s, d, Ulo, Qhi)`. Note: v1's branch_and_bound entry references both DM §9.4 (this match) AND Opt §22.4 (LP-relaxation variant; not in this repo). |
| `gradient_descent` | ✗ | — | — | Optimization-book entry; algforopt-notebooks `first-order.ipynb` is the code source (v1.1 inventory). |
| `simulated_annealing` | ✗ | — | — | Optimization-book entry; algforopt-notebooks `stochastic.ipynb` chapter (v1.1 inventory). |
| `cyclic_coordinate_search` | ✗ | — | — | Optimization-book entry; algforopt-notebooks `direct.ipynb` chapter (v1.1 inventory). |

**Combined v1.1 + v1.2 result**: all 7 v1 entries now have code-repo backing. The earlier "NOT YET FETCHED" note in each DM entry's README can be flipped to point at this file.

## Additional algorithms found (not in v1)

A few high-value entries beyond v1's 7 stand out as natural v1.3+ candidates from this repo:

| Slug | Lines | Section | Notes |
|---|---|---|---|
| `policy_evaluation` (linear-algebra variant) | L575-580 | exact-solutions 4 | DM Algorithm 7.4: `(I - γT')\R'` direct linear solve; complement to iterative version. |
| `policy_iteration` | (in exact-solutions 5-6 area) | exact-solutions 5-6 | DM Algorithm 7.5-7.6. |
| `gauss_seidel_value_iteration` | L637-650 | exact-solutions 9 | DM Algorithm 7.9; in-place backup variant. |
| `linear_program_formulation` | L653-676 | exact-solutions 10 | DM Algorithm 7.10; LP solution to MDP via GLPK. |
| `linear_quadratic_problem` | L678-? | exact-solutions 11 | DM §7.8: LQR closed-form. |
| `monte_carlo_tree_search` | (online-approximations area) | online-approximations | DM §9.6. |
| `pomdp_value_iteration` | L2803-2820 | exact-solutions 9 (POMDP) | DM Algorithm 20.8; alpha-vector + plan pruning. |
| `discrete_state_filter` | (beliefs area) | beliefs | DM §19.2. |
| `kalman_filter` | (beliefs area) | beliefs | DM §19.3. |
| `markov_game` | (sequential_problems area) | sequential_problems | DM §25.1. |
| `dec_pomdp` | (collaborative_agents area) | collaborative_agents | DM §27.1. |

These are **not added in v1.2** — same "inventory before expansion" rule as v1.1. v1.3+ (when greenlit) can add database rows for any of these by copying TEMPLATE/ and filling in fields, citing this inventory's line ranges as the code-repo backing.

## What this inventory does NOT do

- Does **not** copy decision_making_code.jl into QA-MEM. The cloned repo is a fetched reference, not a SourceWork. If a future cert needs the Julia source as primary, that's a separate Phase 4.x ingestion decision.
- Does **not** retrofit the 3 DM entries with new algorithm content. The v1 README + classical.py files use the QA-MEM-anchored book pseudocode, which is unchanged. Only the "Original code location" line in each DM entry's README flips from "NOT YET FETCHED" to point at this inventory + the specific line range.
- Does **not** create new sharp-claim certs. v1.3+ expansion may add database rows but not certs.

## Reproducing the fetch

```bash
cd algorithm_database/external_sources
git clone --depth 1 https://github.com/algorithmsbooks/decisionmaking-code.git
# .gitignore in external_sources/ already excludes the clone
```

Repo size after clone: ~360 KB (single Julia file, no notebooks).
