<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database front-door / catalog index. Source citations live in entries/<algorithm>/qa_mapping.md and sources/<source>.md manifests; this file is a queryable table of contents over QA-MEM corpus + Kochenderfer bridge + sharp-claim certs (Kochenderfer, 2026; Dale, 2026; Sole, 2026). Not a research-claim doc — pure catalog layer." -->

# Algorithm Database — Front Door

> **Adding work to this lane?** Read [`OPERATING_GUIDE.md`](OPERATING_GUIDE.md) first. It locks in lane boundaries (corpus / bridge / certs / database) and documents the add-an-entry checklist + anti-patterns. The guide exists specifically to prevent the scope-drift cycles that delayed this lane's completion in the 2026-04-26/27 work blocks.

A queryable catalog of classical algorithms with their QA conversions. Each row is one algorithm; the columns are source location, classical formulation status, QA mapping status, and evidence pointer.

**Scope discipline (per Will + ChatGPT, 2026-04-27):**
- This database is a **front-door / query layer** over the QA-MEM primary-source corpus, the Kochenderfer bridge spec, and the sharp-claim cert ecosystem. It is **not** a replacement corpus or a new research framework.
- Each entry references existing artifacts (SourceWorks in `qa_kg.db`, mapping rows in `docs/specs/QA_KOCHENDERFER_BRIDGE.md`, certs in `qa_alphageometry_ptolemy/<cert>_cert_v1/`) — does not duplicate them.
- No new sharp-claim certs are created in the database lane. New certs land in the cert ecosystem and the database row's `qa_mapping.md` references them.
- Per Theorem NT, the database documents the firewall direction at every entry: which side of the Theorem NT firewall the classical algorithm and its QA counterpart live on.

**Schema (per row):**
- `entries/<algorithm>/source_reference.md` — anchor to source notebook/book/chapter/section
- `entries/<algorithm>/classical.md` — classical mathematical form + pseudocode (with attribution if from book; transcribed only when source is QA-MEM-anchored)
- `entries/<algorithm>/classical.py` — Python port (where feasible; not always present)
- `entries/<algorithm>/qa_mapping.md` — narrative: QA counterpart + status + cert/utility evidence + Theorem NT boundary note
- `entries/<algorithm>/qa_native.py` — QA-side equivalent (where one exists; not always present)
- `entries/<algorithm>/test_equivalence.py` — only where a real equivalence claim exists; absent otherwise

**Status field values** (mirrors bridge spec):
- `established` — QA already has a concrete artifact (cert, validator, doc) implementing the algorithm's QA counterpart
- `candidate` — mapping is mechanically clear and the QA artifact already exists; needs vocabulary alignment, not new code
- `open` — no QA counterpart exists; future cert candidate (listed in `docs/specs/QA_KOCHENDERFER_BRIDGE.md` Standing Rule #2)
- `rejected` — Kochenderfer concept does not transfer to QA-discrete (usually a Theorem NT firewall constraint)

---

## Catalog (initial 7 entries, 2026-04-27)

| Algorithm | Source | Classical | QA Mapping Status | QA Counterpart / Evidence |
|---|---|---|---|---|
| [iterative_policy_evaluation](entries/iterative_policy_evaluation/) | (Kochenderfer 2022) DM §7.2 | Bellman lookahead `U^π_{k+1}(s) = R(s,π(s)) + γ Σ T(s'\|s,π(s)) U^π_k(s')`; iterative DP | candidate | QA reachability descent on orbit graph; cert [263] utility (`orbit_failure_enumeration.py`) provides class-population enumeration; bridge §7.1 row |
| [value_iteration](entries/value_iteration/) | (Kochenderfer 2022) DM §7.5 | Bellman backup `U_{k+1}(s) = max_a [R(s,a) + γ Σ T(s'\|s,a) U_k(s')]` | established | cert [191] tier-classification = Bellman backup with deterministic transitions on QA orbit graph; bridge §7.1 row established |
| [forward_search](entries/forward_search/) | (Kochenderfer 2022) DM §9.3 | Depth-d expansion of all transitions; DFS with terminal `U(s)`; complexity `O((\|S\|·\|A\|)^d)` | established | cert [191] tiered reachability on `S_9` does forward BFS from each state; bridge §7.2 row established |
| [branch_and_bound](entries/branch_and_bound/) | (Kochenderfer 2022) DM §9.4 + (Kochenderfer 2026 2e) Opt §22.4 | Prune subtrees via upper-bound `Q(s,a)_hi < Q(s,a*)_lo`; LP relaxation + integer branch | rejected (small `S`) / candidate (large `S`) | QA enumeration dominates B&B on `S_9`-scale orbit graphs; B&B becomes useful only at modulus where `\|S\|^h` is intractable; bridge §8.3 row |
| [simulated_annealing](entries/simulated_annealing/) | (Kochenderfer 2026 2e) Opt §8.4 | Metropolis criterion `accept iff Δy ≤ 0 ∨ rand() < e^{-Δy/t}`; exponential annealing schedule | rejected | continuous + stochastic Metropolis crosses Theorem NT firewall; QA enumeration dominates on QA-discrete side; bridge §8.4 row |
| [gradient_descent](entries/gradient_descent/) | (Kochenderfer 2026 2e) Opt §5.1 | Steepest descent `d^(k) = -g^(k)/‖g^(k)‖`; first-order Taylor approximation | rejected (continuous) / candidate (vocabulary alignment) | continuous-domain firewall-rejected as causal QA; cert [191] tier-1 generator selection is the QA-discrete vocabulary analog; bridge §8.2 row |
| [cyclic_coordinate_search](entries/cyclic_coordinate_search/) | (Kochenderfer 2026 2e) Opt §7.1 | Direct/zero-order method; alternates basis vectors for line search | candidate | cert [191] tier hierarchy (L_1 / L_2a / L_2b / L_3) IS the QA-native cyclic generator search analog; bridge §8.3 row |

---

## How to add a new algorithm

1. Pick a slug (lowercase, snake_case): e.g., `policy_iteration`.
2. Create `entries/<slug>/` with the files listed in `TEMPLATE/`.
3. Add a row to the catalog above with the appropriate status.
4. Reference an existing cert / utility / bridge row as evidence; if none exists, mark status as `open` and add the candidate to `docs/specs/QA_KOCHENDERFER_BRIDGE.md` Standing Rule #2.
5. **Do not** create a new sharp-claim cert in this lane; that's the cert lane. The database is a catalog; new certs land in `qa_alphageometry_ptolemy/`.

## Sources represented

- `sources/kochenderfer_trilogy_manifest.md` — pointers into the QA-MEM SourceWorks for the three textbooks (Validation, Decision Making, Optimization 2e + 1e). **FETCHED + INGESTED**.
- `sources/algforopt_notebooks_manifest.md` — Jupyter-notebooks repo `github.com/algorithmsbooks/algforopt-notebooks`. **FETCHED + INVENTORIED 2026-04-27 (v1.1)**. See full inventory at `sources/algforopt_notebooks_inventory.md` (24 notebooks indexed).
- `sources/algorithmsbooks_org.md` — broader GitHub org `github.com/algorithmsbooks` (11 repos total). **`algforopt-notebooks` FETCHED + INVENTORIED 2026-04-27 (v1.1)**, **`decisionmaking-code` FETCHED + INVENTORIED 2026-04-27 (v1.2)** — see [`sources/decisionmaking_code_inventory.md`](sources/decisionmaking_code_inventory.md). Other 9 repos (validation-code, ancillaries, etc.) NOT YET FETCHED.

## Notebook + code coverage (v1.1 + v1.2, 2026-04-27)

Per `sources/algforopt_notebooks_inventory.md`, the 24 algforopt-notebooks map to v1's 7 entries as follows:

| v1 entry | Notebook backing (algforopt-notebooks v1.1)? | Code backing (decisionmaking-code v1.2)? |
|---|---|---|
| `gradient_descent` | ✓ `first-order.ipynb` (`GradientDescent` struct + `init!()`/`step!()`) | n/a (Optimization-book entry) |
| `cyclic_coordinate_search` | partial — `direct.ipynb` has Divided Rectangles support structs | n/a (Optimization-book entry) |
| `simulated_annealing` | partial — `stochastic.ipynb` chapter; helper structs only | n/a (Optimization-book entry) |
| `branch_and_bound` | partial — `discrete.ipynb` chapter; viz-only | ✓ `decision_making_code.jl` L952-973 (DM Algorithm 9.3) |
| `iterative_policy_evaluation` | n/a (DM-book entry) | ✓ `decision_making_code.jl` L564-571 (DM Algorithm 7.3) |
| `value_iteration` | n/a (DM-book entry) | ✓ `decision_making_code.jl` L617-635 (DM Algorithms 7.7-7.8) |
| `forward_search` | n/a (DM-book entry) | ✓ `decision_making_code.jl` L926-944 (DM Algorithm 9.2) |

**All 7 v1 entries now have code-repo backing.** Combined v1.1 + v1.2 closes the coverage gap. No book-canonical algorithm body was lost or duplicated — for the 4 entries shared between Optimization and Decision Making contexts, the canonical pseudocode in the entry's `classical.py` came from the QA-MEM-anchored book PDF, and the code-repo references are documentary additions.

**v1.3+ candidates from inventories** (not added in v1.2; see `algforopt_notebooks_inventory.md` and `decisionmaking_code_inventory.md` "Additional algorithms found" sections):
- From algforopt-notebooks: `expected_improvement`, `pareto_optimality`, `dominates`, `linear_regression`, `newton_method`, `divided_rectangles`, `genetic_algorithm`, `simplex_algorithm`, `policy_evaluation` (linear-algebra variant), etc. — ~30 candidates.
- From decisionmaking-code: `policy_iteration`, `gauss_seidel_value_iteration`, `linear_program_formulation` (MDP), `linear_quadratic_problem`, `monte_carlo_tree_search`, `pomdp_value_iteration` (alpha-vector), `discrete_state_filter`, `kalman_filter`, `markov_game`, `dec_pomdp`, etc. — ~20 candidates.

Per the standing rule "inventory before expansion": v1.3+ adds database rows by copying `TEMPLATE/` and filling in fields, citing the inventory line ranges as code-repo backing. No new sharp-claim certs in this lane.

## Connection to research-bridge artifacts

Each entry's `qa_mapping.md` cites:
- The relevant verbatim anchor in `docs/theory/kochenderfer_*_excerpts.md` (the QA-MEM corpus side).
- The relevant row in `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §1-§8 (the bridge mapping side).
- The relevant cert family in `qa_alphageometry_ptolemy/<cert>_cert_v1/` (the empirical-evidence side, where applicable).

The research-bridge work and the database are complementary, not substitutable. The bridge is the *mapping*; the database is the *catalog with row-level evidence pointers*; the certs are the *empirical claims*. The three together are the Kochenderfer-trilogy unit.
