<!-- PRIMARY-SOURCE-EXEMPT: reason="QA ↔ Kochenderfer 2026 controlled mapping index. Source citations + DOI live below in §Provenance and per-row source-anchor links to docs/theory/kochenderfer_validation_excerpts.md (15 verbatim anchors). Companion fixture: tools/qa_kg/fixtures/source_claims_kochenderfer.json. Bridge spec, not a research claim doc." -->

# QA ↔ Kochenderfer/Wheeler Bridge — Controlled Mapping Index

**Source corpus:** Kochenderfer, Wheeler, Katz, Corso, Moss (2026). *Algorithms for Validation*. MIT Press. CC-BY-NC-ND. 441 pp. Plus the companion textbook (Kochenderfer, Wheeler, Wray, 2022). *Algorithms for Decision Making*. MIT Press. 700 pp. (ingested 2026-04-27 — bridge §7 below). Optimization textbook (Kochenderfer + Wheeler 2019) queued.

**On-disk:**
- `Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_validation.pdf` — excerpts: `docs/theory/kochenderfer_validation_excerpts.md` (15 anchors); fixture: `tools/qa_kg/fixtures/source_claims_kochenderfer.json` (1 SourceWork, 15 SourceClaims).
- `Documents/kochenderfer_corpus/kochenderfer_wheeler_wray_2022_algorithms_for_decision_making.pdf` — excerpts: `docs/theory/kochenderfer_decision_making_excerpts.md` (15 anchors); fixture: `tools/qa_kg/fixtures/source_claims_dm.json` (1 SourceWork, 15 SourceClaims).

**Purpose:** Translate Kochenderfer's external vocabulary for validation algorithms into QA terms, and document which QA artifacts (certs, validators, theorems, infrastructure) already implement Kochenderfer's concepts under different names. This bridge serves the Terminal Goal: making the QA cert ecosystem legible to skeptical technical readers who already know the Kochenderfer formalism.

**Scope discipline (per Will, 2026-04-26):**
- This is a *controlled mapping index*, not a prose dump.
- No taxonomy / classification cert is created.
- Future cert families are listed only when there is a concrete falsifiable pairwise empirical claim, not for "Kochenderfer concept X has a QA name."
- Each row carries a status — `established` / `candidate` / `open` / `rejected` — interpreted as:
  - **established** = QA already has a concrete artifact (cert, validator, doc) that implements the same idea; the row documents where to find it.
  - **candidate** = the mapping is mechanically clear and the QA artifact already exists; the next step is a small docs/vocabulary edit, not a new cert.
  - **open** = no QA counterpart exists yet; a future cert family with a sharp claim is plausible. The "Future cert claim" column names the empirical question the cert would adjudicate.
  - **rejected** = the Kochenderfer concept does not transfer to QA (usually a Theorem NT firewall constraint).

---

## §1. Validation framework — mapping table

| Kochenderfer concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| Validation = "establishing confidence that a system will behave as desired when deployed in the real world" | `#val-1-1-validation-definition` | The QA cert ecosystem (mapping_protocol → validator → meta_validator) is exactly an instance of this. Each cert family is a `(system, specification) → metrics` triple. | `qa_alphageometry_ptolemy/qa_meta_validator.py` (registry of ~262 cert families); `docs/families/README.md` (index) | established | — |
| Verification (special case of validation) = "guarantees about the correctness of a system with respect to a specification" | `#val-1-1-validation-definition` | TLA+ specifications under `qa_alphageometry_ptolemy/QARM_v02_Failures*.tla` give Kochenderfer-grade verification (model-checked, exhaustive). Cert validators give validation-grade evidence (PASS/FAIL on enumerated witnesses). The QA stack uses both layers — TLA+ as constitution, Python validator as runtime. | `qa_alphageometry_ptolemy/QARM_v02_Failures.tla` + `.cfg`; `docs/specs/QA_TLA_PLUS.md` | established | — |
| Testing (special case of validation) = "evaluating the system on a discrete set of test cases" | `#val-1-1-validation-definition` | Cert fixtures (PASS / FAIL / negative) under `<cert>/fixtures/` ARE Kochenderfer-style discrete test cases. | every cert family has a `fixtures/` directory; `qa_meta_validator.py::FAMILY_SWEEPS` runs them | established | — |
| Alignment problem = "mismatch between desired and deployed behavior due to imperfect model / objective / optimization" | `#val-1-1-alignment-problem` | Theorem NT (Observer Projection Firewall) is QA's structural answer: never let continuous optimization objectives feed back as causal inputs to the QA discrete layer. The alignment problem in QA terms is `T2-b` (float × modulus → int cast); the firewall makes the misalignment a hard linter error rather than a runtime drift. | `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`; `tools/qa_axiom_linter.py`; `QA_AXIOMS_BLOCK.md` | candidate | — (vocabulary alignment, no new cert needed) |
| Validation algorithm signature `(system, specification) → metrics`; system = (agent, environment, sensor); specification = operating requirement | `#val-1-4-validation-algorithm-inputs` | Each QA cert family has the exact same signature: `(mapping_protocol.json, validator.py) → {PASS\|FAIL, gate-level telemetry}`. Mapping protocol = system + specification jointly; validator output = metrics. | `qa_mapping_protocol/schema.json`; every `<cert>/validator.py` | established | — |
| Swiss cheese / safety case — "we cannot guarantee … using a single validation algorithm or metric. Instead, we use a combination of these techniques to build a safety case" | `#val-1-4-swiss-cheese-safety-case` | This is the empirical justification for QA's many-small-certs-not-one-mega-cert design. The cert ecosystem (262 families) is the safety case; each cert is a Swiss-cheese slice. Confirmed by Will's standing rule (see `MEMORY.md` §Cert Discipline) and validated by [127] retrospective. | `qa_alphageometry_ptolemy/qa_meta_validator.py::FAMILY_SWEEPS`; `docs/theory/QA_GATE0_VALIDATOR_EMPIRICAL.md` (N=127 cert retrospective, p≈0.0002) | established | — |

## §2. Property specification — mapping table

| Kochenderfer concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| Metric : behavior → ℝ; Specification : behavior → {true, false}; specifications derive from metric thresholds | `#val-3-1-metric-vs-specification` | QA certs return Boolean PASS/FAIL externally and ratio metrics internally (e.g. cert [194] uses `\|reachable_set\| / 81 ∈ {1/81, 8/81, 72/81}` as the metric, threshold = orbit-class match). | `qa_meta_validator.py` (Boolean output); per-cert telemetry (ratio metrics) | established | — |
| Reachability specification ψ = ◇R(s_t) — reach a target set of discrete states; ψ = ¬◇R(s_t) = □¬R(s_t) for avoidance | `#val-3-6-reachability-spec-formula` | QA reachability descent cert operationalizes exactly this on the QA orbit graph. State space = (b,e,d,a) tuples on mod-9 / mod-24; target set = orbit-class membership. | `qa_reachability_descent_run_cert_v1/{validator.py, mapping_protocol.json, schema.json}` | established | — |
| Reachability is the canonical form of model checking — LTL specs reduce to reachability via state-space augmentation (Büchi automaton) | `#val-3-6-reachability-as-canonical-form` | Cert [211] (Cayley-Bateson Filtration) explicitly reduces "Bateson learning hierarchy" (an LTL-shaped specification with strict containment levels) to "connected components of nested undirected Cayley graphs on S_9" — the QA equivalent of LTL→reachability via state-space augmentation, with augmentation = extending the generator set Γ. | `qa_alphageometry_ptolemy/qa_meta_validator.py::_validate_cayley_bateson_filtration_cert` (cert [211], depends on [191]) | established | — |
| Robustness metric ρ(τ, ψ) used as falsification objective; smooth softmin/softmax variants for gradient-based search | `#val-4-5-objective-function-falsification` | No QA-native scalar robustness metric currently exists. QA reachability descent returns `{PASS, FAIL, distance-to-target}` discretely; there is no smoothed-robustness analogue because softmin/softmax over a discrete orbit is malformed (Theorem NT firewall — float over orbit indices = T2-b). The natural QA-discrete robustness is integer path-length to nearest orbit-witness, but this is not currently exposed. | (none) | open | A `qa_discrete_robustness_cert_v1` claim could establish: "QA path-length-to-nearest-witness is a valid falsification objective for QA reachability specs, equivalent up to integer-rational order to Kochenderfer's robustness on the canonical embedding `(b,e) ↪ ℝ²`." Falsifiable; unique to QA. |

## §3. Failure distribution + probability estimation — mapping table

| Kochenderfer concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| Failure distribution `p(τ \| τ ∉ ψ) = 1{τ∉ψ}p(τ) / ∫1{τ∉ψ}p(τ)dτ`; denominator = `p_fail` (normalizing constant) | `#val-6-1-failure-distribution-conditional` | QA evaluates this exactly (not by sampling) on finite orbits. On `S_9` (size 81), `1{τ∉ψ}p(τ)` is enumerable and the integral becomes a finite sum with variance identically zero. The continuous-distribution sampling machinery (rejection sampling, MCMC, probabilistic programming) is replaced by deterministic enumeration on the QA-discrete side of the firewall — Kochenderfer's expensive case becomes QA's free case, in exchange for the discrete-domain restriction. **Validated 2026-04-27 by cert [263] `qa_failure_density_enumeration_cert_v1`**: reproduces cert [194]'s ratios `1/81`, `8/81`, `72/81` bit-exact via `tools/qa_kg/orbit_failure_enumeration.py`; head-to-head against Kochenderfer Algorithm 7.1 direct sampling at `N ∈ {100, 1000, 10000}` confirms the empirical estimator falls inside the `4σ = 4·sqrt(p(1−p)/N)` envelope from Kochenderfer eq. 7.3 while QA enumeration error is identically zero. Scope: mod-9 only in v1; mod-24 deferred until a canonical mod-24 orbit-family classifier lands. | cert [263] `qa_failure_density_enumeration_cert_v1` (anchor: cert [194] cognition-space morphospace `\|reachable_set\| / 81`); cert [191] tiered reachability (26% / 52.67% / 20% on S_9) is a candidate consumer of the enumeration utility | established (2026-04-27, mod-9) | A `qa_failure_density_enumeration_cert_v2` mod-24 extension would land once a published mod-24 orbit-family classifier exists. The orbit graph cardinality on mod-24 is 576 — still small enough for deterministic enumeration, so the variance-vs-Kochenderfer-sampling argument continues to hold. |
| Direct estimation MLE `p̂_fail = (1/m) Σ 1{τ_i ∉ ψ} = n/m` — n failures out of m samples; unbiased and consistent | `#val-7-1-direct-estimation-pfail` | Cert [194] `Agency = \|reachable_set\| / 81` is exactly this MLE form, except: (a) inverted (success rate not failure rate), and (b) deterministic enumeration replaces sampling, so variance is identically zero (no Bernoulli-variance term). | `qa_meta_validator.py::_validate_cognition_space_morphospace_cert` ([194]) | established | — |
| Importance sampling, adaptive IS, sequential Monte Carlo, multilevel splitting (rare-event estimators) | (Ch. 7 §7.2-§7.6, not anchored as quote — full chapter scope) | None — QA enumeration handles rare events directly. The Kochenderfer rare-event machinery is not rejected, but is not load-bearing in QA. | (none) | rejected | — (rare-event ML methods are continuous-domain by construction; QA enumeration dominates on the QA side of the firewall. Importing them would be reverse-direction mapping.) |

## §4. Reachability for discrete systems (Ch. 10) — mapping table

| Kochenderfer concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| Discrete system as directed graph: state = node, transition = edge; probabilistic edges → finite automaton / FSM | `#val-10-1-graph-formulation` | The QA orbit graph IS this directed graph. Nodes = (b,e,d,a) tuples; edges = QA generators (qa_step T, scalar multiplication, modulus reduction). On mod-9 the graph has 81 nodes; on mod-24 it has 576. | `qa_alphageometry_ptolemy/qa_meta_validator.py::_validate_cayley_bateson_filtration_cert` ([211] — explicit Cayley graph construction); orbit definitions in `CLAUDE.md` §Core QA Architecture | established | — |
| Forward reachability via BFS from initial set S_1 within horizon h; backward reachability via reverse-BFS from target set S_T within horizon h | `#val-10-2-forward-backward-reachable-sets` | Cert [191] tiered reachability theorem performs exhaustive forward BFS on S_9 from each starting tuple, computing the reachable-set cardinality at each generator-tier (L_1, L_2a, L_2b). Result: only 26% of the 6561 (s, s') pairs are L_1-reachable. | `qa_alphageometry_ptolemy/qa_meta_validator.py::_validate_bateson_learning_levels_cert` ([191]); `EXPECTED_TIER_COUNTS_S9` constants | established | — |
| Satisfiability = `S_T ∩ R_{1:h} ≠ ∅` (forward) or `S_1 ∩ B_{1:h} ≠ ∅` (backward) | `#val-10-3-satisfiability-via-intersection` | QA cert PASS/FAIL is exactly this set-intersection check. cert [185] (Keely Sympathetic Transfer) explicitly frames its `BLOCK` gate as "discord = reachability obstruction" = `S_T ∩ R_{1:h} = ∅`. | `qa_meta_validator.py::_validate_keely_sympathetic_transfer_cert` ([185]); reachability descent cert PASS condition | established | — |

## §5. Explainability + runtime monitoring — mapping table

| Kochenderfer concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| Failure-mode characterization via clustering of failure trajectories (k-means on extracted features) | `#val-11-6-failure-mode-clustering` | QA orbit classification IS structural clustering with discrete equivalence classes — but the centroid-based vocabulary is wrong for QA: QA uses exact orbit membership, not feature-space distance to a centroid. The Kochenderfer name for what QA does already is "clustering with structural (not feature-distance) similarity." | orbit classification: `_validate_cayley_bateson_filtration_cert` ([211]); `_validate_cognition_space_morphospace_cert` ([194]) | candidate | — (vocabulary adoption only; no new cert. Document in `docs/theory/QA_DIAGONAL_CLASSES_CATALOG.md` that QA orbit classification is the QA-native counterpart of k-means failure-mode clustering, with discrete equivalence classes replacing centroid distance.) |
| Aleatoric (output, irreducible, stochastic-input) vs epistemic (model, reducible, lack-of-data) uncertainty | `#val-12-2-aleatoric-vs-epistemic-uncertainty` | The Theorem NT firewall already separates these two: aleatoric = uncertainty in the observer projection at the input boundary (sensor noise on continuous inputs before discretization); epistemic = uncertainty in cert coverage at the output boundary (cert ecosystem gaps). QA does not yet use Kochenderfer's vocabulary, but the structural distinction is built in. | `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md` (input-boundary firewall = aleatoric); `docs/specs/QA_GATE0_VALIDATOR_EMPIRICAL.md` (cert-coverage gaps = epistemic) | candidate | — (vocabulary adoption: rename input-boundary uncertainty → "aleatoric / observer projection uncertainty" and cert-coverage gaps → "epistemic / cert-ecosystem uncertainty" in the relevant specs.) |
| Operational design domain (ODD) monitoring at runtime — flag inputs that fall outside the validated regime | (Ch. 12 §12.1, not anchored as quote — full chapter scope) | No QA artifact currently watches QA inputs at runtime to confirm they fall inside the orbit-class regime the cert was validated for. Cert validation is offline / fixture-based. The closest analogue is the qa-collab bus broadcast on cert FAIL, but that's reactive not predictive. | (none) | open | A `qa_runtime_odd_monitor_cert_v1` claim could establish: "Runtime monitoring of QA tuple inputs against a declared 'orbit-class ODD' detects outside-validated-regime inputs at zero variance (deterministic membership check), strictly dominating Kochenderfer §12.1's classifier-superlevel-set ODD on the QA-discrete side of the firewall." Falsifiable. |
| Counterfactual explanations (Ch. 11 §11.5) — "smallest input change that flips the output" | (not anchored as quote — chapter scope) | No QA-native counterfactual machinery yet. QA could compute counterfactuals via reachability descent: "smallest generator-path that moves a tuple from PASS-orbit to FAIL-orbit." | (none) | open | A `qa_counterfactual_descent_cert_v1` claim could establish: "Counterfactual explanations for QA PASS/FAIL decisions are computed exactly by Dijkstra-shortest-path on the orbit graph with edge weight = generator-cost." Falsifiable. |

## §6. Sections of *Algorithms for Validation* not mapped here (Validation book scope)

For honesty: these chapters were not anchored in the v1 ingest because they are continuous-domain and cross the Theorem NT firewall. They become candidates only as observer projections (input boundary) or as off-QA baselines for empirical comparison.

- Ch. 2 — System Modeling (probability, parameter learning, agent models). Continuous domain; observer-projection candidates only.
- Ch. 5 — Falsification through Planning (shooting, tree search, heuristic search, MCTS, RL). Off-QA baselines if a future cert wants empirical comparison.
- Ch. 8 — Reachability for Linear Systems. Continuous domain; firewall-rejected as causal input.
- Ch. 9 — Reachability for Nonlinear Systems (interval arithmetic, Taylor models, optimization-based reachability, neural networks). Continuous domain; firewall-rejected as causal input.
- Ch. 11 §11.1-§11.5 — Policy visualization, feature importance, surrogate models, counterfactual explanations. Mostly continuous; counterfactual is a candidate (see §5 above).
- Ch. 12 §12.1, §12.3 — ODD monitoring, failure monitoring. Anchored conceptually in §5 above; specific runtime-monitor cert is open.

---

## §7. Decision Making bridge — added 2026-04-27

This section was added when *Algorithms for Decision Making* (Kochenderfer, 2022) ingested into QA-MEM. Mirrors the §1-§5 structure of the Validation bridge: status-coded mapping rows + future sharp-claim cert candidates. Same scope discipline applies (no taxonomy certs).

### §7.1 — MDP foundation (the QA-rich chapter)

| Kochenderfer (DM 2022) concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| MDP definition: `(S, A, T(s'\|s,a), R(s,a), γ)`, Markov assumption (next state depends only on current state + action) | `#dm-7-1-mdp-definition` | The QA orbit graph IS a deterministic MDP: state space `S = {1..m}²`, action space `A = {qa_step}` (single deterministic generator on the QA-discrete side), transition `T(s'\|s,a) = δ(s', qa_step(s,a))` (no randomness — Theorem NT firewall: stochasticity enters only as observer projection at the input boundary, never as causal QA dynamics), reward = orbit-class membership indicator `1{s ∈ ψ}`. | orbit-graph: `_validate_cayley_bateson_filtration_cert` ([211]); `_validate_bateson_learning_levels_cert` ([191]); `qa_reachability_descent_run_cert_v1` | established | — |
| Bellman expectation equation `U^π(s) = R(s,π(s)) + γ Σ T(s'\|s,π(s)) U^π(s')`; convergence by contraction mapping; `O(\|S\|³)` direct solution via linear algebra | `#dm-7-2-bellman-expectation-equation`, `#dm-7-2-policy-evaluation-lookahead-equation` | On the QA orbit graph with deterministic T, the Bellman expectation equation collapses to `U^π(s) = R(s, π(s)) + γ U^π(qa_step(s, π(s)))` — a single-successor recursion. Convergence is bit-exact in `O(orbit_period(s))` integer steps; the contraction-mapping argument is unnecessary because there's no fixed-point iteration on a continuous space, just orbit traversal. | implicit in cert [191] tier-classification (each tier-i target is reachable in exactly i generator steps); cert [263] enumerates the analogous `p_fail = E[1{τ ∉ ψ}]` over the same finite graph | candidate | — (vocabulary alignment: rename "tier-i reachability" → "Bellman-backup-i value" in the [191] family doc to mirror Kochenderfer's terminology, no new cert needed) |
| Bellman backup / value iteration `U_{k+1}(s) = max_a [ R(s,a) + γ Σ T(s'\|s,a) U_k(s') ]`; converges to optimal `U*` satisfying Bellman optimality equation | `#dm-7-5-value-iteration-bellman-backup` | On the QA orbit graph the `max_a` is a no-op (single deterministic generator), so value iteration on QA = forward-orbit traversal. The QA reachability descent cert `qa_reachability_descent_run_cert_v1` operationalizes exactly this for the `R(s,a) = 1{s ∈ ψ}` reward, with the integer-path-length witness as the `U*` analog. | `qa_reachability_descent_run_cert_v1/{validator.py, mapping_protocol.json, fixtures/}` | established | — |
| Linear-program formulation: `min Σ U(s) s.t. U(s) ≥ R(s,a) + γ Σ T(s'\|s,a) U(s')` for all s,a; `\|S\|` variables, `\|S\|·\|A\|` constraints; polynomial-time by Khachiyan 1980 | `#dm-7-7-linear-program-formulation` | Not directly used by QA — the deterministic single-generator structure makes LP overkill. However, the LP framework is useful for *non-canonical* QA generators (e.g., the L_2a / L_2b / L_3 operator classes from cert [191]) where multiple actions exist; cert [191] already exhaustively enumerates all 81×81 pairs, which is the LP's all-constraints case. | (none beyond [191] enumeration) | rejected | — (LP is dominated by cert [191]'s enumeration on the QA-discrete side; importing LP machinery would add no falsifiable claim) |

### §7.2 — Online planning

| Kochenderfer (DM 2022) concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| Forward search: depth-d expansion of all transitions, search tree with worst-case branching `\|S\|·\|A\|`, complexity `O((\|S\|·\|A\|)^d)` | `#dm-9-3-forward-search` | The QA forward-reachability BFS at cert [191] performs *exactly this* on `S_9` for `\|A\|=1` (single deterministic generator), so the worst-case `O(\|S\|^d)` collapses to `O(\|S\|·d)` linear time. Result: 81+1712+3456+1312 = 6561 (s,s') pairs classified exhaustively at d=4. | `_validate_bateson_learning_levels_cert` ([191]) — uses the utility from cert [263] for the orbit-class enumeration | established | — |
| Branch and bound: prune subtrees via `Q(s,a)_hi < Q(s,a*)_lo`; same worst-case as forward search but better pruned | `#dm-9-4-branch-and-bound` | No QA-native branch-and-bound implementation. The deterministic single-generator structure makes pruning trivial (each subtree is a single linear orbit chain), so QA enumeration is already optimal — branch-and-bound machinery would yield no speedup. For multi-generator settings (L_2a / L_2b / L_3 from [191]), branch-and-bound on `Q_hi`/`Q_lo` from the cert-[263] enumeration utility *would* prune productively, but no concrete QA cert claim hinges on this. | (none) | rejected | — (no falsifiable claim where QA branch-and-bound dominates QA enumeration; Kochenderfer's pruning advantage assumes stochastic transitions which QA-discrete doesn't have) |

### §7.3 — POMDP / partial observability

| Kochenderfer (DM 2022) concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| POMDP observation function `O(o\|a,s')` and discrete state filter for finite spaces — recursive Bayesian estimation produces categorical belief vector `b(s)` of length `\|S\|`; `B ⊂ ℝ^\|S\|` is the probability simplex | `#dm-19-2-discrete-state-filter` | Direct map to Theorem NT input-boundary observer projection: continuous sensor readings → discrete-orbit-class belief (pmf over `{singularity, satellite, cosmos}` on `S_9`). The categorical-pmf representation is finite-dimensional and respects the firewall — `b(s) ∈ {0, 1/81, …, 81/81}` rationals only on the QA-discrete side. cert [259] qa_heartmath_coherence_cert already does this informally for cardiac-rhythm orbit-class labels under continuous HRV input. | `_validate_qa_heartmath_coherence_cert` ([259]); `qa_kg/observer_projection_*` predicates (Theorem NT firewall enforcement) | candidate | — (vocabulary alignment: re-frame [259]'s "orbit-class label assignment" as "QA-side discrete state filter" with reference to Kochenderfer §19.2; no new cert) |
| Belief-state MDP — POMDP-to-MDP reduction with continuous belief simplex as state space; `R(b,a) = Σ R(s,a) b(s)` | `#dm-20-1-belief-state-mdp` | The continuous belief simplex is firewall-rejected as causal QA state (Theorem NT — float QA state = T2-b violation). However, the belief-pmf-as-rational-vector projection respects the firewall: instead of continuous belief simplex `b ∈ ℝ^\|S\|`, QA uses the *discrete pmf lattice* `b ∈ ℚ^\|S\|` with denominators bounded by the orbit-graph cardinality. cert [263]'s `Fraction`-based exact ratios are the correct primitive. | (no QA-specific belief-state cert yet) | open | A `qa_belief_state_lattice_cert_v1` claim could establish: "For QA-discrete POMDPs over orbit-class state spaces, the rational-pmf belief lattice `ℚ^\|S\|_d` (denominators ≤ d) is closed under recursive Bayesian update with a deterministic generator — no continuous belief simplex needed." Falsifiable on small-orbit POMDPs. |
| POMDP value iteration via alpha-vector pruning of conditional plans; expand-then-prune to horizon h | `#dm-20-5-pomdp-value-iteration` | No QA artifact yet. The conditional-plan tree representation maps onto QA generator-sequence enumeration; the alpha-vector pruning maps onto orbit-class equivalence pruning. | (none) | open | A `qa_alpha_vector_orbit_pruning_cert_v1` claim could establish: "QA orbit-class equivalence prunes the conditional-plan tree at exactly the same rate as Kochenderfer alpha-vector pruning on the rational-pmf belief lattice." Falsifiable on small finite-horizon POMDPs. |

### §7.4 — Multi-agent extensions

| Kochenderfer (DM 2022) concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| Markov game: shared state `s ∈ S`, joint action transitions `T(s'\|s, a)`, per-agent reward `R_i(s, a)`; per-agent Bellman with joint policy | `#dm-25-1-markov-games` | No multi-agent QA cert exists. The structural map is: each agent is an `(b_i, e_i)` orbit position; joint state is the tuple `(s_1, ..., s_n)` ∈ `S^n`; deterministic per-agent generator means joint transition is a product of single-agent transitions. The product structure makes the QA multi-agent cardinality `81^n` — manageable for small `n`. | (none) | open | A `qa_multi_agent_orbit_product_cert_v1` claim could establish: "Joint QA orbits on `S^n` for `n` agents factor into per-agent orbits when generators commute; non-commutative cases yield novel coupled orbit families." Falsifiable on `n=2,3` cases at `m=9`. |
| Dec-POMDP — collaborative agents, shared reward, local observations; NEXP-complete for finite horizons | `#dm-27-1-dec-pomdp` | The NEXP-completeness comes from belief-coordination across agents under partial observability. On the QA-discrete side the hard part is the *belief-coordination*; per-agent policies with shared reward over orbit-class targets are still tractable when the joint state space is `81^n`. | (none) | open | A `qa_dec_pomdp_orbit_coordination_cert_v1` claim could establish: "QA Dec-POMDPs with reward-independence (per Kochenderfer's reward-decomposition condition §27.2) admit polynomial-time joint-policy enumeration on `S^n` for orbit-class targets — the Kochenderfer P-complete subclass." Falsifiable. |

### §7.5 — Foundation rows (book framing, not algorithm-specific)

| Kochenderfer (DM 2022) concept | Source anchor | QA counterpart | Current QA artifact | Status | Future cert claim |
|---|---|---|---|---|---|
| Decision-making framing: agent + environment + observe-act loop | `#dm-1-1-decision-making-observe-act-cycle` | The QA observer-projection firewall (Theorem NT) IS this loop with the projection direction declared at every boundary crossing. cert [257] qa_integer_state_pipeline already enforces "exactly two boundary crossings" as a structural invariant. | `_validate_integer_state_pipeline_cert` ([257]) | established | — |
| Four uncertainty types: outcome, model, state, interaction — book is organized around these | `#dm-1-1-four-uncertainties` | Theorem NT partitions causality across the firewall: outcome uncertainty = stochastic input projection; model uncertainty = cert-coverage gaps; state uncertainty = belief-pmf rational lattice (see §7.3); interaction uncertainty = multi-agent (see §7.4). The four-axis partition is structurally compatible with QA's existing cert ecosystem boundary discipline. | (book-level structural mapping, not a single cert) | candidate | — (documentation alignment: reference Kochenderfer's four-uncertainty axis in `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md` as the canonical external taxonomy; no new cert) |
| Maximum expected utility principle: rational agent chooses `a* = arg max_a Σ P(s'\|a,o) U(s')` | `#dm-6-4-maximum-expected-utility` | On the QA-discrete side with deterministic transitions, MEU collapses to `a* = arg max_a U(qa_step(s, a))` — a one-step lookahead. With multi-generator action spaces (L_2a / L_2b / L_3 from cert [191]), MEU recovers the canonical max-over-actions form. | implicit in cert [191] tier-classification (always-pick-cheapest-tier policy) | candidate | — (vocabulary alignment with [191] family doc) |

### §7.6 — Sections of *Algorithms for Decision Making* not mapped here (DM book scope)

For honesty: these chapters were not anchored because they are continuous-domain or off-QA baselines that cross the firewall.

- Part I §2-§5 — Bayesian networks, inference, parameter learning, structure learning. Continuous-domain probabilistic graphical models; observer-projection candidates only.
- Part II §8 — Approximate Value Functions (parametric, neural network, kernel smoothing). Continuous-domain function approximators; firewall-rejected as QA causal state.
- Part II §10-§12 — Policy Search, Policy Gradient, Policy Gradient Optimization (gradient ascent, restricted/natural gradient, trust region, clamped surrogate). Continuous-domain; firewall-rejected.
- Part II §13 — Actor-Critic methods (deep RL). Continuous-domain; off-QA baseline.
- Part III §16-§18 — Exploration/Exploitation, Model-Based Methods, Model-Free Methods (Q-Learning, SARSA, eligibility traces). Continuous-domain on QA-discrete side; observer-projection only at input boundary if used.
- Part III §19 — Imitation Learning (IRL, GAIL). Continuous-domain; off-QA baseline.
- Part IV §19.3-§19.5 — Kalman filter, Extended/Unscented KF, Particle filter. Continuous belief representations; firewall-rejected (only the discrete state filter §19.2 maps to QA).
- Part IV §21-§22 — Offline/Online Belief State Planning (point-based VI, MCTS over belief). Continuous belief simplex; observer-projection at input boundary only.
- Part V §23-§24 — Multiagent Reasoning (Nash equilibrium, fictitious play in simple games). Foundational but non-sequential; bridge §7.4 already covers the sequential variants.

### §7.7 — Open sharp-claim certs from §7

Listed for the "no taxonomy cert, only sharp empirical claims" rule:
- `qa_belief_state_lattice_cert_v1` — rational-pmf belief lattice on QA orbit graphs.
- `qa_alpha_vector_orbit_pruning_cert_v1` — orbit-class equivalence pruning vs Kochenderfer alpha-vector pruning.
- `qa_multi_agent_orbit_product_cert_v1` — joint orbits on `S^n` factor when generators commute.
- `qa_dec_pomdp_orbit_coordination_cert_v1` — polynomial-time Dec-POMDP joint-policy enumeration under reward independence.

Each is independent; none built yet. None requires the others. Same discipline as §1-§5 candidates.

---

## Standing rules

1. **No taxonomy cert** — there will be no `qa_kochenderfer_bridge_cert_v1` or similar that "validates the mapping" by classification. Such a cert would PASS by construction and prove nothing. (Per Will's directive 2026-04-26.)
2. **One cert per sharp claim** — future Kochenderfer-derived certs land *only* when they make a falsifiable empirical claim that compares QA's discrete machinery against Kochenderfer's continuous machinery on a benchmark. Listed candidates above. **First sharp claim landed 2026-04-27**: cert [263] `qa_failure_density_enumeration_cert_v1` — see §3 row 1. Remaining candidates: `qa_discrete_robustness_cert_v1`, `qa_runtime_odd_monitor_cert_v1`, `qa_counterfactual_descent_cert_v1`.
3. **Vocabulary alignment is preferred over new artifacts** — for `candidate`-status rows, the next move is a docs-only edit to the named QA artifact (rename a section, add a Kochenderfer cross-reference, update terminology). No new cert family.
4. **This document evolves additively** — when subsequent ingestions land (`dm.pdf`, `optimization.pdf`), extend this bridge with §7+ sections rather than replacing the existing tables.

---

## References / Provenance

- Kochenderfer, M. J., Wheeler, T. A., Katz, S., Corso, A., & Moss, R. J. (2026). *Algorithms for Validation*. MIT Press. CC-BY-NC-ND. ISBN forthcoming. Companion repos: github.com/algorithmsbooks/{validation, validation-code, validation-ancillaries, validation-figures}.
- Kochenderfer, M. J., & Wheeler, T. A. (2019). *Algorithms for Optimization*. MIT Press. (queued for ingestion)
- Kochenderfer, M. J., Wheeler, T. A., & Wray, K. H. (2022). *Algorithms for Decision Making*. MIT Press. (queued for ingestion)
- Reason, J. (2000). Human Error: Models and Management. *British Medical Journal*, 320(7237), 768–770. (Swiss cheese model citation — used in Kochenderfer §1.4)
- Christian, B. (2020). *The Alignment Problem: Machine Learning and Human Values*. W. W. Norton & Company. (alignment-problem citation — used in Kochenderfer §1.1)
- Bateson, G. (1972). *Steps to an Ecology of Mind*. (cited by QA cert [191] for learning-level filtration)

| Artifact | Path | Rev |
|---|---|---|
| Source PDF | `Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_validation.pdf` | 13.9 MB; staged 2026-04-26 |
| Verbatim excerpts | `docs/theory/kochenderfer_validation_excerpts.md` | 15 anchors |
| QA-MEM fixture (Validation) | `tools/qa_kg/fixtures/source_claims_kochenderfer.json` | 1 SourceWork + 15 SourceClaims |
| Source PDF (Decision Making) | `Documents/kochenderfer_corpus/kochenderfer_wheeler_wray_2022_algorithms_for_decision_making.pdf` | 12.1 MB; staged 2026-04-27 |
| Verbatim excerpts (Decision Making) | `docs/theory/kochenderfer_decision_making_excerpts.md` | 15 anchors |
| QA-MEM fixture (Decision Making) | `tools/qa_kg/fixtures/source_claims_dm.json` | 1 SourceWork + 15 SourceClaims |
| Corpus index entry | `tools/qa_kg/CORPUS_INDEX.md` | section "Kochenderfer / algorithmsbooks corpus" |
| This bridge | `docs/specs/QA_KOCHENDERFER_BRIDGE.md` | v2, §7 added 2026-04-27 |
