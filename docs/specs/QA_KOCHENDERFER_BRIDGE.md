<!-- PRIMARY-SOURCE-EXEMPT: reason="QA ↔ Kochenderfer 2026 controlled mapping index. Source citations + DOI live below in §Provenance and per-row source-anchor links to docs/theory/kochenderfer_validation_excerpts.md (15 verbatim anchors). Companion fixture: tools/qa_kg/fixtures/source_claims_kochenderfer.json. Bridge spec, not a research claim doc." -->

# QA ↔ Kochenderfer/Wheeler Bridge — Controlled Mapping Index

**Source corpus:** Kochenderfer, Wheeler, Katz, Corso, Moss (2026). *Algorithms for Validation*. MIT Press. CC-BY-NC-ND. 441 pp. Companion textbooks (queued for future ingestion): *Algorithms for Optimization* (Kochenderfer + Wheeler 2019) and *Algorithms for Decision Making* (Kochenderfer + Wheeler + Wray 2022).

**On-disk:** `Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_validation.pdf`. Excerpts: `docs/theory/kochenderfer_validation_excerpts.md` (15 verbatim anchors). Fixture: `tools/qa_kg/fixtures/source_claims_kochenderfer.json` (1 SourceWork, 15 SourceClaims).

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
| Failure distribution `p(τ \| τ ∉ ψ) = 1{τ∉ψ}p(τ) / ∫1{τ∉ψ}p(τ)dτ`; denominator = `p_fail` (normalizing constant) | `#val-6-1-failure-distribution-conditional` | QA evaluates this exactly (not by sampling) for small orbits. Over the full orbit graph (size 81 on mod-9, 576 on mod-24), `1{τ∉ψ}p(τ)` is enumerable; the integral becomes a finite sum. The continuous-distribution sampling machinery (rejection sampling, MCMC, probabilistic programming) is replaced by deterministic enumeration — Kochenderfer's expensive case becomes QA's free case, in exchange for the discrete-domain restriction. | cert [194] cognition-space morphospace (`\|reachable_set\| / 81`); cert [191] tiered reachability (26% / 52.67% / 20% on S_9) | candidate | A `qa_failure_density_enumeration_cert_v1` claim could establish: "For QA reachability specifications on mod-9 / mod-24 orbits, deterministic enumeration of `1{τ∉ψ}` over the finite state space yields exact `p_fail`, avoiding the importance-sampling / SMC machinery of Kochenderfer Ch. 7. Specifically: enumeration scales as O(\|S\|·k) where \|S\| ≤ 576 and k = path horizon, while Kochenderfer's importance sampling has unbounded variance in the tail." Falsifiable. |
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

## §6. Sections of *Algorithms for Validation* not mapped here

For honesty: these chapters were not anchored in the v1 ingest because they are continuous-domain and cross the Theorem NT firewall. They become candidates only as observer projections (input boundary) or as off-QA baselines for empirical comparison.

- Ch. 2 — System Modeling (probability, parameter learning, agent models). Continuous domain; observer-projection candidates only.
- Ch. 5 — Falsification through Planning (shooting, tree search, heuristic search, MCTS, RL). Off-QA baselines if a future cert wants empirical comparison.
- Ch. 8 — Reachability for Linear Systems. Continuous domain; firewall-rejected as causal input.
- Ch. 9 — Reachability for Nonlinear Systems (interval arithmetic, Taylor models, optimization-based reachability, neural networks). Continuous domain; firewall-rejected as causal input.
- Ch. 11 §11.1-§11.5 — Policy visualization, feature importance, surrogate models, counterfactual explanations. Mostly continuous; counterfactual is a candidate (see §5 above).
- Ch. 12 §12.1, §12.3 — ODD monitoring, failure monitoring. Anchored conceptually in §5 above; specific runtime-monitor cert is open.

---

## Standing rules

1. **No taxonomy cert** — there will be no `qa_kochenderfer_bridge_cert_v1` or similar that "validates the mapping" by classification. Such a cert would PASS by construction and prove nothing. (Per Will's directive 2026-04-26.)
2. **One cert per sharp claim** — future Kochenderfer-derived certs land *only* when they make a falsifiable empirical claim that compares QA's discrete machinery against Kochenderfer's continuous machinery on a benchmark. Listed candidates above (`qa_discrete_robustness_cert_v1`, `qa_failure_density_enumeration_cert_v1`, `qa_runtime_odd_monitor_cert_v1`, `qa_counterfactual_descent_cert_v1`).
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
| QA-MEM fixture | `tools/qa_kg/fixtures/source_claims_kochenderfer.json` | 1 SourceWork + 15 SourceClaims |
| Corpus index entry | `tools/qa_kg/CORPUS_INDEX.md` | section "Kochenderfer / algorithmsbooks corpus" |
| This bridge | `docs/specs/QA_KOCHENDERFER_BRIDGE.md` | v1, 2026-04-26 |
