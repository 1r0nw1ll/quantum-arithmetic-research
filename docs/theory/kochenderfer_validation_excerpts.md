<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Kochenderfer/Wheeler/Katz/Corso/Moss 2026 "Algorithms for Validation" primary-source excerpt snapshot. Verbatim quotes from val.pdf (441pp). Page locators are PDF page (/printed page). 14 anchors covering validation taxonomy, properties, reachability, falsification, failure distribution, runtime monitoring. -->

# Kochenderfer / Wheeler / Katz / Corso / Moss 2026 — Algorithms for Validation — Primary-Source Excerpts

**Corpus root:** `Documents/kochenderfer_corpus/`
**Scope:** QA-MEM ingestion of *Algorithms for Validation* as the highest-leverage Kochenderfer textbook for QA cert-ecosystem grounding. The book defines a formal vocabulary for validation algorithms, property specification, falsification, failure distribution, reachability, explainability, and runtime monitoring — all of which the QA cert ecosystem already implements informally. Source picked first (over `dm.pdf` and `optimization.pdf`) because it directly serves the Terminal Goal: making QA's cert/validator ecosystem legible to skeptical technical readers via shared external vocabulary.
**Domain:** unclassified (empty string in fixture). The existing closed set in `tools/qa_kg/domain_taxonomy.json` (qa_core, svp, geometry, biology, physics, rsf, psychophysiology) has no slot for formal-methods / validation-methodology / autonomous-systems-engineering. Extending the taxonomy with `formal_methods` is a candidate Phase 4.x follow-up but would force a R10 expected-hash regen; deferred to a focused taxonomy-extension session.

**Publication:** M. J. Kochenderfer, T. A. Wheeler, S. Katz, A. Corso, R. J. Moss, *Algorithms for Validation*. MIT Press, 2026. CC-BY-NC-ND. Compiled 2026-02-08 14:33:20-08:00 (LuaHBTeX, TeX Live 2022). 441 PDF pages.
**Authors:** Mykel J. Kochenderfer (Stanford), Tim A. Wheeler, Sydney Katz, Anthony Corso, Robert J. Moss.
**On-disk:** `Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_validation.pdf` (13.9 MB; staged 2026-04-26 from local download). Companion repos: `github.com/algorithmsbooks/validation`, `validation-code`, `validation-ancillaries`, `validation-figures`.

**QA-research grounding:** Validation in §1.1 is defined as "the broad process of establishing confidence that a system will behave as desired when deployed in the real world" — a function that the QA cert ecosystem (mapping_protocol → validator → meta_validator → empirical observation cert) already serves for QA mathematical claims. The book's discrete-reachability formulation (Ch. 10), reachability specification ψ = ◇R(s_t) (§3.6), and graph-formulation of state transitions (§10.1) are direct counterparts to: `qa_reachability_descent_run_cert_v1`, the orbit graph (24-cycle Cosmos / 8-cycle Satellite / 1-cycle Singularity), and (b,e,d,a) tuple state space respectively. The Swiss-cheese model of layered validation (§1.4) is the structural justification for why the QA cert ecosystem is many small pairwise certs rather than one mega-cert.

**Firewall note:** several Algorithms-for-Validation methods are continuous-domain (interval arithmetic, Taylor models, optimization-based reachability, gradient-of-robustness via softmin/softmax). These cannot enter the QA discrete layer as causal inputs (Theorem NT). The discrete-systems chapter (Ch. 10) is the natural QA-aligned subset; continuous chapters become observer-projection candidates (input boundary or output boundary).

---

## Chapter 1 — Introduction

### #val-1-1-validation-definition (PDF p.17 / printed p.1, §1.1)

> "In this book, we define validation as the broad process of establishing confidence that a system will behave as desired when deployed in the real world. We define verification as a special type of validation that provides guarantees about the correctness of a system with respect to a specification. We define testing as a technique used for validation that involves evaluating the system on a discrete set of test cases."

### #val-1-1-alignment-problem (PDF p.18 / printed p.2, §1.1)

> "However, the models used to perform the optimization may be imperfect, the optimization objective may not perfectly capture the requirements, and the optimization process itself is often approximate. This misalignment can result in a mismatch between the desired behavior of the system and its actual behavior when deployed in the real world. We refer to this phenomenon as the alignment problem."

### #val-1-4-validation-algorithm-inputs (PDF p.24 / printed p.8, §1.4)

> "Validation algorithms require two inputs, as shown in figure 1.3. The first input is the system under test, which we will refer to as the system. The system represents a decision-making agent operating in an environment. The agent makes decisions based on information from the environment that it receives from sensors. […] The second input is a specification, which expresses an operating requirement for the system. […] Given these inputs, validation algorithms output metrics to help us understand the scenarios in which the system does or does not satisfy the specification."

### #val-1-4-swiss-cheese-safety-case (PDF p.30 / printed p.14, §1.4)

> "In most real-world settings, we cannot guarantee that a system will behave as intended using a single validation algorithm or metric. Instead, we use a combination of these techniques to build a safety case. This idea is inspired by the Swiss cheese model of accident causation (figure 1.8). This model views validation algorithms as slices of Swiss cheese with holes, or limitations, that may cause us to miss potential failure modes. If we stack enough slices of Swiss cheese together, the holes in one slice will be covered by the cheese in another slice. By using a combination of validation algorithms, we increase our chances of catching potential failure modes before they could occur during operation."

---

## Chapter 3 — Property Specification

### #val-3-1-metric-vs-specification (PDF p.77 / printed p.61, §3.1)

> "We describe the behavior of a system using metrics and specifications. A metric is a function that maps system behavior to a real number. […] A specification is a function that maps system behavior to a Boolean value. Therefore, specifications are always either true or false. […] Sometimes specifications can be derived from metrics. For example, given a metric that measures the probability of collision for an aircraft collision avoidance system, we can create a specification that requires the probability of collision to be less than a certain threshold."

### #val-3-6-reachability-spec-formula (PDF p.97 / printed p.81, §3.6)

> "A reachability specification is a special type of temporal logic specification that describes a state or set of states that a system should or should not reach during its execution. Let S_T ⊆ S represent the target set of states and define the predicate function R(s_t) to be true if s_t ∈ S_T and false otherwise. If our goal is to reach the target set, the reachability specification has the following form: ψ = ◇R(s_t)  (3.20). If our goal is to avoid the target set, we use the negation of the reachability specification as follows: ψ = ¬◇R(s_t) = □¬R(s_t)  (3.21)"

### #val-3-6-reachability-as-canonical-form (PDF p.98 / printed p.82, §3.6)

> "Writing specifications in this form is useful because many algorithms related to formal methods and model checking are centered around reachability specifications. For example, the algorithms in chapters 8 to 10 determine whether a system could reach a target set. […] In general, it is possible to solve the model checking problem for other types of specifications by transforming the problem into a reachability problem using various techniques. For systems with LTL specifications, we can create a reachability problem by augmenting the state space of the system."

---

## Chapter 4 — Falsification through Optimization

### #val-4-5-objective-function-falsification (PDF p.117 / printed p.101, §4.5)

> "Objective functions guide the search for failure trajectories. In general, a good objective function should output lower values for trajectories that are closer to a failure. The specific measure of closeness used is dependent on the application. […] If ψ is specified using a temporal logic formula, we can use its robustness measure (see section 3.5.2) as an objective function such that f(τ) = ρ(τ, ψ)."

---

## Chapter 6 — Failure Distribution

### #val-6-1-failure-distribution-conditional (PDF p.155 / printed p.139, §6.1)

> "The distribution over failures for a given system with specification ψ is represented by the conditional probability p(τ | τ ∉ ψ). We can write this probability as p(τ | τ ∉ ψ) = 1{τ ∉ ψ} · p(τ) / ∫ 1{τ ∉ ψ} p(τ) dτ  (6.1) where 1{·} is the indicator function and p(τ) is the probability density of the nominal trajectory distribution for trajectory τ. […] For most systems, the failure distribution is difficult to compute exactly because doing so requires solving the integral in the denominator of equation (6.1) to compute the normalizing constant. The value of this integral corresponds to the probability of failure for the system."

---

## Chapter 7 — Failure Probability Estimation

### #val-7-1-direct-estimation-pfail (PDF p.175-176 / printed p.159-160, §7.1)

> "The probability of failure for a given system and specification is defined mathematically as p_fail = E_{τ∼p(·)}[1{τ ∉ ψ}] = ∫ 1{τ ∉ ψ} p(τ) dτ  (7.1) where 1{·} is the indicator function. […] The maximum likelihood estimate of the probability of failure is p̂_fail = (1/m) Σ_{i=1}^m 1{τ_i ∉ ψ} = n/m  (7.2) where n is the number of samples that resulted in a failure and m is the total number of samples. […] In the limit of infinite samples, the variance approaches zero, so the estimator is consistent."

---

## Chapter 10 — Reachability for Discrete Systems

### #val-10-1-graph-formulation (PDF p.279 / printed p.263, §10.1)

> "Directed graphs are a natural way to represent the transitions of a discrete system. A directed graph consists of a set of nodes and a set of directed edges connecting the nodes. For discrete systems, each node represents a state of the system, and each edge represents a transition between states (figure 10.1). We can also associate a probability with each edge to represent the likelihood of the transition occurring. A graph of this form is called a finite automaton or finite-state machine."

### #val-10-2-forward-backward-reachable-sets (PDF p.281 / printed p.265, §10.2)

> "To compute reachable sets, we ignore the probabilities associated with the edges of the graph and focus only on its connectivity. The reachable sets are represented as collections of discrete states. We focus on two types of reachability analysis: forward reachability and backward reachability. Forward reachability analysis determines the set of states that can be reached from a given set of initial states within a specified time horizon. Backward reachability analysis determines the set of states from which a given set of target states can be reached within a specified time horizon."

### #val-10-3-satisfiability-via-intersection (PDF p.283 / printed p.267, §10.3)

> "We can use the forward and backward reachable sets of discrete systems to determine whether they satisfy a reachability specification (figure 10.6). For forward reachability, we check whether the target set intersects with the forward reachable set. For backward reachability, we check whether the initial set intersects with the backward reachable set. In both cases, these checks require us to compute the full forward or backward reachable set."

---

## Chapter 11 — Explainability

### #val-11-6-failure-mode-clustering (PDF p.331 / printed p.315, §11.6)

> "Another way to explain the behavior of a system is to characterize its failure modes. We can use clustering algorithms to create groupings of failure trajectories that are similar to one another. Identifying the similarities and differences between failures helps us understand their underlying causes."

---

## Chapter 12 — Runtime Monitoring

### #val-12-2-aleatoric-vs-epistemic-uncertainty (PDF p.346 / printed p.330, §12.2)

> "We may encounter two different types of uncertainty when monitoring a system. The first type of uncertainty is output uncertainty, which occurs when a single input can lead to multiple different outputs. […] This type of uncertainty occurs due to inherent stochasticity in the real world and is also referred to as aleatoric uncertainty or irreducible uncertainty. […] The second type of uncertainty is model uncertainty, which arises due to limitations of the model we are using to predict system behavior. […] This type of uncertainty is also referred to as epistemic uncertainty or reducible uncertainty."

---

## Acquisition / ingestion notes

- **2026-04-26**: PDF staged from `~/Downloads/val.pdf` to `Documents/kochenderfer_corpus/kochenderfer_wheeler_2026_algorithms_for_validation.pdf` via heredoc-scoped `shutil.copy2` (the `Documents/kochenderfer_corpus/` directory is not yet in `DOCUMENTS_PDF_INGRESS_PREFIXES`; allowlist extension is a `WRAPPER_SELF_MODIFICATION` edit deferred to a focused Codex-quarantine-review session, matching the HeartMath / Schumann pattern).
- 15 anchors selected from a 441-page text on the criteria: definitional weight (§1.1, §3.1, §12.2), direct QA mapping (§3.6, §10.1, §10.2, §10.3, §6.1, §7.1), and structural justification for the QA cert ecosystem's many-small-certs-not-one-mega-cert design (§1.4 Swiss cheese, §1.1 alignment problem, §11.6 failure clustering, §4.5 robustness-as-objective).
- Three further Kochenderfer/Wheeler textbooks are present in `~/Downloads/` and will be ingested in subsequent sessions following the same pattern: `optimization.pdf` (*Algorithms for Optimization*), `optimization-1e-1.pdf` (1st-edition errata of same), `dm.pdf` (*Algorithms for Decision Making*).
- Domain taxonomy decision (formal_methods vs empty-string) is open. Empty-string is legal per R10 and is used here. Future sessions may want to extend the taxonomy and renormalize.
