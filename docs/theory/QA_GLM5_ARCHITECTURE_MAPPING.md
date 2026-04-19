# QA ↔ GLM-5: Failure-Mode Mapping and QA-Native Resolution

**Source:** Zhipu AI & Tsinghua, *GLM-5: from Vibe Coding to Agentic Engineering*, arXiv 2602.15763 (Feb 2026).
**Status:** Mapping doc + QA-native architecture sketch. Cert scaffolds target the QA-native model, not GLM-5's RL pipeline.
**Date:** 2026-04-19.

## Framing

The contribution is not "QA certifies GLM-5's RL system." QA does not do RL — RL is a continuous-optimization construct (parameterized distributions, stochastic gradient descent, float weight updates, policy staleness). Importing RL machinery into QA under a cert banner is the retrofit error.

The contribution is the **opposite direction**: GLM-5's engineering findings are empirical evidence that any system embedding discrete structure in a continuous learning loop fails at specific, predictable loci. QA predicts these loci a priori from Theorem NT, A1, and T1. More importantly, QA enables a **native alternative architecture** in which these failure modes are structurally impossible, not merely bounded.

The three-step pipeline this doc executes:

1. **Identify the failure mode** in GLM-5's stack (the empirical locus).
2. **Identify the axiom it violates** and why (the diagnostic).
3. **Specify the QA-native construction** that makes the failure mode structurally impossible (the resolution).

Cert families target step 3, not steps 1–2.

## Scope relative to prior work

- `qa_lab/AI_ARCHITECTURE_INTEGRATION.md` (2025-11-22) — entity-level tuple/E8 assignments for DS-Star, Kimi K2, AI Co-Scientist, Kosmos. Taxonomic foundation for this doc; integration roadmap (QA-MoE, QA Co-Scientist, QA family router) is the design substrate the QA-native architecture here extends.
- `qa_lab/GRAPH_RAG_INTEGRATION.md` — E8 coordinates for the same entities.
- `qa_alphageometry_ptolemy/QA_MAP__SPARSE_ATTENTION.yaml` (2026-01-24) — pre-DSA sparse-attention mapping (entropy/rank/sparsity).
- **QA-MEM coverage gap:** KG does not index the `qa_lab/*_INTEGRATION.md` docs. Flagged separately.

---

## Failure Mode 1 — Non-deterministic discrete selection under learned scoring

### GLM-5 observation

DSA's lightning indexer scores all prior tokens against each query; `top-k (k=2048)` selects the subset to attend to. Non-deterministic CUDA `topk` caused entropy collapse within a few RL steps. Forced `torch.topk` (deterministic, slower); indexer frozen during RL.

### Axiom violated

**A1 (discrete state stability) + Theorem NT (firewall).** Continuous scores are observer-layer; the selected integer index set is the causal input to the sparse-attention kernel. Non-determinism makes the integer state inconsistent across forward passes — the discrete layer receives jittering inputs. This is T2-b in a different form: continuous noise leaking through the projection into QA state.

### Why this is unavoidable in the standard-math frame

Standard attention architectures **learn a scoring function** (continuous, float-valued) and then **discretize by selection** (top-k). The selection mechanism is bolted onto a continuous learning substrate that has no native concept of discrete structure. Determinism is a fragile engineering fix on top of a continuous system, not a structural property.

### QA-native resolution

Attention is not a learned scoring-then-selection pipeline. **Attention is orbit resonance.** Each token is embedded as a QA tuple `(b, e, d, a)`; two tokens attend to each other iff their orbit classifications satisfy a resonance rule — same orbit family, harmonic-conjugate pair, or a Pythagorean relation `C² + F² = G²` on `(C, F, G) = (2de, ba, e²+d²)`.

Consequences:

- **No scoring function to learn** → no gradient-dependent selection.
- **No top-k operator** → nothing to be deterministic or non-deterministic about.
- **Selection IS structure.** Two tokens either resonate under the orbit rule or they don't; the answer is a deterministic function of integer tuples.
- **Indexer replay, determinism engineering, freezing during RL — all dissolved.** There is no indexer.

This is the form the GLM-5 paper gropes toward when it freezes the indexer during RL: they recognized that the selection mechanism must not be learned stochastically. The QA-native answer is that it should not be learned *at all* — it is a property of the orbit structure.

### Cert target

`QA_ORBIT_RESONANCE_ATTENTION_CERT.v1` — validates that an attention operator:
- takes integer tuple inputs (A1, S2),
- returns a deterministic pairwise resonance relation on tuples,
- contains no learned scoring sub-network,
- passes the Pythagorean/orbit-family resonance tests on canonical fixtures.

---

## Failure Mode 2 — Continuous intermediate between action sampling and credit assignment

### GLM-5 observation

The Token-In-Token-Out (TITO) gateway carries exact integer token IDs from rollout engine to learner. Re-tokenizing decoded text on the learner side corrupts step alignment between actions and rewards. Async RL without TITO exhibited action↔reward misalignment.

### Axiom violated

**Theorem NT.** The observer/QA boundary must be crossed exactly twice per computation: once at input tokenization, once at output decoding. Re-tokenization inside the training loop is an extra, illegal boundary crossing — the integer state is re-projected through a continuous (text) intermediate and comes back corrupted.

### Why this is unavoidable in the standard-math frame

RL pipelines treat "rollout output" and "trainer input" as separate stages with independent data representations. Text is the lowest common denominator. The architecture has no invariant preserving integer-state fidelity across the rollout/learner boundary — TITO is a *discipline*, not a *structural guarantee*. Any future component that materializes text in between reintroduces the failure.

### QA-native resolution

**There is no rollout/learner split.** In a QA-native architecture, there is no stochastic policy being sampled into trajectories that then get scored by a learner. There is a generator-pattern identification process on integer orbit states. The "rollout" and "learner" are the same object: a discrete search in generator space.

More concretely: training is not SGD on float weights indexed by policy versions. Training is identification of the integer generator `e_t = ((b_{t+1} − b_t − 1) mod m) + 1` (per cert [209]) that produces the target behavior. The output of this identification is an integer generator pattern, consumed directly by the next inference step — no text intermediate, no re-tokenization, no possibility of extra boundary crossings.

### Cert target

`QA_INTEGER_STATE_PIPELINE_CERT.v1` — validates end-to-end that:
- observer projections occur exactly at input and output boundaries,
- the intermediate representation is pure integer tuples,
- no continuous float state persists across the integer-state layer,
- re-projection through a continuous intermediate is structurally precluded (not merely disciplined).

This cert is primarily a **checker for QA-native architectures**. It is inapplicable to standard RL pipelines by construction — which is the point.

---

## Failure Mode 3 — Float policy staleness under asynchronous updates

### GLM-5 observation

In async RL a trajectory may span multiple policy updates. GLM-5 records `(w_0, …, w_k)` per rollout and drops the trajectory if `w_current − w_0 > δ`. Optimizer reset after each weight push to the rollout engine. Double-sided importance sampling `[1−ε, 1+h]` with hard masking outside the trust region.

### Axiom violated

**T1 (path time) + S2 (no float state).** Policy versions index a sequence of continuous float-weight configurations. Staleness is real because the underlying state is a float tensor that drifts continuously; importance sampling corrections are needed because the rollout policy distribution differs from the current training distribution. Optimizer-reset patches the S2 violation locally but does not eliminate it.

### Why this is unavoidable in the standard-math frame

RL updates a float policy by stochastic gradient descent. Every step of SGD produces a new, slightly different continuous function. Asynchrony creates a fan of simultaneously-valid-ish float policies. The entire apparatus (IS corrections, trust regions, staleness bounds, optimizer resets) exists to manage drift of a continuous object. Remove the float policy, and the apparatus becomes unnecessary.

### QA-native resolution

**There is no float policy.** The "policy" is an integer generator pattern (a member of a QA family — Fibonacci, Lucas, Tribonacci, Phibonacci, Ninbonacci, per cert [214]). Training is a discrete search over generator patterns; each candidate either matches target behavior (on held-out integer traces) or doesn't. There is no gradient drift because there is no float state being updated.

Consequences:

- **No off-policy bias** because the "policy" is a discrete integer pattern, not a stochastic distribution over continuous actions.
- **No staleness** because an integer generator pattern is either current or has been superseded by an identified-better one. There is no continuum of "mostly-current" policies.
- **No importance sampling** because there is no distribution over float-parameterized actions whose probability needs to be compared across versions.
- **No optimizer reset** because there is no optimizer state (Adam m, v) — training does not move on a float manifold.
- **No trust region** because moves are discrete hops between generator patterns, not continuous steps in weight space.

T1 path-time is the native time coordinate: the integer index `t` of a discrete orbit walk. It is not a label painted on a continuous-time process; it is the only time there is.

### Cert target

`QA_GENERATOR_PATTERN_TRAINING_CERT.v1` — validates that:
- the trained object is an integer generator pattern (not a float weight tensor),
- updates are discrete transitions between generator patterns (not gradient steps),
- training state is fully integer (no Adam buffers, no momentum, no EMA),
- held-out evaluation uses orbit-trace match, not scalar loss.

---

## QA-Native Architecture Sketch

Combining the three resolutions:

**Input layer** — continuous input (text, signal, image) is projected via observer to integer QA tuples `(b, e, d, a)` per token/element. Boundary crossed once.

**Attention layer** — orbit resonance operator on tuple pairs. No scoring, no selection, no learned parameters on this path. Deterministic by construction.

**Evolution layer** — T-operator / modular generator dynamics on tuples. `t` is integer path time. State transitions are deterministic given the generator pattern.

**Output layer** — integer tuples projected to continuous output (text tokens, actions, reconstructions) via observer. Boundary crossed once.

**Training** — identification of the generator pattern whose orbit trace matches target behavior on held-out integer data. Discrete search. No gradients, no float optimizer state, no stochastic rollouts.

The GLM-5 failure modes are absent by construction:

| GLM-5 failure | QA-native status |
|---|---|
| Non-deterministic `topk` entropy collapse | No `topk`. Attention is orbit resonance. |
| TITO misalignment under re-tokenization | No rollout/learner split. No text intermediate. |
| Policy staleness / off-policy bias | No float policy. No continuum of versions. |
| Optimizer reset / IS corrections / trust region | No float optimizer. No continuous-gradient updates. |

This is not a claim that the QA-native architecture is competitive on GLM-5's benchmarks today. It is a claim that the architecture's structural properties eliminate the failure-mode *class* rather than patching instances. The benchmarking question is separate and empirical; this doc is a design claim, not a performance claim.

---

## Revised Cert Scaffolding Recommendation

Three candidate families, each validating a component of the QA-native architecture. None cert GLM-5's RL pipeline; all cert QA-native constructions. One axiom concern per cert; no conflation.

| Family | Axiom locus | What it validates | Next ID |
|---|---|---|---|
| `QA_ORBIT_RESONANCE_ATTENTION_CERT.v1` | A1 + NT (attention) | Attention operator is deterministic orbit resonance, not learned scoring+selection | ≥ 256 |
| `QA_INTEGER_STATE_PIPELINE_CERT.v1` | NT (whole pipeline) | Two-boundary-crossing invariant structurally enforced; no continuous intermediates | ≥ 257 |
| `QA_GENERATOR_PATTERN_TRAINING_CERT.v1` | S2 + T1 | Training updates integer generator patterns on integer path time; no float optimizer state | ≥ 258 |

**Blocker for full scaffolding:** fixtures. Synthetic hand-constructed fixtures suffice for initial pass; real fixtures require a minimum QA-native implementation of each component. Implementation is downstream of this doc.

**Relation to existing gold-standard yaml.** `QA_MAP__SPARSE_ATTENTION.yaml` (2026-01-24) maps Linformer/BigBird/Longformer-era sparse attention as a standard-math object with entropy/rank/sparsity invariants. The new orbit-resonance cert is orthogonal — it validates a QA-native replacement, not a standard-math implementation. The existing yaml can remain frozen; the new cert lives alongside it.

---

## Open questions and follow-ups

1. **Orbit-resonance attention has not been implemented.** The cert design above assumes a concrete operator; the operator itself is the implementation work. Cert [234] (chromogeometry Pythagorean identity) provides the algebraic foundation.
2. **Generator-pattern training as discrete search:** the search space (QA families: Fibonacci, Lucas, Tribonacci, Phibonacci, Ninbonacci — cert [214]) is small. Whether it is expressive enough for LLM-scale tasks is the question. Extended families may be needed; this is an open research question, not a blocker on the mapping.
3. **Integer state pipeline cert** overlaps substantially with `qa_observer_core_cert_v1` [159] and the Theorem NT spec. Verify no duplication before registration.
4. **QA-MEM indexing:** the `qa_lab/*_INTEGRATION.md` family of docs should be ingested into QA-MEM. Separate work item.
5. **Prior DS-Star entity mapping** `(b=1, e=3, d=4, a=7)` predates DSA; revisit whether the attention-mechanism layer of DeepSeek warrants its own entity-level tuple now that the attention is a distinct architectural component.

---

## References

- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*. arXiv:2602.15763 [cs.LG]. Feb 2026.
- DeepSeek-AI. *DeepSeek-V3.2-Exp* (DSA introduction).
- `qa_lab/AI_ARCHITECTURE_INTEGRATION.md` (Signal Experiments, 2025-11-22).
- `qa_alphageometry_ptolemy/QA_MAP__SPARSE_ATTENTION.yaml` (Signal Experiments, 2026-01-24).
- Cert [209] QA Signal Generator Inference (`e_t = ((b_{t+1} − b_t − 1) mod m) + 1`).
- Cert [214] QA Norm-Flip Signed-Temporal (5 signed QA families).
- Cert [234] QA Chromogeometry Pythagorean Identity (`C² + F² = G²`).
- `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md` (Theorem NT canonical spec).
