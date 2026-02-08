# QA Canonical Mapping Registry

This document tracks **Gold Standard** QA mappings—papers/theories that have been fully translated into the QA certificate framework with:

1. Conceptual mapping (YAML spec)
2. Certificate dataclasses
3. Strict v3 validator
4. Recompute hooks
5. Example certificates (success + failure)
6. End-to-end validation passing

---

## Gold Standard Mappings

### 1. Architecture-Independent Generalization Bounds (Bapu–Chen et al.)

**Source**: arXiv:2504.05695 — "Architecture independent generalization bounds for overparametrized deep ReLU networks"

**Status**: ✅ Complete (2026-01-22)

**Key Insight**: Generalization is an **invariant-controlled reachability property**, not an architectural one.

#### Concept Mapping

| Paper Concept | QA Interpretation |
|---------------|-------------------|
| D_geom (metric geometry) | Terrain roughness of state space |
| Operator norms (||W||₂, ||b||₂) | Energy constraints on transitions |
| ReLU regularity | Gates partitioning linear regions |
| Zero-loss constructor | Constructive success certificate |
| Overparametrization | **Gauge freedom** (null directions that don't change certificate) |
| Generalization bound | Reachability certificate |

#### Why This Mapping Matters

1. **Overparametrization ≠ capacity**: Extra parameters are gauge coordinates, not memorization capacity
2. **Zero-loss ≠ overfitting**: Explicit constructors prove this for n ≤ d
3. **Failures are informative**: Vacuous bounds produce structured obstruction certificates
4. **Architecture-independent**: Same certificate schema works across MLP, ResNet, VGG

#### Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__ARCH_INDEP_RELU_GENERALIZATION.yaml` |
| Certificate module | `qa_generalization_certificate.py` |
| Strict v3 validator | `qa_generalization_validator_v3.py` |
| Recompute hooks | `qa_generalization_hooks.py` |
| MNIST success cert | `examples/generalization/mnist_mlp_success.json` |
| Vacuous bound failure | `examples/generalization/vacuous_bound_failure.json` |
| Zero-loss constructor | `examples/generalization/zero_loss_constructor.json` |
| Complete bundle | `examples/generalization/complete_bundle.json` |

#### Validation Commands

```bash
# Validate individual certificates
python qa_generalization_validator_v3.py --demo
python qa_generalization_validator_v3.py examples/generalization/mnist_mlp_success.json
python qa_generalization_validator_v3.py examples/generalization/vacuous_bound_failure.json

# Validate full bundle
python qa_generalization_validator_v3.py --bundle examples/generalization/complete_bundle.json

# Test recompute hooks
python qa_generalization_hooks.py
```

#### Validation Results (2026-01-22)

```
✔ Demo validation:           15 passed, 0 failed
✔ MNIST MLP success:         15 passed, 0 failed
✔ Vacuous bound failure:      6 passed, 0 failed
✔ Complete bundle:           31 passed, 0 failed
```

#### Certificate Schema Summary

| Schema | Description |
|--------|-------------|
| `QA_GENERALIZATION_CERT_V1` | Main generalization bound certificate |
| `QA_ZERO_LOSS_CONSTRUCTOR_V1` | Zero-loss network construction (n ≤ d) |
| `QA_GENERALIZATION_BUNDLE_V1` | Multi-architecture comparison bundle |

#### Failure Modes (12 total)

| Category | Modes |
|----------|-------|
| Data geometry | `insufficient_samples`, `data_not_separable`, `metric_degeneracy` |
| Norm control | `norm_explosion`, `spectral_overflow`, `bias_overflow` |
| Architecture | `depth_too_shallow`, `width_insufficient` |
| Training | `no_zero_loss_solution`, `optimization_stuck` |
| Bound validity | `bound_vacuous`, `bound_not_computable` |

#### Recompute Hooks (4 total)

| Hook ID | Description |
|---------|-------------|
| `metric_geometry_v1` | Recompute D_geom from raw data |
| `operator_norm_v1` | Recompute spectral/bias norms from weights |
| `generalization_bound_v1` | Recompute bound from witnesses |
| `zero_loss_constructor_v1` | Verify zero-loss construction |

---

## Template for Future Mappings

To add a new Gold Standard mapping:

1. **Identify key concepts** that map to QA structures (states, transitions, invariants, obstructions)
2. **Create YAML module spec** following `QA_MAP__<PAPER_ID>.yaml` pattern
3. **Implement certificate dataclasses** with exact scalar arithmetic
4. **Implement strict v3 validator** with schema/consistency/recompute levels
5. **Implement recompute hooks** for independent verification
6. **Create example certificates** (at least one success, one failure)
7. **Run full validation** and document results
8. **Add entry to this registry**

---

### 2. NeuralGCM — Physics-Constrained Weather/Climate Models

**Source**: Neural General Circulation Models (Google DeepMind, 2024)

**Status**: ✅ Complete (2026-01-24)

**Key Insight**: Physical validity comes from **conservation law invariants**, not architecture.

#### Concept Mapping

| Paper Concept | QA Interpretation |
|---------------|-------------------|
| Mass conservation | Hard invariant (packet preservation) |
| Energy conservation | Hard invariant (bounded drift) |
| Momentum conservation | Hard invariant (angular momentum) |
| Neural parameterizations | Gauge freedom within physics constraints |
| Forecast skill | Reachability certificate (drift from truth) |
| Physical bounds | State space constraints |
| CFL condition | Numerical stability invariant |

#### Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__NEURALGCM.yaml` |
| Certificate module | `qa_neuralgcm_certificate.py` |
| Strict v3 validator | `qa_neuralgcm_validator_v3.py` |
| 10-day success cert | `examples/neuralgcm/10day_forecast_success.json` |
| Mass violation failure | `examples/neuralgcm/mass_violation_failure.json` |

#### Validation Commands

```bash
# Validate success certificate
python qa_neuralgcm_validator_v3.py examples/neuralgcm/10day_forecast_success.json

# Validate failure certificate
python qa_neuralgcm_validator_v3.py examples/neuralgcm/mass_violation_failure.json

# Run demos
python qa_neuralgcm_validator_v3.py --demo
python qa_neuralgcm_validator_v3.py --demo-failure
```

#### Validation Results (2026-01-24)

```
✔ 10-day success:     24 passed, 0 failed
✔ Mass violation:      6 passed, 0 failed
```

#### Failure Modes (11 total)

| Category | Modes |
|----------|-------|
| Conservation | `mass_violation`, `energy_violation`, `momentum_violation` |
| Physical bounds | `negative_humidity`, `negative_pressure`, `unphysical_temperature`, `unphysical_wind` |
| Numerical | `cfl_violation`, `numerical_instability`, `divergence_detected` |
| Skill | `skill_below_climatology`, `acc_below_threshold`, `skill_collapse` |

#### Unification with Generalization Bounds

| Concept | Generalization | NeuralGCM |
|---------|---------------|-----------|
| Invariants | Operator norms | Conservation laws |
| Gauge freedom | Overparametrization | Neural parameterizations |
| Failure modes | Vacuous bounds | Conservation violation |

---

### 3. Sparse Attention — Transformer Efficiency

**Source**: Efficient Transformers literature (Linformer, BigBird, etc.)

**Status**: ✅ Complete (2026-01-24)

**Key Insight**: Attention health is an **invariant-controlled efficiency property**, not a capacity one.

#### Concept Mapping

| Paper Concept | QA Interpretation |
|---------------|-------------------|
| Attention entropy | State exploration breadth |
| Effective rank | Representational capacity utilization |
| Sparsity pattern | Constrained reachability graph |
| Head redundancy | **Gauge freedom** (prunable without perf loss) |
| Rank collapse | Structural degeneracy failure mode |
| Information flow | Layer-wise residual/attention balance |

#### Why This Mapping Matters

1. **Redundant heads ≠ capacity**: Extra heads are gauge coordinates, not expressiveness
2. **Rank collapse = wasted compute**: Deep layers can degenerate to effective rank-1
3. **Entropy bounds**: Both collapse (deterministic) and explosion (uniform) are pathological
4. **Architecture-agnostic**: Same certificate works for BERT, GPT, ViT, etc.

#### Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__SPARSE_ATTENTION.yaml` |
| Certificate module | `qa_sparse_attention_certificate.py` |
| Strict v3 validator | `qa_sparse_attention_validator_v3.py` |
| BERT-base success | `examples/sparse_attention/bert_base_success.json` |
| Rank collapse failure | `examples/sparse_attention/rank_collapse_failure.json` |

#### Validation Commands

```bash
# Validate success certificate
python qa_sparse_attention_validator_v3.py examples/sparse_attention/bert_base_success.json

# Validate failure certificate
python qa_sparse_attention_validator_v3.py examples/sparse_attention/rank_collapse_failure.json

# Run demos
python qa_sparse_attention_validator_v3.py --demo
```

#### Validation Results (2026-01-24)

```
✔ BERT-base success:      22 passed, 0 failed
✔ Rank collapse failure:   6 passed, 0 failed
```

#### Failure Modes (8 total)

| Category | Modes |
|----------|-------|
| Entropy | `entropy_collapse`, `entropy_explosion` |
| Rank | `rank_collapse`, `representation_collapse` |
| Sparsity | `disconnected_tokens`, `sparsity_too_aggressive` |
| Approximation | `linear_approximation_error`, `kernel_feature_instability` |

#### Certificate Schema Summary

| Schema | Description |
|--------|-------------|
| `QA_SPARSE_ATTENTION_V1` | Full attention stack certificate |

#### Witness Types (5 total)

| Witness | Description |
|---------|-------------|
| `AttentionEntropyWitness` | Per-head entropy statistics |
| `EffectiveRankWitness` | Attention matrix rank analysis |
| `SparsityPatternWitness` | Allowed pair count and reachability |
| `HeadRedundancyWitness` | Prunable head count per layer |
| `InformationFlowWitness` | Residual vs attention contribution |

#### Unification with Generalization Bounds

| Concept | Generalization | Sparse Attention |
|---------|---------------|------------------|
| Gauge freedom | Overparametrization | Redundant heads |
| Invariants | Operator norms | Entropy/rank bounds |
| Failure modes | Vacuous bounds | Rank collapse |

---

### 4. Axiom AI + Execution-Grounded Research — Formal Reasoning

**Source**: Axiom AI (AxiomProver, Putnam 2025) + Stanford arXiv:2601.14525

**Status**: ✅ Complete (2026-01-24)

**Key Insight**: Formal reasoning is **certificate-controlled reachability** with the kernel as invariant oracle.

#### Concept Mapping

| Paper Concept | QA Interpretation |
|---------------|-------------------|
| Proof state (Lean) | QA state in proof space |
| Tactics | Generators over proof states |
| Lean kernel | Invariant oracle (ACCEPT/REJECT) |
| Human intuition | **Observer projection** (non-executable) |
| Execution-grounded | Certificate-controlled reachability |
| Formalization gap | Missing generator (F1 failure) |

#### Why This Mapping Matters

1. **Intuition ≠ proof**: Observer projection doesn't guarantee reachability
2. **Difficulty is generator-relative**: Adding lemmas changes what's reachable
3. **Execution is validation**: Ideas must run, not just sound plausible
4. **QA subsumes both**: Axiom and AlphaGeometry are QA special cases

#### Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__AXIOM_AI.yaml` |
| Ledger document | `appendix/QA_AXIOM_LEDGER.md` |
| Stratification theorem | `appendix/QA_AXIOM_STRATIFICATION_THEOREM.md` |
| Comparison table | `appendix/QA_COMPARISON_TABLE.md` |
| Manifesto | `appendix/CHECKING_IS_NOT_ENOUGH.md` |
| Failure algebra schema | `schemas/QA_FAILURE_ALGEBRA.json` |

#### Stratification Theorem

```
For any reasoning task R, there exist three strata:
1. Intuition Stratum (IS): Observer-projected, non-replayable
2. Formalization Stratum (FS): Generator-restricted, discrete
3. Kernel Stratum (KS): Deterministic invariant oracle

Valid solution iff reachable in FS and accepted by KS.
IS plausibility does NOT imply validity.
```

#### Failure Modes (6 classes)

| Class | Name | Description |
|-------|------|-------------|
| F1 | Formalization Gap | Obvious step has no generator |
| F2 | Case Explosion | Exponential branching |
| F3 | Rewrite Blocked | Representation mismatch |
| F4 | Budget Exhaustion | Timeout/token limit |
| F5 | Kernel Violation | Invariant oracle rejects |
| F6 | Component Isolation | Structural unreachability |

#### Unification with Other Mappings

| Concept | Generalization | Axiom/Execution |
|---------|---------------|-----------------|
| State space | Function class | Proof states / hypotheses |
| Invariants | Operator norms | Kernel acceptance / empirical results |
| Gauge freedom | Overparametrization | Tactic/method choice |
| Failure modes | Vacuous bounds | Formalization gaps |

---

### 5. Datastore Semantics + Retrieval Soundness

**Source**: QA Datastore Control Formalization (internal module spec)

**Status**: ✅ Complete (2026-02-07)

**Key Insight**: Storage/retrieval is a **generator-controlled QA system** with auditable observations.

#### Concept Mapping

| Datastore Concept | QA Interpretation |
|-------------------|-------------------|
| `put/get/del/migrate/compact` | Generator set over store states |
| Composite QA key | Coordinate chart (`family`, `phase_24`, `phase_9`, tuple, field) |
| Retrieval response | Observation operator with witness |
| Merkle root | Global invariant anchor |
| Migration/projection | Scale-preserving isomorphism or explicit observer projection |

#### Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__DATASTORE.yaml` |
| Validator | `qa_datastore_validator.py` |
| Snapshot builder | `qa_datastore_build_snapshot.py` |
| Semantics cert | `certs/QA_DATASTORE_SEMANTICS_CERT.v1.json` |
| Witness pack | `certs/witness/QA_DATASTORE_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_DATASTORE_COUNTEREXAMPLES_PACK.v1.json` |
| Merkle proof schema | `schemas/QA_MERKLE_PROOF.v1.schema.json` |
| View semantics cert | `certs/QA_DATASTORE_VIEW_CERT.v1.json` |
| View witness pack | `certs/witness/QA_DATASTORE_VIEW_WITNESS_PACK.v1.json` |
| View counterexamples pack | `certs/counterexamples/QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1.json` |
| View validator | `qa_datastore_view_validator.py` |

#### Validation Commands

```bash
# Validate bundled datastore family
python qa_datastore_validator.py --demo

# Validate explicit files
python qa_datastore_validator.py \
  --semantics certs/QA_DATASTORE_SEMANTICS_CERT.v1.json \
  --witness certs/witness/QA_DATASTORE_WITNESS_PACK.v1.json \
  --counterexamples certs/counterexamples/QA_DATASTORE_COUNTEREXAMPLES_PACK.v1.json

# Validate datastore view family (dual-root store/view proofs)
python qa_datastore_view_validator.py --demo
```

#### Failure Modes

| Category | Modes |
|----------|-------|
| Address/schema | `KEY_NOT_FOUND`, `SCHEMA_MISMATCH` |
| Hash/domain | `HASH_MISMATCH`, `DOMAIN_SEP_VIOLATION`, `NON_CANONICAL_JSON` |
| Proof/replay | `UNVERIFIABLE_PROOF`, `FORK_DETECTED` |
| Scale legality | `SCALE_COLLAPSE`, `MIGRATION_NON_INVERTIBLE` |

`KEY_NOT_FOUND` is valid only with a verified non-inclusion proof; otherwise classify as `UNVERIFIABLE_PROOF`.

---

### 6. Topology Resonance — Generator-Induced Geometry

**Source**: QA topology reachability notes (`QA_topology.odt` lineage)

**Status**: ✅ Complete (2026-02-07, first cert emission)

**Key Insight**: Topological reachability is a **generator-controlled invariant system** with SCC growth and phase lock as certifiable contracts.

#### Concept Mapping

| Topology Concept | QA Interpretation |
|------------------|-------------------|
| Reachability complex | State manifold under legal transitions |
| Σ={sigma,mu,lambda2,nu} | Canonical generator set |
| SCC monotonicity | Connectivity invariant |
| Phase-24/Phase-9 lock | Tuple packet coherence invariant |
| Resonance score | Thresholded success witness |

#### Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__TOPOLOGY_RESONANCE.yaml` |
| Validator | `qa_topology_resonance_validator_v1.py` |
| JSON Schema | `schemas/QA_TOPOLOGY_RESONANCE_CERT.v1.schema.json` |
| Success example | `examples/topology/topology_resonance_success.json` |
| Failure example | `examples/topology/topology_phase_break_failure.json` |
| Reference emitted cert | `certs/QA_TOPOLOGY_RESONANCE_CERT.v1.json` |
| Cert hash sidecar | `certs/QA_TOPOLOGY_RESONANCE_CERT.v1.sha256` |
| Bundle manifest | `certs/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.json` |
| Bundle schema | `schemas/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.schema.json` |
| Bundle emitter/validator | `qa_topology_resonance_bundle_v1.py` |

#### Validation Commands

```bash
python qa_topology_resonance_validator_v1.py --demo
python qa_topology_resonance_validator_v1.py examples/topology/topology_resonance_success.json
python qa_topology_resonance_validator_v1.py --level recompute examples/topology/topology_resonance_success.json
python qa_topology_resonance_validator_v1.py examples/topology/topology_phase_break_failure.json
python qa_topology_resonance_bundle_v1.py --emit --check
```

#### Failure Modes

| Category | Modes |
|----------|-------|
| Phase coherence | `phase_break` |
| Connectivity | `scc_drop` |
| Resonance | `resonance_below_threshold` |
| Packet integrity | `packet_drift` |
| Generator legality | `invalid_generator` |

---

### 7. Agentic RAG Interface — Retrieval as Control

**Source**: `ingestion candidates/qa_agentic_rag.odt`

**Status**: ✅ Complete Scaffold (2026-02-07)

**Key Insight**: Agentic RAG is a **generator-controlled retrieval interface** over certified datastore/view projections.

#### Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__ARAG_INTERFACE.yaml` |
| Validator | `qa_arag_validator.py` |
| Semantics cert | `certs/QA_ARAG_INTERFACE_CERT.v1.json` |
| Witness pack | `certs/witness/QA_ARAG_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_ARAG_COUNTEREXAMPLES_PACK.v1.json` |
| Semantics schema | `schemas/QA_ARAG_INTERFACE_CERT.v1.schema.json` |
| Witness schema | `schemas/QA_ARAG_WITNESS_PACK.v1.schema.json` |
| Counterexamples schema | `schemas/QA_ARAG_COUNTEREXAMPLES_PACK.v1.schema.json` |

#### Validation Commands

```bash
# Validate A-RAG interface family
python qa_arag_validator.py --demo
```

#### Core Tool Set

| Tool | QA Generator Meaning |
|------|-----------------------|
| `keyword_search` | lexical anchoring move |
| `semantic_search` | semantic neighborhood move |
| `chunk_read` | materialization move (store-root read) |

---

### 8. Ingest->View Bridge — Provenance-Grounded Views

**Source**: QA ingestion provenance bridge (composes [18]/[20]/[21])

**Status**: ✅ Complete Scaffold (2026-02-08)

**Key Insight**: A certified view entry is valid only if it is rooted to store/view snapshots and grounded in ingested document proofs.

#### Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__INGEST_VIEW_BRIDGE.yaml` |
| Validator | `qa_ingest_view_bridge_validator.py` |
| Semantics cert | `certs/QA_INGEST_VIEW_BRIDGE_CERT.v1.json` |
| Witness pack | `certs/witness/QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1.json` |
| Semantics schema | `schemas/QA_INGEST_VIEW_BRIDGE_CERT.v1.schema.json` |
| Witness schema | `schemas/QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1.schema.json` |
| Counterexamples schema | `schemas/QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1.schema.json` |

#### Validation Commands

```bash
# Validate ingest->view bridge family
python qa_ingest_view_bridge_validator.py --demo
```

#### Core Invariants

| Invariant | Meaning |
|-----------|---------|
| Document grounding required | Every bridge entry cites doc refs with ingest inclusion proofs |
| Root binding | proof bundle store/view roots must match certified snapshots |
| Typed view provenance | KEYWORD_VIEW / SEMANTIC_VIEW bind to typed snapshot ids |
| Budget control | Entry count and token budget remain bounded |

---

## Stub Mappings (In Progress)

*No current stub mappings. All planned mappings have been completed to Gold Standard.*

---

## Cross-Paper Unification

**Document**: `QA_CROSS_PAPER_UNIFICATION.md`

All Gold Standard mappings demonstrate the unified QA thesis:

> **ML/AI theory is invariant-controlled reachability with gauge freedom.**

| Concept | Generalization | NeuralGCM | Sparse Attention | Axiom/Execution | Topology Resonance |
|---------|---------------|-----------|------------------|-----------------|--------------------|
| **Domain** | Statistics | Physics-ML | Efficiency | Formal reasoning | Topological reachability |
| **Invariants** | Operator norms | Conservation laws | Entropy/rank bounds | Kernel acceptance | SCC monotonicity + phase lock |
| **Gauge freedom** | Overparametrization | Neural params | Redundant heads | Tactic choice | Generator sequence choices |
| **Failure modes** | Vacuous bounds | Conservation violation | Rank collapse | Formalization gap | Phase break / SCC drop |

---

### Candidate Papers for Future Mappings

| Paper | Key Concept | QA Fit |
|-------|-------------|--------|
| Physics of AI | Thermodynamic learning bounds | Strong (energy → operator norms) |
| Scaling laws | Power-law relationships | Medium (scale → gauge freedom) |
| Lottery ticket | Sparse subnetworks | Strong (sparsity → gauge fixing) |
| Grokking | Delayed generalization | Medium (phase transition → reachability threshold) |

---

## Citation

```bibtex
@misc{qa_canonical_mappings_2026,
  title={QA Canonical Mapping Registry},
  author={Signal Experiments Research Group},
  year={2026},
  note={Gold Standard ML/AI theory mappings to QA framework}
}
```
