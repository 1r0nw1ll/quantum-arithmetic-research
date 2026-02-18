# QA Certificate Family Index

Every QA certificate family is developed on **two tracts**:

1. **Machine tract**: schema, validator, cert bundle, counterexamples, meta-validator hook
2. **Human tract**: this documentation — what it is, how to run it, what breaks

A family **does not count as shipped** unless both tracts are present.

---

## Families

| ID | Name | Type | Status |
|----|------|------|--------|
| [18] | [QA Datastore](18_datastore.md) | Triplet (semantics + witness + counterexamples) | PASS |
| [19] | [Topology Resonance Bundle](19_topology_resonance.md) | Bundle manifest | PASS |
| [20] | [QA Datastore View](20_datastore_view.md) | Triplet | PASS |
| [21] | [QA A-RAG Interface](21_arag_interface.md) | Triplet | PASS |
| [22] | [QA Ingest->View Bridge](22_ingest_view_bridge.md) | Triplet | PASS |
| [23] | [QA Ingestion](23_ingestion.md) | Triplet | PASS |
| [24] | [QA SVP-CMC](24_svp_cmc.md) | Triplet + Ledger | PASS |
| [26] | [QA Competency Detection](26_competency_detection.md) | Bundle + Metrics | PASS |
| [27] | [QA Elliptic Correspondence](27_elliptic_correspondence.md) | Bundle + Deterministic Replay | PASS |
| [28] | [QA Graph Structure](28_graph_structure.md) | Bundle + Paired Deltas | PASS |
| [29] | [QA Agent Traces](29_agent_traces.md) | Schema + Validator + Fixtures | PASS |
| [30] | [QA Agent Trace Competency Cert](30_agent_trace_competency_cert.md) | Schema + Validator + Fixtures | PASS |
| [31] | [QA Math Compiler Stack](31_math_compiler_stack.md) | Validator + Fixtures (trace + pair) | PASS |
| [32] | [QA Conjecture-Prove Control Loop](32_conjecture_prove_loop.md) | Validator + Fixtures (episode + frontier + receipt) | PASS |
| [33] | [QA Discovery Pipeline](33_discovery_pipeline.md) | Validator + Fixtures (run + plan + bundle) + Batch Runner | PASS |
| [34] | [QA Rule 30 Certified Discovery](34_rule30_cert.md) | Cert Pack + Witness Manifests + File-Hash Verified | PASS |
| [35] | [QA Mapping Protocol](35_mapping_protocol.md) | Schema + Validator + Fixtures | PASS |
| [36] | [QA Mapping Protocol REF](36_mapping_protocol_ref.md) | Schema + Validator + Fixtures | PASS |
| [37] | [QA EBM Navigation Cert](37_ebm_navigation_cert.md) | Schema + Validator + Fixtures | PASS |
| [38] | [QA Energy–Capability Separation Cert](38_energy_capability_separation.md) | Schema + Validator + Fixtures | PASS |
| [39] | [QA EBM Verifier Bridge Cert](39_ebm_verifier_bridge_cert.md) | Schema + Validator + Fixtures | PASS |
| [44] | [QA Rational Trig Type System](44_rational_trig_type_system.md) | Schema + Validator + Fixtures | PASS |
| [45] | [QA ARTexplorer Scene Adapter](45_artexplorer_scene_adapter.md) | Schema + Validator + Fixtures | PASS |
| [50] | [QA ARTexplorer Scene Adapter v2 (Exact)](50_artexplorer_scene_adapter_v2_exact.md) | Schema + Validator + Fixtures (exact arithmetic) | PASS |
| [55] | [QA Three.js Scene Adapter](55_threejs_scene_adapter.md) | Schema + Validator + Fixtures | PASS |
| [56] | [QA GeoGebra Scene Adapter (Exact)](56_geogebra_scene_adapter_exact.md) | Schema + Validator + Fixtures (exact substrate, Z/Q typed coords) | PASS |
| [62] | [QA Kona EBM MNIST](62_kona_ebm_mnist.md) | Schema + Validator + Fixtures (RBM CD-1, real MNIST training, typed failure algebra) | PASS |
| [63] | [QA Kona EBM QA-Native](63_kona_ebm_qa_native.md) | Schema + Validator + Fixtures (QA orbit manifold as latent space, orbit alignment analysis) | PASS |

## Quick validation

```bash
# Run all families [18]-[56] + external validation + doc gate
cd qa_alphageometry_ptolemy
python qa_meta_validator.py

# Policy guard (fails if any `must_have_dedicated_root=True` family shares a family root)
python qa_meta_validator.py --strict

# Fast mode (manifest integrity only)
python qa_meta_validator.py --fast
```

## Provenance chain

Families [18]-[23] form a certified provenance pipeline:

```
[23] Ingestion  -->  [22] Ingest->View Bridge  -->  [20] Datastore View  -->  [21] A-RAG Interface
                                                          |
                                                    [18] Datastore
```

[24] SVP-CMC is an independent domain family (cause-first physics).

[26] Competency Detection is a standalone portable module (`qa_competency/`)
aligned with Michael Levin's Platonic Space competency-detection programme.

## Two-tract checklist (for contributors)

Before shipping any family:

- [ ] Schema(s) committed
- [ ] Validator passes `--demo`
- [ ] Witness pack + counterexamples pack present
- [ ] Meta-validator hook wired in `qa_meta_validator.py`
- [ ] `docs/families/[NN]_<slug>.md` written
- [ ] This README index updated
- [ ] Release notes in `QA_MAP_CANONICAL.md` updated
