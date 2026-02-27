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
| [64] | [QA Kona EBM QA-Native Orbit Reg](64_kona_ebm_qa_native_orbit_reg.md) | Schema + Validator + Fixtures (orbit-coherence regularizer, permutation gap analysis) | PASS |
| [71] | [QA Curvature Stress-Test Bundle](71_curvature_stress_test.md) | Schema + Validator + 4 Fixtures (cross-family κ universality, monoidal bottleneck law) | PASS |
| [72] | [QA Guarded Operator Category](72_guarded_operator_category.md) | Schema + Validator + 4 Fixtures (matrix embedding + guarded obstruction algebra) | PASS |
| [73] | [QA Structural Algebra Cert](73_structural_algebra_cert.md) | Schema + Validator + 3 Fixtures (bounded normal forms + uniqueness + scaling + ν guards + bridge hash binding) | PASS |
| [74] | [QA Component Decomposition Cert](74_component_decomposition_cert.md) | Schema + Validator + 3 Fixtures (gcd decomposition + scaled-seed roundtrip + ν power-of-two contraction characterization + bridge hash binding) | PASS |
| [75] | [QA Algebra Bridge Cert](75_algebra_bridge_cert.md) | Schema + Validator + 2 Fixtures (semantics anchor + word convention + component bridge + semantics hash binding) | PASS |
| [76] | [QA Failure Algebra Structure Cert](76_failure_algebra_structure_cert.md) | Schema + Validator + 2 Fixtures (finite poset + join-semilattice + monotone associative composition + propagation law) | PASS |
| [77] | [QA Neighborhood Sufficiency Cert](77_neighborhood_sufficiency_cert.md) | Schema v1.1 + Validator (branching Gate 3) + 8 Fixtures (4 valid: houston, indian_pines, salinas, ksc_failure; 4 negative: not_dominant, no_plateau, digest, claims_dominant_but_negative_delta) | PASS |
| [78] | [QA Locality Boundary Cert](78_locality_boundary_cert.md) | Schema v1.2 + Validator (6-gate incl. Gate 6 adjacency witness Mode A/B) + 7 Fixtures (3 valid: ksc_boundary v1/v1.1/v1.2 path mode; 4 negative: not_a_boundary_case, digest_mismatch, adj_rate_wrong, gt_mask_sha_mismatch) | PASS |
| [79] | [QA Locality Regime Separator Cert](79_locality_regime_sep_cert.md) | Schema v1.1 + Validator (6-gate incl. Gate 6 adjacency witness Mode A/B) + 8 Fixtures (4 v1: DOMINANT, BOUNDARY, regime_inconsistent, digest_mismatch; 4 v1.1: Salinas Mode A 5×5, KSC Mode B .npy, adj_rate_mismatch, adj_hash_mismatch) | PASS |
| [80] | [QA Energy Cert v1.1 (CAPS_TR cognitive domain)](80_energy_cert.md) | Schema + Validator (6-gate: schema/domain/ih-lock, BFS energy, reverse BFS return, monotonicity, return-in-k, SCC+power+family+interaction) + 6 Fixtures (PASS_FEAR, PASS_LOVE, PASS_MIXED, FAIL_POWER, FAIL_INTERACTION, FAIL_HORIZON) | PASS |
| [81] | [QA Episode Regime Transitions Cert v1.0](81_episode_regime_transitions.md) | Schema + Validator (5-gate: schema, label validity, regime/counts/transition matrix, drift declaration, max-run) + 6 Fixtures (PASS_RECOVERING, PASS_ESCALATING, PASS_MIXED, FAIL_LABEL, FAIL_TRANSITION, FAIL_DRIFT) | PASS |
| [82] | [QA Raman KNN Results Cert v1](82_raman_knn_results_cert.md) | Schema + Validator (5-gate: schema, canonical hash, best-acc consistency, k=1 parity, model assessment) + 3 Fixtures (PASS + FAIL_best_acc_mismatch + FAIL_model_not_realizable) | PASS |
| [83] | [QA Bell CHSH Cert v1](83_bell_chsh_cert.md) | Schema + Validator (5-gate: schema, canonical hash, 8\|N divisibility, Tsirelson bound values, model assessment) + 3 Fixtures (PASS + FAIL_wrong_condition + FAIL_wrong_value) | PASS |
| [84] | [QA PAC-Bayes Constant Cert v1.1](84_pac_bayes_constant_cert.md) | Schema v1.1 + Validator (6-gate: schema, canonical hash, K1 recompute 2C²N(M/2)², PAC bound+improvement ratio, kernel ref binding→[85] formula_id+kernel_block_sha256, DPI scope structured_only) + 4 Fixtures (PASS + FAIL_k1_mismatch + FAIL_kernel_ref_mismatch + FAIL_dpi_claim_universal) | PASS |
| [85] | [QA D_QA PAC Bound Kernel Cert v1](85_dqa_pac_bound_kernel_cert.md) | Schema + Validator (5-gate: schema, triple digest canonical+kernel_block+schema, kernel definition lock formula_id=PAC_BAYES_QA_DQA_LOGDELTA_V1, per-case recompute with witness intermediates, cross-case monotonicity) + 3 Fixtures (PASS + FAIL_digest_mismatch + FAIL_wrong_log_term) | PASS |

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
