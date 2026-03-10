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
| [26] | [QA Competency Detection](26_competency_detection.md) | Bundle + Metrics + Levin mapping semantic gates (`--levin-audit`) | PASS |
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
| [40] | [QA Reachability Descent Run Cert v1](40_reachability_descent_run_cert.md) | Schema + Validator + Fixtures (PASS + negative fixtures) | PASS |
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
| [82] | [QA BSD Local Euler Cert v1](82_bsd_local_euler_cert.md) | Schema v1/v1.1 + Validator (schema/recompute/reduction-type gates; v1.1 optional invariants `delta_mod_p`/`is_good_reduction` + `ap_source`) + 3 Fixtures (pass_good_p5, pass_good_p7_v1_1, fail_wrong_ap) | PASS |
| [83] | [QA BSD Local Euler Batch Cert v1](83_bsd_local_euler_batch_cert.md) | Schema + Validator (per-prime deterministic recompute + stable hash-manifest binding) + 3 Fixtures (pass_batch_p5_p7, pass_batch_p5_p11, fail_corrupt_record_p7_ap) | PASS |
| [84] | [QA BSD Partial L-series Proxy Cert v1](84_bsd_partial_lseries_proxy_cert.md) | Schema + Validator (exact non-reduced Π(#E(F_p)/p) proxy + source-manifest binding) + 3 Fixtures (pass_proxy_p5_p7, pass_proxy_p5_p11, fail_wrong_proxy_denominator) | PASS |
| [85] | [QA BSD Rank Squeeze Cert v1](85_bsd_rank_squeeze_cert.md) | Schema + Validator (local recompute + manifest binding + exact proxy + monotone rank-trace consistency/closure checks) + 5 Fixtures (pass_closed_p5_p7, pass_open_p5_p11, fail_bad_trace_crossing, fail_wrong_proxy_denominator, fail_wrong_ap_p7) | PASS |
| [86] | [QA Generator-Failure Algebra Unification Cert v1](86_generator_failure_unification_cert.md) | Schema + Validator (5-gate: carrier cross-check, digest, T1 finite image, T2 SCC + T3 path propagation, T4 energy monotonicity) + 3 Fixtures (valid_caps_tr_fear_love, invalid_tag_not_in_carrier, invalid_energy_drift) + cross-binding to [76] failure algebra ref + [80] energy cert ref | PASS |
| [87] | [QA Failure Compose Operator Cert v1](87_failure_compose_operator_cert.md) | Schema + Validator (formal `compose(Fi,Fj,form)` with closure/table completeness + per-form associativity checks) + 3 Fixtures (pass_feedback_escalation, fail_closure_incomplete_table, fail_associativity_feedback_violation) | PASS |
| [88] | [QA Failure Algebra Structure Classification Cert v1](88_failure_algebra_structure_classification_cert.md) | Schema + Validator (form-indexed semigroup/monoid classification with identity/absorber/commutativity and optional monotonicity checks) + 5 Fixtures (pass_classify_from_family87_tables, fail_identity_claim_wrong, fail_absorber_claim_wrong, fail_commutative_claim_wrong, fail_monotonicity_violation) | PASS |
| [89] | [QA QALM Curvature Cert v1](89_qalm_curvature_cert.md) | Schema + Validator (H_QA recompute and curvature-scaled update-rule pin) + 3 Fixtures (pass_default_tuple, fail_h_qa_mismatch, fail_update_sign) | PASS |
| [90] | [QA Fairness Demographic Parity Cert v1](90_fairness_demographic_parity_cert.md) | Schema + Validator (demographic parity gap with constructive failure witness) + 2 Fixtures (valid_min, invalid_gap) | PASS |
| [91] | [QA Fairness Equalized Odds Cert v1](91_fairness_equalized_odds_cert.md) | Schema + Validator (equalized odds TPR/FPR gap with constructive failure witness) + 2 Fixtures (valid_min, invalid_gap) | PASS |
| [92] | [QA Safety Prompt Injection Refusal Cert v1](92_safety_prompt_injection_refusal_cert.md) | Schema + Validator (prompt injection refusal rate with judge contract hash and failure witness) + 2 Fixtures (valid_min, invalid_rate) | PASS |
| [93] | [QA GNN Message-Passing Curvature Cert v1](93_gnn_mp_curvature_cert.md) | Schema + Validator (H_QA recompute + agg_gain update-rule pin + kappa pin + graph metadata) + 4 Fixtures (pass_default_graph, fail_agg_gain_mismatch, fail_h_qa_mismatch, fail_graph_metadata_invalid) | PASS |
| [94] | [QA Attention Layer Curvature Cert v1](94_attn_curvature_cert.md) | Schema + Validator (H_QA recompute + attn_gain update-rule pin + kappa pin + attention metadata) + 4 Fixtures (pass_default_attn, fail_attn_gain_mismatch, fail_h_qa_mismatch, fail_seq_len_invalid) | PASS |
| [95] | [QA QARM Curvature Cert v1](95_qarm_curvature_cert.md) | Schema + Validator (H_QA recompute + qarm_gain update-rule pin + kappa pin + QARM metadata) + 4 Fixtures (pass_default_qarm, fail_qarm_gain_mismatch, fail_h_qa_mismatch, fail_modulus_invalid) | PASS |
| [96] | [QA Symbolic Search Curvature Cert v1](96_symbolic_search_curvature_cert.md) | Schema + Validator (H_QA recompute + sym_gain update-rule pin + kappa pin + search metadata) + 4 Fixtures (pass_default_sym, fail_sym_gain_mismatch, fail_h_qa_mismatch, fail_beam_width_invalid) | PASS |\n| [97] | [QA Orbit Curvature Cert v1](97_orbit_curvature_cert.md) | Schema + Validator (orbit enumeration, H_QA series, kappa_min stability margin) + 4 Fixtures (pass_orbit_12, fail_orbit_length_mismatch, fail_kappa_min_mismatch, fail_schema) | PASS |
| [98] | [QA GNN Spectral Gain Cert v1](98_gnn_spectral_gain_cert.md) | Schema + Validator (H_QA recompute + sigma_max power iteration + update-rule pin + kappa pin) + 4 Fixtures (pass_gnn_weight, fail_sigma_mismatch, fail_h_qa_mismatch, fail_schema) | PASS |
| [99] | [QA Attention Spectral Gain Cert v1](99_attn_spectral_gain_cert.md) | Schema + Validator (H_QA recompute + sigma_max(QK^T/sqrt(d_k)) power iteration + update-rule pin + kappa pin) + 4 Fixtures (pass_attn_score, fail_sigma_mismatch, fail_h_qa_mismatch, fail_schema) | PASS |
| [100] | [QA E8 Alignment Audit Cert v1](100_e8_alignment_audit_cert.md) | Schema + Validator (full orbit enumeration, 240 E8 cosines, pre-registered decision rule) + 4 Fixtures (pass_mod9_incidental, fail_wrong_verdict, fail_stats_mismatch, fail_schema) | PASS |
| [101] | [QA Gradient Lipschitz Gain Cert v1](101_gradient_lipschitz_gain_cert.md) | Schema + Validator (H_QA recompute + grad_norm L2 derivation + update-rule pin + kappa pin) + 4 Fixtures (pass_grad_l2, fail_grad_norm_mismatch, fail_h_qa_mismatch, fail_schema) | PASS |
| [102] | [QA Lojasiewicz Orbit Descent Cert v1](102_lojasiewicz_orbit_cert.md) | Schema + Validator (orbit feasibility + C(O) recompute Gate 2D + H-crit witness + phi-bound + orbits-bound) + 4 Fixtures (pass_default, fail_co_mismatch, fail_hcrit, fail_schema) | PASS |
| [103] | [QA Lojasiewicz Orbit Descent Cert v2 (intrinsic)](103_lojasiewicz_orbit_cert_v2.md) | Schema + Validator (orbit feasibility + C(O) recompute Gate 2D + phi-bound + orbits-bound; h_crit_witnessed removed — H-crit derived via B3) + 3 Fixtures (pass_default, fail_co_mismatch, fail_schema) | PASS |

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

## Tooling / CI

### LaTeX Claim Linter (`tools/qa_latex_claim_linter.py`)

Scans `.tex` sources for DPI-anchor and overclaim trigger phrases and enforces
the PAC-Bayes tripwire bundle (families [84], [85], [86]).

```bash
# Lint the full PAC-Bayes workspace (strict mode for pre-submission)
python3 tools/qa_latex_claim_linter.py papers/in-progress/phase1-pac-bayes/phase1_workspace --strict

# Lint + JSON output (for CI pipelines)
python3 tools/qa_latex_claim_linter.py papers/ --json

# Custom rules
python3 tools/qa_latex_claim_linter.py papers/ --config tools/qa_latex_claim_linter_rules.json
```

Tag: `tool-qa-latex-claim-linter-v1.0.0`

---

## Two-tract checklist (for contributors)

Before shipping any family:

- [ ] Schema(s) committed
- [ ] Validator passes `--demo`
- [ ] Witness pack + counterexamples pack present
- [ ] Meta-validator hook wired in `qa_meta_validator.py`
- [ ] `docs/families/[NN]_<slug>.md` written
- [ ] This README index updated
- [ ] Release notes in `QA_MAP_CANONICAL.md` updated
