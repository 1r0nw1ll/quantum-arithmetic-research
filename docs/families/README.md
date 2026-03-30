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
| [104] | [QA Feuerbach Parent Scale](104_feuerbach_parent_scale.md) | Schema + Validator (4 gates: schema anchor + root exception [3,4,5] + sample recompute + batch sweep) + 3 Fixtures (pass l50_full, fail bad_scale_value, fail bad_root_qa_scale) | PASS |
| [105] | [QA Cymatics Correspondence](105_cymatics.md) | 4 schemas + validator + 12 fixtures; full recognition→reachability→control→planning stack; planner certs carry minimality_witness (BFS frontier exhaustion) + hash-pinned replay link to compiled control cert; failure algebra: OFF_RESONANCE, ILLEGAL_TRANSITION, GOAL_NOT_REACHED, NONLINEAR_ESCAPE, NO_PLAN_WITHIN_BOUND, REPLAY_INCONSISTENCY, MINIMALITY_WITNESS_INCOMPLETE | PASS |
| [106] | [QA Plan-Control Compiler](106_plan_control_compiler.md) | 1 schema + validator (CC1–CC9) + 2 fixtures; generic certifiable compilation relation (search→plan→execution→witness); cymatics first instantiation; failure algebra: SOURCE_CERT_MISSING, TARGET_CERT_MISSING, GENERATOR_SEQUENCE_MISMATCH, COMPILATION_HASH_MISMATCH, REPLAY_RESULT_MISMATCH | PASS |
| [107] | [QA Core Spec Kernel](107_qa_core_spec.md) | 1 schema + validator (V1–V5) + 4 fixtures (1 PASS, 3 FAIL); base executable ontology for all QA cert families; defines state_space, generators, invariants, reachability, failure_algebra, logging, and gate policy [0..5]; failure algebra: DUPLICATE_GENERATOR_NAME, MISSING_FAILURE_ALGEBRA, BAD_GATE_SEQUENCE, LOGGING_INCOMPLETE, INVARIANT_REFERENCE_UNRESOLVED | PASS |
| [108] | [QA Area Quantization](108_qa_area_quantization.md) | 1 schema + validator (IH1–IH4 + AQ1–AQ2) + 2 fixtures; first family_extension of [107]; certifies discrete quadrea spectrum of Q(sqrt(5)) norm form mod m; mod-9 theorem: spectrum={0,1,2,4,5,7,8}, forbidden={3,6} (3 inert in Z[phi]); failure algebra: INVALID_KERNEL_REFERENCE, SPEC_SCOPE_MISMATCH, GATE_POLICY_INCOMPATIBLE, FAILURE_ALGEBRA_BREAKS_KERNEL, QUADREA_MISMATCH, FORBIDDEN_QUADREA_INCORRECT | PASS |
| [109] | [QA Inheritance Compat](109_qa_inheritance_compat.md) | 1 schema + validator (IC1–IC8) + 4 fixtures (3 PASS edges: [107]→[108], [106]→[105], [106]→[110]; 1 FAIL: gate policy deleted); certifies inheritance edges as first-class objects in the QA spec graph; failure algebra: PARENT_CERT_MISSING, CHILD_CERT_MISSING, INVALID_INHERITANCE_EDGE, GATE_POLICY_INCOMPATIBLE, FAILURE_ALGEBRA_BREAKS_PARENT, LOGGING_CONTRACT_INCOMPATIBLE, INVARIANT_REFERENCE_UNRESOLVED, SCOPE_TRANSITION_INVALID | PASS |
| [110] | [QA Seismic Pattern Control](110_qa_seismic_control.md) | 1 schema + validator (IH1–IH2 + S1–S6) + 2 fixtures (1 PASS: quiet→p_wave→surface_wave k=2; 1 FAIL: illegal quiet→surface_wave direct jump); second domain_instance of [106]; proves QA_PLAN_CONTROL_COMPILER_CERT.v1 is cross-domain (seismology ≠ cymatics); failure algebra: INVALID_KERNEL_REFERENCE, SPEC_SCOPE_MISMATCH, ILLEGAL_TRANSITION, GOAL_NOT_REACHED, ORBIT_CLASS_MISMATCH, PATH_LENGTH_EXCEEDED, PATTERN_CLASS_UNRECOGNIZED, NONLINEAR_ESCAPE | PASS |
| [111] | [QA Inert Prime Area Quantization](111_qa_area_quantization_pk.md) | 1 schema + validator (IH1–IH3 + PK1–PK4) + 4 fixtures (3 PASS: p=3/k=2 anchor, p=3/k=3, p=7/k=2 second inert prime; 1 FAIL: wrong forbidden set); family_extension of [107]; generalises [108] to mod p^k for all inert p; theorem: Im(f)={r: v_p(r)≠1}, forbidden={r: v_p(r)=1}; exhaustive recompute O(m²); failure algebra: INVALID_KERNEL_REFERENCE, SPEC_SCOPE_MISMATCH, GATE_POLICY_INCOMPATIBLE, PRIME_NOT_INERT, MODULUS_MISMATCH, SPECTRUM_MISMATCH, FORBIDDEN_SET_MISMATCH | PASS |
| [112] | [QA Obstruction-Compiler Bridge](112_qa_obstruction_compiler_bridge.md) | 1 schema + validator (IH1–IH3 + B1–B7) + 3 fixtures (2 PASS: forbidden class 3 blocked, valid class 4 unblocked; 1 FAIL: forbidden class 6 claimed reachable); first bridge family; cross-references [111] arithmetic obstruction + [106] control compiler; theorem: v_p(r)=1 ⟹ no valid plan/control cert may claim r as reachable; failure algebra: ARITHMETIC_FAMILY_MISMATCH, CONTROL_FAMILY_MISMATCH, OBSTRUCTION_VERDICT_WRONG, FORBIDDEN_TARGET_REACHABILITY_CLAIM | PASS |
| [113] | [QA Obstruction-Aware Planner](113_qa_obstruction_aware_planner.md) | 1 schema + validator (IH1–IH3 + BR1–BR5 + PA1–PA3) + 3 fixtures (2 PASS: pruned class 3 with 0 nodes, search class 4 plan found; 1 FAIL: forbidden class 6 but 47 nodes expanded); certifies planner applies [112] bridge before search; theorem: v_p(r)=1 ⟹ nodes_expanded=0; failure algebra: OBSTRUCTION_REF_MISMATCH, OBSTRUCTION_VERDICT_WRONG, OBSTRUCTION_NOT_APPLIED, PRUNE_DECISION_INCONSISTENT | PASS |
| [114] | [QA Obstruction Efficiency](114_qa_obstruction_efficiency.md) | 1 schema + validator (IH1–IH3 + EF1–EF9) + 3 fixtures (2 PASS: forbidden class 6 saves 47/47 nodes pruning_ratio=1.0, valid class 4 saves 0 with false_pruning=false; 1 FAIL: valid class 4 falsely pruned claiming ratio=1.0); quantifies search-cost savings from obstruction-aware pruning; validator recomputes saved_nodes and pruning_ratio from raw traces; completes obstruction chain [111]→[112]→[113]→[114]; failure algebra: PLANNER_REF_MISMATCH, MODULUS_MISMATCH, PRIME_NOT_INERT, TARGET_OUT_OF_RANGE, OBSTRUCTION_VERDICT_WRONG, EFFICIENCY_CLAIM_INCORRECT, FALSE_PRUNING_EFFICIENCY, AWARE_TRACE_MISMATCH | PASS |
| [115] | [QA Obstruction Stack](115_qa_obstruction_stack.md) | 1 schema + validator (IH1–IH3 + OS1–OS12) + 2 fixtures (1 PASS: canonical r=6 full chain — v_p=1 → unreachable → pruned(0 nodes) → ratio=1.0; 1 FAIL: r=6 obstruction declared but planner expanded 12 nodes, ratio=0.74 — PRUNING_CONCLUSION_MISMATCH + EFFICIENCY_CONCLUSION_MISMATCH + STACK_INCONSISTENCY); synthesis spine compressing [111]–[114] into one theorem-bearing cert; recomputes all four layers independently; failure algebra: ARITHMETIC_REF_MISMATCH, CONTROL_REF_MISMATCH, PLANNER_REF_MISMATCH, EFFICIENCY_REF_MISMATCH, OBSTRUCTION_VERDICT_WRONG, PRUNING_CONCLUSION_MISMATCH, EFFICIENCY_CONCLUSION_MISMATCH, STACK_INCONSISTENCY | PASS |
| [116] | [QA Obstruction Stack Report](116_qa_obstruction_stack_report.md) | 1 schema + validator (IH1–IH3 + RP1–RP9) + 2 fixtures (1 PASS: canonical r=6, two-row table verified, theorem + 5 layer summaries + witnesses; 1 FAIL: r=6 row claims pruned=false/aware=12/ratio=0.74: SUMMARY_TABLE_MISMATCH); reader-facing report packaging [115] for external audiences; validator recomputes entire summary table from arithmetic params; failure algebra: STACK_REF_MISMATCH, THEOREM_STATEMENT_MISSING, LAYER_SUMMARY_INCOMPLETE, SUMMARY_TABLE_MISMATCH, WITNESS_MISMATCH | PASS |
| [117] | [QA Control Stack](117_qa_control_stack.md) | 1 schema + validator (IH1–IH3 + CS1–CS11) + 2 fixtures (1 PASS: cymatics + seismology both orbit singularity→satellite→cosmos k=2; 1 FAIL: seismology declares orbit ending in satellite, cross_domain_claim lies — ORBIT_TRAJECTORY_MISMATCH + CROSS_DOMAIN_CLAIM_INCONSISTENT + STACK_INCONSISTENCY); control-side synthesis spine; certifies QA_PLAN_CONTROL_COMPILER_CERT.v1 is domain-generic across physically distinct domains; failure algebra: COMPILER_REF_MISMATCH, DOMAIN_INSTANCE_MISMATCH, ORBIT_TRAJECTORY_MISMATCH, PATH_LENGTH_MISMATCH, CROSS_DOMAIN_CLAIM_INCONSISTENT, STACK_INCONSISTENCY | PASS |
| [118] | [QA Control Stack Report](118_qa_control_stack_report.md) | 1 schema + validator (IH1–IH3 + CR1–CR7) + 2 fixtures (1 PASS: two-row comparison table — cymatics flat→hexagons and seismology quiet→surface_wave both orbit singularity→satellite→cosmos k=2; 1 FAIL: seismology row claims orbit ends in satellite, path_length_k=3 — COMPARISON_TABLE_MISMATCH); reader-facing validated report packaging [117]; validator checks orbit_path and path_length_k are cross-row consistent; failure algebra: CONTROL_STACK_REF_MISMATCH, THEOREM_STATEMENT_MISSING, DOMAIN_SUMMARY_INCOMPLETE, COMPARISON_TABLE_MISMATCH, WITNESS_MISMATCH | PASS |
| [119] | [QA Dual Spine Unification Report](119_qa_dual_spine_unification_report.md) | 1 schema + validator (IH1–IH3 + DU1–DU14) + 2 fixtures (1 PASS: canonical two-spine table — obstruction [116] and control [118] side by side, both spine refs correct, both theorems present, synthesis statement unifying both; 1 FAIL: obstruction_spine_ref points to QA_OBSTRUCTION_STACK_CERT.v1 instead of QA_OBSTRUCTION_STACK_REPORT.v1: OBSTRUCTION_SPINE_REF_MISMATCH); top-level validated overview; validator checks spine refs, theorem presence, comparison table structure and content (v_p in obstruction theorem; singularity/cosmos in control theorem), synthesis completeness, witness values; failure algebra: OBSTRUCTION_SPINE_REF_MISMATCH, CONTROL_SPINE_REF_MISMATCH, OBSTRUCTION_THEOREM_MISSING, CONTROL_THEOREM_MISSING, COMPARISON_TABLE_INCOMPLETE, COMPARISON_TABLE_MISMATCH, SYNTHESIS_STATEMENT_MISSING, SYNTHESIS_STATEMENT_INCOMPLETE, WITNESS_MISMATCH | PASS |
| [120] | [QA Public Overview Doc](120_qa_public_overview_doc.md) | 1 schema + validator (IH1–IH3 + PO1–PO9) + 2 fixtures (1 PASS: canonical overview — executive summary, two-spine diagram with chain+theorem per spine, obstruction example r=6/p=3/k=2/ratio=1.0, control example cymatics+seismology k=2, why-it-matters, both spine entry points; 1 FAIL: spine_entry_points missing control spine: SPINE_ENTRY_POINTS_INCOMPLETE); presentation-grade export derived from [119]; validator checks faithfulness to [119] including substantive content (v_p/ratio in obstruction example; cymatics+seismology in control example domains); failure algebra: OVERVIEW_REF_MISMATCH, EXECUTIVE_SUMMARY_MISSING, SPINE_DIAGRAM_MISSING, OBSTRUCTION_EXAMPLE_MISSING, OBSTRUCTION_EXAMPLE_INCOMPLETE, CONTROL_EXAMPLE_MISSING, CONTROL_EXAMPLE_INCOMPLETE, WHY_IT_MATTERS_MISSING, SPINE_ENTRY_POINTS_INCOMPLETE, WITNESS_MISMATCH | PASS |
| [122] | [QA Empirical Observation Cert](122_qa_empirical_observation_cert.md) | 1 validator (V1-V5) + 3 fixtures (2 PASS: audio orbit CONSISTENT with [107] state_space, finance script 26 CONTRADICTS curvature→vol cross-asset claim; 1 FAIL: EMPTY_EVIDENCE); bridge between Open Brain / experiment script results and the cert ecosystem; certifies verdict (CONSISTENT/CONTRADICTS/PARTIAL/INCONCLUSIVE) of an empirical finding against a named parent cert claim; domain-specific fail_ledger entries allowed when verdict=CONTRADICTS; failure algebra: UNKNOWN_OBSERVATION_SOURCE, INVALID_PARENT_CERT_REF, INVALID_VERDICT, CONTRADICTS_WITHOUT_FAIL_LEDGER, EMPTY_EVIDENCE | PASS |
| [128] | [QA Spread Period Cert](128_qa_spread_period.md) | 1 validator (SP1–SP5) + 3 fixtures (2 PASS: m=9 period=24, m=7 period=16; 1 FAIL: PISANO_PERIOD_MISMATCH+MATRIX_PERIOD_WRONG — claimed period=12 for m=9, F^12=-I≢I); certifies QA cosmos orbit period = π(m) = Pisano period of Fibonacci sequence mod m = ord(F) in GL₂(Z/mZ); F_matrix^π(m)≡I; spread polynomial S_n(s) cycles after π(m) steps; failure algebra: SCHEMA_VERSION_WRONG, PISANO_PERIOD_MISMATCH, MATRIX_PERIOD_WRONG, PERIOD_NOT_MINIMAL, ORBIT_TYPE_MISMATCH | PASS |
| [127] | [QA UHG Null Cert](127_qa_uhg_null.md) | 1 validator (UN1–UN7) + 3 fixtures (2 PASS: d=2e=1 → 3-4-5 null point [3:4:5], d=3e=2 → 5-12-13 null point [5:12:13]; 1 FAIL: BLUE_QUADRANCE_MISMATCH+NULL_CONDITION_VIOLATED+GAUSSIAN_DECOMP_MISMATCH — G=6 instead of 5); certifies every QA triple (F,C,G)=(d²-e²,2de,d²+e²) is a null point in UHG satisfying F²+C²-G²=0; Gaussian integer interpretation Z=d+ei: F=Re(Z²), C=Im(Z²), G=|Z|²; QA triples = absolute conic of UHG; failure algebra: SCHEMA_VERSION_WRONG, GREEN_QUADRANCE_MISMATCH, RED_QUADRANCE_MISMATCH, BLUE_QUADRANCE_MISMATCH, NULL_CONDITION_VIOLATED, GAUSSIAN_DECOMP_MISMATCH, NULL_QUADRANCE_WRONG | PASS |
| [126] | [QA Red Group Cert](126_qa_red_group.md) | 1 validator (RG1–RG7) + 3 fixtures (2 PASS: m=9 cosmos period=24, m=3 cosmos period=8; 1 FAIL: ORBIT_PERIOD_WRONG — claimed period=12 but F^12=-I≢I for m=9); certifies QA T-operator = Fibonacci shift F=[[0,1],[1,1]] = red isometry by φ in Z[√5]/mZ[√5]; det(F)=-1=N_red(φ), trace(F)=1; orbit period = ord(F) in GL₂(Z/mZ); affine period equals linear order when F^{P/2}=-I (proven for m=3,9); failure algebra: SCHEMA_VERSION_WRONG, T_MATRIX_WRONG, DET_NOT_MINUS_ONE, TRACE_NOT_ONE, ORBIT_PERIOD_WRONG, PERIOD_NOT_MINIMAL, ORBIT_TYPE_MISMATCH | PASS |
| [125] | [QA Chromogeometry Cert](125_qa_chromogeometry.md) | 1 validator (CG1–CG7) + 3 fixtures (2 PASS: 3-4-5 hyperbola b=1e=1, 20-21-29 ellipse b=3e=2; 1 FAIL: GREEN_QUADRANCE_MISMATCH+PYTHAGORAS_VIOLATED — C=2·b·e instead of 2·d·e); certifies C=Q_green(d,e)=2de, F=Q_red(d,e)=d²-e²=ab, G=Q_blue(d,e)=d²+e²; C²+F²=G² is Wildberger Chromogeometric Theorem 6; I=|C-F| conic discriminant (C>F→hyperbola, C=F→parabola, C<F→ellipse); failure algebra: SCHEMA_VERSION_WRONG, GREEN_QUADRANCE_MISMATCH, RED_QUADRANCE_MISMATCH, BLUE_QUADRANCE_MISMATCH, PYTHAGORAS_VIOLATED, SEMI_LATUS_MISMATCH, CONIC_TYPE_MISMATCH | PASS |
| [124] | [QA Security Competency Cert](124_qa_security_competency_cert.md) | 1 validator (SC1-SC11) + 3 fixtures (2 PASS: ml_kem membrane/fips_final/FIPS-203, ed25519 identity/classical_only+migration_path; 1 FAIL: SC5_PQ_MIGRATION_REQUIRED — rsa_1024_legacy identity/classical_only no migration path); extends [123] with immune system architecture (security_role, immune_function, pq_readiness); SC5 quantum resilience invariant: identity/membrane + classical_only → pq_migration_path required; SC6: fips_final → nist_fips required; SC9 CELL_ORBIT_MISMATCH inherited; failure algebra: SCHEMA_VERSION_MISMATCH, UNKNOWN_SECURITY_ROLE, UNKNOWN_IMMUNE_FUNCTION, UNKNOWN_PQ_READINESS, PQ_MIGRATION_REQUIRED, MISSING_FIPS_DESIGNATION, EMPTY_FAILURE_MODES, EMPTY_COMPOSITION_RULES, CELL_ORBIT_MISMATCH, GOAL_TOO_SHORT, RESULT_MISMATCH | PASS |
| [123] | [QA Agent Competency Cert](123_qa_agent_competency_cert.md) | 1 validator (V1-V10) + 3 fixtures (2 PASS: merge_sort_agent cosmos/differentiated/guaranteed, gradient_descent_agent mixed/progenitor/conditional; 1 FAIL: CELL_ORBIT_MISMATCH — stem agent declaring cosmos orbit); formalizes Levin morphogenetic agent architecture for QA Lab; machine-checks that (orbit_signature, levin_cell_type) are consistent (differentiated↔cosmos, progenitor↔satellite/mixed, stem↔singularity); failure algebra: SCHEMA_VERSION_MISMATCH, UNKNOWN_COGNITIVE_HORIZON, UNKNOWN_CONVERGENCE_TYPE, UNKNOWN_ORBIT_SIGNATURE, UNKNOWN_LEVIN_CELL_TYPE, EMPTY_FAILURE_MODES, EMPTY_DEDIFFERENTIATION_COND, CELL_ORBIT_MISMATCH, GOAL_TOO_SHORT, EMPTY_COMPOSITION_RULES | PASS |
| [121] | [QA Engineering Core Cert](121_qa_engineering_core_cert.md) | 1 schema + validator (IH1–IH3 + EC1–EC11) + 3 fixtures (1 PASS: spring-mass-damper still/transient/steady_oscillation mapped to singularity/satellite/cosmos mod-9, k=2, target_r=2 not obstructed; 1 FAIL: same system with target (b=1,e=3) giving r=3, v₃(3)=1 inert — ARITHMETIC_OBSTRUCTION_IGNORED; 1 FAIL: state encoded at b=0 outside QA domain — STATE_ENCODING_INVALID); family_extension of [107]; certifies classical engineering systems (state-space models, Lyapunov stability, Kalman controllability) against QA arithmetic; key contribution EC11: v_p(target_r)=1 for inert prime overrides classical full-rank controllability; failure algebra: INVALID_KERNEL_REFERENCE, SPEC_SCOPE_MISMATCH, GATE_POLICY_INCOMPATIBLE, STATE_ENCODING_INVALID, TRANSITION_NOT_GENERATOR, FAILURE_TAXONOMY_INCOMPLETE, TARGET_NOT_ORBIT_FAMILY, ORBIT_FAMILY_CLASSIFICATION_FAILURE, LYAPUNOV_QA_MISMATCH, CONTROLLABILITY_QA_MISMATCH, ARITHMETIC_OBSTRUCTION_IGNORED | PASS |
| [133] | [QA Eisenstein Cert](133_qa_eisenstein.md) | 1 validator (EIS_1-EIS_7+EIS_W/U) + 2 fixtures (PASS: fundamental (1,1,2,3) gives F=3,W=8,Z=7,Y=5 both Eisenstein triples 49=7²; 6-witness general with algebraic proof via u=b²+3be); certifies F²-FW+W²=Z² and Y²-YW+W²=Z² for ALL QA tuples; W=equilateral side, Z=Eisenstein companion per QA Law 15; ℤ[ω] norm N(F+Wω)=Z²; classical (3,5,7) Eisenstein triple appears at b=e=1 | PASS |
| [132] | [QA HAT Cert](132_qa_hat.md) | HAT₁=e/d=C/(G+F), HAT₂=(d-e)/(d+e)=F/(G+C), spread s=E/G=HAT₁²/(1+HAT₁²); Fibonacci box [[e,d-e],[d,d+e]]; certifies H. Lee Price 2008 half-angle tangents = QA direction ratios; Price Pythagorean tree = QA Koenig tree in HAT coordinates; checks HAT_1-8+HAT_W/F; 2 PASS (anchor 3-4-5, 5 witnesses d≤5) | PASS |
| [130] | [QA Origin of 24 Cert](130_qa_origin_of_24.md) | 1 validator (O24_1-O24_9 + O24_G/W/F/D) + 2 fixtures (PASS: anchor 3-4-5 fundamental direction, general theorem 6 witnesses d≤5); certifies dual derivation of mod-24: H²-G²=G²-I²=2CF where C=2de (green quadrance) and F=d²-e² (red quadrance); always divisible by 24 for all primitive Pythagorean directions; minimum=24 at (d,e)=(2,1) ↔ π(9)=24 Pisano period (same 24, not coincidence); Ben Iverson's Pyth-1 + QA-4 Crystal routes | PASS |
| [129] | [QA Projection Obstruction Cert](129_qa_projection_obstruction_cert.md) | 1 validator (IH1-IH3 + PO1-PO9) + 3 fixtures (1 PASS: Arto ternary lawful natively, representation debt explicit, physical realization UNASSESSED/INCONCLUSIVE; 2 FAIL: physical conflation under UNASSESSED status, unresolved invariant reference); family_extension of [121]; certifies the distinction between native symbolic closure, discrete representation-basis mismatch, and physical device realization so representation debt is not misreported as device failure; failure algebra: INVALID_PARENT_ENGINEERING_REFERENCE, SPEC_SCOPE_MISMATCH, GATE_POLICY_INCOMPATIBLE, EMPTY_NATIVE_INVARIANTS, NATIVE_WITNESS_INVALID, INVARIANT_REFERENCE_UNRESOLVED, REPRESENTATION_LAYER_INVALID, REPRESENTATION_LEDGER_MISMATCH, REPRESENTATION_VERDICT_MISMATCH, PHYSICAL_LAYER_INVALID, PHYSICAL_LEDGER_MISMATCH, PHYSICAL_VERDICT_MISMATCH, OBSTRUCTION_LEDGER_REQUIRED, OVERALL_VERDICT_MISMATCH, plus domain tags STATE_SPACE_RESIDUAL, COST_INFLATION, SELECTOR_AND_MERGE_DEBT, TOPOLOGY_PART_COUNT_DEBT, INSUFFICIENT_STABLE_STATES, THRESHOLD_MARGIN_WEAK, NOISE_MARGIN_WEAK, FANOUT_LIMITED, TIMING_UNVERIFIED | PASS |
| [131] | [QA Prime Bounded Certificate Scaling Cert](131_qa_prime_bounded_certificate_scaling_cert.md) | 1 schema + validator (7 gates: schema, canonical hash, artifact parity, endpoint ordering, row recomputation, row-flag honesty, overall result honesty) + 2 validator-valid fixtures (PASS using the real `100,250,500,1000` artifact; FAIL using a mock `500 -> 17 != 19` artifact); formalizes the empirical claim that the observed minimal passing `prime_max` matches `largest_prime_leq_sqrt_end` on the tested endpoints only | PASS |

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
