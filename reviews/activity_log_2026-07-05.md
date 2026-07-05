# QA Project Activity Log — Last 10 Days
**Generated:** 2026-07-05 13:23 UTC  |  **Period:** 2026-06-25 → 2026-07-05

## Summary
| Source | Count |
|--------|-------|
| Open Brain entries | 0 |
| Git commits | 47 |
| Experiment result files | 12 |
| qa-collab events | 0 |
| Modified/new source files | 6223 |
| **Total log entries** | **59** |

## Activity by Agent
- **will/codex**: 47 entries
- **unknown**: 11 entries
- **codex**: 1 entries

## Git Commits

- `9b625b01` `2026-07-05` — chore(health): daily QA cert health report 2026-07-05
- `41399d0e` `2026-07-04` — feat(cert:[516]): Witt Tower discrimination ladder AR(1)-baseline audit
- `0a6b0154` `2026-07-04` — feat(cert:[515]): QA orbit-derived NTRU keys weak when modulus divisible by 3
- `0860d59c` `2026-07-03` — chore(cert:[228]): refresh determinism fixture after [496] registry edit
- `88062a00` `2026-07-03` — fix(cert:[496]): downgrade E8 Satellite Chamber theorem, retract 3 checks
- `5ca3808f` `2026-07-03` — experiments(geometry): fix derive_from_f_qa_mpqs missing Fermat fast-path
- `c19121ed` `2026-07-03` — experiments(geometry): switch qa_fermat_factor to mod-8-only pruning (1.4-2.5x faster)
- `118c253b` `2026-07-03` — experiments(geometry): add ECM stage-2 continuation, re-tune defaults
- `e5b5f5ea` `2026-07-03` — experiments(geometry): fix qa_fermat_factor infinite loop, add difference-of-squares fast reject
- `56aeeca4` `2026-07-03` — experiments(geometry): tune ECM defaults for better auto-pipeline success odds
- `955c6f46` `2026-07-03` — experiments(geometry): add Lenstra ECM bridge to the Pythagorean auto-factoring pipeline
- `1d70546e` `2026-07-03` — docs: investigate scalar EM claim boundaries
- `14db50cd` `2026-07-03` — fix(cert:[514]): clarify scalar EM boundary is not disproof
- `c4bd70c0` `2026-07-03` — feat(cert:[514]): bound longitudinal scalar EM claims
- `dc67eee5` `2026-07-03` — feat(cert:[513]): bounded Maxwell derivation assembly
- `37d8e200` `2026-07-03` — feat(cert:[512]): recover inhomogeneous Maxwell equation
- `aeb60580` `2026-07-03` — feat(cert:[510],[511]): add native hodge and source-carrier seeds
- `9502a3df` `2026-07-03` — feat(cert:[511]): source continuity from declared current
- `031187ca` `2026-07-03` — chore(quarantine): approve temp scratchpad packets
- `46605d75` `2026-07-03` — feat(cert:[510]): Hodge constitutive boundary gate
- `de480031` `2026-07-03` — merge: integrate remote automated health-check/scan commits
- `83ee397a` `2026-07-03` — feat(qa_ml): NT-compliant zero-float BitNet training matches/beats float baseline
- `47bac690` `2026-07-03` — feat(cert:[509]): field two-form Bianchi identity
- `18295c56` `2026-07-03` — feat(cert:[508]): discrete exterior nilpotency
- `46218b72` `2026-07-03` — docs(maxwell): scaffold QA derivation proof program
- `b5c59df5` `2026-07-03` — docs(whittaker): map Layer 5 and Mie claim boundaries
- `adc362ac` `2026-07-03` — feat(cert:[507]): scaffold Whittaker two-scalar bridge
- `374582c4` `2026-07-03` — health: daily cert check 2026-07-03 — 1 linter error, [31] FAIL (Lean missing)
- `8f6ddd9f` `2026-07-02` — health: daily cert check 2026-07-02 — 1 linter error, [31] FAIL (Lean missing)
- `188305b1` `2026-07-02` — sota: scan #17 — 2026-07-02, 3 HIGH finds in NN modular arithmetic cluster
- `94f74a81` `2026-07-01` — feat(eet): add grind proofs for all 5 failing Stage 2 problems
- `ee8cd8f6` `2026-06-29` — Extend QA orbit pack disjointness coverage
- `f77627ae` `2026-06-29` — Extend QA orbit pack cardinality coverage
- `56ede644` `2026-06-29` — chore(igp24): trim generator whitespace
- `7d9ee73a` `2026-06-29` — Use recorded kernel certs in default meta sweep
- `9dc1b8ee` `2026-06-29` — Make Witt Tower crash-pair validator deterministic
- `66bd088d` `2026-06-29` — Record green QA meta-validator checkpoint
- `978468ff` `2026-06-25` — feat(igp24): batches 30-36, cyclotomic strategy pivot (700 polys)
- `142eec19` `2026-06-25` — feat(igp24): batch29 — 100 cyclotomic degree-24 polys, m=3003–7560
- `2fc33495` `2026-06-25` — feat(igp24): batch28 — r=0 polys from complex degree-8/12 seeds, more cyclotomic
- `079f1da9` `2026-06-25` — feat(igp24): batch26/27 generators — fixed PARI VEC handling, new T-numbers
- `03093e77` `2026-06-29` — chore(health): daily QA cert health report 2026-06-29
- `9cd783d0` `2026-06-28` — chore(review): add activity log artifacts for 2026-06-28
- `9a3ca6cb` `2026-06-28` — docs: weekly review 2026-06-28
- `38bb5fd1` `2026-06-28` — chore(health): daily QA cert health report 2026-06-28
- `e45569c0` `2026-06-27` — chore(health): daily QA cert health report 2026-06-27
- `f253ba96` `2026-06-26` — chore(health): daily QA cert health report 2026-06-26

## Modified / New Files

- `.claude/agents/qa-cert-auditor.md`
- `.claude/hooks/collab_broadcast_edit.sh`
- `.claude/hooks/collab_commit_gate.sh`
- `.claude/hooks/collab_file_lock_check.sh`
- `.claude/hooks/collab_session_gate.sh`
- `.claude/hooks/pretool_guard.sh`
- `.claude/hooks/qa_axiom_check.sh`
- `.claude/hooks/session_start.sh`
- `.claude/skills/cert-new/SKILL.md`
- `.claude/skills/cert-status/SKILL.md`
- `.claude/skills/ob-status/SKILL.md`
- `.claude/skills/validate/SKILL.md`
- `.claude/skills/weekly-review/SKILL.md`
- `.claude/statusline.sh`
- `.openclaw/config.example.json`
- `.openclaw/extensions/qa-guardrail/README.md`
- `.openclaw/extensions/qa-guardrail/package.json`
- `46_seismic_expanded.py`
- `46_seismic_expanded_results.json`
- `46_seismic_surrogate_results.json`
- `46_seismic_surrogates.py`
- `46_seismic_topographic_observer.py`
- `47_climate_topographic_observer.py`
- `48_teleconnection_results.json`
- `48_teleconnection_surrogate_v2_results.json`
- `48_teleconnection_surrogates.py`
- `48_teleconnection_surrogates_v1_CIRCULAR_BUG.py`
- `48_teleconnection_topographic_observer.py`
- `49_era5_multilayer.py`
- `49_era5_multilayer_results.json`
- `49_forecast_coherence_observer.py`
- `49_forecast_coherence_surrogate_v2_results.json`
- `49_forecast_coherence_surrogates.py`
- `49_forecast_coherence_surrogates_v1_CIRCULAR_BUG.py`
- `50_cardiac_preregistered.py`
- `50_cardiac_preregistered_results.json`
- `51_bearing_preregistered.py`
- `51_bearing_preregistered_results.json`
- `52_network_preregistered.py`
- `52_network_preregistered_results.json`
- `53_emg_preregistered.py`
- `53_emg_preregistered_results.json`
- `54_qa_timeseries_invariants.py`
- `55_qa_synchronous_harmonics.py`
- `56_qa_witt_tower_oracle.py`
- `57_qa_dilution_detector.py`
- `58_qa_wss_orbit_characterisation.py`
- `59_qa_filter_bank.py`
- `60_qa_witt_empirical_validation.py`
- `61_qa_witt_financial_filterbank.py`
- `62_qa_witt_multi_instrument.py`
- `63_qa_witt_rolling_window.py`
- `AGENTS.md`
- `CHANGELOG.md`
- `CLAUDE.md`
- `CONSTITUTION.md`
- `CONTRIBUTING.md`
- `DEMO.md`
- `Documents/BUILD_A_BRAIN_PROMPT_KIT.md`
- `Documents/MULTIMODAL_FUSION_OVERVIEW.md`
- `Documents/OPEN_BRAIN_COMPANION_PROMPTS.md`
- `Documents/PROJECT_FORENSICS_CONSOLIDATION.md`
- `Documents/QA_ADAPTER_PATTERN_SPEC.md`
- `Documents/QA_CARTOGRAPHY_MAP.md`
- `Documents/QA_EXECUTION_MAP.md`
- `Documents/QA_MAPPING_PROTOCOL__ARTEXPLORER_SCENE_ADAPTER.v1.json`
- `Documents/QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json`
- `Documents/QA_MAPPING_PROTOCOL__GEOGEBRA_SCENE_ADAPTER.v1.json`
- `Documents/QA_MAPPING_PROTOCOL__RATIONAL_TRIG_TYPE_SYSTEM.v1.json`
- `Documents/QA_MAPPING_PROTOCOL__THREEJS_SCENE_ADAPTER.v1.json`
- `Documents/QA_MAP__EBM_REASONING_KONA_PODCAST.md`
- `Documents/QA_META_MAP.md`
- `Documents/QA_ONTOLOGY_MAP.md`
- `Documents/QA_SYNTHESIS_2026-03-27.md`
- `Documents/QA_TOOLCHAIN_MAP.md`
- `Documents/RESULTS_CURATED.md`
- `Documents/RESULTS_REGISTRY.md`
- `Documents/canonical_expansion_v2.tex`
- `Documents/figure_captions.tex`
- `Documents/letter_to_dale_pond_from_claude_2026-03-27.md`
- *(+6143 more)*

## Experiment Results (by date)

- `2026-06-30T12:17` whittaker_phase_packet_raman_probe.json: {"best_ablation": "phase_no_intensity_features", "best_ablation_mean_accuracy": 0.4967456005212561, "phase_mean_accuracy": 0.4956397258988079, "phase_minus_raw
- `2026-06-30T12:17` whittaker_em_direction_sweep.json: 
- `2026-06-30T12:17` whittaker_em_qa_observer_null_test.json: {"alpha": 0.05, "block_bootstrap_rejects": [{"channel": "unwrapped_phase", "observables": ["mean_return_time"], "ordering": "raster_frequency_order"}, {"channe
- `2026-06-30T12:17` qa_prime_bounded_certificate_scaling_experiment.json: PASS
- `2026-06-30T12:17` qa_hysteresis_real_loop_observer.json: {"leakage_controls": "H/B quantile edges, marginal shell centers, and joint QA state centers are fit on calibration loops only. Direct predictors use no held-out
- `2026-06-30T12:17` eeg_feat_plv.json: 
- `2026-06-30T12:17` eeg_feat_coherence_ags.json: 
- `2026-06-30T12:17` eeg_feat_harmonics.json: 
- `2026-06-30T12:17` eeg_feat_entropy.json: 
- `2026-06-30T12:17` eeg_feat_combined.json: 
- `2026-06-30T12:17` eeg_feat_multiband.json: 
- `2026-06-30T12:17` eeg_feat_combined2.json: 

## Open Brain Timeline


## Open Brain Entry Type Breakdown
