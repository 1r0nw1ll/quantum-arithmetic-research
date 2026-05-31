# QA Project Activity Log — Last 10 Days
**Generated:** 2026-05-31 13:24 UTC  |  **Period:** 2026-05-21 → 2026-05-31

## Summary
| Source | Count |
|--------|-------|
| Open Brain entries | 0 |
| Git commits | 27 |
| Experiment result files | 5 |
| qa-collab events | 0 |
| Modified/new source files | 4208 |
| **Total log entries** | **32** |

## Activity by Agent
- **will/codex**: 27 entries
- **unknown**: 4 entries
- **codex**: 1 entries

## Git Commits

- `42171399` `2026-05-30` — experiments(qa_cert): QA I Ching Hexagram Orbit cert [286]
- `2ebc634c` `2026-05-30` — experiments(qa_cert): QA I Ching Trigram Orbit cert [285] — Kun=0 A1-excluded; Dui=3 and Xun=6 Satellite; no Singularity
- `18ddc341` `2026-05-30` — experiments(qa_cert): QA RT Quadrance Orbit cert [283] — v3(G)=2*v3(gcd(b,e)) for all 81 pairs
- `45f9bfbf` `2026-05-30` — experiments(qa_cert): QA Fibonacci-Orbit Index cert [282] — F_n class determined by n mod 12 (Wall 1960)
- `df4a272c` `2026-05-30` — experiments(qa_cert): QA Pisano-Orbit Correspondence cert [281] — pi(3)=8=Satellite, pi(9)=24=Cosmos
- `f1f97bac` `2026-05-30` — cert(qa_nuclear_magic_orbit)[280]: a=126 uniquely has Sat+Sing among magic numbers
- `2c576da3` `2026-05-30` — cert(qa_orbit_access_theorem)[279]: Satellite iff 3
- `c6fc32e7` `2026-05-30` — experiments(geometry): Pollard's rho layer between Fermat and MPQS
- `74a32852` `2026-05-30` — feat(qa_mpqs): QA-Fermat engine with orbit pruning as primary factoring path
- `29fdff0c` `2026-05-30` — fix(qa_mpqs): correct A-value target to Silverman optimal sqrt(2F/M)
- `69a58a2a` `2026-05-30` — experiments(geometry): QA-MPQS sparse LA — Wiedemann BM GF(2) nullspace
- `1b32a0e5` `2026-05-25` — experiments(qa_network_routing): QA Substrate Ladder Level 4 — network/routing benchmark
- `1a6c1f11` `2026-05-28` — scan(sota): biweekly SOTA scan #7 — 2026-05-28 (window May 25-28)
- `80e6992f` `2026-05-27` — health: daily cert health report 2026-05-27
- `aa08d173` `2026-05-26` — health: daily cert health report 2026-05-26
- `5200dc99` `2026-05-25` — health: daily cert health report 2026-05-25
- `d6e78a0d` `2026-05-25` — scan(sota): biweekly SOTA scan #6 — 2026-05-25 (window May 21-25)
- `396d7513` `2026-05-25` — experiments(qa_ml): POSE-1 Table 4.2 Cambridge column — closes 13/13 rows; GCAN weak outdoor
- `0daee2f8` `2026-05-24` — experiments(qa_ml): POSE-1 Table 4.2 7-Scenes column — 5/7 strong, GCAN beats PoseNet by 26mm/2deg
- `68ab9647` `2026-05-24` — experiments(qa_runtime_scheduler): fix H4 adversarial_trap — bait-and-trap design
- `fcbb603a` `2026-05-24` — experiments(qa_runtime_scheduler): QA Substrate Ladder Level 3 — runtime/scheduler
- `fcbf2398` `2026-05-24` — fix(qa_vfs): proper H2 — two-tier corruption distinguishes law-recovery from record-read
- `62adfdea` `2026-05-24` — fix(qa_vfs): correct H2 — deviation record IS content, not unrecoverable
- `1d197490` `2026-05-24` — experiments(qa_vfs): QA Substrate Ladder Level 2 — virtual filesystem benchmark
- `891155c7` `2026-05-24` — experiments(qa_db): law-structured memory benchmark with falsifier modes
- `d8036961` `2026-05-24` — exp(qa_ml): colab notebook for real auto-gptq + autoawq CUDA run
- `1667c20d` `2026-05-22` — exp(qa_ml): import distilgpt2 QA packet-quant handoff + reproduce on Linux

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
- `Documents/methods_qa_friendly_encodings.tex`
- `Documents/results_full_merged.tex`
- `Documents/results_graphs_addendum.tex`
- `Documents/results_raman.tex`
- `Documents/results_section.tex`
- `Documents/synopsis_for_dale_pond_2026-03-27.md`
- `Documents/table_flagship_results.tex`
- `Documents/table_football.tex`
- `Documents/table_manifolds_kmeans.tex`
- `Documents/table_qa_knowledge_graph.tex`
- *(+4128 more)*

## Experiment Results (by date)

- `2026-05-30T12:17` whittaker_em_direction_sweep.json: 
- `2026-05-30T12:17` whittaker_phase_packet_raman_probe.json: {"best_ablation": "phase_no_intensity_features", "best_ablation_mean_accuracy": 0.4967456005212561, "phase_mean_accuracy": 0.4956397258988079, "phase_minus_raw
- `2026-05-30T12:17` whittaker_em_qa_observer_null_test.json: {"alpha": 0.05, "block_bootstrap_rejects": [{"channel": "unwrapped_phase", "observables": ["mean_return_time"], "ordering": "raster_frequency_order"}, {"channe
- `2026-05-30T12:17` qa_hysteresis_real_loop_observer.json: {"leakage_controls": "H/B quantile edges, marginal shell centers, and joint QA state centers are fit on calibration loops only. Direct predictors use no held-out
- `2026-05-30T12:17` qa_prime_bounded_certificate_scaling_experiment.json: PASS

## Open Brain Timeline


## Open Brain Entry Type Breakdown
