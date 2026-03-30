# Results Registry (Auto-Generated)

- Generated: `2026-02-16T00:02:46.218229+00:00`
- Forensics input: `/home/player2/signal_experiments/_forensics/forensics_20260216_000134`
- Control-theorem cutoff (UTC date): `2026-01-10` (pre-cutoff = revet required)
- This is a *triage index*: it ranks likely “result nodes” using keyword hotspots and script→artifact links.

## How to update
- Re-run forensics: `python tools/project_forensics.py`
- Re-generate this file: `python tools/generate_results_registry.py`

## Hotspots (claim/evidence keyword density)
### R-001: qa_meta_validator.py
- Path: `qa_alphageometry_ptolemy/qa_meta_validator.py`
- Category: `Meta-validator`
- Hotspot score: `640` (evidence=210, claims=10)
- File: exists=`1` tracked=`1` revet_required=`0` size=`144.3 KiB` mtime=`2026-02-15` chat_mentions=`260`
- Linked artifacts (from script strings):
  - `qa_alphageometry_ptolemy/certs/QA_ARAG_INTERFACE_CERT.v1.json`
  - `qa_alphageometry_ptolemy/certs/QA_DATASTORE_SEMANTICS_CERT.v1.json`
  - `qa_alphageometry_ptolemy/certs/QA_DATASTORE_VIEW_CERT.v1.json`
  - `qa_alphageometry_ptolemy/certs/QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.json`
  - `qa_alphageometry_ptolemy/certs/QA_GRAPH_STRUCTURE_BUNDLE.v1.json`
  - `qa_alphageometry_ptolemy/certs/QA_INGEST_SEMANTICS_CERT.v1.json`
  - `qa_alphageometry_ptolemy/certs/QA_INGEST_VIEW_BRIDGE_CERT.v1.json`
  - `qa_alphageometry_ptolemy/certs/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.json`
  - (+11 more)
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_meta_validator.py --help`
  - `python qa_alphageometry_ptolemy/qa_meta_validator.py`
- Suggested evidence check:
  - `python qa_alphageometry_ptolemy/qa_meta_validator.py qa_alphageometry_ptolemy/certs/QA_ARAG_INTERFACE_CERT.v1.json`

### R-002: check this for accuracy, generalizations, extentio.md
- Path: `docs/ai_chats/check this for accuracy, generalizations, extentio.md`
- Category: `Imported notes (ai_chats)`
- Hotspot score: `475` (evidence=125, claims=100)
- File: exists=`1` tracked=`0` revet_required=`1` size=`641.8 KiB` mtime=`2025-11-12` chat_mentions=`0`

### R-003: qa_certificate.py
- Path: `qa_alphageometry_ptolemy/qa_certificate.py`
- Category: `Certificate schema`
- Hotspot score: `453` (evidence=140, claims=33)
- File: exists=`1` tracked=`1` revet_required=`0` size=`233.4 KiB` mtime=`2026-01-21` chat_mentions=`19`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_certificate.py`

### R-004: Unit tests for UnderstandingCertificate validity rules.
- Path: `qa_alphageometry_ptolemy/test_understanding_certificate.py`
- Category: `Certificate schema`
- Hotspot score: `344` (evidence=101, claims=41)
- File: exists=`1` tracked=`1` revet_required=`0` size=`188.1 KiB` mtime=`2026-01-21` chat_mentions=`30`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/test_understanding_certificate.py`

### R-005: miner49er100.txt
- Path: `docs/Google AI Studio/miner49er100.txt`
- Category: `Imported notes (Google AI Studio)`
- Hotspot score: `261` (evidence=85, claims=6)
- File: exists=`1` tracked=`0` revet_required=`1` size=`253.0 KiB` mtime=`2025-06-12` chat_mentions=`0`

### R-006: qa_topology_resonance_validator_v1.py
- Path: `qa_alphageometry_ptolemy/qa_topology_resonance_validator_v1.py`
- Category: `Validator`
- Hotspot score: `222` (evidence=74, claims=0)
- File: exists=`1` tracked=`1` revet_required=`0` size=`34.9 KiB` mtime=`2026-02-07` chat_mentions=`9`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_topology_resonance_validator_v1.py --demo`
  - `python qa_alphageometry_ptolemy/qa_topology_resonance_validator_v1.py --help`
  - `python qa_alphageometry_ptolemy/qa_topology_resonance_validator_v1.py`

### R-007: # Title: Quantum Arithmetic Graph Model
- Path: `docs/Google AI Studio/context.txt`
- Category: `Imported notes (Google AI Studio)`
- Hotspot score: `210` (evidence=37, claims=99)
- File: exists=`1` tracked=`0` revet_required=`1` size=`170.1 KiB` mtime=`2025-08-22` chat_mentions=`0`

### R-008: knowledgebase_qa(2).txt
- Path: `docs/Google AI Studio/knowledgebase_qa(2).txt`
- Category: `Imported notes (Google AI Studio)`
- Hotspot score: `208` (evidence=26, claims=130)
- File: exists=`1` tracked=`0` revet_required=`1` size=`333.1 KiB` mtime=`2025-07-20` chat_mentions=`0`

### R-009: knowledgebase_qa.txt
- Path: `docs/Google AI Studio/knowledgebase_qa.txt`
- Category: `Imported notes (Google AI Studio)`
- Hotspot score: `208` (evidence=26, claims=130)
- File: exists=`1` tracked=`0` revet_required=`1` size=`333.1 KiB` mtime=`2025-07-20` chat_mentions=`0`

### R-010: knowledgebase_qa(1).txt
- Path: `docs/Google AI Studio/knowledgebase_qa(1).txt`
- Category: `Imported notes (Google AI Studio)`
- Hotspot score: `208` (evidence=26, claims=130)
- File: exists=`1` tracked=`0` revet_required=`1` size=`333.1 KiB` mtime=`2025-07-20` chat_mentions=`0`

### R-011: QA Canonical Mapping Registry
- Path: `qa_alphageometry_ptolemy/QA_MAP_CANONICAL.md`
- Category: `QA core (ptolemy)`
- Hotspot score: `196` (evidence=60, claims=16)
- File: exists=`1` tracked=`1` revet_required=`0` size=`29.5 KiB` mtime=`2026-02-09` chat_mentions=`50`

### R-012: qa_graph_structure_validator_v1.py
- Path: `qa_alphageometry_ptolemy/qa_graph_structure_validator_v1.py`
- Category: `Validator`
- Hotspot score: `195` (evidence=65, claims=0)
- File: exists=`1` tracked=`1` revet_required=`0` size=`34.5 KiB` mtime=`2026-02-09` chat_mentions=`0`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_graph_structure_validator_v1.py --demo`
  - `python qa_alphageometry_ptolemy/qa_graph_structure_validator_v1.py --help`
  - `python qa_alphageometry_ptolemy/qa_graph_structure_validator_v1.py`

### R-013: Here’s an executable proof that QA‑style exact rational arithmetic (using canonical tuples and rational series) approxim
- Path: `docs/Google AI Studio/Validation of Structural Analysis on Deterministic and Fractal Graphs.txt`
- Category: `Imported notes (Google AI Studio)`
- Hotspot score: `192` (evidence=56, claims=24)
- File: exists=`1` tracked=`0` revet_required=`1` size=`135.7 KiB` mtime=`2025-09-30` chat_mentions=`0`

### R-014: qa_neuralgcm_validator_v3.py
- Path: `qa_alphageometry_ptolemy/qa_neuralgcm_validator_v3.py`
- Category: `Validator`
- Hotspot score: `183` (evidence=61, claims=0)
- File: exists=`1` tracked=`0` revet_required=`0` size=`26.6 KiB` mtime=`2026-01-24` chat_mentions=`0`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_neuralgcm_validator_v3.py --demo`
  - `python qa_alphageometry_ptolemy/qa_neuralgcm_validator_v3.py --help`
  - `python qa_alphageometry_ptolemy/qa_neuralgcm_validator_v3.py`

### R-015: Validate the QA Math Compiler stack artifacts.
- Path: `qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py`
- Category: `Validator`
- Hotspot score: `180` (evidence=55, claims=15)
- File: exists=`1` tracked=`1` revet_required=`0` size=`74.2 KiB` mtime=`2026-02-12` chat_mentions=`0`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py`

### R-016: # Title: AZR to QA Mapping
- Path: `docs/Google AI Studio/azr_qa.txt`
- Category: `Imported notes (Google AI Studio)`
- Hotspot score: `172` (evidence=40, claims=52)
- File: exists=`1` tracked=`0` revet_required=`1` size=`182.1 KiB` mtime=`2025-09-30` chat_mentions=`0`

### R-017: qa_elliptic_correspondence_validator_v3.py
- Path: `qa_alphageometry_ptolemy/qa_elliptic_correspondence_validator_v3.py`
- Category: `Validator`
- Hotspot score: `165` (evidence=55, claims=0)
- File: exists=`1` tracked=`1` revet_required=`0` size=`24.4 KiB` mtime=`2026-02-10` chat_mentions=`0`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_elliptic_correspondence_validator_v3.py --demo`
  - `python qa_alphageometry_ptolemy/qa_elliptic_correspondence_validator_v3.py --help`
  - `python qa_alphageometry_ptolemy/qa_elliptic_correspondence_validator_v3.py`

### R-018: Title: AI in the terminal
- Path: `docs/ai_chats/AI in the terminal.md`
- Category: `Imported notes (ai_chats)`
- Hotspot score: `162` (evidence=31, claims=69)
- File: exists=`1` tracked=`0` revet_required=`1` size=`93.5 KiB` mtime=`2025-11-12` chat_mentions=`0`

### R-019: QA Certificate Verifier CLI
- Path: `qa_alphageometry_ptolemy/qa_verify.py`
- Category: `QA core (ptolemy)`
- Hotspot score: `158` (evidence=51, claims=5)
- File: exists=`1` tracked=`1` revet_required=`0` size=`24.7 KiB` mtime=`2026-01-22` chat_mentions=`0`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_verify.py --demo`
  - `python qa_alphageometry_ptolemy/qa_verify.py --help`
  - `python qa_alphageometry_ptolemy/qa_verify.py`

### R-020: qa_sparse_attention_validator_v3.py
- Path: `qa_alphageometry_ptolemy/qa_sparse_attention_validator_v3.py`
- Category: `Validator`
- Hotspot score: `153` (evidence=51, claims=0)
- File: exists=`1` tracked=`0` revet_required=`0` size=`25.8 KiB` mtime=`2026-01-24` chat_mentions=`0`
- Suggested reproduction commands (best-effort):
  - `python qa_alphageometry_ptolemy/qa_sparse_attention_validator_v3.py --demo`
  - `python qa_alphageometry_ptolemy/qa_sparse_attention_validator_v3.py --help`
  - `python qa_alphageometry_ptolemy/qa_sparse_attention_validator_v3.py`

## Artifact-Producing Scripts (top)
### E-001: qa_build_pipeline.py
- Script: `qa_build_pipeline.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `10`
  - `qa_chunk_edges.json`
  - `qa_entities.json`
  - `qa_entities_discovered.json`
  - `qa_entities_merged.json`
  - `qa_entities_repo.json`
  - `qa_entities_seed.json`
  - `qa_entity_encodings.json`
  - `qa_lexicon_defines_edges.json`
  - `qa_typed_edges.json`
  - `qa_typed_edges_combined.json`
- Suggested run:
  - `python qa_build_pipeline.py --help`

### E-002: qa_fst_validate.py
- Script: `qa_alphageometry_ptolemy/qa_fst/qa_fst_validate.py` (exists=1, tracked=1, revet_required=0, chat_mentions=16)
- Artifacts: `9`
  - `qa_alphageometry_ptolemy/qa_fst/qa_fst_cert_bundle.json`
  - `qa_alphageometry_ptolemy/qa_fst/qa_fst_manifest.json`
  - `qa_alphageometry_ptolemy/qa_fst/qa_fst_module_spine.json`
  - `qa_alphageometry_ptolemy/qa_fst/qa_fst_submission_packet_spine.json`
  - `qa_alphageometry_ptolemy/qa_fst/schemas/FAIL_RECORD.v1.schema.json`
  - `qa_alphageometry_ptolemy/qa_fst/schemas/QA_CERT_BUNDLE.v1.schema.json`
  - `qa_alphageometry_ptolemy/qa_fst/schemas/QA_MAP_MODULE_SPINE.v1.schema.json`
  - `qa_alphageometry_ptolemy/qa_fst/schemas/QA_SHA256_MANIFEST.v1.schema.json`
  - `qa_alphageometry_ptolemy/qa_fst/schemas/QA_SUBMISSION_PACKET_SPINE.v1.schema.json`
- Suggested run:
  - `python qa_alphageometry_ptolemy/qa_fst/qa_fst_validate.py`

### E-003: HI 2.0 Ablation Study Visualization Suite
- Script: `hi_2_0_visualization.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `7`
  - `hi_2_0_ablation_results.csv`
  - `hi_2_0_ablation_summary.json`
  - `hi_2_0_figure1_performance_comparison.png`
  - `hi_2_0_figure2_gender_discrimination.png`
  - `hi_2_0_figure3_domain_heatmaps.png`
  - `hi_2_0_figure4_3d_component_space.png`
  - `hi_2_0_figure5_tuple_analysis.png`
- Suggested run:
  - `python hi_2_0_visualization.py`

### E-004: qa_kayser_validate.py
- Script: `qa_alphageometry_ptolemy/qa_kayser/qa_kayser_validate.py` (exists=1, tracked=1, revet_required=0, chat_mentions=38)
- Artifacts: `7`
  - `qa_alphageometry_ptolemy/qa_kayser/qa_kayser_basin_separation_cert.json`
  - `qa_alphageometry_ptolemy/qa_kayser/qa_kayser_conic_optics_cert.json`
  - `qa_alphageometry_ptolemy/qa_kayser/qa_kayser_lambdoma_cycle_cert.json`
  - `qa_alphageometry_ptolemy/qa_kayser/qa_kayser_manifest.json`
  - `qa_alphageometry_ptolemy/qa_kayser/qa_kayser_primordial_leaf_cert.json`
  - `qa_alphageometry_ptolemy/qa_kayser/qa_kayser_rhythm_time_cert.json`
  - `qa_alphageometry_ptolemy/qa_kayser/qa_kayser_tcross_generator_cert.json`
- Suggested run:
  - `python qa_alphageometry_ptolemy/qa_kayser/qa_kayser_validate.py --help`

### E-005: qa_elliptic_correspondence_bundle_v1.py
- Script: `qa_alphageometry_ptolemy/qa_elliptic_correspondence_bundle_v1.py` (exists=1, tracked=1, revet_required=0, chat_mentions=0)
- Artifacts: `5`
  - `qa_alphageometry_ptolemy/certs/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.json`
  - `qa_alphageometry_ptolemy/examples/elliptic_correspondence/elliptic_correspondence_ramification_failure.json`
  - `qa_alphageometry_ptolemy/examples/elliptic_correspondence/elliptic_correspondence_success.json`
  - `qa_alphageometry_ptolemy/schemas/QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.schema.json`
  - `qa_alphageometry_ptolemy/schemas/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.schema.json`
- Suggested run:
  - `python qa_alphageometry_ptolemy/qa_elliptic_correspondence_bundle_v1.py --help`

### E-006: qa_graph_structure_bundle_v1.py
- Script: `qa_alphageometry_ptolemy/qa_graph_structure_bundle_v1.py` (exists=1, tracked=1, revet_required=0, chat_mentions=0)
- Artifacts: `5`
  - `qa_alphageometry_ptolemy/certs/QA_GRAPH_STRUCTURE_CERT.v1.json`
  - `qa_alphageometry_ptolemy/examples/graph_structure/graph_structure_parity_failure.json`
  - `qa_alphageometry_ptolemy/examples/graph_structure/graph_structure_success.json`
  - `qa_alphageometry_ptolemy/schemas/QA_GRAPH_STRUCTURE_BUNDLE.v1.schema.json`
  - `qa_alphageometry_ptolemy/schemas/QA_GRAPH_STRUCTURE_CERT.v1.schema.json`
- Suggested run:
  - `python qa_alphageometry_ptolemy/qa_graph_structure_bundle_v1.py --help`

### E-007: qa_topology_resonance_bundle_v1.py
- Script: `qa_alphageometry_ptolemy/qa_topology_resonance_bundle_v1.py` (exists=1, tracked=1, revet_required=0, chat_mentions=12)
- Artifacts: `5`
  - `qa_alphageometry_ptolemy/certs/QA_TOPOLOGY_RESONANCE_CERT.v1.json`
  - `qa_alphageometry_ptolemy/examples/topology/topology_phase_break_failure.json`
  - `qa_alphageometry_ptolemy/examples/topology/topology_resonance_success.json`
  - `qa_alphageometry_ptolemy/schemas/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.schema.json`
  - `qa_alphageometry_ptolemy/schemas/QA_TOPOLOGY_RESONANCE_CERT.v1.schema.json`
- Suggested run:
  - `python qa_alphageometry_ptolemy/qa_topology_resonance_bundle_v1.py --help`

### E-008: # geometric_autopsy.py
- Script: `geometric_autopsy.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `4`
  - `1_angular_spectrum.png`
  - `2_tda_persistence_diagram.png`
  - `3a_dimensionality.png`
  - `3b_clustering.png`
- Suggested run:
  - `python geometric_autopsy.py`

### E-009: Rule 30 Center Column Non-Periodicity Witness Generator
- Script: `qa_alphageometry_ptolemy/rule30_submission_package/rule30_witness_generator.py` (exists=1, tracked=1, revet_required=0, chat_mentions=0)
- Artifacts: `4`
  - `qa_alphageometry_ptolemy/rule30_submission_package/center_rule30_T16384.txt`
  - `qa_alphageometry_ptolemy/rule30_submission_package/computation_summary.txt`
  - `qa_alphageometry_ptolemy/rule30_submission_package/witness_rule30_center_P1024_T16384.csv`
  - `qa_alphageometry_ptolemy/rule30_submission_package/witness_rule30_center_P1024_T16384.json`
- Suggested run:
  - `python qa_alphageometry_ptolemy/rule30_submission_package/rule30_witness_generator.py`

### E-010: Rule 30 Center Column Non-Periodicity Witness Generator
- Script: `qa_alphageometry_ptolemy/rule30_witness_generator.py` (exists=1, tracked=1, revet_required=1, chat_mentions=0)
- Artifacts: `4`
  - `qa_alphageometry_ptolemy/center_rule30_T16384.txt`
  - `qa_alphageometry_ptolemy/computation_summary.txt`
  - `qa_alphageometry_ptolemy/witness_rule30_center_P1024_T16384.csv`
  - `qa_alphageometry_ptolemy/witness_rule30_center_P1024_T16384.json`
- Suggested run:
  - `python qa_alphageometry_ptolemy/rule30_witness_generator.py`

### E-011: QA Hyperspectral Dashboard
- Script: `qa_hyperbolic_dashboard.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `4`
  - `hyperspectral_comparison.csv`
  - `hyperspectral_comparison.png`
  - `spread_periodicity.png`
  - `spread_periodicity_analysis.csv`
- Suggested run:
  - `python qa_hyperbolic_dashboard.py`

### E-012: #!/usr/bin/env python3
- Script: `qa_modal_reachability.py` (exists=1, tracked=0, revet_required=0, chat_mentions=0)
- Artifacts: `4`
  - `cycle_impossible_ir_hsi.json`
  - `cycle_impossible_rgb_hsi.json`
  - `return_constructed_hsi_ir.json`
  - `return_constructed_hsi_lidar.json`
- Suggested run:
  - `python qa_modal_reachability.py --help`

### E-013: Piezoelectric Tensor Visualizations for Quartz
- Script: `quartz_piezo_tensor_viz.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `4`
  - `quartz_converse_effect.png`
  - `quartz_coupling_modes.png`
  - `quartz_energy_efficiency.png`
  - `quartz_piezo_tensor_3d.png`
- Suggested run:
  - `python quartz_piezo_tensor_viz.py`

### E-014: qa_generate_seed_candidates.py
- Script: `qa_generate_seed_candidates.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `3`
  - `qa_entities_discovered.json`
  - `qa_entities_merged.json`
  - `qa_seed_candidates.json`
- Suggested run:
  - `python qa_generate_seed_candidates.py --help`

### E-015: qa_lexicon_defines.py
- Script: `qa_lexicon_defines.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `3`
  - `qa_entities.json`
  - `qa_entities_merged.json`
  - `qa_lexicon_defines_edges.json`
- Suggested run:
  - `python qa_lexicon_defines.py --help`

### E-016: QA Multi-AI Collaborative Orchestrator
- Script: `qa_multi_ai_orchestrator.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `3`
  - `qa_dataset_placeholder.csv`
  - `qa_demo_graph.pt`
  - `qa_embeddings.pt`
- Suggested run:
  - `python qa_multi_ai_orchestrator.py --demo`

### E-017: Quantum-Phonon Coupling Simulation for Helium-Doped Quartz Piezoelectric System
- Script: `quartz_quantum_phonon_coupling.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `3`
  - `quartz_coupled_dynamics.png`
  - `quartz_phonon_spectrum.png`
  - `quartz_power_output.png`
- Suggested run:
  - `python quartz_quantum_phonon_coupling.py`

### E-018: Create comparison table and visualizations for QA vs baseline methods
- Script: `create_comparison_report.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `2`
  - `results/comparison_table.csv`
  - `results/comparison_visualization.png`
- Suggested run:
  - `python create_comparison_report.py`

### E-019: QA Decision Certificate Spine Demo
- Script: `demos/decision_spine_demo.py` (exists=1, tracked=1, revet_required=0, chat_mentions=3)
- Artifacts: `2`
  - `demos/spine_bundle.json`
  - `demos/spine_manifest.json`
- Suggested run:
  - `python demos/decision_spine_demo.py`

### E-020: Diagnostic script to investigate low variance in (b,e) encoding
- Script: `diagnose_encoding_variance.py` (exists=1, tracked=0, revet_required=1, chat_mentions=0)
- Artifacts: `2`
  - `results/encoding_diagnostic_summary.json`
  - `results/encoding_variance_analysis.png`
- Suggested run:
  - `python diagnose_encoding_variance.py`
