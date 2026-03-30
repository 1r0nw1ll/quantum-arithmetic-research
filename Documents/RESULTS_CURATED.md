# Results Registry (Curated)

This file is **manual**.

Use it to record “claim → evidence → reproduce” entries that should not be overwritten by automation.

Related:
- Auto triage index (overwritten on regen): `Documents/RESULTS_REGISTRY.md`
- Toolchain map: `Documents/QA_TOOLCHAIN_MAP.md`

## Entry template

Copy/paste and fill:

```
### C-<YYYYMMDD>-<short_id>: <title>

- Claim:
- Evidence:
  - (paths to certs/logs/plots; prefer validator-verifiable artifacts)
- Reproduce:
  - (exact commands + working directory)
- Source:
  - (primary source files)
- Status: draft | verified | superseded
- Notes:
```

## Entries

### C-20260215-rule30-nonperiod-v2: Rule 30 center-column nonperiodicity (P≤1024) up to T=65536

- Claim: No period `p ∈ [1,1024]` detected for Rule 30 center column at any `T ∈ {4096,8192,16384,32768,65536}` (single-1 initial condition), up to `T_max=65536` (verified `5120/5120`, failures `0`).
- Evidence:
  - `qa_alphageometry_ptolemy/qa_rule30/certpacks/rule30_nonperiodicity_v2/QA_RULE30_NONPERIODICITY_CERT.v1.json`
  - `qa_alphageometry_ptolemy/qa_rule30/certpacks/rule30_nonperiodicity_v2/witnesses/T16384/MANIFEST.json`
  - `trace/runs/20260215_012337_TOOL.revett.rule30_cert_v2_validator.v1/stdout.log`
  - `trace/runs/20260215_012347_TOOL.revett.rule30_manifest_T16384_verify.v1/stdout.log`
  - `trace/runs/20260215_012400_TOOL.revett.rule30_certpack_v2_T16384_independent_verify.v1/stdout.log`
  - Legacy artifact integrity (T=16384):
    - `qa_alphageometry_ptolemy/rule30_submission_package/computation_summary.txt`
    - `qa_alphageometry_ptolemy/witness_rule30_center_P1024_T16384.csv`
    - `qa_alphageometry_ptolemy/witness_rule30_center_P1024_T16384.json`
    - `trace/runs/20260215_012420_TOOL.revett.rule30_legacy_witness_hashes.v1/stdout.log`
- Reproduce (from repo root):
  - `python qa_alphageometry_ptolemy/qa_rule30/qa_rule30_cert_validator.py cert qa_alphageometry_ptolemy/qa_rule30/certpacks/rule30_nonperiodicity_v2/QA_RULE30_NONPERIODICITY_CERT.v1.json`
  - `python qa_alphageometry_ptolemy/qa_rule30/qa_rule30_cert_validator.py manifest qa_alphageometry_ptolemy/qa_rule30/certpacks/rule30_nonperiodicity_v2/witnesses/T16384/MANIFEST.json --verify-files`
  - `python qa_alphageometry_ptolemy/qa_rule30/verify_certpack.py qa_alphageometry_ptolemy/qa_rule30/certpacks/rule30_nonperiodicity_v2 --T 16384`
- Source:
  - `qa_alphageometry_ptolemy/qa_rule30/qa_rule30_cert_validator.py`
  - `qa_alphageometry_ptolemy/qa_rule30/verify_certpack.py`
  - Legacy generator: `qa_alphageometry_ptolemy/rule30_witness_generator.py` (pre-2026-01-10; not used for reproduction)
- Status: verified
- Notes:
  - Re-vetted post-cutover: the authoritative, reproducible artifact is the post-2026-01-10 certpack `rule30_nonperiodicity_v2` plus independent verification.

### C-20260215-cert-invariants: Certificate invariants (fixed_q_mode=null + generator closure) validated

- Claim: Proof certificates satisfy core invariants across adapters: `fixed_q_mode` is `null` when absent (not `{}`), and every generator used in a success witness is included in `generator_set` (generator closure). Existing reflection artifacts validate under the same invariant suite.
- Evidence:
  - `trace/runs/20260215_031015_TOOL.revett.certificate_invariants.v1/stdout.log`
  - `qa_alphageometry_ptolemy/artifacts/reflection_GeometryAngleObserver.success.cert.json`
  - `qa_alphageometry_ptolemy/artifacts/reflection_NullObserver.obstruction.cert.json`
- Reproduce:
  - `cd qa_alphageometry_ptolemy && python3 test_certificate_invariants.py`
- Source:
  - `qa_alphageometry_ptolemy/test_certificate_invariants.py`
  - `qa_alphageometry_ptolemy/qa_certificate.py`
- Status: verified
- Notes:
  - Legacy summary: `qa_alphageometry_ptolemy/ALL_INVARIANTS_VALIDATED.md` (pre-2026-01-10) is re-vetted by the trace run above.

### C-20260215-qarm-v02-glfsi: QARM v0.2 constitutional mirror + GLFSI theorem verified

- Claim: Under CAP=20, KSet={2,3}, per-generator failure signatures are invariant under generator set changes (GLFSI): σ has (OOB=21, FQ=100) and λ has (OOB=105, FQ=35) in both {σ,μ,λ} and {σ,λ}; μ contributes (OOB=40, FQ=74) and vanishes when removed. Reachable state counts match: 504 (full) and 383 (no-μ).
- Evidence:
  - `trace/runs/20260215_031022_TOOL.revett.qarm_constitutional_cargo_test.v1/stdout.log`
  - `trace/runs/20260215_031028_TOOL.revett.qarm_constitutional_verify_glfsi.v1/stdout.log`
  - `trace/runs/20260215_031229_TOOL.revett.qarm_parse_tlc_dumps.v1/stdout.log`
  - TLC artifacts (legacy, parsed in revet run): `qa_alphageometry_ptolemy/states_full.txt.dump`, `qa_alphageometry_ptolemy/states_nomu.txt.dump`
- Reproduce:
  - `cd qa_alphageometry_ptolemy/qarm_constitutional && cargo test`
  - `cd qa_alphageometry_ptolemy/qarm_constitutional && cargo run --bin verify_glfsi`
  - `cd qa_alphageometry_ptolemy && python3 parse_tlc_dump.py`
- Source:
  - `qa_alphageometry_ptolemy/qarm_constitutional/src/qarm_glfsi.rs`
  - `qa_alphageometry_ptolemy/QARM_v02_Failures.tla`
  - `qa_alphageometry_ptolemy/QARM_v02_NoMu.tla`
  - `qa_alphageometry_ptolemy/parse_tlc_dump.py`
- Status: verified
- Notes:
  - Legacy docs: `qa_alphageometry_ptolemy/TLC_FAILURE_ANALYSIS_REPORT.md`, `qa_alphageometry_ptolemy/RUST_MIRROR_COMPLETE.md` (both pre-2026-01-10) are re-vetted by the trace runs above.

### C-20260215-graphrag-lexicon-graph: Lexicon → QA tuple encodings → GraphML knowledge graph (local, deterministic)

- Claim: The canonical lexicon can be deterministically extracted and encoded into QA tuples, producing a local GraphML knowledge graph (current run: 21 entities → 21 nodes, 38 edges).
- Evidence:
  - `trace/runs/20260215_032020_TOOL.revett.graphrag_minimal.v1/stdout.log`
  - `trace/runs/20260215_032020_TOOL.revett.graphrag_minimal.v1/artifacts/qa_entities.json`
  - `trace/runs/20260215_032020_TOOL.revett.graphrag_minimal.v1/artifacts/qa_entity_encodings.json`
  - `trace/runs/20260215_032020_TOOL.revett.graphrag_minimal.v1/artifacts/qa_knowledge_graph.graphml`
- Reproduce:
  - `OUT=/tmp/qa_graphrag_minimal && mkdir -p \"$OUT\"`
  - `python3 qa_entity_extractor.py --in private/QAnotes/research_log_lexicon.md --out \"$OUT/qa_entities.json\"`
  - `python3 qa_entity_encoder.py --in \"$OUT/qa_entities.json\" --overrides qa_entity_overrides.yaml --out \"$OUT/qa_entity_encodings.json\"`
  - `python3 qa_knowledge_graph.py --enc \"$OUT/qa_entity_encodings.json\" --out \"$OUT/qa_knowledge_graph.graphml\"`
- Source:
  - `qa_entity_extractor.py`
  - `qa_entity_encoder.py`
  - `qa_knowledge_graph.py`
- Status: verified
- Notes:
  - `qa_build_pipeline.py` (legacy) currently calls `qa_knowledge_graph.py` with a mismatched CLI; prefer the three-step minimal pipeline above until the toolchain is re-wired.

### C-20260329-eeg-threshold-policy-slice: EEG HI 2.0 threshold-policy live slice preserves the regime split while reducing overrides

- Claim: On a capped live CHB-MIT slice spanning the borderline override case `chb01`, the stable-full case `chb03`, and the robust override case `chb08`, the `stability_threshold` policy with `min_trigger_rate=0.5` preserved positive override-gate gain over full HI 2.0 while reducing overrides relative to the anchor policy. The intended regime split held prospectively: anchor rerouted `chb01` and `chb08`; threshold-0.5 rerouted only `chb08`; `chb03` stayed on full under both.
- Evidence:
  - `results/eeg_hi2_0_live_slice_anchor.json`
  - `results/eeg_hi2_0_live_slice_threshold_0p5.json`
  - `results/eeg_hi2_0_live_policy_slice_compare.json`
  - `qa_alphageometry_ptolemy/qa_empirical_observation_cert/results/eoc_pass_eeg_live_policy_slice_threshold_consistent.json`
  - `experiments/registry.json`
- Reproduce (from repo root):
  - Validator family:
    - `python3 qa_alphageometry_ptolemy/qa_empirical_observation_cert/qa_empirical_observation_cert_validate.py --self-test`
    - `python3 qa_alphageometry_ptolemy/qa_empirical_observation_cert/qa_empirical_observation_cert_validate.py --file qa_alphageometry_ptolemy/qa_empirical_observation_cert/results/eoc_pass_eeg_live_policy_slice_threshold_consistent.json`
  - Rebuild the comparison artifact from saved run outputs:
    - `python3 eeg_hi2_0_live_policy_slice_compare.py --anchor results/eeg_hi2_0_live_slice_anchor.json --threshold results/eeg_hi2_0_live_slice_threshold_0p5.json --anchor-only-reroute chb01 --both-reroute chb08 --both-full chb03 --output /tmp/eeg_hi2_0_live_policy_slice_compare.repro.json`
  - Rebuild the detector decision artifact from saved family/stability artifacts:
    - `python3 eeg_hi2_0_regime_detector.py --artifact results/eeg_hi2_0_family_gated_classifier.json --stability-audit results/eeg_hi2_0_override_gate_patient_stability_audit.json --policy-mode stability_threshold --min-trigger-rate 0.5 --output /tmp/eeg_hi2_0_regime_detector_threshold_0p5.repro.json`
- Source:
  - `eeg_hi2_0_chbmit_scale.py`
  - `eeg_hi2_0_regime_detector.py`
  - `eeg_hi2_0_live_policy_slice_compare.py`
  - `eeg_hi2_0_stability_threshold_policy_sweep.py`
- Status: verified
- Notes:
  - Cert-backed result: `qa.cert.empirical.eeg_hi2_live_policy_slice_threshold_consistent.v1`.
  - The machine-level safety profile for this repo now forbids live raw-EDF policy validation above 3 patients on this host without explicit override; use saved artifacts by default on this laptop.
