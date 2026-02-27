# Family [82]: QA Raman KNN Results Cert v1

**Schema:** `QA_RAMAN_KNN_RESULTS_CERT.v1`
**Root:** `qa_raman_knn_results_cert_v1/`
**Mapping protocol:** `mapping_protocol_ref.json` → `qa_mapping_protocol/canonical_mapping_protocol.json`

## Purpose

Locks in the verified LOO kNN classification results for the QA Raman spectroscopy
paper. Provides a CI-enforceable certificate that:

- The headline **95.2% accuracy** (k=5, weighted LOO kNN) is reproducible from the
  frozen features artifact (`qa_raman_features.csv`, SHA-256 locked).
- The **k-sweep table** (k ∈ {1,3,5,7}, weighted vs. unweighted) is internally
  consistent and matches the manuscript Table B.2.
- The classifier is a **valid classical model** (not subject to quantum realizability issues).

This cert arose from the 2026-02-27 audit of the Raman spectroscopy paper, which found
the k-sweep table had stale values from an old pipeline run (k=1,3,7 rows were wrong).
The live re-run fixed the values; this cert locks them in permanently.

## Certified Results

| k | Unweighted | Weighted |
|---|-----------|---------|
| 1 | 0.9585    | 0.9585  |
| 3 | 0.8508    | 0.9544  |
| **5** | 0.8373 | **0.9523** |
| 7 | 0.8124    | 0.9534  |

**Headline:** k=5, weighted, accuracy=0.9523 (95.2%)

**χ²** = 1930.0 (C\_mod24 vs. class dependence)

**Artifact SHA-256:** `defa2f205265ba4990cc8fd504ce1a93228c02bd5071a5c6504fbb611668234a`

## Gate Architecture (5 gates)

| Gate | Name | What it checks |
|------|------|----------------|
| 1 | `gate_1_schema_validity` | JSON schema `QA_RAMAN_KNN_RESULTS_CERT.v1.schema.json` |
| 2 | `gate_2_canonical_hash` | Self-referential SHA-256 (canonical JSON with placeholder zeroed) |
| 3 | `gate_3_best_acc_consistency` | `k_sweep` best row accuracy matches `classifier.best_accuracy` |
| 4 | `gate_4_k1_parity` | k=1 unweighted == k=1 weighted (physical sanity: distance weighting irrelevant for k=1 LOO) |
| 5 | `gate_5_model_assessment` | `model_valid=True`, `model_not_physically_realizable=False` |

## Failure Modes

| Label | Gate | Meaning |
|-------|------|---------|
| `SCHEMA_INVALID` | 1 | Certificate does not match schema |
| `HASH_MISMATCH` | 2 | Canonical SHA-256 does not match declared value (cert was tampered) |
| `BEST_ACC_MISMATCH` | 3 | `classifier.best_accuracy` contradicts the k-sweep table |
| `K1_PARITY_MISMATCH` | 4 | k=1 unweighted ≠ weighted (impossible for LOO kNN) |
| `MODEL_NOT_PHYSICALLY_REALIZABLE` | 5 | Model flagged as physically unrealizable (should never happen for kNN) |
| `MODEL_INVALID` | 5 | `model_valid=False` |

## Fixtures

| File | Expected | Trigger |
|------|----------|---------|
| `valid_raman_knn_v1.json` | PASS | All gates pass with live sweep results |
| `invalid_best_acc_mismatch.json` | FAIL gate 3 | `best_accuracy=0.96` but k-sweep says 0.9523 |
| `invalid_model_not_realizable.json` | FAIL gate 5 | `model_not_physically_realizable=true` (wrong for kNN) |

## Contrast with I3322 Bell Test

The `model_not_physically_realizable` flag in Gate 5 explicitly distinguishes this
cert from the retracted I3322 Bell test results. The I3322 implementation
(`qa_i3322_bell_test.py`) optimized over unconstrained independent correlators and
produced scores of ~6.0 (24× the quantum bound of 0.25). That model IS
`MODEL_NOT_PHYSICALLY_REALIZABLE`. The LOO kNN classifier here is NOT.

See `BELL_TESTS_FINAL_SUMMARY.md` and `TSIRELSON_BOUND_RESEARCH_SUMMARY.md`
for the full I3322 retraction.

## Usage

```bash
# Run self-test
python3 qa_raman_knn_results_cert_v1/validator.py --self-test

# Validate a specific certificate
python3 qa_raman_knn_results_cert_v1/validator.py path/to/cert.json

# JSON output
python3 qa_raman_knn_results_cert_v1/validator.py --self-test --json
```

## Provenance

- Sweep script: `papers/ready-for-submission/qa-raman-spectroscopy/submission/qa_knn_sweep.py`
- Sweep results locked: `submission/artifacts/qa_knn_sweep.csv`
- Features artifact: `submission/artifacts/qa_raman_features.csv`
- Audit date: 2026-02-27
