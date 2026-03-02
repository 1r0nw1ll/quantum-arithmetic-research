# Family [89]: QA QALM Curvature Cert v1

## What it is

Family [89] pins the QALM 2.0 fallback harmonic curvature computation
(`H_QA`) and the curvature-scaled optimizer update rule used by `QAOptimizer`.

It is intended as a drift detector: if the curvature formula or the update
scaling changes, the cert fails with a concrete mismatch witness.

## Cert root

`qa_qalm_curvature_cert_v1/`

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for QALM curvature certs |
| `validator.py` | 3-gate validator with deterministic recompute |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/pass_default_tuple.json` | PASS fixture (default tuple from QALM2 test results) |
| `fixtures/fail_h_qa_mismatch.json` | FAIL fixture (Gate 2: curvature mismatch) |
| `fixtures/fail_update_sign.json` | FAIL fixture (Gate 3: update-rule mismatch) |

## Anchors

- `qalm_2.0/qa_markovian_integration.py` lines 47–54 (fallback `harmonic_descent` curvature/loss formula)
- `qalm_2.0/qa_markovian_integration.py` line 260 (update rule `p -= lr * gain * h * p.grad`)
- `qalm_2.0/QALM2_TEST_RESULTS.md` (default tuple yields `H_QA ≈ 0.0497`)

## Gates

| Gate | Check |
|---|---|
| 1 | Schema validity |
| 2 | Deterministic recompute of `H_QA_raw`, normalized `H_QA`, and `loss_hat` |
| 3 | Deterministic recompute of the curvature-scaled update rule |

## How to run

```bash
python qa_qalm_curvature_cert_v1/validator.py --self-test
python qa_qalm_curvature_cert_v1/validator.py \
  --schema qa_qalm_curvature_cert_v1/schema.json \
  --cert qa_qalm_curvature_cert_v1/fixtures/pass_default_tuple.json
```

## Failure types

- `SCHEMA_INVALID`
- `EPS_MISMATCH`
- `H_QA_MISMATCH`
- `LOSS_HAT_MISMATCH`
- `UPDATE_RULE_MISMATCH`

