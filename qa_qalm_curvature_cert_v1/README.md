# QA QALM Curvature Cert v1

Machine-tract family for pinning the QALM 2.0 harmonic curvature formula (`H_QA`)
and the curvature-scaled update rule used in `QAOptimizer`.

## Source anchors

- Curvature formula (`H_QA`): `qalm_2.0/qa_markovian_integration.py` lines 47–54 (fallback `harmonic_descent`)
- Update rule: `qalm_2.0/qa_markovian_integration.py` line 260
- Ground-truth fixture: `qalm_2.0/QALM2_TEST_RESULTS.md` (default tuple yields `H_QA ≈ 0.0497`)

## Fixtures

- `fixtures/pass_default_tuple.json` (PASS)
- `fixtures/fail_h_qa_mismatch.json` (FAIL Gate 2)
- `fixtures/fail_update_sign.json` (FAIL Gate 3)

## Run

```bash
python qa_qalm_curvature_cert_v1/validator.py --self-test
python qa_qalm_curvature_cert_v1/validator.py \
  --schema qa_qalm_curvature_cert_v1/schema.json \
  --cert qa_qalm_curvature_cert_v1/fixtures/pass_default_tuple.json
```

