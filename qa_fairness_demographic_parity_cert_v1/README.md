# QA Fairness — Demographic Parity Certificate v1

Machine-checkable certificate for a common fairness claim:

> The **demographic parity gap** (max selection-rate difference across groups) is at most `dp_gap_max`.

This certificate is **exact-arithmetic**: counts are integers and rates are serialized as rational strings (e.g. `"3/10"`).

## Files

- `schema.json` — `QA_FAIRNESS_DEMOGRAPHIC_PARITY_CERT.v1`
- `validator.py` — 5-gate validator + CLI + self-test fixtures
- `fixtures/valid_min.json` — passing example
- `fixtures/invalid_gap.json` — failing example (gap too large)
- `mapping_protocol_ref.json` — intake protocol reference

## Run

```bash
python qa_fairness_demographic_parity_cert_v1/validator.py --self-test
python qa_fairness_demographic_parity_cert_v1/validator.py qa_fairness_demographic_parity_cert_v1/fixtures/valid_min.json
```

