# QA Fairness — Equalized Odds Certificate v1

Machine-checkable certificate for the standard fairness claim:

> **Equalized odds gaps** are bounded: the maximum TPR gap and maximum FPR gap across protected groups are at most
> `tpr_gap_max` and `fpr_gap_max`.

This certificate uses **exact arithmetic**:
- counts are integers
- rates are serialized as rational strings (e.g. `"7/10"`)

## Files

- `schema.json` — `QA_FAIRNESS_EQUALIZED_ODDS_CERT.v1`
- `validator.py` — 5-gate validator + CLI + self-test fixtures
- `fixtures/valid_min.json` — passing example
- `fixtures/invalid_gap.json` — failing example (TPR/FPR gap too large)
- `mapping_protocol_ref.json` — intake protocol reference

## Run

```bash
python qa_fairness_equalized_odds_cert_v1/validator.py --self-test
python qa_fairness_equalized_odds_cert_v1/validator.py qa_fairness_equalized_odds_cert_v1/fixtures/valid_min.json
```

