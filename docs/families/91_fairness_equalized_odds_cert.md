# Family [91]: QA Fairness Equalized Odds Cert v1

## What it is

Family [91] certifies that a classifier's **equalized odds** constraints are met
across protected groups.

Equalized odds requires that both the true positive rate (TPR) and the false
positive rate (FPR) are equal across all protected groups. The cert records the
observed confusion-matrix counts per group, recomputes TPR and FPR for each
group, measures the max absolute TPR gap and max absolute FPR gap across all
group pairs, and verifies that the reported gaps match the recomputed values.

When either gap exceeds its declared threshold the validator requires a
constructive **failure witness**: a concrete pair of groups `(group_a, group_b)`
with their observed TPR or FPR values that demonstrates the violation.

## Cert root

`qa_fairness_equalized_odds_cert_v1/`

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for equalized odds certs |
| `validator.py` | 4-gate validator with deterministic TPR/FPR recompute + failure witness check |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/valid_min.json` | PASS fixture (both TPR gap and FPR gap below thresholds) |
| `fixtures/invalid_gap.json` | FAIL fixture (GAP_MISMATCH — reported TPR or FPR gap does not match recompute) |

## Gates

| Gate | Check |
|---|---|
| 0 | Mapping protocol intake constitution |
| 1 | Schema validation |
| 2 | Gap recompute: deterministic recomputation of per-group TPR and FPR from confusion-matrix counts; compare declared `tpr_gap` and `fpr_gap` to recomputed values |
| 3 | Failure witness: if `tpr_gap > tpr_threshold` or `fpr_gap > fpr_threshold`, a `failure_witness` block with `metric` (`tpr` or `fpr`), `group_a`, `group_b`, and their observed rates must be present and consistent |

## Deterministic output

On PASS, the validator emits:

- `groups`: list of protected group identifiers
- `tpr_per_group`: map from group id to recomputed true positive rate
- `fpr_per_group`: map from group id to recomputed false positive rate
- `tpr_gap`: max absolute TPR difference across all group pairs
- `fpr_gap`: max absolute FPR difference across all group pairs
- `tpr_threshold`: declared acceptable TPR gap ceiling
- `fpr_threshold`: declared acceptable FPR gap ceiling
- `verdict`: `PASS` or `FAIL`

## How to run

```bash
python qa_fairness_equalized_odds_cert_v1/validator.py --self-test
python qa_fairness_equalized_odds_cert_v1/validator.py \
  --schema qa_fairness_equalized_odds_cert_v1/schema.json \
  --cert qa_fairness_equalized_odds_cert_v1/fixtures/valid_min.json
```

## Failure types

- `SCHEMA_INVALID`
- `GAP_MISMATCH` (declared `tpr_gap` or `fpr_gap` does not match recomputed value)
- `WITNESS_MISSING` (a gap exceeds its threshold but no failure witness provided)
- `WITNESS_INCONSISTENT` (witness group rates do not reproduce the declared gap)
- `THRESHOLD_EXCEEDED` (gap > threshold without a valid witness block)
- `CONFUSION_COUNTS_INVALID` (TP+FN=0 or TN+FP=0 for a group, making rate undefined)
