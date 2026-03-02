# Family [90]: QA Fairness Demographic Parity Cert v1

## What it is

Family [90] certifies that a classifier's **demographic parity** (selection-rate
equality across protected groups) meets a declared gap threshold.

Demographic parity requires that the positive prediction rate is the same for
all protected groups. The cert records the observed selection rate per group,
recomputes the max absolute gap between any two groups, and verifies that the
reported gap matches the recomputed value within a declared tolerance.

When the gap exceeds the declared threshold the validator requires a constructive
**failure witness**: a concrete pair of groups `(group_a, group_b)` together with
their observed selection rates that demonstrates the violation.

## Cert root

`qa_fairness_demographic_parity_cert_v1/`

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for demographic parity certs |
| `validator.py` | 4-gate validator with deterministic gap recompute + failure witness check |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/valid_min.json` | PASS fixture (gap below threshold, no witness required) |
| `fixtures/invalid_gap.json` | FAIL fixture (GAP_MISMATCH — reported gap does not match recompute) |

## Gates

| Gate | Check |
|---|---|
| 0 | Mapping protocol intake constitution |
| 1 | Schema validation |
| 2 | Gap recompute: deterministic recomputation of per-group selection rates and max absolute gap; compare to declared `observed_gap` |
| 3 | Failure witness: if `observed_gap > threshold`, a `failure_witness` block with `group_a`, `group_b`, and their rates must be present and consistent with the recomputed gap |

## Deterministic output

On PASS, the validator emits:

- `groups`: list of protected group identifiers
- `selection_rates`: map from group id to recomputed positive prediction rate
- `observed_gap`: max absolute difference across all group pairs
- `threshold`: declared acceptable gap ceiling
- `verdict`: `PASS` (gap <= threshold) or `FAIL` (gap > threshold with witness)

## How to run

```bash
python qa_fairness_demographic_parity_cert_v1/validator.py --self-test
python qa_fairness_demographic_parity_cert_v1/validator.py \
  --schema qa_fairness_demographic_parity_cert_v1/schema.json \
  --cert qa_fairness_demographic_parity_cert_v1/fixtures/valid_min.json
```

## Failure types

- `SCHEMA_INVALID`
- `GAP_MISMATCH` (declared `observed_gap` does not match recomputed value)
- `WITNESS_MISSING` (gap exceeds threshold but no failure witness provided)
- `WITNESS_INCONSISTENT` (witness group rates do not reproduce the declared gap)
- `THRESHOLD_EXCEEDED` (gap > threshold without a valid witness block)
