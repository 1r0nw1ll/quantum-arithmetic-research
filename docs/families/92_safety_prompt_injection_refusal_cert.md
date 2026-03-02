# Family [92]: QA Safety Prompt Injection Refusal Cert v1

## What it is

Family [92] certifies that a model or pipeline's **prompt injection refusal
rate** meets a declared safety threshold.

The cert records pass/fail/skip counts over a fixed benchmark of prompt
injection test cases, verifies the refusal rate against a declared minimum,
and binds the result to a specific judge contract via `judge_contract_hash`.
When the refusal rate falls below the threshold the validator requires a
**failure witness**: an explicit list of `test_case_id` values for cases where
the injection was not refused, enabling targeted regression investigation.

## Cert root

`qa_safety_prompt_injection_refusal_cert_v1/`

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for prompt injection refusal certs |
| `validator.py` | 4-gate validator with rate recompute + judge hash check + failure witness |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/valid_min.json` | PASS fixture (refusal rate meets threshold, no failure witness required) |
| `fixtures/invalid_rate.json` | FAIL fixture (RATE_MISMATCH or RATE_BELOW_THRESHOLD with witness) |

## Gates

| Gate | Check |
|---|---|
| 0 | Mapping protocol intake constitution |
| 1 | Schema validation |
| 2 | Rate recompute: `refusal_rate = pass_count / (pass_count + fail_count)` (skip not counted); compare declared `refusal_rate` to recomputed value; verify `judge_contract_hash` is a 64-character hex digest |
| 3 | Failure witness: if `refusal_rate < min_refusal_rate`, a `failure_witness` block listing `test_case_ids` of non-refused injections must be present; count must equal `fail_count` |

## Deterministic output

On PASS, the validator emits:

- `pass_count`: number of test cases where injection was refused
- `fail_count`: number of test cases where injection was not refused
- `skip_count`: number of test cases skipped (excluded from rate)
- `refusal_rate`: `pass_count / (pass_count + fail_count)`
- `min_refusal_rate`: declared acceptable floor
- `judge_contract_hash`: SHA-256 hex digest binding the judge prompt/contract
- `verdict`: `PASS` (rate >= threshold) or `FAIL` (rate < threshold with witness)

## How to run

```bash
python qa_safety_prompt_injection_refusal_cert_v1/validator.py --self-test
python qa_safety_prompt_injection_refusal_cert_v1/validator.py \
  --schema qa_safety_prompt_injection_refusal_cert_v1/schema.json \
  --cert qa_safety_prompt_injection_refusal_cert_v1/fixtures/valid_min.json
```

## Failure types

- `SCHEMA_INVALID`
- `RATE_MISMATCH` (declared `refusal_rate` does not match recomputed value)
- `JUDGE_HASH_INVALID` (not a 64-character lowercase hex string)
- `WITNESS_MISSING` (rate below threshold but no failure witness provided)
- `WITNESS_COUNT_MISMATCH` (witness `test_case_ids` length does not equal `fail_count`)
- `RATE_BELOW_THRESHOLD` (rate < `min_refusal_rate` and witness is valid — this is the cert FAIL verdict)
