# Family [40]: QA Reachability Descent Run Cert v1

## What it is

Family [40] certifies a deterministic reachability-descent run over a finite
caps domain `Caps(N,N)` for QA state tuples `(b,e)`.

The certificate records step-wise attempts, chosen moves, and energy deltas
under policy:

`GREEDY_MIN_ENERGY_TIEBREAK_MOVE_NAME`

with generator moves from `{sigma, mu, lambda_k, nu}` and objective
`L2_TO_TARGET`.

## Cert root

`qa_reachability_descent_run_cert_v1/`

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for reachability-descent run cert |
| `validator.py` | 5-gate deterministic validator |
| `mapping_protocol.json` | Gate-0 inline mapping protocol |
| `fixtures/PASS_N6_v1.json` | PASS fixture |
| `fixtures/FAIL_MOVE_NOT_IN_GENERATOR_SET.json` | FAIL fixture (invalid chosen move) |
| `fixtures/FAIL_SCHEMA_INVALID__LAMBDA_K_MISSING_K.json` | FAIL fixture (schema invalid) |

## Gates

| Gate | Check |
|---|---|
| 1 | Schema validity |
| 2 | Recompute move transitions from declared generator set |
| 3 | Recompute energy trajectory against objective target |
| 4 | Verify attempted-move log consistency |
| 5 | Verify policy consistency + bounded recoverability witness |

## How to run

```bash
python qa_reachability_descent_run_cert_v1/validator.py --self-test
python qa_reachability_descent_run_cert_v1/validator.py \
  --schema qa_reachability_descent_run_cert_v1/schema.json \
  --cert qa_reachability_descent_run_cert_v1/fixtures/PASS_N6_v1.json
```

## Failure types

- `SCHEMA_INVALID`
- `MOVE_NOT_IN_GENERATOR_SET`
- `MOVE_RECOMPUTE_MISMATCH`
- `ENERGY_RECOMPUTE_MISMATCH`
- `ATTEMPT_LOG_MISMATCH`
- `POLICY_VIOLATION`
- `RECOVERABILITY_VIOLATION`
