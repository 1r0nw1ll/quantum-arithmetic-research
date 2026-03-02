# Family [88]: QA Failure Algebra Structure Classification Cert v1

## What it is

Family [88] certifies the algebraic structure type induced by a finite
form-indexed operator:

`compose(Fi, Fj, form) -> Fk`

with `form ∈ {serial, parallel, feedback}`.

Given a carrier and explicit compose tables, it deterministically classifies each
form as semigroup/monoid and verifies identity, absorber, commutativity, and
optional monotonicity under a provided preorder.

## Cert root

`qa_failure_algebra_structure_classification_cert_v1/`

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for structure-classification certs |
| `validator.py` | 5-gate validator with deterministic recompute + classification payload |
| `operator_anchor_ref.py` | Family [87] table reference utilities |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/pass_classify_from_family87_tables.json` | PASS fixture |
| `fixtures/fail_identity_claim_wrong.json` | FAIL fixture (Gate 3) |
| `fixtures/fail_absorber_claim_wrong.json` | FAIL fixture (Gate 3) |
| `fixtures/fail_commutative_claim_wrong.json` | FAIL fixture (Gate 4) |
| `fixtures/fail_monotonicity_violation.json` | FAIL fixture (Gate 5) |

## Gates

| Gate | Check |
|---|---|
| 1 | Schema + reference pin + deterministic ordering checks |
| 2 | Closure + associativity per form |
| 3 | Identity/absorber discovery and assertive claim checks |
| 4 | Commutativity recompute and assertive claim checks |
| 5 | Optional preorder validation + monotonicity witness checks |

## Deterministic output

On PASS, the validator emits per-form classification payload:

- `type`: `semigroup` or `monoid`
- `commutative`: bool
- `identity`: string or `null`
- `absorber`: string or `null`
- `monotone`: bool or `null`

## How to run

```bash
python qa_failure_algebra_structure_classification_cert_v1/validator.py --self-test
python qa_failure_algebra_structure_classification_cert_v1/validator.py \
  --schema qa_failure_algebra_structure_classification_cert_v1/schema.json \
  --cert qa_failure_algebra_structure_classification_cert_v1/fixtures/pass_classify_from_family87_tables.json
```

## Failure types

- `SCHEMA_INVALID`
- `FAMILY87_REF_MISSING`
- `FAMILY87_REF_HASH_MISMATCH`
- `DUPLICATE_COMPOSE_ENTRY`
- `COMPOSE_TABLE_NONCANONICAL_ORDER`
- `COMPOSE_TABLE_INCOMPLETE`
- `CLOSURE_VIOLATION`
- `COMPOSE_ASSOCIATIVITY_VIOLATION`
- `IDENTITY_CLAIM_MISMATCH`
- `ABSORBER_CLAIM_MISMATCH`
- `COMMUTATIVITY_CLAIM_MISMATCH`
- `PREORDER_INVALID`
- `MONOTONICITY_VIOLATION`
- `MONOTONICITY_CLAIM_MISMATCH`
- `RECOMPUTE_MISMATCH`
