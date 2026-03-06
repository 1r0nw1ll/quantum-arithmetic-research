# Family [87]: QA Failure Compose Operator Cert v1

## What it is

Family [87] certifies an explicit ternary operator:

`compose(Fi, Fj, form) -> Fk`

where `Fi, Fj, Fk` are failure tags from a finite carrier and `form` is a
composition mode (`serial`, `parallel`, `feedback`).

This family adds a machine-checkable composition layer on top of Family [76].

## Cert root

`qa_failure_compose_operator_cert_v1/`

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for carrier/forms/compose table/claims |
| `validator.py` | 5-gate validator with deterministic recompute |
| `compose_operator_anchor.py` | Canonical `compose(Fi, Fj, form)` reference implementation |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/pass_feedback_escalation.json` | PASS fixture |
| `fixtures/fail_closure_incomplete_table.json` | FAIL fixture (closure/table completeness) |
| `fixtures/fail_associativity_feedback_violation.json` | FAIL fixture (associativity witness) |

## Gates

| Gate | Check |
|---|---|
| 1 | Schema and type validity |
| 2 | Closure + compose-table completeness over `(form, a, b)` |
| 3 | Associativity for each `form` |
| 4 | Claimed booleans match recompute (`closure_holds`, `associativity_holds`) |
| 5 | Deterministic output payload (`compose_digest`, `cert_sha256`) |

## How to run

```bash
python qa_failure_compose_operator_cert_v1/validator.py --self-test
python qa_failure_compose_operator_cert_v1/validator.py \
  --schema qa_failure_compose_operator_cert_v1/schema.json \
  --cert qa_failure_compose_operator_cert_v1/fixtures/pass_feedback_escalation.json
```

## Failure types

- `SCHEMA_INVALID`
- `DUPLICATE_COMPOSE_ENTRY`
- `COMPOSE_TABLE_INCOMPLETE`
- `CLOSURE_VIOLATION`
- `COMPOSE_ASSOCIATIVITY_VIOLATION`
- `RECOMPUTE_MISMATCH`
