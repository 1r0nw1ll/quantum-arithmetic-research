# QA Failure Compose Operator Cert v1

Machine-tract family for explicit `compose(Fi, Fj, form)` certification.

## What this cert checks

- Canonical operator anchor `compose(Fi, Fj, form)` in `compose_operator_anchor.py`
- Finite carrier and declared composition forms
- Explicit compose table rows `(form, a, b) -> comp`
- Closure over the carrier for every form
- Associativity for each form independently
- Claim binding for `closure_holds` and `associativity_holds`

## Fixtures

- `fixtures/pass_feedback_escalation.json` (PASS)
- `fixtures/fail_closure_incomplete_table.json` (FAIL, missing compose row)
- `fixtures/fail_associativity_feedback_violation.json` (FAIL, associativity witness)

## Run

```bash
python qa_failure_compose_operator_cert_v1/validator.py --self-test
python qa_failure_compose_operator_cert_v1/validator.py --schema qa_failure_compose_operator_cert_v1/schema.json --cert qa_failure_compose_operator_cert_v1/fixtures/pass_feedback_escalation.json
```
