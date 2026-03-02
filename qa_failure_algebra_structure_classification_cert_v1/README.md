# QA Failure Algebra Structure Classification Cert v1

Machine-tract family for classifying form-indexed failure composition operators.

## Operator

`compose(Fi, Fj, form) -> Fk`, where `form ∈ {serial, parallel, feedback}`.

## What this cert checks

- Family [87] reference pin (`path` + `cert_sha256`)
- Table closure and associativity per form
- Identity and absorber discovery per form
- Commutativity and monotonicity checks
- Deterministic classification payload

## Fixtures

- `fixtures/pass_classify_from_family87_tables.json` (PASS)
- `fixtures/fail_identity_claim_wrong.json` (FAIL Gate 3)
- `fixtures/fail_absorber_claim_wrong.json` (FAIL Gate 3)
- `fixtures/fail_commutative_claim_wrong.json` (FAIL Gate 4)
- `fixtures/fail_monotonicity_violation.json` (FAIL Gate 5)

## Run

```bash
python qa_failure_algebra_structure_classification_cert_v1/validator.py --self-test
python qa_failure_algebra_structure_classification_cert_v1/validator.py \
  --schema qa_failure_algebra_structure_classification_cert_v1/schema.json \
  --cert qa_failure_algebra_structure_classification_cert_v1/fixtures/pass_classify_from_family87_tables.json
```
