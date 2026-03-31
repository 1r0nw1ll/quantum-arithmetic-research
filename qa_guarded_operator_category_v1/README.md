# QA Guarded Operator Category v1

Machine-checkable family for matrix embedding of QA generator core with guarded extensions.

## Files

- `schema.json` — `QA_GUARDED_OPERATOR_CATEGORY_CERT.v1`
- `validator.py` — 5-gate validator with recompute checks
- `generator_semantics_registry.json` — explicit `sigma_unit` and `sigma_shear` semantics
- `fixtures/valid_min.json` — valid certificate fixture
- `fixtures/invalid_nu_unguarded.json` — negative fixture (missing `nu` guard)
- `fixtures/invalid_lambda_det_claim.json` — negative fixture (wrong λ obstruction determinant)
- `fixtures/invalid_nu_guard_malformed.json` — negative fixture (malformed `nu` guard spec)
- `mapping_protocol_ref.json` — intake protocol reference

## Run

```bash
python qa_guarded_operator_category_v1/validator.py --self-test
python qa_guarded_operator_category_v1/validator.py qa_guarded_operator_category_v1/fixtures/valid_min.json
```
