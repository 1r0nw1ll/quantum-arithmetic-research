# Family [72]: QA Guarded Operator Category

## Purpose

Family [72] certifies a matrix embedding for the QA core generators `{sigma, mu}` and
certifies explicit obstruction witnesses for guarded extensions `{lambda_k, nu}`.

This family separates:

- unimodular core action (integer matrix group layer), and
- guarded partial-map extension layer (failure-complete obstructions).

## Location

`qa_guarded_operator_category_v1/`

## Schema

`QA_GUARDED_OPERATOR_CATEGORY_CERT.v1`

## Generator Semantics Registry

The family ships an explicit semantics registry:

- `QA_GENERATORS_SIGMA_UNIT.v1`: `sigma(b,e)=(b,e+1)`
- `QA_GENERATORS_SIGMA_SHEAR.v1`: `sigma(b,e)=(b,e+b)`

This cert family requires `generator_semantics_ref=QA_GENERATORS_SIGMA_SHEAR.v1`.

## Gates

1. **Gate 1 — Schema anchor validity**
2. **Gate 2 — Semantics + rho anchor coherence**
   - checks `rho(sigma)=[[1,0],[1,1]]`, `rho(mu)=[[0,1],[1,0]]`
3. **Gate 3 — Composition recompute integrity**
   - recomputes `word -> matrix -> state`
4. **Gate 4 — Determinant + finite mod-n checks**
   - verifies determinant claims and generated subgroup sizes
5. **Gate 5 — Obstruction completeness**
   - `lambda_k_det_not_pm1` witness
   - `nu_not_integer_total_linear` witness + required `nu` guard

## Artifacts

| File | Purpose |
|------|---------|
| `schema.json` | Schema for `QA_GUARDED_OPERATOR_CATEGORY_CERT.v1` |
| `validator.py` | Five-gate validator with recompute checks |
| `generator_semantics_registry.json` | Explicit semantics refs (`sigma_unit`, `sigma_shear`) |
| `mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `fixtures/valid_min.json` | PASS fixture |
| `fixtures/invalid_nu_unguarded.json` | FAIL fixture (`NU_GUARD_MISSING`) |
| `fixtures/invalid_lambda_det_claim.json` | FAIL fixture (`LAMBDA_K_OBSTRUCTION_MISMATCH`) |
| `fixtures/invalid_nu_guard_malformed.json` | FAIL fixture (`NU_GUARD_INVALID`) |

## CLI

```bash
python qa_guarded_operator_category_v1/validator.py --self-test
python qa_guarded_operator_category_v1/validator.py qa_guarded_operator_category_v1/fixtures/valid_min.json
```
