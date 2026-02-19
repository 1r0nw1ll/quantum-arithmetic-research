# Family [71]: QA Curvature Stress-Test Bundle

## Purpose

Family [71] is a **meta-family** (bundle validator) that validates cross-family
curvature stress-test bundles. Each bundle contains one or more entries, each
representing a QA cert family's generator curvature anchor, and one or more
TENSOR compositions that must satisfy the **monoidal bottleneck law**.

The family certifies that κ̂ universality holds across QA families:
when two certified processes are composed via TENSOR product, the bottleneck
curvature of the composed system equals the minimum curvature of its components.

## Research Question

Does the generator interaction curvature κ̂ = 1 − |1 − lr·λ_orbit|, defined
per-family in the Dynamics Spine, satisfy a monoidal bottleneck law under
TENSOR composition across QA cert families?

## Location

`qa_curvature_stress_test_v1/`

## Schema

`QA_CURVATURE_STRESS_TEST_BUNDLE.v1`

## Curvature Definition

For each entry in the bundle:

```
κ̂_i = 1 − |1 − lr_i · lambda_orbit_i|
```

Computed per epoch; minimum over all epochs = `min_kappa_hat`.

## Monoidal Bottleneck Law (TENSOR)

For a TENSOR composition of entries {e₁, e₂, ..., eₙ}:

```
κ̂_composed = min(κ̂_e1, κ̂_e2, ..., κ̂_en)
```

The composed system is only as stable as its least stable component.

## Gates

1. **Gate 1 — Schema**: `schema_id`, required top-level fields, `result` structure
2. **Gate 2 — Entry anchor coherence**: `entry_id` pattern, `model_config` ranges,
   `generator_curvature` required fields, non-empty `kappa_hat_per_epoch`
3. **Gate 3 — κ recompute integrity**: recompute `kappa_hat_per_epoch` from
   `(lr, lambda_orbit)`, verify `kappa_hash`, `min_kappa_hat`, `min_kappa_epoch`;
   fail `NEGATIVE_GENERATOR_CURVATURE` or `CURVATURE_RECOMPUTE_MISMATCH`
4. **Gate 4 — κ sign prediction alignment**: expected sign from recomputed
   `min_kappa_hat` vs `predicted.kappa_sign`; fail `KAPPA_SIGN_MISMATCH`
5. **Gate 5 — Monoidal bottleneck law**: TENSOR compositions must satisfy
   `kappa_hat_composed == min(component kappas)`; fail `BOTTLE_NECK_VIOLATION`
   or `COMPOSITION_REF_MISSING`

## Artifacts

| File | Purpose |
|------|---------|
| `schema.json` | Schema for `QA_CURVATURE_STRESS_TEST_BUNDLE.v1` |
| `validator.py` | Five-gate validator |
| `mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `fixtures/valid_cross_family_bundle.json` | PASS: 2 entries, 1 composition |
| `fixtures/invalid_missing_family.json` | FAIL: COMPOSITION_REF_MISSING |
| `fixtures/invalid_kappa_sign_mismatch.json` | FAIL: KAPPA_SIGN_MISMATCH |
| `fixtures/invalid_bottleneck_violation.json` | FAIL: BOTTLE_NECK_VIOLATION |

## Failure Taxonomy

| fail_type | Gate | Trigger |
|-----------|------|---------|
| `SCHEMA_REQUIRED_FIELD_MISSING` | 1–2 | Required field absent |
| `SCHEMA_TYPE_MISMATCH` | 1–2 | Wrong type or enum value |
| `SCHEMA_ID_MISMATCH` | 1 | `schema_id` is not `QA_CURVATURE_STRESS_TEST_BUNDLE.v1` |
| `CONFIG_INVALID` | 2 | `lr <= 0` or `lambda_orbit < 0` |
| `CURVATURE_RECOMPUTE_MISMATCH` | 3 | Recomputed kappa or hash does not match cert |
| `NEGATIVE_GENERATOR_CURVATURE` | 3 | Any recomputed epoch kappa < 0 |
| `KAPPA_SIGN_MISMATCH` | 4 | `predicted.kappa_sign` disagrees with recomputed sign |
| `COMPOSITION_REF_MISSING` | 5 | Composition references unknown `entry_id` |
| `BOTTLE_NECK_VIOLATION` | 5 | `kappa_hat_composed != min(component kappas)` |

## CLI

```bash
# Self-test (4 fixtures: 1 valid, 3 negative)
python qa_curvature_stress_test_v1/validator.py --self-test

# Validate a bundle cert
python qa_curvature_stress_test_v1/validator.py \
  qa_curvature_stress_test_v1/fixtures/valid_cross_family_bundle.json
```

## Relation to Family [64]

Family [64] first defined the closed-form curvature κ̂ = 1 − |1 − lr·λ| and
Gate 3 recomputation. Family [71] elevates this from a per-family property to a
**cross-family universality law**, certifying that curvature bounds compose
multiplicatively under TENSOR product via the bottleneck construction.

The monoidal structure is: (QA cert families, ⊗, Identity) forms a monoidal
category where the curvature functor κ̂ is lax monoidal with the min bottleneck
as the structural morphism.
