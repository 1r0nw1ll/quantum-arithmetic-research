# QA Whittaker Scalar Angular-Kernel Sampling Cert v1

Candidate family ID: `[274]`. This artifact is not registered yet.

This cert builds Layer 3 v1 of the Whittaker -> QA development ladder. It
uses the registered `[273]` exact rational `S2` direction substrate and tests
only deterministic finite scalar sampling over that substrate.

## Scope

The v1 claim is exact finite scalar averaging only:

- `WKB_1`: dependency provenance requires registered `[273]`.
- `WKB_2`: profiles are restricted to `{const,z,z2}`.
- `WKB_3`: `weight_rule` must be `uniform_points`.
- `WKB_4`: the exact `Fraction` `discrete_uniform_average` is recomputed.
- `WKB_5`: observer diagnostics are display-only; float pass/fail and
  quadrature/Whittaker/physics overclaims are rejected.

Large rational averages are witnessed by:

```text
canonical_fraction_sha256 = sha256((num + "/" + den).encode("ascii")).hexdigest()
numerator_digit_count
denominator_digit_count
```

Observer floats are not used for pass/fail.

## Non-Claims

This cert does not prove Whittaker 1903, Whittaker's wave kernel, spherical
quadrature, density, convergence, Maxwell equations, electromagnetism,
scalar-potential physics, geodesy, ellipsoid physics, or any physical field
reconstruction.

## Fixtures

PASS:

- `fixtures/pass_wkb_m3_const_uniform.json`
- `fixtures/pass_wkb_m5_z_uniform.json`
- `fixtures/pass_wkb_m9_z2_uniform_hash.json`

FAIL:

- `fixtures/fail_wkb_wrong_provenance.json`
- `fixtures/fail_wkb_wrong_profile_value.json`
- `fixtures/fail_wkb_wrong_hash_witness.json`
- `fixtures/fail_wkb_wrong_digit_count.json`
- `fixtures/fail_wkb_float_used_as_pass_fail.json`
- `fixtures/fail_wkb_overclaimed_spherical_integral.json`
- `fixtures/fail_wkb_overclaimed_whittaker_error.json`

## Run

```bash
python3 qa_alphageometry_ptolemy/qa_whittaker_scalar_angular_kernel_sampling_cert_v1/qa_whittaker_scalar_angular_kernel_sampling_cert_validate.py --self-test
```

Expected result:

```json
{"ok": true}
```

The full output includes all 10 fixture classifications: 3 PASS and 7 FAIL.
