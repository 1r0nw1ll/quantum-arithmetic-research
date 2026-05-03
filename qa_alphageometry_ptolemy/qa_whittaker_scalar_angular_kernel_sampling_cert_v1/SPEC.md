# SPEC: QA Whittaker Scalar Angular-Kernel Sampling Cert v1

Candidate family ID: `[274]`, unregistered pending hostile review.

## Dependency

Layer 3 v1 depends on registered family `[273]`:

```text
qa_whittaker_rational_direction_s2_cert_v1
chart = inverse_stereographic_excluding_south_pole
```

The validator recomputes `[273]` packets locally through the `[273]`
construction logic.

## Construction

Inputs:

```text
m in {3,5,9}
profile_name in {const,z,z2}
weight_rule = uniform_points
```

For each generated `[273]` packet:

```text
omega_i = (x_i/den_i, y_i/den_i, z_i/den_i)
```

profile samples are exact `Fraction` values:

```text
const = 1
z     = z_i/den_i
z2    = (z_i/den_i)*(z_i/den_i)
```

The finite-set average is:

```text
discrete_uniform_average = (sum_i h_i) / |D_m^(2)|
```

This is a deterministic finite-set average, not a spherical quadrature rule.

## Exact Witnesses

Small exact averages may use direct numerator/denominator fields. Large exact
averages use hash witnesses:

```text
canonical_fraction_sha256 = sha256((num + "/" + den).encode("ascii")).hexdigest()
numerator_digit_count
denominator_digit_count
```

The validator recomputes the exact `Fraction`, builds the canonical ASCII
string `num/den`, and checks the full SHA-256 hash plus digit counts.
`canonical_fraction_sha256_16` is display/reporting only.

## Gates

`WKB_1`: dependency provenance:

```text
dependency.family_id = 273
dependency.slug = qa_whittaker_rational_direction_s2_cert_v1
dependency.chart = inverse_stereographic_excluding_south_pole
dependency.registered = true
```

`WKB_2`: profile set restricted to `{const,z,z2}`.

`WKB_3`: `weight_rule = uniform_points`.

`WKB_4`: exact `Fraction` `discrete_uniform_average` recomputation and direct
or hash witness validation.

`WKB_5`: observer diagnostics are display-only. The validator rejects:

```text
uses_observer_float_for_pass_fail = true
claims_spherical_quadrature = true
claims_whittaker_kernel_error = true
claims_physics = true
claims_maxwell_em = true
claims_scalar_potential = true
claims_density = true
claims_convergence = true
```

`SRC/F`: source attribution and intended FAIL-fixture ledger checks.

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
