# [274] QA Whittaker Scalar Angular-Kernel Sampling Cert

## What this is

Layer 3 v1 of the Whittaker -> QA development ladder. Layer 1 is `[266]`,
the registered rational direction cert on `S1`; Layer 2 is `[273]`, the
registered rational direction cert on `S2`.

This family certifies exact finite scalar-profile sampling over the `[273]`
`S2` direction substrate. It does not certify Whittaker's wave kernel.

Primary source anchor:

- E. T. Whittaker (1903), *On the partial differential equations of
  mathematical physics*, Math. Annalen 57:333-355. DOI:
  10.1007/BF01444290.

Whittaker 1903 motivates angular superposition over directions. This cert
only validates deterministic finite scalar sampling over the exact rational
direction packets supplied by `[273]`.

## Claim

For:

```text
m in {3,5,9}
profile_name in {const,z,z2}
weight_rule = uniform_points
```

the validator recomputes `[273]` `D_m^(2)` packets and evaluates:

```text
const = 1
z     = z_i/den_i
z2    = (z_i/den_i)*(z_i/den_i)
```

as exact `Fraction` values. It then checks:

```text
discrete_uniform_average = (sum_i h_i) / |D_m^(2)|
```

This is a finite-set average, not a spherical quadrature rule.

## Exact Witnesses

Small rational averages may be declared directly by numerator and
denominator. Large exact averages are witnessed by:

```text
canonical_fraction_sha256 = sha256((num + "/" + den).encode("ascii")).hexdigest()
numerator_digit_count
denominator_digit_count
```

The validator recomputes the exact `Fraction`, builds the canonical `num/den`
string, and checks the full SHA-256 hash plus digit counts.
`canonical_fraction_sha256_16` is display/reporting only.

## Fixture Coverage

PASS:

| Fixture | Purpose |
|---------|---------|
| `pass_wkb_m3_const_uniform.json` | sanity fixture: `m=3`, `const`, direct `1/1` average |
| `pass_wkb_m5_z_uniform.json` | chart-bias fixture: `m=5`, `z`, full hash witness |
| `pass_wkb_m9_z2_uniform_hash.json` | stress fixture: `m=9`, `z2`, large-rational hash witness |

FAIL:

| Fixture | Rejected condition |
|---------|--------------------|
| `fail_wkb_wrong_provenance.json` | dependency family is not registered `[273]` |
| `fail_wkb_wrong_profile_value.json` | declared rational average does not match recomputation |
| `fail_wkb_wrong_hash_witness.json` | full canonical SHA-256 hash mismatch |
| `fail_wkb_wrong_digit_count.json` | numerator/denominator digit count mismatch |
| `fail_wkb_float_used_as_pass_fail.json` | observer float used as pass/fail basis |
| `fail_wkb_overclaimed_spherical_integral.json` | `uniform_points` overclaimed as spherical quadrature |
| `fail_wkb_overclaimed_whittaker_error.json` | finite average overclaimed as Whittaker-kernel error |

## Validator Gates

| Gate | Check |
|------|-------|
| WKB_1 | Dependency provenance requires registered `[273]`, expected slug, and `inverse_stereographic_excluding_south_pole` chart |
| WKB_2 | Profile restricted to `{const,z,z2}` |
| WKB_3 | `weight_rule = uniform_points` |
| WKB_4 | Exact `Fraction` `discrete_uniform_average` recomputation and direct/hash witness validation |
| WKB_5 | Observer diagnostics are display-only; float pass/fail and overclaims are rejected |
| WKB_SRC | Source attribution must cite Whittaker 1903 and DOI |
| WKB_F | Fail ledgers well-formed for FAIL fixtures |

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_alphageometry_ptolemy/qa_whittaker_scalar_angular_kernel_sampling_cert_v1/qa_whittaker_scalar_angular_kernel_sampling_cert_validate.py` |
| Mapping ref | `qa_alphageometry_ptolemy/qa_whittaker_scalar_angular_kernel_sampling_cert_v1/mapping_protocol_ref.json` |
| PASS fixtures | `qa_alphageometry_ptolemy/qa_whittaker_scalar_angular_kernel_sampling_cert_v1/fixtures/pass_wkb_*.json` |
| FAIL fixtures | `qa_alphageometry_ptolemy/qa_whittaker_scalar_angular_kernel_sampling_cert_v1/fixtures/fail_wkb_*.json` |
| Design draft | `docs/specs/QA_WHITTAKER_WAVE_KERNEL_BRIDGE_CERT_DRAFT.md` |

## How to run

```bash
python3 qa_alphageometry_ptolemy/qa_whittaker_scalar_angular_kernel_sampling_cert_v1/qa_whittaker_scalar_angular_kernel_sampling_cert_validate.py --self-test
```

Expected: `{"ok": true, "results": [...]}` with all 10 fixtures classified
correctly.

## Non-Claims

This cert does not prove:

- Whittaker 1903.
- Whittaker's wave kernel.
- Spherical quadrature.
- Density or convergence.
- Maxwell equations.
- Electromagnetism.
- Scalar-potential physics.
- Geodesy or ellipsoid physics.
- Any physical field reconstruction.

## Status

Registered as family `[274]` after standalone artifact commit `055df15` and
pre-registration review. Registry integration is intentionally separate from
the artifact build.
