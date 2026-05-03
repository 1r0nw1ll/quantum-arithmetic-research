# [273] QA Whittaker Rational Direction S2 Cert

## What this is

Layer 2 of the Whittaker -> QA development ladder. Layer 1 is `[266]`, the registered rational direction cert on `S1`. This family certifies an exact rational direction set on `S2` as a geometry substrate for later Whittaker wave-kernel work.

Primary source anchor:

- E. T. Whittaker (1903), *On the partial differential equations of mathematical physics*, Math. Annalen 57:333-355. DOI: 10.1007/BF01444290.

Whittaker 1903 motivates the spherical direction substrate. This cert does not prove Whittaker's theorem.

## Claim

For each `m in {3,5,9}`, define the QA-derived rational parameter pool:

```text
R_m = { C/G, F/G : (b,e) in {1..m}^2, gcd(b,e)=1,
                    d=b+e, a=b+2e,
                    C=2*d*e, F=a*b, G=d*d+e*e }
```

Then generate:

```text
D_m^(2) = { S(r,s) : r in R_m, s in R_m }
```

using the rational inverse stereographic chart:

```text
S(r,s) = (
  2r / (1 + r*r + s*s),
  2s / (1 + r*r + s*s),
  (1 - r*r - s*s) / (1 + r*r + s*s)
)
```

The cert proves finite exact `S2` construction and separation properties only.

## Bit-Exact Predictions

| m | seed_count | unique_R_count | unique_S2_direction_count | pair_count | N_max | C/F provenance collisions |
|---|-----------:|---------------:|--------------------------:|-----------:|------:|--------------------------:|
| 3 | 7          | 10             | 100                       | 4,950      | 1,260,041 | 4 |
| 5 | 19         | 26             | 676                       | 228,150    | 175,808,753 | 12 |
| 9 | 55         | 74             | 5,476                     | 14,990,550 | 32,889,577,313 | 36 |

True finite-set minimum normalized `sin_sq` witnesses:

| m | true_min_normalized_sin_sq | observer angle approx |
|---|----------------------------|----------------------:|
| 3 | `187385728680000/271030516650563569` | 0.0262971822419 |
| 5 | `1187255997390145600/39768198848578581524641` | 0.00546394584372 |
| 9 | `33955705631283190632000/39974350779836816952622268161` | 0.000921649372821 |

## Validator Gates

| Gate | Check |
|------|-------|
| W3D_1 | Exact sphere identity: `x*x + y*y + z*z == den*den` for every generated packet |
| W3D_2 | Recompute `seed_count`, `raw_ratio_count`, `unique_R_count`, `raw_pair_count`, `unique_S2_direction_count`, `duplicate_count`, and `R_channel_provenance_collision_count` |
| W3D_3 | Enforce `inverse_stereographic_excluding_south_pole` chart discipline and reject full-sphere, antipodal, sign-reflection, density, and equidistribution overclaims |
| W3D_4 | Check all pairs for the finite denominator separation theorem and report the true minimum normalized `sin_sq` witness |
| W3D_SRC | Source attribution must cite Whittaker 1903 and DOI |
| W3D_F | Fail ledgers well-formed for FAIL fixtures |

For distinct generated packets:

```text
sin_sq(theta_ij)
= cross_norm_sq_num / (den_i*den_i * den_j*den_j)
>= 1 / (den_i*den_i * den_j*den_j)
>= 1 / N_max(m)^4
```

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s2_cert_v1/qa_whittaker_rational_direction_s2_cert_validate.py` |
| Mapping ref | `qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s2_cert_v1/mapping_protocol_ref.json` |
| PASS fixtures | `qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s2_cert_v1/fixtures/pass_s2_*.json` |
| FAIL fixtures | `qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s2_cert_v1/fixtures/fail_s2_*.json` |
| Design draft | `docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_S2_CERT_DRAFT.md` |

## How to run

```bash
python3 qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s2_cert_v1/qa_whittaker_rational_direction_s2_cert_validate.py --self-test
```

Expected: `{"ok": true, "results": [...]}` with all 8 fixtures classified correctly.

## Non-Claims

This cert does not prove:

- Whittaker 1903.
- Maxwell equations.
- Electromagnetism.
- Scalar-potential physics.
- Geodesy or ellipsoid physics.
- Density, equidistribution, or convergence.
- A Whittaker wave-kernel approximation.
- Full-sphere, antipodal, or sign-reflection closure.

`W3D_5` spherical Lipschitz sampling is deferred to a later layer or revision.

## Status

Registered as family `[273]` after standalone artifact commit `997bfcc` and pre-registration review. Registry integration is intentionally separate from the artifact build.
