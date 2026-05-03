# QA Whittaker Rational Direction S2 Cert v1

Candidate family ID: `[273]`. This artifact is not registered yet.

This cert builds Layer 2 of the Whittaker -> QA direction ladder: an exact finite rational direction set on `S2` generated from Layer 1 QA-rational ratios.

## Scope

The v1 claim is exact geometry only:

- `W3D_1`: every generated packet satisfies `x*x + y*y + z*z == den*den`.
- `W3D_2`: enumeration, duplicate counts, and `C/F` provenance collision counts are recomputed exactly.
- `W3D_3`: chart discipline is enforced for `inverse_stereographic_excluding_south_pole`.
- `W3D_4`: all generated pairs satisfy the finite denominator separation theorem.

`W3D_5` spherical Lipschitz sampling is intentionally deferred.

## C/F Provenance

Geometry uses the pooled ratio set `R_m = {C/G, F/G}`, but fixtures retain labeled `C/F` provenance metadata. This matters because the same rational ratio can arise from both QA channels.

Baseline provenance collision counts:

| m | values seen in both C and F channels |
|---|-------------------------------------:|
| 3 | 4                                    |
| 5 | 12                                   |
| 9 | 36                                   |

## Non-Claims

This cert does not prove Whittaker 1903, Maxwell equations, electromagnetism, scalar-potential physics, geodesy, ellipsoid physics, density, equidistribution, convergence, or a Whittaker wave-kernel approximation.

## Run

```bash
python3 qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s2_cert_v1/qa_whittaker_rational_direction_s2_cert_validate.py --self-test
```

Expected result:

```json
{"ok": true}
```

The full output includes all 8 fixture classifications: 3 PASS and 5 FAIL.
