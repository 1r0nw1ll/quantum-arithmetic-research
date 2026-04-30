# QA_STEINMETZ_WHITTAKER_BRIDGE_CERT.v1

## Status

Documentation-first empirical cert scaffold.

## Purpose

Validate transform consistency for the proposed bridge:

\[
\oint H\,dB \leftrightarrow \iint_{\Sigma}F_{\mathrm{QA}} \leftrightarrow \Delta\int d\phi
\]

under fixed QA tuple, fixed material/drive convention, and fixed calibration constants.

## Non-Claim

This cert validates deterministic transform consistency only. It does not validate or prove a universal physical identity between Steinmetz, Whittaker, Dollard, Bearden, or QA.

## Fixture Schema

Top-level object:

| Field | Type | Meaning |
|---|---|---|
| `cert_family` | string | Must be `QA_STEINMETZ_WHITTAKER_BRIDGE_CERT.v1`. |
| `case_id` | string | Fixture identifier. |
| `tolerance` | number | Absolute tolerance for transform consistency. |
| `tuple` | object | QA tuple with integer fields `b`, `e`, `d`, `a`. |
| `declared_invariants` | object | Declared QA invariants `J`, `X`, `K`, `F`, `C`, `G`. |
| `calibration` | object | Calibration section. |
| `evaluation` | object | Evaluation section. |
| `guardrail` | string | Human-readable non-claim statement. |

`calibration` object:

| Field | Type | Meaning |
|---|---|---|
| `material_drive_convention` | string | Fixed material/drive convention label. |
| `calibration_constants` | object | Numeric `alpha_X`, `alpha_J`, `alpha_K`. |

`evaluation` object:

| Field | Type | Meaning |
|---|---|---|
| `material_drive_convention` | string | Must match calibration convention. |
| `calibration_constants` | object | Must exactly match calibration constants. |
| `H` | number[] | Sampled magnetizing-field values. |
| `B` | number[] | Sampled flux-density values. |
| `theta` | number[] | Sampled unwrapped QA phase values. |
| `expected_hysteresis_area` | number | Declared reference for \(\oint H\,dB\). |
| `expected_curvature_proxy` | number | Declared reference for \(\oint \Pi\,d\theta\). |

`theta` is interpreted as an unwrapped phase coordinate. A full cycle may be represented as `0.0` to `1.0` or any other fixed convention, as long as the same convention is used consistently.

## Required Checks

1. `cert_family` must match `QA_STEINMETZ_WHITTAKER_BRIDGE_CERT.v1`.
2. Tuple relations must hold:
   - `d = b + e`
   - `a = b + 2*e`
3. Declared invariants must match:
   - `J = b*d`
   - `X = d*e`
   - `K = d*a`
   - `F = b*a`
   - `C = 2*e*d`
   - `G = e^2 + d^2`
4. `material_drive_convention` must be identical in calibration and evaluation sections.
5. `calibration_constants` must be identical in calibration and evaluation sections.
6. `H`, `B`, and `theta` must be same-length numeric arrays with at least two samples.
7. The closed-loop trapezoid integral \(\oint H\,dB\) must match `expected_hysteresis_area` within `tolerance`.
8. The QA curvature proxy \(\oint \Pi\,d\theta\) must match `expected_curvature_proxy` within `tolerance`.
9. Hysteresis area and QA curvature proxy must match each other within `tolerance`.

## Failure Fixture

`fixtures/fail_changed_calibration.json` must fail because `evaluation.calibration_constants` differs from `calibration.calibration_constants`.

## Output Contract

Normal fixture validation prints one line:

- `PASS: <reason>`
- `FAIL: <reason>`

`--self-test` exits `0` and prints canonical JSON:

```json
{"ok":true}
```

