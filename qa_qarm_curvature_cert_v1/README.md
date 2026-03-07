# QA QARM Curvature Cert v1

Pins a QARM curvature scalar `kappa_QARM` together with the substrate curvature `H_QA` and an optimizer update-rule witness scaled by `qarm_gain`.

## Definitions

- `G = e*e + d*d`
- `F = b*a`
- `loss_hat = F / (G + eps)`
- `h_raw = 0.25 * (F/(G+eps) + (e*d)/(a+b+eps))`
- `H_QA = abs(h_raw) / (1 + abs(h_raw))`
- `kappa_QARM = 1 - abs(1 - lr * qarm_gain * H_QA)`
- Update witness: `p_after = p_before - lr * qarm_gain * grad`

## Fields

- `qarm_gain`: user-supplied scalar witness in `(0,2]` (effective spectral gain of one QARM generator application on the state orbit); strict rejection if out of range.
- `modulus`, `orbit_size`, `generator`: structural metadata for the QARM substrate (arithmetic modulus, size of the orbit containing the current state, and generator name).

## Gates

- Gate 1 (schema): JSON-schema validation (`schema.json`).
- Gate 2A (substrate): recompute `h_raw`, `H_QA`, `loss_hat` and pin them to `claimed.*`.
- Gate 2B (optimizer): strict `qarm_gain ∈ (0,2]` and pin the update witness.
- Gate 2C (kappa): recompute `kappa_QARM` and pin it to `claimed.kappa`.

## Pass / fail

- PASS iff all gates pass and `claimed.*` matches deterministic recompute within tolerance.
- FAIL with a concrete `fail_type` and `invariant_diff` witness on first mismatch.

## Run

```bash
python qa_qarm_curvature_cert_v1/validator.py --self-test
```
