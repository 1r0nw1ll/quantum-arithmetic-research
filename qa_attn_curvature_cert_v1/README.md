# QA Attention Layer Curvature Cert v1

Pins an attention-layer curvature scalar `kappa` together with the substrate curvature `H_QA` and an optimizer update-rule witness scaled by `attn_gain`.

## Definitions

- `G = e*e + d*d`
- `F = b*a`
- `loss_hat = F / (G + eps)`
- `h_raw = 0.25 * (F/(G+eps) + (e*d)/(a+b+eps))`
- `H_QA = abs(h_raw) / (1 + abs(h_raw))`
- `kappa = 1 - abs(1 - lr * attn_gain * H_QA)`
- Update witness: `p_after = p_before - lr * attn_gain * grad`

## Fields

- `attn_gain`: user-supplied scalar witness in `(0,2]` (interpretable as a spectral-norm witness of the attention output projection); strict rejection if out of range.
- `n_heads`, `d_model`, `seq_len`: structural metadata for the attention layer / sequence.

## Gates

- Gate 1 (schema): JSON-schema validation (`schema.json`).
- Gate 2A (substrate): recompute `h_raw`, `H_QA`, `loss_hat` and pin them to `claimed.*`.
- Gate 2B (optimizer): strict `attn_gain ∈ (0,2]` and pin the update witness.
- Gate 2C (kappa): recompute `kappa` and pin it to `claimed.kappa`.

## Pass / fail

- PASS iff all gates pass and `claimed.*` matches deterministic recompute within tolerance.
- FAIL with a concrete `fail_type` and `invariant_diff` witness on first mismatch.

## Run

```bash
python qa_attn_curvature_cert_v1/validator.py --self-test
```

