# Family [94]: QA Attention Layer Curvature Cert v1

## What it is

Family [94] pins an attention-layer curvature scalar `kappa` together with:

- the substrate curvature `H_QA` recomputed from a scalar tuple `(b,e,d,a,eps)`, and
- a concrete optimizer update witness using `attn_gain`.

It is intended as a drift detector: if the substrate formula, the update rule, or the `kappa` definition changes, the cert fails with a concrete mismatch witness.

## Cert root

`qa_attn_curvature_cert_v1/`

## Definitions

- `G = e*e + d*d`
- `F = b*a`
- `loss_hat = F / (G + eps)`
- `h_raw = 0.25 * (F/(G+eps) + (e*d)/(a+b+eps))`
- `H_QA = abs(h_raw) / (1 + abs(h_raw))`
- `kappa = 1 - abs(1 - lr * attn_gain * H_QA)`
- Update witness: `p_after = p_before - lr * attn_gain * grad`

## Structural metadata

- `n_heads`, `d_model`, `seq_len`: attention layer / sequence structure metadata (schema-gated).

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for attention curvature certs |
| `validator.py` | schema + deterministic recompute + update + kappa pin |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/pass_default_attn.json` | PASS fixture |
| `fixtures/fail_attn_gain_mismatch.json` | FAIL fixture (Gate 2B: update-rule mismatch) |
| `fixtures/fail_h_qa_mismatch.json` | FAIL fixture (Gate 2A: curvature mismatch) |
| `fixtures/fail_seq_len_invalid.json` | FAIL fixture (Gate 1: schema invalid) |

## Gates

| Gate | Check |
|---|---|
| 1 | Schema validity |
| 2A | Deterministic recompute of `H_QA_raw`, normalized `H_QA`, and `loss_hat` |
| 2B | Strict `attn_gain ∈ (0,2]` and deterministic update witness |
| 2C | Deterministic recompute of `kappa` |

## How to run

```bash
python qa_attn_curvature_cert_v1/validator.py --self-test
python qa_attn_curvature_cert_v1/validator.py \
  --schema qa_attn_curvature_cert_v1/schema.json \
  --cert qa_attn_curvature_cert_v1/fixtures/pass_default_attn.json
```

## Failure types

- `SCHEMA_INVALID`
- `EPS_MISMATCH`
- `H_QA_MISMATCH`
- `LOSS_HAT_MISMATCH`
- `ATTN_GAIN_OUT_OF_RANGE`
- `UPDATE_RULE_MISMATCH`
- `KAPPA_MISMATCH`

