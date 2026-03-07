# Family [95]: QA QARM Curvature Cert v1

## What it is

Family [95] pins a QARM curvature scalar `kappa` together with:

- the substrate curvature `H_QA` recomputed from a scalar tuple `(b,e,d,a,eps)`, and
- a concrete optimizer update witness using `qarm_gain`.

It is intended as a drift detector: if the substrate formula, the update rule, or the `kappa` definition changes, the cert fails with a concrete mismatch witness.

## Cert root

`qa_qarm_curvature_cert_v1/`

## Definitions

- `G = e*e + d*d`
- `F = b*a`
- `loss_hat = F / (G + eps)`
- `h_raw = 0.25 * (F/(G+eps) + (e*d)/(a+b+eps))`
- `H_QA = abs(h_raw) / (1 + abs(h_raw))`
- `kappa = 1 - abs(1 - lr * qarm_gain * H_QA)`
- Update witness: `p_after = p_before - lr * qarm_gain * grad`

## Structural metadata

- `modulus`: arithmetic modulus (e.g. 9 or 24).
- `orbit_size`: size of the state orbit containing the current state.
- `generator`: name of the QARM generator (e.g. `"sigma"`, `"mu"`, `"lambda_2"`, `"nu"`).

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for QARM curvature certs |
| `validator.py` | schema + deterministic recompute + update + kappa pin |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/pass_default_qarm.json` | PASS fixture |
| `fixtures/fail_qarm_gain_mismatch.json` | FAIL fixture (Gate 2B: update-rule mismatch) |
| `fixtures/fail_h_qa_mismatch.json` | FAIL fixture (Gate 2A: curvature mismatch) |
| `fixtures/fail_modulus_invalid.json` | FAIL fixture (Gate 1: schema invalid) |

## Gates

| Gate | Check |
|---|---|
| 1 | Schema validity |
| 2A | Deterministic recompute of `H_QA_raw`, normalized `H_QA`, and `loss_hat` |
| 2B | Strict `qarm_gain ∈ (0,2]` and deterministic update witness |
| 2C | Deterministic recompute of `kappa` |

## How to run

```bash
python qa_qarm_curvature_cert_v1/validator.py --self-test
python qa_qarm_curvature_cert_v1/validator.py \
  --schema qa_qarm_curvature_cert_v1/schema.json \
  --cert qa_qarm_curvature_cert_v1/fixtures/pass_default_qarm.json
```

## Failure types

- `SCHEMA_INVALID`
- `EPS_MISMATCH`
- `H_QA_MISMATCH`
- `LOSS_HAT_MISMATCH`
- `QARM_GAIN_OUT_OF_RANGE`
- `UPDATE_RULE_MISMATCH`
- `KAPPA_MISMATCH`

