# Family [96]: QA Symbolic Search Curvature Cert v1

## What it is

Family [96] pins a symbolic-search curvature scalar `kappa` together with:

- the substrate curvature `H_QA` recomputed from integer substrate parameters `(b,e,d,a)`, and
- a concrete optimizer update witness using `sym_gain` (plus audit-only search metadata: `beam_width`, `search_depth`, `rule_count`).

It is intended as a drift detector: if the substrate curvature formula, the curvature-scaled update rule, or the `kappa` definition changes, the cert fails with a concrete mismatch witness.

## Cert root

`qa_symbolic_search_curvature_cert_v1/`

## Definitions

- `G = e*e + d*d`
- `F = b*a`
- `eps = 1e-12`
- `H_raw = 0.25 * (F/(G+eps) + e*d/(a+b+eps))`
- `H_QA = abs(H_raw) / (1 + abs(H_raw))`
- Update witness: `p_after = p_before - lr * sym_gain * H_QA * grad`
- `kappa = 1 - abs(1 - lr * sym_gain * H_QA)`

## Files

| File | Purpose |
|---|---|
| `schema.json` | v1 schema for symbolic search curvature certs |
| `validator.py` | schema + deterministic recompute + update + kappa pin |
| `mapping_protocol_ref.json` | Gate-0 mapping protocol pin |
| `fixtures/pass_default_sym.json` | PASS fixture |
| `fixtures/fail_sym_gain_mismatch.json` | FAIL fixture (Gate B: update-rule mismatch) |
| `fixtures/fail_h_qa_mismatch.json` | FAIL fixture (Gate A: curvature mismatch) |
| `fixtures/fail_beam_width_invalid.json` | FAIL fixture (Gate 1: schema invalid) |

## Gates

| Gate | Check |
|---|---|
| 1 | Schema validity |
| A | Deterministic recompute of `H_QA` |
| B | Strict `sym_gain ∈ (0,2]` and deterministic update witness |
| C | Deterministic recompute of `kappa` |

## How to run

```bash
python qa_symbolic_search_curvature_cert_v1/validator.py --self-test
python qa_symbolic_search_curvature_cert_v1/validator.py \
  --schema qa_symbolic_search_curvature_cert_v1/schema.json \
  --cert qa_symbolic_search_curvature_cert_v1/fixtures/pass_default_sym.json
```

## Failure types

- `SCHEMA_INVALID`
- `H_QA_MISMATCH`
- `UPDATE_RULE_MISMATCH`
- `KAPPA_MISMATCH`

