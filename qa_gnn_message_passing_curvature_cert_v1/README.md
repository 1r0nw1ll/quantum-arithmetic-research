# QA GNN Message-Passing Curvature Cert v1

Pins a GNN message-passing curvature scalar `kappa` together with the substrate curvature `H_QA` and an optimizer update-rule witness.

## Definitions

- `G = e*e + d*d`
- `F = b*a`
- `loss_hat = F / (G + eps)`
- `h_raw = 0.25 * (F/(G+eps) + (e*d)/(a+b+eps))`
- `H_QA = abs(h_raw) / (1 + abs(h_raw))`
- `kappa = 1 - abs(1 - lr * agg_gain * H_QA)`
- Update witness: `p_after = p_before - lr * agg_gain * grad`

## Gates

- Gate 1 (schema): JSON-schema validation (`schema.json`).
- Gate 2A (substrate): recompute `h_raw`, `H_QA`, `loss_hat` and pin them to `claimed.*`.
- Gate 2B (optimizer): strict `agg_gain ∈ (0,2]` and pin the update witness.
- Gate 2C (kappa): recompute `kappa` and pin it to `claimed.kappa`.

## Run

```bash
python qa_gnn_message_passing_curvature_cert_v1/validator.py --self-test
```

