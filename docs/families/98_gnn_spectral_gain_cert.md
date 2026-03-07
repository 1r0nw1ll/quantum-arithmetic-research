# Family [98] — QA GNN Spectral Gain Cert

**Cert root:** `qa_gnn_spectral_gain_cert_v1/`  
**Validator:** `qa_gnn_spectral_gain_cert_v1/validator.py --self-test`  
**Schema:** `QA_GNN_SPECTRAL_GAIN_CERT.v1.schema.json`

## What it certifies

This cert upgrades the curvature gain from a **free witness** to a **native
structural object**: the spectral norm σ_max(W) of the GNN weight matrix W,
computed via power iteration on W^T W.

In a GCN, the Lipschitz constant of the feature update w.r.t. input embeddings
is determined by the operator norm of the weight matrix. By deriving gain from
σ_max(W) rather than accepting a free scalar, the certificate becomes a
structural analysis of the actual weight geometry.

## Update rule

```
p_after = p_before - lr · σ_max(W) · H_QA · grad
```

with κ = 1 − |1 − lr · σ_max(W) · H_QA|

## Three gates

| Gate | Check |
|------|-------|
| A | Recompute H_QA from substrate via canonical formula; verify `claimed.H_QA` |
| B | Derive σ_max(W) via power iteration on W^T W; verify `claimed.sigma_max`; pin update rule |
| C | Recompute κ = 1 − |1 − lr·σ_max·H_QA|; verify `claimed.kappa` |

## Why this matters

| Cert type | gain source | What it proves |
|-----------|-------------|----------------|
| [93]–[96] | free scalar witness | consistency with κ formula |
| [98] | σ_max(W) via power iteration | structural spectral stability |

The distinction: [93]–[96] say "if you provide a gain in (0,2], κ behaves
this way." [98] says "the gain IS the spectral norm of your weight matrix,
and here is the certified stability margin."

## Canonical fixture

- `weight_matrix`: [[0.8, 0.2], [0.1, 0.6]] (2×2)
- `sigma_max`: 0.88206 (derived, not provided)
- `substrate`: (b=3, e=5, d=8, a=13), H_QA ≈ 0.4235
- `lr=0.01`, `kappa` ≈ 0.003735
