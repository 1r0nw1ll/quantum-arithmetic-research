# Family [99] — QA Attention Spectral Gain Cert

**Cert root:** `qa_attn_spectral_gain_cert_v1/`  
**Validator:** `qa_attn_spectral_gain_cert_v1/validator.py --self-test`  
**Schema:** `QA_ATTN_SPECTRAL_GAIN_CERT.v1.schema.json`

## What it certifies

This cert derives the curvature gain from the natural Lipschitz constant of
the scaled-dot-product attention score map:

```
attn_gain = sigma_max(Q K^T / sqrt(d_k))
```

computed via power iteration on A^T A where A = Q K^T / sqrt(d_k).

This is the tightest spectral bound on how much the attention score matrix
can amplify a perturbation in key/query space, making it the correct
structural object for certifying attention-layer update stability.

## Update rule

```
p_after = p_before - lr · sigma_max(QK^T/sqrt(d_k)) · H_QA · grad
```

with κ = 1 − |1 − lr · σ_max · H_QA|

## Three gates

| Gate | Check |
|------|-------|
| A | Recompute H_QA from substrate; verify `claimed.H_QA` |
| B | Compute A = QK^T/sqrt(d_k); derive σ_max(A) via power iteration; verify `claimed.sigma_max`; pin update rule |
| C | Recompute κ = 1 − |1 − lr·σ_max·H_QA|; verify `claimed.kappa` |

## Derived-gain architecture comparison

| Cert | Architecture | gain source |
|------|-------------|-------------|
| [93]–[96] | GNN/Attn/QARM/Search | free scalar witness |
| [98] | GNN feature update | σ_max(W) — weight spectral norm |
| [99] | Attention score map | σ_max(QK^T/√d_k) — attention Lipschitz constant |

Together [98] and [99] establish that derived structural gain applies across
both graph/message-passing and sequence/attention architecture classes.

## Canonical fixture

- `Q`: [[0.6, 0.4], [0.3, 0.7]]
- `K`: [[0.5, 0.3], [0.2, 0.8]]
- `d_k`: 2
- `sigma_max`: 0.66033 (derived, not provided)
- `substrate`: (b=3, e=5, d=8, a=13), H_QA ≈ 0.4235
- `lr=0.01`, κ ≈ 0.002796
