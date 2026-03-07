# Family [101] — QA Gradient Lipschitz Gain Cert

**Cert root:** `qa_gradient_lipschitz_gain_cert_v1/`  
**Validator:** `qa_gradient_lipschitz_gain_cert_v1/validator.py --self-test`  
**Schema:** `QA_GRADIENT_LIPSCHITZ_GAIN_CERT.v1.schema.json`

## What it certifies

Derives the curvature gain from the L2 norm of the gradient vector — the
natural local Lipschitz constant of the gradient descent update step:

```
gain = min(||grad_vector||_2, 2.0)
```

The gradient vector is the native object. Its norm is computed internally
by the validator; the claimer cannot inject an arbitrary gain value.

## Update rule

```
p_after = p_before - lr · min(||grad||_2, 2.0) · H_QA · grad
```

with κ = 1 − |1 − lr · gain · H_QA|

## Three gates

| Gate | Check |
|------|-------|
| A | Recompute H_QA from substrate; verify `claimed.H_QA` |
| B | Recompute `||grad_vector||_2`; verify `claimed.grad_norm` and `claimed.gain = min(grad_norm, 2.0)`; pin update rule |
| C | Recompute κ = 1 − |1 − lr·gain·H_QA|; verify `claimed.kappa` |

## Three-architecture derived-gain comparison

| Cert | Architecture | Native object | gain derivation |
|------|-------------|---------------|-----------------|
| [98] | GNN feature update | Weight matrix W | σ_max(W) via power iter. |
| [99] | Attention score map | Score matrix QKᵀ/√d_k | σ_max(·) via power iter. |
| [101] | Gradient descent | Gradient vector g | min(||g||_2, 2.0) |

Together these three establish that derived structural gain is applicable
across graph, sequence, and gradient-descent architecture classes. The
paper can now say: **"derived gain demonstrated in three architecture classes."**

## Canonical fixture

- `grad_vector`: [0.3, 0.4] (2D)
- `grad_norm`: 0.5 (||[0.3, 0.4]||_2 = 0.5 exactly)
- `gain`: 0.5 (below 2.0 cap)
- `substrate`: (b=3, e=5, d=8, a=13), H_QA ≈ 0.4235
- `lr=0.01`, κ ≈ 0.002117
