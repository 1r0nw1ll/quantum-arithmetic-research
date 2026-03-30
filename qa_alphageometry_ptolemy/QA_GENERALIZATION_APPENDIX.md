# Appendix: Generalization as Gauge-Invariant Reachability

**A QA Framework Interpretation of Architecture-Independent Generalization Bounds**

---

## 1. The Core Reframing

Classical learning theory asks: *How many parameters can a network have before it memorizes?*

QA asks a different question: *What invariants control the network's reachable function class?*

The answer from arXiv:2504.05695 is striking:

> **Generalization depends on operator norms and data geometry, not parameter count.**

In QA terms, this means:

| Classical View | QA View |
|---------------|---------|
| Parameters = capacity | Parameters = coordinates (most are gauge) |
| Overparametrization = risk | Overparametrization = gauge freedom |
| Architecture = hypothesis class | Architecture = coordinate choice |
| Training = search | Training = gauge fixing |

---

## 2. Overparametrization as Gauge Freedom

### The Key Insight

When a network has more parameters than needed for zero training loss, the "extra" parameters form a **null space**. Changing them doesn't change the network's behavior on training data.

In physics, this is called **gauge freedom**—extra degrees of freedom that can be fixed arbitrarily without changing physical predictions.

### QA Formalization

```
total_params = 11,173,962  (ResNet-18 on CIFAR-10)
minimal_params = 153,600   (sufficient for zero loss)
gauge_dim = 11,020,362     (can be set arbitrarily)
```

The gauge dimension is **not** "extra capacity for memorization." It's coordinates that don't affect the certificate.

### Why This Matters

1. **VC dimension is irrelevant**: It counts parameters, not invariant-controlled quantities
2. **"Double descent" demystified**: The second descent is gauge fixing, not regularization
3. **Implicit regularization explained**: SGD picks a gauge, not a simpler function

---

## 3. Zero-Loss ≠ Overfitting

### The Constructor Theorem

For n ≤ d (samples ≤ input dimension), the paper proves:

> There exists an **explicit constructor** that achieves zero training loss without any gradient descent.

This is a QA **SUCCESS certificate** with:
- Explicit weight matrices
- Explicit biases
- Verified zero residuals
- No optimization involved

### QA Interpretation

Zero training loss is **not** evidence of overfitting when:
1. The loss is achieved by a controlled constructor (not unconstrained search)
2. The operator norms are bounded
3. The data geometry is well-behaved

The certificate schema captures this:

```json
{
  "schema": "QA_ZERO_LOSS_CONSTRUCTOR_V1",
  "n_samples": 50,
  "input_dim": 100,
  "method": "interpolation",
  "verified_zero_loss": true,
  "residuals": ["0/1", "0/1", "0/1"]
}
```

---

## 4. Failures Are First-Class Objects

### The QA Difference

Classical learning theory: "The bound is vacuous" → end of story.

QA: "The bound is vacuous" → **failure certificate with obstruction witness**.

### Structured Failure Modes

| Failure | Obstruction | Remediation |
|---------|-------------|-------------|
| `bound_vacuous` | Computed bound > 1 | Spectral normalization, more data |
| `norm_explosion` | Specific layer identified | Regularize that layer |
| `insufficient_samples` | n vs. threshold | Data augmentation |
| `metric_degeneracy` | Collapsed distances | Feature preprocessing |

### Why This Matters

Failures become **actionable**. Instead of "the theory doesn't apply," we get "here's what to fix."

Example failure certificate:
```json
{
  "success": false,
  "failure_mode": "bound_vacuous",
  "failure_witness": {
    "computed_bound": "147/100",
    "threshold": "1/1",
    "contributing_factors": [
      "spectral_product too large",
      "insufficient samples"
    ],
    "remediation": [
      "Apply spectral normalization",
      "Increase training set size"
    ]
  }
}
```

---

## 5. Architecture-Independence via Certificates

### The Validation

We validated the same certificate schema across:
- MLP (simple feedforward)
- ResNet (residual connections)
- VGG (deep convolutions)

All produce valid certificates with:
- Same metric geometry (data is fixed)
- Different operator norms (architecture-dependent)
- Different gauge freedom (architecture-dependent)
- Coherent bounds (architecture-independent formula)

### The Implication

**Architecture is a coordinate choice, not a hypothesis class.**

The generalization bound formula:

```
gap ≤ C × D_geom × (∏ ||W_l||₂) × (1 + Σ ||b_l||₂) / √n
```

Contains:
- `D_geom`: Data property (architecture-independent)
- `||W_l||₂`: Controllable via regularization
- `||b_l||₂`: Controllable via regularization
- `n`: Data property (architecture-independent)

Nothing about depth, width, or parameter count.

---

## 6. The Certificate as Proof Object

### What the Certificate Proves

A valid `QA_GENERALIZATION_CERT_V1` with `success: true` proves:

1. **Data geometry is well-behaved**: `metric_geometry` witness shows D_geom is finite
2. **Norms are controlled**: `operator_norms` witness shows bounded energy
3. **Bound is non-vacuous**: `generalization_bound < 1` for classification
4. **Empirical tracking**: `tracking_error` shows theory matches practice

### What the Certificate Cannot Prove

- That the specific network will generalize (only that the *bound* holds)
- That training will find a good solution (only that one exists)
- That the bound is tight (only that it's valid)

### Recompute Hooks

Independent verification via:
- `metric_geometry_v1`: Recompute D_geom from raw data
- `operator_norm_v1`: Recompute norms from weight matrices
- `generalization_bound_v1`: Recompute bound from witnesses

---

## 7. Summary

| Classical Learning Theory | QA Certificate Framework |
|--------------------------|-------------------------|
| Bounds on hypothesis class | Bounds on invariant-controlled reachability |
| VC dimension matters | Gauge dimension is irrelevant |
| Overparametrization is dangerous | Overparametrization is gauge freedom |
| Zero loss implies overfitting | Zero loss is constructive (for n ≤ d) |
| Vacuous bounds are failures | Vacuous bounds are **informative** failures |
| Architecture determines generalization | Norms + geometry determine generalization |

---

## Citation

```bibtex
@misc{qa_generalization_appendix_2026,
  title={Generalization as Gauge-Invariant Reachability: A QA Framework Interpretation},
  author={Signal Experiments Research Group},
  year={2026},
  note={Appendix to QA mapping of arXiv:2504.05695}
}
```
