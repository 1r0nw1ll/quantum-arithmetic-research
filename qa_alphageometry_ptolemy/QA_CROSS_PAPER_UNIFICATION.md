# QA Cross-Paper Unification

**Three Papers, One Principle: Invariant-Controlled Reachability**

---

## The Unified Claim

Three apparently different ML/AI papers all reduce to the same QA structure:

| Paper | Domain | Surface Question |
|-------|--------|------------------|
| arXiv:2504.05695 | Learning Theory | Why do overparametrized networks generalize? |
| NeuralGCM | Climate/Weather | How do hybrid physics-ML models maintain physical validity? |
| Sparse Attention | NLP/Transformers | Why can we prune attention without losing performance? |

**Unified QA Answer:**

> **Performance guarantees come from invariant control, not architectural choices.**
> **Extra capacity is gauge freedom, not memorization risk.**

---

## The Three Pillars

### Pillar 1: Invariant Control

Each paper identifies quantities that *must* be controlled for valid certificates:

| Paper | Controlled Invariants | QA Structure |
|-------|----------------------|--------------|
| Generalization | Operator norms (||W||₂, ||b||₂) | `OperatorNormWitness` |
| NeuralGCM | Conservation laws (mass, energy, momentum) | `ConservationWitness` |
| Sparse Attention | Attention entropy, effective rank | `EntropyWitness`, `RankWitness` |

**Unified principle:** The certificate depends on these invariants, not on architectural details.

---

### Pillar 2: Gauge Freedom

Each paper has "extra" degrees of freedom that don't affect the certificate:

| Paper | Gauge Freedom | QA Interpretation |
|-------|--------------|-------------------|
| Generalization | Overparametrization (params beyond minimal) | Null space of loss Hessian |
| NeuralGCM | Neural parameterizations (within physics constraints) | Subgrid physics choices |
| Sparse Attention | Redundant attention heads | Prunable without performance loss |

**Unified principle:** Gauge coordinates can be fixed arbitrarily without changing validity.

```
Generalization:   gauge_dim = total_params - minimal_params
NeuralGCM:        gauge_dim = neural_params (physics is fixed)
Sparse Attention: gauge_dim = prunable_heads × head_dim
```

---

### Pillar 3: Structured Failures

Each paper has failure modes that produce *informative* obstruction certificates:

| Paper | Failure Mode | Obstruction Type |
|-------|--------------|------------------|
| Generalization | Vacuous bound | `bound > 1` with contributing factors |
| Generalization | Norm explosion | Specific layer identified |
| NeuralGCM | Conservation violation | Mass/energy delta with location |
| NeuralGCM | Skill collapse | Forecast horizon where skill < climatology |
| Sparse Attention | Rank collapse | Effective rank << sequence length |
| Sparse Attention | Entropy collapse | All attention on single token |

**Unified principle:** Failures are first-class objects with remediation paths.

---

## The Mapping Table

| Concept | Generalization | NeuralGCM | Sparse Attention |
|---------|---------------|-----------|------------------|
| **State space** | Function class | Atmospheric state | Token representations |
| **Transitions** | Layer operations | Dynamics + parameterization | Attention + FFN |
| **Invariants** | Norm bounds | Conservation laws | Entropy/rank bounds |
| **Gauge freedom** | Overparametrization | Neural params | Redundant heads |
| **Certificate** | Generalization bound | Forecast skill | Task performance |
| **Failure** | Vacuous/explosion | Violation/collapse | Collapse/instability |

---

## Obstruction Taxonomy (Unified)

All failure modes across the three papers can be classified into four categories:

### Category 1: Invariant Violation
The controlled quantity exceeds its bound.

| Paper | Example |
|-------|---------|
| Generalization | `spectral_product > threshold` |
| NeuralGCM | `mass_delta > tolerance` |
| Sparse Attention | `entropy < min_threshold` |

### Category 2: Structural Degeneracy
The system loses effective dimensionality.

| Paper | Example |
|-------|---------|
| Generalization | Rank-deficient weight matrix |
| NeuralGCM | Collapsed atmospheric layer |
| Sparse Attention | `effective_rank << sequence_length` |

### Category 3: Computational Intractability
The certificate cannot be computed.

| Paper | Example |
|-------|---------|
| Generalization | Non-Lipschitz activation |
| NeuralGCM | CFL violation (unstable numerics) |
| Sparse Attention | Kernel feature instability |

### Category 4: Empirical Invalidity
The bound exists but doesn't match reality.

| Paper | Example |
|-------|---------|
| Generalization | Bound valid but vacuous (> 1) |
| NeuralGCM | Forecast worse than climatology |
| Sparse Attention | Sparse pattern loses critical information |

---

## Certificate Coherence Across Papers

When applying QA to a system that spans multiple papers (e.g., a transformer-based weather model), certificates must be **coherent**:

```yaml
cross_certificate_coherence:
  - check: "attention_entropy_bounds_imply_norm_control"
    description: "Low attention entropy → bounded effective operator norm"

  - check: "conservation_implies_bounded_drift"
    description: "Conservation witness → bounded state space drift"

  - check: "gauge_freedom_additive"
    description: "Total gauge = sum of component gauge freedoms"
```

---

## Implementation Status

| Mapping | Status | Certificates | Validator |
|---------|--------|--------------|-----------|
| Generalization (arXiv:2504.05695) | ✅ Complete | 4 examples | Strict v3 |
| NeuralGCM | ✅ Complete | 2 examples | Strict v3 |
| Sparse Attention | ✅ Complete | 2 examples | Strict v3 |
| **Cross-paper unification** | ✅ This document | — | — |

### Validation Results (2026-01-24)

All three Gold Standard mappings pass validation:

```
Generalization Bounds:
  ✔ mnist_mlp_success.json:     15 passed
  ✔ vacuous_bound_failure.json:  6 passed

NeuralGCM (Physics-ML):
  ✔ 10day_forecast_success.json: 24 passed
  ✔ mass_violation_failure.json:  6 passed

Sparse Attention:
  ✔ bert_base_success.json:      22 passed
  ✔ rank_collapse_failure.json:   6 passed
```

---

## Completed Phases

### Phase 1: Individual Mappings ✅
1. ✅ `qa_generalization_certificate.py` with metric geometry, operator norm, activation regularity witnesses
2. ✅ `qa_neuralgcm_certificate.py` with conservation, forecast skill, physical bounds witnesses
3. ✅ `qa_sparse_attention_certificate.py` with entropy, rank, sparsity, head redundancy witnesses
4. ✅ Strict v3 validators for all three domains

### Phase 2: Cross-Certificate Coherence (Partial)
1. ✅ Unified obstruction taxonomy (4 categories documented above)
2. ⏳ Bundle coherence validator (future work)
3. ⏳ Cross-domain example bundles (future work)

### Phase 3: Unified Obstruction Algebra (Partial)
1. ✅ Four obstruction categories formalized (Invariant Violation, Structural Degeneracy, Computational Intractability, Empirical Invalidity)
2. ⏳ Remediation move mapping (future work)
3. ⏳ Obstruction lattice visualization (future work)

---

## The Bottom Line

These three papers appear to be about different things:
- Learning theory
- Climate modeling
- NLP efficiency

But they're all answering the same question:

> **Why does this system work despite having "too many" degrees of freedom?**

And the answer is always:

> **Because the extra degrees of freedom are gauge—they don't affect the invariants that control the certificate.**

This is the QA unification thesis:

> **ML/AI theory is invariant-controlled reachability with gauge freedom.**

---

## Citation

```bibtex
@misc{qa_cross_paper_unification_2026,
  title={QA Cross-Paper Unification: Invariant-Controlled Reachability Across ML Domains},
  author={Signal Experiments Research Group},
  year={2026},
  note={Unifies generalization bounds, physics-ML hybrids, and sparse attention}
}
```
