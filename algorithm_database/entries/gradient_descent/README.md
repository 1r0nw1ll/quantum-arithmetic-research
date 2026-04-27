<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database entry: gradient_descent. Source attribution to (Kochenderfer, 2026) Opt §5.1 + (Dale, 2026) bridge §8.2. Catalog row." -->

# `gradient_descent`

## Source reference

- **Source**: (Kochenderfer 2026) *Algorithms for Optimization*, 2nd ed., MIT Press, CC-BY-NC-ND, 631 pp
- **Chapter / section**: §5.1 Gradient Descent
- **Anchor in QA-MEM**: [`docs/theory/kochenderfer_optimization_excerpts.md#opt-5-1-gradient-descent-steepest-direction`](../../../docs/theory/kochenderfer_optimization_excerpts.md)
- **Original code location** (when fetched): `algorithmsbooks/algforopt-notebooks/first-order.ipynb` Algorithm 5.1 — NOT YET FETCHED.

## Classical mathematical form

Steepest-descent direction:

$$
d^{(k)} = -\frac{g^{(k)}}{\|g^{(k)}\|}, \qquad x^{(k+1)} = x^{(k)} + \alpha^{(k)} d^{(k)}
$$

where `g^(k) = ∇f(x^(k))`. First-order Taylor justification: `f(x + αd) ≈ f(x) + α d^T g`; minimizer of the linear approximation under `‖d‖ = 1` is `d = -g/‖g‖`.

## Classical code

Pseudocode (transcribed from Opt Algorithm 5.1, attribution: Kochenderfer 2026):

```python
# See classical.py for runnable Python port.
def gradient_descent(f, grad_f, x, alpha, k_max):
    for _ in range(k_max):
        x = x - alpha * grad_f(x)
    return x
```

## QA mapping

- **Status**: `rejected (continuous-domain)` / `candidate (vocabulary alignment with cert [191] tier-1 generator selection)`
- **QA counterpart**: cert [191] `qa_bateson_learning_levels_cert_v1` tier-1 generator selection — the QA-discrete analog of "pick the descent direction that most reduces the objective". On the QA orbit graph with `1{s ∈ ψ}` as the objective, the tier-1 (orbit-preserving) generator that minimizes orbit-class distance to ψ IS the discrete steepest-descent direction.
- **Bridge spec row**: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §8.2 row 3 (status `candidate`).
- **Theorem NT boundary**: gradient computation requires the objective be differentiable (continuous-domain); classical `gradient_descent` is firewall-rejected as causal QA dynamics. The QA-discrete analog at the orbit-class boundary IS canonically defined (cert [191] tier-1 generator selection); the mapping is vocabulary-only — no new code.
- **Evidence link**: cert [191] PASS for the discrete analog; classical gradient_descent has no QA-side runnable equivalent and shouldn't have one (different ontology).

## Cross-references

- Related entry: [`cyclic_coordinate_search`](../cyclic_coordinate_search/) — derivative-free cousin with the same QA-tier-hierarchy mapping.
- Bridge spec §8.2 row 3 — full mapping detail.
- Future cert candidate `qa_hgd_descent_baseline_cert_v1` — distinct from this entry; concerns Hypergradient Descent (Baydin 2018), which Kochenderfer §5.9 names but is mathematically different from any QA HGD; precondition: pin the QA-side HGD canonical reference first (bridge §8.2).
