<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database entry: simulated_annealing. Source attribution to (Kochenderfer, 2026) Opt §8.4 + (Dale, 2026) bridge §8.4. Catalog row." -->

# `simulated_annealing`

## Source reference

- **Source**: (Kochenderfer 2026) *Algorithms for Optimization*, 2nd ed., MIT Press, CC-BY-NC-ND, 631 pp
- **Chapter / section**: §8.4 Simulated Annealing
- **Anchor in QA-MEM**: [`docs/theory/kochenderfer_optimization_excerpts.md#opt-8-4-simulated-annealing`](../../../docs/theory/kochenderfer_optimization_excerpts.md)
- **Original code location** (when fetched): `algorithmsbooks/algforopt-notebooks/stochastic.ipynb` Algorithm 8.5 — NOT YET FETCHED.

## Classical mathematical form

Metropolis acceptance criterion at temperature `t`:

$$
P(\text{accept } x') = \begin{cases} 1 & \text{if } \Delta y \le 0 \\ e^{-\Delta y / t} & \text{if } \Delta y > 0 \end{cases}
$$

where `Δy = f(x') - f(x)`. Exponential annealing schedule: `t^(k+1) = γ t^(k)` for some `γ ∈ (0, 1)`.

## Classical code

Pseudocode (transcribed from Opt Algorithm 8.5, attribution: Kochenderfer 2026):

```python
# See classical.py for runnable Python port.
def simulated_annealing(f, x, T, t_schedule, k_max):
    y = f(x)
    best = (x, y)
    for k in range(1, k_max+1):
        x_prime = x + sample(T)
        y_prime = f(x_prime)
        dy = y_prime - y
        if dy <= 0 or random() < math.exp(-dy / t_schedule(k)):
            x, y = x_prime, y_prime
        if y_prime < best[1]:
            best = (x_prime, y_prime)
    return best[0]
```

## QA mapping

- **Status**: `rejected`
- **QA counterpart**: none. Continuous + stochastic Metropolis criterion crosses the Theorem NT firewall; QA enumeration dominates on QA-discrete side.
- **Bridge spec row**: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §8.4 row 1 (status `rejected`).
- **Theorem NT boundary**: stochastic acceptance + continuous transition distribution = firewall-rejected as causal QA dynamics. SA is an off-QA baseline that QA enumeration dominates by construction on small orbit graphs (`|S| ≤ 576` empirically). For genuinely intractable scale, SA could become useful — that's the future cert candidate `qa_generator_search_vs_simulated_annealing_cert_v1` flagged in bridge §8.4.
- **Evidence link**: none. Status is rejected; no QA-side implementation exists or is planned in v1.

## Cross-references

- Bridge spec §8.4 row 1 — full mapping detail.
- Future cert candidate `qa_generator_search_vs_simulated_annealing_cert_v1` — bridge Standing Rule #2.
- Cert [263] `qa_failure_density_enumeration_cert_v1` — already demonstrates QA enumeration dominating sampling-based estimation; SA-vs-enumeration would be a parallel comparison.
