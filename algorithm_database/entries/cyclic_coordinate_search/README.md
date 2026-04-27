<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database entry: cyclic_coordinate_search. Source attribution to (Kochenderfer, 2026) Opt §7.1 + (Dale, 2026) bridge §8.3 + cert [191]. Catalog row." -->

# `cyclic_coordinate_search`

## Source reference

- **Source**: (Kochenderfer 2026) *Algorithms for Optimization*, 2nd ed., MIT Press, CC-BY-NC-ND, 631 pp
- **Chapter / section**: §7.1 Cyclic Coordinate Search
- **Anchor in QA-MEM**: [`docs/theory/kochenderfer_optimization_excerpts.md#opt-7-1-cyclic-coordinate-search`](../../../docs/theory/kochenderfer_optimization_excerpts.md)
- **Original code location** (when fetched): `algorithmsbooks/algforopt-notebooks/discrete.ipynb` Algorithm 7.2 + 7.3 — NOT YET FETCHED.

## Classical mathematical form

Cyclic coordinate search alternates line searches along the `n` basis vectors `e^(1), …, e^(n)`:

$$
x^{(2)}_1 = \arg\min_{x_1} f(x_1, x^{(1)}_2, \dots, x^{(1)}_n), \quad x^{(3)}_2 = \arg\min_{x_2} f(x^{(2)}_1, x_2, x^{(2)}_3, \dots), \quad \dots
$$

Optionally augmented with an "acceleration step": after a full cycle through `e^(1), …, e^(n)`, do a line search along the net direction `x^(n+1) - x^(1)`.

## Classical code

Pseudocode (transcribed from Opt Algorithm 7.2 + 7.3, attribution: Kochenderfer 2026):

```python
# See classical.py for runnable Python port.
def cyclic_coordinate_descent(f, x, eps):
    n = len(x)
    delta = float("inf")
    while abs(delta) > eps:
        x_orig = list(x)
        for i in range(n):
            d = basis(i, n)        # i-th basis vector
            x = line_search(f, x, d)
        delta = norm(x - x_orig)
    return x
```

## QA mapping

- **Status**: `candidate` (vocabulary alignment with cert [191] tier hierarchy)
- **QA counterpart**: cert [191] `qa_bateson_learning_levels_cert_v1` tier-classification IS the QA-native cyclic generator search analog. For each `(s, s')` pair: try L_1 generators (orbit-preserving) first; if no L_1 path, try L_2a (orbit-changing, family-preserving); then L_2b (family-changing); then L_3 (modulus-changing). Each tier corresponds to a "coordinate" in the operator-class hierarchy.
- **Bridge spec row**: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §8.3 row 2 (status `candidate`).
- **Theorem NT boundary**: classical CCS uses continuous line searches along basis vectors (continuous-domain); firewall-rejected as causal QA dynamics. The QA-discrete cyclic-generator-search analog operates on the integer tier hierarchy from cert [191]; same algorithmic shape (cycle through basis directions, take cheapest move at each step), different ontology (continuous coordinates vs discrete generator tiers).
- **Evidence link**: cert [191] PASS — exhaustive tier-classification on `S_9` is the QA-native cyclic-coordinate-search execution.

## Cross-references

- Related entry: [`gradient_descent`](../gradient_descent/) — gradient-based cousin; same QA-tier-hierarchy mapping but at a finer level.
- Bridge spec §8.3 row 2 — full mapping detail.
- Cert [191] — empirical evidence on `S_9`.
- Cert [265] `qa_counterfactual_descent_cert_v1` uses the same generator hierarchy (`qa_step` = L_1, `scalar_mult_2` = L_2a, `scalar_mult_3` = L_2b) for BFS counterfactual-path search; the algorithmic shape mirrors CCS but on the orbit graph.
