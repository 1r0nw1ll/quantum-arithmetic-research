<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database entry: forward_search. Source attribution to (Kochenderfer, 2022) DM §9.3 + (Dale, 2026) cert [191] tiered reachability. Catalog row." -->

# `forward_search`

## Source reference

- **Source**: (Kochenderfer 2022) *Algorithms for Decision Making*, MIT Press, CC-BY-NC-ND, 700 pp
- **Chapter / section**: §9.3 Forward Search
- **Anchor in QA-MEM**: [`docs/theory/kochenderfer_decision_making_excerpts.md#dm-9-3-forward-search`](../../../docs/theory/kochenderfer_decision_making_excerpts.md)
- **Original code location** (when fetched): `algorithmsbooks/decisionmaking-code` Algorithm 9.2 — NOT YET FETCHED.

## Classical mathematical form

Forward search is depth-`d` recursive expansion. From state `s`:

$$
\text{ForwardSearch}(s, d) = \arg\max_{a \in A} \left[ R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) \, V(s', d-1) \right]
$$

where `V(s, 0) = U(s)` (terminal value function estimate). Worst-case complexity `O((|S|·|A|)^d)`.

## Classical code

Pseudocode (transcribed from DM Algorithm 9.2, attribution: Kochenderfer 2022):

```python
# See classical.py for runnable Python port.
def forward_search(P, s, d, U):
    if d <= 0:
        return (None, U(s))
    best_a, best_u = None, float("-inf")
    def U_next(sp):
        return forward_search(P, sp, d-1, U)[1]
    for a in P.A:
        u = lookahead(P, U_next, s, a)
        if u > best_u:
            best_a, best_u = a, u
    return (best_a, best_u)
```

## QA mapping

- **Status**: `established`
- **QA counterpart**: cert [191] `qa_bateson_learning_levels_cert_v1` tiered reachability — exhaustive forward BFS on `S_9` from each of 81 starting tuples, computing reachable-set cardinality at each generator-tier.
- **Bridge spec row**: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §7.2 row 1 (forward search; status `established`).
- **Theorem NT boundary**: classical forward search is depth-`d` DFS on an MDP with worst-case branching `|S|·|A|`. On the QA orbit graph with `|A| = 1` (single deterministic generator), the worst-case `O(|S|^d)` collapses to `O(|S|·d)` linear time. Cert [191] does the unbounded-horizon variant via BFS until orbit closure (`81 + 1712 + 3456 + 1312 = 6561` (s, s') pairs classified exhaustively at d=4).
- **Evidence link**: cert [191] PASS — exhaustive forward-reachability classification on `S_9`. Used by cert [263] utility (`enumerate_orbit_class_counts`) and cert [264] (`orbit_family_s9` membership) and cert [265] (BFS counterfactual descent).

## Cross-references

- Related entry: [`branch_and_bound`](../branch_and_bound/) — pruned variant of forward search.
- Related entry: [`value_iteration`](../value_iteration/) — infinite-horizon limit of forward search.
- Bridge spec §7.2 row 1 — full mapping detail.
- Cert [191] — empirical evidence; cert [265] uses BFS forward-search directly for counterfactual paths.
