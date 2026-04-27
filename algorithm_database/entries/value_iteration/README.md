<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database entry: value_iteration. Source attribution to (Kochenderfer, 2022) DM §7.5 + (Dale, 2026) cert [191] tier-classification. Catalog row." -->

# `value_iteration`

## Source reference

- **Source**: (Kochenderfer 2022) *Algorithms for Decision Making*, MIT Press, CC-BY-NC-ND, 700 pp
- **Chapter / section**: §7.5 Value Iteration
- **Anchor in QA-MEM**: [`docs/theory/kochenderfer_decision_making_excerpts.md#dm-7-5-value-iteration-bellman-backup`](../../../docs/theory/kochenderfer_decision_making_excerpts.md)
- **Original code location**: `algorithmsbooks/decisionmaking-code` `decision_making_code.jl` L617-635 (sections `exact-solutions 7` + `exact-solutions 8`); FETCHED 2026-04-27 v1.2. See [`sources/decisionmaking_code_inventory.md`](../../sources/decisionmaking_code_inventory.md). Julia: `function backup(𝒫::MDP, U, s)` + `struct ValueIteration` + `solve(M::ValueIteration, 𝒫::MDP)` — matches book Algorithms 7.7-7.8.

## Classical mathematical form

Bellman backup converges to the optimal value function `U*`:

$$
U_{k+1}(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) \, U_k(s') \right]
$$

At convergence, the Bellman optimality equation holds:

$$
U^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) \, U^*(s') \right]
$$

Termination criterion (DM §7.5): Bellman residual `||U_{k+1} - U_k||_∞ < δ` gives `||U_k - U*||_∞ < δγ/(1-γ)`.

## Classical code

Pseudocode (transcribed from DM Algorithm 7.8, attribution: Kochenderfer 2022):

```python
# See classical.py for runnable Python port.
def value_iteration(P, k_max):
    U = {s: 0.0 for s in P.S}
    for _ in range(k_max):
        U = {s: max(lookahead(P, U, s, a) for a in P.A) for s in P.S}
    return U
```

## QA mapping

- **Status**: `established`
- **QA counterpart**: cert [191] `qa_bateson_learning_levels_cert_v1` tier-classification with `EXPECTED_TIER_COUNTS_S9 = {0: 81, 1: 1712, 2a: 3456, 2b: 1312}` exhaustively verified.
- **Bridge spec row**: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §7.1 row 3 (Bellman backup; status `established`).
- **Theorem NT boundary**: classical Bellman backup is continuous-domain on a general MDP. On the QA orbit graph with deterministic single-action transitions (`A = {qa_step}`), the `max_a` collapses to the single-successor case, and the backup becomes orbit traversal: `U(s) = R(s) + γ U(qa_step(s))`. Cert [191] tier-classification IS this backup with `R(s) = 1{s ∈ ψ}` (orbit-class membership indicator). The QA-discrete reduction makes the backup integer-only.
- **Evidence link**: cert [191] PASS — `81 + 1712 + 3456 + 1312 = 6561` ordered pairs classified exhaustively on `S_9`.

## Cross-references

- Related entry: [`iterative_policy_evaluation`](../iterative_policy_evaluation/) — fixed-policy variant of the same Bellman backup.
- Related entry: [`forward_search`](../forward_search/) — finite-horizon unroll of value iteration.
- Bridge spec §7.1 row 3 — full mapping detail.
- Cert [191] — empirical evidence on `S_9`.
