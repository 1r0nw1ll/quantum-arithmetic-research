<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database entry: iterative_policy_evaluation. Source attribution to (Kochenderfer, 2022) DM §7.2 + (Dale, 2026) cert [191]; full citations in QA-MEM excerpts file. Catalog row pointing at evidence, not a research claim doc." -->

# `iterative_policy_evaluation`

## Source reference

- **Source**: (Kochenderfer 2022) *Algorithms for Decision Making*, MIT Press, CC-BY-NC-ND, 700 pp
- **Chapter / section**: §7.2 Policy Evaluation
- **Anchor in QA-MEM**: [`docs/theory/kochenderfer_decision_making_excerpts.md#dm-7-2-policy-evaluation-lookahead-equation`](../../../docs/theory/kochenderfer_decision_making_excerpts.md)
- **Original code location**: `algorithmsbooks/decisionmaking-code` `decision_making_code.jl` L564-571 (section `exact-solutions 3`); FETCHED 2026-04-27 v1.2. See [`sources/decisionmaking_code_inventory.md`](../../sources/decisionmaking_code_inventory.md). Julia signature: `function iterative_policy_evaluation(𝒫::MDP, π, k_max)` — matches book Algorithm 7.3.

## Classical mathematical form

Iterative DP via Bellman lookahead:

$$
U^\pi_{k+1}(s) = R(s, \pi(s)) + \gamma \sum_{s'} T(s' \mid s, \pi(s)) \, U^\pi_k(s')
$$

Convergence guaranteed because the update is a contraction mapping (DM §7.2). The Bellman expectation equation `U^π = R^π + γ T^π U^π` admits a direct linear-algebra solution `U^π = (I - γ T^π)^{-1} R^π` in `O(|S|^3)` (DM Algorithm 7.4 `policy_evaluation`).

## Classical code

Pseudocode (transcribed from DM Algorithm 7.3, attribution: Kochenderfer 2022):

```python
# See classical.py for runnable Python port.
def iterative_policy_evaluation(P, pi, k_max):
    S, T, R, gamma = P.S, P.T, P.R, P.gamma
    U = {s: 0.0 for s in S}
    for _ in range(k_max):
        U = {s: lookahead(P, U, s, pi(s)) for s in S}
    return U
```

## QA mapping

- **Status**: `candidate`
- **QA counterpart**: cert [191] `qa_bateson_learning_levels_cert_v1` tier-classification + cert [263] `tools/qa_kg/orbit_failure_enumeration.py` utility.
- **Bridge spec row**: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §7.1 row 2 (Bellman expectation equation; status `candidate`).
- **Theorem NT boundary**: classical algorithm is continuous-domain (real-valued `U^π`, possibly stochastic `T`); the QA-discrete analog runs on the orbit graph with `T = qa_step` (deterministic), where the Bellman expectation equation collapses to the single-successor recursion `U^π(s) = R(s, π(s)) + γ U^π(qa_step(s, π(s)))` — no fixed-point iteration needed, just orbit traversal in `O(orbit_period(s))` integer steps. Cert [191] tier-classification implements this on `S_9` exhaustively. The next step is vocabulary alignment in [191]'s family doc, not new code.
- **Evidence link**: cert [191] PASS in meta-validator + cert [263] utility's `enumerate_orbit_class_counts(9) → {1, 8, 72, 81}` reproduces the canonical mod-9 orbit cardinalities (the QA-discrete analog of policy-evaluation fixed points).

## Cross-references

- Related entry: [`value_iteration`](../value_iteration/) — same Bellman backup pattern with `max_a` instead of fixed policy.
- Related entry: [`forward_search`](../forward_search/) — forward expansion is the unrolled-by-depth version of policy evaluation.
- Bridge spec §7.1 row 2 — full mapping detail.
- Cert [191] — empirical evidence on `S_9`.
