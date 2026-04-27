<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database entry: branch_and_bound. Source attribution to (Kochenderfer, 2022) DM §9.4 + (Kochenderfer, 2026) Opt §22.4 + (Dale, 2026) cert [191] enumeration. Catalog row." -->

# `branch_and_bound`

## Source reference

- **Source (DM context)**: (Kochenderfer 2022) *Algorithms for Decision Making*, MIT Press, §9.4 (online planning, MDP context)
- **Source (Opt context)**: (Kochenderfer 2026) *Algorithms for Optimization*, 2nd ed., MIT Press, §22.4 (integer programming context)
- **Anchors in QA-MEM**:
  - [`docs/theory/kochenderfer_decision_making_excerpts.md#dm-9-4-branch-and-bound`](../../../docs/theory/kochenderfer_decision_making_excerpts.md) (forward-search pruning)
  - [`docs/theory/kochenderfer_optimization_excerpts.md#opt-22-4-branch-and-bound-integer`](../../../docs/theory/kochenderfer_optimization_excerpts.md) (LP-relaxation + integer branch)
- **Original code location (DM context)**: `algorithmsbooks/decisionmaking-code` `decision_making_code.jl` L952-973 (section `online-approximations`); FETCHED 2026-04-27 v1.2. See [`sources/decisionmaking_code_inventory.md`](../../sources/decisionmaking_code_inventory.md). Julia signature: `function branch_and_bound(𝒫, s, d, Ulo, Qhi)` — matches book Algorithm 9.3.
- **Original code location (Opt context)**: `algorithmsbooks/optimization` Algorithm 22.5 (LP-relaxation + integer branch variant) — NOT YET FETCHED. The DM-context Julia code already covers our v1 entry; Opt-context would only be needed for the LP-relaxation variant in v2+.

## Classical mathematical form

**DM context (forward-search pruning)**: prune subtrees via `Q(s, a)_hi < Q(s, a*)_lo`. Same worst-case as forward search; better pruning depending on lower-bound oracle.

**Opt context (integer programming)**: LP relaxation produces fractional solution `x*`; branch on the most fractional component `x_i`:

$$
x_i \le \lfloor x^*_{i,c} \rfloor \quad \text{or} \quad x_i \ge \lceil x^*_{i,c} \rceil
$$

Priority queue ordered by LP-objective lower bound; prune subtrees whose lower bound exceeds best integral solution found.

## Classical code

Pseudocode (transcribed from DM Algorithm 9.3 + Opt Algorithm 22.5, attribution: Kochenderfer 2022/2026):

```python
# See classical.py for runnable Python port (DM forward-search variant).
def branch_and_bound(P, s, d, U_lo, Q_hi):
    if d <= 0:
        return (None, U_lo(s))
    best_a, best_u = None, float("-inf")
    def U_next(sp):
        return branch_and_bound(P, sp, d-1, U_lo, Q_hi)[1]
    for a in sorted(P.A, key=lambda a: -Q_hi(s, a)):
        if Q_hi(s, a) < best_u:
            return (best_a, best_u)  # safe to prune
        u = lookahead(P, U_next, s, a)
        if u > best_u:
            best_a, best_u = a, u
    return (best_a, best_u)
```

## QA mapping

- **Status**: `rejected` (small `|S|`) / `candidate` (large `|S|`)
- **QA counterpart**: none direct in v1. Cert [263] utility's enumeration over `S_9` (|S|=81) dominates B&B by exhaustive search; B&B's pruning advantage requires `|S|^d` to be intractable.
- **Bridge spec row**: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §7.2 row 2 (DM context) + §8.3 row 1 (Opt context).
- **Theorem NT boundary**: classical B&B uses LP relaxation (continuous-domain) on the Opt side; the relaxed LP step crosses the firewall as causal computation. On the DM side B&B is structurally fine for QA (discrete actions + discrete states + integer rewards), but enumeration dominates on small orbit graphs. **For modulus where `|S|^h` is intractable (e.g., mod-72 or higher), B&B becomes useful**; that's the future cert candidate `qa_generator_search_vs_branch_and_bound_cert_v1` listed in bridge §8.3 standing rule.
- **Evidence link**: none in v1 (status is rejected for `S_9` scale; candidate for larger scale; no implementation exists).

## Cross-references

- Related entry: [`forward_search`](../forward_search/) — base algorithm; B&B prunes its tree.
- Bridge spec §7.2 row 2 + §8.3 row 1 — full mapping detail.
- Future cert candidate `qa_generator_search_vs_branch_and_bound_cert_v1` — bridge Standing Rule #2.
