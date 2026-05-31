# [288] QA Anchor Geodesic Separation Cert

**Family ID**: 288
**Slug**: `qa_anchor_geodesic_separation_cert_v1`
**Status**: Active
**Registered**: 2026-05-31

## Claim (narrow, falsifiable)

In any finite unweighted tree T with distinct anchor nodes L and R, for every edge (v, u) define:

```
Δb = d(u, L) − d(v, L)
Δe = d(u, R) − d(v, R)
```

Then:

1. **AGS_1**: Δb, Δe ∈ {−1, +1} for every edge (BFS distances change by exactly ±1 in a tree)
2. **AGS_2**: Δb·Δe = −1 iff edge (v,u) lies on the unique L-R path P_LR
3. **AGS_3**: Δb·Δe = +1 iff edge (v,u) does NOT lie on P_LR

**Node-level corollary:**

```
qa_monotone_dir_score(v) = |{u ∈ N(v) : Δb·Δe > 0}|
                         = count of incident edges NOT on P_LR
```

- **AGS_4**: score(v) = number of off-path incident edges
- **AGS_5**: score(v) = 0 iff all incident edges are on P_LR (v is on P_LR with no branches)

## Proof Sketch

**AGS_1** follows from the tree property: any two adjacent nodes differ by exactly 1 in BFS distance from any fixed source.

**AGS_2/3**: Removing an on-path edge separates L from R into two components, forcing one endpoint closer to L and the other closer to R — opposite signs, product −1. Removing an off-path edge does NOT separate L from R; both endpoints share the same "foot" B on P_LR, so moving away from B increases both d(v,L) and d(v,R) simultaneously — same signs, product +1.

## Why Degree Is Naturally Embedded

The node-level score counts off-path incident edges. For the standard path-with-branch construction:

| Node type | score | Reason |
|---|---|---|
| Path interior (non-junction) | 0 | Both incident edges are on P_LR |
| Path endpoint | 0 | Single incident edge is on P_LR |
| Branch attachment (junction, on P_LR) | 1 | One on-path + one off-path edge |
| Branch body interior | 2 | Both incident edges are off P_LR |
| Branch leaf | 1 | Single incident edge is off P_LR |

No explicit degree term is needed — the count already distinguishes degree-2 body (score=2) from degree-1 leaf (score=1).

## Connection to Benchmark

This theorem explains why `qa_monotone_dir_score` achieves:
- AUROC = 0.7779 (vs koenig_gap 0.6939)
- AP = 0.7803 (vs koenig_gap 0.3779)
- top-k = 0.8262 (vs koenig_gap 0.3025)

on the 192-case path-with-branch anomaly benchmark (`experiments/registry.json` id `qa_koenig_graph_anomaly_benchmark_2026-05-31c`). Branch body nodes (label=1) have score=2; path interior (label=0) has score=0. The only ambiguous score=1 group mixes branch attachments and branch leaves.

## Boundary Condition: Cycles

The theorem is **tree-specific**. For graphs with cycles, AGS_1 can fail: in a 5-cycle with anchors two hops apart, nodes equidistant from an anchor produce Δb=0 (not in {−1,+1}). AGS_3 also fails in cycle configurations where equal-length alternative paths exist. The `fail_cycle_breaks_ags1.json` fixture demonstrates this with a 5-cycle.

## Primary Sources

- Wildberger, N. J. (2005). *Divine Proportions: Rational Trigonometry to Universal Geometry*. Wild Egg Books. ISBN 978-0-9757492-0-8. QA distance coordinate framework: b = d(v,L)+1, e = d(v,R)+1.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press. ISBN 978-0-262-03384-8. Chapter 22: BFS, shortest-path distances, ±1 adjacency property.

## Mechanism Chain

- [279] QA Orbit Access Theorem — provides the QA coordinate framework
- Benchmark `qa_koenig_graph_anomaly_benchmark_2026-05-31c` — empirical validation

## Checks

| ID     | Description                                                              |
|--------|--------------------------------------------------------------------------|
| AGS_1  | Δb, Δe ∈ {−1, +1} for every edge (tree ±1 BFS property)               |
| AGS_2  | On-path edges: Δb·Δe = −1                                               |
| AGS_3  | Off-path edges: Δb·Δe = +1                                              |
| AGS_4  | score(v) = count of incident edges NOT on P_LR                          |
| AGS_5  | score(v) = 0 iff all incident edges are on P_LR                         |
| SRC    | Primary-source exempt marker present                                     |
| F      | Fixture on_path/off_path node sets match BFS computation                 |

**Fixtures**: 6 PASS + 4 FAIL
**Self-test**: path (n=3,5,8,12,24), star (n=3,5,8), binary tree (depth=2,3,4), caterpillar (various)
