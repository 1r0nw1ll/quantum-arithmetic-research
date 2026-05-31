# QA Koenig Graph Anomaly Benchmark

Hash: `8c8207f113fc4f38706cee643cf24dabba6d04c2bd3f1b384f5b73a9048eed0d`

Generated path-with-branch anomaly sweeps. Positive nodes are true branch nodes; decoy branches and shortcut edges are controls.

Koenig gap and QA gap are algebraically the same `2*C*F` square-gap quantity. The benchmark reports both names to make the derivation explicit.

`koenig_depth_score` and `koenig_rank_score` are row-set branch diagnostics from the supplied QA projection, not independent graph invariants. Treat them as explanation/reranking features, not canonical graph labels.

`spread_score` is the Wildberger rational spread between the two GLOBAL anchor-pair QA projections. Benchmarked but found to be uninformative (AUROC below random) because it measures path-position differences, not local graph geometry.

`local_edge_spread_score` uses LOCAL edge-direction vectors: for node v, each incident edge (v,u) gets direction Δ=(d(u,L)-d(v,L), d(u,R)-d(v,R)) using BFS distances from the first anchor pair. The score is the sum of Wildberger pairwise spreads over all incident edge pairs. Degree is embedded: a degree-k node contributes C(k,2) pairs. The branch ATTACHMENT POINT is the geometric corner (spread=2); straight path/branch-body nodes score 0.

`qa_monotone_dir_score` counts incident edges with Δb*Δe > 0 (same-sign QA direction = branch-type). On the main path both components change with opposite signs; on a branch both increase. Score=2 for branch body, score=0 for path interior, score=1 for branch/decoy attachment and leaf. Degree is naturally encoded: body=2, leaf=1, path=0.

`qa_branch_composite_score` = monotone_dir + local_edge_spread. Combines the body-detection signal (monotone_dir) with the attachment-detection signal (spread). Evaluated against both `label` (body only) and `label_extended` (body + attachment).

`anchor_free_*` scores use eccentricity-derived anchors via double-BFS pseudo-diameter — no domain knowledge required. The pair (u,v) is found by: BFS from an arbitrary node → u = farthest node; BFS from u → v = farthest from u. On simple path-with-branch graphs without shortcuts, these recover P0/P{n-1}; with shortcuts the diameter endpoints may differ.

`[ext]` suffix indicates the metric was evaluated against `label_extended` (branch body + branch attachment node), which captures the full structural anomaly region.

## Summary

| score | cases | AUROC mean | AP mean | top-k hit rate |
|---|---:|---:|---:|---:|
| anchor_free_branch_composite_score | 192 | 0.5070 +/- 0.3891 | 0.3567 +/- 0.2568 | 0.3083 +/- 0.3594 |
| anchor_free_branch_composite_score[ext] | 192 | 0.5677 +/- 0.3627 | 0.4843 +/- 0.3373 | 0.4562 +/- 0.3875 |
| anchor_free_local_edge_spread_score | 192 | 0.4346 +/- 0.0475 | 0.4802 +/- 0.2007 | 0.4044 +/- 0.3493 |
| anchor_free_monotone_dir_score | 192 | 0.5335 +/- 0.4113 | 0.5296 +/- 0.4121 | 0.4414 +/- 0.4561 |
| anchor_free_monotone_dir_score[ext] | 192 | 0.5522 +/- 0.3920 | 0.5352 +/- 0.3875 | 0.4325 +/- 0.4250 |
| degree_score | 192 | 0.4077 +/- 0.0398 | 0.4218 +/- 0.1592 | 0.4211 +/- 0.3259 |
| distance_imbalance_score | 192 | 0.1344 +/- 0.1158 | 0.1073 +/- 0.0338 | 0.0020 +/- 0.0166 |
| distance_product_score | 192 | 0.6344 +/- 0.1218 | 0.3272 +/- 0.2135 | 0.2481 +/- 0.2289 |
| distance_sum_score | 192 | 0.3235 +/- 0.1547 | 0.1652 +/- 0.1222 | 0.0752 +/- 0.1326 |
| koenig_depth_score | 192 | 0.5125 +/- 0.0217 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| koenig_g_score | 192 | 0.4291 +/- 0.0705 | 0.1655 +/- 0.0888 | 0.0811 +/- 0.1184 |
| koenig_gap_score | 192 | 0.6939 +/- 0.1143 | 0.3779 +/- 0.1918 | 0.3025 +/- 0.1907 |
| koenig_h_score | 192 | 0.4738 +/- 0.0757 | 0.1845 +/- 0.1052 | 0.1014 +/- 0.1375 |
| koenig_rank_score | 192 | 0.5589 +/- 0.0818 | 0.7450 +/- 0.1956 | 0.6812 +/- 0.2763 |
| local_edge_spread_score | 192 | 0.3929 +/- 0.0833 | 0.4196 +/- 0.2353 | 0.3396 +/- 0.3692 |
| permuted_koenig_gap_score | 192 | 0.4999 +/- 0.2208 | 0.2324 +/- 0.1540 | 0.1536 +/- 0.1867 |
| qa_branch_composite_score | 192 | 0.6602 +/- 0.2582 | 0.4628 +/- 0.1934 | 0.4211 +/- 0.3259 |
| qa_branch_composite_score[ext] | 192 | 0.7150 +/- 0.2342 | 0.6094 +/- 0.2567 | 0.6044 +/- 0.3123 |
| qa_degree_score | 192 | 0.4485 +/- 0.1248 | 0.1705 +/- 0.0634 | 0.0848 +/- 0.1117 |
| qa_g_score | 192 | 0.4291 +/- 0.0705 | 0.1655 +/- 0.0888 | 0.0811 +/- 0.1184 |
| qa_gap_score | 192 | 0.6939 +/- 0.1143 | 0.3779 +/- 0.1918 | 0.3025 +/- 0.1907 |
| qa_h_score | 192 | 0.4738 +/- 0.0757 | 0.1845 +/- 0.1052 | 0.1014 +/- 0.1375 |
| qa_monotone_dir_score | 192 | 0.7779 +/- 0.1861 | 0.7803 +/- 0.2148 | 0.8262 +/- 0.1629 |
| qa_monotone_dir_score[ext] | 192 | 0.8056 +/- 0.1557 | 0.8532 +/- 0.1419 | 0.8411 +/- 0.1235 |
| spread_score | 192 | 0.4742 +/- 0.1268 | 0.2055 +/- 0.0800 | 0.1635 +/- 0.1394 |

## Shortcut Split (AGS Theorem Tree-Specificity)

AGS cert [288] proves the monotone-direction invariant holds for trees. Shortcut edges create cycles; the theorem does not apply, and scores degrade. The split below validates this boundary.

| score | no-shortcut (96 cases) | shortcut (96 cases) |
|---|---:|---:|
| `qa_monotone_dir_score` | 0.9605 | 0.5952 |
| `qa_monotone_dir_score[ext]` | 0.9558 | 0.6554 |
| `anchor_free_monotone_dir_score` | 0.9062 | 0.1608 |
| `anchor_free_monotone_dir_score[ext]` | 0.9057 | 0.1986 |
| `koenig_gap_score` | 0.7701 | 0.6177 |

Anchor-free on no-shortcut cases (0.9062) is within 5.4% of anchored (0.9605). On shortcut cases, both degrade — confirming that the score failure is a graph-topology issue (cycles), not an anchor selection issue.

## Verdict Inputs

- `koenig_gap_score`: AUROC `0.6939`, AP `0.3779`, top-k `0.3025`
- `qa_gap_score`: AUROC `0.6939`, AP `0.3779`, top-k `0.3025`
- `qa_monotone_dir_score`: AUROC `0.7779`, AP `0.7803`, top-k `0.8262`
- `qa_monotone_dir_score[ext]`: AUROC `0.8056`, AP `0.8532`, top-k `0.8411`
- `qa_branch_composite_score`: AUROC `0.6602`, AP `0.4628`, top-k `0.4211`
- `qa_branch_composite_score[ext]`: AUROC `0.7150`, AP `0.6094`, top-k `0.6044`
- `anchor_free_monotone_dir_score`: AUROC `0.5335`, AP `0.5296`, top-k `0.4414`
- `anchor_free_monotone_dir_score[ext]`: AUROC `0.5522`, AP `0.5352`, top-k `0.4325`
- `anchor_free_branch_composite_score`: AUROC `0.5070`, AP `0.3567`, top-k `0.3083`
- `anchor_free_branch_composite_score[ext]`: AUROC `0.5677`, AP `0.4843`, top-k `0.4562`
- `local_edge_spread_score`: AUROC `0.3929`, AP `0.4196`, top-k `0.3396`
- `spread_score`: AUROC `0.4742`, AP `0.2055`, top-k `0.1635`
- `qa_degree_score`: AUROC `0.4485`, AP `0.1705`, top-k `0.0848`
- `permuted_koenig_gap_score`: AUROC `0.4999`, AP `0.2324`, top-k `0.1536`
- `distance_product_score`: AUROC `0.6344`, AP `0.3272`, top-k `0.2481`
- `degree_score`: AUROC `0.4077`, AP `0.4218`, top-k `0.4211`
