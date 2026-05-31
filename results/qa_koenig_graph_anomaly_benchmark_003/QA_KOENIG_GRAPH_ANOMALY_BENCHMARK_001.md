# QA Koenig Graph Anomaly Benchmark 001

Hash: `0121b96c78a485169eef343cca1d3924118ff644f50f3c13560bb32782172ee7`

Generated path-with-branch anomaly sweeps. Positive nodes are true branch nodes; decoy branches and shortcut edges are controls.

Koenig gap and QA gap are algebraically the same `2*C*F` square-gap quantity. The benchmark reports both names to make the derivation explicit.

`koenig_depth_score` and `koenig_rank_score` are row-set branch diagnostics from the supplied QA projection, not independent graph invariants. Treat them as explanation/reranking features, not canonical graph labels.

`spread_score` is the Wildberger rational spread between the two GLOBAL anchor-pair QA projections. Benchmarked but found to be uninformative (AUROC below random) because it measures path-position differences, not local graph geometry.

`local_edge_spread_score` uses LOCAL edge-direction vectors: for node v, each incident edge (v,u) gets direction Δ=(d(u,L)-d(v,L), d(u,R)-d(v,R)) using BFS distances from the first anchor pair. The score is the sum of Wildberger pairwise spreads over all incident edge pairs. Degree is embedded: a degree-k node contributes C(k,2) pairs. The branch ATTACHMENT POINT is the geometric corner (spread=2); straight path/branch-body nodes score 0.

`qa_monotone_dir_score` counts incident edges with Δb*Δe > 0 (same-sign QA direction = branch-type). On the main path both components change with opposite signs; on a branch both increase. Score=2 for branch body, score=0 for path interior, score=1 for branch/decoy attachment and leaf. Degree is naturally encoded: body=2, leaf=1, path=0.

## Summary

| score | cases | AUROC mean | AP mean | top-k hit rate |
|---|---:|---:|---:|---:|
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
| qa_degree_score | 192 | 0.4485 +/- 0.1248 | 0.1705 +/- 0.0634 | 0.0848 +/- 0.1117 |
| qa_g_score | 192 | 0.4291 +/- 0.0705 | 0.1655 +/- 0.0888 | 0.0811 +/- 0.1184 |
| qa_gap_score | 192 | 0.6939 +/- 0.1143 | 0.3779 +/- 0.1918 | 0.3025 +/- 0.1907 |
| qa_h_score | 192 | 0.4738 +/- 0.0757 | 0.1845 +/- 0.1052 | 0.1014 +/- 0.1375 |
| qa_monotone_dir_score | 192 | 0.7779 +/- 0.1861 | 0.7803 +/- 0.2148 | 0.8262 +/- 0.1629 |
| spread_score | 192 | 0.4742 +/- 0.1268 | 0.2055 +/- 0.0800 | 0.1635 +/- 0.1394 |

## Verdict Inputs

- `koenig_gap_score`: AUROC `0.6939`, AP `0.3779`, top-k `0.3025`
- `qa_gap_score`: AUROC `0.6939`, AP `0.3779`, top-k `0.3025`
- `qa_monotone_dir_score`: AUROC `0.7779`, AP `0.7803`, top-k `0.8262`
- `local_edge_spread_score`: AUROC `0.3929`, AP `0.4196`, top-k `0.3396`
- `spread_score`: AUROC `0.4742`, AP `0.2055`, top-k `0.1635`
- `qa_degree_score`: AUROC `0.4485`, AP `0.1705`, top-k `0.0848`
- `permuted_koenig_gap_score`: AUROC `0.4999`, AP `0.2324`, top-k `0.1536`
- `distance_product_score`: AUROC `0.6344`, AP `0.3272`, top-k `0.2481`
- `degree_score`: AUROC `0.4077`, AP `0.4218`, top-k `0.4211`
