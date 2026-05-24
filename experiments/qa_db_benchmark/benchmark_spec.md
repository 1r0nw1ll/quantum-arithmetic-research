# Benchmark Specification

## Domain

- b, e ∈ [1, N], default N=250
- All derived fields computed from (b, e) by canonical QA law (no float)
- Generators: sigma=(b,e+1), mu=(e,b), lambda2=(2b,2e), nu=(b/2,e/2) when even
- All targets checked to stay inside domain [1,N]×[1,N]

## Query Structure

Each query specifies:
| Field | Type | Description |
|-------|------|-------------|
| seeds | list[(b,e)] | starting packets (1-3) |
| k | int {2,3,4} | BFS depth |
| orbit_mod | int {9,24} | orbit modulus |
| orbit_val | int | required (b+e) % orbit_mod |
| i_gap_max | int | maximum I = |C−F| |
| area_max | int | maximum B·E |
| shape_sig | (int,int,int) or None | (C%9,F%9,G%9) |
| parity | str or None | 'odd'/'even'/None |
| require_primitive | bool | gcd(b,e)==1 |
| axis_check | bool | enforce axis_split==2D |
| pyth_check | bool | enforce C²+F²==G² |
| broad_expand | bool | k≥3 and tight buckets |

## Correctness Contract

- table and graph backends MUST return identical result sets.
- qa backend is path_constrained: results ⊆ table results (tighter semantics).
- Path-constrained divergence is reported separately, not as a mismatch.

## Metrics Per Query

| Metric | Unit | Description |
|--------|------|-------------|
| latency_ns | ns | wall-clock query time |
| candidate_count_before | int | after index pre-filter |
| candidate_count_after | int | after full predicate filter |
| expansion_count | int | edge traversals in BFS |
| results_count | int | final result set size |
| collapse_ratio | float | after/before |
| bytes_estimate | int | approximate storage bytes |

## Aggregate Metrics

p50, p95, p99, mean latency; mean expansion_count, collapse_ratio, bytes_estimate;
correctness agreement/mismatch counts; path_constrained count.

## Hypotheses

H1: mean_collapse_ratio(qa) < mean_collapse_ratio(table)
H2: mean_expansion_count(qa) < mean_expansion_count(table)
H3: bytes_estimate(qa) < bytes_estimate(table)
H4: QA win-rate on non-broad queries ≠ 100%
