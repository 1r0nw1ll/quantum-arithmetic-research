# QA Graph Mapping Guide — Domain-Natural (b,e) Assignments

**Authority**: MAP-1 linter rule (tools/qa_axiom_linter.py)
**Hard rule**: Never use generic `b=degree, e=core` as primary method without domain justification.

## Principle

The QA contribution to graph analysis is the **nonlinear algebraic transform** of two integer features (b, e) into 102 invariants (84 qa_invariants + 18 Diophantine). The transform is fixed — the MAPPING chooses WHAT gets transformed.

**The mapping IS the method.** Same 102 features, different (b,e) → order-of-magnitude difference in lift.

## Proven-Best Mappings by Graph Type

| Graph Type | b | e | Δ ARI | Benchmark |
|---|---|---|---|---|
| **Dense-block** (football, caveman) | degree | core_number | +0.056 / +0.418 | unified_graph_bench.py |
| **Hub-dominated** (florentine, barbell) | dist_to_hub_0 | dist_to_hub_1 | +0.193 | hub_distance_descriptor_benchmark.py |
| **Overlapping** | degree | avg_neighbor_degree | +0.041 | overlapping_largescale_spatial_bench.py |
| **Hierarchical** | core_number | clustering_coeff × 10 | +0.126 | hierarchical_bench.py |
| **Scale-free** (multi-community) | log2(degree) + 1 | core_number | +0.332 | hierarchical_bench.py |
| **Molecular** (PROTEINS, ENZYMES) | atom_degree | atom_type (1-indexed) | +3.9pp / +5.3pp | molecular_bench.py |
| **Brain connectivity** (EEG) | channel_degree | core_number | d=0.40 | eeg_brain_connectivity_graph.py |
| **Signed** (alliances/enmities) | positive_degree | negative_degree | matches SOTA | signed_graph_bench.py |
| **Spatial** (geographic) | x_coordinate | y_coordinate | matches (ceiling) | overlapping_largescale_spatial_bench.py |
| **Temporal signals** | b_t (current state) | e_t = generator [209] | per domain | eeg_209_full_stack.py |

## Selection Heuristic

1. **What generates the communities?** The (b, e) mapping should encode the TWO structural properties that CAUSE community membership.

2. **Domain-specific always beats generic.** If the graph has edge signs, use signed degree. If it has coordinates, use coordinates. If it has types, use types.

3. **Core number captures hierarchy.** For hierarchical or nested communities, `b=core_number` (shell depth in k-core decomposition) is consistently better than `b=degree`.

4. **Average neighbor degree captures density gradients.** For overlapping or fuzzy communities, `e=avg_neighbor_degree` captures whether a node sits in a dense or sparse region.

5. **Log-degree for power-law.** Scale-free networks have extreme degree variation; `b=log2(degree)+1` compresses the tail into a workable integer range.

6. **The signed Laplacian IS QA.** For signed graphs, the Laplacian diagonal `d_i = pos_deg + neg_deg = b_i + e_i` is the QA derived coordinate A2. Don't add QA features on top — recognize the Laplacian AS the QA method.

## Anti-Patterns (enforced by MAP-1)

- ❌ `b = degree, e = core` on signed graphs → ignores edge signs
- ❌ `b = degree, e = core` on spatial graphs → ignores coordinates
- ❌ `b = degree, e = core` on molecular graphs → ignores atom types
- ❌ `b = degree, e = core` on hierarchical graphs → misses shell depth
- ❌ Declaring "QA doesn't help" when the mapping was wrong
- ❌ Adding QA features ON TOP of a SOTA method instead of recognizing the SOTA AS QA

## Feature Dimensionality

- **degree_only** (4 features: mean+std of b, e): Best for small graphs (n < 50) where higher dimensions overfit.
- **qa21** (42 features): Middle ground, works for n > 50.
- **full/qa102** (204 features): Best for n > 100 with sufficient community structure. The Diophantine features (Eisenstein norm, family code, inradius) add +2-3pp on molecular benchmarks.

## Adding a New Graph Type

1. Identify the TWO most informative integer features for community membership
2. Declare them with `# QA_MAP: b=<feature> (<rationale>)` and `# QA_MAP: e=<feature> (<rationale>)`
3. Run with `degree_only`, `qa21`, and `full` modes
4. If `degree_only` beats `full`, you have a dimensionality problem — reduce n or use feature selection
5. Commit the results honestly — if baseline is at ceiling, QA matches (not "fails")
