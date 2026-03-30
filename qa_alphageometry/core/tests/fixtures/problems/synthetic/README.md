# Discriminative Synthetic Problems

## Purpose

These problems are specifically designed to **test heuristic discrimination** in beam search. Unlike the Week 4 problems (which have unique proof paths), these problems have:

- **High branching factor** (≥30 successors per expansion)
- **Multiple valid proof paths** (≥2 distinct routes to goal)
- **Structural variation** between paths (not just length)

## Problem Families

### Family S: Perpendicular Lattices with Decoys

**Design:**
- Dense perpendicular structure (QA-sensitive)
- Many irrelevant parallel/equality distractors
- Goal reachable via multiple orthogonal propagation routes

**QA Hypothesis:**
- QA should prefer low-entropy perpendicular chains
- Should avoid distractor-heavy paths
- Should demonstrate reduced `states_expanded` vs geometric baseline

**Problems:**
- `s01_lattice_3x3` - Small lattice (baseline)
- `s02_lattice_4x4` - Medium lattice
- `s03_lattice_5x5` - Large lattice
- `s04_lattice_with_parallels` - Heavy parallel distractors
- `s05_lattice_with_equalities` - Heavy equality distractors
- `s06_mixed_noise` - Combined distractors
- `s07_diagonal_goal` - Corner-to-corner propagation
- `s08_center_goal` - Edge-to-center propagation
- `s09_sparse_lattice` - Less dense structure
- `s10_asymmetric_lattice` - Non-square grid

### Family T: Competing Proof Routes

**Design:**
- Route A: Short (5-7 steps), structurally clean
- Route B: Longer (10-15 steps), transitivity-heavy, noisy

**QA Hypothesis:**
- QA should prefer Route A if it has better harmonic structure
- Should show early divergence at first branch point

**Problems:**
- `t01_two_routes_simple` - Clear A/B choice
- `t02_three_routes` - More complex branching
- `t03_route_a_clean` - A is QA-favored
- `t04_route_b_noisy` - B is distractor-heavy
- `t05_equal_length_routes` - Same length, different structure
- `t06_early_branch` - Divergence at depth 1
- `t07_late_branch` - Divergence at depth 5
- `t08_parallel_paths` - Independent proof chains
- `t09_converging_paths` - Paths merge midway
- `t10_backtrack_required` - Dead ends force exploration

### Family C: Coordinate-Derived Right Triangles

**Design:**
- No explicit `RightTriangle` facts
- Only coordinate geometry → infer perpendicularity
- QA must activate from derived structure

**QA Hypothesis:**
- QA extraction works on coordinate-derived facts
- Demonstrates QA is not purely symbolic

**Problems:**
- `c01_simple_3_4_5` - Classic Pythagorean triple
- `c02_scaled_triangle` - (6,8,10)
- `c03_multiple_triangles` - Several right triangles
- `c04_nested_triangles` - Triangles within triangles
- `c05_grid_coordinates` - Integer grid points
- `c06_decimal_coordinates` - Non-integer coordinates
- `c07_origin_centered` - Symmetric about origin
- `c08_arbitrary_placement` - Random positions
- `c09_overlapping_triangles` - Shared vertices
- `c10_triangle_chain` - Sequential construction

## Success Metrics

### Primary: Beam Divergence

**Metric:** `first_divergence_depth`
- Depth at which `beam_signature(QA=0) ≠ beam_signature(QA=0.7)`
- Lower divergence → stronger heuristic effect

**Target:** ≥ 80% of problems show divergence within 3 depths

### Secondary: Search Efficiency

**Metrics:**
- `states_expanded` - Total beam states popped
- `successors_generated` - Total successors created
- `successors_kept` - Beam utilization

**Target:** QA reduces `states_expanded` by ≥15% on Family S

### Tertiary: Correctness Preservation

**Metric:** Solve status consistency across QA weights

**Target:** 100% consistency (non-negotiable)

## Implementation Strategy

1. **Generate problems programmatically** (Python script)
2. **Validate solvability** before adding to suite
3. **Measure branching factor** empirically
4. **Verify multiple paths exist** via exhaustive search
5. **Run Phase 6 telemetry** with beam signatures
6. **Analyze divergence patterns**

## Expected Timeline

- Week 1: Generate + validate all 30 problems
- Week 2: Run benchmarks + analyze results
- Week 3: Write results section + create plots
- Week 4: Draft paper + prepare arXiv submission
