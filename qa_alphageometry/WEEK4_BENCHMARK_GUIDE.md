# Week 4 Benchmark Guide

## Overview

Week 4 introduces a 50-problem step-depth ladder dataset (Tiers 0-3) with enhanced telemetry to demonstrate QA efficiency gains on harder problems.

## Quick Start

```bash
# Run Tier 0 (sanity checks, fast)
cargo test --release test_week4_full_benchmark_tier0 -- --nocapture

# Run Tier 1 (basic complexity, moderate)
cargo test --release test_week4_full_benchmark_tier1 -- --ignored --nocapture

# Run ALL tiers (Tier 0-3, comprehensive - takes longer)
cargo test --release week4 -- --ignored --nocapture
```

## Dataset Structure

### Tier 0: Sanity Checks
- **Problems**: 10
- **Steps**: 1-2
- **Purpose**: Verify solver correctness on trivial cases
- **Expected**: Minimal QA benefit (branching factor ~1.0)

### Tier 1: Basic Complexity
- **Problems**: 15
- **Steps**: 3-4
- **Purpose**: Test QA guidance on non-trivial problems with basic branching
- **Expected**: 5-15% reduction in states explored

### Tier 2: Moderate Complexity
- **Problems**: 15
- **Steps**: 5-7
- **Purpose**: Stress test beam search with heavy distractors and complex branching
- **Expected**: 10-30% reduction in states explored

### Tier 3: High Complexity
- **Problems**: 10
- **Steps**: 8-12
- **Purpose**: Ultimate challenge problems with exponential search spaces
- **Expected**: 30-60% reduction in states explored

## Configurations Tested

All problems are run with 5 QA weight configurations:

1. **Geometry Only** (QA weight = 0.0)
   - Pure symbolic baseline
   - No QA guidance

2. **QA 10%** (QA weight = 0.1, Geometric = 0.9)
   - Minimal QA influence
   - Slight search reordering

3. **QA 30%** (QA weight = 0.3, Geometric = 0.7)
   - Moderate QA guidance
   - Balanced hybrid

4. **QA 50%** (QA weight = 0.5, Geometric = 0.5)
   - Equal weighting
   - Strong QA influence

5. **QA 70%** (QA weight = 0.7, Geometric = 0.3)
   - Dominant QA guidance
   - Experimental configuration

## Telemetry Metrics

Each benchmark run collects the following metrics:

### Search Metrics
- `states_explored`: Total beam search states expanded
- `depth_reached`: Maximum proof depth reached
- `proof_steps`: Number of steps in final proof (if solved)
- `time_ms`: Wall-clock time in milliseconds

### QA Telemetry
- `qa_prior_mean`: Mean QA posterior (0.0 to ~10.0)
- `phase_entropy`: Entropy of mod-24 phase distribution (0.0 to ~3.2)
- `primitive_mass`: Fraction of primitive Pythagorean triples (0.0 to 1.0)
- `female_mass`: Fraction of "female" tuples (0.0 to 1.0)
- `fermat_mass`: Fraction of Fermat family tuples (0.0 to 1.0)
- `mean_jk`: Mean J+K invariant
- `mean_harmonic_index`: Mean |C-F| harmonic index
- `num_candidates`: Number of candidate QA tuples extracted
- `qa_confidence`: QA extraction confidence (0.0 to 1.0)

## Output Files

Each tier produces two output files:

### CSV: `benchmark_results_week4_tier{N}.csv`
Full per-problem results with all telemetry columns. Use for:
- Detailed analysis
- Plotting efficiency curves
- Statistical tests
- Correlation analysis

### JSON: `benchmark_summary_week4_tier{N}.json`
Aggregated tier-level statistics with mean±std. Use for:
- Quick overview
- Paper tables
- Comparing configurations

## Interpreting Results

### Success Criteria

1. **100% Correctness Preservation**
   - All QA weight configurations must solve the same set of problems
   - Assertion failure indicates a bug (QA is changing semantics)

2. **Efficiency Gains on Tier 2-3**
   - QA 30-50% should reduce `states_explored` by 10-60% on harder problems
   - Gains should scale with problem branching factor

3. **No Degradation on Tier 0-1**
   - Simple problems may show no benefit (already optimal)
   - QA should not increase states explored by more than 10%

### Key Plots to Generate (Session 4)

1. **States Explored vs QA Weight** (per tier)
   - Line plot showing efficiency curve
   - Error bars from standard deviation

2. **Time vs QA Weight** (per tier)
   - Check if QA overhead is negligible

3. **Phase Entropy vs States Explored**
   - Correlation analysis: does higher entropy predict harder problems?

4. **Solve Rate by Tier and Configuration**
   - Heatmap: Tier × QA Weight

## Session 3: Ablation Study

Run the coordinate-derived facts ablation:

```bash
cargo test --release test_week4_coord_facts_ablation -- --ignored --nocapture
```

This tests whether adding coordinate-derived facts (when available) improves QA extraction quality.

## Example Output

```
🔬 Week 4 Benchmark: TIER1 (15 problems)

📝 t1_p01_parallel_chain_4 (difficulty: 3)
   Geometry Only | ✅ | Steps: 4, States: 12, Time: 0ms | QA Prior: 2.145, Entropy: 1.234
   QA 10%        | ✅ | Steps: 4, States: 11, Time: 0ms | QA Prior: 2.145, Entropy: 1.234
   QA 30%        | ✅ | Steps: 4, States: 9, Time: 0ms | QA Prior: 2.145, Entropy: 1.234
   QA 50%        | ✅ | Steps: 4, States: 7, Time: 0ms | QA Prior: 2.145, Entropy: 1.234
   QA 70%        | ✅ | Steps: 4, States: 7, Time: 0ms | QA Prior: 2.145, Entropy: 1.234

...

📊 TIER1 Summary Statistics:

QA 30% (QA weight: 0.3)
   Solve Rate:     100.0% (15/15)
   Avg States:     8.73 ± 2.45
   Avg Steps:      3.80
   Avg Time:       0.12ms
   Avg QA Prior:   2.156
   Avg Entropy:    1.289

✅ CORRECTNESS VERIFIED: QA preserves solve status across all 5 configs!
```

## Next Steps

1. **Session 2 (Current)**: Run Tier 0-1 benchmarks to validate harness
2. **Session 3**: Implement and run coordinate facts ablation
3. **Session 4**: Generate plots and add results appendix to paper
