# QA-AlphaGeometry Week 4 Step-Depth Ladder Dataset

**Total: 50 problems across 4 tiers**

Created for Week 4 experiments to demonstrate QA efficiency gains on harder problems with longer inference chains and increased branching complexity.

## Dataset Structure

### Tier 0: Sanity Checks (10 problems, 1-2 steps)
Basic problems requiring minimal inference, designed to verify solver correctness on trivial cases.

| Problem ID | Description | Steps | Difficulty |
|------------|-------------|-------|------------|
| t0_p01 | Direct parallel transitivity L1→L3 | 1 | 1 |
| t0_p02 | Perpendicular to parallel L1⊥L2⊥L3 => L1‖L3 | 1 | 1 |
| t0_p03 | Direct segment equality chain (3 segments) | 2 | 2 |
| t0_p04 | Direct angle equality chain (3 angles) | 2 | 2 |
| t0_p05 | Direct concentric circles chain (3 circles) | 2 | 2 |
| t0_p06 | Direct collinearity transitivity | 1 | 1 |
| t0_p07 | Simple parallel-perpendicular L1‖L2, L1⊥L3 | 1 | 1 |
| t0_p08 | Two-step parallel chain | 2 | 2 |
| t0_p09 | Two simple independent goals | 2 | 2 |
| t0_p10 | Trivial equality sanity check | 1 | 1 |

### Tier 1: Basic Complexity (15 problems, 3-4 steps)
Introduces longer chains and basic branching to test QA guidance on non-trivial problems.

| Problem ID | Description | Steps | Difficulty | Key Features |
|------------|-------------|-------|------------|--------------|
| t1_p01 | Parallel chain 4-step | 4 | 3 | Long chain |
| t1_p02 | Mixed parallel/perpendicular | 3 | 3 | Mixed rules |
| t1_p03 | Equality chain 4-step | 4 | 3 | Long chain |
| t1_p04 | **Parallel branching** | 3 | 3 | **Branching introduced** |
| t1_p05 | Circle chain 4-step | 4 | 3 | Circle rules |
| t1_p06 | Multi-goal simple | 3-4 | 4 | Multiple goals |
| t1_p07 | Perp/parallel with distractor | 3 | 3 | 1 distractor |
| t1_p08 | Angle equality chain | 3 | 3 | Angle rules |
| t1_p09 | Collinear transitivity | 3 | 3 | Collinearity |
| t1_p10 | Parallel with 3 distractors | 3 | 4 | **3 distractors** |
| t1_p11 | Equality branching | 3 | 3 | Branching paths |
| t1_p12 | Mixed circle/parallel | 3-4 | 4 | Multi-goal, mixed |
| t1_p13 | Perpendicular chain 3-step | 4 | 4 | Perp cascade |
| t1_p14 | Angle/parallel/equality multi-goal | 3-4 | 4 | 3 goals |
| t1_p15 | Complex branching | 3-4 | 4 | 6 givens, branching |

### Tier 2: Moderate Complexity (15 problems, 5-7 steps)
Significantly longer chains, heavy distractors, and complex branching to challenge beam search efficiency.

| Problem ID | Description | Steps | Difficulty | Key Features |
|------------|-------------|-------|------------|--------------|
| t2_p01 | Parallel chain 7-step | 7 | 5 | Ultra-long chain |
| t2_p02 | Perp/parallel cascade | 5-6 | 6 | Complex cascade |
| t2_p03 | Equality web | 6 | 6 | Multiple paths |
| t2_p04 | Multi-goal chains | 5 | 6 | 3 independent goals |
| t2_p05 | **Parallel with 8 distractors** | 5 | 6 | **Heavy noise** |
| t2_p06 | Mixed perp/parallel complex | 4-5 | 5 | Rule mixing |
| t2_p07 | Collinear complex | 5-6 | 6 | Complex collinearity |
| t2_p08 | Angle chain 6-step | 6 | 5 | Long angle chain |
| t2_p09 | Circle chain 6-step | 6 | 5 | Long circle chain |
| t2_p10 | Equality branching complex | 6-7 | 6 | Web structure |
| t2_p11 | Parallel branching heavy | 5-6 | 6 | Exponential branching |
| t2_p12 | **Mixed all rules** | 5-7 | 7 | **4 goals, all types** |
| t2_p13 | Segment equality chain 7-step | 7 | 5 | Ultra-long |
| t2_p14 | Perp cascade complex | 5-6 | 6 | Long perp chain |
| t2_p15 | **Ultimate distractor** | 6 | 7 | **15 distractors** |

### Tier 3: High Complexity (10 problems, 8-12 steps)
Ultimate challenge problems with ultra-long chains, exponential branching, and massive distractor sets.

| Problem ID | Description | Steps | Difficulty | Key Features |
|------------|-------------|-------|------------|--------------|
| t3_p01 | **Parallel chain 12-step** | 12 | 8 | **Longest chain** |
| t3_p02 | Perp/parallel mega cascade | 7-8 | 9 | 4 perp transitivity |
| t3_p03 | Equality mega web | 8-10 | 9 | 3 parallel paths |
| t3_p04 | **Multi-goal mega** | 8-10 | 10 | **5 goals, all types** |
| t3_p05 | Angle chain 10-step | 10 | 8 | Ultra-long angles |
| t3_p06 | Circle chain 10-step | 10 | 8 | Ultra-long circles |
| t3_p07 | Segment chain 10-step | 10 | 8 | Ultra-long segments |
| t3_p08 | **Ultimate branching** | 8-10 | 10 | **Exponential search** |
| t3_p09 | **Mega distractor** | 8 | 10 | **25 distractors** |
| t3_p10 | **Ultimate mixed** | 10-12 | 12 | **6 goals, all types** |

## Design Principles

1. **Gradual Difficulty Progression**: Each tier doubles the inference depth from the previous
2. **Branching Complexity**: Progressive introduction of multiple valid paths (t1_p04, t2_p11, t3_p08)
3. **Distractor Noise**: Systematic increase in irrelevant givens (1 → 3 → 8 → 15 → 25)
4. **Rule Coverage**: All rule types tested at each tier (parallel, perpendicular, equality, angles, circles, collinearity)
5. **Multi-Goal Challenges**: Increasing number of simultaneous goals (2 → 3 → 4 → 5 → 6)
6. **Real Difficulty**: Problems designed to stress beam search, not just increase depth

## Expected QA Impact

**Tier 0-1**: Minimal QA benefit (problems too simple, branching factor ~1.0)
**Tier 2**: QA should demonstrate 10-30% reduction in states explored via branching pruning
**Tier 3**: QA should demonstrate 30-60% reduction in states explored on high-branching problems

**Key Hypothesis**: QA guidance efficiency scales with problem branching factor, not just depth.

## Usage

```bash
# Run full benchmark
cargo test --release -- --nocapture week4_full_benchmark

# Run specific tier
cargo test --release -- --nocapture tier2_benchmark

# Run ablation study
cargo test --release -- --nocapture ablation_qa_weight_sweep
```

## Validation Criteria

- All problems must be solvable by geometry-only baseline (QA weight = 0.0)
- 100% correctness preservation across all QA weight configurations
- Proof lengths match expected step counts (±1 for multi-goal problems)
- Beam search should explore ≥ 1.0 states per problem on average

## Next Steps (Week 4 Sessions 2-4)

- Session 2: Expand benchmark harness with telemetry (qa_prior_mean, phase_entropy, etc.)
- Session 3: Run ablation studies (QA weight sweep 0.0-0.7, coordinate-derived facts on/off)
- Session 4: Generate results figures and add appendix to paper
