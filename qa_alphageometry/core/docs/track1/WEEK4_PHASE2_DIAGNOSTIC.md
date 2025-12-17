# Week 4 Phase 2: Diagnostic Report

## Experiment Configuration

**Phase 2 Changes:**
- Increased `max_states`: 2000 → 8000 (4x)
- Increased `max_depth`: 20 → 35 (75%)
- Added termination telemetry: `hit_max_states`, `hit_max_depth`
- Fixed QA extraction to infer from perpendicular structure
- Ran Tier 2 only (15 problems × 10 configs = 150 runs)

**Goal:** Achieve ≥90% solve rate so search reaches discriminative depth where QA weight differences can be measured.

## Results Summary

### Overall Metrics
- **Solve Rate:** 46.7% (70/150) ❌ BELOW 90% TARGET
- **QA Activation:** 33.3% (50/150) ✅ WORKING CORRECTLY
- **Mean QA Prior (when active):** 2.206
- **Mean Phase Entropy (when active):** 1.309

### Critical Finding: States Explored STILL IDENTICAL

**All 15 problems show IDENTICAL states explored across all QA weights**
- No variance in search behavior despite QA activation
- Coefficient of variation: 0.0% for all problems

### Termination Analysis

**Budget limits NOT the issue:**
- Hit `max_states` limit: 0/150 runs (0.0%)
- Hit `max_depth` limit: 0/150 runs (0.0%)

Problems terminate LONG BEFORE reaching budget limits:
- Max states explored: 1356 (vs 8000 limit)
- Max depth reached: 12 (vs 35 limit)

### Problem-by-Problem Breakdown

| Problem | Solve Rate | Avg States | Status |
|---------|-----------|------------|--------|
| t2_p01_parallel_chain_7 | 100% | 1356 | ✅ Trivially solved |
| t2_p02_perp_parallel_cascade | 0% | **0** | ❌ **ZERO STATES** |
| t2_p03_equality_web | 0% | 187 | ❌ Never solved |
| t2_p04_multi_goal_chains | 100% | 1336 | ✅ Trivially solved |
| t2_p05_parallel_with_heavy_distractors | 100% | 160 | ✅ Trivially solved |
| t2_p06_mixed_perp_parallel_complex | 0% | **0** | ❌ **ZERO STATES** |
| t2_p07_collinear_complex | 0% | **0** | ❌ **ZERO STATES** |
| t2_p08_angle_chain_6 | 100% | 471 | ✅ Trivially solved |
| t2_p09_circle_chain_6 | 100% | 471 | ✅ Trivially solved |
| t2_p10_equality_branching_complex | 0% | 506 | ❌ Never solved |
| t2_p11_parallel_branching_heavy | 0% | 49 | ❌ Never solved |
| t2_p12_mixed_all_rules | 0% | 523 | ❌ Never solved |
| t2_p13_segment_equality_chain_7 | 100% | 1356 | ✅ Trivially solved |
| t2_p14_perp_cascade_complex | 0% | **0** | ❌ **ZERO STATES** |
| t2_p15_ultimate_distractor | 100% | 471 | ✅ Trivially solved |

**Bimodal Distribution:**
- 7 problems solve trivially (100% solve rate, same search path every time)
- 4 problems fail immediately with ZERO states explored
- 4 problems explore but never solve despite budget headroom

## Diagnostic Analysis

### Finding 1: QA Activation is Correct ✅

QA extraction is working as designed:
- Activates on problems with perpendicular structure (e.g., t2_p02: QA Prior = 2.484)
- Does NOT activate on problems without perpendicular structure (e.g., t2_p01: QA Prior = 0.000)
- This is the expected behavior after the extraction fix

### Finding 2: Budget Exhaustion is NOT the Problem ❌

The hypothesis that "search terminates before reaching discriminative depth due to budget exhaustion" is **FALSIFIED**.

Evidence:
- NO runs hit budget limits
- Problems terminate at ~50-1356 states (vs 8000 limit)
- Problems terminate at depth 0-12 (vs 35 limit)

### Finding 3: Search is Deterministic Regardless of Heuristic ❌

**All problems show IDENTICAL states explored** across:
- QA weight: 0%, 10%, 30%, 50%, 70%
- Coord facts: ON vs OFF

This suggests:
1. The beam search is following the same exploration order regardless of scoring
2. The heuristic weights are not affecting tie-breaking
3. Problems either:
   - Solve via a unique optimal path (7 problems), or
   - Fail immediately (4 problems with 0 states), or
   - Explore the same prefix then give up (4 problems)

### Finding 4: Zero-State Problems are Broken 🔥

4 problems show **ZERO states explored**:
- t2_p02_perp_parallel_cascade
- t2_p06_mixed_perp_parallel_complex
- t2_p07_collinear_complex
- t2_p14_perp_cascade_complex

This indicates:
- Problem fails to initialize, or
- Goal is immediately unreachable, or
- Loader/parser error

Example JSON (t2_p02) looks valid:
```json
{
  "id": "t2_p02_perp_parallel_cascade",
  "givens": [
    {"Perpendicular": [1, 2]},
    {"Perpendicular": [2, 3]},
    {"Perpendicular": [3, 4]},
    {"Perpendicular": [4, 5]},
    {"Perpendicular": [5, 6]}
  ],
  "goals": [
    {"Perpendicular": [1, 6]}
  ]
}
```

This should be solvable via transitive perpendicular inference.

## Root Cause Hypotheses

### Hypothesis A: Beam Search Implementation Issue

The beam search may not be using the scoring function correctly for beam ordering.

**Evidence:**
- IDENTICAL states explored across all weight configurations
- Weights have no effect on search trajectory

**Next Step:** Inspect `BeamSolver::solve()` to verify scoring is used for beam selection.

### Hypothesis B: Problems Are Degenerate

The "step-depth ladder" problems may be too simple or have unique proof paths that eliminate heuristic influence.

**Evidence:**
- 7/15 problems solve trivially with same path
- Low state counts even when unsolved (49-523 states)

**Next Step:** Create more branching problems with multiple valid proof paths.

### Hypothesis C: Missing Rules or Broken Loader

The 4 zero-state problems may be hitting missing rule implementations or loader bugs.

**Evidence:**
- t2_p02 (perpendicular cascade) should be solvable but shows 0 states
- t2_p07 (collinear) may need collinear inference rules

**Next Step:** Run single-problem debug on t2_p02 to see failure mode.

### Hypothesis D: Goal Checking is Too Strict

Problems may be solving but not recognized as solved due to goal normalization issues.

**Evidence:**
- 4 problems explore states but never "solve"
- States explored varies (49-506) but all fail

**Next Step:** Add debug logging to goal-checking logic.

## Recommended Next Steps

### Priority 1: Debug Zero-State Problems (1-2 hours)

Run t2_p02 with verbose logging to diagnose why 0 states are explored:

```bash
RUST_LOG=debug cargo test --release -- --ignored t2_p02
```

Expected outcomes:
- Loader error → Fix JSON format or loader
- Missing rule → Implement perpendicular transitivity
- Goal unreachable → Problem is invalid

### Priority 2: Verify Beam Scoring is Used (30 min)

Add debug logging to `BeamSolver::solve()` to confirm:
1. Scores are computed for each beam candidate
2. Beam selection sorts by score
3. Different weights produce different orderings

If beam scoring is NOT being used → this is a CRITICAL BUG.

### Priority 3: Create Discriminative Test Problems (2-3 hours)

Design new problems with:
- Multiple valid proof paths of different lengths
- High branching factor (>100 candidates per step)
- QA-sensitive structure (perpendicular heavy vs perpendicular light)

These problems will force heuristic discrimination.

### Priority 4: Comparative Baseline (OPTIONAL)

Compare against pure geometric baseline:
- AlphaGeometry without QA
- Random search
- Greedy search

This validates whether the issue is QA-specific or systemic.

## Status of Original Week 4 Goals

### ✅ Completed
- QA extraction fix (infer from perpendicular structure)
- QA activation verification (nonzero priors, entropy)
- Termination telemetry implementation
- Budget increase experiment

### ❌ Blocked
- **Cannot measure QA efficiency** until search is non-deterministic
- **Cannot claim QA improves solve rate** (it's 46.7% regardless of QA weight)
- **Cannot proceed with Session 4 plotting** (no variance to visualize)

## Revised Timeline

### Immediate (Today)
1. Debug zero-state problems
2. Verify beam scoring implementation
3. Report findings

### This Week
1. Fix identified bugs (beam scoring, missing rules, etc.)
2. Re-run Phase 2 with fixes
3. If still deterministic → create new discriminative problems

### Next Week
1. Comparative baseline experiments
2. Session 4 plotting (IF we have variance)
3. Paper writing

## Key Takeaway

**The increased budgets revealed the real problem: search is deterministic.**

This is actually EXCELLENT science - we've discovered that the issue is not "insufficient search depth" but rather "heuristic has no effect on search." This is a more fundamental and fixable problem than budget tuning.

The next step is NOT to increase budgets further, but to:
1. Fix the beam search to actually use the scoring function, and/or
2. Create problems where heuristic choice matters

We're making progress - just in a different direction than anticipated.
