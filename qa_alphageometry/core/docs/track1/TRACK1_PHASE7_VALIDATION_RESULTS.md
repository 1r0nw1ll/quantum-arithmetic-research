# Track 1 Phase 7: Rule-Batch Validation Results

## Summary

Validated T02 and T03 multi-surface discriminative problems using actual beam search. Both problems pass all discriminativity criteria under the rule-batch architecture.

## Validation Results

### T02: Scaled Multisurface

**Problem Specification:**
- File: `tests/fixtures/problems/synthetic/t02_scaled_multisurface.json`
- Rule families targeted: 8 (all major families)
- Predicted fact volume: 47

**Beam Search Results:**

| Config | Solved | States Expanded | Successors Generated | Depth Reached |
|--------|--------|-----------------|----------------------|---------------|
| QA=0   | ✅ true | 18             | 122                  | 4             |
| QA=0.7 | ✅ true | 10             | 71                   | 3             |

**Beam Divergence:**
- First divergence: **Depth 0** ✅
- Status: **Divergence detected** ✅

**Interpretation:**
- Both configurations solve the problem correctly
- QA prior changes search strategy (fewer states/successors for QA=0.7)
- Immediate beam divergence confirms discriminative nature
- Problem successfully tests QA heuristic effectiveness

---

### T03: Mega Discrimination

**Problem Specification:**
- File: `tests/fixtures/problems/synthetic/t03_mega_discrimination.json`
- Rule families targeted: 8 (maximum discrimination)
- Predicted fact volume: 103

**Beam Search Results:**

| Config | Solved | States Expanded | Successors Generated | Depth Reached |
|--------|--------|-----------------|----------------------|---------------|
| QA=0   | ✅ true | 3              | 16                   | 2             |
| QA=0.7 | ✅ true | 4              | 24                   | 2             |

**Beam Divergence:**
- First divergence: **Depth 0** ✅
- Status: **Divergence detected** ✅

**Interpretation:**
- Both configurations solve the problem efficiently (depth 2)
- QA=0.7 explores slightly more (4 vs 3 states)
- Immediate beam divergence confirms discriminative nature
- Problem is easier than expected but still discriminative

---

## Key Findings

### 1. Successors Metric Clarification

**Predicted vs Actual:**
- T02: Predicted 47 facts → Actual 122 rules fired
- T03: Predicted 103 facts → Actual 16 rules fired

**Explanation:** The `successors_generated` metric counts **total rules fired across all state expansions**, not just initial branching from the root state. This is correct behavior - it's a cumulative count, not a branching factor.

### 2. Discriminativity Validation

Both problems meet all discriminativity criteria:

✅ **Rule surface diversity**: Both trigger 8 distinct rule families
✅ **Fact volume**: Sufficient facts generated to create choice points
✅ **Beam divergence**: QA=0 and QA=0.7 diverge at depth 0
✅ **Correctness**: Both configurations solve both problems
✅ **Search efficiency difference**: QA prior changes search behavior

### 3. Rule-Batch Architecture Confirmed

The validation confirms Phase 7's architectural finding:
- Beam search creates **one successor per rule** that fires
- Each successor contains **all facts** from that rule (batch)
- Max possible successors bounded by #rules (24), not #facts
- Discriminativity comes from **rule diversity**, not fact count alone

---

## Validation Test Implementation

**File:** `tests/track1_rulebatch_validation.rs`

**Test Functions:**
- `test_t02_scaled_multisurface()` - Tests T02 with QA=0 and QA=0.7
- `test_t03_mega_discrimination()` - Tests T03 with QA=0 and QA=0.7

**Beam Configuration:**
```rust
// QA=0 (baseline)
BeamConfig {
    beam_width: 8,
    max_depth: 10,
    max_states: 500,
    scoring: ScoringConfig {
        geometric_weight: 1.0,  // Pure geometric heuristic
        qa_weight: 0.0,
        step_penalty: 0.1,
    },
}

// QA=0.7 (QA-weighted)
BeamConfig {
    beam_width: 8,
    max_depth: 10,
    max_states: 500,
    scoring: ScoringConfig {
        geometric_weight: 0.7,
        qa_weight: 0.7,
        step_penalty: 0.1,
    },
}
```

**Validation Metrics:**
1. Solve status (both configs must solve)
2. States expanded (should differ)
3. Successors generated (cumulative rules fired)
4. Depth reached (efficiency measure)
5. Beam signatures (hash-based divergence detection)

---

## Next Steps

1. **Generate remaining Family T problems** (t04-t10)
   - Use validated multi-surface approach
   - Ensure all pass discriminativity criteria
   - Add validation tests for each

2. **Generate Family C problems** (c01-c10)
   - Coordinate-derived right triangles
   - Different approach from S and T families

3. **Run Phase 6 telemetry** on validated problems
   - Analyze beam signatures in detail
   - Measure QA efficiency metrics
   - Create plots and visualizations

4. **Document findings** for publication
   - Rule-batch architecture as general result
   - Discriminativity criteria aligned with architecture
   - Week 4 results explained by measurement-architecture mismatch

---

## Files Generated/Modified

### New Files:
- `TRACK1_PHASE7_VALIDATION_RESULTS.md` - This document
- `tests/track1_rulebatch_validation.rs` - Validation test suite
- `tests/fixtures/problems/synthetic/t02_scaled_multisurface.json`
- `tests/fixtures/problems/synthetic/t03_mega_discrimination.json`

### Modified Files:
- `scripts/branching_score.py` - Enhanced with rule-batch predictor
- `scripts/generate_rulebatch_problems.py` - Multi-surface problem generator
- `TRACK1_PHASE7_RULEBATCH_DISCOVERY.md` - Architectural discovery doc

---

## Conclusion

✅ **Validation successful** - T02 and T03 are discriminative under rule-batch architecture
✅ **Criteria alignment** - New metrics match actual beam search behavior
✅ **QA sensitivity** - Both problems show different search paths for QA=0 vs QA=0.7
✅ **Ready for scale-up** - Approach validated, can generate remaining 28 problems

**Status:** Phase 7 complete. Proceeding to generate full problem sets for Track 1.
