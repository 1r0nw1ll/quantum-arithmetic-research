# Week 4 Phase 3: Beam Search Tie-Breaking Fix

## Problem Identified

ChatGPT's analysis identified the **root cause of determinism** in our beam search:

### Issue 1: Single-Level Sorting (CRITICAL BUG)

**Location:** `src/search/beam.rs:151-153`

**Original Code:**
```rust
next_beam.sort_by(|a, b| {
    b.1.total.partial_cmp(&a.1.total).unwrap_or(std::cmp::Ordering::Equal)
});
```

**Problem:**
- Sorts ONLY by `total` score
- No tie-breaking when scores are identical or near-identical
- Rust's stable sort preserves insertion order for ties
- Insertion order is deterministic (fixed rule order → fixed fact order)
- Result: **Same top-k selected regardless of QA weight**

**Why This Caused Our Results:**
- Early in search, many states have identical scores (common in symbolic search)
- QA component contributes to `total`, but many ties remain
- When ties occur, stable sort + deterministic insertion → same beam selected
- This perfectly explains: "QA activates but states explored are identical"

### Issue 2: Single-Fact Successors (AMPLIFIES TIES)

**Location:** `src/search/beam.rs:179-202`

**Pattern:**
```rust
for rule in rules {
    let new_facts = rule.apply(state);
    for fact in new_facts {
        // One successor per fact
        let mut new_state = state.clone();
        new_state.facts.insert(fact.clone());
        successors.push((new_state, new_trace));
    }
}
```

**Problem:**
- Creates one successor per individual fact
- Successors differ by exactly one fact
- Early states have nearly identical scores
- Massively increases number of ties
- Reduces discriminative power of any heuristic

### Issue 3: Zero-State Problems

**Explanation:**
When `expand_state()` returns empty (no rules fire):
- `next_beam` stays empty
- Solver returns immediately with `states_explored = 0`

**This explains:**
- t2_p02_perp_parallel_cascade: 0 states
- t2_p06_mixed_perp_parallel_complex: 0 states
- t2_p07_collinear_complex: 0 states
- t2_p14_perp_cascade_complex: 0 states

**Root cause:** Missing rules or representation mismatch (not a beam search bug).

## Fix Implemented: Multi-Level Tie-Breaking

### Fix A: Score-Sensitive Comparator

**New Code:**
```rust
next_beam.sort_by(|a, b| {
    b.1.total.partial_cmp(&a.1.total).unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.1.qa_prior.partial_cmp(&a.1.qa_prior).unwrap_or(std::cmp::Ordering::Equal))
        .then_with(|| b.1.geometric_score.partial_cmp(&a.1.geometric_score).unwrap_or(std::cmp::Ordering::Equal))
        .then_with(|| a.0.facts.len().cmp(&b.0.facts.len())) // Prefer simpler states
});
```

**Ordering:**
1. **Primary:** `total` score (as before)
2. **Tie-break 1:** `qa_prior` (harmonic structure)
3. **Tie-break 2:** `geometric_score` (distance to goal)
4. **Tie-break 3:** `facts.len()` (prefer simpler states)

**Why This Works:**
- When `total` ties, `qa_prior` becomes decisive
- Different QA weights → different `qa_prior` → different `total` → different ordering
- Even if `total` identical, `qa_prior` directly breaks ties
- Creates **score-dependent variance** across QA weight sweep

### Fix C: Diagnostic Logging

**Added Code:**
```rust
// FIX C: Diagnostic logging at depth 0 for first problems
if depth == 0 && next_beam.len() > 5 {
    if cfg!(test) || std::env::var("RUST_LOG").is_ok() {
        eprintln!("DEBUG [depth={}]: Top-5 scores before truncation:", depth);
        for (i, (state, score, _)) in next_beam.iter().take(5).enumerate() {
            eprintln!("  #{} total={:.4} qa={:.4} geo={:.4} facts={}",
                      i + 1, score.total, score.qa_prior, score.geometric_score, state.facts.len());
        }
    }
}
```

**Purpose:**
- Verify scorer is wired correctly
- Show score distribution before truncation
- Confirm QA weight affects ordering
- Enable with `RUST_LOG=debug` environment variable

## Expected Outcome

### If Tie-Breaking Fix Works:

**Variance should appear in `states_explored`:**
- Different QA weights → different beam orderings
- Different beams → different search paths
- Different paths → different state counts

**Metric to check:**
```python
states_std = group['states_explored'].std()
states_mean = group['states_explored'].mean()
cv = (states_std / states_mean * 100)
```

If CV > 1-5% → **Tie-breaking is working!**

### If Still Deterministic:

**Possible causes:**
1. Scores are ALL identical (need to verify with diagnostic logs)
2. QA extractor returning same value for all states (unlikely after our fix)
3. Deeper architecture issue

## Next Steps After This Run

### Scenario A: Variance Appears ✅

1. **Analyze variance pattern**
   - Which problems show variance?
   - How much does QA weight affect states explored?
   - Does higher QA weight reduce or increase exploration?

2. **Measure QA efficiency**
   - Does QA reduce states explored on perpendicular-heavy problems?
   - Does QA maintain solve rate?
   - Generate Session 4 plots

3. **Proceed with paper writing**

### Scenario B: Still Deterministic ❌

1. **Check diagnostic logs** (RUST_LOG=debug run)
   - Are scores varying across QA weights?
   - If yes → deeper issue
   - If no → scorer not working

2. **Implement Fix B: Rule-Batch Successors**
   - Generate one successor per rule application
   - Add all new facts at once
   - Increases discriminative power

3. **Debug zero-state problems**
   - Add missing rules (perpendicular transitivity, collinear inference)
   - Fix representation mismatches

## Technical Notes

### Why Stable Sort Matters

Rust's `sort_by` is **stable**:
- Preserves relative order of equal elements
- If `a.score == b.score` and `a` was inserted before `b`, then `a` remains before `b`

Combined with deterministic insertion (fixed rule/fact iteration order):
- Same ties → same preserved order → same top-k selected

### Why Multi-Level Comparison Works

The `then_with` chain creates lexicographic ordering:
```
(total_a, qa_a, geo_a, len_a) vs (total_b, qa_b, geo_b, len_b)
```

This is exactly what we need - QA weight affects **both** `total` and `qa_prior`, so changes propagate through the comparison.

### Alternative Fix (Not Implemented Yet)

**Fix B: Rule-Batch Successors** would be more architecturally correct:

```rust
for rule in rules {
    let new_facts = rule.apply(state);
    if !new_facts.is_empty() {
        // ONE successor with ALL facts from this rule
        let mut new_state = state.clone();
        for fact in &new_facts {
            new_state.facts.insert(fact.clone());
        }
        // Single proof step with multiple conclusions
        let step = ProofStep {
            id: step_id,
            rule_id: rule.id().to_string(),
            premises: vec![],
            conclusions: new_facts,  // All facts together
            score: rule.cost(),
            explanation: None,
        };
        successors.push((new_state, new_trace_with_step));
    }
}
```

This would:
- Reduce number of successors (fewer ties)
- Increase score differences between successors
- Make heuristic more discriminative
- Be more semantically correct (rule application = atomic operation)

## Files Changed

- `core/src/search/beam.rs` - Added multi-level tie-breaking and diagnostic logging

## Experiment Running

**Command:**
```bash
cargo test --release test_week4_session3_tier2_coord_ablation -- --ignored --nocapture 2>&1 | tee tier2_phase3_tiebreaking.log
```

**Expecting:**
- 15 problems × 10 configs = 150 runs
- CSV: `benchmark_results_week4_session3_tier2.csv`
- Analysis will show if variance appeared

**Key Metric:**
```
states_explored variance across QA weights
```

If variance > 0% consistently → **TIE-BREAKING FIX SUCCESSFUL!**
