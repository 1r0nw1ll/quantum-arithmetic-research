# Week 4 Phase 5: Comprehensive Telemetry Implementation

## Motivation

After implementing Fix B (rule-batch successors), states_explored remained identical across all QA weights despite:
- ✅ QA extraction working (nonzero priors)
- ✅ Scorer producing varying values
- ✅ Multi-level tie-breaking implemented
- ✅ Rule-batch successors reducing state count dramatically

**Hypothesis:** Test problems have unique proof paths with low branching factor, eliminating heuristic influence.

**Solution:** Implement comprehensive telemetry to definitively prove or disprove this hypothesis.

## Implementation (ChatGPT Track 0)

### Three Telemetry Metrics

**Previous (single metric):**
- `states_explored`: Total states visited (ambiguous semantics)

**New (three distinct metrics):**
1. **`states_expanded`**: Number of beam states **popped from beam and expanded**
   - Counts how many times we call `expand_state()`
   - Measures actual search work done

2. **`successors_generated`**: Total successors **created across all expansions**
   - Counts every successor produced by rule applications
   - Replaces old `states_explored` for budget limiting

3. **`successors_kept`**: Successors **retained after beam truncation**
   - Counts states added to next beam
   - Tracks beam utilization efficiency

### Beam Signatures for Divergence Detection

**Purpose:** Detect if/when beam contents diverge between QA weights.

**Implementation:**
```rust
fn beam_signature(beam: &[(GeoState, StateScore, ProofTrace)]) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Sort beam by stable key and hash structure
    let mut identifiers: Vec<(usize, usize)> = beam.iter()
        .map(|(state, score, _)| (
            state.facts.len(),
            (score.total * 1000.0) as usize,  // Discretize score
        ))
        .collect();

    identifiers.sort();
    identifiers.hash(&mut hasher);
    hasher.finish()
}
```

**Usage:**
- Record signature at each depth
- Compare signatures between QA=0 and QA=0.7 runs
- If signatures **identical** → unique proof paths confirmed
- If signatures **differ** → beam search WAS affected, but old metric didn't capture it

### SearchResult Updates

**Before:**
```rust
pub struct SearchResult {
    pub solved: bool,
    pub proof: Option<ProofTrace>,
    pub states_explored: usize,
    pub depth_reached: usize,
    pub best_score: f64,
}
```

**After:**
```rust
pub struct SearchResult {
    pub solved: bool,
    pub proof: Option<ProofTrace>,

    // TELEMETRY: Comprehensive search metrics
    pub states_expanded: usize,        // Beam states popped and expanded
    pub successors_generated: usize,   // Total successors created
    pub successors_kept: usize,        // Successors kept after truncation

    pub depth_reached: usize,
    pub best_score: f64,

    // TELEMETRY: Beam signatures at each depth
    pub beam_signatures: Vec<(usize, u64)>,
}
```

### solve() Loop Updates

**Tracking states_expanded:**
```rust
for (state, _score, trace) in beam.iter() {
    states_expanded += 1;  // Count beam state expanded
    let successors = self.expand_state(state, trace, depth);
    // ...
}
```

**Tracking successors_generated:**
```rust
for (successor_state, successor_trace) in successors {
    successors_generated += 1;  // Count successor generated
    // ... scoring and goal checking ...
    next_beam.push((successor_state, successor_score, successor_trace));
}
```

**Tracking successors_kept and beam signatures:**
```rust
next_beam.truncate(self.config.beam_width);

// Track successors kept after truncation
successors_kept += next_beam.len();

// Record beam signature for divergence detection
let signature = Self::beam_signature(&next_beam);
beam_signatures.push((depth, signature));
```

## Files Changed

### Core Library

**`core/src/search/beam.rs`:**
- Updated `SearchResult` struct (lines 37-64)
- Added `beam_signature()` function (lines 77-95)
- Updated `solve()` loop tracking (lines 120-224)
- Updated test assertions (lines 315, 337, 372-376, 422-425, 473-480)

### Benchmark Tests

**`tests/week4_session3_ablation.rs`:**
- Updated `BenchmarkResult` struct (lines 21-25)
- Updated telemetry population (lines 124, 136-139)
- Updated output formatting (lines 212-218, 283-289)

## Verification

### Unit Test Output

**test_parallel_transitivity_proof:**
```
✅ Parallel transitivity proof found!
   Steps: 1
   States expanded: 1
   Successors generated: 1
   Successors kept: 0
   Depth: 1
```

**test_qa_guidance_comparison:**
```
📊 QA Guidance Comparison:
   QA OFF - Solved: true, Expanded: 2, Generated: 4, Kept: 2, Depth: 2
   QA ON  - Solved: true, Expanded: 2, Generated: 4, Kept: 2, Depth: 2
```

Both tests pass with new telemetry showing detailed breakdown.

## Next Steps (ChatGPT Tracks 0-2)

### Track 0: Divergence Analysis (IMMEDIATE)

Run benchmark comparing QA=0 vs QA=0.7 and analyze beam signatures:

```bash
cargo test --release test_week4_session3_tier2_coord_ablation -- --ignored --nocapture 2>&1 | tee tier2_phase5_telemetry.log
```

**Analysis script:**
```python
# Compare beam signatures between runs
qa_0_sigs = results[results['qa_weight'] == 0.0]['beam_signatures']
qa_70_sigs = results[results['qa_weight'] == 0.7]['beam_signatures']

for problem in problems:
    sigs_0 = qa_0_sigs[problem]
    sigs_70 = qa_70_sigs[problem]

    # Find first divergence depth
    first_divergence = None
    for depth, (sig0, sig70) in enumerate(zip(sigs_0, sigs_70)):
        if sig0[1] != sig70[1]:  # Compare hash values
            first_divergence = depth
            break

    if first_divergence is None:
        print(f"{problem}: IDENTICAL beams (unique path confirmed)")
    else:
        print(f"{problem}: Divergence at depth {first_divergence}")
```

### Track 1: Discriminative Synthetic Problems

Create 30 problems with:
- High branching factor (>50 successors/expansion)
- Multiple valid proof paths
- QA-sensitive structure (perpendicular-heavy)
- Known solution depth (5-7 steps)

**Families:**
- **Family S:** Perpendicular lattice with decoy parallels
- **Family T:** Two competing proof routes (one QA-favored, one not)
- **Family C:** Coordinate-derived right triangles

### Track 2: External Benchmark Validation

Adapt 50-problem subset from:
- Geometry3K / GeoEval
- AlphaGeometry's synthetic dataset

Validate that QA guidance affects search on external problems.

## Expected Outcomes

### If Beam Signatures Identical

**Conclusion:** Test problems have unique proof paths. QA cannot influence search when there's no choice.

**Action:**
1. Create discriminative problems (Track 1)
2. Use external benchmarks (Track 2)
3. Report findings honestly in paper

### If Beam Signatures Differ

**Conclusion:** Beam search WAS affected by QA, but old single metric missed it.

**Action:**
1. Analyze divergence patterns
2. Measure efficiency differences (successors_generated, time)
3. Generate plots for paper

### If Signatures Differ BUT Metrics Still Identical

**Conclusion:** Beam contents change but search cost remains constant (lucky convergence).

**Action:**
1. Investigate why different paths cost the same
2. Check if both paths are optimal
3. This would be a fascinating result worth reporting

## Success Criteria

- ✅ Telemetry compiles and passes unit tests
- ✅ Benchmark runs without errors
- ✅ CSV includes new fields
- ⏳ Beam signature analysis reveals variance or confirms unique paths
- ⏳ Paper narrative updated based on findings

## Scientific Value

This telemetry upgrade transforms our investigation from:
- "QA doesn't work" → vague conclusion
- To: "Test problems have property X which prevents heuristic discrimination" → specific, falsifiable hypothesis

Regardless of outcome, we gain scientific clarity about **why** QA did or didn't affect search.
