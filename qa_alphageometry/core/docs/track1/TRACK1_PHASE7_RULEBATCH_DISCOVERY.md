# Track 1 Phase 7: Rule-Batch Architecture Discovery

## Critical Discovery (2025-12-15)

### The Root Cause of Low Branching

Investigation into why generated problems showed max_successors=3-4 instead of target ≥30 revealed a **fundamental architectural constraint**:

**Beam search creates ONE successor per RULE, not per fact.**

From `src/search/beam.rs:246-265`:
```rust
fn expand_state(...) -> Vec<(GeoState, ProofTrace)> {
    for rule in rules {
        let new_facts = rule.apply(state);

        if !new_facts.is_empty() {
            // FIX B: ONE successor with ALL facts from this rule
            let mut new_state = state.clone();
            for fact in &new_facts {
                new_state.facts.insert(fact.clone());
```

### Implications

1. **Max successors bounded by #rules** (24 total), not by #facts
2. **Target of "30 successors" was mis-specified** - implicitly assumed fact-level granularity
3. **Week 4 problems had unique proof paths** due to rule batching, not QA heuristic failure
4. **Branching score predictor was counting facts** (correct!), but architecture maps facts → rules (batching)

### Example: s01_lattice_3x3

**Givens create:**
- Hub 1 (line 1): 6 perpendicular spokes → C(6,2) = 15 parallel facts
- Hub 2 (line 4): 6 perpendicular spokes → C(6,2) = 15 parallel facts
- Hub 3 (line 7): 5 perpendicular spokes → C(5,2) = 10 parallel facts
- Parallel chain: 8 lines → transitivity facts

**Predicted:** branching_score = 77 (fact volume)

**Actual:** max_successors = 4 (number of rules that fire):
1. `PerpendicularSymmetry`
2. `PerpendicularToParallel` (generates 40 parallel facts in ONE batch)
3. `ParallelTransitivity`
4. `ParallelPerpendicular`

### Decision: Adopt Rule-Batch Metrics

**Chosen approach:** Option 1 + Option 2 (combined)

1. **Keep rule-batch architecture** (do NOT switch to per-fact successors)
   - Preserves Fix B benefits (massive redundancy reduction)
   - Clean, publishable story
   - Per-fact would cause combinatorial explosion

2. **Redefine discriminativity criteria** to match architecture:

**Old (fact-level, mis-specified):**
- Target: max_successors_generated ≥ 30

**New (rule-batch, architecturally aligned):**
- Rule surface score ≥ 4 (distinct rule families fire)
- Fact volume score ≥ 25 (total new facts generated)
- Batch diversity: Jaccard distance ≥ 0.3 between successor fact-batches
- Beam divergence: signatures differ by depth ≤ 2

### Updated Scoring Function

Enhanced `scripts/branching_score.py` to predict:

**Rule families tracked:**
1. `PerpendicularToParallel` (shared-perp hubs)
2. `ParallelTransitivity` (parallel chains)
3. `ParallelPerpendicular` (perp+parallel propagation)
4. `OnCircleToConcyclic` (C(n,4) from n points on circle)
5. `OnLineToCollinear` (C(n,3) from n points on line)
6. `CoincidentLineTransitivity` (coincident chains)
7. `ConcentricTransitivity` (concentric chains)
8. `EqualityTransitivity` (equality chains)

**Metrics:**
- `rule_surface_score` = count(nonzero components)
- `fact_volume_score` = sum(all components)

### Validation Results

**T01 (multi-surface design):**
- Rule surface: **8** ✅ (exceeds target of 4)
- Fact volume: **16** ❌ (below target of 25)
- Fires 8 distinct rule families but with small batches

**s01-s03 (perp/parallel only):**
- Rule surface: **3** ❌ (below target of 4)
- Fact volume: **77-290** ✅ (exceeds target of 25)
- Fires only 3 rule families but with massive batches

**Conclusion:** Need **hybrid approach** - T01's multi-surface design with larger injectors.

## Next Steps

1. **Update telemetry** (beam search) to track:
   - `rules_fired_total`
   - `new_facts_generated_total`
   - Per-depth: `rules_fired_count`, `new_facts_generated_total`, `successor_batch_sizes`, `batch_jaccard_min/mean`

2. **Generate Family T problems** using hybrid approach:
   - Multi-surface design (8+ rule families)
   - Larger injectors (fact_volume ≥ 25)
   - 2 distinct proof routes

3. **Update structure probe** to validate new metrics

4. **Scale to 30 problems** across families S, T, C

## Files Created

- `tests/fixtures/problems/synthetic/t01_dual_route_rulebatch.json` - Multi-surface reference problem
- `scripts/branching_score.py` - Enhanced rule-batch discriminativity scorer
- `TRACK1_PHASE7_RULEBATCH_DISCOVERY.md` - This document

## Publishable Finding

**Framing:** "Discriminative problems require choice points where multiple rules generate distinct fact-batches with divergent reachability - this is a general result about symbolic search spaces under rule-batch successor generation."

This explains Week 4 results as an architectural property, not a QA heuristic failure.
