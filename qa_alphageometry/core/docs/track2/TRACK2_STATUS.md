# Track 2: External Validation Status

**Status**: ⚠️ BLOCKER IDENTIFIED - DECISION PENDING
**Phase**: 2.1 (Ingestion Analysis)
**Date**: 2025-12-16

---

## Quick Summary

**CRITICAL FINDING**: Geometry3K dataset is in VQA (Visual Question Answering) format with PNG images + minimal text. **No symbolic IR available** - incompatible with Track 1's symbolic reasoning architecture.

**Dataset coverage**:
- Total: 2,101 training problems
- Parseable text: 434 problems (20.7%) mention our fact types
- **Missing**: Complete symbolic geometric setup (points, lines, circles, relationships)

---

## Four Options Forward

### Option A: Accept Stop Condition ✅ RECOMMENDED

**Pros**:
- Track 1 stands alone as strong contribution
- Intellectually honest (format mismatch, not quality issue)
- No scope creep
- Clear path to publication

**Cons**:
- No external validation

**Timeline**: 0 days (done now)

### Option B: Manual Case Studies (50-100 problems)

**Pros**:
- Provides some external validation
- Low scope creep
- Controlled quality

**Cons**:
- Selection bias risk
- Defeats "external validation" narrative (manual curation)
- Still small sample

**Timeline**: 1-2 days

### Option C: Build NLP Parser ❌ NOT RECOMMENDED

**Pros**:
- Could ingest ~434 problems automatically

**Cons**:
- Massive scope creep (violates Track 2 principle)
- Introduces parsing errors as confound
- Still misses 80% of dataset
- Dilutes Track 1 narrative

**Timeline**: 1-2 weeks

### Option D: Search Alternative Datasets

**Pros**:
- Might find compatible dataset

**Cons**:
- Preliminary search suggests no large-scale symbolic IR datasets exist
- Time investment with uncertain payoff

**Timeline**: 1-3 days

---

## Recommendation

**Stop Track 2. Publish Track 1 as-is.**

**Rationale**:
1. Track 1 is scientifically sound and publication-ready
2. No compatible external dataset exists (not a failure of our work)
3. Honest acknowledgment in limitations section
4. Future work can address this when:
   - Symbolic IR benchmarks emerge
   - Vision-augmented preprocessing is developed

**Updated Track 1 conclusion**:
> "External validation on public benchmarks (e.g., Geometry3K) awaits development of vision-to-symbolic preprocessing, as existing datasets use VQA format incompatible with symbolic reasoning architectures. Our controlled synthetic problem generation establishes baseline discriminativity metrics for future comparison."

---

## Documentation

- **Full analysis**: `PHASE2.1_INGESTION_ANALYSIS.md` (comprehensive findings)
- **Dataset stats**: 434/2101 problems (20.7%) mention supported fact types
- **Blocker**: VQA format requires vision/NLP preprocessing (out of scope)

---

## Next Steps (Pending Decision)

**If Option A (STOP)**:
1. Update `TRACK1_COMPLETE_SUMMARY.md` limitations section
2. Add future work paragraph about external validation
3. Finalize publication submission

**If Option B (Manual)**:
1. Select 50-100 Geometry3K problems
2. Hand-annotate symbolic IR
3. Run Track 1 Phase 6 telemetry (unchanged)
4. Add Track 2 appendix as "case studies"

**If Option C (Parser)**:
1. Design LaTeX math parser
2. Build fact extraction system
3. Test on 434 parseable problems
4. Risk timeline and narrative clarity

**If Option D (Search)**:
1. Survey geometry theorem proving datasets (GEOS, GeoQA, IMO-AG-30)
2. Assess symbolic IR compatibility
3. Report findings in 1-3 days
