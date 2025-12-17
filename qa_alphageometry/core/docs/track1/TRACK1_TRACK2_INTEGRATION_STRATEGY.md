# Track 1 + Track 2 Integration Strategy

## Overview

We have two complementary testing approaches:

**Track 1 (Discriminative Synthetic)**: Custom-designed problems to test QA heuristic effectiveness
**Track 2 (Real-World Validation)**: Geometry3K dataset for generalization testing

## Track 1: Discriminative Synthetic Problems

### Status: Phase 7 Complete ✅

**Validated Problems:**
- T02: Scaled multisurface (8 rule families, fact_volume=47)
- T03: Mega discrimination (8 rule families, fact_volume=103)

**Validation Results:**
- Both problems solved by QA=0 and QA=0.7
- Beam divergence detected at depth 0
- Different search paths confirmed

**Next Steps:**
1. Generate remaining Family T problems (t04-t10)
2. Generate Family S problems (s01-s10) - perpendicular lattices
3. Generate Family C problems (c01-c10) - coordinate-derived
4. Run Phase 6 telemetry on all 30 problems
5. Analyze QA efficiency metrics

**Purpose:** Controlled testing of QA prior under known discriminative conditions

---

## Track 2: Geometry3K Real-World Dataset

### Dataset Details

**Source:** https://github.com/lupantech/InterGPS
**Location:** `/home/player2/signal_experiments/qa_alphageometry/data/`

**Splits:**
- Train: 2,101 problems (default_train.parquet)
- Validation: 300 problems (default_validation.parquet)
- Test: 601 problems (default_test.parquet)
- **Total: 3,002 problems**

**Format:**
```python
{
  "images": [image],          # PNG diagram
  "problem": "<image>text",   # Natural language problem statement
  "answer": "choice"          # Multiple choice answer
}
```

**Task Type:** Visual question answering with geometry diagrams

### Integration Challenges

The Geometry3K dataset is in a different format than our symbolic IR:

**Current Format:**
- Natural language problem text
- PNG diagram images
- Multiple choice answers

**Required Format:**
- Structured facts (Parallel, Perpendicular, etc.)
- Symbolic goal representation
- Theorem proving (not multiple choice)

### Track 2 Implementation Options

**Option 1: Use as Baseline Comparison**
- Run Geometry3K through original AlphaGeometry (if available)
- Compare solve rates with/without QA prior
- Focus on problems that AlphaGeometry can already parse

**Option 2: Create Parquet→IR Parser**
- Extract structured facts from natural language using LLM
- Convert diagram coordinates to symbolic representation
- Transform multiple choice to proof goals
- **Effort:** High (requires NLP + vision)

**Option 3: Use Subset with Manual Annotation**
- Manually convert 30-50 representative problems to IR format
- Create structured fact annotations
- Use as real-world test set alongside Track 1
- **Effort:** Medium (manual labor)

**Option 4: Defer to Future Work**
- Focus on Track 1 for initial publication
- Add Geometry3K integration in follow-up work
- Document as "planned extension"
- **Effort:** None (immediate)

---

## Recommended Strategy

### Short-Term (Week 4 completion)

**Focus on Track 1:**
1. ✅ Complete Phase 7 validation (DONE)
2. Generate remaining 28 synthetic problems (Families S, T, C)
3. Run Phase 6 telemetry on all 30 problems
4. Analyze and visualize results
5. Document findings for publication

**Track 2 Baseline:**
1. Document Geometry3K dataset availability
2. Note as future work in paper
3. Reserve for follow-up publication if Track 1 shows strong results

### Long-Term (Post-publication)

**Track 2 Full Integration:**
1. Develop Geometry3K→IR parser
   - NLP-based fact extraction
   - Coordinate-to-symbolic conversion
   - Goal synthesis from multiple choice
2. Run full 3,002 problem benchmark
3. Compare QA vs baseline on real-world generalization
4. Publish as extension paper

---

## Dual-Track Architecture Benefits

### Scientific Rigor

**Track 1 (Synthetic):**
- ✅ Controlled variables
- ✅ Known discriminative properties
- ✅ Direct QA hypothesis testing
- ✅ Reproducible results

**Track 2 (Real-World):**
- ✅ Generalization validation
- ✅ Diverse problem types
- ✅ Established benchmark
- ✅ Comparison with prior work

### Publication Strategy

**Initial Paper (Track 1 Focus):**
- Title: "Quantum Arithmetic Priors for Discriminative Geometric Theorem Proving"
- Main contribution: Rule-batch discriminativity framework
- Validation: 30 controlled synthetic problems
- Future work: Geometry3K integration

**Extension Paper (Track 2):**
- Title: "Scaling Quantum Arithmetic Priors to Real-World Geometry Problems"
- Main contribution: NLP→IR parser for Geometry3K
- Validation: 3,002 real-world problems
- Comparison: QA vs AlphaGeometry baseline

---

## Implementation Priority

### Immediate (This Week)
1. ✅ Phase 7 validation complete
2. Generate Family S problems (s01-s10)
3. Generate Family T problems (t04-t10)
4. Generate Family C problems (c01-c10)
5. Run telemetry on all 30 problems

### Near-Term (Next 2 Weeks)
1. Analyze Track 1 results
2. Create plots and visualizations
3. Write Track 1 paper sections
4. Document Track 2 dataset

### Long-Term (Post-submission)
1. Build Geometry3K→IR parser
2. Run Track 2 benchmark
3. Prepare extension paper

---

## Files and Locations

### Track 1 Files
- Validated problems: `tests/fixtures/problems/synthetic/t02_*.json`, `t03_*.json`
- Generator: `scripts/generate_rulebatch_problems.py`
- Scorer: `scripts/branching_score.py`
- Validation tests: `tests/track1_rulebatch_validation.rs`
- Documentation: `TRACK1_PHASE7_*.md`

### Track 2 Files
- Dataset location: `/home/player2/signal_experiments/qa_alphageometry/data/`
- Train: `default_train.parquet` (2,101 problems)
- Validation: `default_validation.parquet` (300 problems)
- Test: `default_test.parquet` (601 problems)
- Metadata: `README.md`, `dataset_metadata.md`

---

## Decision Point

**Immediate Action Required:**

Should we:
1. **Focus exclusively on Track 1** (30 synthetic problems) for Week 4 completion? ✅ RECOMMENDED
2. **Add Track 2 baseline** (manual conversion of 30 Geometry3K problems)? ⏸️ DEFER
3. **Build full Track 2 parser** (NLP-based automatic conversion)? ⏸️ FUTURE WORK

**Recommendation:** Option 1 - Complete Track 1 with full 30-problem suite. Track 2 provides excellent future work but would delay Week 4 completion.

---

## Summary

✅ **Track 1 (Immediate):** Generate 28 more synthetic problems, run telemetry, publish results
📊 **Track 2 (Future):** Build Geometry3K→IR parser, run 3,002-problem benchmark
🎯 **Strategy:** Dual-track approach balances rigor (synthetic) with generalization (real-world)

**Status:** Track 1 Phase 7 complete. Ready to scale up to 30 problems.
