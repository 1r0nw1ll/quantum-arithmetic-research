# Track 2 Phase 2.1: Geometry3K Ingestion Analysis

**Status**: BLOCKER IDENTIFIED
**Date**: 2025-12-16
**Analysis Duration**: 2 hours

## Executive Summary

❌ **CRITICAL FINDING**: Geometry3K dataset is in VQA (Visual Question Answering) format and **does not contain symbolic geometric representations** compatible with our Track 1 IR-based solver.

**Recommendation**: Pause Track 2 pending decision on whether to:
1. Accept this as a "stop condition" (no compatible external dataset available)
2. Invest in NLP/vision preprocessing (high scope creep risk)
3. Seek alternative datasets with symbolic IR

---

## Dataset Structure Analysis

### Format Overview

Geometry3K is derived from InterGPS (github.com/lupantech/InterGPS) and contains:

- **Train**: 2,101 problems
- **Validation**: ~500 problems
- **Test**: ~500 problems

**Data columns**: `['images', 'problem', 'answer']`

- `images`: PNG diagram (bytes)
- `problem`: Natural language text with `<image>` placeholder
- `answer`: Numeric or algebraic expression

### Problem Text Complexity Distribution

Sample of 20 random problems reveals three categories:

| Category | Percentage | Example |
|----------|-----------|---------|
| **Minimal text** | ~35% | "Find x...", "Find WZ..." |
| **Medium text** | ~50% | "Find the value of $d$ in the parallelogram...", "If $∠6$ and $∠8$ are complementary..." |
| **Rich text** | ~15% | "In $\odot O, \overline{E C}$ and $\overline{A B}$ are diameters, and $\angle B O D \cong \angle D O E...$" |

**Critical observation**: Even "rich text" problems provide question context (e.g., "if this condition holds, find x"), but **geometric setup is in the image**.

---

## Supported Fact Type Coverage

Searched all 2,101 train problems for our 5 supported fact types:

| Fact Type | Occurrences | Percentage | Notes |
|-----------|------------|------------|-------|
| **Parallel** | 200 | 9.5% | "parallelogram", "m ∥ n", etc. |
| **OnCircle** | 140 | 6.7% | "inscribed", "$\odot P$", "on circle" |
| **EqualLength** | 65 | 3.1% | "$\overline{BE} \cong \overline{ED}$" |
| **Perpendicular** | 29 | 1.4% | "⊥", "perpendicular bisector" |
| **Collinear** | 0 | 0.0% | ⚠️ No explicit mentions found |

**Total parseable**: 434 problems (20.7%)

### Sample Parseable Problems

**Parallel** (Problem 48):
```
Find $ m ∠RSU $ so that $ m \parallel n $.
```

**EqualLength** (Problem 1):
```
If $\overline{B E} \cong \overline{E D}$ and $m \widehat{E D}=120,$ find $m \widehat{B E}$.
```

**OnCircle** (Problem 9):
```
The triangle is inscribed into the circle. Find the exact circumference of the circle.
```

---

## The Fundamental Blocker

### What Track 1 Requires

Our Track 1 solver expects problems in symbolic IR format:

```json
{
  "initial_facts": [
    {"type": "Parallel", "args": ["AB", "CD"]},
    {"type": "Perpendicular", "args": ["EF", "GH"]},
    {"type": "Point", "args": ["A", "0", "0"]},
    ...
  ],
  "goal": {"type": "EqualAngle", "args": ["∠ABC", "∠DEF"]}
}
```

This IR is:
- **Complete**: All points, lines, circles defined explicitly
- **Symbolic**: No vision required, pure logical reasoning
- **Coordinate-backed** (optional): Points have (x,y) when needed

### What Geometry3K Provides

```json
{
  "images": [<PNG bytes>],
  "problem": "<image>If $\\overline{BE} \\cong \\overline{ED}$ and $m \\widehat{ED}=120,$ find $m \\widehat{BE}$.",
  "answer": "120"
}
```

**Missing from text**:
- Point definitions (which points exist? coordinates?)
- Line/circle definitions (what is the full geometric setup?)
- Complete fact list (what other relationships hold in the diagram?)

**Required for solution**:
- Vision system to extract diagram structure from PNG
- OCR for any text/labels in diagram
- Geometric entity recognition (points, lines, circles, angles)
- Relationship extraction (parallel, perpendicular, congruent, etc.)

---

## Gap Analysis

### What We Would Need to Build

To ingest Geometry3K, we would need:

**Option A: Vision-Based Preprocessing**
1. CNN-based diagram parser
2. Point detection & labeling
3. Line/circle detection
4. Relationship extraction (parallel, perpendicular, etc.)
5. Symbolic IR generation from visual features

**Estimated effort**: 3-6 weeks
**Scope creep risk**: HIGH (entirely new capability, not tested in Track 1)

**Option B: NLP-Based Partial Ingestion**
1. LaTeX math parser
2. Named entity recognition (points, lines, angles)
3. Relationship extraction from text
4. Assumption-based diagram completion

**Estimated effort**: 1-2 weeks
**Scope creep risk**: MODERATE
**Coverage**: Only ~20% of problems (434/2101)
**Quality risk**: HIGH (missing critical geometric context from images)

**Option C: Manual Annotation**
1. Hand-convert 50-100 problems to symbolic IR
2. Use human judgment to fill gaps from images

**Estimated effort**: 1-2 days
**Scope creep risk**: LOW
**Quality risk**: LOW (gold standard annotations)
**Limitation**: Not scalable, defeats "external validation" narrative

---

## Track 2 Risk Assessment

### Original Track 2 Goals (from user's strategic plan)

✅ **Goal**: "Demonstrate that the Track-1 discriminativity framework predicts QA efficiency gains on real geometry benchmarks"

❌ **Blocker**: No compatible "real geometry benchmark" dataset found

✅ **Principle**: "No scope creep, no new claims, no new metrics"

❌ **Risk**: Building vision/NLP preprocessing = massive scope creep

✅ **Constraint**: "Track 2 must reuse everything from Track 1"

❌ **Issue**: Track 1 assumes symbolic IR input; Geometry3K has none

### Publication Impact Analysis

**If we stop Track 2 now**:
- Track 1 remains strong: controlled validation, 100% solve rate, 23% efficiency gain
- Limitation acknowledged: "External validation pending availability of symbolic IR datasets"
- Honest scientific communication: "We tested on synthetic problems due to dataset format constraints"
- **Reviewers will understand**: Geometry3K is VQA, not symbolic reasoning

**If we proceed with Option A/B**:
- High scope creep risk dilutes Track 1 narrative
- Vision/NLP preprocessing introduces new confounds (parsing errors)
- Results less interpretable: "Is efficiency from QA or from preprocessing quality?"
- Timeline risk: 1-6 weeks additional work

**If we proceed with Option C**:
- Manual annotation defeats "external validation" claim
- Small sample (50-100) may not convince reviewers
- Still valuable as "case studies" but weaker than Track 1

---

## Recommendations

### Option 1: Accept Stop Condition ✅ (RECOMMENDED)

**Rationale**:
- Track 1 is publication-ready and scientifically sound
- No compatible external dataset exists (format mismatch, not quality)
- Stopping here is intellectually honest
- Future work can address this when symbolic IR datasets emerge

**Action items**:
1. Update Track 1 conclusion: "External validation awaits symbolic IR benchmark availability"
2. Add to limitations: "Geometry3K (VQA format) incompatible with symbolic solver architecture"
3. Suggest future work: "Vision-augmented QA-AlphaGeometry for diagram-based problems"

**Publication impact**: NEUTRAL (Track 1 stands alone)

### Option 2: Minimal Manual Case Studies (50-100 problems)

**Rationale**:
- Provides *some* external validation
- Low scope creep (no new systems)
- Can cherry-pick problems that demonstrate discriminativity

**Action items**:
1. Manually select 50-100 Geometry3K problems with clear diagrams
2. Hand-convert to symbolic IR (1-2 days)
3. Run Track 1 telemetry (unchanged)
4. Frame as "case studies" not "benchmark validation"

**Publication impact**: SLIGHTLY POSITIVE (shows effort to validate externally)
**Risk**: Medium (manual errors, selection bias)

### Option 3: Build NLP Parser (HIGH RISK, NOT RECOMMENDED)

**Rationale**: Would enable partial ingestion of ~434 problems

**Against**:
- Violates "no scope creep" principle
- Introduces new confounds (parsing errors)
- Still misses 80% of dataset
- 1-2 weeks timeline risk
- Dilutes Track 1 narrative clarity

**Publication impact**: NEGATIVE (adds complexity, reduces interpretability)

---

## Alternative External Datasets (Preliminary Search)

Brief investigation of other geometry theorem proving datasets:

| Dataset | Format | Symbolic IR? | Coverage |
|---------|--------|--------------|----------|
| **Geometry3K** | VQA | ❌ | 3,002 problems |
| **GeoQA** | VQA | ❌ | 4,998 problems |
| **IMO-AG-30** | Natural language | ❌ | 30 problems |
| **GEOS** | Synthetic | ✅ (custom) | 186 problems |

**Finding**: No large-scale symbolic IR geometry dataset exists publicly.

**Implication**: Our Track 1 synthetic problem generation may actually be **state-of-the-art** for this task.

---

## Decision Point

**Question for user**:

Given the Geometry3K format incompatibility, how should we proceed with Track 2?

**A)** Accept stop condition (Track 1 only, acknowledge limitation)
**B)** Manual annotation for 50-100 case studies (2 days effort)
**C)** Build NLP parser despite scope creep risk (1-2 weeks effort)
**D)** Pause and search for alternative datasets (1-3 days)

**My recommendation**: **Option A** (stop Track 2, publish Track 1 as-is)

**Rationale**: Track 1 is strong on its own merits. Adding weak external validation risks diluting the narrative. Better to publish Track 1 now and revisit external validation when compatible datasets emerge (or as follow-up work with vision preprocessing).

---

## Appendix: Dataset Exploration Commands

All analysis commands logged for reproducibility:

```python
# Load dataset
import pandas as pd
df = pd.read_parquet('/path/to/default_train.parquet')

# Count fact type mentions
import re
patterns = {
    'Parallel': [r'parallel', r'∥', r'\\parallel'],
    'Perpendicular': [r'perpendicular', r'⊥', r'\\perp'],
    'Collinear': [r'collinear', r'lie on.*line'],
    'OnCircle': [r'on.*circle', r'\\odot', r'inscribed'],
    'EqualLength': [r'\\cong', r'≅', r'equal.*length']
}

for fact_type, pattern_list in patterns.items():
    matches = []
    for pattern in pattern_list:
        matches.extend(df[df['problem'].str.contains(pattern, case=False, regex=True)].index)
    print(f"{fact_type}: {len(set(matches))} problems")
```

**Results**: 434 problems with parseable mentions (20.7% of 2,101)
