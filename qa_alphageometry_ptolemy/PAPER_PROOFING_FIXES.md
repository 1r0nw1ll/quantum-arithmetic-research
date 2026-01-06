# Paper-Proofing Fixes - 2026-01-06

**Based on ChatGPT's detailed code review of physics certificates**

---

## Issues Identified and Fixed

### 1. fixed_q_mode: {} vs null Consistency ✅

**Issue:** Physics certificates had `"fixed_q_mode": {}` (empty object) instead of `null` when unused.

**Root cause:** In `InvariantContract.to_dict()`, line 330:
```python
# BEFORE (incorrect):
"fixed_q_mode": {k: str(to_scalar(v)) for k, v in (self.fixed_q_mode or {}).items()},
# This converts None → {} → empty dict
```

**Fix:**
```python
# AFTER (correct):
"fixed_q_mode": ({k: str(to_scalar(v)) for k, v in self.fixed_q_mode.items()}
                if self.fixed_q_mode is not None else None),
# This preserves None → null in JSON
```

**Impact:**
- AlphaGeometry adapter already used `None` correctly
- Physics adapter already used `None` correctly
- Only the serialization was wrong

**Verification:**
```python
GeometryAngleObserver: fixed_q_mode: null ✓
NullObserver:          fixed_q_mode: null ✓
```

---

### 2. Generator Set Consistency ✅

**Issue:** `generator_set` contained `PHYS:reflection_probe` but `success_path[0].gen.name` was `PHYS:observe` - inconsistent.

**Semantics clarified:** `generator_set` should include **all generators that may appear in witness steps**, not just "available generators."

**Fix:** Added both to generator_set:
```python
generator_set: Set[Generator] = {
    Generator("PHYS:reflection_probe", ()),  # Theorem context
    Generator("PHYS:observe", ()),           # Actual operation in success_path
}
```

**Verification:**
```python
GeometryAngleObserver:
  generator_set: ['PHYS:observe', 'PHYS:reflection_probe']
  witness gen:    PHYS:observe
  Consistency: ✓ witness generator in generator_set
```

---

## Physics Certificates Regenerated

Both certificates regenerated with corrected adapter:

### artifacts/reflection_GeometryAngleObserver.success.cert.json
```json
{
  "generator_set": [
    {"name": "PHYS:observe", "params": []},
    {"name": "PHYS:reflection_probe", "params": []}
  ],
  "contracts": {
    "non_reduction_enforced": true,
    "fixed_q_mode": null  // ← WAS {}, NOW null
  },
  "success_path": [
    {
      "gen": {"name": "PHYS:observe", ...}  // ← NOW in generator_set
    }
  ]
}
```

**Key result:** θ_i = θ_r = 63.43° (perfect reflection under GeometryAngleObserver)

### artifacts/reflection_NullObserver.obstruction.cert.json
```json
{
  "generator_set": [
    {"name": "PHYS:observe", "params": []},
    {"name": "PHYS:reflection_probe", "params": []}
  ],
  "contracts": {
    "non_reduction_enforced": true,
    "fixed_q_mode": null  // ← WAS {}, NOW null
  },
  "obstruction": {
    "fail_type": "observer_undefined"  // ← Angles undefined
  }
}
```

**Key result:** Law undefined because NullObserver can't compute angles

---

## Demonstration of "Projection-Dependent Law"

These two certificates cleanly show:

1. **Success (GeometryAngleObserver):**
   - Measures equal incident/reflected angles (63.43° / 63.43°)
   - Sets `law_holds: true`
   - Law **emerges** from projection

2. **Obstruction (NullObserver):**
   - Cannot compute angles
   - `fail_type: "observer_undefined"`
   - Law **undefined** without appropriate observer

**This is exactly the story for the paper:** Laws are projection properties, not QA substrate properties.

---

## Semantic Consistency Achieved

### fixed_q_mode

| Adapter | Value | Serializes To | Correct? |
|---------|-------|---------------|----------|
| AlphaGeometry | `None` | `null` | ✅ |
| Physics | `None` | `null` | ✅ (was `{}`) |

### generator_set

| Certificate | Contains | Used in Path | Consistent? |
|-------------|----------|--------------|-------------|
| AG (future) | `AG:<rules>` | `AG:<rules>` | ✅ |
| Physics | `PHYS:observe`, `PHYS:reflection_probe` | `PHYS:observe` | ✅ (was incomplete) |

---

## Files Modified

1. **qa_certificate.py** (line 330)
   - Fixed `fixed_q_mode` serialization to preserve `None` → `null`

2. **qa_physics/adapters/certificate_adapter.py** (lines 68-72)
   - Added both `PHYS:observe` and `PHYS:reflection_probe` to generator_set
   - Clarified semantics with comment

3. **artifacts/reflection_*.cert.json** (2 files)
   - Regenerated with corrected adapter
   - Now consistent with schema v1.0+

---

## Verification

### Consistency Checks ✅

```bash
✓ fixed_q_mode serializes to null (not {})
✓ generator_set includes all generators in witness path
✓ Both physics certificates valid
✓ Schema version 1.0 consistent across all artifacts
```

### Test Results ✅

```python
GeometryAngleObserver Success:
  fixed_q_mode: None ✓
  generator_set: ['PHYS:observe', 'PHYS:reflection_probe'] ✓
  witness gen in generator_set: ✓

NullObserver Obstruction:
  fixed_q_mode: None ✓
  generator_set: ['PHYS:observe', 'PHYS:reflection_probe'] ✓
  fail_type: observer_undefined ✓
```

---

## Impact on Paper

### Strengthens Story ✅

The regenerated certificates **more clearly demonstrate**:

1. **Projection-dependent laws:** Only GeometryAngleObserver computes angles
2. **Null model validation:** NullObserver proves angles are projection-added
3. **Schema consistency:** All fields follow same semantics

### Reviewer Confidence ✅

- Consistent `fixed_q_mode` handling across adapters
- Clear generator_set semantics (all used generators included)
- No ambiguity about "available" vs "used" generators

---

## Next Steps (Unchanged)

**Still blocked on:** Export Ptolemy SearchResult JSON from Rust

**When ready:**
```bash
python3 generate_ptolemy_certificates.py --in ptolemy_searchresult.json
```

**Timeline:** 2-3 weeks to JAR/ITP submission

---

## Summary

✅ **fixed_q_mode** - Now correctly serializes `None` → `null` (not `{}`)
✅ **generator_set** - Now includes all generators used in witness path
✅ **Physics certificates** - Regenerated with corrected semantics
✅ **Schema consistency** - All adapters follow same patterns
✅ **Paper story** - Projection-dependent law demonstration cleaner
✅ **Backward compatible** - Only affects how None is serialized

**Status:** Physics certificates are now production-ready and reviewer-proof.

---

**Credits:** Paper-proofing review by ChatGPT o1
