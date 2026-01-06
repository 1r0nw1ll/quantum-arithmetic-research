# Certificate System Corrections - 2026-01-06

**Based on ChatGPT feedback after bundle verification**

---

## Corrections Applied

### 1. Added OBS:* Namespace Support ✅

**Issue:** Generator class only supported `AG:*` and `PHYS:*` namespaces. The `OBS:*` namespace (for observers like GeometryAngleObserver) was mentioned in documentation but not actually implemented.

**Fix:**
```python
# In qa_certificate.py Generator.__post_init__()
elif self.name.startswith("OBS:"):
    pass
```

**Updated docstring:**
```python
Namespaces:
  - QA core: "σ", "λ", "μ", "ν", "σ_inv", "λ_inv"
  - AlphaGeometry rule: "AG:<rule_id>"
  - Physics/projection: "PHYS:<thing>"
  - Observer/measurement: "OBS:<observer_id>"  # NEW
```

**Verification:**
```bash
✓ OBS:* namespace working: OBS:GeometryAngleObserver
✓ PHYS:* namespace working: PHYS:law_of_reflection
✓ AG:* namespace working: AG:similar_triangles
✓ Core QA working: σ
✓ Invalid namespace rejected correctly
```

---

### 2. Fixed non_reduction_enforced for AlphaGeometry ✅

**Issue:** AlphaGeometry adapter was setting `non_reduction_enforced=True`, which is misleading. AlphaGeometry doesn't operate under QA's Non-Reduction Axiom (it uses its own geometric algebra).

**Fix:**
```python
# In qa_alphageometry/adapters/certificate_adapter.py
contracts = InvariantContract(
    tracked_invariants=tracked_invariants or [],
    non_reduction_enforced=False,  # AG uses its own algebra (not QA non-reduction axiom)
    fixed_q_mode=None
)
```

**Rationale:** The flag should reflect actual operational constraints. AG certificates now accurately indicate that they don't enforce QA's non-reduction constraint. The explanatory comment makes this transparent.

**Verification:**
```bash
AlphaGeometry adapter test:
  non_reduction_enforced: False
  ✓ Correctly set to False (AG uses its own algebra)
```

---

## Bundle Clarification

**Canonical source:** `files(17).zip`
- Contains complete system (schema, adapters, docs)
- Clean source files (no .pyc artifacts)

**Ignore:** `qa_certificate_bundle.zip`
- Partial export with `__pycache__/` artifacts
- Not source-of-truth

---

## Documentation Corrections

### Walkthrough Numbers are Illustrative

The walkthrough shows example outputs like:
- "Proof length: 7 steps"
- "States explored: 245"
- "Depth reached: 7"

These are **illustrative expectations**, not actual results. Real numbers will come from exporting the actual Ptolemy SearchResult JSON.

### Critical Path Unchanged

**Still blocked on:** Exporting Ptolemy SearchResult JSON from Rust

**Once exported:**
```bash
python3 generate_ptolemy_certificates.py --in ptolemy_searchresult.json
```

This remains the one-command solution.

---

## Backward Compatibility

### Existing Certificates Unaffected

Both corrections are **additive** or **clarifying**:

1. **OBS:* addition:** New namespace doesn't break existing certificates
2. **non_reduction_enforced fix:** Physics adapter already had correct value (`True`), only AG adapter changed to `False`

**Existing physics artifacts still valid:**
- `artifacts/reflection_GeometryAngleObserver.success.cert.json` ✅
- `artifacts/reflection_NullObserver.obstruction.cert.json` ✅

These use `non_reduction_enforced=True` correctly (QA physics respects the axiom).

---

## Testing

### All Tests Pass

**Schema validation:**
```bash
✓ Schema loaded successfully
✓ Generator test: σ
✓ Scalar test: 6343/100 (exact rational)
```

**Namespace validation:**
```bash
✓ QA core: σ, λ, μ, ν
✓ AG:* (AlphaGeometry rules)
✓ PHYS:* (physical laws)
✓ OBS:* (observers) [NEW]
✓ Invalid namespaces rejected
```

**Adapter validation:**
```bash
✓ AlphaGeometry: non_reduction_enforced=False
✓ Physics: non_reduction_enforced=True
✓ Both adapters produce valid certificates
```

---

## Next Steps (Unchanged)

1. **Export Ptolemy SearchResult JSON** from Rust beam search
2. **Generate certificate** with one command
3. **Validate structure**
4. **Draft paper §3** with results

**Timeline:** Still 2-3 weeks to JAR/ITP submission

---

## Files Modified

1. `qa_certificate.py` (line 107: added `OBS:*` branch)
2. `qa_alphageometry/adapters/certificate_adapter.py` (line 74: changed to `False`)

**Total impact:** 2 lines changed, critical namespace added, semantic correctness improved.

---

## Summary

✅ **OBS:* namespace** - Now actually implemented (was only documented)
✅ **non_reduction_enforced** - Correctly reflects AlphaGeometry's algebra
✅ **Backward compatible** - Existing certificates unaffected
✅ **All tests passing** - Verified with actual code execution
✅ **Critical path unchanged** - Still need Ptolemy SearchResult export

**Status:** System remains production-ready with improved accuracy.

---

**Credits:** Corrections based on ChatGPT o1 code review of bundle exports
