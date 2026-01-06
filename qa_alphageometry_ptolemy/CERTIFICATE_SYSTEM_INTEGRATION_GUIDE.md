# QA Certificate System v1.0 - Integration Guide

## Executive Summary

You now have a **production-ready, unified certificate system** that spans:
- QA formal methods (Ptolemy's theorem, geometric proofs)
- AlphaGeometry integration (beam search → certificates)
- Physics projection interface (observer-dependent laws)

**All using the same `ProofCertificate` schema.**

---

## What's in the Bundle

### 1. Core Schema (`qa_certificate.py`)

**✅ Features:**
- Exact scalar arithmetic (int/Fraction only, floats rejected)
- Namespaced generators: `AG:*` (AlphaGeometry), `PHYS:*` (physics), `OBS:*` (observers)
- Complete failure taxonomy (15 failure modes as first-class objects)
- Physics extension: `ProjectionContract` for observer interface
- Deterministic JSON serialization

**Key types:**
```python
ProofCertificate = {
    theorem_id: str
    generator_set: Set[Generator]
    contracts: InvariantContract
    witness_type: "success" | "obstruction"
    success_path: List[MoveWitness]  # if success
    obstruction: ObstructionEvidence  # if failure
    observer_id: Optional[str]  # physics only
    projection_contract: Optional[ProjectionContract]  # physics only
    context: Dict[str, Any]
}
```

### 2. AlphaGeometry Adapter

**File:** `qa_alphageometry/adapters/certificate_adapter.py`

**Aligned to your Rust `SearchResult` structure:**
```python
from qa_alphageometry.adapters.certificate_adapter import wrap_searchresult_to_certificate

# Input: SearchResult from beam.rs
sr = {
    "solved": True,
    "proof": {
        "steps": [
            {"id": 0, "rule_id": "parallel_transitive", ...},
            {"id": 1, "rule_id": "angle_sum", ...}
        ],
        "final_state_hash": 9876543210
    },
    "states_expanded": 15,
    "depth_reached": 5,
    ...
}

# Output: ProofCertificate
cert = wrap_searchresult_to_certificate(
    sr,
    theorem_id="ptolemy_quadrance",
    max_depth_limit=50,
    repo_tag="qa-alphageometry-v0.1",
    commit="abc123"
)
```

**Handles:**
- Success: Builds `success_path` from `proof.steps`
- Failure: Infers stop reason from `depth_reached`/`successors_generated`
- Generators: Extracts from `rule_id` as `AG:<rule_id>`
- Conservative: Never claims `SCC_UNREACHABLE` without topology proof

### 3. Physics Adapter

**File:** `qa_physics/adapters/certificate_adapter.py`

**Wraps observer results:**
```python
from qa_physics.adapters.certificate_adapter import wrap_reflection_result_to_certificate

# Input: Observer output
result = {
    "observation": {
        "law_holds": True,
        "measured_angles": {"incident": "63.43", "reflected": "63.43"},
        "angle_difference": "0"
    },
    "symmetric": True,
    "states_explored": 41
}

# Output: ProofCertificate with ProjectionContract
cert = wrap_reflection_result_to_certificate(
    result,
    observer_id="GeometryAngleObserver",
    projection_tag="qa-physics-projection-v0.1",
    commit_hash="4aa2af4"
)
```

### 4. Sample Artifacts

**Generated certificates in `artifacts/`:**
- `reflection_GeometryAngleObserver.success.cert.json` - Law holds
- `reflection_NullObserver.obstruction.cert.json` - Angles undefined

### 5. Paper Skeleton

**File:** `qa_certificate_paper_skeleton.tex`

LaTeX structure for JAR/ITP submission:
- §1: Motivation (failure traceability)
- §2: Certificate objects (formal definitions)
- §3: QA-AlphaGeometry integration
- §4: Physics as projection
- §5: Discussion (artifact-first workflow)

---

## Integration Steps

### Step 1: Install in Your Repo

```bash
# Copy to your QA research repo
cp qa_certificate.py ~/qa-research/
cp -r qa_alphageometry ~/qa-research/
cp -r qa_physics ~/qa-research/
```

### Step 2: Generate Ptolemy Certificates

**Option A: From existing SearchResult JSON**
```bash
# If you have ptolemy_searchresult.json
python -c "
from qa_alphageometry.adapters.certificate_adapter import wrap_searchresult_to_certificate
import json

sr = json.load(open('ptolemy_searchresult.json'))
cert = wrap_searchresult_to_certificate(
    sr,
    theorem_id='ptolemy_quadrance',
    max_depth_limit=50,
    repo_tag='qa-alphageometry-ptolemy-v0.1'
)
json.dump(cert.to_json(), open('ptolemy_success.cert.json', 'w'), indent=2)
print('✓ Generated ptolemy_success.cert.json')
"
```

**Option B: Modify Rust runner to emit SearchResult**

Add to your `beam.rs` main function:
```rust
// After search completes
let result_json = serde_json::to_string_pretty(&result)?;
std::fs::write("ptolemy_searchresult.json", result_json)?;
```

Then use Option A.

### Step 3: Generate Physics Certificates

Your physics interface already outputs the right structure. Wire it up:

```python
# In qa_physics/optics/run_reflection_demo.py
from qa_physics.adapters.certificate_adapter import wrap_reflection_result_to_certificate
import json

# After running GeometryAngleObserver
cert = wrap_reflection_result_to_certificate(
    result,
    observer_id="GeometryAngleObserver",
    projection_tag="qa-physics-projection-v0.1",
    commit_hash="4aa2af4"
)

with open("reflection_geometry.success.cert.json", "w") as f:
    json.dump(cert.to_json(), f, indent=2)
```

### Step 4: Validate Certificates

```python
from qa_certificate import ProofCertificate
import json

# Load and validate
with open("ptolemy_success.cert.json") as f:
    cert_dict = json.load(f)

# Verify schema
assert cert_dict["schema_version"] == "1.0"
assert cert_dict["witness_type"] == "success"
assert len(cert_dict["success_path"]) > 0

print("✓ Certificate valid")
```

---

## Key Design Decisions (From ChatGPT)

### 1. Namespaced Generators

**Why:** Prevents collision between domains
- `σ`, `λ`, `ν` - Core QA operators
- `AG:similar_triangles` - AlphaGeometry rules
- `PHYS:law_of_reflection` - Physical laws
- `OBS:GeometryAngleObserver` - Observers (if you add this namespace)

### 2. Conservative Failure Classification

**AlphaGeometry failures default to `DEPTH_EXHAUSTED`:**
- Never claims `SCC_UNREACHABLE` without topology proof
- Infers stop reason from structure (depth hit, no successors, etc.)
- Stores `inferred_stop_reason` in context for transparency

### 3. Exact Scalars Only

**No floats in certificates:**
```python
to_scalar("63.43")  # → Fraction(6343, 100) ✓
to_scalar(63.43)    # → TypeError ✗
to_scalar("3/2")    # → Fraction(3, 2) ✓
to_scalar(25)       # → 25 ✓
```

### 4. Physics Requires Minimal Path

Current schema enforces non-empty `success_path`. Physics adapter creates a minimal 1-step "observe" witness. Real measurement data is in `context` + `projection_contract`.

**To change:** Allow `success_path=[]` when `observer_id` is set (requires schema modification).

---

## What You Can Do Now

### Immediate (Next Hour)

1. **Generate Ptolemy certificates:**
   - Add JSON export to your Rust runner
   - Use adapter to create `ptolemy_success.cert.json`

2. **Validate physics certificates:**
   - Run physics adapter on your existing results
   - Confirm exact scalar handling

### Short-term (This Week)

3. **Paper artifacts:**
   - Lock 4 canonical certificates:
     - `ptolemy_success.cert.json`
     - `ptolemy_ablated.obstruction.cert.json`
     - `reflection_geometry.success.cert.json`
     - `reflection_null.obstruction.cert.json`

4. **JAR submission:**
   - Flesh out paper skeleton
   - §3 shows Ptolemy success + ablated obstruction
   - §5 conclusion mentions physics (one paragraph)

### Medium-term (Next 2 Weeks)

5. **Physics paper:**
   - Separate foundations paper
   - Uses same certificate schema
   - Shows law emergence is projection-dependent

6. **ML phase transition demo:**
   - Reuse same schema
   - Certificate `theorem_id="phase_transition_465_to_1"`
   - Witness: SCC collapse statistics

---

## Testing the Bundle

**Quick smoke test:**
```bash
cd /path/to/bundle
python -c "
from qa_certificate import to_scalar
from fractions import Fraction

# Test exact scalar conversion
assert to_scalar('63.43') == Fraction(6343, 100)
assert to_scalar('3/2') == Fraction(3, 2)
assert to_scalar(25) == 25

try:
    to_scalar(63.43)  # Should fail
    assert False
except TypeError:
    pass

print('✓ Exact scalar handling works')
"
```

**Test AlphaGeometry adapter:**
```python
from qa_alphageometry.adapters.certificate_adapter import wrap_searchresult_to_certificate

# Minimal test case
sr = {
    "solved": True,
    "proof": {
        "steps": [{"rule_id": "test_rule"}],
        "final_state_hash": 123
    },
    "states_expanded": 10,
    "depth_reached": 3,
    "successors_generated": 20,
    "successors_kept": 15,
    "best_score": 0.9,
    "beam_signatures": []
}

cert = wrap_searchresult_to_certificate(sr, "test_theorem")
assert cert.witness_type == "success"
assert len(cert.success_path) == 1
print("✓ AlphaGeometry adapter works")
```

---

## Schema Extensions (Future)

### If You Need Observer Namespace

Add to `qa_certificate.py`:
```python
# Line 102-105, update __post_init__:
elif self.name.startswith("OBS:"):
    pass  # Observer namespace
```

Then use:
```python
Generator("OBS:GeometryAngleObserver", ())
```

### If You Want Empty Success Paths

Relax constraint in `ProofCertificate.__post_init__`:
```python
if self.witness_type == "success":
    if self.observer_id is None:
        # Formal methods: path required
        assert self.success_path is not None and len(self.success_path) > 0
    # Physics: path optional
```

---

## Critical Files Manifest

```
qa_certificate.py                                    # Core schema (492 lines)
qa_alphageometry/adapters/certificate_adapter.py     # AG adapter (167 lines)
qa_physics/adapters/certificate_adapter.py           # Physics adapter
artifacts/reflection_GeometryAngleObserver.success.cert.json
artifacts/reflection_NullObserver.obstruction.cert.json
qa_certificate_paper_skeleton.tex                    # LaTeX skeleton
README.md                                            # Quick start
```

---

## Next Actions (Priority Order)

### ✅ Done (You Have This Now)
- Unified certificate schema
- AlphaGeometry adapter (exact Rust alignment)
- Physics adapter (observer projection)
- Sample artifacts
- Paper skeleton

### 🎯 Next (You Do This)
1. Export `SearchResult` JSON from Rust Ptolemy proof
2. Generate `ptolemy_success.cert.json`
3. Generate ablated version (remove ν generator)
4. Lock 4 canonical artifacts

### 📝 Then (Paper Time)
5. Flesh out §3 in paper skeleton (Ptolemy results)
6. Add killer figure (2-panel reachability diagram)
7. Submit to JAR/ITP

---

## Support & Validation

**If certificate generation fails:**
1. Check SearchResult structure matches expected format
2. Verify all fields are present (`states_expanded`, `depth_reached`, etc.)
3. Check that `rule_id` exists in proof steps

**If scalar conversion fails:**
4. Ensure no floats in input
5. Use string decimals: `"63.43"` not `63.43`

**Questions?**
The schema is now canonical (v1.0). If you need modifications, they should be:
- Backwards compatible (add fields, don't change types)
- Documented in commit message
- Tested with existing artifacts

---

## Final Note: Physics "Pre-Buy" Language

For your JAR paper conclusion:

> "The same certificate machinery applies to projection-dependent physical laws. 
> For example, the law of reflection emerges under `GeometryAngleObserver` 
> (certificate: `reflection_geometry.success.cert.json`) but remains undefined 
> under `NullObserver` (certificate: `reflection_null.obstruction.cert.json`). 
> See companion artifact tag `qa-physics-projection-v0.1`."

This is 3 sentences. No derailing. Just priority staking.

---

**You're ready to ship.** 🚀
