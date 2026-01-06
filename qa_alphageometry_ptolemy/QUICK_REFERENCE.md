# QA Certificate System - Quick Reference

## One-Line Commands

### Generate AlphaGeometry Certificate
```python
from qa_alphageometry.adapters.certificate_adapter import wrap_searchresult_to_certificate
import json

sr = json.load(open("searchresult.json"))
cert = wrap_searchresult_to_certificate(sr, "theorem_name", max_depth_limit=50)
json.dump(cert.to_json(), open("output.cert.json", "w"), indent=2)
```

### Generate Physics Certificate
```python
from qa_physics.adapters.certificate_adapter import wrap_reflection_result_to_certificate
import json

result = {"observation": {"law_holds": True, "measured_angles": {"incident": "63.43", "reflected": "63.43"}}}
cert = wrap_reflection_result_to_certificate(result, "GeometryAngleObserver", projection_tag="qa-physics-projection-v0.1")
json.dump(cert.to_json(), open("physics.cert.json", "w"), indent=2)
```

### Validate Certificate
```python
import json
cert = json.load(open("certificate.json"))
assert cert["schema_version"] == "1.0"
assert cert["witness_type"] in ("success", "obstruction")
```

## Schema Cheat Sheet

### Success Certificate Structure
```json
{
  "schema_version": "1.0",
  "theorem_id": "...",
  "generator_set": [{"name": "AG:rule", "params": []}],
  "witness_type": "success",
  "success_path": [
    {"gen": {...}, "src_id": "...", "dst_id": "...", "packet_delta": {}}
  ],
  "search": {"max_depth": 50, "states_explored": 100}
}
```

### Obstruction Certificate Structure
```json
{
  "schema_version": "1.0",
  "theorem_id": "...",
  "witness_type": "obstruction",
  "obstruction": {
    "fail_type": "depth_exhausted",
    "max_depth_reached": 50,
    "states_explored": 1000
  }
}
```

### Physics Certificate (Success)
```json
{
  "witness_type": "success",
  "observer_id": "GeometryAngleObserver",
  "projection_contract": {
    "observer_id": "GeometryAngleObserver",
    "time_projection": "discrete: t = k (path length)",
    "preserves_topology": true,
    "continuous_observables": ["theta_incident_deg", "theta_reflected_deg"]
  },
  "context": {
    "measured_angles_deg": {"incident": "63.43", "reflected": "63.43"}
  }
}
```

## Generator Namespaces

| Prefix | Domain | Example |
|--------|--------|---------|
| (none) | Core QA | `σ`, `λ`, `ν` |
| `AG:` | AlphaGeometry | `AG:similar_triangles` |
| `PHYS:` | Physical laws | `PHYS:law_of_reflection` |
| `OBS:` | Observers | `OBS:GeometryAngleObserver` |

## Fail Types Quick Reference

| Type | Use When |
|------|----------|
| `DEPTH_EXHAUSTED` | Search hit max depth |
| `GENERATOR_INSUFFICIENT` | More generators would solve it |
| `SCC_UNREACHABLE` | Topology proven disconnected |
| `INVARIANT_VIOLATION` | Move breaks tracked invariant |
| `OBSERVER_UNDEFINED` | Observer can't compute observable |
| `LAW_VIOLATION` | Law fails under observer |

## Common Patterns

### Ptolemy Success Certificate
```python
# From Rust SearchResult with solved=True
cert = wrap_searchresult_to_certificate(
    sr, "ptolemy_quadrance",
    repo_tag="qa-alphageometry-ptolemy-v0.1"
)
# → success_path has proof steps
# → generator_set has AG:<rule_id> entries
```

### Ptolemy Ablated (No ν)
```python
# From SearchResult with solved=False (restricted generators)
cert = wrap_searchresult_to_certificate(
    sr, "ptolemy_quadrance_no_nu",
    max_depth_limit=50
)
# → obstruction with fail_type="depth_exhausted"
# → context["inferred_stop_reason"] explains why
```

### Reflection Law Success
```python
cert = wrap_reflection_result_to_certificate(
    {"observation": {"law_holds": True, "measured_angles": {...}}},
    "GeometryAngleObserver"
)
# → success with projection_contract
# → context has measured angles (exact scalars)
```

### Null Observer Obstruction
```python
cert = wrap_reflection_result_to_certificate(
    {"observation": {"law_holds": False, "reason": "no angles"}},
    "NullObserver"
)
# → obstruction with fail_type="observer_undefined"
# → proves angles are projection-added
```

## Exact Scalar Conversion

```python
from qa_certificate import to_scalar

# Valid
to_scalar(25)        # → 25
to_scalar("3/2")     # → Fraction(3, 2)
to_scalar("63.43")   # → Fraction(6343, 100)

# Invalid (raises TypeError)
to_scalar(63.43)     # Float rejected
to_scalar(True)      # Bool rejected
```

## File Structure

```
your_repo/
├── qa_certificate.py                          # Core schema
├── qa_alphageometry/
│   └── adapters/
│       └── certificate_adapter.py             # AG → Certificate
├── qa_physics/
│   └── adapters/
│       └── certificate_adapter.py             # Physics → Certificate
└── artifacts/
    ├── ptolemy_success.cert.json
    ├── ptolemy_ablated.obstruction.cert.json
    ├── reflection_geometry.success.cert.json
    └── reflection_null.obstruction.cert.json
```

## Testing Checklist

- [ ] `to_scalar()` rejects floats
- [ ] AlphaGeometry adapter handles `solved=True`
- [ ] AlphaGeometry adapter handles `solved=False`
- [ ] Physics adapter creates `ProjectionContract`
- [ ] All certificates serialize to valid JSON
- [ ] Certificates have `schema_version="1.0"`
- [ ] Success certificates have non-empty `success_path`
- [ ] Obstruction certificates have `fail_type`

## Paper Integration Points

**§3 (QA-AlphaGeometry):**
- Show `ptolemy_success.cert.json` excerpt
- Show `ptolemy_ablated.obstruction.cert.json` excerpt
- Explain generator extraction from `proof.steps`

**§4 (Physics Projection):**
- Show `reflection_geometry.success.cert.json` excerpt
- Highlight `ProjectionContract` section
- Note `observer_id` field

**§5 (Conclusion):**
> "The same certificate machinery applies to projection-dependent physical laws 
> (see companion artifact tag `qa-physics-projection-v0.1`)."

## Common Pitfalls

❌ **Don't:** Use floats anywhere in input
✅ **Do:** Convert to strings first: `str(angle)`

❌ **Don't:** Claim `SCC_UNREACHABLE` without proof
✅ **Do:** Use `DEPTH_EXHAUSTED` for beam search failures

❌ **Don't:** Mix generator namespaces
✅ **Do:** Keep `AG:*` for AlphaGeometry, `PHYS:*` for physics

❌ **Don't:** Modify certificates after generation
✅ **Do:** Regenerate with correct inputs

## Support Commands

### Verify Schema
```python
python -c "from qa_certificate import ProofCertificate; print('✓ Schema loaded')"
```

### Check Artifact
```python
import json
cert = json.load(open("artifact.cert.json"))
print(f"Type: {cert['witness_type']}")
print(f"Theorem: {cert['theorem_id']}")
if cert['witness_type'] == 'success':
    print(f"Steps: {len(cert['success_path'])}")
else:
    print(f"Fail: {cert['obstruction']['fail_type']}")
```

### Compare Certificates
```python
import json
c1 = json.load(open("cert1.json"))
c2 = json.load(open("cert2.json"))
print(f"Same schema: {c1['schema_version'] == c2['schema_version']}")
print(f"Same theorem: {c1['theorem_id'] == c2['theorem_id']}")
print(f"Same result: {c1['witness_type'] == c2['witness_type']}")
```

---

**Ready to generate your first certificates!** 🎯
