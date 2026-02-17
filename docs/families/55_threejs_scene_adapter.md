# Family [55]: QA Three.js Scene Adapter

## Summary

Family [55] is a QA adapter cert family for ingesting [Three.js](https://threejs.org/) scene exports and certifying their geometric objects with Rational Trigonometry (RT) invariants over a **float64 compute substrate**.

Three.js is the dominant WebGL 3D library. Its scene graph exports (JSON format `three_scene_export_v1`) contain typed geometric objects — triangles, meshes — with floating-point vertex coordinates. This family bridges Three.js geometry into the QA certified geometry pipeline.

## Relationship to Other Adapter Families

| Family | Upstream | Format | Substrate |
|--------|----------|--------|-----------|
| [45]   | ARTexplorer | `artexplorer_scene_export_v1` | float64 (rel tol 1e-9) |
| [50]   | ARTexplorer | `artexplorer_scene_export_v1` | exact (`qa_rational_pair_noreduce`) |
| **[55]** | **Three.js** | **`three_scene_export_v1`** | **float64 (rel tol 1e-9)** |

## Upstream

- **Library**: Three.js (`https://github.com/mrdoob/three.js`)
- **Docs**: `https://threejs.org/docs/`
- **Export format**: `three_scene_export_v1` — constrained JSON with `format` key and `objects` array

## Compute Substrate: float64

Coordinates are IEEE-754 doubles (`"type": "number"` in schema). Quadrances and spreads are floats. Law verification uses **relative tolerance** `REL_TOL = 1e-9`:

```
rel_residual = |lhs - rhs| / max(|lhs|, |rhs|, 1.0) <= 1e-9
```

Rule: always use `x*x` not `x**2` in validator/generator arithmetic to avoid libm `pow()` ULP drift.

## RT Invariants

For each triangle with vertices A, B, C, the adapter certifies:

```
Q = [Q(BC), Q(CA), Q(AB)]    — three quadrances (floats)
s = [sA, sB, sC]             — three spreads (floats in [0,1])
```

Where:
```
Q(PQ) = (px-qx)^2 + (py-qy)^2 + (pz-qz)^2   (always use x*x form)
sA    = 1 - dot(AB,AC)^2 / (Q(AB)*Q(AC))
```

## 5-Gate Validation Structure

| Gate | Name | Checks |
|------|------|--------|
| 1 | Schema validity | `jsonschema` Draft-07 against `schema.json` |
| 2 | Determinism contract + invariant_diff | All 4 flags true; `result.invariant_diff` present; `compute_substrate == "float64"` |
| 3 | Typed formation | `scene_raw_sha256` verified; parse `three_scene_export_v1`; detect non-finite coordinates; stable sort by `object_id` |
| 4 | Base algebra adequacy | `base_algebra.properties.field == true` and `no_zero_divisors == true` |
| 5 | Step hash chain + RT recomputation + law verification | SHA256 step hashes; RT recomputation vs claimed values; `RT_VALIDATE_LAW_EQUATION` steps verified with relative tolerance |

## Failure Modes

| Fail Type | Gate | Cause |
|-----------|------|-------|
| `NONFINITE_COORDINATE` | 3 | `null`, `NaN`, `Infinity`, or non-numeric coordinate value |
| `TYPED_FORMATION_ERROR` | 3 | Wrong scene format, missing/malformed objects, `scene_raw_sha256` mismatch |
| `MISSING_DETERMINISM_CONTRACT` | 2 | Determinism flags not all `true` |
| `MISSING_INVARIANT_DIFF` | 2 | `result.invariant_diff` absent |
| `UNSUPPORTED_COMPUTE_SUBSTRATE` | 2 | `compute_substrate != "float64"` |
| `STEP_HASH_MISMATCH` | 5 | Step hash does not match canonical JSON payload |
| `LAW_EQUATION_MISMATCH` | 5 | RT law exceeds relative tolerance, or `outputs.verified != true` |
| `ZERO_DIVISOR_OBSTRUCTION` | 5 | Degenerate triangle (zero-length edge) |

## Supported Move IDs

| Move ID | Description |
|---------|-------------|
| `THREE_PARSE_SCENE` | Parse `three_scene_export_v1` into typed objects |
| `RT_COMPUTE_TRIANGLE_INVARIANTS` | Compute Q and s for a triangle |
| `RT_VALIDATE_LAW_EQUATION` | Verify RT law (Cross Law RT_LAW_04, Triple Spread RT_LAW_05, Pythagoras RT_LAW_01) |

## NONFINITE_COORDINATE Detection

Gate 3 checks every coordinate value: `isinstance(x, (int, float)) and math.isfinite(float(x))`. JSON `null` becomes Python `None`, failing the `isinstance` check → `NONFINITE_COORDINATE` failure.

## Files

| File | Description |
|------|-------------|
| `qa_threejs_scene_adapter_v1/schema.json` | JSON Schema Draft-07 |
| `qa_threejs_scene_adapter_v1/validator.py` | 5-gate validator with `--self-test` and `--json` |
| `qa_threejs_scene_adapter_v1/mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `qa_threejs_scene_adapter_v1/fixtures/valid_minimal_triangle_scene.json` | 3-4-5 right triangle (valid) |
| `qa_threejs_scene_adapter_v1/fixtures/invalid_nonfinite_coordinate.json` | null coordinate → NONFINITE_COORDINATE |
| `qa_threejs_scene_adapter_v1/fixtures/invalid_missing_invariant_diff.json` | missing invariant_diff → MISSING_INVARIANT_DIFF |
| `qa_threejs_scene_adapter_v1/fixtures/invalid_law_equation_mismatch.json` | wrong spread value → LAW_EQUATION_MISMATCH |

## Quick Validation

```bash
# Self-test (all 4 fixtures)
python3 qa_threejs_scene_adapter_v1/validator.py --self-test

# Validate a single certificate
python3 qa_threejs_scene_adapter_v1/validator.py path/to/cert.json

# JSON output for toolchain integration
python3 qa_threejs_scene_adapter_v1/validator.py --self-test --json

# Full meta-validator (all families)
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

## Example: 3-4-5 Right Triangle

Vertices A=(0,0,0), B=(3,0,0), C=(0,4,0):

```
Q = [25.0, 16.0, 9.0]
s = [1.0, 0.64, 0.36]
```

Cross Law check (RT_LAW_04 at vertex C with Q3=Q(AB)=9, s3=sC=0.36):
```
(Q1+Q2-Q3)^2 = (25+16-9)^2 = 1024
4*Q1*Q2*(1-s3) = 4*25*16*(1-0.36) = 1024  ✓
```

## Notes

- Family [55] completes the "Three.js" leg of the Adapter Multiplication roadmap (GeoGebra, Blender, Three.js).
- The float64 substrate is identical in design to Family [45] (ARTexplorer v1) — only the upstream format and parser differ.
- For exact-arithmetic geometry, use Families [44] (RT type system) or [50] (ARTexplorer v2 exact).
