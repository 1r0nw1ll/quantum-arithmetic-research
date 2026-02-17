# Family [56]: QA GeoGebra Scene Adapter (Exact Substrate)

## Summary

Family [56] is a QA adapter cert family for ingesting [GeoGebra](https://www.geogebra.org/) scene exports and certifying their geometric objects with Rational Trigonometry (RT) invariants over an **exact compute substrate** (`qa_rational_pair_noreduce`).

GeoGebra is a widely-used dynamic geometry application. Its scene exports contain geometric objects with coordinates that may be integers or unreduced rational pairs. This family bridges GeoGebra geometry into the QA certified geometry pipeline using **zero-tolerance exact arithmetic** — no floating-point, no tolerance, no approximation.

## Relationship to Other Adapter Families

| Family | Upstream | Format | Substrate |
|--------|----------|--------|-----------|
| [45]   | ARTexplorer | `artexplorer_scene_export_v1` | float64 (rel tol 1e-9) |
| [50]   | ARTexplorer | `artexplorer_scene_export_v1` | exact (`qa_rational_pair_noreduce`) |
| [55]   | Three.js | `three_scene_export_v1` | float64 (rel tol 1e-9) |
| **[56]** | **GeoGebra** | **`geogebra_scene_export_v1`** | **exact (`qa_rational_pair_noreduce`)** |

## Upstream

- **Application**: GeoGebra (`https://www.geogebra.org/`)
- **Export format**: `geogebra_scene_export_v1` — constrained JSON with `version` key and `objects` array
- **Object types**: `Triangle` (with vertices A, B, C each having XYZ coordinates typed as Z or Q)

## Compute Substrate: `qa_rational_pair_noreduce`

Coordinates are typed as either:
- **Z** (integer): `{"k": "Z", "v": <int>}`
- **Q** (unreduced rational pair): `{"k": "Q", "n": <int>, "d": <int>}` with `d != 0`

Per-triangle, a LCM lift converts all coordinates to an integer lattice:
```
L = LCM(denominators of all coords in triangle, counting Z coords as denominator=1)
scale = L // abs(d)    # for each coordinate
lifted_coord = v * scale  (for Z: v*L, for Q: n*scale)
```

Denominators are used **as-given** — no GCD simplification before LCM. This is the **Non-Reduction Axiom**: `{"k":"Q","n":1,"d":2}` and `{"k":"Q","n":2,"d":4}` represent different states even though they have the same numeric value.

## RT Invariants (Exact)

For each triangle with integer-lifted vertices A, B, C:

```
Q = [Q(BC), Q(CA), Q(AB)]    — three quadrances (integers)
s = [sA, sB, sC]             — three spreads (unreduced pairs {n, d})
```

Where:
```
Q(PQ) = dx*dx + dy*dy + dz*dz          (explicit multiply, never **2)
sA: v1=AB, v2=AC,  d=Q(AB)*Q(AC),  dot=dot(v1,v2),  n=d-dot*dot
```

All arithmetic uses explicit `*` multiplication — never `**2` or `pow()` — to avoid ULP drift.

## Spread Pair Identity (ILLEGAL_NORMALIZATION)

Spreads are stored as **unreduced pairs** `{n, d}`. The validator recomputes `d = Q1*Q2` and `n = d - dot*dot` from first principles and checks **exact pair equality**:

- Claimed `{n:144, d:144}` vs computed `{n:144, d:144}` → PASS
- Claimed `{n:1, d:1}` vs computed `{n:144, d:144}` → `ILLEGAL_NORMALIZATION` (pair identity mismatch)

Fraction reduction is never "just simplification" — it's a geometric action (scale collapse) that must be recorded explicitly.

## Law Verification (Zero Tolerance)

RT laws are verified by integer equality with **zero tolerance** (cross-multiplication):

**Cross Law (RT_LAW_04)**:
```
t = Q1 + Q2 - Q3
check: t*t*s3.d == 4*Q1*Q2*(s3.d - s3.n)
```

**Pythagoras (RT_LAW_01)** (for right triangles, s3=1 meaning n=d):
```
check: Q3 == Q1 + Q2
```

For the 3-4-5 triangle: `t=32, 32²×400=409600 == 4×25×16×256=409600 ✓`

## 5-Gate Validation Structure

| Gate | Name | Checks |
|------|------|--------|
| 1 | Schema validity | `jsonschema` Draft-07 against `schema.json` |
| 2 | Determinism contract + invariant_diff | All 4 flags true; `result.invariant_diff` present; `compute_substrate == "qa_rational_pair_noreduce"` |
| 3 | Typed formation | `scene_raw_sha256` verified; parse `geogebra_scene_export_v1`; check Z/Q type discipline; reject `d==0`; stable sort by `object_id` |
| 4 | Base algebra adequacy | `base_algebra.properties.field == true` and `no_zero_divisors == true` |
| 5 | Step hash chain + RT recomputation + law verification | SHA256 step hashes; LCM lift + exact RT recomputation; spread pair identity check; zero-tolerance law verification |

## Failure Modes

| Fail Type | Gate | Cause |
|-----------|------|-------|
| `SCHEMA_INVALID` | 1 | JSON Schema Draft-07 validation fails |
| `MISSING_DETERMINISM_CONTRACT` | 2 | Determinism flags not all `true` |
| `MISSING_INVARIANT_DIFF` | 2 | `result.invariant_diff` absent |
| `UNSUPPORTED_COMPUTE_SUBSTRATE` | 2 | `compute_substrate != "qa_rational_pair_noreduce"` |
| `TYPED_FORMATION_ERROR` | 3 | Wrong scene format, malformed objects, `scene_raw_sha256` mismatch, unsupported object type, unstable ordering |
| `NON_RATIONAL_COORDINATE` | 3 | Coordinate not typed as Z or Q(n,d) |
| `ZERO_DENOMINATOR` | 3 | Q coordinate has `d == 0` |
| `BASE_ALGEBRA_INADEQUATE` | 4 | Base algebra not a field with no zero divisors |
| `ZERO_DIVISOR_OBSTRUCTION` | 5 | Degenerate triangle (zero-length edge, Q=0 → spread denominator=0) |
| `STEP_HASH_MISMATCH` | 5 | Step hash does not match canonical JSON payload |
| `LAW_EQUATION_MISMATCH` | 5 | Integer equality check for RT law fails, or `outputs.verified != true` |
| `ILLEGAL_NORMALIZATION` | 5 | Spread pair identity mismatch: claimed `{n,d}` != computed `{n,d}` (unreduced pairs must match exactly) |

## Supported Move IDs

| Move ID | Description |
|---------|-------------|
| `GG_PARSE_SCENE` | Parse `geogebra_scene_export_v1` into typed TRIANGLE objects |
| `RT_COMPUTE_TRIANGLE_INVARIANTS` | Compute Q (integers) and s (unreduced pairs) via LCM lift |
| `RT_VALIDATE_LAW_EQUATION` | Verify RT law by integer equality (zero tolerance) |

## Files

| File | Description |
|------|-------------|
| `qa_geogebra_scene_adapter_v1/schema.json` | JSON Schema Draft-07 with Z/Q typed coordinates |
| `qa_geogebra_scene_adapter_v1/validator.py` | 5-gate exact validator with `--self-test` and `--json` |
| `qa_geogebra_scene_adapter_v1/mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `qa_geogebra_scene_adapter_v1/fixtures/valid_exact_345_triangle.json` | 3-4-5 right triangle (valid, exact) |
| `qa_geogebra_scene_adapter_v1/fixtures/invalid_missing_invariant_diff.json` | missing invariant_diff → Gate 2 MISSING_INVARIANT_DIFF |
| `qa_geogebra_scene_adapter_v1/fixtures/invalid_zero_denominator.json` | `{"k":"Q","n":1,"d":0}` coordinate → Gate 3 ZERO_DENOMINATOR |
| `qa_geogebra_scene_adapter_v1/fixtures/invalid_law_equation_mismatch.json` | sA.n=145 vs computed 144 → Gate 5 ILLEGAL_NORMALIZATION |

## Quick Validation

```bash
# Self-test (all 4 fixtures)
python3 qa_geogebra_scene_adapter_v1/validator.py --self-test

# Validate a single certificate
python3 qa_geogebra_scene_adapter_v1/validator.py path/to/cert.json

# JSON output for toolchain integration
python3 qa_geogebra_scene_adapter_v1/validator.py --self-test --json

# Full meta-validator (all families)
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

## Example: 3-4-5 Right Triangle

Vertices A=(0,0,0), B=(3,0,0), C=(0,4,0) — all integer Z coordinates:

```
L = LCM(1,1,1,...) = 1   (all denominators are 1)
Lifted: A=(0,0,0), B=(3,0,0), C=(0,4,0)

Q(BC) = 9+16 = 25,  Q(CA) = 16,  Q(AB) = 9

sA (at A, edges AB and AC):
  v1=AB=(3,0,0), v2=AC=(0,4,0)
  d = Q(AB)*Q(AC) = 9*16 = 144
  dot = 0
  n = 144 - 0*0 = 144    → sA = {n:144, d:144}

sB (at B, edges BC and BA):
  v1=BC=(-3,4,0), v2=BA=(-3,0,0)
  d = Q(BC)*Q(BA) = 25*9 = 225
  dot = (-3)*(-3) + 4*0 = 9
  n = 225 - 9*9 = 225 - 81 = 144    → sB = {n:144, d:225}

sC (at C, edges CA and CB):
  v1=CA=(0,-4,0), v2=CB=(3,-4,0)
  d = Q(CA)*Q(CB) = 16*25 = 400
  dot = 0*3 + (-4)*(-4) = 16
  n = 400 - 16*16 = 400 - 256 = 144    → sC = {n:144, d:400}
```

Cross Law verification at C (Q1=25, Q2=16, Q3=9, s3=sC={n:144,d:400}):
```
t = 25 + 16 - 9 = 32
LHS = t*t*s3.d = 32*32*400 = 409600
RHS = 4*Q1*Q2*(s3.d-s3.n) = 4*25*16*(400-144) = 4*25*16*256 = 409600  ✓
```

## Design Decisions

**Exact-only from day one**: Unlike the ARTexplorer family (float v1 [45] + exact v2 [50]), GeoGebra ships as exact-only. GeoGebra natively supports exact rational coordinates, making float approximation unnecessary.

**Non-Reduction Axiom**: Fraction reduction is a geometric action (scale collapse). An unreduced pair `{n:144, d:144}` and reduced pair `{n:1, d:1}` are pair-identity different even though numerically equal. Any cert claiming a reduced pair without an explicit `RT_REDUCE_FRACTION` move fails with `ILLEGAL_NORMALIZATION`.

**Typed Z/Q coordinates**: The schema enforces `oneOf [{k:Z, v:int}, {k:Q, n:int, d:int}]`. Floating-point values like `1.5` fail Gate 1 schema validation before reaching Gate 3 — there is no "float with typing" escape hatch.

## Notes

- Family [56] completes the "GeoGebra" leg of the Adapter Multiplication roadmap.
- For float64 geometry ingestion, use Families [45] (ARTexplorer v1) or [55] (Three.js v1).
- The exact substrate is identical in design to Family [50] (ARTexplorer v2 exact) — only the upstream format and parser differ.
