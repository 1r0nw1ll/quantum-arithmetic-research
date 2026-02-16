# [45] QA ARTexplorer Scene Adapter (QA_ARTEXPLORER_SCENE_ADAPTER.v1)

Adapter cert family that ingests an **ARTexplorer** exported JSON scene and certifies:

- Parsed typed objects (Point3/Triangle/Mesh/Polyhedron)
- RT invariants (quadrance/spread) computed deterministically
- Step hashes over canonical JSON
- Typed obstruction taxonomy (degenerate triangles, illegal normalization, zero divisors)

**ARTexplorer** is an open-source algebraic geometry visualization tool that implements
Wildberger's Rational Trigonometry with quadray (WXYZ) coordinate support:
https://github.com/arossti/ARTexplorer

## Machine tract

Directory: `qa_artexplorer_scene_adapter_v1/`

Files:
- `qa_artexplorer_scene_adapter_v1/schema.json`
- `qa_artexplorer_scene_adapter_v1/validator.py`
- `qa_artexplorer_scene_adapter_v1/fixtures/` (3 valid, 3 negative)
- `qa_artexplorer_scene_adapter_v1/mapping_protocol_ref.json` (Gate 0 intake)
- `Documents/QA_MAPPING_PROTOCOL__ARTEXPLORER_SCENE_ADAPTER.v1.json` (mapping protocol)

### What it validates (gates)

- **Gate 1 -- Schema validity**: JSON Schema Draft-07 conformance
- **Gate 2 -- Determinism contract**: Contract flags strict + `result.invariant_diff` present (typed obstruction `MISSING_INVARIANT_DIFF` lives here, not at schema level)
- **Gate 3 -- Typed formation**: Triangle non-collinearity (3D cross product); WXYZ zero-sum normalization legality (must record explicit `ART_NORMALIZE_WXYZ_ZERO_SUM` move)
- **Gate 4 -- Base algebra adequacy**: Spread computation requires `no_zero_divisors`
- **Gate 5 -- Step hash + RT recomputation**: Verifies `step_hash_sha256 = sha256(canonical_json({move_id, inputs, outputs}))` and recomputes RT invariants to match claimed values

### Failure taxonomy

| Fail type | Meaning | Gate |
|-----------|---------|------|
| `SCENE_PARSE_ERROR` | Upstream scene JSON cannot be parsed into typed objects | Gate 1/3 |
| `MISSING_INVARIANT_DIFF` | `result.invariant_diff` absent or malformed | Gate 2 |
| `NONDETERMINISM_CONTRACT_VIOLATION` | Contract flags wrong or step hash mismatch | Gate 2/5 |
| `DEGENERATE_TRIANGLE_COLLINEAR` | Triangle formation rule violated (cross product = 0) | Gate 3 |
| `ILLEGAL_NORMALIZATION` | WXYZ zero-sum normalization without recorded generator move | Gate 3 |
| `BASE_ALGEBRA_TOO_WEAK` | Spread needs `no_zero_divisors` but base ring lacks it | Gate 4 |
| `ZERO_DIVISOR_OBSTRUCTION` | Q=0 in spread denominator | Gate 5 |
| `LAW_EQUATION_MISMATCH` | Claimed RT invariants don't match recomputation | Gate 5 |

### Generator moves (adapter pipeline)

| Move ID | Description |
|---------|-------------|
| `ART_PARSE_SCENE` | Parse upstream ARTexplorer JSON into typed objects |
| `RT_COMPUTE_TRIANGLE_INVARIANTS` | Compute Q1,Q2,Q3 and s1,s2,s3 for each Triangle |
| `ART_NORMALIZE_WXYZ_ZERO_SUM` | Project WXYZ quadray coordinates to zero-sum form |
| `ART_PROJECT_WXYZ_TO_XYZ` | Certified projection from WXYZ to Cartesian XYZ |
| `RT_VALIDATE_LAW_EQUATION` | Validate claimed RT law equation numerically |

### Run

```bash
python qa_artexplorer_scene_adapter_v1/validator.py --self-test
python qa_artexplorer_scene_adapter_v1/validator.py qa_artexplorer_scene_adapter_v1/fixtures/valid_minimal.json
```

## Human tract

### Why ARTexplorer is QA-shaped

ARTexplorer implements Wildberger's Rational Trigonometry natively:

- **Quadrance** replaces distance: `Q(A,B) = dx^2 + dy^2 + dz^2` (no sqrt)
- **Spread** replaces angle: `s = 1 - (dot^2)/(Q1*Q2)` (no arctan/sin/cos)
- **sqrt deferred to GPU boundary**: all invariants are algebraic until final render
- **Quadray coordinates (WXYZ)**: tetrahedral 4-axis system with zero-sum normalization

This is the same "algebraic invariants, no transcendentals" discipline that QA enforces.
The adapter cert bridges ARTexplorer's scene JSON format into QA's typed manifold +
generator move + obstruction framework.

### Adapter pattern

The adapter pipeline is:

```
ARTexplorer scene JSON
  --[ART_PARSE_SCENE]--> typed objects (Point3, Triangle, Mesh)
  --[RT_COMPUTE_TRIANGLE_INVARIANTS]--> RT observables (Q, s)
  --[RT_VALIDATE_LAW_EQUATION]--> verified law constraints (optional)
```

Each step is deterministic with `step_hash_sha256` over canonical JSON.

### WXYZ normalization discipline

Quadray coordinates (WXYZ) are 4D tetrahedral axes. ARTexplorer supports a
"Zero-Sum Normalize" toggle that projects `W+X+Y+Z=0`. In QA terms, this is a
**projection/normalization move** that must be explicitly recorded:

- If `zero_sum_normalized=true` in scene but `ART_NORMALIZE_WXYZ_ZERO_SUM` is not
  in derivation steps: `ILLEGAL_NORMALIZATION` (Gate 3)
- If recorded but vertices aren't actually zero-sum: `ILLEGAL_NORMALIZATION` (Gate 3)

This enforces the "Types vs Sets" principle: normalization is a typed move with
formation rules, not a silent convenience.

### External semantics anchoring

- **Gate 0**: `mapping_protocol_ref.json` sha256-locks
  `Documents/QA_MAPPING_PROTOCOL__ARTEXPLORER_SCENE_ADAPTER.v1.json`
  (enforced by meta-validator `require_mapping_protocol()`)
- **Source semantics**: Each cert instance carries `source_semantics.upstream`
  pointing to the ARTexplorer repo and app URL

### Sources

- ARTexplorer: https://github.com/arossti/ARTexplorer
- ARTexplorer live app: https://arossti.github.io/ARTexplorer/
- Wildberger, N.J. (2007). "A Rational Approach to Trigonometry."
- Thomson, A.R. (2026). "Spread-Quadray Rotors." (ResearchGate preprint)
- Family [44] QA_RATIONAL_TRIG_TYPE_SYSTEM.v1 (foundational RT + type theory family)

## Notes

- v1 adapter supports XYZ invariant computation directly; WXYZ is supported via
  certified `ART_PROJECT_WXYZ_TO_XYZ` projection step (recorded in derivation)
- This family complements [44]: family [44] is the pure RT type system, family [45]
  is the adapter that connects an external tool (ARTexplorer) to that type system

### Fixtures (3 valid, 3 negative)

| Fixture | Type | What it demonstrates |
|---------|------|---------------------|
| `valid_minimal.json` | PASS | Right isosceles triangle (XYZ), cross-family consistency with [44] |
| `valid_provenance_tetrahedron.json` | PASS | **Provenance-grade**: real ARTexplorer scene_raw (Tetrahedron, halfSize=1), scene_raw_sha256 locked, ART_PARSE_SCENE + RT_COMPUTE steps, equilateral face Q=[8,8,8] s=[0.75,0.75,0.75] |
| `valid_wxyz_projected.json` | PASS | **WXYZ pipeline**: quadray coords → ART_NORMALIZE_WXYZ_ZERO_SUM → ART_PROJECT_WXYZ_TO_XYZ → RT_COMPUTE, same tetrahedron face, proves WXYZ→XYZ→RT pipeline with typed moves |
| `invalid_missing_invariant_diff.json` | FAIL Gate 2 | `MISSING_INVARIANT_DIFF` (typed obstruction, not schema reject) |
| `invalid_degenerate_triangle.json` | FAIL Gate 3 | `DEGENERATE_TRIANGLE_COLLINEAR` (collinear points 0,0,0 → 1,1,1 → 2,2,2) |
| `invalid_illegal_normalization.json` | FAIL Gate 3 | `ILLEGAL_NORMALIZATION` (WXYZ zero_sum_normalized but no normalization move recorded) |
