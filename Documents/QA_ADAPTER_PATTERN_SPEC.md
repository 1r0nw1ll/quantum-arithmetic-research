# QA Adapter Pattern Specification

## Overview

The **5-gate adapter pattern** is a proven architecture for ingesting external geometry data into the QA certificate ecosystem. It was established by family [45] (ARTexplorer Scene Adapter) and generalizes to any external source that produces typed geometric objects.

## Pipeline Stages

```
scene_raw  →  ART_PARSE_SCENE  →  RT_COMPUTE  →  RT_VALIDATE  →  certified result
  (gate 1-2)     (gate 3)          (gate 5)       (gate 5)         (gate 5)
```

## The 5 Gates

| Gate | Name | Purpose |
|------|------|---------|
| 1 | Schema Validity | JSON Schema (Draft-07) conformance |
| 2 | Determinism Contract | canonical_json, stable_sorting, no_rng, invariant_diff |
| 3 | Typed Formation | Parse scene → typed objects, detect illegal normalization |
| 4 | Base Algebra Adequacy | Confirm algebra supports required operations (e.g. no_zero_divisors) |
| 5 | Step Hash + RT Verification | Deterministic step hashes, recompute invariants, verify law equations |

## Reusable vs Adapter-Specific Concerns

### Reusable (in `qa_cert_validator_base.py`)

- `GateStatus` / `GateResult` dataclasses
- `canonical_json_compact(obj)` — deterministic serialization
- `sha256_hex(s)` — SHA-256 hex digest
- `load_json(path)` / `validate_schema(obj, schema_path)`
- `report_ok(results)` / `print_human(results)` / `print_json(results)`
- `run_self_test(...)` — fixture-driven self-test harness
- `build_cli(...)` — standard `--self-test` / `--json` / file CLI

### Adapter-Specific (implement per family)

- Schema definition (`schema.json`)
- Gate 3 typed formation rules (coordinate system, object types, degeneracy checks)
- Gate 4 algebra requirements
- Gate 5 invariant recomputation logic (RT, EBM, or domain-specific)
- Law equation verification (law dispatch table)
- Fixture creation and self-test fixture list

## Step Hash Contract

Every derivation step records a `step_hash_sha256` computed as:

```python
payload = {
    "inputs":  step["inputs"],
    "move_id": step["move_id"],
    "outputs": step["outputs"],
}
step_hash_sha256 = sha256_hex(canonical_json_compact(payload))
```

This ensures:
- **Determinism**: same inputs + move + outputs always produce the same hash
- **Tamper detection**: any change to step content invalidates the hash
- **Auditability**: validators independently recompute and compare

## Law Equation Verification

For `RT_VALIDATE_LAW_EQUATION` steps, the validator:

1. Extracts `law_id`, `Q`, `s` from `inputs`
2. Dispatches to the appropriate law computation:
   - `RT_LAW_04` (Cross Law): `(Q1+Q2-Q3)^2 = 4*Q1*Q2*(1-s3)`
   - `RT_LAW_05` (Triple Spread): `(s1+s2+s3)^2 = 2*(s1^2+s2^2+s3^2) + 4*s1*s2*s3`
   - `RT_LAW_01` (Pythagoras): `Q3 = Q1 + Q2` when `s3 = 1`
3. Checks residual < 1e-9
4. Confirms `outputs.verified == true`
5. Fails with `LAW_EQUATION_MISMATCH` on any discrepancy

## Numerical Tolerance

Law verification uses **relative tolerance 1e-9**: `residual / max(|lhs|, |rhs|, 1.0) < 1e-9`. This acknowledges the floating-point substrate when `base_algebra = Q` (semantic target: rational trigonometry) but computations run on IEEE-754 doubles. RT laws are algebraically exact over the rationals; the tolerance exists solely because the computational substrate introduces rounding in intermediate products (e.g., `(Q1+Q2-Q3)^2` vs `4*Q1*Q2*(1-s3)` accumulate different ULP errors).

Relative (not absolute) tolerance is required because law equation terms scale as `O(coord^4)`. At coordinate scale 1e6, absolute residuals reach ~1e9 even though relative error stays at ~1e-15. A 10k burn-in at `--coord-scale 1e6` confirmed zero failures with relative tolerance.

**Default REL_TOL = 1e-9** is intentionally loose vs observed ~7e-14 relative error at scales 10 and 1e6, providing 5 orders of margin.

Family [50] (ARTexplorer Scene Adapter v2) implements an exact-arithmetic substrate using unreduced rational pairs — see "Exact Arithmetic Substrate" section below.

### Known Substrate Pitfall: `pow(x,2)` vs `x*x` (ULP drift)

IEEE-754 + libm `pow()` can diverge by 1 ULP from explicit multiplication. CPython's `float.__pow__` calls `pow(x, 2.0)` from libm, which is **not** guaranteed to produce the same bits as `x * x`. This was demonstrated during the 10k burn-in: generators using `d**2` produced Q/s values that differed from validators using `d*d`, causing exact-match failures at 0.24% rate (24/10000).

**Rule:** For quadrance and all invariant computations, always use explicit multiply-add form (`dx*dx + dy*dy + dz*dz`). Never use `**2`, `pow()`, or `math.pow()` for squaring in deterministic pipelines.

## Checklist for New Adapters

1. [ ] Define `schema.json` (Draft-07) with cert_type, derivation.steps.move_id enum
2. [ ] Create `mapping_protocol.json` or `mapping_protocol_ref.json` in family root
3. [ ] Implement 5-gate `validate_cert(obj)` function
4. [ ] Arithmetic form is standardized (no `pow`/`sqrt`; use explicit multiply-add forms for invariants)
5. [ ] Create at least 1 valid + 1 invalid fixture per gate being tested
6. [ ] Implement `--self-test` with all fixtures
7. [ ] Add family to `FAMILY_SWEEPS` in `qa_meta_validator.py`
8. [ ] Write human-tract doc in `docs/families/[NN]_*.md`
9. [ ] Verify: `python <family>/validator.py --self-test` all PASS

## Exact Arithmetic Substrate (Family [50])

Family [50] (`QA_ARTEXPLORER_SCENE_ADAPTER.v2`) eliminates all floating-point tolerance by operating on **unreduced rational pairs** with integer arithmetic.

### Compute Substrate: `qa_rational_pair_noreduce`

- **Coordinates**: must be integers (schema enforces `"type": "integer"`)
- **Quadrance Q**: integer (sum of squared integer differences, explicit `d*d` form)
- **Spread s**: unreduced rational pair `{"n": int, "d": int}` where:
  - `d = Q1 * Q2` (product of adjacent quadrances)
  - `dot_val = dot(v1, v2)`, `n = d - dot_val * dot_val` (explicit multiply, never `**2`)
  - Validator recomputes and checks **pair equality** (not numeric equality)
  - Any pair not matching the unreduced computation → `ILLEGAL_NORMALIZATION`

### Law Verification by Cross-Multiplication (Zero Tolerance)

All RT laws are verified as strict integer equalities after cross-multiplication:

| Law | Cross-Multiplied Form |
|-----|-----------------------|
| Cross Law (RT_LAW_04) | `t*t*s3.d == 4*Q1*Q2*(s3.d - s3.n)` where `t = Q1+Q2-Q3` |
| Triple Spread (RT_LAW_05) | Numerators compared at common denominator `(d1*d2*d3)^2` |
| Pythagoras (RT_LAW_01) | `Q3 == Q1 + Q2` (integer equality) |

### `RT_REDUCE_FRACTION`: Certified Projection

Fraction reduction is **not** simplification — it is a **geometric projection** (scale collapse). To reduce a spread pair, the cert must include an explicit `RT_REDUCE_FRACTION` step:

- **Inputs**: `n`, `d`, `target` (field path being reduced)
- **Outputs**: `n_reduced`, `d_reduced`, `gcd`, `scale_before`, `scale_after`, `non_reduction_axiom_ack`
- **Verification**: `n == n_reduced * gcd`, `d == d_reduced * gcd`, `gcd(n_reduced, d_reduced) == 1`, `non_reduction_axiom_ack == true`

### Additional Failure Modes

| Fail Type | Cause |
|-----------|-------|
| `NON_INTEGER_COORDINATE` | Vertex coordinate is not an integer |
| `ILLEGAL_NORMALIZATION` | Spread reduced without `RT_REDUCE_FRACTION` move |
| `UNSUPPORTED_COMPUTE_SUBSTRATE` | `compute_substrate` missing or wrong |

## Proven Instances

| Family | Adapter | Source | Substrate |
|--------|---------|--------|-----------|
| [44] | RT Type System | Internal (Wildberger RT + Martin-Lof) | float64 |
| [45] | ARTexplorer Scene Adapter v1 | External (ARTexplorer JSON) | float64 (rel tol 1e-9) |
| [50] | ARTexplorer Scene Adapter v2 | External (ARTexplorer JSON) | exact (rational pairs, tol=0) |
| [55] | Three.js Scene Adapter v1 | External (Three.js JSON) | float64 (rel tol 1e-9) |
