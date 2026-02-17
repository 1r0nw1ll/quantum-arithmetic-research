# Family [50]: QA ARTexplorer Scene Adapter v2 (Exact Substrate)

## Summary

Family [50] extends the ARTexplorer scene adapter ([45]) with an **exact-arithmetic compute substrate**. Where v1 uses IEEE-754 doubles with relative tolerance for RT law verification, v2 uses **unreduced rational pairs** with **zero tolerance** — law verification becomes strict integer equality via cross-multiplication.

## Relationship to Family [45]

| Property | [45] v1 (float) | [50] v2 (exact) |
|----------|-----------------|-----------------|
| Coordinates | `number` (float) | `integer` |
| Quadrance Q | `number` | `integer` |
| Spread s | `number` (float) | `{"n": int, "d": int}` (unreduced pair) |
| Law tolerance | relative 1e-9 | **0** (integer equality) |
| Law verification | direct computation | cross-multiplication |
| Fraction reduction | implicit (float) | **certified move** (`RT_REDUCE_FRACTION`) |
| `compute_substrate` | absent | `"qa_rational_pair_noreduce"` |
| cert_type | `QA_ARTEXPLORER_SCENE_ADAPTER.v1` | `QA_ARTEXPLORER_SCENE_ADAPTER.v2` |

v1 is **unchanged** — existing v1 certs remain valid. v2 is a new family in its own directory.

## Compute Substrate: `qa_rational_pair_noreduce`

### Unreduced Spread Pairs

Given vectors `v1`, `v2` from a triangle vertex:

```
q1 = dot(v1, v1)          # integer
q2 = dot(v2, v2)          # integer
d  = q1 * q2              # denominator (explicit multiply)
dot_val = dot(v1, v2)     # integer
n  = d - dot_val * dot_val  # numerator (explicit multiply, never **2)
spread = {"n": n, "d": d}   # unreduced rational pair
```

The pair is **never auto-reduced**. Any spread that does not match this unreduced computation triggers `ILLEGAL_NORMALIZATION`.

### Why Not Auto-Reduce?

In QA's formal framework, normalization is a **geometric projection** (scale collapse), not harmless simplification. Reducing `144/144` to `1/1` destroys the information that the denominator was `Q_AB * Q_AC = 9 * 16 = 144`. This scale information is part of the certified derivation trace.

### `RT_REDUCE_FRACTION`: Certified Projection

To obtain a reduced fraction, the cert must include an explicit `RT_REDUCE_FRACTION` step with:

- **Inputs**: `n`, `d`, `target` (field path being reduced)
- **Outputs**: `n_reduced`, `d_reduced`, `gcd`, `scale_before`, `scale_after`, `non_reduction_axiom_ack`
- **Verification**:
  - `n == n_reduced * gcd` and `d == d_reduced * gcd`
  - `gcd(n_reduced, d_reduced) == 1`
  - `scale_before == d`, `scale_after == d_reduced`
  - `non_reduction_axiom_ack == true`

## Law Verification (Zero Tolerance)

All RT laws are verified by **cross-multiplication to integer equality**:

### Cross Law (RT_LAW_04)

Original: `(Q1+Q2-Q3)^2 = 4*Q1*Q2*(1-s3)`

Cross-multiplied with `s3.d`:
```
t = Q1 + Q2 - Q3
t * t * s3.d == 4 * Q1 * Q2 * (s3.d - s3.n)
```

### Triple Spread (RT_LAW_05)

All terms cross-multiplied to common denominator `(d1*d2*d3)^2`.

### Pythagoras (RT_LAW_01)

`Q3 == Q1 + Q2` (integer equality, when spread = 1).

## 5-Gate Structure

Same 5-gate pattern as family [45]:

1. **Schema validity** — JSON Schema Draft-07 (v2 schema)
2. **Determinism contract** — canonical JSON + `compute_substrate = "qa_rational_pair_noreduce"`
3. **Typed formation** — integer coordinate enforcement (`NON_INTEGER_COORDINATE` on failure)
4. **Base algebra adequacy** — `no_zero_divisors` required
5. **Step hash + exact RT** — deterministic hashes, exact recomputation, zero-tolerance laws

## Failure Modes

| Fail Type | Cause |
|-----------|-------|
| `ILLEGAL_NORMALIZATION` | Spread pair doesn't match unreduced computation (fraction reduced without `RT_REDUCE_FRACTION` move) |
| `NON_INTEGER_COORDINATE` | Vertex coordinate is not an integer |
| `LAW_EQUATION_MISMATCH` | Cross-multiplied law equation fails integer equality |
| `UNSUPPORTED_COMPUTE_SUBSTRATE` | `compute_substrate` missing or not `"qa_rational_pair_noreduce"` |

## Quick Validation

```bash
# Self-test (2 fixtures: 1 valid, 1 negative)
python qa_artexplorer_scene_adapter_v2/validator.py --self-test

# Validate a specific cert
python qa_artexplorer_scene_adapter_v2/validator.py path/to/cert.json

# Full meta-validator (includes [50])
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

## Fixtures

| Fixture | Expected | Tests |
|---------|----------|-------|
| `valid_exact_345_triangle.json` | PASS | 3-4-5 right triangle with exact spreads, Cross Law verified |
| `invalid_illegal_reduction.json` | FAIL | sA reduced from `{n:144,d:144}` to `{n:1,d:1}` without `RT_REDUCE_FRACTION` |

## Substrate Pitfall: `x**2` vs `x*x`

Even though v2 operates on integers (where `**2` and `*` produce identical results in Python), the codebase convention is to **always use explicit multiply** (`x * x`, never `x**2`). This convention was established during the v1 burn-in where `pow(x,2)` caused ULP drift at 0.24% rate on floats, and is maintained for v2 as a determinism hygiene rule.

## Directory Structure

```
qa_artexplorer_scene_adapter_v2/
  schema.json
  validator.py
  mapping_protocol_ref.json
  fixtures/
    valid_exact_345_triangle.json
    invalid_illegal_reduction.json
```
