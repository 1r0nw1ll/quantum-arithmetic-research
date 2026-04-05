# FAIL: ORBIT_FAMILY_CLASSIFICATION_FAILURE

**Trigger**: the `orbit_family` declared for a state disagrees with the orbit family recomputed
from `f(b, e) = b·b + b·e - e·e` and the modulus.

---

## Broken cert (minimal)

```json
"state_encoding": [
  { "label": "active", "b": 1, "e": 2, "orbit_family": "satellite" }
],
"modulus": 9
```

## Validator output

```
[FAIL]
  errors:
    - Recomputed engineering failures ['ORBIT_FAMILY_CLASSIFICATION_FAILURE'] but result=PASS — inconsistency
```

## Why it fails

The validator recomputes orbit family from scratch:

```
f(1, 2) = 1 + 2 - 4 = -1 ≡ 8 mod 9
v₃(8) = 0   →   cosmos
```

The cert declared `satellite` but `(1, 2)` is a cosmos state. The validator rejects the
inconsistency.

## The three rules (mod 9)

| Condition | Orbit family |
|-----------|--------------|
| `(b,e) ≡ (0,0) mod 9` | singularity |
| `f(b,e) ≡ 0 mod 9` (i.e. `v₃(f) ≥ 2`) | satellite |
| `f(b,e) mod 3 ≠ 0` (i.e. `v₃(f) = 0`) | cosmos |
| `f(b,e) mod 3 = 0` but `f(b,e) mod 9 ≠ 0` (i.e. `v₃(f) = 1`) | obstruction (not a valid QA state for this cert type) |

## The fix

Either change the declared orbit family to match:

```json
{ "label": "active", "b": 1, "e": 2, "orbit_family": "cosmos" }
```

Or choose a different `(b, e)` that actually produces a satellite orbit, e.g. `(3, 6)`:

```
f(3, 6) = 9 + 18 - 36 = -9 ≡ 0 mod 9, v₃(-9) = 2 ≥ 2 → satellite ✓
```

## Quick verification table (mod 9, common encodings)

| (b, e) | f(b,e) | f mod 9 | Orbit family |
|--------|--------|---------|--------------|
| (9, 9) | 81 | 0 | singularity |
| (3, 6) | -9 | 0 | satellite |
| (6, 3) | 27 | 0 | satellite |
| (1, 2) | -1 | 8 | cosmos |
| (2, 1) | 5 | 5 | cosmos |
| (1, 1) | 1 | 1 | cosmos |
| (2, 3) | -1 | 8 | cosmos |

For an exhaustive table: `05_reference/QUICK_REFERENCE.md`.

---

**Cert family reference**: EC5 in `qa_engineering_core_cert/qa_engineering_core_cert_validate.py`
