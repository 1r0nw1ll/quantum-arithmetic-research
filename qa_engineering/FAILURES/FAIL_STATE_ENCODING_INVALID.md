# FAIL: STATE_ENCODING_INVALID

**Trigger**: a state has `b = 0`, `e = 0`, or any value outside `{1,...,modulus}`.

---

## Broken cert (minimal)

```json
"state_encoding": [
  { "label": "still", "b": 0, "e": 9, "orbit_family": "singularity" }
]
```

## Validator output

```
[FAIL]
  errors:
    - Recomputed engineering failures ['STATE_ENCODING_INVALID'] but result=PASS — inconsistency
```

## Why it fails

QA arithmetic is defined on `{1,...,N}`. Zero has no role in QA: `f(0, e) = -e²`, which does not
classify correctly as singularity, and the orbit arithmetic breaks. The most common cause is
translating a 0-indexed classical state vector directly into QA coordinates.

## The fix

```json
{ "label": "still", "b": 9, "e": 9, "orbit_family": "singularity" }
```

Use `{1,...,9}` not `{0,...,8}`. For modulus 9, the value `9` represents the additive identity
(since `9 ≡ 0 mod 9`), which is correct for the singularity state.

---

**Cert family reference**: EC1 in `qa_engineering_core_cert/qa_engineering_core_cert_validate.py`
**Live fixture**: `qa_engineering_core_cert/fixtures/engineering_core_fail_invalid_encoding.json`
