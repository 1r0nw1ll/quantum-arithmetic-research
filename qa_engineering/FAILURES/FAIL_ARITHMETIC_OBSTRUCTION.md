# FAIL: ARITHMETIC_OBSTRUCTION_IGNORED

**Trigger**: `target_r = b·e` has `v_p(target_r) = 1` for an inert prime `p`, but the cert
declares `obstructed: false`.

---

## Broken cert (minimal)

```json
"state_encoding": [
  { "label": "target", "b": 1, "e": 3, "orbit_family": "cosmos" }
],
"target_condition": { "label": "target", "orbit_family": "cosmos" },
"obstruction_check": {
  "modulus": 9,
  "inert_primes": [3],
  "target_r": 3,
  "v_p_values": { "3": 0 },
  "obstructed": false
}
```

## Validator output

```
[FAIL]
  errors:
    - Recomputed engineering failures ['ARITHMETIC_OBSTRUCTION_IGNORED'] but result=PASS — inconsistency
```

## Why it fails

`target_r = 1·3 = 3`. `v₃(3) = 1` (3 divides 3 exactly once). Prime 3 is inert in Z[φ]
(the ring of integers of Q(√5)). When `v_p(r) = 1` for an inert prime, the target is
arithmetically unreachable — no sequence of generators can reach it, regardless of what the
classical Kalman rank test says.

Note: EC5 (orbit family check) **passes** for this state — `f(1,3) = 1+3-9 = -5 ≡ 4 mod 9`,
`v₃(4) = 0`, so the state is correctly classified as cosmos. The orbit classification is fine.
The problem is in `target_r`, not `f(b,e)`. These are independent checks.

## The fix

Choose a target encoding where `v_p(b·e) ≠ 1` for all inert primes:

```json
{ "label": "target", "b": 1, "e": 2, "orbit_family": "cosmos" }
```

`target_r = 2`, `v₃(2) = 0` → not obstructed. Recompute `f(1,2) = -1 ≡ 8 mod 9`, `v₃(8) = 0`
→ cosmos ✓.

---

**Inert primes by modulus**:
- mod 9: `{3}`
- mod 24: `{3, 7}`

**Cert family reference**: EC11 in `qa_engineering_core_cert/qa_engineering_core_cert_validate.py`
**Live fixture**: `qa_engineering_core_cert/fixtures/engineering_core_fail_arithmetic_obstruction.json`
**Background**: `FOUNDATIONS_OF_ENGINEERING_AND_APPLIED_MATH_FOR_QA.md` — Control section
