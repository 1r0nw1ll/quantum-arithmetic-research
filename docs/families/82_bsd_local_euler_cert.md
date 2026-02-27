# Family [82] — QA BSD Local Euler Cert v1

**Machine tract:** `qa_bsd_local_euler_cert_v1/`  
**Schema version:** `QA_BSD_LOCAL_EULER_CERT.v1`

---

## What it certifies

For a short Weierstrass curve `E: y^2 = x^3 + Ax + B` over `F_p`, this family
deterministically certifies:

- `#E(F_p)` by brute-force point counting
- `a_p = p + 1 - #E(F_p)`
- good/bad reduction flag from `Delta mod p`, where `Delta = -16(4A^3 + 27B^2)`

---

## Gate summary

| Gate | What it checks |
|---|---|
| Gate 1 | JSON schema validity (`schema_id`, `schema_version`, curve model, prime, claimed fields) |
| Gate 2 | Deterministic recomputation of `point_count_fp` and `a_p` |
| Gate 3 | Optional `claimed.reduction_type` consistency with recomputed good/bad reduction |

---

## Failure taxonomy

| `fail_type` | Trigger condition |
|---|---|
| `SCHEMA_INVALID` | JSON schema/type/required-field mismatch |
| `RECOMPUTE_MISMATCH` | Claimed `point_count_fp` or `a_p` differs from deterministic recomputation |
| `REDUCTION_TYPE_MISMATCH` | Claimed `reduction_type` conflicts with recomputed good/bad reduction flag |

---

## Fixtures

| File | Type | Notes |
|---|---|---|
| `pass_good_p5.json` | PASS | `E: y^2 = x^3 + 1`, `p=5`, expects `#E(F_5)=6`, `a_5=0`, good reduction |
| `fail_wrong_ap.json` | FAIL | Same curve/prime but intentionally wrong `a_p` to trigger `RECOMPUTE_MISMATCH` |

---

## Self-test

```bash
python3 qa_bsd_local_euler_cert_v1/validator.py --self-test
```

Expected: `self_test_ok: true`.
