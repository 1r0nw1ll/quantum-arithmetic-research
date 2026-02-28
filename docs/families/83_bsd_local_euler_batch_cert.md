# Family [83] — QA BSD Local Euler Batch Cert v1

**Machine tract:** `qa_bsd_local_euler_batch_cert_v1/`  
**Schema version:** `QA_BSD_LOCAL_EULER_BATCH_CERT.v1`

---

## What it certifies

This family certifies a deterministic **batch** of local Euler records for one
short Weierstrass curve `E: y^2 = x^3 + Ax + B`.

For each prime in `records[]`, validator recomputes:

- `#E(F_p)`
- `a_p = p + 1 - #E(F_p)`
- `delta_mod_p = -16(4A^3 + 27B^2) mod p`
- `is_good_reduction = (delta_mod_p != 0)`

Then it verifies a stable claimed manifest:

- `claimed_manifest.record_hashes[]` (per-prime hashes)
- `claimed_manifest.batch_sha256` (hash over sorted per-prime hash list)

---

## Gate summary

| Gate | What it checks |
|---|---|
| Gate 1 | JSON schema validity (`schema_id`, curve model, records, claimed manifest) |
| Gate 2 | Deterministic per-prime recomputation + claimed invariant checks |
| Gate 3 | Manifest hash binding (`record_hashes[]` and `batch_sha256`) |

---

## Failure taxonomy

| `fail_type` | Trigger condition |
|---|---|
| `SCHEMA_INVALID` | Schema/type/required-field mismatch |
| `DUPLICATE_PRIME` | Same prime appears more than once in `records` |
| `RECOMPUTE_MISMATCH` | Claimed per-prime value disagrees with deterministic recomputation |
| `MANIFEST_MISMATCH` | Claimed record hashes or batch hash disagree with recomputed manifest |

---

## Fixtures

| File | Type | Notes |
|---|---|---|
| `pass_batch_p5_p7.json` | PASS | Two-prime batch (`p=5,7`) with full optional per-prime fields |
| `pass_batch_p5_p11.json` | PASS | Two-prime batch (`p=5,11`) with mixed claim granularity |
| `fail_corrupt_record_p7_ap.json` | FAIL | Corrupted `a_p` for `p=7` (expected `-4`) |

---

## Self-test

```bash
python3 qa_bsd_local_euler_batch_cert_v1/validator.py --self-test
```

Expected: `self_test_ok: true`.
