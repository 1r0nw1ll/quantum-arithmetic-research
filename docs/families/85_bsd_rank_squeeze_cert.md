# Family [85] — QA BSD Rank Squeeze Cert v1

**Machine tract:** `qa_bsd_rank_squeeze_cert_v1/`  
**Schema version:** `QA_BSD_RANK_SQUEEZE_CERT.v1`

---

## What it certifies

This family certifies a discrete **rank squeeze trace** over a certified local Euler batch:

- local record recomputation is valid per prime;
- source manifest hashes are bound;
- exact non-reduced proxy product \(\prod_{p \in P} \#E(\mathbb{F}_p)/p\) is consistent;
- rank interval traces are coherent:
  - `lower_bounds` is monotone non-decreasing,
  - `upper_bounds` is monotone non-increasing,
  - and each step satisfies `lower <= upper`.

`squeeze_closed=true` requires an observed closing step where lower and upper meet.

---

## Gate summary

| Gate | What it checks |
|---|---|
| Gate 1 | JSON schema validity |
| Gate 2 | Deterministic per-prime recomputation from source batch |
| Gate 3 | Source batch manifest binding (`record_hashes`, `batch_sha256`) |
| Gate 4 | Exact non-reduced proxy equality (`factors`, `numerator`, `denominator`) |
| Gate 5 | Rank trace consistency (`len`, monotonicity, interval crossing, closure semantics) |

Local point-count recomputation uses an \(O(p)\) odd-prime method (Legendre-symbol based)
with exact handling for \(p=2\).

---

## Failure taxonomy

| `fail_type` | Trigger condition |
|---|---|
| `SCHEMA_INVALID` | Schema/type/required-field mismatch |
| `DUPLICATE_PRIME` | Duplicate prime detected in source batch records |
| `RECOMPUTE_MISMATCH` | Claimed local invariant or proxy value mismatches recomputation |
| `MANIFEST_MISMATCH` | Claimed manifest hashes mismatch recomputed manifest |
| `TRACE_LENGTH_MISMATCH` | `lower_bounds` and `upper_bounds` lengths differ |
| `TRACE_MONOTONICITY_VIOLATION` | Lower decreases or upper increases along the trace |
| `TRACE_INCONSISTENT` | Any interval crossing (`lower > upper`) or closure claim mismatch |

---

## Fixtures

| File | Type | Notes |
|---|---|---|
| `pass_closed_p5_p7.json` | PASS | Squeeze closes at step 1 for primes `5,7` |
| `pass_open_p5_p11.json` | PASS | Squeeze remains open for primes `5,11` |
| `fail_bad_trace_crossing.json` | FAIL | Interval crossing (`lower > upper`) |
| `fail_wrong_proxy_denominator.json` | FAIL | Gate 4 proxy mismatch (wrong denominator) |
| `fail_wrong_ap_p7.json` | FAIL | Gate 2 local recompute mismatch (`ap`) |

---

## Self-test

```bash
python3 qa_bsd_rank_squeeze_cert_v1/validator.py --self-test
```

Expected: `self_test_ok: true`.
