# Family [84] — QA BSD Partial L-series Proxy Cert v1

**Machine tract:** `qa_bsd_partial_lseries_proxy_cert_v1/`  
**Schema version:** `QA_BSD_PARTIAL_LSERIES_PROXY_CERT.v1`

---

## What it certifies

This family certifies a deterministic finite proxy built from local Euler data:

\[
\text{proxy\_product} = \prod_{p \in P}\frac{\#E(\mathbb{F}_p)}{p}.
\]

The proxy is represented as an exact **non-reduced** product object:

- `numerator_factors`
- `denominator_factors`
- `numerator`
- `denominator`

It consumes an embedded [83]-compatible source batch and rechecks per-prime data
plus manifest hashes before proxy validation.

---

## Gate summary

| Gate | What it checks |
|---|---|
| Gate 1 | JSON schema validity |
| Gate 2 | Deterministic per-prime recomputation from source batch |
| Gate 3 | Source batch manifest binding (`record_hashes`, `batch_sha256`) |
| Gate 4 | Exact non-reduced proxy equality (`factors`, `numerator`, `denominator`) |

---

## Failure taxonomy

| `fail_type` | Trigger condition |
|---|---|
| `SCHEMA_INVALID` | Schema/type/required-field mismatch |
| `DUPLICATE_PRIME` | Duplicate prime detected in source batch records |
| `RECOMPUTE_MISMATCH` | Claimed local invariant or proxy value mismatches deterministic recomputation |
| `MANIFEST_MISMATCH` | Claimed manifest hashes mismatch recomputed manifest |

---

## Fixtures

| File | Type | Notes |
|---|---|---|
| `pass_proxy_p5_p7.json` | PASS | Exact proxy from primes `5,7` with full local claim fields |
| `pass_proxy_p5_p11.json` | PASS | Exact proxy from primes `5,11` with mixed local claim granularity |
| `fail_wrong_proxy_denominator.json` | FAIL | Corrupted proxy denominator (`36` instead of `35`) |

---

## Self-test

```bash
python3 qa_bsd_partial_lseries_proxy_cert_v1/validator.py --self-test
```

Expected: `self_test_ok: true`.
