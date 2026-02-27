# Family [85]: QA D_QA PAC Bound Kernel Cert v1

**Schema:** `QA_DQA_PAC_BOUND_KERNEL_CERT.v1`
**Root:** `qa_dqa_pac_bound_kernel_cert_v1/`
**formula_id:** `PAC_BAYES_QA_DQA_LOGDELTA_V1`
**Mapping protocol:** `mapping_protocol_ref.json` → `qa_mapping_protocol/canonical_mapping_protocol.json`

## Purpose

Single source of truth for the QA_DQA PAC bound computation kernel.
Where Family [84] certifies the *constants* used in the Phase-1 paper (K₁, improvement ratio,
DPI scope), Family [85] certifies the *formula itself* as a reusable kernel — ensuring that
every downstream consumer computes exactly the same bound from the same inputs.

**Locked invariants:**

1. **Formula identity** — `formula_id=PAC_BAYES_QA_DQA_LOGDELTA_V1` is the canonical name
2. **Log term** — `ln(1/δ)` (not `ln(m/δ)`, not `ln(2m/δ)`)
3. **Rounding** — `ROUND_HALF_UP` to 1 decimal place (not Python banker's rounding)
4. **Per-case numeric recomputation** — each case's witness intermediates are verified
5. **Cross-case monotonicity** — bound increases with D_QA and with `ln(1/δ)` (i.e. smaller δ)

## Kernel Formula

```
L     = ln(1/delta)
slack = sqrt((D_QA + L) / (2 * m))
bound = risk_hat + slack
bound_clipped = min(1.0, max(0.0, bound))
bound_percent_unrounded = 100.0 * bound_clipped
bound_percent = ROUND_HALF_UP(bound_percent_unrounded, decimals=1)
```

where `ROUND_HALF_UP(x, d) = floor(x × 10^d + 0.5) / 10^d`.

Note: this is a simplified variant of the Family [84] formula — there is no K₁ multiplier.
D_QA already incorporates the QA complexity (passed pre-multiplied by the caller).

## Triple Digest Architecture

This cert carries **three** SHA-256 digests:

| Digest | Covers |
|--------|--------|
| `canonical_sha256` | Entire cert (self-referential, canonical_sha256 zeroed during compute) |
| `kernel_block_sha256` | `cert.kernel` block only (tamper-evidence for formula lock) |
| `schema_sha256` | `schema.json` file bytes (locks schema version in cert) |

## Certified Cases

| case_id | D_QA | m | δ | risk_hat | bound_percent |
|---------|------|---|---|----------|--------------|
| `base` | 67.06 | 1000 | 0.05 | 0.08 | 26.7% |
| `smaller_delta` | 67.06 | 1000 | 0.01 | 0.08 | 26.9% |
| `bigger_DQ` | 100.13 | 1000 | 0.05 | 0.08 | 30.7% |

The `base` case uses the Phase-1 tight-experiment average D_QA (≈ 67.06) at δ=0.05.
`bigger_DQ` uses the Phase-1 initial-experiment average D_QA (≈ 100.13) to confirm
monotone increase with D_QA.

## Gate Architecture (5 gates)

| Gate | Name | What it checks |
|------|------|----------------|
| 1 | `gate_1_schema_validity` | JSON schema `QA_DQA_PAC_BOUND_KERNEL_CERT.v1.schema.json` |
| 2 | `gate_2_digest_integrity` | Triple digest: canonical_sha256 → kernel_block_sha256 → schema_sha256 |
| 3 | `gate_3_kernel_definition_lock` | All kernel constants (name, version, formula_id, log_term, rounding, tolerance) match locked values |
| 4 | `gate_4_case_recompute` | Each case: recomputes L, unrounded%, bound% and checks vs witness intermediates + expected |
| 5 | `gate_5_cross_case_sanity` | Monotonicity: smaller δ → larger bound; larger D_QA → larger bound |

## Failure Modes

| Label | Gate | Meaning |
|-------|------|---------|
| `SCHEMA_INVALID` | 1 | Certificate does not match schema |
| `DIGEST_MISMATCH` | 2 | Tampered cert (canonical, kernel_block, or schema digest wrong) |
| `KERNEL_DEFINITION_DRIFT` | 3 | Kernel constants changed (formula_id, log_term, rounding, tolerance) |
| `CASE_RECOMPUTE_MISMATCH` | 4 | Witness intermediates or expected.bound_percent inconsistent with recomputation |
| `KERNEL_SANITY_VIOLATION` | 5 | Monotonicity violated across cases |

## Fixtures

| File | Expected | Trigger |
|------|----------|---------|
| `valid_dqa_pac_bound_kernel_v1.json` | PASS | All gates pass, 3 cases |
| `invalid_digest_mismatch.json` | FAIL gate 2 | `canonical_sha256` set to placeholder zeros |
| `invalid_wrong_log_term.json` | FAIL gate 4 | Case `base` witness uses `ln(m/δ)` (=9.903) instead of `ln(1/δ)` (=2.996); bound_percent 27.6% vs correct 26.7% (diff=0.9 > tol=0.1) |

## Usage

```bash
# Self-test
python3 qa_dqa_pac_bound_kernel_cert_v1/validator.py --self-test

# JSON output
python3 qa_dqa_pac_bound_kernel_cert_v1/validator.py --self-test --json

# Validate a specific certificate
python3 qa_dqa_pac_bound_kernel_cert_v1/validator.py path/to/cert.json
```

## Relationship to Family [84]

| | Family [84] | Family [85] |
|---|---|---|
| What is locked | K₁ constant, improvement ratio, DPI scope | Bound formula itself (kernel) |
| Inputs | N, M, C, D_QA, m, δ, risk_hat + K₁ | D_QA, m, δ, risk_hat (no K₁) |
| Formula variant | `risk_hat + sqrt((K1·D_QA + ln(1/δ))/m)` | `risk_hat + sqrt((D_QA + ln(1/δ))/(2m))` |
| Scope | Phase-1 paper constants | Reusable kernel for downstream certs |

## Provenance

- Source paper: `papers/in-progress/phase1-pac-bayes/phase1_workspace/pac_bayes_qa_theory_complete.tex`
- Audit date: 2026-02-27 (revet campaign)
- JSON results: `signal_pac_results_tight.json` (D_QA values)
- Related cert: Family [84] `84_pac_bayes_constant_cert.md`
