# Family [86]: QA PAC-Bayes DPI Scope Cert v1

**Schema:** `QA_PAC_BAYES_DPI_SCOPE_CERT.v1`
**Root:** `qa_pac_bayes_dpi_scope_cert_v1/`
**Mapping protocol:** `mapping_protocol_ref.json` → `qa_mapping_protocol/canonical_mapping_protocol.json`

## Purpose

Locks the **DPI scope claim** for the Phase-1 PAC-Bayes paper to `structured_only`.
This is the cert that makes it impossible to write "universal DPI" in a LaTeX document
without a corresponding cert update.

Certifies four invariants:

1. **Structured distribution definition** — `STRUCTURED_ONLY_V1` with explicit generator, seed, orbit family, and constraints
2. **Empirical evidence** — violation rates are numerically recomputed from `n_violations / n_trials`; structured passes, random fails at expected rate
3. **Scope separation** — structured violation rate ≤ pass_threshold AND random violation rate ≥ min_expected (the two domains are demonstrably different)
4. **Claim policy** — `dpi_claim="structured_only"`, `claim_level ∈ {empirical_only,proof_sketch,open}`, `forbidden_phrases` must include `"universal"`, `"proven"`, and `"for all distributions"`

## Certified Evidence

| Experiment | Trials | Violations | Violation rate | Threshold |
|-----------|--------|-----------|---------------|-----------|
| Structured (seed=42, Cosmos) | 5 | 0 | 0.0% | ≤ 0.0% ✓ |
| Random (UNSTRUCTURED_RANDOM_V1) | 250 | 129 | 51.6% | ≥ 10.0% ✓ |
| Multi-step k=5 (structured) | 5 | 0 | 0.0% | — (audit only) |

The separation is enforced by Gate 4: structured must pass, random must fail.

## Cross-Family References

This cert explicitly references:
- **Family [84]** `QA_PAC_BAYES_CONSTANT_CERT.v1.1` — constants cert
- **Family [85]** kernel block `PAC_BAYES_QA_DQA_LOGDELTA_V1` (sha256 `553b7588...`) — kernel cert

These are stored in `refs` but not numerically verified by the validator (they are metadata for traceability). Gate 5 does not recheck them — that binding is enforced at the [84] cert level.

## Gate Architecture (5 gates)

| Gate | Name | What it checks |
|------|------|----------------|
| 1 | `gate_1_schema_validity` | JSON schema `QA_PAC_BAYES_DPI_SCOPE_CERT.v1.schema.json` |
| 2 | `gate_2_digest_integrity` | Self-referential canonical SHA-256 |
| 3 | `gate_3_evidence_rate_consistency` | Recomputes `violation_rate = n_violations / n_trials` for structured + random (+ multistep if enabled); abs_tol=1e-12; enforces `0 ≤ n_violations ≤ n_trials` |
| 4 | `gate_4_scope_separation_assertion` | `structured.violation_rate ≤ pass_threshold` AND `random.violation_rate ≥ min_expected_violation_rate` |
| 5 | `gate_5_claim_policy_enforcement` | `dpi_claim == "structured_only"`; `forbidden_phrases` ⊇ `{"universal","proven","for all distributions"}` |

## Failure Modes

| Label | Gate | Meaning |
|-------|------|---------|
| `SCHEMA_INVALID` | 1 | Certificate does not match schema |
| `DIGEST_MISMATCH` | 2 | Tampered certificate |
| `EVIDENCE_RATE_MISMATCH` | 3 | Stored violation_rate ≠ n_violations/n_trials (> 1e-12 abs); or n_violations out of bounds |
| `SCOPE_SEPARATION_VIOLATION` | 4 | Structured violation rate exceeds threshold, or random violation rate below minimum expected |
| `CLAIM_POLICY_VIOLATION` | 5 | `dpi_claim` ≠ `"structured_only"`, or required forbidden phrases missing from list |

## Fixtures

| File | Expected | Trigger |
|------|----------|---------|
| `valid_pac_bayes_dpi_scope_v1.json` | PASS | All gates pass |
| `invalid_random_trials_too_clean.json` | FAIL gate 4 | random_trials: n_trials=100, n_violations=0, violation_rate=0.0 — fails separation (0.0 < min_expected 0.10) |
| `invalid_claim_policy_missing_forbidden_phrase.json` | FAIL gate 5 | `forbidden_phrases=["for all distributions","proven"]` — missing `"universal"` |

## Structured Distribution Definition

`definition_id: STRUCTURED_ONLY_V1`
- Generator: `QA_STRUCTURED_DEMO_GENERATOR v1`
- Seed: 42
- N=16, M=24
- Family: `Cosmos_24_cycle`
- Constraints: `orbit_family=Cosmos`, `reachable_from_seed_start`, `no_satellite_mixing`
- Canonicalization: `CANONICAL_STRUCTURED_V1`

This definition is what separates "structured" from "unstructured random".
Any change to these parameters requires a new cert.

## Usage

```bash
# Self-test
python3 qa_pac_bayes_dpi_scope_cert_v1/validator.py --self-test

# JSON output
python3 qa_pac_bayes_dpi_scope_cert_v1/validator.py --self-test --json

# Validate a specific certificate
python3 qa_pac_bayes_dpi_scope_cert_v1/validator.py path/to/cert.json
```

## Provenance

- Source paper: `papers/in-progress/phase1-pac-bayes/phase1_workspace/pac_bayes_qa_theory_complete.tex`
- Audit date: 2026-02-27 (revet campaign)
- Evidence source: `DPI_REFINEMENT_RESULTS.md` (DPI_REFINEMENT section)
- Related certs: [84] `84_pac_bayes_constant_cert.md`, [85] `85_dqa_pac_bound_kernel_cert.md`
