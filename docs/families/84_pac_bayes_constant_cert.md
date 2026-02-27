# Family [84]: QA PAC-Bayes Constant Cert v1

**Schema:** `QA_PAC_BAYES_CONSTANT_CERT.v1`
**Root:** `qa_pac_bayes_constant_cert_v1/`
**Mapping protocol:** `mapping_protocol_ref.json` → `qa_mapping_protocol/canonical_mapping_protocol.json`

## Purpose

Machine-tract lock for the Phase-1 PAC-Bayes theory paper (`pac_bayes_qa_theory_complete.tex`).
Certifies four invariants that were identified during the 2026-02-27 revet audit:

1. **K₁ formula correctness** — recomputes K₁ = 2C²N(M/2)² from raw system parameters
2. **PAC bound recomputation** — derives bound percent directly from (D_QA, m, δ, risk_hat)
3. **Improvement ratio consistency** — verifies 5585% → 1813% = 3.1× (not 3.2×)
4. **DPI scope declaration** — enforces `claim="structured_only"` (universal is prohibited)

## Corrected K₁ Formula

The Phase-1 paper contained an arithmetic error:

| | Formula shown | Arithmetic |
|---|---|---|
| **Old (wrong)** | `2 × 1 × 24 × 24² = 6912` | 2 × 24 × 576 = **27,648** |
| **Fixed** | `2 × 1² × 24 × (24/2)² = 6912` | 2 × 24 × 144 = **6912** ✓ |

The validator enforces: `K1_formula == "2*C^2*N*(M/2)^2"` and recomputes
`K1_calc = 2 * C² * N * (M/2)²` from the cert's system fields.

For N=16 (actual experiments): `K1 = 2 × 1 × 16 × 144 = 4608`
For N=24 (theoretical reference): `K1 = 2 × 1 × 24 × 144 = 6912`

## PAC Bound Variant: QA_DQA

The Phase-1 scripts compute the bound using `log(1/δ)` (not `log(m/δ)`):

```
bound = risk_hat + sqrt((K1 * D_QA + log(1/delta)) / m)
bound_percent = round(bound * 100, percent_round_dp)
```

For the certified fixture (Pure Tone, tight experiment):
- N=16, M=24, C=1, K1=4608, D_QA=57.2596, m=1000, δ=0.05, risk_hat=1.0
- bound = 1.0 + sqrt((4608×57.2596 + 2.9957)/1000) = 17.244 → **1724%**

## Certified Results

| Experiment | K₁ | m | D_QA (avg) | Bound % (avg) |
|-----------|-----|---|-----------|--------------|
| Uniform prior, m=150 | 4608 | 150 | 100.13 | 5585% |
| Informed prior, m=1000 | 4608 | 1000 | 66.86 | 1813% |

**Improvement ratio: 5585 / 1813 = 3.1×** (corrected from earlier incorrect 3.2×)

## DPI Scope

The Data Processing Inequality was validated for structured QA distributions (seed=42)
but fails on **51.8% of random initial distributions** (100 random trials). The cert
enforces `dpi.claim="structured_only"` to prevent future over-claims.

| | Structured (seed=42) | 100 random trials |
|---|---|---|
| Single-step DPI | ✓ PASS | 51.8% FAIL |
| Multi-step (5 steps) | ✓ PASS (0/5 violations) | not tested |

## Gate Architecture (5 gates)

| Gate | Name | What it checks |
|------|------|----------------|
| 1 | `gate_1_schema_validity` | JSON schema `QA_PAC_BAYES_CONSTANT_CERT.v1.schema.json` |
| 2 | `gate_2_digest_integrity` | Self-referential SHA-256 |
| 3 | `gate_3_k1_recompute` | K1_calc = 2C²N(M/2)² matches K1_expected and K1_recomputed within tolerance_abs; K1_formula string correct; M even |
| 4 | `gate_4_pac_bound_recompute` | Recomputes bound_percent from (risk_hat, D_QA, m, delta, K1_calc, bound_variant); verifies improvement_ratio = round(initial/tight, rdp); tight < initial |
| 5 | `gate_5_dpi_scope` | `dpi.claim == "structured_only"`; violation_rate in [0,1] |

## Failure Modes

| Label | Gate | Meaning |
|-------|------|---------|
| `SCHEMA_INVALID` | 1 | Certificate does not match schema |
| `DIGEST_MISMATCH` | 2 | Tampered certificate |
| `K1_RECOMPUTE_MISMATCH` | 3 | K1_expected/K1_recomputed don't match 2C²N(M/2)²; or M_is_even wrong; or K1_formula string wrong |
| `PAC_BOUND_RECOMPUTE_MISMATCH` | 4 | bound_percent inconsistent with recomputed value; improvement_ratio wrong; tight ≥ initial |
| `DPI_SCOPE_VIOLATION` | 5 | `dpi.claim` is not `"structured_only"` |

## Fixtures

| File | Expected | Trigger |
|------|----------|---------|
| `valid_pac_bayes_constant_v1.json` | PASS | All gates pass |
| `invalid_k1_mismatch.json` | FAIL gate 3 | K1_expected=27648 (old wrong 2×N×M² path) vs correct 4608 |
| `invalid_dpi_claim_universal.json` | FAIL gate 5 | `dpi.claim="universal"` (overclaims universality despite 51.8% violation rate) |

## Usage

```bash
# Self-test
python3 qa_pac_bayes_constant_cert_v1/validator.py --self-test

# Validate a specific certificate
python3 qa_pac_bayes_constant_cert_v1/validator.py path/to/cert.json

# JSON output
python3 qa_pac_bayes_constant_cert_v1/validator.py --self-test --json
```

## Provenance

- Source paper: `papers/in-progress/phase1-pac-bayes/phase1_workspace/pac_bayes_qa_theory_complete.tex`
- Audit date: 2026-02-27 (revet campaign)
- JSON results: `signal_pac_results.json`, `signal_pac_results_tight.json`
- Related audit doc: `memory/pac_bayes_audit.md`
