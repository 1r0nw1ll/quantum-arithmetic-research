# Family [83]: QA Bell CHSH Cert v1

**Schema:** `QA_BELL_CHSH_CERT.v1`
**Root:** `qa_bell_chsh_cert_v1/`
**Mapping protocol:** `mapping_protocol_ref.json` → `qa_mapping_protocol/canonical_mapping_protocol.json`

## Purpose

Certifies the **"8|N theorem"**: the QA cosine correlator

```
E_N(s,t) = cos(2π(s-t)/N)
```

achieves the CHSH Tsirelson bound |S| = 2√2 ≈ 2.8284 **if and only if 8 divides N**.

This cert was created during the 2026-02-27 Bell test audit, which:
- **Verified** the CHSH result by exhaustive search over all N⁴ settings
- **Retracted** the I3322 "6|N theorem" (see `BELL_TESTS_FINAL_SUMMARY.md`) due to an
  unconstrained correlator model (`MODEL_NOT_PHYSICALLY_REALIZABLE`)

## Certified Results

| N  | 8\|N | max\|S\|   | Hit Tsirelson |
|----|------|-----------|---------------|
| 6  | ✗    | 2.500000  | ✗             |
| 8  | ✓    | **2.8284** | ✓             |
| 12 | ✗    | 2.732051  | ✗             |
| 16 | ✓    | **2.8284** | ✓             |
| 24 | ✓    | **2.8284** | ✓             |
| 32 | ✓    | **2.8284** | ✓             |

**Optimal N=24 settings:** (a, a', b, b') = (0, 6, 15, 21)
→ Alice: sectors 0 and 6 (0° and 90°), Bob: sectors 15 and 21 (225° and 315°)

## Physical Interpretation

The cosine correlator is **continuous-valued** (not binary ±1). This evades Bell's
theorem constraints, which apply to binary pre-assignments. The model is:

- **Deterministic**: no randomness, outcomes computed from hidden variable (clock position s or t)
- **Local**: no communication between Alice and Bob
- **Valid**: `model_not_physically_realizable = False` — it IS a physically realizable
  classical deterministic LHV model

This contrasts with the **retracted** I3322 implementation where correlators were
optimized independently without a single quantum state ρ (`MODEL_NOT_PHYSICALLY_REALIZABLE = True`).

## Gate Architecture (5 gates)

| Gate | Name | What it checks |
|------|------|----------------|
| 1 | `gate_1_schema_validity` | JSON schema `QA_BELL_CHSH_CERT.v1.schema.json` |
| 2 | `gate_2_canonical_hash` | Self-referential SHA-256 |
| 3 | `gate_3_divisibility` | `divisible_by_8 == (N % 8 == 0)` for each sweep entry |
| 4 | `gate_4_tsirelson_values` | `hit_tsirelson` consistent with `max_abs_S` vs. threshold 2√2−1e-6; 8\|N theorem enforced |
| 5 | `gate_5_model_assessment` | `model_valid=True`, `model_not_physically_realizable=False` |

## Failure Modes

| Label | Gate | Meaning |
|-------|------|---------|
| `SCHEMA_INVALID` | 1 | Certificate does not match schema |
| `HASH_MISMATCH` | 2 | Tampered certificate (canonical SHA-256 mismatch) |
| `DIVISIBILITY_MISMATCH` | 3 | `divisible_by_8` flag contradicts N%8 computation |
| `TSIRELSON_VALUE_MISMATCH` | 4 | `hit_tsirelson` flag inconsistent with `max_abs_S`, or 8\|N theorem violated |
| `MODEL_NOT_PHYSICALLY_REALIZABLE` | 5 | Correlator flagged as physically unrealizable (should be False) |
| `MODEL_INVALID` | 5 | `model_valid=False` |

## Fixtures

| File | Expected | Trigger |
|------|----------|---------|
| `valid_chsh_8n_theorem.json` | PASS | All gates pass (N=6,8,12,16,24,32 sweep) |
| `invalid_wrong_condition.json` | FAIL gate 3 | N=8 marked `divisible_by_8=False` (wrong) |
| `invalid_wrong_value.json` | FAIL gate 4 | N=6 has `max_abs_S=2.9` but `hit_tsirelson=False` (inconsistent) |

## Contrast with I3322 Retraction

| | CHSH [83] | I3322 (retracted) |
|-|-----------|-------------------|
| `model_not_physically_realizable` | **False** | True |
| Max score | 2.8284 (= 2√2) | ~6.0 (24× quantum bound) |
| Reason valid/invalid | Continuous-valued LHV, single hidden variable | Independent unconstrained correlators, no joint state ρ |
| Theorem status | **VERIFIED** (8\|N) | **RETRACTED** ("6\|N") |

## Usage

```bash
# Run self-test
python3 qa_bell_chsh_cert_v1/validator.py --self-test

# Validate a specific certificate
python3 qa_bell_chsh_cert_v1/validator.py path/to/cert.json

# JSON output
python3 qa_bell_chsh_cert_v1/validator.py --self-test --json
```

## Provenance

- Verification script: `qa_chsh_bell_test.py` (exhaustive N^4 search)
- Audit date: 2026-02-27
- Related: `BELL_TESTS_FINAL_SUMMARY.md`, `TSIRELSON_BOUND_RESEARCH_SUMMARY.md`
