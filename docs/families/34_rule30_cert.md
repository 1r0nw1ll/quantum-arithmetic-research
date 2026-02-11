# Family [34] — Rule 30 Certified Discovery

## Purpose

Certify bounded nonperiodicity of Rule 30's center column through
machine-verifiable witness data with hash-locked manifests.

## Scope

| Parameter | Value |
|-----------|-------|
| Rule | 30 (ECA) |
| Initial condition | single 1 at position 0 |
| Period range | p in [1, 1024] |
| Time horizons | T in {4096, 8192, 16384, 32768, 65536} |
| Claim | No period p in [1,1024] detected at any T value |

## Schemas

- `QA_RULE30_NONPERIODICITY_CERT_SCHEMA.v1` — top-level cert referencing
  witness manifests and discovery pipeline bundles
- `QA_RULE30_WITNESS_MANIFEST.v1` — per-T witness data manifest with
  SHA-256 file hashes

## Machine Tract

| Component | Path |
|-----------|------|
| Cert schema | `qa_rule30/schemas/QA_RULE30_NONPERIODICITY_CERT_SCHEMA.v1.json` |
| Cert validator | `qa_rule30/qa_rule30_cert_validator.py` (9 self-tests) |
| Witness generator | `qa_rule30/generate_witnesses.py` (numpy-vectorized) |
| Plan generator | `qa_rule30/generate_rule30_plan.py` (batch plan factory) |
| Certpack assembler | `qa_rule30/assemble_certpack.py` (end-to-end) |
| Cert pack v1 | `qa_rule30/certpacks/rule30_nonperiodicity_v1/` |

## Validator Gates

### Cert gates (6)
1. `invariant_diff` presence → MISSING_INVARIANT_DIFF
2. Schema shape → SCHEMA_INVALID
3. Scope consistency (P_max >= P_min, k_range) → SCOPE_INVALID
4. Aggregate vs witness_refs consistency → AGGREGATE_MISMATCH
5. Hash chain integrity → HASH_CHAIN_INVALID
6. Zero failures across all witness_refs → FAILURE_DETECTED

### Manifest gates (3)
1. Schema shape → SCHEMA_INVALID
2. File hashes match on disk → HASH_MISMATCH
3. Failure count == 0 → FAILURE_DETECTED

## Witness Generation

The generator evolves Rule 30 using numpy-vectorized operations:
```
new[i] = left XOR (center OR right)
```

For each period p, it finds the smallest t where center(t) != center(t+p).
If found, that (p, t) pair is a counterexample witness proving center is
not periodic with period p within [0, T].

Performance: T=65536 completes in ~5 seconds on commodity hardware.

## Cert Packs

### v1 (P_max=256)
- 1280 / 1280 periods verified (5 T values x 256 periods)
- 0 failures
- Cert self-hash: `a612e01acba27b39f45d1239d83d35e8f838857e8db7dec8bac125c6a2bfedcc`

### v2 (P_max=1024)
- 5120 / 5120 periods verified (5 T values x 1024 periods)
- 0 failures
- Cert self-hash: `05775f6a412afdf9bf961eda6b7a1f5d89953e28d17c0058e1fb4f9266c0c675`

Both cert packs are witness-manifest file-hash verified and independently
verified by `verify_certpack.py` (zero-trust recomputation).

## Negative Fixtures

Three negative fixtures in `qa_rule30/fixtures/` prove gates are enforced
(meta-validator checks exit code + fail marker + reason substring):

- `cert_neg_missing_invariant_diff.json` → MISSING_INVARIANT_DIFF
- `cert_neg_scope_invalid.json` → SCOPE_INVALID
- `cert_neg_aggregate_mismatch.json` → AGGREGATE_MISMATCH

## Verification

Independent verifier — trusts nothing except the Rule 30 truth table:

```bash
# Full verification (~7 seconds)
python qa_rule30/verify_certpack.py qa_rule30/certpacks/rule30_nonperiodicity_v1

# With center sequence byte-for-byte check
python qa_rule30/verify_certpack.py qa_rule30/certpacks/rule30_nonperiodicity_v1 --full

# Spot-check a single T value
python qa_rule30/verify_certpack.py qa_rule30/certpacks/rule30_nonperiodicity_v1 --T 4096
```

The verifier:
1. Recomputes Rule 30 evolution from scratch (numpy-vectorized)
2. Checks every witness `(p, t)` pair: `center(t) != center(t+p)`
3. Verifies manifest file hashes match on disk
4. Optionally verifies center sequence files byte-for-byte

Public digest: `qa_rule30/certpacks/rule30_nonperiodicity_v1/DIGEST.json`
