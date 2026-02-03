# Certificate Family Hardening Playbook

This document describes the standard pattern for hardening QA certificate modules, based on the Kayser reference implementation.

## Overview

A hardened certificate module provides three layers of integrity:

| Layer | What it checks | How to verify |
|-------|----------------|---------------|
| File integrity | File bytes unchanged | `sha256` in manifest |
| Semantic identity | JSON content unchanged | `canonical_sha256` in manifest |
| Behavioral integrity | Validation logic correct | `merkle_root` from validator |

## Required Components

### 1. Manifest (`qa_{module}_manifest.json`)

Every module must have a manifest with:

```json
{
  "schema_version": "QA_MANIFEST.v1",
  "manifest_id": "qa.manifest.{module}.v{N}",
  "hash_spec": {
    "id": "qa.hash_spec.v1",
    "version": "1.0",
    "canonical_spec": "json.dumps(obj, sort_keys=True, separators=(',',':'), ensure_ascii=False)",
    "source": "qa_cert_core.canonical_json_compact"
  },
  "certificates": {
    "{cert_name}": {
      "id": "qa.cert.{module}.{cert_name}.v1",
      "file": "qa_{module}_{cert_name}_cert.json",
      "sha256": "<64 hex chars>",
      "canonical_sha256": "<64 hex chars>",
      "evidence_level": "PROVEN|STRUCTURAL_PROVEN|STRUCTURAL_ANALOGY|ENGINEERING_VALIDATED",
      ...
    }
  },
  "validation_summary": {
    "merkle_root": "<64 hex chars>",
    "all_passed": true
  }
}
```

### 2. Validator (`qa_{module}_validate.py`)

Required CLI interface:

```bash
# Full validation
python qa_{module}_validate.py --all

# JSON output
python qa_{module}_validate.py --all --json

# Manifest integrity check (fast gate)
python qa_{module}_validate.py --check-manifest

# Summary
python qa_{module}_validate.py --summary
```

Required imports from `qa_cert_core`:

```python
from qa_cert_core import (
    canonical_json_compact,
    sha256_canonical,
    sha256_file,
    check_manifest_integrity,  # optional, for standardized manifest check
)
```

### 3. Schemas (optional but recommended)

Store in `schemas/`:
- `qa_manifest.schema.json` - manifest structure
- `qa_structural_analogy_test.schema.json` - test row format

## Evidence Levels

| Level | Meaning | Requirements |
|-------|---------|--------------|
| PROVEN | Mathematical isomorphism | Invariant-preserving mapping |
| STRUCTURAL_PROVEN | Structure preserved | Interpretation may differ |
| STRUCTURAL_ANALOGY | Partial alignment | Must include `limitation_class` |
| ENGINEERING_VALIDATED | Third-party connection | Must include benchmark data |
| CONJECTURAL | Suggestive resemblance | Requires formalization |

## For `structural_analogy` Certificates

### Required fields in test rows:

```json
{
  "id": "L1",
  "name": "Test Name",
  "result": "PASS|PARTIAL_MATCH|FAIL",
  "expected_outcome": "PASS|PARTIAL_MATCH|FAIL_EXPECTED"
}
```

### Required limitation_class:

If `evidence_level == "STRUCTURAL_ANALOGY"`, must include:
- `limitation_class` in certificate (from registry)
- `limitation` description

### Allowed limitation_class values:

```python
LIMITATION_CLASS_REGISTRY = {
    "GEOMETRY_MODEL_MISMATCH",   # Curved vs linear, different manifolds
    "DOMAIN_MISMATCH",           # Different input/output domains
    "PARAMETER_RANGE_LIMIT",     # Works in subset of parameter space
    "SYMMETRY_BREAK",            # Symmetry present in one, not other
    "CARDINALITY_MISMATCH",      # Different set sizes
    "TOPOLOGY_MISMATCH",         # Different topological structure
}
```

## Enum Registries

### Result enum:
```python
RESULT_ENUM = {"PASS", "PARTIAL_MATCH", "FAIL"}
```

### Expected outcome enum:
```python
EXPECTED_OUTCOME_ENUM = {"PASS", "PARTIAL_MATCH", "FAIL_EXPECTED"}
```

## Two-Phase Meta-Validator Integration

Add to `qa_meta_validator.py`:

```python
# Phase A: Fast manifest integrity check
result_a = subprocess.run([validator, "--check-manifest", "--json"], ...)

# Phase B: Full behavioral validation
result_b = subprocess.run([validator, "--all", "--json"], ...)
```

## Checklist for New Module

- [ ] Create manifest with `hash_spec.id = "qa.hash_spec.v1"`
- [ ] Add `sha256` (file bytes) for each certificate
- [ ] Add `canonical_sha256` for each certificate
- [ ] Implement `--check-manifest` mode in validator
- [ ] Import canonicalization from `qa_cert_core`
- [ ] Add to meta-validator test suite
- [ ] For structural_analogy: require `expected_outcome` and `limitation_class`

## Reference Implementation

See `qa_kayser/` for the complete reference implementation:
- `qa_kayser_manifest.json` - manifest with full hash_spec
- `qa_kayser_validate.py` - validator with all modes
- `qa_kayser_primordial_leaf_cert.json` - structural_analogy example
