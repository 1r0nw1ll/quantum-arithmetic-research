# [36] QA Mapping Protocol REF

## What this is

Defines **QA_MAPPING_PROTOCOL_REF.v1**: a tiny deterministic reference object that lets certificate families point at a shared `QA_MAPPING_PROTOCOL.v1` mapping object.

This supports low-friction adoption:

- families can start with `mapping_protocol_ref.json`
- later upgrade to a self-contained `mapping_protocol.json` if needed

## Artifacts

| Artifact | Path |
|----------|------|
| REF schema | `qa_mapping_protocol_ref/schema.json` |
| REF validator | `qa_mapping_protocol_ref/validator.py` |
| REF root file (example) | `qa_mapping_protocol_ref/mapping_protocol_ref.json` |
| Valid reference fixture | `qa_mapping_protocol_ref/fixtures/valid_ref.json` |
| Invalid reference fixture | `qa_mapping_protocol_ref/fixtures/invalid_ref_missing_path.json` |

All paths relative to repository root.

## How to run

```bash
# From repo root:
python qa_mapping_protocol_ref/validator.py --self-test

# Or via meta-validator (runs as test [36])
cd qa_alphageometry_ptolemy
python qa_meta_validator.py
```

## Semantics

The REF validator enforces:

- schema validity (`QA_MAPPING_PROTOCOL_REF.v1`)
- `ref_path` must resolve within repo root (no path escape)
- referenced mapping file must exist
- optional `ref_sha256` must match file-bytes sha256 (tamper detection)
- referenced mapping must validate as `QA_MAPPING_PROTOCOL.v1`
- referenced mapping must include the v1 determinism contract essentials

## Changelog

- **v1** (2026-02-14): Initial schema + validator + fixtures; wired into meta-validator family sweeps.

