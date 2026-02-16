# QA Mapping Protocol REF (Machine Tract)

This folder defines **QA_MAPPING_PROTOCOL_REF.v1**: a tiny, deterministic reference object used by certificate families to point at a shared `QA_MAPPING_PROTOCOL.v1` mapping.

## Files

- `schema.json`: JSON Schema for `QA_MAPPING_PROTOCOL_REF.v1`
- `validator.py`: validator for refs (resolves path, optional sha256 pin, validates referenced mapping)
- `fixtures/valid_ref.json`: valid reference fixture (points at canonical mapping object)
- `fixtures/invalid_ref_missing_path.json`: invalid ref fixture (missing `ref_path`)

## How to run

```bash
python qa_mapping_protocol_ref/validator.py --self-test
python qa_mapping_protocol_ref/validator.py qa_mapping_protocol_ref/fixtures/valid_ref.json
```

