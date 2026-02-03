# Golden Test Vectors

This directory contains golden test vectors that enforce protocol-level specifications.
Any drift in canonicalization, hashing, or merkle computation will break these tests.

## Purpose

Golden fixtures prevent:
- Canonicalization drift (someone "fixes" json.dumps params)
- Hash domain confusion (wrong function used for wrong purpose)
- Merkle ordering changes (someone sorts differently)
- Protocol spec drift without explicit migration

## Fixture Format

Each fixture is a JSON file with this structure:

```json
{
  "fixture_id": "qa.fixture.<family>.<name>.v1",
  "description": "What this fixture tests",
  "hash_spec_id": "qa.hash_spec.v1",

  "input": {
    "object": { ... }  // The JSON object to canonicalize
  },

  "expected": {
    "canonical_json_compact": "exact string output",
    "sha256_canonical": "64 hex chars",
    "merkle_leaf": "optional: leaf string if applicable",
    "merkle_root": "optional: root if testing merkle"
  }
}
```

## Adding a New Fixture

1. Create the input object
2. Compute expected values using qa_cert_core functions
3. Verify manually that the values are correct
4. Add the fixture file: `<family>_<name>.json`
5. Run `pytest tests/test_golden_vectors.py` to verify

## Hash Domains

| Domain | Function | Purpose |
|--------|----------|---------|
| semantic_identity | `sha256_canonical()` | Manifest canonical_sha256 |
| file_integrity | `sha256_file()` | Manifest sha256 |
| cert_id | `certificate_hash()` | Tetrad cert IDs |

## Merkle Specification

- Leaf ordering: `sorted(keys)` lexicographically
- Leaf format: `sha256("{name}:{canonical_hash}:{result_label}")`
- Tree construction: pairwise hashing until single root

## Migration

If a fixture breaks intentionally (protocol upgrade):
1. Bump `hash_spec.id` in manifests (e.g., `qa.hash_spec.v2`)
2. Update fixture `hash_spec_id` field
3. Document the change in `docs/PROTOCOL_CHANGELOG.md`
