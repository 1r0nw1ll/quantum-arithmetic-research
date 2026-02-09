# [18] QA Datastore

## What this is

Certifies that the QA datastore — the key-value store underlying all retrieval operations — obeys its semantic contract: composite QA keys, domain-separated hashing, Merkle-anchored integrity, and generator-controlled mutations (`put`, `get`, `del`, `migrate`, `compact`). This is the foundational storage layer that [20], [21], [22], and [23] all depend on.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__DATASTORE.yaml` |
| Semantics cert | `certs/QA_DATASTORE_SEMANTICS_CERT.v1.json` |
| Witness pack | `certs/witness/QA_DATASTORE_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_DATASTORE_COUNTEREXAMPLES_PACK.v1.json` |
| Validator | `qa_datastore_validator.py` |
| Snapshot builder | `qa_datastore_build_snapshot.py` |
| Merkle proof schema | `schemas/QA_MERKLE_PROOF.v1.schema.json` |
| Semantics schema | `schemas/QA_DATASTORE_SEMANTICS_CERT.v1.schema.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate full datastore family (semantics + witness + counterexamples)
python qa_datastore_validator.py --demo

# Or via meta-validator (runs as test [18])
python qa_meta_validator.py
```

## Semantics

The datastore semantics cert (`QA_DATASTORE_SEMANTICS_CERT.v1`) declares:

- **Hash algorithm**: SHA256 with domain separation
- **Hash domains**: `cert`, `keyed_leaf`, `merkle_node`, `record`
- **Canonical JSON**: `json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`
- **Generators**: `put`, `get`, `del`, `migrate`, `compact` over store states
- **Composite QA key**: (`family`, `phase_24`, `phase_9`, tuple, field)
- **Merkle root**: global invariant anchor binding all entries

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `KEY_NOT_FOUND` | QA key absent from store (valid only with non-inclusion proof) | Provide verified non-inclusion proof or ingest the missing entry |
| `HASH_MISMATCH` | Declared hash does not match recomputed hash | Recompute using `--rehash` or fix the payload |
| `SCHEMA_MISMATCH` | Entry does not conform to declared schema | Fix entry structure to match schema |
| `DOMAIN_SEP_VIOLATION` | Hash computed without correct domain prefix | Use `ds_sha256(domain, payload)` |
| `NON_CANONICAL_JSON` | JSON not in canonical form | Re-serialize with `sort_keys=True, separators=(',',':')` |
| `UNVERIFIABLE_PROOF` | Merkle proof cannot be verified against root | Regenerate proof from current snapshot |
| `FORK_DETECTED` | Two valid entries share a key (conflicting roots) | Resolve to single authoritative root |
| `MIGRATION_NON_INVERTIBLE` | Scale migration loses information | Use invertible migration only |
| `SCALE_COLLAPSE` | Scale operation degenerates state | Prevent degenerate scale transforms |

## Example

**Passing** — a witness entry with valid Merkle proof:
```json
{
  "doc_id": "qa_agentic_rag.odt",
  "doc_record_hash": "<ds_sha256('QA/RECORD/v1', canonical_json)>",
  "leaf_hash": "<ds_sha256('QA/INGEST_LEAF/v1', doc_id + '\\x00' + record_hash)>"
}
```

**Failing** — `DOMAIN_SEP_VIOLATION`:
```json
{
  "doc_record_hash": "<plain sha256 without domain prefix>",
  "expected_fail_type": "DOMAIN_SEP_VIOLATION"
}
```

## Changelog

- **v1.3.1** (2026-02-08): Initial triplet (semantics + witness + counterexamples) shipped with meta-validator hook [18].
