# [20] QA Datastore View

## What this is

Certifies that datastore **view projections** (e.g., posting lists indexed by `phase_24`) are sound derivations from the underlying store. A view is a certified read path — it projects store entries into lookup-optimized structures while maintaining Merkle-anchored provenance back to the store root. Depends on [18] QA Datastore.

## Artifacts

| Artifact | Path |
|----------|------|
| View semantics cert | `certs/QA_DATASTORE_VIEW_CERT.v1.json` |
| Witness pack | `certs/witness/QA_DATASTORE_VIEW_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1.json` |
| Validator | `qa_datastore_view_validator.py` |
| Semantics schema | `schemas/QA_DATASTORE_VIEW_CERT.v1.schema.json` |
| Witness schema | `schemas/QA_DATASTORE_VIEW_WITNESS_PACK.v1.schema.json` |
| Counterexamples schema | `schemas/QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1.schema.json` |
| View proof schema | `schemas/QA_VIEW_PROOF.v1.schema.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate datastore view family
python qa_datastore_view_validator.py --demo

# Or via meta-validator (runs as test [20])
python qa_meta_validator.py
```

## Semantics

- **View kind**: `POSTING_LIST` — entries indexed by a projection key
- **Projection**: `phase_24_posting_view` in `SCALE_PRESERVING` mode
- **Dual-root binding**: every view entry binds to both a store Merkle root and a view Merkle root
- **Hash domains**: `cert`, `merkle_node`, `posting`, `view_leaf`
- **Merkle padding**: `DUPLICATE_LAST` for odd leaf counts
- **Store semantics ref**: depends on `QA_DATASTORE_SEMANTICS_CERT.v1` ([18])

### Hash computation rules (invariants)

1. Manifest self-hash uses zeroed `canonical_json_sha256` (HEX64_ZERO placeholder)
2. `posting_hash = ds_sha256('posting', canonical_json(posting))`
3. `view_leaf_hash = ds_sha256('view_leaf', posting_hash + view_key)`
4. Internal Merkle nodes: `ds_sha256('merkle_node', left_hash + right_hash)`
5. Non-inclusion proofs: adjacent key binding

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `KEY_NOT_FOUND` | View key absent | Provide non-inclusion proof or add entry |
| `HASH_MISMATCH` | Recomputed hash differs from declared | Use `--rehash` to recompute |
| `SCHEMA_MISMATCH` | Entry doesn't match view schema | Fix entry structure |
| `DOMAIN_SEP_VIOLATION` | Wrong domain prefix in hash | Use correct domain string |
| `NON_CANONICAL_JSON` | JSON not canonical | Re-serialize canonically |
| `UNVERIFIABLE_PROOF` | Merkle proof invalid | Regenerate from snapshot |
| `FORK_DETECTED` | Conflicting roots for same view | Resolve to single root |
| `MIGRATION_NON_INVERTIBLE` | Lossy scale transform | Use invertible projection |
| `SCALE_COLLAPSE` | Degenerate projection | Fix projection parameters |

## Changelog

- **v1.3.1** (2026-02-08): Initial triplet shipped with meta-validator hook [20].
