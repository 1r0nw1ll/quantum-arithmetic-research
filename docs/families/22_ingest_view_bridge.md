# [22] QA Ingest->View Bridge

## What this is

Certifies the **provenance bridge** from ingested documents to datastore view entries. Every view entry must be grounded in an actual ingested document with a verifiable inclusion proof. This family composes [18] Datastore, [20] View, and [23] Ingestion to form the middle link of the end-to-end provenance chain.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__INGEST_VIEW_BRIDGE.yaml` |
| Semantics cert | `certs/QA_INGEST_VIEW_BRIDGE_CERT.v1.json` |
| Witness pack | `certs/witness/QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1.json` |
| Validator | `qa_ingest_view_bridge_validator.py` |
| Semantics schema | `schemas/QA_INGEST_VIEW_BRIDGE_CERT.v1.schema.json` |
| Witness schema | `schemas/QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1.schema.json` |
| Counterexamples schema | `schemas/QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1.schema.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate bridge family
python qa_ingest_view_bridge_validator.py --demo

# Or via meta-validator (runs as test [22])
python qa_meta_validator.py
```

## Semantics

### Hash chain (critical path)

The bridge certifies a four-layer hash chain:

```
source file bytes
    |
    v
doc_record = {doc_id, source_ref, file_hash, ...}
    |  canonical_json -> ds_sha256('QA/RECORD/v1', payload)
    v
doc_record_hash
    |  ds_sha256('QA/INGEST_LEAF/v1', doc_id + '\x00' + doc_record_hash)
    v
leaf_hash  -->  Merkle tree  -->  root_hash
```

**Changing any layer cascades through all downstream hashes.**

### Hash domains

| Domain | Used for |
|--------|----------|
| `QA/RECORD/v1` | `doc_record_hash` computation |
| `QA/INGEST_LEAF/v1` | `leaf_hash` computation |
| `QA/INGEST_NODE/v1` | Internal Merkle nodes |
| `QA/VIEW_NODE/v1` | View-side Merkle nodes |

### Bridge contract

- **Budget**: `max_entries=8`, `max_total_tokens=1200`
- **Typed view roots**: `KEYWORD_VIEW` and `SEMANTIC_VIEW` bind to specific snapshot IDs
- **Doc inclusion proofs**: every bridge entry cites document refs with Merkle inclusion proofs
- **Root binding**: proof bundle store/view roots must match certified snapshots

### Invariants

| Invariant | Meaning |
|-----------|---------|
| Document grounding | Every bridge entry cites doc refs with ingest inclusion proofs |
| Root binding | Proof store/view roots match certified snapshots |
| Typed view provenance | Views bind to typed snapshot IDs |
| Budget control | Entry count and token budget remain bounded |

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `HASH_MISMATCH` | Recomputed hash differs | Recompute with `--rehash` |
| `DOMAIN_SEP_VIOLATION` | Wrong domain prefix | Use correct domain string |
| `NON_CANONICAL_JSON` | Non-canonical JSON | Re-serialize canonically |
| `UNVERIFIABLE_PROOF` | Merkle proof fails | Regenerate from snapshot |
| `FORK_DETECTED` | Conflicting roots | Resolve root conflict |
| `SCHEMA_MISMATCH` | Entry doesn't match schema | Fix structure |
| `INGEST_DOC_MISSING` | Source file not found at `source_ref` | See "Gitignored sources" below |
| `INGEST_DOC_HASH_MISMATCH` | Source file hash changed | Re-ingest from original source |
| `VIEW_ENTRY_UNGROUNDED` | View entry has no doc provenance | Add document reference |
| `VIEW_DERIVATION_UNSOUND` | View derivation logic invalid | Fix derivation chain |
| `VIEW_DERIVATION_INCOMPLETE` | Missing derivation steps | Complete the derivation |
| `BUDGET_EXCEEDED` | Too many entries or tokens | Reduce bridge scope |

## Gitignored sources and committed mirrors

**Critical rule**: `source_ref` MUST NOT point into `ingestion candidates/` (gitignored at `.gitignore:131`).

When ingesting from that directory:
1. Copy the source file to `qa_ingestion_sources/` (committed to git)
2. Update `source_ref` in all witness and counterexample packs
3. Recompute the full hash chain: `doc_record_hash` -> `leaf_hash` -> `root_hash`
4. Use `--rehash` on the validator to recompute manifest hashes

The hash chain uses **domain-separated hashing**:
```python
def ds_sha256(domain: str, payload: bytes) -> str:
    return sha256(domain.encode() + b'\x00' + payload).hexdigest()
```

## Changelog

- **v1.3.1** (2026-02-08): Initial triplet shipped.
- **v1.4.1** (2026-02-09): Fixed source paths from gitignored `ingestion candidates/` to committed `qa_ingestion_sources/`. Recalculated domain-separated hashes across full chain.
