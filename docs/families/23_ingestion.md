# [23] QA Ingestion

## What this is

Certifies the **document ingestion pipeline** — the deterministic transform from source files (`.odt`, `.txt`, `.md`) into normalized, chunked artifacts with cryptographic hashes. This is the entry point of the provenance chain: raw documents go in, hashed/chunked artifacts come out, and [22] Bridge connects them to views.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__INGEST_SEMANTICS.yaml` |
| Semantics cert | `certs/QA_INGEST_SEMANTICS_CERT.v1.json` |
| Witness pack | `certs/witness/QA_INGEST_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_INGEST_COUNTEREXAMPLES_PACK.v1.json` |
| Validator | `qa_ingest_validator.py` |
| Semantics schema | `schemas/QA_INGEST_SEMANTICS_CERT.v1.schema.json` |
| Witness schema | `schemas/QA_INGEST_WITNESS_PACK.v1.schema.json` |
| Counterexamples schema | `schemas/QA_INGEST_COUNTEREXAMPLES_PACK.v1.schema.json` |
| Committed source mirror | `../qa_ingestion_sources/` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate ingestion family
python qa_ingest_validator.py --demo

# Or via meta-validator (runs as test [23])
python qa_meta_validator.py
```

## Semantics

### Accepted file types

`odt`, `txt`, `md`

### Ingest contract

| Parameter | Value |
|-----------|-------|
| `max_docs` | 4 |
| `max_total_chars` | 500,000 |
| `max_total_chunks` | 5,000 |
| Chunk strategy | `fixed_chars` |
| Chunk max | 1,200 chars |
| Chunk min | 1 char |
| Chunk overlap | 120 chars |

### Normalization rules

1. LF line endings (no CR)
2. NFKC unicode normalization
3. Whitespace collapse (runs of whitespace -> single space)
4. Control character removal

### Hash domains

| Domain | Used for |
|--------|----------|
| `QA/RECORD/v1` | Document record hash |
| `QA/CHUNK/v1` | Individual chunk hash |
| `QA/DOC_ROOT/v1` | Per-document chunk root |

### Provenance requirements

Each ingested document must carry:
- `container_sig`: container/format signature
- `file_hash`: SHA256 of raw source bytes
- `source_path`: path to source file (must exist and be committed)

## Gitignored sources and committed mirrors

**Critical rule**: `source_ref` MUST NOT point into `ingestion candidates/` (gitignored).

Source files referenced by witness/counterexample packs must live in `qa_ingestion_sources/` (committed to git). When adding a new document:

1. Copy to `qa_ingestion_sources/<filename>`
2. Set `source_ref` to the committed path
3. Run `--rehash` on the validator to recompute manifest hashes

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `HASH_MISMATCH` | Recomputed hash differs | Recompute with `--rehash` |
| `SCHEMA_MISMATCH` | Entry doesn't match schema | Fix structure |
| `DOMAIN_SEP_VIOLATION` | Wrong hash domain | Use correct domain |
| `NON_CANONICAL_JSON` | Non-canonical JSON | Re-serialize |
| `UNVERIFIABLE_PROOF` | Merkle proof invalid | Regenerate |
| `FORK_DETECTED` | Conflicting roots | Resolve |
| `INGEST_DOC_MISSING` | Source file not at `source_ref` | Move to `qa_ingestion_sources/` |
| `INGEST_DOC_HASH_MISMATCH` | Source file bytes changed | Re-ingest from original |
| `CONTAINER_SIG_TAMPER` | Container signature mismatch | Re-extract from original |
| `EXTRACTION_DRIFT` | Extracted text doesn't match | Fix parser or re-extract |
| `NORMALIZATION_DRIFT` | Normalized text doesn't match | Fix normalization contract |
| `CHUNKING_DRIFT` | Chunks don't match | Fix chunking parameters |
| `BUDGET_EXCEEDED` | Too many docs/chunks/chars | Reduce ingestion scope |

## Example

**Passing** — a witness entry for an ingested `.odt` file:
```json
{
  "doc_id": "qa_agentic_rag.odt",
  "source_ref": "qa_ingestion_sources/qa_agentic_rag.odt",
  "file_hash": "abc123...",
  "container_sig": "PK...",
  "chunks": [
    {"chunk_id": 0, "chunk_hash": "def456...", "char_count": 1200}
  ]
}
```

**Failing** — `INGEST_DOC_MISSING` when source points to gitignored path:
```json
{
  "source_ref": "ingestion candidates/qa_agentic_rag.odt",
  "expected_fail_type": "INGEST_DOC_MISSING"
}
```

## Changelog

- **v1.3.1** (2026-02-08): Initial triplet shipped.
- **v1.4.1** (2026-02-09): Fixed `source_ref` paths from gitignored directory to committed `qa_ingestion_sources/`. Rehashed all manifests.
