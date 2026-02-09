# [21] QA A-RAG Interface

## What this is

Certifies the **Agentic RAG (Retrieval-Augmented Generation) interface** — the tool layer that agents use to retrieve information from the certified datastore. It defines a bounded tool set, maps each tool to a view type, and enforces budget/depth constraints on retrieval traces. Depends on [18] QA Datastore and [20] QA Datastore View.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__ARAG_INTERFACE.yaml` |
| Semantics cert | `certs/QA_ARAG_INTERFACE_CERT.v1.json` |
| Witness pack | `certs/witness/QA_ARAG_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_ARAG_COUNTEREXAMPLES_PACK.v1.json` |
| Validator | `qa_arag_validator.py` |
| Semantics schema | `schemas/QA_ARAG_INTERFACE_CERT.v1.schema.json` |
| Witness schema | `schemas/QA_ARAG_WITNESS_PACK.v1.schema.json` |
| Counterexamples schema | `schemas/QA_ARAG_COUNTEREXAMPLES_PACK.v1.schema.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate A-RAG interface family
python qa_arag_validator.py --demo

# Or via meta-validator (runs as test [21])
python qa_meta_validator.py
```

## Semantics

### Tool set (generators)

| Tool | QA Generator Meaning | Binds to View |
|------|---------------------|---------------|
| `keyword_search` | Lexical anchoring move | `KEYWORD_VIEW` |
| `semantic_search` | Semantic neighborhood move | `SEMANTIC_VIEW` |
| `chunk_read` | Materialization move (store-root read) | `CHUNK_STORE` |

### Trace contract

- **Root binding required**: every retrieval step must cite a store/view root
- **Step index monotone**: trace steps must be strictly ordered
- **View root provenance by kind**: each tool binds to its declared view type
- **Proof per retrieval step**: every step carries a Merkle proof

### Budget model

- `max_retrieved_tokens`: 500
- `max_steps`: 5

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `HASH_MISMATCH` | Hash recomputation fails | Recompute with `--rehash` |
| `SCHEMA_MISMATCH` | Entry doesn't match schema | Fix structure |
| `DOMAIN_SEP_VIOLATION` | Wrong hash domain | Use correct domain |
| `NON_CANONICAL_JSON` | Non-canonical serialization | Re-serialize |
| `UNVERIFIABLE_PROOF` | Merkle proof invalid | Regenerate |
| `FORK_DETECTED` | Conflicting roots | Resolve root conflict |
| `BUDGET_EXCEEDED` | Token or step budget exceeded | Reduce retrieval scope |
| `WRONG_GENERATOR_SELECTION` | Tool doesn't match query type | Use appropriate tool |
| `INSUFFICIENT_DEPTH` | Not enough retrieval steps | Add more steps |
| `OVERREAD_NOISE` | Retrieved irrelevant content | Tighten retrieval query |
| `ENTITY_CONFUSION` | Mixed up distinct entities | Improve entity resolution |
| `REDUNDANT_PATH` | Duplicate retrieval work | Deduplicate steps |

## Changelog

- **v1.3.1** (2026-02-08): Initial triplet shipped with meta-validator hook [21].
