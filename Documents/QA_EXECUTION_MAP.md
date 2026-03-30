# QA Execution Map (How to run, validate, reproduce)

This is a **runbook**: primary entrypoints and “how to verify reality” commands.

See also:
- `Documents/QA_CARTOGRAPHY_MAP.md`
- `Documents/QA_TOOLCHAIN_MAP.md`

## QA verification (ptolemy)

Run from `qa_alphageometry_ptolemy/`:
- `python qa_verify.py --demo`
- `python qa_meta_validator.py`

## Repo forensics (cartography inputs)

From repo root:
- Forensics scan (writes `_forensics/forensics_*/`): `python tools/project_forensics.py`
- Cartography integrity gate (compare latest two forensics runs): `python tools/qa_cartography_gate.py`
- Archive recovery scan (chat-missing targets -> zip hits): `python tools/qa_archive_hunter.py`
- Results registry refresh (uses latest `_forensics/...`): `python tools/generate_results_registry.py`
 - Curated results (manual): `Documents/RESULTS_CURATED.md`

## Trace wrapper (recommended for new runs)

Wrap any command so it produces a standard `run_id` folder under `trace/runs/`:
- `python tools/qa_trace_ledger.py run --tool-id TOOL.project_forensics.v1 -- python tools/project_forensics.py`

Default ledger target (ignored by git): `trace/TRACE_RUNS_LOCAL.md`  
Commit-worthy ledger target: `trace/TRACE_INDEX.md` (pass `--ledger trace/TRACE_INDEX.md`)

## Fast local search (agent-friendly)

Build/query SQLite FTS index:
- Build: `python tools/qa_local_search.py --overwrite build`
- Search: `python tools/qa_local_search.py search "meta validator" --top 10`
- Show: `python tools/qa_local_search.py show qa_alphageometry_ptolemy/qa_meta_validator.py`

## QA-certified search snapshot (auditable)

Build a QA datastore + posting-list view *from the sqlite forensics index*:
- Build: `python tools/qa_forensics_cert_index.py build`
- View keys: `python tools/qa_forensics_cert_index.py view-keys`
- View get: `python tools/qa_forensics_cert_index.py view-get view:forensics/chat_top_k --limit 50`
- Store get: `python tools/qa_forensics_cert_index.py store-get Documents/RESULTS_REGISTRY.md`

Validate the generated packs (use the printed `_forensics/qa_cert_index_*/` directory):
- Store: `python -m qa_alphageometry_ptolemy.qa_datastore_validator --semantics <DIR>/store_semantics.json --witness <DIR>/store_witness_pack.json --counterexamples <DIR>/store_counterexamples_pack.json`
- View: `python -m qa_alphageometry_ptolemy.qa_datastore_view_validator --store-semantics <DIR>/store_semantics.json --view-semantics <DIR>/view_semantics.json --witness <DIR>/view_witness_pack.json --counterexamples <DIR>/view_counterexamples_pack.json`

## QA-GraphRAG prototype (knowledge graph)

Quickstart: `QA_GRAPHRAG_README.md`

Canonical pipeline command:
- `python qa_build_pipeline.py --chunk-limit 500 --full-e8`
