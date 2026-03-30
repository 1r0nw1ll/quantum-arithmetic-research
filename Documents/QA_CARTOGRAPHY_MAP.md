# QA Cartography Map (Layers, Indices, Navigation)

This repo is not “noisy.” It is **layered**.

The goal of cartography is to preserve **all trace** while making the system cognitively navigable by separating:
- **Definitions** (what is true/valid),
- **Execution** (how you run/verify),
- **Trace** (what happened),
- **Results** (what was proven / shipped / reproduced),
- **Meta/Steering** (why choices were made).

## Five Layers (working definitions)

1) **Ontology (Semantic Core)** — schemas, cert families, validators, invariants, canonical maps/specs.  
2) **Execution (Machinery)** — CLIs, build/test runners, recompute harnesses, ingestion pipelines.  
3) **Trace (History)** — run outputs, logs, datasets, snapshots, exported chats, caches.  
4) **Results (Extracted Knowledge)** — claims + evidence + reproduction commands (registry-style).  
5) **Meta/Steering (Intent/Evolution)** — governance, guardrails, handoffs, pipeline drift notes.

## Indices you can use immediately (agent-friendly)

### Fast search (SQLite FTS)
- Build/search: `tools/qa_local_search.py`
- DB location: `_forensics/qa_local_search.sqlite` (local, ignored by git)

### Cartography integrity gate (regression detection)
- Compare latest two forensics runs: `python tools/qa_cartography_gate.py`

### QA-certified snapshot (Merkle store + posting-list view)
- Builder/query: `tools/qa_forensics_cert_index.py`
- Output: `_forensics/qa_cert_index_*/` (local, ignored by git)
- Validatable via:
  - `python -m qa_alphageometry_ptolemy.qa_datastore_validator ...`
  - `python -m qa_alphageometry_ptolemy.qa_datastore_view_validator ...`

## Where each layer lives (current mapping)

### Layer 1 — Ontology
- `qa_alphageometry_ptolemy/` (certificate spine, schemas, validators, TLA+ specs)
- Root QA axioms/theorems:
  - `QA_AXIOMS_BLOCK.md`
  - `QA_CONTROL_THEOREMS.md`
  - `QA_PIPELINE_AXIOM_DRIFT.md`
- Canonical mapping registry:
  - `qa_alphageometry_ptolemy/QA_MAP_CANONICAL.md`

See also: `Documents/QA_ONTOLOGY_MAP.md`

### Layer 2 — Execution
- Forensics + search tooling:
  - `tools/project_forensics.py`
  - `tools/qa_local_search.py`
  - `tools/qa_forensics_cert_index.py`
- QA verification entrypoints:
  - `qa_alphageometry_ptolemy/qa_meta_validator.py`
  - `qa_alphageometry_ptolemy/qa_verify.py`
- QA-GraphRAG prototype:
  - `QA_GRAPHRAG_README.md`
  - `qa_build_pipeline.py`, `qa_graph_query.py`, `qa_knowledge_graph.py`

See also: `Documents/QA_EXECUTION_MAP.md`
See also: `Documents/QA_TOOLCHAIN_MAP.md`

### Layer 3 — Trace
These are meaningful historical artifacts and should be treated as **trace**, not clutter.
- Big trace-heavy subtrees (example from `_forensics/forensics_20260214_154253/REPORT.md`):
  - `qa_lab/` (logs/data/build history)
  - `chat_data_extracted/`, `chat_data/` (conversational trace / exports)
  - `vault_audit_cache/` (vault ingestion cache)
  - `phase2_data/`, `data/` (datasets / derived products)

Trace ledger template: `trace/TRACE_INDEX.md`
New runs (standard run_id manifests): `python tools/qa_trace_ledger.py run --tool-id TOOL.<name>.v1 -- <command...>`
Local auto-ledger (ignored by git): `trace/TRACE_RUNS_LOCAL.md`

### Layer 4 — Results
- Curated registry (manual): `Documents/RESULTS_CURATED.md`
- Triage index (auto-generated): `Documents/RESULTS_REGISTRY.md`
- Forensics-based summary: `Documents/PROJECT_FORENSICS_CONSOLIDATION.md`
- Additional public-facing summaries (examples):
  - `FINAL_PUBLICATION_REPORT.md`
  - `REAL_FINAL_RESULTS.md`

### Layer 5 — Meta/Steering
- Repo-local rules: `AGENTS.md`
- Pipeline governance / process drift:
  - `QA_PIPELINE_README.md`
  - `QA_PIPELINE_AXIOM_DRIFT.md`
- Session and handoff chronicles (examples):
  - `HANDOFF*.md`, `SESSION_CLOSEOUT*.md`, `SESSION_SUMMARY*.md`

See also: `Documents/QA_META_MAP.md`

## “Spine” view keys (QA-certified, small lists)

If you build a certified index, you get small, navigational posting lists:
- `view:cartography/ontology_spine`
- `view:cartography/execution_spine`
- `view:cartography/results_spine`
- `view:cartography/meta_spine`

Query example:
`python tools/qa_forensics_cert_index.py view-get view:cartography/ontology_spine --limit 200`
