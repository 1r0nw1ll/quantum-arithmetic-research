# QA Toolchain Map (Execution Layer)

This map is the **execution-layer index**: which tools exist, what job they do, and how they connect Ontology ↔ Trace ↔ Results.

Cartography principle: tools may read semantic core, but should write **trace** (or trace-like) outputs so execution history stays queryable.

## Tool Buckets (by job)

1) **Build / Validate** — run validators, certify invariants, run test gates.  
2) **Generate / Produce** — produce derived artifacts (certs, graphs, datasets, reports).  
3) **Analyze / Audit** — forensics, drift, redundancy, chat↔repo reconciliation.  
4) **Operate / Orchestrate** — safe automation, agent governance/security kernels.

## Tool Cards (starter set)

Below are “Tool Cards” for high-frequency tools. Add more as needed.

---

### TOOL.qa_meta_validator.v1
- **Entrypoint:** `python qa_alphageometry_ptolemy/qa_meta_validator.py`
- **Bucket:** Build / Validate
- **Inputs:** built-in cert fixtures under `qa_alphageometry_ptolemy/certs/` + example packs
- **Outputs:** stdout/stderr only (pass/fail); no required file writes
- **Determinism contract:** exact arithmetic + canonical JSON + stable fail_type taxonomy (per cert families)
- **Trace contract:** capture run output under `trace/` (or `_forensics/`) when used in audits
- **Result linkage:** validates “evidence artifacts” that appear in `Documents/RESULTS_REGISTRY.md`

### TOOL.qa_verify_demo.v1
- **Entrypoint:** `python qa_alphageometry_ptolemy/qa_verify.py --demo`
- **Bucket:** Build / Validate
- **Inputs:** decision spine examples under `qa_alphageometry_ptolemy/`
- **Outputs:** stdout/stderr only
- **Determinism contract:** canonical JSON + recompute hooks (per spine)
- **Trace contract:** record output hashes in trace when used as an acceptance gate
- **Result linkage:** supports “verified demo” results entries

---

### TOOL.project_forensics.v1
- **Entrypoint:** `python tools/project_forensics.py`
- **Bucket:** Analyze / Audit
- **Inputs:** repo filesystem + optional `chat_data/*.zip`
- **Outputs:** `_forensics/forensics_*/REPORT.md` + TSVs (no raw chat text by default)
- **Determinism contract:** stable keyword/path/command extraction; stable ordering in TSV outputs
- **Trace contract:** writes only under `_forensics/` (trace-like)
- **Result linkage:** drives `Documents/RESULTS_REGISTRY.md` seeding and hotspot prioritization

### TOOL.qa_local_search.v1
- **Entrypoint:** `python tools/qa_local_search.py --overwrite build`
- **Bucket:** Analyze / Audit
- **Inputs:** repo text files + latest `_forensics/forensics_*` metadata
- **Outputs:** `_forensics/qa_local_search.sqlite`
- **Determinism contract:** stable scan scope rules + stable metadata joins; FTS rank is best-effort (not a proof)
- **Trace contract:** writes only under `_forensics/`
- **Result linkage:** powers fast retrieval for “where is X?” during result extraction

### TOOL.qa_forensics_cert_index.v1
- **Entrypoint:** `python tools/qa_forensics_cert_index.py build`
- **Bucket:** Analyze / Audit
- **Inputs:** `_forensics/qa_local_search.sqlite`
- **Outputs:** `_forensics/qa_cert_index_*/` (Merkle store + posting-list view + witness/counterexamples packs)
- **Determinism contract:** canonical JSON + domain-separated hashing; witness packs validate via QA validators
- **Trace contract:** writes only under `_forensics/`
- **Result linkage:** produces auditable “spine views” to navigate Ontology/Execution/Results/Meta

### TOOL.qa_archive_hunter.v1
- **Entrypoint:** `python tools/qa_archive_hunter.py`
- **Bucket:** Analyze / Audit
- **Inputs:** `_forensics/forensics_*/chat_python_targets.tsv` + repo `*.zip` snapshots (configurable)
- **Outputs:** `_forensics/archive_hunt_*/REPORT.md` + `matches.tsv` + `missing.tsv`
- **Determinism contract:** stable TSV parsing + stable zip name listing + stable match rules (path/basename)
- **Trace contract:** writes only under `_forensics/` (trace-like)
- **Result linkage:** helps recover “missing work” referenced by chat so it can be curated/validated

### TOOL.generate_results_registry.v1
- **Entrypoint:** `python tools/generate_results_registry.py`
- **Bucket:** Analyze / Audit
- **Inputs:** latest `_forensics/forensics_*` + existing `Documents/RESULTS_REGISTRY.md`
- **Outputs:** updates `Documents/RESULTS_REGISTRY.md`
- **Determinism contract:** stable ranking inputs; emits best-effort reproduction pointers (not proof)
- **Trace contract:** registry is semantic distillation (Layer 4), not raw trace
- **Result linkage:** this *is* the results layer index

### TOOL.qa_cartography_gate.v1
- **Entrypoint:** `python tools/qa_cartography_gate.py`
- **Bucket:** Analyze / Audit
- **Inputs:** two `_forensics/forensics_*` runs (default: latest + previous)
- **Outputs:** `_forensics/cartography_gate_*/REPORT.md` + `summary.json`
- **Determinism contract:** stable JSON/TSV parsing + stable sorted diff output
- **Trace contract:** writes only under `_forensics/` (trace-like)
- **Result linkage:** guards cartography itself (alerts when indexing/drift signals are suspicious)

---

### TOOL.qa_build_pipeline.v1
- **Entrypoint:** `python qa_build_pipeline.py --chunk-limit 500 --full-e8`
- **Bucket:** Generate / Produce
- **Inputs:** lexicon + optional `vault_audit_cache/chunks`
- **Outputs:** graph artifacts (e.g. `qa_knowledge_graph.graphml`, entity/edge JSONs) in repo paths
- **Determinism contract:** deterministic encoding + stable graph export formats (GraphML/JSON)
- **Trace contract:** prefer recording run manifests + outputs under trace for new runs (future)
- **Result linkage:** populates entities/graphs used by downstream analysis and reports

### TOOL.qa_graph_query.v1
- **Entrypoint:** `python qa_graph_query.py "<question>" --top-k 5 --method hybrid`
- **Bucket:** Analyze / Audit
- **Inputs:** `qa_knowledge_graph.graphml` + encodings
- **Outputs:** stdout only (ranked answers)
- **Determinism contract:** deterministic encodings; query is best-effort (not a validator proof)
- **Trace contract:** record query + output hash for reproducible “investigation trails”
- **Result linkage:** supports discovery and cross-linking; not a proof artifact by itself

---

### TOOL.qa_agent_security_kernel.v1
- **Entrypoint:** `python qa_agent_security/qa_agent_security.py --validate`
- **Bucket:** Operate / Orchestrate
- **Inputs:** policy kernel + schemas under `qa_agent_security/schemas/`
- **Outputs:** JSON pass/fail summary (`--validate`) or stdout self-test log
- **Determinism contract:** canonical JSON + fixed fail_type taxonomy + merkle trace semantics
- **Trace contract:** store policy pass/fail + obstruction certs under trace when used operationally
- **Result linkage:** prevents “untrusted → exec” moves; supports auditable agent operation

### TOOL.qa_agent_security_runner.v1
- **Entrypoint:** `python -m qa_agent_security.tool_runner`
- **Bucket:** Operate / Orchestrate
- **Inputs:** capability tokens + tool schemas (see `qa_agent_security/README.md`)
- **Outputs:** stdout self-test log (no network required)
- **Determinism contract:** strict URL + schema invariants; deterministic obstruction generation
- **Trace contract:** log every move as `{move, fail_type, invariant_diff}` when used for real runs
- **Result linkage:** produces security trace/evidence for governance

---

### TOOL.qa_trace_ledger.v1
- **Entrypoint:** `python tools/qa_trace_ledger.py`
- **Bucket:** Operate / Orchestrate
- **Inputs:** `--tool-id` + optional `--run-id` + optional `--ledger`; for `run`, a command after `--`
- **Outputs:** `trace/runs/<run_id>/` (manifest + stdout/stderr + artifacts/checks dirs) + optional ledger markdown append
- **Determinism contract:** stable run_id generation; canonical JSON (`RUN_MANIFEST.json`); sha256 hashes of stdout/stderr
- **Trace contract:** writes only under `trace/` by default (and the chosen ledger file)
- **Result linkage:** makes reproduction trails auditable; attach `RUN_MANIFEST.json` as “evidence of execution” when curating results
