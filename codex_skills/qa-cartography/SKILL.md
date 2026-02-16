---
name: qa-cartography
description: "Use when working in `signal_experiments/` to build/query repo-wide forensic indices (forensics report, SQLite FTS, QA-certified datastore+views) and to keep `Documents/RESULTS_REGISTRY.md` evidence-linked and searchable."
metadata:
  short-description: "QA cartography & search indices"
---

# QA Cartography

Cartography is **mapping**, not cleanup: preserve all trace and make it navigable.

## Quick start (safe + fast)

From repo root:

1) Refresh forensics (writes trace-like outputs under `_forensics/`):
- `python tools/project_forensics.py --control-cutoff-utc-date 2026-01-10`
  - Anything with mtime **before 2026-01-10 UTC** is treated as *legacy* and must be re-vetted.

2) Optional: cartography integrity gate (compare latest two forensics runs):
- `python tools/qa_cartography_gate.py`

2) Build/refresh fast local search (SQLite FTS5):
- `python tools/qa_local_search.py --overwrite build`
- Query (human): `python tools/qa_local_search.py search "meta validator" --top 10`
- Query (agent): `python tools/qa_local_search.py --json search "meta validator" --top 10`

3) Build QA-certified navigation index (Merkle store + posting-list views):
- `python tools/qa_forensics_cert_index.py build`
- Views: `python tools/qa_forensics_cert_index.py view-keys`
- Spine example: `python tools/qa_forensics_cert_index.py view-get view:cartography/ontology_spine --limit 200`
- Record example: `python tools/qa_forensics_cert_index.py store-get qa_alphageometry_ptolemy/qa_meta_validator.py`

4) Refresh Results Registry (Layer 4 distillation):
- `python tools/generate_results_registry.py --control-cutoff-utc-date 2026-01-10`

## Retrieval workflow (when asked “where is X?”)

1) Start with certified spines:
- `view:cartography/ontology_spine`
- `view:cartography/execution_spine`
- `view:cartography/results_spine`
- `view:cartography/meta_spine`

2) Use SQLite FTS for precise term search:
- `python tools/qa_local_search.py search "<query>" --top 20`

3) When a candidate result is found:
- Add/update an entry in `Documents/RESULTS_REGISTRY.md` with:
  - claim (1–3 sentences)
  - evidence path(s)
  - reproduction command(s)
  - status (draft/verified/superseded)

## Guardrails

- Prefer `_forensics/` for heavy audit outputs (already ignored by git).
- Avoid emitting raw ChatGPT export text; prefer aggregates + file/path references.
- If you need to touch “trace-heavy” trees (`qa_lab/`, `vault_audit_cache/`, `chat_data_extracted/`), do it intentionally and record what you did in `trace/TRACE_INDEX.md`.
- Treat **pre-2026-01-10** artifacts as legacy until re-validated under trace/ledgered runs.
