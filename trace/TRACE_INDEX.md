# Trace Index (ledger)

This file is a lightweight ledger of “what ran” and “what it produced.”

## Tooling

- Local run wrapper (run_id + manifest + stdout/stderr): `python tools/qa_trace_ledger.py run --tool-id TOOL.<name>.v1 -- <command...>`
- Auto-ledger output (ignored by git): `trace/TRACE_RUNS_LOCAL.md`

## Current trace-heavy subtrees (not yet relocated)

- `qa_lab/` — lab trace (logs/data/build outputs; large)
- `vault_audit_cache/` — vault ingestion cache (large)
- `chat_data/`, `chat_data_extracted/` — conversational trace / exports (large)
- `data/`, `phase2_data/` — datasets / derived products

## New trace entries (template)

Add one entry per run/trace bundle:

- `run_id`: `<yyyy-mm-dd>_<short_name>`
  - `when`: `<ISO-8601 UTC>`
  - `who/host`: `<machine or operator>`
  - `tool`: `<script/command>`
  - `inputs`: `<paths>`
  - `outputs`: `<paths>`
  - `status`: `PASS|FAIL|PARTIAL`
  - `notes`: `<1-3 lines>`
