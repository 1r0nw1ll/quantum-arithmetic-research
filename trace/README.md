# Trace Layer

The trace layer is where execution history lives:
- runs, logs, artifacts, snapshots, exports, caches

Nothing here is “noise.” It’s **trace**.

This repo already contains large trace-heavy directories (e.g. `qa_lab/`, `vault_audit_cache/`, `chat_data_extracted/`).  
`trace/` is a **navigation/ledger root** so future runs can be placed consistently (without mixing with semantic core files).

Start with the ledger: `trace/TRACE_INDEX.md`.

For new runs, prefer the trace wrapper:
- `python tools/qa_trace_ledger.py run --tool-id TOOL.<name>.v1 -- <command...>`

By default it appends to a **local** ledger file:
- `trace/TRACE_RUNS_LOCAL.md` (ignored by git)
