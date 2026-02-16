---
name: qa-trace-ledger
description: "Use when you want a standard run_id folder (manifest + stdout/stderr logs) for a command under trace/runs/, plus an optional ledger entry (default: trace/TRACE_RUNS_LOCAL.md)."
metadata:
  short-description: "Standardize run trace + manifests"
---

# QA Trace Ledger

This skill standardizes **execution trace** into run folders and a small ledger entry.

## Quick start

Wrap a command and capture logs + a `RUN_MANIFEST.json`:
- `python tools/qa_trace_ledger.py run --tool-id TOOL.project_forensics.v1 -- python tools/project_forensics.py`

It writes:
- `trace/runs/<run_id>/RUN_MANIFEST.json`
- `trace/runs/<run_id>/stdout.log`
- `trace/runs/<run_id>/stderr.log`
- `trace/runs/<run_id>/artifacts/`
- `trace/runs/<run_id>/checks/`

And appends a ledger entry to:
- `trace/TRACE_RUNS_LOCAL.md` (default; ignored by git)

## Common workflows

### Record a “commit-worthy” entry

Append to the tracked index instead of the local ledger:
- `python tools/qa_trace_ledger.py run --tool-id TOOL.qa_forensics_cert_index.v1 --ledger trace/TRACE_INDEX.md -- python tools/qa_forensics_cert_index.py build`

### Create a run folder without executing a command

- `python tools/qa_trace_ledger.py init --tool-id TOOL.manual_investigation.v1 --note "review hotspot XYZ"`

### Disable ledger writes

- `python tools/qa_trace_ledger.py run --tool-id TOOL.scratch.v1 --ledger none -- <command...>`

## Guardrails

- Prefer writing new outputs under `trace/` or `_forensics/`, not into semantic core directories.
- Treat `trace/runs/<run_id>/` as **evidence storage**, then promote only the distilled result into `Documents/RESULTS_CURATED.md`.
