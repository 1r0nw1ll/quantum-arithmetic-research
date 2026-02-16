---
name: qa-archive-hunter
description: "Use when chat/forensics mention scripts or paths that are missing from the current tree, to scan repo zip snapshots and locate where those missing files likely live (outputs a report under `_forensics/`)."
metadata:
  short-description: "Find missing work in archives"
---

# QA Archive Hunter

Use this skill to close the **chat ↔ repo gap** when targets referenced in chat no longer exist in the working tree.

## Quick start

1) Refresh forensics (updates chat-mentioned targets lists):
- `python tools/project_forensics.py`

2) Hunt missing targets in repo zip snapshots:
- `python tools/qa_archive_hunter.py`

3) Open the report it writes:
- `_forensics/archive_hunt_<timestamp>/REPORT.md`

## Common workflows

### Focus on “high-frequency missing” targets

- `python tools/qa_archive_hunter.py --min-count 25 --top 200`

### Scan additional zip locations (beyond repo root)

If you keep snapshots under a folder like `archives/`:
- `python tools/qa_archive_hunter.py --zip-root archives --zip-max-depth 3`

### After a hit: recover a candidate file safely

1) Inspect the zip entry path (from `matches.tsv`)
2) Recover into a trace folder (keeps provenance; avoids overwriting working tree):
- `mkdir -p trace/recovered/<run_id>`
- `python -c "import zipfile; z=zipfile.ZipFile('<zip>'); print(z.read('<entry>').decode('utf-8','replace'))" > trace/recovered/<run_id>/<basename>`
3) Record the recovery in `trace/TRACE_INDEX.md`

## Guardrails

- The hunter only uses filenames/paths and counts (no raw chat text).
- Default scanning is intentionally conservative (top-level `*.zip`); widen scope via `--zip-root`.
- Treat recoveries as **trace first**, then promote into the semantic core only after review + validation.
