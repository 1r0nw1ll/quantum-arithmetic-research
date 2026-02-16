---
name: qa-results-curator
description: "Use when converting triage hotspots/artifacts into curated results entries (claim → evidence → reproduce) and running the relevant validators so “buried results” become retrievable and reproducible."
metadata:
  short-description: "Curate reproducible results"
---

# QA Results Curator

The auto-generated registry (`Documents/RESULTS_REGISTRY.md`) is a **triage index**.  
This skill helps you produce **curated, reproducible** entries in `Documents/RESULTS_CURATED.md`.

## Quick start

1) Refresh the triage signals:
- `python tools/project_forensics.py --control-cutoff-utc-date 2026-01-10`
- `python tools/generate_results_registry.py --control-cutoff-utc-date 2026-01-10`
  - Anything with mtime **before 2026-01-10 UTC** is treated as *legacy* and must be re-vetted.

2) Build fast search (optional but recommended):
- `python tools/qa_local_search.py --overwrite build`

3) Pick candidates to curate:
- `python tools/qa_forensics_cert_index.py view-get view:forensics/hotspot_top_k --limit 50`
- `python tools/qa_local_search.py search "witness pack" --top 20`

4) Write curated entries (manual, not overwritten):
- `Documents/RESULTS_CURATED.md`

## Curation workflow (claim → evidence → reproduce)

For each candidate “result node”:

1) **Identify the claim**
  - What is being asserted (1–3 sentences)?

2) **Attach evidence**
  - Prefer validator-verifiable artifacts (cert JSON, witness packs, bundles, logs with PASS markers).

3) **Add reproduction commands**
  - Exact command(s) + working directory.
  - If evidence is a QA cert family: run the family validator or `qa_meta_validator.py`.

4) **Mark status**
  - `draft` until you can reproduce.
  - `verified` once reproduced and (where applicable) validated.
  - `superseded` if replaced by a newer, stronger, or corrected result.

## Guardrails

- Don’t manually edit `Documents/RESULTS_REGISTRY.md` (it’s overwritten on regen).
- Prefer writing new execution artifacts under `_forensics/` or `trace/` (ledger them).
- Validation is part of the result: capture validator output (hashes/logs) as trace.
- For pre-cutoff artifacts: mark entries `legacy` until you have a post-cutoff reproduction run + evidence.
