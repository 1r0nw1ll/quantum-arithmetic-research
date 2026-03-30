# QA Meta Map (Steering, governance, “why we did it this way”)

This layer answers: **why** the system looks like it does, and what rules constrain changes.

## Repo-local rules (highest priority)

- Agent/repo instructions: `AGENTS.md`

## Pipeline governance + drift tracking

- Pipeline overview: `QA_PIPELINE_README.md`
- Axiom / invariant drift log: `QA_PIPELINE_AXIOM_DRIFT.md`

## Session lineage (chronicles)

These files are not “junk.” They are the project’s remembered history.

Common patterns at repo root:
- `HANDOFF*.md`
- `SESSION_CLOSEOUT*.md`
- `SESSION_SUMMARY*.md`
- `PROGRESS_SUMMARY.md`, `PROJECT_STATUS_*.md`

## Chat exports (conversational trace)

- Raw export(s): `chat_data/`
- Extracted/staged bulk: `chat_data_extracted/`
- Aggregated steering audit (no raw text): `chat_data/steering_audit.md`

## Cartography pointers

- Layer model + navigation: `Documents/QA_CARTOGRAPHY_MAP.md`
- Ontology pointers: `Documents/QA_ONTOLOGY_MAP.md`
- Execution runbook: `Documents/QA_EXECUTION_MAP.md`
- Results registry: `Documents/RESULTS_REGISTRY.md`

