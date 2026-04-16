# MEMORY.md — Phase 5 Determinism Fixture Sample

<!-- PRIMARY-SOURCE-EXEMPT: reason=Phase 5 determinism fixture; frozen subset -->

This file is a frozen subset of MEMORY.md maintained as an input to the
QA-KG fixture-driven deterministic rebuild (cert [228]). The live
MEMORY.md lives at:

    ~/.claude/projects/-home-player2-signal-experiments/memory/MEMORY.md

and is NOT tracked in the repo. This snapshot contains only the pieces
the `memory_rules` extractor depends on (Hard Rules with `(HARD)` tag).
Content is selected to have zero secret risk per
`memory/feedback_no_secrets_in_commits.md`.

## Hard Rules (sampled)

### No Secrets In Fixture (HARD — 2026-04-16)

Fixture files must contain no credentials, PII, or personal documents.
Reviewed on capture.

### Phase 5 Fixture Is Frozen (HARD — 2026-04-16)

Editing this file invalidates the Phase 5 determinism cert until the
fixture is refreshed and the expected_hash.json is regenerated. Use
`python -m tools.qa_kg.cli fixture-refresh corpus_snapshot_v1` to
regenerate the manifest atomically.
