<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 5 cert family doc; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), tools/qa_kg/canonicalize.py (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026) -->
# [228] QA-KG Determinism Cert v1

**Status:** PASS (D1, D2, D3, D4, D5, D6, D7; D1.5 WARN-only)
**Created:** 2026-04-16 (Phase 5 of QA-MEM)
**Source:** Will Dale + Claude; `docs/specs/QA_MEM_SCOPE.md` (Dale, 2026); `tools/qa_kg/canonicalize.py` (Dale, 2026); `tools/qa_kg/build_context.py` (Dale, 2026); `tools/qa_kg/fixtures/corpus_snapshot_v1/CANONICAL_MANIFEST.json` (Dale, 2026); `memory/project_qa_mem_review_role.md` (Dale, 2026).

## Claim

QA-MEM Phase 5 adds a **determinism contract** over the QA-KG build
pipeline. Two independent invariants:

1. **Fixture rebuild determinism** — given the frozen fixture at
   `tools/qa_kg/fixtures/corpus_snapshot_v1/`, running
   `python -m tools.qa_kg.cli build --fixture <dir>` produces a
   byte-stable canonical graph hash. The canonical serialization
   excludes build-time metadata (`created_ts`, `updated_ts`, `vetted_ts`,
   `last_check_*`), NFC-normalizes every string value, orders nodes by
   `id` and edges by `(src_id, dst_id, edge_type)`, and whitelists the
   `meta` table to `{schema_version}`. See
   `tools/qa_kg/canonicalize.py`.

2. **Ledger-vs-live consistency** — `_meta_ledger.json["228"].graph_hash`
   records the canonical hash of the live production `qa_kg.db` at
   meta-validator run time. `kg.promote()` enforces this HARD: any live
   hash diverging from the recorded hash raises `FirewallViolation` with
   a `graph_hash drift` message (see `tools/qa_kg/kg.py` Phase 5 block).
   The remediation is a meta-validator rerun.

## Gates

| Gate  | Level         | Description |
|-------|---------------|-------------|
| D1    | HARD          | `sha256(canonical(manifest.files))` matches `expected_hash.fixture_hash`. Bootstrap PASS on first run (expected_hash.json absent); D2 writes the initial value. |
| D1.5  | WARN          | `manifest.repo_head` vs current `git rev-parse HEAD`. Mismatch emits `[WARN]`; never blocks. Phase 5.1 will pin reproducibility to `manifest.repo_head` via `git archive`. |
| D2    | HARD          | In-process rebuild twice (`:memory:` DB, `BuildContext.from_fixture`) produces byte-identical `graph_hash`. Bootstrap path writes `expected_hash.json` on first run. |
| D3    | HARD          | Subprocess rebuild twice (`python -m tools.qa_kg.cli build --fixture <> --hash-only`) produces the same hash as D2. Fresh interpreter state must not perturb output. |
| D4    | SHOULD or N-A | Linux: PASS (Linux parity scaffolded). macOS / Windows: N-A (non-Linux reviewer required). Cross-platform parity is deferred; Phase 5 ships the scaffold. |
| D5    | HARD          | `_meta_ledger.json["228"].graph_hash` matches live `graph_hash(conn)` against production `qa_kg.db`. **Bootstrap PASS** on first run with explicit rationale string distinguishing it from compare-mode PASS (tri-state honesty discipline from [254] R4). |
| D6    | HARD          | Ephemeral test: creates agent + promoter nodes in a temp DB, writes a fake ledger with sentinel `graph_hash`, calls `kg.promote()`. Asserts `FirewallViolation` with `graph_hash drift` in the message. |
| D7    | HARD          | AST scan for `except Exception: pass` / bare `except: pass` in `tools/qa_kg/canonicalize.py`, `tools/qa_kg/build_context.py`, and this validator. Mirrors [254] R8 and [225] KG10. |

## Open decisions (resolved in plan v2; locked here for audit)

| ID | Decision | Locked value |
|----|----------|--------------|
| C#1 | ARAG_DB parameterization | Minimal `db_path=None` kwarg on `arag.search` / `arag.promote_to_kg`. `ARAG_DB` retained as default. |
| C#2 | D6 HARD vs WARN | HARD. Operational pattern: rerun `qa_meta_validator.py` after any DB-mutating extractor pass before promote-dependent calls. Documented in README.md "Operational contract" section. |
| C#3 | Pinned-source reproducibility | Phase 5.1 work. Phase 5 initial commit tests against working-tree code. `tools/qa_kg/build_context.py` carries `BuildContext` + `run_pipeline`; `materialize_pinned_repo` and PYTHONPATH overlay are deferred. |
| C#4 | `captured_at_utc` in `fixture_hash` | EXCLUDED. `compute_fixture_hash` hashes only `manifest.files[]`, not metadata. Re-capturing byte-identical content yields identical hash. |
| F1  | D5 bootstrap distinctness | PASS rationale strings distinguish "bootstrap" (no prior hash) from "match" (comparison). Future auditor reading the ledger can tell them apart. |

## Operational contract

**After any DB-mutating extractor pass** (`tools.qa_kg.cli build`, OB
ingest, A-RAG promote, manual edge insertion):

```
python qa_alphageometry_ptolemy/qa_meta_validator.py
```

must be rerun before subsequent `kg.promote()` calls will succeed. This
is by design — staleness is HARD. The live hash drifts whenever the
graph content changes; only a meta-validator sweep refreshes the ledger's
recorded hash. Phase 5.5 may add self-healing `kg.promote()` (inline
`[228]` D-gate rerun on mismatch) if the friction becomes operational.

## Out of scope

- **Phase 4.5** corpus expansion (Ben/Dale full books, Wildberger PDFs,
  Levin 2026 as SourceClaims).
- **Phase 5.1** pinned-source reproducibility via `git archive`. The
  helper lives in `tools/qa_kg/build_context.py` infrastructure (not
  shipped in Phase 5).
- **Phase 6** `[255]` agent-write surface + MCP tool integration.

Per `memory/project_qa_mem_review_role.md`'s alpha-bar rule, QA-MEM
still cannot be marketed as "agent memory" until `[228] + [254] + [255]`
all PASS. Phase 5 unlocks parallel-session safety
(determinism-on-disk); it does not, on its own, make QA-MEM
agent-facing.

## References

- `docs/specs/QA_MEM_SCOPE.md` (Dale, 2026)
- `memory/project_qa_mem_review_role.md` (Dale, 2026) — reviewer role and alpha-bar
- `tools/qa_kg/canonicalize.py` (Dale, 2026) — canonicalization module under test
- `tools/qa_kg/build_context.py` (Dale, 2026) — `BuildContext` + `run_pipeline`
- `tools/qa_kg/fixtures/corpus_snapshot_v1/CANONICAL_MANIFEST.json` (Dale, 2026) — frozen fixture
- `qa_alphageometry_ptolemy/qa_kg_determinism_cert_v1/expected_hash.json` — bootstrap/compare anchor
- Cert [227] v1 — firewall effectiveness (peer; shared `promote()` surface)
- Cert [254] v1 — authority-tiered ranker (peer; shared ledger shape)
- Cert [225] v4 — graph consistency (peer; shared schema contract)
