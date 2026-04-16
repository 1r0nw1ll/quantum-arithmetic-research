<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 5 internal cert family; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026), tools/qa_kg/canonicalize.py (Dale, 2026) -->
# Cert Family [228] — QA_KG_DETERMINISM_CERT.v1

**Phase 5 of QA-MEM.** Validates that the QA-KG build pipeline produces
byte-stable canonical output given a frozen corpus fixture, and that the
live production graph_hash is recorded in `_meta_ledger.json` so
`kg.promote()` can enforce freshness against it.

Primary sources:

- `docs/specs/QA_MEM_SCOPE.md` (Dale, 2026)
- `memory/project_qa_mem_review_role.md` (Dale, 2026) — reviewer-role +
  alpha-bar discipline that gates Phase 5 landing
- `tools/qa_kg/canonicalize.py` (Dale, 2026) — canonicalization module
  under test
- `tools/qa_kg/fixtures/corpus_snapshot_v1/CANONICAL_MANIFEST.json` (Dale,
  2026) — frozen fixture for D2/D3 regression

## Gates

### D1 (HARD) — Fixture content hash

`sha256(canonical(manifest.files))` matches `expected_hash.fixture_hash`.
Hashes ONLY the files array (sorted by `path`) — `captured_at_utc` and
`repo_head` are manifest metadata and EXCLUDED from the hash per Phase 5
plan C#4. Re-capturing byte-identical content produces identical hash.

### D1.5 (WARN) — Manifest repo_head drift

`manifest.repo_head` compared against `git rev-parse HEAD`. Mismatch
emits `[WARN] D1.5`; never blocks. The fixture remains valid across HEAD
advances because the pinning mechanism (git-archive materialization in
`tools/qa_kg/build_context.py`) is infrastructure reserved for Phase 5.1;
Phase 5 initial commit tests reproducibility against working-tree code.

### D2 (HARD) — In-process rebuild idempotence

Running `build --fixture <corpus_snapshot_v1>` twice, in the same process,
against ephemeral `:memory:` DBs, produces byte-identical `graph_hash`
output. Validates the canonicalization layer and extractor determinism at
a single point in time.

### D3 (HARD) — Subprocess rebuild idempotence

Running `python -m tools.qa_kg.cli build --fixture <corpus_snapshot_v1>
--hash-only` in two separate subprocesses with isolated DBs produces
byte-identical `graph_hash` and that hash matches D2's result. Validates
that fresh Python interpreter state does not perturb the output. Phase 5
initial: subprocess runs working-tree code (no pinning). Phase 5.1: adds
`git archive` materialization so reproducibility is tested against code
pinned at `manifest.repo_head`.

### D4 (SHOULD) — Cross-platform scaffold

Cross-platform NFC normalization + line-ending handling — Linux
scaffolded; macOS + Windows require a non-Linux reviewer. Phase 5 returns
**N-A** on non-Linux hosts and **PASS** on Linux (the scaffold runs;
scoped contract is "Linux stable, other platforms deferred").

### D5 (HARD) — Ledger graph_hash matches live

Reads `_meta_ledger.json["228"].graph_hash`. Computes live canonical
`graph_hash(conn)` against the production `qa_kg.db`. Compares.

**Bootstrap behavior:** on the first `qa_meta_validator.py` run after
Phase 5 lands (or any operation that removes `graph_hash` from the
ledger entry for [228]), D5 reports **PASS** with the explicit rationale:

```
PASS — bootstrap: ledger has no prior graph_hash to compare; validator
       wrote graph_hash=<sha12>… for subsequent runs to validate against.
```

From the second run onward, D5 PASS attests that the ledger and live
canonical hash match byte-for-byte. A fresh-ledger PASS is not equivalent
to a comparison PASS — the rationale string distinguishes them. This
follows the tri-state honesty discipline established by cert [254] R4
(N-A when the `valid_until` axis is unpopulated): vacuous-PASS replays
Phase 0 KG3's mistake and is explicitly rejected here.

### D6 (HARD) — promote() enforces graph_hash staleness

Ephemeral DB fixture: creates an agent Thought + promoter internal node +
writes a ledger entry with a known graph_hash. Mutates a node to induce
graph drift. Invokes `kg.promote()` and asserts a `FirewallViolation` is
raised whose message references `graph_hash drift`. Validates the
additive Phase 5 check in `kg.py` (line `if p5_expected: ...`).

### D7 (HARD) — No-swallow AST scan

AST walk over `tools/qa_kg/canonicalize.py`, the Phase-5 additions in
`tools/qa_kg/build_context.py`, and this validator itself. Flags any
`ExceptHandler` whose body is `[Pass]`, or any bare `except:` with no
exception type declared. Mirrors [254] R8 and [225] KG10 discipline.

## Operational contract

After any DB-mutating extractor pass (`tools.qa_kg.cli build`, OB ingest,
A-RAG promote, manual edge insertion), `qa_meta_validator.py` **must be
rerun** before subsequent `kg.promote()` calls will succeed. This is by
design — staleness is enforced HARD (see D6). The ledger's recorded
`graph_hash` is the authority; a live hash diverging from the ledger
means agent promotions are rejected until the ledger is refreshed.

**Remediation when blocked**:

```
python qa_alphageometry_ptolemy/qa_meta_validator.py
```

Re-runs the sweep, rewrites ledger with current `graph_hash`.

**Phase 5.5 consideration (not shipped)**: self-healing `kg.promote()` —
on `graph_hash` mismatch, attempt inline [228] D-gate rerun; if all PASS,
auto-refresh ledger and proceed. Deferred until operational friction
warrants it.

## Out of scope

- **Phase 4.5** corpus expansion (Ben/Dale full books, Wildberger PDFs,
  Levin 2026 as SourceClaims).
- **Phase 5.1** pinned-source reproducibility via `git archive` — the
  helper lives in `tools/qa_kg/build_context.py` infrastructure but is
  not exercised by Phase 5 gates.
- **Phase 6** `[255]` agent-write surface + MCP tool integration.

Per `memory/project_qa_mem_review_role.md`'s alpha-bar rule, QA-MEM still
cannot be marketed as "agent memory" until `[228] + [254] + [255]` all
PASS. Phase 5 unlocks parallel-session safety (determinism-on-disk); it
does not, on its own, make QA-MEM agent-facing.

## References

- `docs/specs/QA_MEM_SCOPE.md` (Dale, 2026)
- `memory/project_qa_mem_review_role.md` (Dale, 2026)
- `tools/qa_kg/canonicalize.py` (Dale, 2026)
- `tools/qa_kg/build_context.py` (Dale, 2026)
- `tools/qa_kg/fixtures/corpus_snapshot_v1/CANONICAL_MANIFEST.json` (Dale, 2026)
- Cert [227] v1 — firewall effectiveness (peer cert; shared `promote()` surface)
- Cert [254] v1 — authority-tiered ranker (peer cert; shared ledger shape)
- Cert [225] v4 — graph consistency (peer cert; shared schema contract)
