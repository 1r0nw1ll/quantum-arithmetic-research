<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG internal scope document; grounds in cert families [202], [225] v4, [226], [227], [228], [252], [253], [254], [255], and project-internal authority docs -->
# QA-MEM Scope (through Phase 6)

**Status:** Phases 0–6 landed. **Alpha bar achieved 2026-04-16** — QA-MEM is now alpha agent memory AND authoritative project memory per the original three-cert bar ([228] + [254] + [255]). Phase 4.5 (corpus scale) and Phase 5.1 (pinned-source D2/D3) reclassified as operational hardening; not gating.
**Authority:** cert families `[225] v4` (graph consistency) + `[226]` (Candidate F classifier invariants) + `[202]` (Aiq Bekar digital root) + `[227]` (firewall effectiveness) + `[252]` (epistemic fields) + `[253]` (source-claim contracts) + `[254]` (authority-tiered ranker) + `[228]` (graph determinism) + `[255]` (agent write surface).
**Companion files:** `QA_AXIOMS_BLOCK.md`, `qa_orbit_rules.py`, `tools/qa_retrieval/schema.py`, `memory/project_qa_mem_architecture.md`.

## What QA-MEM IS (as of Phase 0)

An indexed, firewalled view over repo artifacts:

- **Candidate F retrieval index** per cert `[202]`:
  `idx_b = dr(char_ord_sum(content))`, `idx_e = NODE_TYPE_RANK[node_type]`.
  These columns partition the graph into 81 retrieval sectors. They are a
  **retrieval index**, not a QA state.
- **Canonical orbit tier** from `qa_orbit_rules.orbit_family(idx_b, idx_e)` —
  singularity/satellite/cosmos/unassigned. Under Candidate F most nodes land
  in Cosmos (72/81 of the lattice is Cosmos); this is a partition, not a
  canonicity judgment.
- **Keyword-co-occurrence edges** (`edge_type="keyword-co-occurs"`,
  `method="keyword"`, `confidence=0.3`) from body-token matches: `[N]` cert
  references, axiom codes, rule-title substrings. These are NOT derivations,
  NOT proofs, NOT authoritative provenance.
- **Subject coord columns** `subject_b/subject_e` — optional, populated only
  from cert metadata that explicitly declares a QA state subject. THIS is
  the field agents should use to identify which QA state a node is about.
- **Firewall guard**: `kg.upsert_edge` raises `FirewallViolation` when an
  Unassigned source attempts a causal edge to a Cosmos/Singularity target
  without `via_cert`. Under Candidate F, Unassigned is effectively unoccupied,
  so this guard rarely fires. It is shape-correct but does not yet implement
  the full Theorem NT firewall.

## What QA-MEM IS NOT (explicit non-goals)

- **Not a canonicity judgment.** Tier is a lattice partition derived from a
  content hash × node-type rank.
- **Not an authority ranker.** `kg.search()` returns FTS5/BM25-ordered rows.
  Do NOT treat results as "what the project believes."
- **Not a provenance graph.** `kg.why()` filters keyword edges OUT; no Phase 0
  extractor emits structural `derived-from`. `why()` returns empty for every
  cert until Phase 3.
- **Not a contradiction system.** `contradicts` is supported but unpopulated.
- **Not an epistemic-status system.** No `SourceClaim/Interpretation/
  Observation/Conjecture` distinctions yet; Phase 1 adds them.
- **Not a memory substrate for agent reasoning.** A-RAG remains corpus
  backbone. Agents must not take QA-MEM as authoritative until Phase 2
  (AgentNote firewall) lands.
- **Not deterministic across sessions.** OB markdown snapshots depend on
  capture time; Phase 5 adds frozen fixtures + graph-hash cert `[228]`.

## Do not do on `idx_b/idx_e`

The column names `idx_b/idx_e` deliberately avoid the QA state names `(b,e)`.
Computing `d = idx_b + idx_e` produces nonsense because `idx_b` is a UTF-8
byte-sum digital root and `idx_e` is a node-type ordinal. QA states for
knowledge nodes live in `subject_b/subject_e`, populated exclusively from
cert metadata.

The `Index` type (formerly `Coord`) intentionally has no `.d` or `.a`
attributes. Back-compat aliases `Coord / compute_be / tier_for_coord`
retained during Phase 0 only.

**Removal target: cert `[225]` v3 / schema v3.** The alias pin is enforced
by `test_back_compat_aliases_scheduled_for_removal_in_v3` in
`tools/qa_kg/tests/test_kg_basic.py`: under `SCHEMA_VERSION == 2` the
aliases MUST exist; under `SCHEMA_VERSION >= 3` the aliases MUST be gone
(test fails and forces cleanup in the same commit as the schema bump).

## Landed phases

### Phase 1 (schema v3) — epistemic fields

Landed 2026-04-15. Schema v3 adds `authority`, `epistemic_status`, `method`,
`source_locator`, `lifecycle_state` to the nodes table. The authority axis
(primary / derived / internal / agent) records WHO produced the knowledge.
The epistemic axis (axiom / source_claim / certified / observation /
interpretation / conjecture) records WHAT KIND of claim. An allowed matrix
(4×6) is enforced by cert [252] EF3 and lives in
`qa_kg_epistemic_fields_cert_v1/allowed_matrix.json`.

Back-compat aliases (Coord / compute_be / tier_for_coord) removed per the
Phase 0 pin, atomic with the v3 bump.

N1/N2/N3 carry-forward fixes from Phase 0 re-audit landed as [225] v3 gates
KG8 (frozen-not-in-sweeps), KG9 (axiom-codes canonical — parsed from
CLAUDE.md, not hardcoded), KG10 (no except-Exception-continue swallows).

Phase 1 also adds a partial authority firewall: `authority=agent` nodes
cannot emit causal edges without `via_cert`, enforced at `edge_allowed()`
and in `kg.upsert_edge()`.

### Phase 3 — contradictions + SourceClaim ingestion

Landed 2026-04-16. First-class representation of quoted material from
primary sources, plus machine-checkable contradictions and supersedes
lifecycle.

Schema: `EPISTEMIC_STATUSES` gains `"source_work"` as an additive enum
value (SCHEMA_VERSION stays 3). Enforcement is application-level on
old DBs + DB-level on fresh DBs; `[252]` EF3 allowed-matrix is the
runtime safety net during the transition. `qa_kg.db` is gitignored and
rebuildable (`python -m tools.qa_kg.cli build`).

Edge types: `quoted-from` (SourceClaim → SourceWork, FK) and
`supersedes` (newer → older, lifecycle) added to
`orbit.STRUCTURAL_EDGE_TYPES`. Both are non-causal — the Theorem NT
firewall does not apply, which is correct: a Work is an external fact
and supersession is lifecycle metadata, not derivation. Regression
guard asserts `CAUSAL_EDGE_TYPES.isdisjoint(STRUCTURAL_EDGE_TYPES)`.

Node factories (`Node.source_work`, `Node.source_claim`) pin the
`(node_type, authority, epistemic_status)` triple so extractors cannot
mis-configure. SourceClaim IDs are `sc:<sha1(locator+quote)[:16]>` =
64 bits, collision-safe at Phase 4.5 corpus scale.

Contradicts: edge provenance is JSON with a `reason` in the closed set
`{ocr, variant, typo, dispute, true}`. The forbidden endpoint set is
`{node_type=Axiom, authority=agent}`: axioms don't contradict (rule-system
bug) and agent dissent must flow through `kg.promote()` first. Both
closed sets live in `qa_kg_source_claims_cert_v1/closed_sets.json`.

Lifecycle bridge: `extractors/certs.py::_lifecycle_for_status` now
translates the file-level `_status: frozen` marker to node-level
`lifecycle_state ∈ {superseded, deprecated}` depending on whether a
sibling `_v<N+1>` directory exists. Keeps KG8 (file) and KG13 (graph)
aligned.

Shared locator resolver: `tools/qa_kg/locators.py` provides
`resolve_file_locator` / `resolve_pdf_locator` / `resolve_cert_locator`
/ `resolve_any`. `[252]` EF4 and `[253]` SC1 both import from it — no
divergence when Phase 4.5 adds URL scheme. `[252]`'s
`mapping_protocol_ref.json` is unchanged (gate semantics unchanged);
audit-trail SHA on the validator file reflects implementation
re-organisation only.

WARN capture: `qa_meta_validator.py` now scrapes `[WARN]` stdout
lines from the four QA-KG certs and appends them as a `warns` list on
each `_meta_ledger.json` entry. Additive — existing consumers
(`kg.promote` freshness check) ignore the new field. Stdout parsing
is a Phase 3 expedient; Phase 5+ switches to structured validator
returns.

New cert families:

- **[253] QA_KG_SOURCE_CLAIMS_CERT.v1** — SC1 quote+locator, SC2
  quoted-from FK to SourceWork, SC3 extraction_method closed set,
  SC4 locator-conflict rule, SC5 reason closed set, SC6 no
  contradicts cycles, SC7 WARN unresolved disputes, SC8 endpoint
  whitelist (no Axiom, no agent).

Cert bumps:

- **[225] v4** — adds KG11 (SourceWork primary+source_work), KG12
  (SourceClaim quoted-from FK), KG13 (supersedes DAG + lifecycle
  consistency; superseded requires ≥1 incoming, deprecated exempt).
  [225] v3 is frozen; v3 directory and doc retained for audit.

Seed corpus:

- `docs/theory/svp_wiki_qa_elements_snapshot.md` — verbatim SVP wiki
  page, fetched 2026-04-16.
- `tools/qa_kg/fixtures/source_claims_phase3.json` — 2 SourceWorks,
  15 SourceClaims, 11 Observations, 13 contradicts edges covering all
  5 `reason` values on real-data exemplars, 3 supersedes edges
  forming the `[225] v4 → v3 → v2 → v1` chain. Volk PDF not located
  under `Documents/wildberger_corpus/`; per plan §4 Volk-absent
  fallback, reason=variant exemplar uses SVP-internal
  L=FC/2 vs L=C·F/12 self-contradiction.

Allowed matrix: `{primary}` now includes `source_work` alongside
`axiom` and `source_claim`.

Gates (all real-data, no mocks): `[225]` v4 KG1–KG13 PASS (KG3 N/A
while agent nodes = 0); `[252]` v1 EF1–EF6 PASS; `[253]` v1 SC1–SC8
PASS (SC7 WARN = 1 unresolved dispute: Law 13 parenthetical labels);
`[227]` v1 FE1–FE6 PASS/N-A; tests 20+/20+ PASS; linter CLEAN.

### Phase 2 — full Theorem NT firewall (AgentNote)

Landed 2026-04-15. The agent-authority firewall is now DB-backed: `edge_allowed`
returns `False` unconditionally for `authority=agent` + causal edge types, and
the only bypass is a `promoted-from` edge in the DB created by `kg.promote()`.
Callers cannot circumvent the firewall by passing a `via_cert` string directly.

`kg.promote(agent_note_id, via_cert, promoter_node_id, broadcast_payload)`:
- Validates agent identity (`authority=agent`) and promoter authority
  (`primary`, `derived`, or `internal`).
- Checks `_meta_ledger.json` staleness: `via_cert` must resolve to PASS
  within 14 days and `git_head` must match current HEAD.
- Validates broadcast payload timestamp within ±60s window.
- Creates `promoted-from` edge with full provenance snapshot.

`_meta_ledger.json` is written by `qa_meta_validator.py` after the family
sweep loop, recording per-cert actual status (PASS/SKIP), timestamp, and
git HEAD. It is gitignored (runtime-derived artifact).

`extractors/agent_notes.py` ingests collab-bus events (session_handoff_note,
kg_promotion), collab_log_activity rows, and OB thoughts with explicit
originSessionId. All produce `authority=agent` Thought nodes.

Cert `[227] QA_KG_FIREWALL_EFFECTIVE_CERT.v1` (FE1–FE6) validates:
- FE1: no unpromoted agent causal edges in DB
- FE2: via_cert on promoted-from edges resolves to PASS in ledger
- FE3: no promoted-from cycles
- FE4: ephemeral test DB with real-shape fixtures
- FE5: WARN oldest unpromoted agent note
- FE6: promoted-from provenance contains broadcast_payload_snapshot

[225] v3 KG3 upgraded to "precondition occupied": N/A only when 0 agent
nodes. When agent nodes exist, checks both firewall layers (unassigned
tier + agent authority). PASS = firewall silent in normal operation.

### Phase 4 — authority-tiered retrieval ranker

Landed 2026-04-16. `KG.search_authority_ranked()` is the agent-facing
retrieval entry point; the legacy `kg.search()` stays available for
cert-validator introspection but its docstring directs agent-facing
callers away. Cert `[254]` enforces formula correctness, determinism,
swallow-discipline, and coverage.

Schema: bumped to v4 with four ranker-input columns (`confidence`,
`valid_from`, `valid_until`, `domain`). Atomic delta — `_migrate_to_v4`
idempotent ALTER ADD; `_check_v4_columns_drift` logs at `init_db`
pointing at `rm tools/qa_kg/qa_kg.db && python -m tools.qa_kg.cli build`.
Application-level `_validate_node_fields` covers confidence range on
old DBs (mirrors Phase 3 source_work pattern).

Closed-form score (single source of truth: `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/ranker_spec.json`):

    score = authority_weight[node.authority]      # primary 10, derived 8, internal 5, agent 1
          × lifecycle_factor[node.lifecycle]       # current 1.0, deprecated/superseded 0.5
          × bm25_norm                              # min-max across candidate pool, FTS5-sign-inverted
          × node.confidence                        # measured signal; default 1.0
          × time_decay                             # 1.0 for axiom/source_*/certified; exp(-Δdays/365) else
          × contradiction_factor                   # 1.5 if contradicts edge present, else 1.0
          × provenance_decay                       # exp(-depth/3) if rooted, no_path_factor (0.5) else

Two-pass candidate pool (`KG._candidate_pool`) — Pass A: ALL FTS5
matches with `authority IN {primary, derived}`, no BM25 cap (primary
material is never silently demoted by low BM25); Pass B: top-K of
remaining authorities by raw BM25. `lifecycle_state='withdrawn'`
excluded entirely from the pool.

Contradiction surfacing is **unconditional** on the public API; the
`include_contradictions` parameter is intentionally absent so agents
cannot opt out of dispute visibility. An internal
`KG._export_clean_subset` exists for documentation export only.

Tiebreak: score DESC, authority_weight DESC, node_id ASC. Deterministic
under fixed `valid_at`. Default `valid_at=None` snapshots wall-clock
once per call; cert `[254]` R7 exercises determinism by passing an
explicit `valid_at`.

Provenance depth materialization threshold: live recursive CTE per
candidate is the Phase 4 implementation. **When nodes > 5,000 OR p95
search latency > 200 ms**, materialize `depth_to_axiom INTEGER` on the
nodes table updated on edge insert. Phase 4 scale (~400 nodes, 113
edges) makes this a no-op; the threshold is documented so the next
reviewer can monitor.

Cert `[254]` gates: R1 (HARD) `min_authority='internal'` excludes agent;
R2 (HARD) per-fixture-query `expected_top_1_authority`; R3 (tri-state)
contradicted material in top-3; R4 (tri-state) `valid_at` filter
correctness — N-A while no node carries `valid_until`; R5 (WARN)
recall@5 vs A-RAG `retrieve_keyword`; R6 (HARD) formula correctness
across ≥6 golden cases; R7 (HARD) determinism; R8 (HARD) no
`except Exception: pass` in `ranker.py` / `search_authority_ranked`;
R9 (HARD) coverage completeness on BOTH axes (`decay_status` ∪
`decay_exempt_status` == `EPISTEMIC_STATUSES`; `authority_weight.keys()`
== `AUTHORITIES`; `lifecycle_factor.keys()` == `LIFECYCLE_STATES \\ {'withdrawn'}`).

Hand-curated 20-query benchmark at `query_fixture.json` — every entry
carries `expected_top_1_authority`, `expected_top_5_node_ids`, `tags`,
and a Will-readable `rationale`. **No LLM-generated queries.**

CLI: `python -m tools.qa_kg.cli search-ranked "<query>" [--min-authority …] [--domain …] [--valid-at …] [-k …]`.

### Phase 5 — graph determinism

Landed 2026-04-16. Canonical serialization of the KG (nodes + edges + a
whitelisted `meta` slice) excludes build-time metadata (`created_ts`,
`updated_ts`, `vetted_ts`, `last_check_*`), NFC-normalizes every string
value, orders nodes by `id` and edges by `(src_id, dst_id, edge_type)`,
and hashes the resulting JSONL byte-streams. Module:
`tools/qa_kg/canonicalize.py`. Single helper: `graph_hash(conn)` returns
the SHA256 hex of the canonical graph.

`_meta_ledger.json` is extended with optional `graph_hash` and
`fixture_hash` fields on the `[228]` entry (additive — all other
entries unchanged). The Phase 5 section of `kg.promote()` enforces
HARD staleness: when the ledger carries a `graph_hash`, a live mismatch
raises `FirewallViolation` with `graph_hash drift` in the message. This
is stronger than Phase 2's 14-day timestamp staleness; the operational
contract is documented in `qa_kg_determinism_cert_v1/README.md` —
after any DB-mutating extractor pass, `qa_meta_validator.py` must be
rerun before subsequent `kg.promote()` calls succeed.

Frozen corpus fixture: `tools/qa_kg/fixtures/corpus_snapshot_v1/` with
`CANONICAL_MANIFEST.json` (hash over `files[]` only — metadata
excluded per C#4), `memory_md_sample.md` (MEMORY.md snapshot; live
MEMORY.md lives outside the repo at `~/.claude/...`), and `ob_sample.md`
(6 thoughts exercising internal + agent authority + cert/axiom keyword
edges). `BuildContext.from_fixture(path)` threads the fixture's
`memory_md_path` and `ob_markdown_path` into the pipeline without
monkey-patching. CLI exposes `build --fixture <dir>` and
`build --hash-only` (D3 contract) and a new `hash` subcommand.

Cert `[228] QA_KG_DETERMINISM_CERT.v1` gates:

- D1 (HARD) fixture content hash matches `expected_hash.fixture_hash`
- D1.5 (WARN) `manifest.repo_head` vs current HEAD (never blocks)
- D2 (HARD) in-process rebuild twice gives byte-identical `graph_hash`
- D3 (HARD) subprocess rebuild matches D2
- D4 (SHOULD or N-A) Linux parity scaffolded; non-Linux N-A
- D5 (HARD) ledger `graph_hash` matches live; bootstrap PASS on first
  run with explicit rationale string (tri-state honesty from [254] R4)
- D6 (HARD) `kg.promote()` rejects on `graph_hash` drift (ephemeral DB
  test with sentinel mismatch)
- D7 (HARD) no `except Exception: pass` / bare `except: pass` in Phase
  5 modules (AST scan)

Side-channel pattern: `[228]` validator publishes the live graph_hash
to `_kg_graph_hash_by_fam_id["228"]` in `qa_meta_validator.py`; the
ledger writer attaches it to the `[228]` entry. Fixture hash reads
from `expected_hash.json` at ledger write time.

C#1 call-site coverage: `arag.search` / `arag.promote_to_kg` accept
`db_path=None` kwarg; `memory_rules.populate` accepts
`memory_md_path=None`. `BuildContext` threads both to the extractors
during fixture-driven builds. Unit test
`test_arag_extractor_respects_db_path_override` guards regression.

### Phase 6 — agent integration (MCP write surface)

Landed 2026-04-16. `tools/qa_kg_mcp/server.py` is a stdlib-only stdio
JSON-RPC server (no `mcp` SDK dependency) exposing EXACTLY four tools:

- `qa_kg_search` — delegates to `kg.search_authority_ranked`. READ ONLY.
- `qa_kg_get_node` — single-node lookup + epistemic fields + lifecycle.
  READ ONLY. Miss → `{error: "not_found"}`, never raises.
- `qa_kg_neighbors` — in/out edges with `edge_types` + `direction`.
  READ ONLY.
- `qa_kg_promote_agent_note` — escalates an existing AgentNote via
  `kg.promote()`. READ_WRITE only. Rate-limited + tamper-evident.

Capability is declared at server spawn via `--cap {read_only,read_write}`.
A READ_ONLY session's `tools/list` response OMITS `qa_kg_promote_agent_note`
entirely — agents cannot discover or self-elevate to the write tool.

The write surface ONLY promotes existing AgentNotes. It cannot create
nodes. AgentNote rows are written exclusively by
`tools/qa_kg/extractors/agent_notes.py` (from collab-bus events / OB
thoughts with `originSessionId` / `collab_log_activity` rows).

Per-session rate limit (`MAX_WRITES_PER_SESSION=50`, configurable via
`QA_KG_MCP_MAX_WRITES` env var) lives in
`qa_alphageometry_ptolemy/_agent_writes.json` under flock.
**File is parallel to `_meta_ledger.json`, never shares its flock.**
This was plan v2 M1 — the two writers (meta-validator sweep vs.
rate_limit per-promote) never race. Counter decays on explicit
`session_done` broadcast; crashed sessions recovered via
`python -m tools.qa_kg_mcp.cli reset-writes <session>`.

`broadcast_payload` is deep-copied by the MCP server and
`mcp_session` stamped AFTER the copy, so any spoofed `mcp_session` key
in the agent-provided dict is OVERWRITTEN, not preserved. The
`qa_security_audit.check_mcp_provenance` check flags `promoted-from`
edges whose provenance lacks the stamp — catching direct-write
bypass attempts.

`qa_security_audit.check_qa_kg_db_direct_writes` scans the LLM-wrapper
ledger for ALLOWed Bash calls matching `sqlite3 ... qa_kg.db ...
INSERT/UPDATE/DELETE/...`. Any match is FAIL. The hook itself
(`llm_qa_wrapper/cert_gate_hook.py`) is `WRAPPER_SELF_MODIFICATION`-
protected, so the W3a forensic detection lives in the audit layer.

W5 authority is IMMUTABLE in both directions: `kg.upsert_node` raises
`FirewallViolation("authority_immutable")` on agent → internal
(upgrade) AND primary → agent (silent downgrade). Real authority
corrections require explicit delete + recreate.

Every MCP tool call writes to `query_log`:
- Reads: one row per returned node, `rank >= 0`.
- Writes: one row, `rank = -1` sentinel.

No new schema columns; `query_log` is reused verbatim (schema v4 stays).

New cert family:

- **[255] QA_KG_AGENT_WRITE_SURFACE_CERT.v1** — W1 MCP surface = 4 tools
  (AST), W2 agent-authority upserts confined to extractor + tests +
  cert validators, W3a direct-write detector fires, W3b MCP provenance
  detector fires, W4 rate limit raises at cap, W5 authority immutable
  both directions, W6 READ_ONLY hides promote from `tools/list`, W7
  every MCP tool call audited, W8 no except-Exception-pass swallows.
  All HARD.

Alpha-bar flipped: QA-MEM is alpha agent memory + authoritative
project memory per the original three-cert bar.

## Roadmap (deferred, operational hardening — NOT alpha-bar items)

- **Phase 4.5:** full primary-source corpus ingestion (Ben/Dale books,
  full Wildberger corpus, Levin 2026); extractor confidence-from-method
  maps (e.g., `source_claims.populate` setting `confidence=0.7` on
  `extraction_method='ocr'`); domain/valid_from/valid_until population
  on real nodes; URL scheme in `tools/qa_kg/locators.py` resolver. Phase 3
  shipped the mechanism + minimum seed; Phase 4.5 is the scaled pass.
  Not alpha-bar — corpus scale is post-alpha polish.
  **Ops item** (separate from corpus work): cert-gate Codex review
  bridge has been dead at cert-submission time in Phase 2/3/4/5/6. Either
  `llm_qa_wrapper/cert_gate_hook.py` grows a pre-commit health-check
  that fails loudly with "bridge is dead, restart it" rather than the
  generic `CODEX_REVIEW_PENDING`, OR `tools/qa_security_audit.py`
  gains a line-item that fails when `qa_lab/logs/codex_bridge.log`
  mtime > 24h. Phase 6 again worked around with `codex exec --full-auto`
  one-shot; that pattern shouldn't become the default.
- **Phase 5.1:** pinned-source reproducibility — extend Phase 5's
  determinism test so that D2/D3 rebuild uses extractor source pinned
  at `manifest.repo_head` via `git archive`, decoupling reproducibility
  from working-tree HEAD advances. `tools/qa_kg/build_context.py`
  carries `BuildContext` + `run_pipeline`; Phase 5.1 adds
  `materialize_pinned_repo` + harness-overlay list. Also: fixture
  tooling (`cli fixture-refresh <dir>`) to regenerate manifest +
  `expected_hash.json` atomically.
  Not alpha-bar — adding a new cert family shifts the fixture rebuild
  hash legitimately (Phase 6 refreshed expected_hash.json once); the
  pinned-source harness eliminates this work in later cert additions.

Per `memory/project_qa_mem_review_role.md`, the alpha bar is
`[228] + [254] + [255]` — all three have PASSed as of 2026-04-16.
QA-MEM may be described as "alpha agent memory" and "authoritative
project memory" from this commit forward.

## References

- Cert family [202] — Aiq Bekar digital root A1-compliance
  (`tools/qa_retrieval/schema.py::dr`, `compute_be`)
- Cert family [225] v3 — QA-KG consistency invariants (KG1–KG10)
  (`qa_alphageometry_ptolemy/qa_kg_consistency_cert_v3/`)
- Cert family [225] v2 — FROZEN, superseded by v3
  (`qa_alphageometry_ptolemy/qa_kg_consistency_cert_v2/`)
- Cert family [226] — Candidate F classifier correctness
  (`qa_alphageometry_ptolemy/qa_semantic_coord_cert_v1/`)
- Cert family [252] — Epistemic fields correctness (EF1–EF6)
  (`qa_alphageometry_ptolemy/qa_kg_epistemic_fields_cert_v1/`)
- Cert family [227] — Firewall effectiveness (FE1–FE6)
  (`qa_alphageometry_ptolemy/qa_kg_firewall_effective_cert_v1/`)
- Cert family [225] v4 — current; KG1–KG13 (Phase 3 adds KG11–KG13)
  (`qa_alphageometry_ptolemy/qa_kg_consistency_cert_v4/`)
- Cert family [253] — SourceClaim / contradicts contracts (SC1–SC8)
  (`qa_alphageometry_ptolemy/qa_kg_source_claims_cert_v1/`)
- Cert family [254] — Authority-tiered ranker (R1–R9; R3+R4 tri-state, R5 WARN)
  (`qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/`)
- Cert family [228] — Graph determinism (D1–D7; D1.5 WARN, D4 Linux-only)
  (`qa_alphageometry_ptolemy/qa_kg_determinism_cert_v1/`)
- Canonicalizer module: `tools/qa_kg/canonicalize.py`
- Build context: `tools/qa_kg/build_context.py` (BuildContext + run_pipeline)
- Phase 5 fixture: `tools/qa_kg/fixtures/corpus_snapshot_v1/` (memory_md_sample.md, ob_sample.md, CANONICAL_MANIFEST.json, repo_head.txt)
- Ranker module: `tools/qa_kg/ranker.py`
- Ranker spec (single source of truth): `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/ranker_spec.json`
- 20-query benchmark: `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/query_fixture.json`
- `docs/theory/svp_wiki_qa_elements_snapshot.md` — SVP wiki primary-source
  snapshot (2026-04-16 fetch), locator target for Phase 3 SourceClaims
- `QA_AXIOMS_BLOCK.md` (Dale 2026) — canonical QA axiom set A/T/S/F groups
- `qa_orbit_rules.py` — canonical `orbit_family` classifier
- `tools/qa_retrieval/schema.py` — A-RAG Candidate F reference implementation
- `memory/project_qa_mem_architecture.md` — living design notes
