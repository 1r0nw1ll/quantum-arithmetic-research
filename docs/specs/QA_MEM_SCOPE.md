<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG internal scope document; grounds in cert families [202], [225] v2, [226], and project-internal authority docs -->
# QA-MEM Scope (Phase 0)

**Status:** Phase 0 — landmine fixes landed; Phase 1+ deferred.
**Authority:** cert families `[225] v2` (graph consistency) + `[226]` (Candidate F classifier invariants) + `[202]` (Aiq Bekar digital root).
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

### Phase 2 (pending) — full Theorem NT firewall

Not yet landed. Will add:
- `kg.promote()` protocol with `_meta_ledger.json` staleness guard and
  collab-bus broadcast corroboration.
- `extractors/agent_notes.py` — OB-with-originSessionId + collab events.
- Cert `[227] QA_KG_FIREWALL_EFFECTIVE` (FE1–FE6).
- [225] v3 KG3 upgrade from tri-state to "precondition occupied."

## Roadmap (deferred, NOT in scope)

- **Phase 3:** contradictions + sources — SourceClaim nodes with quote+locator,
  seed `contradicts` for SVP wiki typos and Volk/Dale reconciliations.
  `supersedes` edge type for lifecycle_state transitions.
- **Phase 4:** authority-filtered retrieval + provenance-aware ranker.
- **Phase 5:** determinism — frozen corpus fixture, graph-hash cert `[228]`.
- **Phase 6:** agent integration — shadow-mode parallel to A-RAG before any
  replacement.

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
- `QA_AXIOMS_BLOCK.md` (Dale 2026) — canonical QA axiom set A/T/S/F groups
- `qa_orbit_rules.py` — canonical `orbit_family` classifier
- `tools/qa_retrieval/schema.py` — A-RAG Candidate F reference implementation
- `memory/project_qa_mem_architecture.md` — living design notes
