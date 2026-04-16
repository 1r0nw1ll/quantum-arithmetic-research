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
retained one release cycle.

## Roadmap (deferred, NOT in scope here)

- **Phase 1:** epistemic fields — `authority`, `epistemic_status`, `method`;
  split node types; backfill from `MEMORY.md` / `CLAUDE.md` / `QA_AXIOMS_BLOCK.md`.
- **Phase 2:** real firewall — `AgentNote → *` requires `promoted-from` with
  `via_cert`; cert `[227] QA_KG_FIREWALL_EFFECTIVE`.
- **Phase 3:** contradictions + sources — SourceClaim nodes with quote+locator,
  seed `contradicts` for SVP wiki typos and Volk/Dale reconciliations.
- **Phase 4:** authority-filtered retrieval + provenance-aware ranker.
- **Phase 5:** determinism — frozen corpus fixture, graph-hash cert `[228]`.
- **Phase 6:** agent integration — shadow-mode parallel to A-RAG before any
  replacement.

## References

- Cert family [202] — Aiq Bekar digital root A1-compliance
  (`tools/qa_retrieval/schema.py::dr`, `compute_be`)
- Cert family [225] v2 — QA-KG consistency invariants
  (`qa_alphageometry_ptolemy/qa_kg_consistency_cert_v2/`)
- Cert family [226] — Candidate F classifier correctness
  (`qa_alphageometry_ptolemy/qa_semantic_coord_cert_v1/`)
- `QA_AXIOMS_BLOCK.md` (Dale 2026) — canonical QA axiom set A/T/S/F groups
- `qa_orbit_rules.py` — canonical `orbit_family` classifier
- `tools/qa_retrieval/schema.py` — A-RAG Candidate F reference implementation
- `memory/project_qa_mem_architecture.md` — living design notes
