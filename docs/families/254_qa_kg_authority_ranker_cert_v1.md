# [254] QA-KG Authority Ranker Cert v1

**Status:** PASS (R1, R2, R3, R6, R7, R8, R9; R4 N-A pending Phase 4.5; R5 WARN)
**Created:** 2026-04-16 (Phase 4)
**Source:** Will Dale + Claude; docs/specs/QA_MEM_SCOPE.md (Dale, 2026); `tools/qa_kg/ranker.py`; `tools/qa_kg/kg.py::KG.search_authority_ranked`; `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/ranker_spec.json`; `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/query_fixture.json`.

## Claim

QA-MEM Phase 4 introduces an authority-tiered retrieval ranker. The
plain `kg.search()` (FTS5/BM25) is preserved for raw introspection but
**must not** be used by agent-facing callers — agents must call
`KG.search_authority_ranked()`, which composes a closed-form score:

```
score = authority_weight[node.authority]      # primary 10, derived 8, internal 5, agent 1
      × lifecycle_factor[node.lifecycle]       # current 1.0, deprecated/superseded 0.5
      × bm25_norm                              # min-max across candidate pool, FTS5-sign-inverted
      × node.confidence                        # measured signal; default 1.0
      × time_decay                             # 1.0 for axiom/source_*/certified; exp(-Δdays/365) else
      × contradiction_factor                   # 1.5 if node has contradicts edge, else 1.0
      × provenance_decay                       # exp(-depth/3) if rooted, no_path_factor (0.5) else
```

Single source of truth for formula constants is
`ranker_spec.json` (loaded once by `tools.qa_kg.ranker.load_spec`,
re-loaded by the validator for cross-check). Cert R6 verifies
`compose_score` matches the spec exactly across ≥6 golden cases; cert
R9 enforces that the spec covers every value in the schema enums and
the `[252]` allowed-matrix.

The candidate pool is built via a **two-pass** policy
(`KG._candidate_pool`):

1. **Pass A** — every FTS5 match with `authority IN {primary, derived}`,
   no BM25 cap. Primary material is never silently demoted by low BM25.
2. **Pass B** — top-`(candidate_pool_k - len(A))` matches with
   `authority IN allowed \ {primary, derived}`, ordered by raw BM25.

Contradiction surfacing is **unconditional** on the public API
(`include_contradictions` parameter is intentionally absent — see
Plan §M3). For documentation export the internal
`KG._export_clean_subset` helper exists; agents must not use it.

## Gates

| Gate | Level | Description |
|------|-------|-------------|
| R1 | HARD | `min_authority='internal'` returns no nodes with `authority='agent'`. Verified per fixture query. Currently vacuous-PASS in the live DB (0 agent nodes), wired so a regression catches once agent extractors land. |
| R2 | HARD | For every fixture query, `results[0].authority == query.expected_top_1_authority`. No aggregate slack — each fixture query carries its own expected tier and rationale. |
| R3 | HARD or N-A | For every fixture query tagged `contradicted_material`, every contradicting node of the matched node appears in top-3 with `contradiction_state != 'none'`. Tri-state: N-A when zero such queries are runnable in the live DB. |
| R4 | HARD or N-A | `valid_at` filter excludes nodes with `valid_until < valid_at`. Tri-state: N-A when no node carries `valid_until != ''` (Phase 4 default — extractors don't populate yet). PASS only when ≥1 node carries `valid_until` AND filter excludes correctly. |
| R5 | WARN | Recall@5 of `search_authority_ranked` vs A-RAG `retrieve_keyword` baseline within ±10%. WARN-only because the two retrieval surfaces have legitimate divergence at the candidate-pool level. |
| R6 | HARD | `compose_score` formula correctness — given fixed inputs, score matches spec to within 1e-6. ≥6 golden cases cover: each authority tier × decayed/exempt status × contradicted/clean × provenance depths in {0,1,3,5,-1} × valid_from precedence × confidence < 1.0 × lifecycle current vs superseded. |
| R7 | HARD | Determinism — 5× re-run on same query/DB/spec under fixed `valid_at` returns identical RankedHit lists (id + score). Catches non-deterministic tiebreak. |
| R8 | HARD | No `except Exception: pass` / bare `except: pass` swallows in `tools/qa_kg/ranker.py` or `KG.search_authority_ranked`. AST scan, not regex. |
| R9 | HARD | Coverage completeness on BOTH axes against `qa_kg_epistemic_fields_cert_v1/allowed_matrix.json`: (a) `decay_exempt_status ∪ decay_status == EPISTEMIC_STATUSES`; (b) `authority_weight.keys() == AUTHORITIES == set(allowed_matrix.allowed.keys())`; (c) `lifecycle_factor.keys() == LIFECYCLE_STATES \ {'withdrawn'}`; (d) decay sets disjoint. Code-side companion: `ranker.authority_weight` and `ranker.lifecycle_factor` raise `KeyError` on unknown values. |

## Open decisions (resolved in plan; locked here for audit)

| ID | Decision | Locked value |
|----|----------|--------------|
| D1 | Schema delta scope | All 4 columns (confidence, valid_from, valid_until, domain) atomic v3 → v4 |
| D2 | provenance_depth | Live recursive CTE; materialize at threshold (nodes > 5,000 OR p95 > 200ms) |
| D3 | bm25 normalization | Per-query min-max across candidate pool, FTS5-sign-inverted |
| D4 | contradiction_prior | 1.5 (≈ +1 authority-tier of visibility; defended via R3) |
| D5 | Old `kg.search()` | Keep, docstring deprecation, no runtime warning (Phase 6 owns the MCP boundary) |

## Hand-curated 20-query benchmark

`query_fixture.json` is hand-drafted against the live DB at fixture-
authoring time. Every entry has `expected_top_1_authority`,
`expected_top_5_node_ids`, `tags`, and a `rationale` paragraph
documenting WHY the ranker should produce that result. **No LLM-
generated queries.** Coverage matrix:

| Tag | # queries | Exercises |
|-----|-----------|-----------|
| `primary_dominance` | 7 (q01-q05, q11-q14 partial) | R2 — primary axioms / source claims top derived/internal alternatives |
| `M2_floor` | 1 (q01) | M2 plan check — primary axiom must surface despite low BM25 |
| `derived_certified` | 6 (q06, q07, q08, q09, q10, q18, q19, q20) | R2 — derived certs top on cert-related queries |
| `internal_authority` | 3 (q15, q16, q17) | R1 + R2 — internal rules surface when no primary/derived competitor |
| `contradicted_material` | 4 (q11, q12, q13, q14) | R3 + R6 contradiction_factor |
| `deprecated_lifecycle` | 1 (q10) | Lifecycle factor — v4 (current) tops superseded v1/v2/v3 |
| `domain_filter` | 1 (q19) | Wired for Phase 4.5; vacuous today |
| `valid_at_filter` | 1 (q20) | R4 tri-state — currently N-A |

`agent_excluded` tag is intentionally absent — there are 0 agent nodes
in the live DB, so R1 runs across all 20 queries as a regression
sentinel rather than a positive proof.

## Companion certs

- [[225]](225_qa_kg_consistency_cert_v4.md) v4 — KG-consistency invariants (lifecycle alignment, supersedes DAG)
- [[252]](252_qa_kg_epistemic_fields_cert_v1.md) — `allowed_matrix.json` is the authoritative axis source for R9
- [[253]](253_qa_kg_source_claims_cert_v1.md) — populates the contradicts edges that R3 surfaces
- [[227]](227_qa_kg_firewall_effective_cert_v1.md) — Phase 2 agent firewall complementing R1

## Artifacts

- Ranker module: `tools/qa_kg/ranker.py`
- Public API: `tools/qa_kg/kg.py::KG.search_authority_ranked`
- CLI: `python -m tools.qa_kg.cli search-ranked "<query>" [--min-authority …] [--domain …] [--valid-at …] [-k …]`
- Validator: `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/qa_kg_authority_ranker_cert_validate.py`
- Spec (single source of truth): `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/ranker_spec.json`
- 20-query benchmark: `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/query_fixture.json`
- Mapping protocol: `qa_alphageometry_ptolemy/qa_kg_authority_ranker_cert_v1/mapping_protocol_ref.json`
- Unit tests: `tools/qa_kg/tests/test_ranker.py`

## Phase 4 boundary (alpha-bar)

After this commit QA-MEM is "indexed, firewalled, source-grounded,
contradiction-aware, **authority-ranked** catalog of repo artifacts."
Per the Phase 3 review's alpha-bar rule, QA-MEM **still cannot** be
called "agent memory" until Phases 4 + 5 + 6 all PASS. Phase 4 ships
the ranker primitives; Phase 6 wires them into MCP.
