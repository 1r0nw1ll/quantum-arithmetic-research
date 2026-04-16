# [253] QA-KG Source Claims Cert v1

**Status:** PASS (SC1–SC8; meta-validator registered)
**Created:** 2026-04-16 (Phase 3)
**Source:** Will Dale + Claude; docs/specs/QA_MEM_SCOPE.md (Dale, 2026); `tools/qa_kg/extractors/source_claims.py`; `tools/qa_kg/fixtures/source_claims_phase3.json`; SVP wiki snapshot at `docs/theory/svp_wiki_qa_elements_snapshot.md` (fetched from svpwiki.com/Quantum+Arithmetic+Elements).

## Claim

QA-MEM Phase 3 contract for SourceClaim / SourceWork / contradicts
ingestion. Eight gates validate that quoted material from primary
sources (SVP wiki, Pond notes, etc.) is well-formed in the live graph
and that contradicts edges carry the closed-set provenance needed for
Phase 4 authority-ranked retrieval.

The single source of truth for closed sets is
`qa_alphageometry_ptolemy/qa_kg_source_claims_cert_v1/closed_sets.json`:

- `reasons`: `{ocr, variant, typo, dispute, true}` — on contradicts edges.
- `extraction_methods`: `{manual, ocr, llm, script}` — on SourceClaim nodes.
- `contradicts_endpoints.forbidden_node_types`: `{Axiom}`.
- `contradicts_endpoints.forbidden_authorities`: `{agent}`.

Locator resolution reuses `tools/qa_kg/locators.py::resolve_any` —
shared with [252] EF4 so the two certs cannot drift on file/pdf/cert
scheme semantics.

## Gates

| Gate | Level | Description |
|------|-------|-------------|
| SC1 | HARD | Every SourceClaim has non-empty body (quote) AND `source_locator` that resolves via `tools.qa_kg.locators.resolve_any`. |
| SC2 | HARD | Every SourceClaim has exactly one outgoing `quoted-from` edge whose target is a SourceWork (`epistemic_status=source_work`). |
| SC3 | HARD | SourceClaim `method` ∈ `{manual, ocr, llm, script}`. |
| SC4 | HARD | No two SourceClaims with identical `source_locator` AND different `body` without a `contradicts` edge between them. (Identical body + different locators is allowed — same quote across editions is expected.) |
| SC5 | HARD | Every `contradicts` edge's provenance JSON parses and contains a `reason` key whose value ∈ `reasons` closed set. |
| SC6 | HARD | No `contradicts` cycles (scoped to the populated graph; complements [225] KG2). |
| SC7 | WARN | Count of `contradicts` edges with `reason=dispute` — triage queue sensor. Value is surfaced into `_meta_ledger.json` via the Phase 3 §12a WARN capture. |
| SC8 | HARD | No `contradicts` edge endpoint (src or dst) has `node_type=Axiom` or `authority=agent`. Axioms don't contradict (rule-system bug, belongs in linter); agent dissent must flow through `kg.promote()` first. |

## Phase 3 seed (real-data exercise)

All 5 `reason` values are exercised against verifiable quotes from the
SVP wiki snapshot:

| reason | exemplar |
|--------|----------|
| typo | L = (a²−h²)/2 (should be (a²−b²)/2); C identity; Law 12(e); music interval mislabels |
| ocr | "choromosomes" (line 66); "3-diminsional" (line 50) |
| variant | L = FC/2 (line 546) contradicts L = C·F/12 (line 385) — same work, internal inconsistency |
| dispute | Laws 13(a)/(b) parenthetical labels `(b)/(a)` — ambiguous; Dale not yet confirmed |
| true | "F are never prime" (line 308) — FALSE for Unity F=3 |

SourceClaim endpoints also include three `supersedes` edges forming the
`cert:fs:qa_kg_consistency_cert_v4 → v3 → v2 → v1` chain — this is what
[225] v4 KG13 checks against.

## Companion certs

- [[225]](225_qa_kg_consistency_cert_v4.md) v4 — shares KG11/KG12/KG13 coverage of the same ingested nodes (node-level invariants).
- [[252]](252_qa_kg_epistemic_fields_cert_v1.md) — EF3 `allowed_matrix.json` gained `source_work` under `primary` in Phase 3.
- [[227]](227_qa_kg_firewall_effective_cert_v1.md) — SC8 forbids `authority=agent` endpoints, complementing the Phase 2 agent firewall.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_kg_source_claims_cert_v1/qa_kg_source_claims_cert_validate.py`
- Closed sets: `qa_alphageometry_ptolemy/qa_kg_source_claims_cert_v1/closed_sets.json`
- Mapping protocol: `qa_alphageometry_ptolemy/qa_kg_source_claims_cert_v1/mapping_protocol_ref.json`
- Extractor: `tools/qa_kg/extractors/source_claims.py`
- Seed fixture: `tools/qa_kg/fixtures/source_claims_phase3.json`
- Primary-source snapshot: `docs/theory/svp_wiki_qa_elements_snapshot.md`
