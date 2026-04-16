# [225] QA-KG Consistency Cert v4

**Status:** PASS (KG1–KG13; meta-validator registered)
**Created:** 2026-04-16
**Supersedes:** [225] v3 (frozen), which superseded v2, which superseded v1.
**Source:** Will Dale + Claude; docs/specs/QA_MEM_SCOPE.md; QA_AXIOMS_BLOCK.md (Dale, 2026); CLAUDE.md; cert [226]; cert [253].

## Claim

QA-KG graph consistency under schema v3 (stable through Phase 3) + Phase 3
SourceWork / SourceClaim / supersedes invariants. Thirteen gates validate
structural, epistemic, static-analysis, and Phase 3 ingestion invariants
on the live `tools/qa_kg/qa_kg.db` graph.

## Gates

| Gate | Level | Description |
|------|-------|-------------|
| KG1 | HARD | No self-vetting — `vetted_by != node.id` |
| KG2 | HARD | No contradicts cycles |
| KG3 | HARD/N-A | Theorem NT firewall — precondition occupied (Phase 2 DB-backed). N/A only when `authority='agent'` count is 0. Checks both Layer 1 (Unassigned→Cosmos/Singularity causal without via_cert) and Layer 2 (agent causal without DB-backed promoted-from edge). PASS = firewall silent. |
| KG4 | WARN | Satellite orphan aging > 30d |
| KG5 | HARD | `tier ≡ qa_orbit_rules.orbit_family(idx_b, idx_e)` |
| KG6 | HARD | Candidate F [202] integrity: `idx_b == dr(char_ord_sum)`, `idx_e == NODE_TYPE_RANK[node_type]` |
| KG7 | HARD | `authority` and `epistemic_status` non-null on every node |
| KG8 | HARD | Frozen cert dirs not in FAMILY_SWEEPS |
| KG9 | HARD | `edges.AXIOM_CODES` canonical — equals `axioms.CANONICAL_AXIOM_CODES` parsed from CLAUDE.md |
| KG10 | HARD | No `except Exception: continue` patterns in `tools/qa_kg/extractors/*.py` (static AST scan) |
| KG11 | HARD | **Phase 3.** Every SourceWork has `authority=primary` AND `epistemic_status=source_work` AND `node_type=Work`. |
| KG12 | HARD | **Phase 3.** Every SourceClaim has a `quoted-from` edge whose target exists and has `epistemic_status=source_work`. |
| KG13 | HARD | **Phase 3.** Supersedes edges form a DAG (no cycles); every node with `lifecycle_state='superseded'` has ≥1 incoming supersedes edge; `lifecycle_state='deprecated'` nodes are exempt (deprecation is retirement without replacement). Pinned direction: newer→older. |

## What's new in v4 vs v3

- Adds KG11/KG12/KG13 for Phase 3 SourceWork / SourceClaim / supersedes invariants.
- Schema version unchanged (v3 → v3 — `EPISTEMIC_STATUSES` gains `"source_work"` as an additive enum value, no column churn).
- The lifecycle bridge (`tools/qa_kg/extractors/certs.py::_lifecycle_for_status`) now translates `_status: frozen` file markers to `lifecycle_state` ∈ {`superseded`, `deprecated`} — keeping KG8 (file-level) and KG13 (node-level) aligned.

## Companion certs

- [[252]](252_qa_kg_epistemic_fields_cert_v1.md) — authority × epistemic_status matrix + source_locator resolution (Phase 1 base).
- [[227]](227_qa_kg_firewall_effective_cert_v1.md) — DB-backed agent firewall effectiveness (Phase 2).
- [[253]](253_qa_kg_source_claims_cert_v1.md) — SourceClaim / contradicts / SC1–SC8 contracts (Phase 3 sibling).

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_kg_consistency_cert_v4/qa_kg_consistency_cert_validate.py`
- Mapping protocol: `qa_alphageometry_ptolemy/qa_kg_consistency_cert_v4/mapping_protocol_ref.json`
- DB under test: `tools/qa_kg/qa_kg.db` (gitignored — rebuild with `python -m tools.qa_kg.cli build`).
