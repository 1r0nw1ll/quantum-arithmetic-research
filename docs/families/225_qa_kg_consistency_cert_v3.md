# [225] QA-KG Consistency Cert v3

> **FROZEN — superseded by v4.** See [225_qa_kg_consistency_cert_v4.md](225_qa_kg_consistency_cert_v4.md).
> Phase 3 (2026-04-16) added gates KG11/KG12/KG13 for SourceWork / SourceClaim /
> supersedes invariants; v3 is retained here for audit symmetry with v1 and v2.

**Status:** FROZEN (was PASS KG1–KG10; no longer run as a production gate)
**Created:** 2026-04-15
**Supersedes:** [225] v2 (frozen)
**Source:** Will Dale + Claude; docs/specs/QA_MEM_SCOPE.md; QA_AXIOMS_BLOCK.md (Dale, 2026); CLAUDE.md; cert [226].

## Claim

QA-KG graph consistency under schema v3 (Phase 1 epistemic fields + alias
removal). Ten gates validate structural, epistemic, and static-analysis
invariants.

## Gates

| Gate | Level | Description |
|------|-------|-------------|
| KG1 | HARD | No self-vetting — vetted_by != node id |
| KG2 | HARD | No contradicts-cycles |
| KG3 | HARD/NA | Theorem NT firewall tri-state (PASS/FAIL/N-A) |
| KG4 | WARN | Satellite orphan aging > 30 days |
| KG5 | HARD | tier matches canonical orbit classifier |
| KG6 | HARD | Candidate F integrity per [202] |
| KG7 | HARD | authority + epistemic_status non-null on every node |
| KG8 | HARD | Frozen cert dirs not in FAMILY_SWEEPS |
| KG9 | HARD | AXIOM_CODES in edges.py equals canonical set from CLAUDE.md |
| KG10 | HARD | No except-Exception-continue patterns in extractors |

## v3 additions (over v2)

- **KG7** ensures Phase 1 epistemic backfill is complete — no node may have
  NULL authority or epistemic_status.
- **KG8** (N1 carry-forward) prevents frozen cert directories from remaining
  in FAMILY_SWEEPS after supersession.
- **KG9** (N2 carry-forward) eliminates phantom axiom codes (previously A3/A4
  existed in edges.py without definitions).
- **KG10** (N3 carry-forward) catches silent exception swallowing in edge
  extractors that would mask FirewallViolation.
- Back-compat aliases (Coord/compute_be/tier_for_coord) removed atomically
  with this schema bump per the Phase 0 pin.

## Artifacts

- `qa_alphageometry_ptolemy/qa_kg_consistency_cert_v3/`
- Validator: `qa_kg_consistency_cert_validate.py`
- Mapping: `mapping_protocol_ref.json`
