# [252] QA-KG Epistemic Fields Cert v1

**Status:** PASS (EF1–EF6; meta-validator registered)
**Created:** 2026-04-15
**Source:** Will Dale + Claude; docs/specs/QA_MEM_SCOPE.md; QA_AXIOMS_BLOCK.md (Dale, 2026); CLAUDE.md.

## Claim

Every node in the QA-KG has a valid authority and epistemic_status assignment,
forming a 4×6 allowed matrix that is orthogonal to the Candidate F retrieval
index. The authority axis records WHO produced the knowledge; the epistemic
axis records WHAT KIND of claim it is.

## Authority values (locked — 4 crisp)

| Value | Meaning |
|-------|---------|
| primary | External canon predating QA (Dale/Ben/Keely/Pond/Wildberger/Levin) |
| derived | Cert machinery outputs (validators, structural provenance) |
| internal | Will-authored project material (MEMORY.md, CLAUDE.md, OB without session) |
| agent | Claude/Codex/OpenCode outputs (OB with session, collab-bus events) |

## Allowed matrix (EF3)

| authority | permitted epistemic_status |
|-----------|---------------------------|
| primary | axiom, source_claim |
| derived | certified, observation |
| internal | interpretation, observation |
| agent | conjecture, observation |

Single source of truth: `allowed_matrix.json` in the cert directory.

## Gates

| Gate | Level | Description |
|------|-------|-------------|
| EF1 | HARD | Every node has non-null authority |
| EF2 | HARD | Every node has non-null epistemic_status |
| EF3 | HARD | (authority, epistemic_status) in allowed matrix |
| EF4 | HARD | authority=primary implies source_locator present + resolves |
| EF5 | WARN | Count of authority=agent nodes (pressure sensor) |
| EF6 | HARD | Axiom nodes must be primary + axiom |

## Artifacts

- `qa_alphageometry_ptolemy/qa_kg_epistemic_fields_cert_v1/`
- Validator: `qa_kg_epistemic_fields_cert_validate.py`
- Matrix: `allowed_matrix.json`
- Mapping: `mapping_protocol_ref.json`
