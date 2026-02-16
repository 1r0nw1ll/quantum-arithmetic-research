# [39] QA EBM Verifier Bridge Cert (QA_EBM_VERIFIER_BRIDGE_CERT.v1)

This family makes **verifier-gated acceptance** a first-class, machine-checkable QA artifact.

It is the minimal bridge that upgrades:

- “terminal state reached” (navigation witness)

into:

- “terminal state accepted by a verifier” (typed verdict + binding witness).

## Machine tract

Directory: `qa_ebm_verifier_bridge_cert/`

Files:

- `qa_ebm_verifier_bridge_cert/schema.json`
- `qa_ebm_verifier_bridge_cert/validator.py`
- `qa_ebm_verifier_bridge_cert/fixtures/`
- `qa_ebm_verifier_bridge_cert/mapping_protocol_ref.json` (Gate 0 intake)

### What it validates (gates)

- **Gate 1 — Schema validity**: strict schema, no extra fields
- **Gate 2 — Canonical hash**: `digests.canonical_sha256` matches canonical compact JSON hash
- **Gate 3 — Bridge coherence**:
  - binds `subject.state_after` → `subject.state_sha256`
  - enforces `verdict.passed ↔ verdict.fail_type` coherence (`passed=true ⇒ fail_type=OK`)
  - binds duplicated fields in `invariant_diff`

### Run

```bash
python qa_ebm_verifier_bridge_cert/validator.py --self-test
python qa_ebm_verifier_bridge_cert/validator.py qa_ebm_verifier_bridge_cert/fixtures/valid_min.json
```

## Integration

`QA_EBM_NAVIGATION_CERT.v1` supports optional verifier-coupling:

- If `trace.outcome.accepted_by_verifier=true`, it must include `verifier_bridge_ref` and digest-link it via
  `digests.refs`.

## Human tract

Concept mapping (EBM reasoning / Kona podcast → QA):

- `Documents/QA_MAP__EBM_REASONING_KONA_PODCAST.md`
- `Documents/QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json`

