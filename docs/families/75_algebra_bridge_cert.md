# Family [75]: QA Algebra Bridge Cert

## Purpose

Family [75] provides a semantic anchor certificate for QA structural algebra so downstream families can reference a pinned semantics hash instead of re-specifying generator conventions.

It certifies:

- generator semantics reference,
- word application order,
- component bridge behavior (`state <-> (word, scale, normalized)`), and
- deterministic semantics hash binding.

## Location

`qa_algebra_bridge_cert_v1/`

## Schema

`QA_ALGEBRA_BRIDGE_CERT.v1`

## Gates

1. **Gate 1 — Schema shape / type checks**
2. **Gate 2 — Deterministic generator probe recomputation**
3. **Gate 3 — Word convention / order probes**
   - includes explicit left-to-right anchors: `RL -> (2,3)`, `LR -> (3,2)` from seed `(1,1)`
4. **Gate 4 — Component-bridge probes (word + scale)**
5. **Gate 5 — Semantics hash binding + invariant diff claim verification**

## Semantic Anchor

- `semantics_id = QA_ALGEBRA_SEMANTICS.v1`
- `generator_semantics_ref = QA_GENERATORS_SIGMA_SHEAR.v1`
- `word_application_order = left_to_right`

The cert includes `semantics.semantics_sha256 = sha256(canonical_json(semantics_payload))` and Gate 5 enforces equality.

## Encoding Convention

- Word token `L` means apply `sigma`.
- Word token `R` means apply generator `R := mu sigma mu`.
- Words are applied left-to-right to the seed state.

## Artifacts

| File | Purpose |
|------|---------|
| `schema.json` | Schema for `QA_ALGEBRA_BRIDGE_CERT.v1` |
| `validator.py` | Five-gate validator |
| `mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `fixtures/valid_min.json` | PASS fixture |
| `fixtures/invalid_bad_word_probe.json` | FAIL fixture (`WORD_PROBE_STATE_MISMATCH`) |

## CLI

```bash
python qa_algebra_bridge_cert_v1/validator.py --self-test
python qa_algebra_bridge_cert_v1/validator.py qa_algebra_bridge_cert_v1/fixtures/valid_min.json
python qa_algebra_bridge_cert_v1/validator.py qa_algebra_bridge_cert_v1/fixtures/invalid_bad_word_probe.json
```
