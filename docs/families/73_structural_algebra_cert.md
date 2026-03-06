# Family [73]: QA Structural Algebra Cert

## Purpose

Family [73] certifies bounded structural properties of the QA structural algebra kernel:

- unique LR normal forms on coprime states,
- deterministic bounded uniqueness audit,
- scaling component classification via gcd normalization,
- guarded contraction behavior for `nu`.

## Location

`qa_structural_algebra_cert_v1/`

## Schema

`QA_STRUCTURAL_ALGEBRA_CERT.v1`

## Gates

1. **Gate 1 — Schema shape / type checks**
   - includes bridge dependency: `subject.algebra_bridge_semantics_sha256` must match Family [75] anchor hash
   - optional soft bridge to Family [76]:
     - `subject.failure_algebra_anchor_ref`
     - `subject.failure_algebra_anchor_rollup_sha256`
     - if present, both are verified against `qa_failure_algebra_structure_cert_v1.failure_algebra_anchor`
2. **Gate 2 — Deterministic sample recomputation**
   - roundtrip samples (`state_to_word`, `word_to_state`)
   - guarded contraction samples (`nu`)
3. **Gate 3 — Bounded uniqueness + normal-form audit**
   - full coprime audit up to bound `N`
4. **Gate 4 — Scaling component checks vs gcd**
   - `component_gcd`, `normalize_to_coprime`, `state_to_word_with_scale`
   - scaled reachability completeness with `allow_scaling=True`
5. **Gate 5 — Invariant diff map claim verification**
   - validator recomputes gate failures as the authoritative `invariant_diff_map.entries`
   - fixture-provided `result.invariant_diff_map` is treated as a claim and must match recomputed entries + rollup
   - mismatch fails with `INVARIANT_DIFF_MAP_CLAIM_MISMATCH`
   - when the [76] soft bridge is present and loadable, validator emits non-gating audit field `failure_join_summary`
     with:
     - direct lens: join over known [76]-carrier failure tags + explicit unknown tags + known/unknown counts
     - projected lens: conservative mapping from family-local fail types to [76]-carrier tags
       (`projected_fail_types`, `projection_unknown_fail_types`, `joined_fail_type_projected`, `projection_complete`,
       `projection_mapped_count`, `projection_unmapped_count`, `projection_coverage`,
       `projection_map_name`, `projection_map_version`,
       `projection_selector_reason`, `projection_selector_hits`, `failure_signature_v1`,
       `failure_signature_v1_sha256`)

## Artifacts

| File | Purpose |
|------|---------|
| `schema.json` | Schema for `QA_STRUCTURAL_ALGEBRA_CERT.v1` |
| `validator.py` | Five-gate validator |
| `mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `fixtures/valid_min.json` | PASS fixture |
| `fixtures/invalid_bad_expected_word.json` | FAIL fixture (`ROUNDTRIP_WORD_MISMATCH`) |
| `fixtures/invalid_bridge_hash_mismatch.json` | FAIL fixture (`BRIDGE_SEMANTICS_HASH_MISMATCH`) |

The FAIL fixture keeps a valid Gate 5 claim. Its intended failure is Gate 2 only (`ROUNDTRIP_WORD_MISMATCH`).

## Encoding Convention

- Word token `L` means apply `sigma`.
- Word token `R` means apply generator `R := mu sigma mu`.
- Words are applied left-to-right to the seed state.

## CLI

```bash
python qa_structural_algebra_cert_v1/validator.py --self-test
python qa_structural_algebra_cert_v1/validator.py qa_structural_algebra_cert_v1/fixtures/valid_min.json
python qa_structural_algebra_cert_v1/validator.py qa_structural_algebra_cert_v1/fixtures/invalid_bad_expected_word.json
python qa_structural_algebra_cert_v1/validator.py qa_structural_algebra_cert_v1/fixtures/invalid_bridge_hash_mismatch.json
```
