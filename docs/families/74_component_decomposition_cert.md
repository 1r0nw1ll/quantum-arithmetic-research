# Family [74]: QA Component Decomposition Cert

## Purpose

Family [74] certifies gcd-based component decomposition for QA structural algebra states and validates a guarded contraction theorem for `nu`.

The family binds three layers into one cert:

- gcd normalization `(b,e) -> (g, (b/g, e/g))`,
- scaled-seed reconstruction by normal-form word application, and
- `nu` contraction behavior controlled by the 2-adic valuation `v2(g)`.

## Location

`qa_component_decomposition_cert_v1/`

## Schema

`QA_COMPONENT_DECOMPOSITION_CERT.v1`

## Gates

1. **Gate 1 — Schema shape / type checks**
   - includes bridge dependency: `subject.algebra_bridge_semantics_sha256` must match Family [75] anchor hash
   - optional soft bridge to Family [76]:
     - `subject.failure_algebra_anchor_ref`
     - `subject.failure_algebra_anchor_rollup_sha256`
     - if present, both are verified against `qa_failure_algebra_structure_cert_v1.failure_algebra_anchor`
2. **Gate 2 — Deterministic decomposition sample recomputation**
   - verifies `gcd`, normalized pair, normal-form word, and scaled-seed roundtrip
3. **Gate 3 — Deterministic `nu` characterization samples**
   - verifies `v2(g)`, post-`v2(g)` state, and next-step guard behavior
4. **Gate 4 — Bounded theorem sweep up to `N`**
   - recomputes decomposition and contraction properties for all states in `[1..N]^2`
5. **Gate 5 — Invariant diff map claim verification**
   - fixture `result.invariant_diff_map` is treated as a claim and must match recomputed failures + rollup
   - when the [76] soft bridge is present and loadable, validator emits non-gating audit field `failure_join_summary`
     with:
     - direct lens: join over known [76]-carrier failure tags + explicit unknown tags + known/unknown counts
     - projected lens: conservative mapping from family-local fail types to [76]-carrier tags
       (`projected_fail_types`, `projection_unknown_fail_types`, `joined_fail_type_projected`, `projection_complete`,
       `projection_mapped_count`, `projection_unmapped_count`, `projection_coverage`,
       `projection_map_name`, `projection_map_version`,
       `projection_selector_reason`, `projection_selector_hits`, `failure_signature_v1`,
       `failure_signature_v1_sha256`)

## Certified Property

For each state `(b,e)`:

- Let `g = gcd(b,e)` and normalized state `(b',e') = (b/g, e/g)`.
- There exists a word `w` over `{L,R}` such that applying `w` to seed `(g,g)` reconstructs `(b,e)`.
- Let `t = v2(g)`.
  - After exactly `t` applications of `nu`, the state reaches normalized `(b',e')` iff `g` is a power of two.

## Encoding Convention

- Word token `L` means apply `sigma`.
- Word token `R` means apply generator `R := mu sigma mu`.
- Words are applied left-to-right to the seed state.

## Artifacts

| File | Purpose |
|------|---------|
| `schema.json` | Schema for `QA_COMPONENT_DECOMPOSITION_CERT.v1` |
| `validator.py` | Five-gate validator |
| `mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `fixtures/valid_min.json` | PASS fixture |
| `fixtures/invalid_bad_decomposition_word.json` | FAIL fixture (`DECOMP_WORD_MISMATCH`) |
| `fixtures/invalid_bridge_hash_mismatch.json` | FAIL fixture (`BRIDGE_SEMANTICS_HASH_MISMATCH`) |

## CLI

```bash
python qa_component_decomposition_cert_v1/validator.py --self-test
python qa_component_decomposition_cert_v1/validator.py qa_component_decomposition_cert_v1/fixtures/valid_min.json
python qa_component_decomposition_cert_v1/validator.py qa_component_decomposition_cert_v1/fixtures/invalid_bad_decomposition_word.json
python qa_component_decomposition_cert_v1/validator.py qa_component_decomposition_cert_v1/fixtures/invalid_bridge_hash_mismatch.json
```
