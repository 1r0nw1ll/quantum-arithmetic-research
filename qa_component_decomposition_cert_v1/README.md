# QA Component Decomposition Cert v1

Machine-tract family for certified gcd-based component decomposition and guarded contraction behavior.

## Schema

`QA_COMPONENT_DECOMPOSITION_CERT.v1`

Subject must include:
- `algebra_bridge_semantics_sha256` (must equal `qa_algebra_bridge_cert_v1.semantics_anchor.BRIDGE_SEMANTICS_SHA256`)

Subject may optionally include:
- `failure_algebra_anchor_ref` (if present, must equal `qa_failure_algebra_structure_cert_v1.failure_algebra_anchor.FAILURE_ALGEBRA_ANCHOR_REF`)
- `failure_algebra_anchor_rollup_sha256` (if present, must equal `qa_failure_algebra_structure_cert_v1.failure_algebra_anchor.FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256`)

## Gates

1. Schema shape / types
2. Deterministic decomposition sample recomputation
3. Deterministic nu-power characterization sample checks
4. Bounded theorem sweep up to `N`
5. Invariant diff map claim verification against recomputed failures (`INVARIANT_DIFF_MAP_CLAIM_MISMATCH` on mismatch)

## Encoding Convention

- Word token `L` means apply `sigma`.
- Word token `R` means apply generator `R := mu sigma mu`.
- Words are applied left-to-right to the seed state.

## Certified Component Property

For each state `(b,e)`:

- `g = gcd(b,e)` and `(b',e') = (b/g, e/g)`.
- `state_to_word((b',e')) = w` exists and is unique.
- Applying `w` to scaled seed `(g,g)` reconstructs `(b,e)`.
- Let `t = v2(g)`. After `t` applications of `nu`, the state reaches normalized `(b',e')` iff `g` is a power of two.

## Fixtures

- `fixtures/valid_min.json` (PASS)
- `fixtures/invalid_bad_decomposition_word.json` (FAIL)
- `fixtures/invalid_bridge_hash_mismatch.json` (FAIL)

`result.invariant_diff_map` is treated as a claim. The validator recomputes failures from gates, checks claim equality (`entries` + rollup), and emits authoritative output from recomputed failures.

If `subject.failure_algebra_anchor_ref` is present and loadable, validator output also includes
`failure_join_summary` as a non-blocking audit field:
- direct lens: known/unknown [76]-carrier tags + joined failure + counts
- projected lens: `projected_fail_types`, `projection_unknown_fail_types`,
  `joined_fail_type_projected`, `projection_complete`,
  `projection_mapped_count`, `projection_unmapped_count`, `projection_coverage` (`{mapped,total}`),
  `projection_map_name`, `projection_map_version`,
  `projection_selector_reason`, `projection_selector_hits`,
  `failure_signature_v1` (`<map_version>:<map_name>:<joined>:<complete>:<mapped>/<total>`),
  and `failure_signature_v1_sha256` (sha256 of `failure_signature_v1`)

## Run

```bash
python qa_component_decomposition_cert_v1/validator.py --self-test
python qa_component_decomposition_cert_v1/validator.py qa_component_decomposition_cert_v1/fixtures/valid_min.json
python qa_component_decomposition_cert_v1/validator.py qa_component_decomposition_cert_v1/fixtures/invalid_bridge_hash_mismatch.json
```

`--self-test` includes a deterministic non-gating probe that demonstrates partial projected coverage (`projection_complete=false`) without fixture changes.
