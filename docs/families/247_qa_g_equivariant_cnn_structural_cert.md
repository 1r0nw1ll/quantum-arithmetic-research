# [247] QA G-Equivariant CNN Structural Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-14
**Primary source:** Cohen and Welling (2016), arxiv.org/abs/1602.07576
**Source:** Will Dale + Claude

## Claim

This family certifies the closed-form algebra behind the Cohen-Welling
group-equivariant CNN rotation index.

- `phi(b) = b mod n` is a bijection from `{1,...,n}` to `Z/nZ`.
- `qa_step(b1,b2,n) = ((b1 + b2 - 1) mod n) + 1` preserves addition
  under `phi`.
- At `n=9`, the single-generator orbit partition over all `81` pairs
  splits into `singularity`, `satellite`, and `cosmos` with no
  exceptions.
- Eq. 10, Eq. 11, and §6.3 correspond structurally to observer IN,
  QA-layer resonance, and observer OUT.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_g_equivariant_cnn_structural_cert_v1/qa_g_equivariant_cnn_structural_cert_v1.py`
- Fixtures:
  - `gecs_pass_n9_full.json` (PASS)
  - `gecs_pass_n24_full.json` (PASS)
  - `gecs_fail_wrong_bijection.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Checks

| Check | Meaning |
|-------|---------|
| SCHEMA | schema version matches |
| C1 | `phi` is a bijection with explicit inverse |
| C2 | composition preservation holds exhaustively |
| C3 | `n=9` orbit partition has family counts 9/18/54 with zero exceptions |
| C4 | equation correspondence table matches the structural claim |
| SRC | source attribution mentions Cohen and Welling |
| F | fail ledger is a list |

## Cross-references

- Theory: `docs/theory/QA_GROUP_EQUIVARIANT_CNN_MAPPING.md`
- Methodology: `docs/theory/QA_CV_METHODOLOGY_MAPPING.md`
- Template family: `docs/families/232_qa_uhg_diagonal_coincidence_cert.md`

## Notes

- This is a structural cert only. It does not train or benchmark a CNN.
- C3 is intentionally n=9-specific; the `n=24` fixture skips it.
