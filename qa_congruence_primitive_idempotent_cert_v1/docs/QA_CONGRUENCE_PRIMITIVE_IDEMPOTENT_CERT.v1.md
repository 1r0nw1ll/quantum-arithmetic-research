# QA_CONGRUENCE_PRIMITIVE_IDEMPOTENT_CERT.v1

This certificate family validates **primitivity** of central idempotents in the center algebra
`Z(Q[PSL2(Z/72Z)])`, strictly **relative to** a certified backend and a declared finite refinement scope.

## Hard dependency

Every certificate must reference a valid:

- `QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1` certificate with `profile == "full_backend"`.

Certificates referencing `profile == "unit_law_only"` must be rejected.

## What this cert proves (v1)

### PASS (`pass_exact`)

Certifies that the provided idempotent `e` is **primitive relative to the declared finite refinement scope**:

- `e^2 = e`
- no nontrivial refinement is found within the finite search space `A` for `a`, where a refinement means:
  - `a^2 = a`
  - `a ≠ 0` and `a ≠ e`
  - `a e = a` (so `a` is a nontrivial sub-idempotent of `e`)
  - with `b := e - a` then automatically `b^2=b` and `ab=0` in a commutative algebra

This is an **honest in-scope primitivity** claim, not a global Wedderburn/character-theoretic primitivity claim.

### FAIL (`fail_witness`)

Certifies explicit non-primitivity by exhibiting a refinement witness:

- `e = a + b`
- `a^2 = a`, `b^2 = b`
- `ab = ba = 0`
- `a ≠ 0`, `b ≠ 0`

All verified exactly via the backend multiplication.

## Representation

- Basis: `psl2_mod72_orbit_basis` (length 375), as defined by `qa_psl2_mod72_center_algebra.py` and certified by the backend.
- Elements `e`, `a`, `b` are sparse lists of triples: `[orbit_index, num, den]` representing `num/den` at that coordinate.
- All arithmetic is exact (`fractions.Fraction`).

