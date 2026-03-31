# QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1

This certificate family validates **packet decompositions of central idempotents** in the center algebra
`Z(Q[PSL2(Z/72Z)])`, strictly **relative to** a certified backend.

## Hard dependency

Every certificate in this family must reference a valid:

- `QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1` certificate, with `profile == "full_backend"`.

Certificates referencing `profile == "unit_law_only"` must be rejected (`DEPENDENCY_PROFILE_MISMATCH`).

## What this cert proves

Given a list of central elements `(e_1, ..., e_m)` represented as sparse rational coefficient vectors in the
`psl2_mod72_orbit_basis`, the validator verifies using exact backend multiplication:

- **Idempotence:** `e_i^2 = e_i` for all `i`.
- **Orthogonality:** `e_i e_j = 0` for all `i ≠ j`.
- **Completeness:** `Σ_i e_i = 1` (algebra identity).

Optionally, a certificate can include a **target resolution** claim:

- `z = Σ_i λ_i e_i` (exact equality), where `λ_i` are exact rationals.

## What this cert does *not* prove

This family does not certify:

- irreducible characters,
- character tables,
- primitivity of idempotents,
- spectral interpretations (trace formula, Laplacian eigenmodes, etc.).

Those require separate families and/or external mathematical inputs.

## Representation

- Basis: `psl2_mod72_orbit_basis` (length 375), as defined by `qa_psl2_mod72_center_algebra.py` and certified by the backend.
- Each element is a sparse list of triples: `[orbit_index, num, den]` representing `num/den` at that orbit coordinate.
- All arithmetic uses `fractions.Fraction` (exact).

