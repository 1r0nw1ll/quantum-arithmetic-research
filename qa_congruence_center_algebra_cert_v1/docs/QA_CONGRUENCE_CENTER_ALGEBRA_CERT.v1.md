# QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1

This certificate family validates the **internal CRT center-algebra backend** implemented in `qa_psl2_mod72_center_algebra.py`.

## Scope (what this cert is and is not)

This family certifies:

- the correctness and determinism of cached center-algebra data for:
  - `SL2(Z/8Z)` (order 384, 30 conjugacy classes),
  - `SL2(Z/9Z)` (order 648, 25 conjugacy classes),
  - and the induced orbit model of conjugacy classes for `PSL2(Z/72Z)` (375 sign-orbits of product conjugacy classes);
- exact class-sum structure constants in each factor;
- exact orbit-basis multiplication for the `PSL2(Z/72Z)` center algebra via the map:
  - `K̄_orbit := (1/2)(K_pair + K_negpair)` (lift),
  - multiply in the product center via factor structure constants,
  - project back to orbit coefficients.

This family does **not** certify:

- character tables,
- primitive central idempotent decompositions,
- “character packets” or any spectral interpretation.

Those belong to downstream families once this backend is certified.

## Data sources (caches)

The backend writes/reads deterministic JSON caches under `qa_center_algebra_cache/`:

- `sl2_mod8_center_cache.json`
- `sl2_mod9_center_cache.json`
- `psl2_mod72_orbit_cache.json`

This certificate locks the exact bytes of these caches via SHA-256.

## Validation strategy (gates)

`validator.py` implements:

- **Gate 0 (Provenance):** engine source hash and cache file hashes match the cert.
- **Gate 1 (Factor correctness):** recompute factor conjugacy partitions from scratch (mod 8 and mod 9) and compare to caches.
- **Gate 2 (Factor structure constants):** verify unit law and associativity on a deterministic battery using cached structure constants.
- **Gate 3 (Orbit correctness):** verify 375 sign-orbits, coverage of all product class pairs, no fixed sign-orbits, orbit sizes, and deterministic CRT representatives.
- **Gate 4 (Quotient center algebra):** verify unit law, commutativity, and associativity on a deterministic battery using orbit-basis multiplication.
- **Gate 5 (Fingerprint):** recompute a deterministic multiplication fingerprint and match the cert (for `pass_exact`).

## Certificate statuses

- `pass_exact`: all required gates passed.
- `fail_witness`: a single, explicit failure is certified with a recomputable witness (e.g. cache-hash mismatch; wrong claimed unit index).

## Profiles

This family supports a `profile` field to scope what “pass_exact” means:

- `full_backend`: runs the full factor/orbit/backend validation and requires an `algebra_fingerprint_sha256`.
- `unit_law_only`: verifies provenance + basic cache sanity + identity-orbit location + unit law on a deterministic battery.
