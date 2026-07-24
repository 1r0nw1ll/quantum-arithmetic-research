# Family [533]: QA Orbit Satellite Ramification Cert

**Schema:** `QA_ORBIT_SATELLITE_RAMIFICATION_CERT.v1`
**Root:** `qa_alphageometry_ptolemy/qa_orbit_satellite_ramification_cert_v1/`
**Validator:** `qa_orbit_satellite_ramification_cert_validate.py --self-test`
**Status:** Active; self-test passing; added to `FAMILY_SWEEPS`

## Purpose

`qa_orbit_rules.py` (the canonical, single-source-of-truth orbit classifier
used throughout the project) computes `orbit_family` by brute-force
simulation because no closed-form characterization of the period-8
("satellite") class was known. It also ships a fast algebraic shortcut,
`(m//3)|b ∧ (m//3)|e`, that the module's own self-test documents as
under-counting satellites by exactly 32 whenever `5 | m` — without deriving
why.

This cert closes that gap. `qa_step(b,e,m)` is conjugate to the Fibonacci
matrix `M=[[0,1],[1,1]]` acting on `(Z/mZ)^2`, with `0` relabeled `m` (all
A1's no-zero rule is doing). `M`'s characteristic polynomial `x^2-x-1` has
discriminant 5:

- **Mod 3**: discriminant `5 ≡ 2` is a non-residue, so `x^2-x-1` is
  irreducible over `F_3`. `M`'s eigenvalues live in `F_9` with order exactly
  8, so `M` has order 8 in `GL_2(F_3)` and fixes only the zero vector —
  every nonzero vector of `(Z/3Z)^2` has orbit period exactly 8. This is the
  origin of the satellite class, and exactly what the divisor shortcut's
  `(m//3)|b,e` condition detects (the subgroup of `(Z/mZ)^2` that reduces to
  `(Z/3Z)^2`).
- **Mod 5**: discriminant `5 ≡ 0`, so `x^2-x-1` has a *repeated* root
  (`λ=3`, the same ramification as "5 ramifies `Z[φ]`" elsewhere in this
  project). `M` mod 5 is a non-diagonalizable Jordan block: its 1-dimensional
  eigenspace has 4 nonzero vectors of period exactly 4 (pure scalar
  multiplication by 3, `ord(3)=4` in `(Z/5Z)^*`); every other nonzero vector
  gets the full Jordan-block order, 20 (`=` Pisano period `π(5)`).
- **CRT composition**: since `4 | 8`, an `(b,e)` that is generic mod 3
  (local period 8) *and* lands in the mod-5 eigenspace (local period 4) has
  combined period `lcm(8,4)=8` — a genuine satellite the divisor shortcut
  misses entirely, because the shortcut only looks for a trivial mod-5 part.
  Count: `8 × 4 = 32`, exactly, matching the module's own documented gap.

## Scope and honesty boundary

The mod-3 and mod-5 (prime-level) facts above are proven for **any** modulus
by the eigenvalue/Jordan-block argument. Their persistence at higher prime
powers (9, 25 — i.e. `m ∈ {45, 75}`) is **verified computationally here**,
not re-derived from a general p-adic/Hensel-lifting argument for arbitrary
exponents. The cert's `non_claims` and overclaim checks enforce this
boundary explicitly (see `HENSEL_GENERALIZATION_OVERCLAIM`).

This cert explains and is diagnostic of the existing canonical
`orbit_family`/`orbit_period` functions — it does not replace the
simulation-based classifier as the source of truth.

## Schema

| Field | Meaning |
|---|---|
| `schema_version` | Must be `QA_ORBIT_SATELLITE_RAMIFICATION_CERT.v1` |
| `cert_type` | Must be `qa_orbit_satellite_ramification_cert` |
| `theorem_status` | Must be `PROVEN_BY_DISCRIMINANT_5_RAMIFICATION` |
| `theorem_statement` | Declares the ramification argument; forbids empirical-orbit-lift, general-Hensel, and classifier-replacement overclaims |
| `proof_obligations` | Fibonacci conjugacy, mod-3 irreducibility/period-8, mod-5 Jordan-eigenspace/period-4, CRT composition, bounded-audit, known-moduli exactness |
| `qa_step_fibonacci_conjugacy_samples` | Sampled `(b,e,m)` rows checked against the matrix-conjugacy identity |
| `mod3_witnesses` / `mod5_witnesses` | Declared periods, cross-checked against a full recomputation over `(Z/3Z)^2` / `(Z/5Z)^2` |
| `crt_examples` | Illustrative shortcut-missed satellites, each independently re-derived from its 3-part/5-part periods |
| `bounded_audit` | Full recomputation of `satellite_count`/`shortcut_miss_count` for `m ∈ {15,30,45,60,75}`, plus per-miss CRT-mechanism verification |
| `known_moduli_check` | Recomputed 0-miss exactness for `m ∈ {9,24}` (no factor of 5) |
| `non_claims` | Must include factorization-shortcut, general-Hensel, and classifier-replacement exclusions |
| `result` | `PASS` or `FAIL` |

## Fixtures

| Fixture | Expected | Purpose |
|---|---|---|
| `orbit_pass_ramification.json` | PASS | Valid certificate: full mod-3/mod-5 grids, five-modulus bounded audit (all `32` misses), and known-moduli exactness |
| `orbit_fail_bad_witness.json` | FAIL | Detects a declared mod-3 period that does not recompute |
| `orbit_fail_hensel_overclaim.json` | FAIL | Rejects claiming a general p-adic/Hensel-lifting proof beyond what was computationally verified |

## Family Relationships

- Explains the empirical gap documented in `qa_orbit_rules.py`'s own
  docstring and `self_test()` (the hard-coded `misses == 32` assertion for
  `m ∈ {15, 30}`).
- Not a companion to the mining-derived certs [529]–[532]: those certify
  properties of QA-derived scalars (`D+F`, `G`, `F`); this one certifies the
  orbit classifier itself, shared infrastructure every cert that reasons
  about Cosmos/Satellite/Singularity depends on.
- Connects to the quaternion-grounding work's finding that "5 ramifies
  `Z[φ]`" — this is that same ramification fact surfacing as the source of
  the divisor-shortcut's blind spot.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_orbit_satellite_ramification_cert_v1/qa_orbit_satellite_ramification_cert_validate.py --self-test
```

Expected result: `{"ok":true,...}` with one passing fixture and two correctly rejected fail fixtures.
