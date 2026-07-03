# QA Whittaker Layer 5/6 Claim Boundary

Status: build map, not a registered cert.

This document continues the Whittaker -> QA ladder after registered Layer 4
cert `[507]`. It turns the current prose mapping into concrete build
boundaries for the EM-reconstruction and Mie-scattering claims.

## Sources And Scope

Primary source already used by `[507]`:

- E. T. Whittaker (1904), "On an expression of the electromagnetic field due
  to electrons by means of two scalar potential functions," Proc. London Math.
  Soc. s2-1:367-372. DOI: `10.1112/plms/s2-1.1.367`.

Mie-scattering source anchors:

- Gustav Mie (1908), "Beitraege zur Optik trueber Medien, speziell
  kolloidaler Metalloesungen," Annalen der Physik, 25:377-445.
- Modern terminology check: Mie scattering is an electromagnetic scattering
  solution for a sphere expressed as spherical multipole partial waves; common
  modern formulations use vector spherical harmonics and spherical Bessel /
  Hankel radial functions.

Do not promote any Bromwich-specific claim to a cert until the Bromwich
primary source is located and quoted by page/equation. Until then, say
"spherical/Debye-potential mode lattice" or "Mie-mode lattice" rather than
"Bromwich proves X".

## Layer 5: Maxwell Scalar-Pair Reconstruction

Layer 5 is the first place where `[507]` is allowed to produce observer-side
field values. It consumes exact Layer 4 packet coefficients and then evaluates
formal trig labels at declared observer coordinates.

Allowed claim:

> Given a finite packet list whose directions are drawn from `[273]` and whose
> per-packet coefficients pass `[507]`, an observer-side evaluator can compute
> the six reconstructed field components at declared sample points. If a
> target smooth vacuum field is represented by Whittaker's angular integral
> with angular data satisfying explicit smoothness and band-limit assumptions,
> a finite QA direction net has an error bound with a second-derivative
> amplification factor.

Rejected claims:

- QA derives Maxwell equations.
- QA proves electromagnetism.
- QA reconstructs arbitrary physical fields.
- QA proves scalar-potential energy, longitudinal free energy, or any
  Bearden/Pond/SVP-style "structured scalar energy" claim.
- Layer 5 certifies Mie scattering.

### Layer 5 Inputs

A future validator should require:

- `source_attribution`: Whittaker 1904 DOI plus `[273]` and `[507]`.
- `observer_boundary`: `true`.
- `firewall_crossing_count`: exactly `2` for the Whittaker ladder.
- `packets`: finite list of `[507]`-compatible packets with exact rational
  coefficients or hashes of `[507]`-validated packet certs.
- `sample_points`: finite observer-side coordinates `(x,y,z,t)`.
- `trig_backend`: declared function family, e.g. `sin` / `cos`, used only
  after the QA exact packet algebra is complete.
- `error_bound_model`: explicit derivative order and band-limit assumptions.

### Layer 5 Bound Shape

Let `Delta_m` denote the angular covering radius inherited from `[273]`.
Layer 3 only sampled scalar angular data. Layer 5 differentiates twice through
Whittaker's operator, so any angular sampling bound must carry a factor that
scales like the square of the largest spatial frequency. The intended bound
shape is:

```text
component_sup_error <= C * L * K_max*K_max * Delta_m
```

where:

- `C` is an explicit constant proved or conservatively bounded in the cert.
- `L` is a declared angular Lipschitz or stronger smoothness constant for the
  Whittaker angular data.
- `K_max` is a declared finite band limit.
- `Delta_m` is supplied or recomputed from `[273]`.

Layer 5 is not build-ready until `C` and the exact smoothness norm are fixed.

### Layer 5 Negative Fixtures

Minimum fail cases for a future cert:

- Claims `claims_maxwell_derivation=true`.
- Claims `claims_electromagnetism=true`.
- Uses trig/float values before `[507]` coefficient algebra is validated.
- Omits `K_max` while declaring an error bound.
- Uses a first-derivative or zeroth-derivative bound for a second-derivative
  Whittaker component.
- Claims `firewall_crossing_count > 2` inside the same cert.

## Layer 6: Mie / Spherical Mode Boundary

Layer 6 is not a field-value cert. The QA-shaped part is the discrete
spherical mode index lattice. The special functions used to evaluate actual
Mie fields are observer-side analysis objects.

Allowed narrow claim:

> For a cutoff `N`, enumerate the finite spherical mode lattice
> `(n,m)` exactly, with `0 <= n <= N` and `-n <= m <= n` for scalar modes, or
> `1 <= n <= N` with TE/TM polarization labels for electromagnetic vector
> modes. Certify counts and indexing discipline only.

Rejected claims:

- Exact QA representation of `j_n(kr)`, `y_n(kr)`, Hankel functions, or
  associated Legendre field values.
- Mie scattering cross sections, efficiencies, resonances, or phase functions.
- Boundary-condition solving at a sphere.
- Any physical scattering prediction.
- Any scalar-wave-energy or longitudinal-energy claim.

### Exact Mode Counts

Scalar spherical-harmonic index lattice:

```text
L_scalar(N) = {(n,m): 0 <= n <= N, -n <= m <= n}
|L_scalar(N)| = sum_{n=0}^N (2*n + 1) = (N + 1)*(N + 1)
```

Electromagnetic vector mode lattice, excluding the scalar `n=0` mode and
splitting TE/TM polarization:

```text
L_em(N) = {(pol,n,m): pol in {TE,TM}, 1 <= n <= N, -n <= m <= n}
|L_em(N)| = 2 * sum_{n=1}^N (2*n + 1) = 2*N*(N + 2)
```

For an axially incident plane wave, a separate observer-side reduction may
select an `m=1` sector depending on convention and parity. That selection is
not the same claim as the full mode lattice and should be a separate fixture
or separate cert.

### Layer 6 Candidate Gates

A future mode-lattice cert can be exact and self-contained:

- `ML_1`: `N` is a non-negative integer; for EM vector lattice, `N >= 1`.
- `ML_2`: every scalar entry satisfies `0 <= n <= N` and `-n <= m <= n`.
- `ML_3`: every EM entry satisfies `pol in {TE,TM}`, `1 <= n <= N`, and
  `-n <= m <= n`.
- `ML_4`: scalar count equals `(N+1)*(N+1)`.
- `ML_5`: EM count equals `2*N*(N+2)`.
- `ML_6`: no field-value functions are present: reject `bessel_values`,
  `legendre_values`, `mie_coefficients`, `cross_sections`, `efficiencies`,
  `phase_function`, `boundary_conditions_solved`.
- `ML_7`: source attribution cites Mie 1908 and a modern vector-spherical-
  harmonic formulation, and explicitly says this cert does not compute Mie
  field values.

### Layer 6 Negative Fixtures

Minimum fail cases:

- Off-by-one scalar count: omits `n=0` while claiming scalar lattice.
- Off-by-one EM count: includes `n=0` in vector EM modes.
- Missing negative `m` values.
- Declares only `m=1` but claims full spherical mode lattice.
- Includes Bessel/Legendre/Mie coefficient numerical values.
- Claims scattering cross section or physical prediction.
- Claims scalar-wave-energy physics.

## Recommended Build Order

1. Do not register a Layer 6 mode-lattice cert until the user explicitly
   accepts that it is a combinatorial boundary cert only.
2. Build Layer 5 only after the derivative-amplified error-bound constant and
   smoothness norm are specified.
3. If a quick next cert is desired, build a narrow `qa_mie_mode_lattice_cert_v1`
   as a pure indexing cert. It should not be called a scattering cert.
4. Keep any numeric Mie calculator as an observer-projection tool outside the
   QA cert family, consuming the mode-lattice cert output but not feeding
   numerical values back into QA exact state.

## Claim Classifier

| Claim text pattern | Disposition |
| --- | --- |
| "Whittaker operators give exact Fraction packet coefficients" | Already certified in `[507]` |
| "Evaluate this finite packet at observer points" | Layer 5 observer boundary, allowed with two-crossing accounting |
| "This approximates a smooth Whittaker angular integral" | Layer 5 only with explicit `C * L * K_max*K_max * Delta_m` bound |
| "QA derives Maxwell" | Reject |
| "QA proves electromagnetism" | Reject |
| "Mie modes are indexed by `(n,m)` up to cutoff `N`" | Layer 6 mode-lattice cert candidate |
| "QA computes Mie scattering cross sections" | Reject inside cert; observer tool only |
| "Scalar potentials contain structured free energy" | Reject |

