# QA Whittaker Wave-Kernel Bridge Cert Draft

**Status**: Design draft only. No cert family ID assigned. Do not register or build from this draft until hostile review and Will sign-off.

**Proposed slug**: `qa_whittaker_wave_kernel_bridge_cert_v1`

**Layer position**: Layer 3 of the Whittaker -> QA development ladder.

Dependencies:

- `[266]` Layer 1: QA Whittaker Rational Direction `S1` Cert.
- `[273]` Layer 2: QA Whittaker Rational Direction `S2` Cert.

**Primary source anchor**: E. T. Whittaker, "On the partial differential equations of mathematical physics," *Math. Annalen* **57**:333-355, 1903. DOI 10.1007/BF01444290.

**Purpose**: Design a source-inspired scalar angular-kernel sampling cert over the exact rational `S2` direction substrate certified by `[273]`.

This draft does **not** assign a cert ID, does **not** create a validator, does **not** create fixtures, and does **not** modify the registry.

---

## 1. Why Layer 3 Exists

Layer 1 certified exact rational directions on `S1`.

Layer 2 certified exact rational directions on `S2`:

```text
D_m^(2) = { S(r,s) : r in R_m, s in R_m }
```

Layer 3 should be the first bridge from exact direction geometry into a wave-kernel setting. The bridge must be narrow: it should test deterministic sampling and finite scalar averaging over the certified `S2` substrate, not claim physics.

Whittaker motivates angular superposition; `[273]` supplies the rational direction substrate; Layer 3 v1 certifies only exact finite scalar sampling over that substrate.

---

## 2. What "Wave-Kernel Bridge" Means In v1

In v1, a wave-kernel bridge means:

```text
given an observer-side scalar angular profile h(omega) on a declared S2 chart,
sample h on D_m^(2) from [273],
construct a discrete angular superposition over those sampled directions,
and validate deterministic provenance plus exact recomputation of the declared finite scalar average.
```

Here `omega` denotes a direction on `S2`. The certified input directions are not arbitrary floats; they are exact rational packets from `[273]`:

```text
omega_i = (x_i/den_i, y_i/den_i, z_i/den_i)
```

The values `h(omega_i)` are observer-side measurements/evaluations. The cert must distinguish:

```text
QA substrate: exact rational directions and provenance from [273]
Observer layer: scalar profile evaluation, weights, optional display diagnostics
```

---

## 3. Recommended v1 Scope

Layer 3 v1 should use toy scalar angular profiles only.

Allowed v1 profile families:

```text
h_const(omega) = 1
h_z(omega)     = z
h_z2(omega)    = z*z
```

These profiles are useful because:

- they are simple enough to compute deterministically from rational packets;
- they test whether the `[273]` direction substrate is being consumed correctly;
- they avoid prematurely claiming Whittaker's full phase construction;
- they can produce exact rational sampled values before observer-side reporting.

v1 should not use empirical physical data.

v1 should not use Maxwell fields, electric fields, magnetic fields, scalar potentials, or gauge language.

---

## 4. Deferred Whittaker-Form Phase Packets

Whittaker 1903 is the source motivation for studying angular superposition over directions; v1 does not certify Whittaker's kernel.

Deferred candidate form:

```text
K(x,t; omega, k) = A(omega,k) * exp(i*k*(omega dot x - v*t))
```

or real-valued projections:

```text
cos(k*(omega dot x - v*t))
sin(k*(omega dot x - v*t))
```

These are observer-side analytic kernels until a later cert defines:

- exact domain;
- phase convention;
- sampling measure;
- weighting convention;
- allowable error metric;
- source-grounded relationship to Whittaker 1903.

Layer 3 v1 may mention these as future bridge targets, but it should not validate them.

---

## 5. Proposed Construction

Inputs:

```text
m in {3,5,9}
D_m^(2) from [273]
profile_name in {const, z, z2}
weight_rule
```

Direction source:

```text
read or recompute [273] D_m^(2)
preserve packet provenance:
  x_num
  y_num
  z_num
  den
  m
  chart = inverse_stereographic_excluding_south_pole
```

Sample values:

```text
h_i = h(omega_i)
```

For polynomial profiles, `h_i` can remain a `Fraction`.

Discrete uniform average:

```text
discrete_uniform_average[h] = sum_i w_i * h_i
```

where the v1 `weight_rule` should be simple and explicit. Candidate choices:

```text
uniform_points: w_i = 1 / |D_m^(2)|
z_band_balanced: future; not v1 unless separately justified
area_weighted: deferred until a spherical Voronoi or chart-Jacobian rule is defined
```

Recommended v1:

```text
weight_rule = uniform_points
```

This is not a claim of spherical area quadrature. It is a deterministic finite-sample averaging rule.

---

## 6. Candidate Error Notions

Layer 3 v1 should separate exact checks from observer-side checks.

Exact checks:

```text
direction provenance from [273]
sample count matches |D_m^(2)|
all polynomial profile samples are exact Fractions
uniform weights sum to 1 exactly
superposition value is exactly recomputed
```

Observer-side checks:

```text
observer_float_discrete_uniform_average
optional non-Whittaker, non-physical reference display
optional convergence trend across m={3,5,9}
```

For the first build, the strongest exact target is not convergence. It is exact recomputation:

```text
validator recomputes `discrete_uniform_average[h]` exactly and matches the declared rational value.
```

If a reference comparison is included, it must be framed carefully:

```text
uniform_points average over D_m^(2) is a finite-set diagnostic only.
It is not a Whittaker-kernel error, not a physical error, and not a
spherical quadrature error unless a separate quadrature theorem is added.
```

---

## 7. Read-only Scalar Average Baseline

A read-only enumeration was run against the registered `[273]` `S2`
construction for `m={3,5,9}`. The values below were recomputed from `[273]`
rational packets, not copied from draft tables.

```text
weight_rule = uniform_points
profiles    = {const, z, z2}
```

Observer-side decimal display:

| m | profile | unique_S2_direction_count | discrete_uniform_average |
|---:|---|---:|---:|
| 3 | const | 100 | 1 |
| 3 | z | 100 | 0.0441973802690126 |
| 3 | z2 | 100 | 0.0539057319420647 |
| 5 | const | 676 | 1 |
| 5 | z | 676 | 0.0513573393223701 |
| 5 | z2 | 676 | 0.0646401367652083 |
| 9 | const | 5476 | 1 |
| 9 | z | 5476 | 0.0549531745155562 |
| 9 | z2 | 5476 | 0.0704289153271853 |

Interpretation:

- `const` averages to exactly `1` for all three moduli.
- `z` averages are positive, not zero. The positive
  inverse-stereographic chart is biased and should not be described as
  full-sphere symmetric.
- `z2` gives a useful nontrivial stress target, but exact rational payloads
  grow quickly at `m=9`.
- These are finite-set averages only. They are not spherical quadrature,
  convergence, density, Whittaker-kernel, or physics claims.

Exact-witness policy:

- For small exact fractions, fixtures may store numerator and denominator
  directly.
- For huge exact fractions, fixtures should store exact recomputation
  witnesses:

```text
profile_name
m
unique_S2_direction_count
numerator_digit_count
denominator_digit_count
canonical_fraction_sha256
observer_float_display
```

The validator must recompute the exact `Fraction`, form the canonical ASCII
string:

```text
num/den
```

and hash that canonical string. Observer floats are display only and must not
be used for pass/fail.

Read-only exact witness fingerprints:

| m | profile | numerator digits | denominator digits | canonical_fraction_sha256_16 |
|---:|---|---:|---:|---|
| 3 | z | 201 | 202 | `2bc60820e263e34a` |
| 3 | z2 | 400 | 401 | `dbd910019ccb9c8f` |
| 5 | z | 1616 | 1617 | `59f1778ee5019937` |
| 5 | z2 | 3232 | 3233 | `9d7c14f745654f09` |
| 9 | z | 15378 | 15379 | `72c8e9d33e2c01a1` |
| 9 | z2 | 30755 | 30756 | `8d0c00cb98c2478f` |

Fixture implication:

- `m=3 const` is a simple sanity PASS fixture.
- `m=5 z` is a useful chart-bias PASS fixture because the average is
  nonzero.
- `m=9 z2` is a useful stress fixture only if v1 intentionally supports
  hash-style exact witnessing for large rationals.

---

## 8. Proposed v1 Gates

### WKB_1 — Dependency Provenance

The fixture must declare dependency on `[273]`:

```text
dependency_family_id = 273
dependency_slug = qa_whittaker_rational_direction_s2_cert_v1
m in {3,5,9}
chart = inverse_stereographic_excluding_south_pole
```

The validator must recompute the `[273]` direction set locally or call the `[273]` construction logic in a controlled way.

### WKB_2 — Profile Declaration

The fixture must declare one allowed v1 scalar profile:

```text
const
z
z2
```

No arbitrary Python expressions. No imported symbolic code. No empirical data.

### WKB_3 — Exact Sample Evaluation

For every generated direction packet:

```text
omega_i = (x_i/den_i, y_i/den_i, z_i/den_i)
h_i = h(omega_i)
```

The validator recomputes every `h_i` as a `Fraction`.

### WKB_4 — Exact Uniform Superposition

For:

```text
w_i = 1 / |D_m^(2)|
discrete_uniform_average[h] = sum_i w_i*h_i
```

the validator recomputes the exact rational `discrete_uniform_average[h]` and compares it to the declared fixture value.

### WKB_5 — Guarded Observer Reporting

If a fixture includes decimal approximations, they are observer-side only:

```text
observer_float_discrete_uniform_average
observer_reference_diagnostic
```

They must not be used as the primary pass/fail basis in v1. Any reference diagnostic must be explicitly non-Whittaker and non-physical. Do not call it quadrature error.

---

## 9. Initial Fixture Plan

Do not create these until build authorization.

PASS candidates:

```text
fixtures/pass_wkb_m3_const_uniform.json
fixtures/pass_wkb_m5_z_uniform.json
fixtures/pass_wkb_m9_z2_uniform.json
```

The `m=9 z2` fixture should use the exact-witness policy above unless the
build intentionally accepts very large numerator/denominator fields.

FAIL candidates:

```text
fixtures/fail_wkb_bad_dependency_id.json
fixtures/fail_wkb_unknown_profile.json
fixtures/fail_wkb_wrong_sample_count.json
fixtures/fail_wkb_wrong_superposition.json
fixtures/fail_wkb_overclaimed_spherical_integral.json
```

The last FAIL fixture is important: it should reject language claiming that `uniform_points` is a certified spherical area quadrature.

---

## 10. Non-Claims

Layer 3 v1 does not claim:

- Whittaker 1903 is proved by QA.
- Whittaker's full wave-kernel expansion has been certified.
- Whittaker 1904 two-scalar-potential theory has been reached.
- Maxwell equations, electromagnetism, gauge theory, or scalar-potential physics follow.
- `uniform_points` is a spherical area quadrature.
- The `S2` direction sets are dense or equidistributed.
- Any physical field has been reconstructed.
- Geodesy or ellipsoid physics follows from Whittaker.

Layer 3 v1 claims only deterministic scalar-profile sampling and exact finite scalar averaging over the `[273]` direction substrate.

---

## 11. Open Design Decisions Before Build

1. Whether v1 should import `[273]` validator logic or duplicate the minimal `[273]` construction locally.
2. Whether `m={3,5,9}` is sufficient for v1 or whether `m=3` and `m=5` are enough for the first build.
3. Whether observer-side reference diagnostics should be omitted entirely in v1.
4. Whether fixture fields should use `discrete_uniform_average` or a shorter `uniform_average` naming.
5. Whether large exact averages should be declared by numerator/denominator
   fields or by `canonical_fraction_sha256` plus digit-count witnesses.
6. The next free cert-family ID at build time.

---

## 12. Recommended Build Boundary

Recommended Layer 3 v1:

```text
exact scalar polynomial profile sampling
uniform finite-set superposition
dependency on [273]
no Whittaker phase packets
no spherical quadrature theorem
no physics
```

Recommended later Layer 3 v1.1:

```text
observer-side Whittaker-form phase packet experiments
declared phase convention
declared finite sampling rule
hostile review before any physical interpretation
```

This preserves the ladder:

```text
[266] S1 exact rational directions
[273] S2 exact rational directions
Layer 3 v1 scalar kernel sampling over S2
Layer 3 v1.1 Whittaker-form phase packet bridge
Layer 4 Whittaker 1904 scalar-potential bridge
Layer 5 Maxwell/scalar-pair reconstruction
```
