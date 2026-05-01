# QA Whittaker Rational Direction `S^2` Cert Draft

**Status**: Design draft only. No cert family ID assigned. Do not register or build from this draft until hostile review and Will sign-off.

**Layer position**: Layer 2 of the Whittaker -> QA development ladder. Layer 1 is registered as cert family `[266]` in `qa_whittaker_rational_direction_s1_cert_v1`.

**Primary source anchor**: E. T. Whittaker, "On the partial differential equations of mathematical physics," *Math. Annalen* **57**:333-355, 1903. DOI 10.1007/BF01444290.

**Purpose**: Design a geometry-first, exact-arithmetic rational direction set on `S^2` suitable as the base layer for a later Whittaker 1903 wave-kernel approximation cert.

This draft does **not** assign a cert ID, does **not** create a validator, and does **not** modify the registry.

---

## 1. Why Layer 2 Is Geometry-First

Whittaker 1903 uses a full direction on the sphere, commonly written in angular form as:

```text
(sin u cos v, sin u sin v, cos u)
```

Layer 1 certified only a rational `S^1` direction net. Layer 2 must therefore certify a rational direction set on `S^2` before any honest wave-kernel bridge is attempted.

The Layer 2 target is not physics. It is the exact finite geometry needed before physics-adjacent approximation claims can be stated.

---

## 2. Primary Construction: Rational `S^2` Map

Use the standard rational parameterization of the unit sphere:

```text
S(r, s) = (
  2r / (1 + r*r + s*s),
  2s / (1 + r*r + s*s),
  (1 - r*r - s*s) / (1 + r*r + s*s)
)
```

For integer parameters `(P, R, Q)` representing `r = P/Q` and `s = R/Q`, use the exact integer form:

```text
N = Q*Q + P*P + R*R
X = 2*P*Q / N
Y = 2*R*Q / N
Z = (Q*Q - P*P - R*R) / N
```

Exact identity:

```text
X*X + Y*Y + Z*Z = 1
```

Validator form should avoid floating state by storing each point as an integer numerator triple over a shared denominator:

```text
(x_num, y_num, z_num, den) =
(2*P*Q, 2*R*Q, Q*Q - P*P - R*R, N)
```

Then check:

```text
x_num*x_num + y_num*y_num + z_num*z_num == den*den
```

No `**2`; use `x*x`.

---

## 3. QA Parameter Source

Layer 2 should source rational parameters from the already-certified Layer 1 QA-rational coordinates.

For each coprime QA seed `(b, e)`:

```text
d = b + e
a = b + 2*e
C = 2*d*e
F = a*b
G = d*d + e*e
```

Layer 1 gives rational coordinates:

```text
C/G, F/G
```

Define the rational parameter pool:

```text
R_m = { C/G, F/G : (b,e) in {1..m}^2, gcd(b,e)=1 }
```

Then define the Layer 2 direction set:

```text
D_m^(2) = { S(r, s) : r in R_m, s in R_m }
```

Implementation detail: if `r = P_r/Q_r` and `s = P_s/Q_s`, convert to a shared integer chart by choosing:

```text
Q = Q_r * Q_s
P = P_r * Q_s
R = P_s * Q_r
```

Then apply the integer `S^2` map above.

This keeps the construction exact and separates the `S^2` geometry from later Whittaker wave-kernel analysis.

---

## 4. Secondary Construction: Paired QA Seeds

Paired QA seeds should remain a secondary or comparison mode in v1:

```text
((b1,e1), (b2,e2))
```

Possible use:

```text
r = C1/G1
s = C2/G2
```

This is QA-native, but it should not be the primary v1 claim because it risks hiding the clean rational-sphere map behind a paired-angle interpretation. Two independent QA angles are not automatically Whittaker's spherical coordinates.

Primary v1 should certify the rational `S^2` direction set. Paired-seed behavior can be recorded as a subfamily, fixture mode, or later comparison cert.

---

## 5. Circle/Sphere Projection Consideration

There is a useful geometric intuition from earlier QA discussions:

- A circle can be treated as a special symmetric case or observer projection of an ellipse.
- The circular view appears when the focal structure is aligned or collapsed from the observer's perspective.
- By analogy, a sphere can be treated as the symmetric direction surface before introducing ellipsoidal, focal, or anisotropic projection structure.

For this Layer 2 cert, that intuition should guide language but not become a claim.

Recommended framing:

```text
Layer 1's S^1 net is the circular/aligned-focus boundary case.
Layer 2's S^2 net is the spherical/aligned-focus direction surface.
Later projection layers may introduce ellipsoidal or focal deformation, but Layer 2 certifies only the exact rational sphere geometry.
```

Guardrail:

```text
Do not claim that Whittaker's sphere is physically an ellipse or ellipsoid.
Do not claim that focal alignment proves a field ontology.
Use the circle/ellipse and sphere/ellipsoid language only as an observer-projection design note.
```

This keeps the insight available without contaminating the exact rational `S^2` cert.

---

## 6. Geodesy Bridge

Layer 2 also connects to existing QA geodesy and navigation work as a precedent for projection discipline:

- `[164]` **QA Gnomonic RT Cert**: great-circle and projection geometry precedent.
- `[165]` **QA Celestial Nav Cert**: spherical direction and sight-reduction precedent.
- `[168]` **QA Ellipsoid Geodesic Cert**: ellipsoid/geodesic deformation precedent.
- `[169]` **QA Ellipsoid Slice Cert**: latitude and meridian slice precedent.
- `[176]` **QA Inertial Nav Cert**: observer-projection and navigation framing precedent.

Key interpretation:

```text
Layer 2 certifies exact rational points on S^2.
Existing QA geodesy/navigation certs show that QA already has a tested projection discipline from spherical direction geometry into navigation and ellipsoidal/geodesic contexts.
Therefore the Whittaker S^2 layer should be treated as a spherical direction substrate.
Ellipsoidal, focal, geodesic, or Earth-specific interpretations belong to later projection/deformation layers, not to the Layer 2 cert itself.
```

Guardrails:

```text
Do not claim Earth geodesy follows from Whittaker.
Do not claim Whittaker proves geodesy.
Do not claim the S^2 net is already an ellipsoid.
Use the geodesy work only as QA precedent for projection discipline.
```

This bridge keeps the design aligned with the circle/ellipse and sphere/ellipsoid projection note: `S^2` is the symmetric/aligned-focus direction substrate; geodesy and ellipsoid models are deformation or observer-projection layers.

---

## 7. Proposed Gates

### W3D_1 — Exact Rational `S^2` Construction

Every generated point satisfies:

```text
x_num*x_num + y_num*y_num + z_num*z_num == den*den
```

All values are integers. No float arithmetic is used in construction.

### W3D_2 — Finite Enumeration and Duplicate Accounting

For declared `m`, the validator builds `R_m`, then `D_m^(2)`.

The fixture must declare:

```text
raw_parameter_count
raw_pair_count
unique_direction_count
duplicate_count
```

The validator recomputes all four exactly.

### W3D_3 — Chart and Coverage Discipline

The cert must declare which part of the sphere is covered:

```text
chart: north_stereographic
parameter_source: R_m x R_m
coverage_claim: one rational chart image from positive QA ratios
```

Sign reflections, antipodal closure, octant closure, or full-sphere closure are not automatic. If included, each must be declared and checked explicitly.

### W3D_4 — Exact Non-Equality and Spherical Separation Data

For distinct rational vectors `v_i`, `v_j`, compute exact dot and cross data:

```text
dot_num = xi*xj + yi*yj + zi*zj
dot_den = den_i * den_j
```

and cross-product numerator:

```text
cross = (
  yi*zj - zi*yj,
  zi*xj - xi*zj,
  xi*yj - yi*xj
)
```

At minimum, v1 should validate exact non-equality after deduplication. A lower separation bound may be added only if the proof is clean and the declared bound is conservative.

### W3D_5 — Spherical Lipschitz Nearest-Neighbor Bound

For an observer-side `L`-Lipschitz profile `g` on the covered spherical region, define:

```text
Delta_max_sphere = maximum sampled spherical nearest-neighbor radius
```

Then nearest-neighbor reconstruction should satisfy:

```text
sup |g(omega) - g(NN(omega))| <= L * Delta_max_sphere
```

This gate may be design-only in v1 if a deterministic spherical test grid is not yet settled. If implemented, all continuous grids and trigonometric functions must be tagged as observer-side checks.

---

## 8. Proposed Fixtures

Do not create these until build authorization.

```text
fixtures/pass_s2_m3_exact_sphere.json
fixtures/pass_s2_m5_duplicate_accounting.json
fixtures/pass_s2_m9_chart_summary.json
fixtures/fail_s2_bad_sphere_identity.json
fixtures/fail_s2_wrong_duplicate_count.json
fixtures/fail_s2_overclaimed_full_sphere.json
```

Keep `m` small until the enumeration growth is measured. The raw pair count grows as `|R_m|*|R_m|`.

---

## 9. Validator Requirements

When built, the validator should be:

- Pure Python stdlib only.
- Standalone.
- `--self-test` exits 0 and returns JSON with `"ok": true`.
- No imports from sibling experiment files.
- No `**2`; use `x*x`.
- Canonical JSON for any hashable output:

```python
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

Exact construction should use integers and `fractions.Fraction`. Floating point appears only in observer-side reporting or optional spherical-grid checks.

---

## 10. Non-Claims

Layer 2 does not claim:

- Whittaker 1903 is proved by QA.
- The full Whittaker wave kernel is approximated.
- Maxwell's equations, electromagnetism, gauge theory, or scalar-potential physics follow.
- The rational `S^2` set is equidistributed.
- The rational `S^2` set is dense as `m -> infinity`.
- Paired QA seed angles are identical to Whittaker's spherical coordinates.
- Sphere/ellipsoid or circle/ellipse projection language proves a physical focal ontology.
- Earth geodesy follows from Whittaker or from the `S^2` net.
- Whittaker proves geodesy.
- The `S^2` net is already an ellipsoid.

Layer 2 claims only exact finite rational `S^2` geometry under declared QA-derived parameter sources.

---

## 11. Next Decision Before Build

Before implementation, choose:

1. Small `m` values for v1 fixtures.
2. Whether v1 includes only exact geometry gates W3D_1-W3D_4, or also includes W3D_5 observer-side Lipschitz sampling.
3. Whether sign reflections or antipodal closure are included in v1 or deferred.
4. The next free cert-family ID at build time.
