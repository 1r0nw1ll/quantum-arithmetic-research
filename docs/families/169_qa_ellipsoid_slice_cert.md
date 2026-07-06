# Family [169] QA_ELLIPSOID_SLICE_CERT.v1

## One-line summary

QA-style slicing of the WGS84 quantum ellipse: latitude circles (R^2 rational in spreads), self-similar meridian ellipses, chromogeometric C/F/G curve families, and 24 Pisano longitude bands.

## Mathematical content

### Three types of QA slicing

**Latitude slices** produce circles with:
```
R^2 = a^2 * d^2 * c_phi / (d^2 - e^2 * s_phi)
```
All rational in spreads + QN components. Circle radius decreases from equator (R = a) to pole (R = 0).

**Meridian slices** produce ellipses with:
- Axis ratio = sqrt(F)/d = sqrt(12019)/110
- Eccentricity^2 = e_QN^2/d_QN^2 = 81/12100
- The meridian IS the shape QN realized as a 2D curve — **self-similar**.

**Chromogeometric slices** — three orthogonal curve families:
- C-constant curves (green/area contours)
- F-constant curves (red/Minkowski contours)
- G-constant curves (blue/Euclidean contours)
- At every point: C^2 + F^2 = G^2

### Pisano longitude bands

pi(24) = 24 divides longitude into 24 bands of 15 degrees each. This corresponds to:
- 24 hours of sidereal rotation
- 24 time zones
- 24-fold cosmos orbit symmetry

**Time zones are a Pisano-period partition of the quantum ellipse.**

## Cross-references

- [156] QA_WGS84_ELLIPSE — shape QN (101,9,110,119)
- [125] QA_CHROMOGEOMETRY — C, F, G curve families
- [128] QA_SPREAD_PERIOD — Pisano period pi(24) = 24
- [168] QA_ELLIPSOID_GEODESIC — curvature on same surface

## Verification Note (2026-07-06)

Self-contained QA-internal derivation on [156]'s Earth shape QN, no new
external citation needed. Independently verified the latitude-circle
formula R²=a²d²c_φ/(d²−e²s_φ) against the standard oblate-spheroid
formula R(φ)=N(φ)cos(φ) at φ=45° — exact match to 1 part in 10¹²,
confirming both the algebra and that the doc's "a" here means the real
equatorial radius (a_earth=6378137 m), not the QN's own element a=119
(same symbol, different referent — worth noting since the doc doesn't
say so explicitly, though the validator's own comment already does).
Confirmed limiting cases (R=a at equator, R=0 at pole) and the meridian
ellipse's axis ratio/eccentricity² (matching [156]/[168] exactly).
Independently recomputed the Pisano period π(24)=24 from scratch (not
just trusted) — confirmed real.

Validator confirmed genuinely computing every check (latitude-circle R²,
meridian axis ratio, chromogeometric C/F/G, band arithmetic) at runtime
using the real WGS84 equatorial radius, not fixture-trusting. One minor
observation: `SL_BAND` only checks `n_bands × band_width = 360°`
arithmetically — it doesn't itself recompute the Pisano period π(24)=24
that justifies n_bands=24 (unlike the other checks, that deeper fact is
a fixed mathematical constant, not empirical data subject to drift, so
the risk profile is much lower than the astronomical-data bugs found
elsewhere this cycle; not treating this as something requiring a fix).
`--self-test` passes on both fixtures. No bugs found.
