# Family [168] QA_ELLIPSOID_GEODESIC_CERT.v1

## One-line summary

Geodesic properties of the WGS84 quantum ellipse expressed entirely in QN components: curvature M/N = F/(d^2 - e^2 s_phi), axis ratio b/a = sqrt(F)/d, discriminant I = C-F = -10039 < 0.

## Mathematical content

### Curvature in QN arithmetic

For shape QN (101, 9, 110, 119) with triple (C,F,G) = (1980, 12019, 12181):

- Prime vertical: N = a_earth * d / sqrt(d^2 - e^2 * s_phi)
- Curvature ratio: **M/N = F / (d^2 - e^2 * s_phi)** = 12019 / (12100 - 81 * s_phi)
- At equator: M/N = F/d^2 = 0.9933 (maximum difference)
- At pole: M/N = F/F = 1 (locally spherical)

### Quantum lattice points

Geophysically significant latitudes coincide with QN-harmonic spreads:

| s_phi | Formula | Latitude | Significance |
|-------|---------|----------|-------------|
| 81/12181 | e^2/G | 4.68° | Eccentricity resonance |
| 1980/12181 | **C/G** | **23.78°** | **Within 0.34° of Tropic of Cancer** |
| 12019/12181 | F/G | 83.38° | Red/blue boundary |

The C/G = green/blue ratio predicting the Tropic is striking — the green quadrance of the shape QN nearly determines Earth's axial tilt boundary.

## Cross-references

- [156] QA_WGS84_ELLIPSE — shape QN definition
- [140] QA_CONIC_DISCRIMINANT — I < 0 = ellipse
- [163]-[167] Navigation stack operates ON this surface
- [169] QA_ELLIPSOID_SLICE — slicing this same surface

## Verification Note (2026-07-06)

Self-contained QA-internal derivation building on [156]'s already-verified
Earth shape QN — no new external citation needed. Independently
recomputed every claim from scratch: C=1980, F=12019, G=12181 (matching
[156]); I=C−F=−10039 exactly; M/N at the equator = F/d² = 0.99331
exactly; all three quantum lattice points via arcsin(√(s_num/s_den)):
e²/G→4.677° (matches "4.68°"), C/G→23.777° (matches "23.78°", the
Tropic-of-Cancer coincidence), F/G→83.378° (matches "83.38°"). 0
mismatches.

Validator confirmed genuinely computing, not fixture-trusting: `C`,
`F`, `G`, `I`, axis ratio, curvature ratio M/N, prime-vertical radius N
(using the real WGS84 equatorial radius 6378137.0 m, confirmed in
[156]'s audit), and all lattice-point latitudes are recomputed from the
raw QN at runtime and checked against declared values, not the reverse.
`--self-test` passes on both fixtures. No bugs found.
