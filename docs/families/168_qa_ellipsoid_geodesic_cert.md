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
