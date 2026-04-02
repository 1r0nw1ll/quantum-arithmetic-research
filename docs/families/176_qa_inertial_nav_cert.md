# Family [176] QA_INERTIAL_NAV_CERT.v1

## One-line summary

Formal proof that QA T-operator navigation has zero computational drift vs classical INS O(epsilon * sqrt(N)) error growth.

## Mathematical content

Classical INS per step: x += d * sin(theta) + noise. After N steps with per-step noise epsilon: RMS error = epsilon * d * sqrt(N).

QA per step: (b,e) -> (e, b+e) mod m. After N steps: T^N * (b0,e0) mod m. Error = 0 for all N.

Error budget at 10000 steps:

| Noise source | sigma | Classical drift | QA drift |
|---|---|---|---|
| IEEE 754 ULP | 1e-15 | ~1e-10 m | 0 |
| Trig table | 1e-10 | ~6e-6 m | 0 |
| MEMS IMU | 1e-6 | ~0.06 m | 0 |
| Cheap IMU | 1e-3 | ~57 m | 0 |

The ratio classical/QA diverges to infinity. Theorem NT: the ONLY error in QA nav is observer projection at input/output boundaries.

## Cross-references

- [163] QA_DEAD_RECKONING — T-operator mechanics
- [164] QA_GNOMONIC_RT — gnomonic chart for waypoints
- [168] QA_ELLIPSOID_GEODESIC — curvature of navigation surface
