# Family [164] QA_GNOMONIC_RT_CERT.v1

## One-line summary

Gnomonic map projection expressed entirely via rational trigonometry (spreads and crosses), with Berggren tree paths as discrete geodesic straight-line segments on the chart.

## Mathematical content

### Classical vs Rational gnomonic projection

Classical gnomonic from tangent point (phi_0, lambda_0):

```
x = cos(phi) sin(Delta_lambda) / cos(c)
y = [cos(phi_0) sin(phi) - sin(phi_0) cos(phi) cos(Delta_lambda)] / cos(c)
```

Rational gnomonic (quadrance form):

```
cos(c) = sqrt(s_0 * s) + sqrt(c_0 * c * c_Delta)
spread_c = 1 - cos^2(c)     (spread of angular distance)
cross_c = cos^2(c)

Gnomonic quadrance: Q = spread_c / cross_c = tan^2(c)
```

where s = sin^2(phi) = spread, c = 1 - s = cross.

### The defining property: great circles to straight lines

The gnomonic is the ONLY map projection where every great circle becomes a straight line. In RT terms: collinearity is verified by zero cross-product quadrance of projected points.

Verified: 5 points along a great circle from London bearing 45 degrees project with cross products < 10^{-15} — machine-epsilon collinearity.

### Berggren tree connection

The three Berggren generators:

| Move | Rule | From (2,1) | Triple (C,F,G) |
|------|------|------------|----------------|
| M_A | (d,e) -> (2d-e, d) | (3,2) | (12, 5, 13) |
| M_B | (d,e) -> (2d+e, d) | (5,2) | (20, 21, 29) |
| M_C | (d,e) -> (d+2e, e) | (4,1) | (8, 15, 17) |

Each generator produces a Pythagorean triple satisfying C^2 + F^2 = G^2. These are discrete steps along geodesics on the Pythagorean cone. Under gnomonic projection, geodesics become straight lines.

Therefore: **Berggren tree navigation = straight-line segments on the gnomonic chart.**

### Plimpton 322 as navigation table

Plimpton 322 (cert [138]) lists Pythagorean triples = Berggren tree nodes. Each row gives a direction (d, e) and its gnomonic quadrance G = d^2 + e^2. The Babylonian tablet IS a gnomonic navigation table.

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_gnomonic_rt_cert_v1
python qa_gnomonic_rt_cert_validate.py --self-test
```

## Validator checks

| Check | Description |
|-------|-------------|
| GN_1 | schema_version == QA_GNOMONIC_RT_CERT.v1 |
| GN_QUAD | Gnomonic quadrance Q = spread_c / cross_c matches classical x^2+y^2 |
| GN_SPREAD | spread_c + cross_c = 1 (fundamental identity) |
| GN_COLLINEAR | Great circle points project to collinear points |
| GN_BERGGREN | Berggren triples satisfy C^2+F^2=G^2 |
| GN_W | At least 3 projection witnesses |
| GN_F | Fail detection |

## Fixtures

| Fixture | Result | Content |
|---------|--------|---------|
| gn_pass_london.json | PASS | London tangent, 5 cities, collinearity test, 5 Berggren moves (2 depths) |
| gn_fail_wrong_triple.json | FAIL | Non-Pythagorean triple (7,11,13) — not on the cone |

## Key insight

The gnomonic projection was always a spread/cross computation in disguise — classical notation with sin/cos obscured the rational structure. The connection to the Berggren tree means that discrete navigation on the Pythagorean cone projects to straight-line segments on the gnomonic chart, giving a complete integer-arithmetic navigation system.

## Applications

1. **Great-circle route planning** — exact waypoints without trig tables
2. **FPGA/embedded chart computation** — integer multiply + mod only
3. **Archaeological cartography** — Plimpton 322, Carnac sites, Megalithic Yard all fit the gnomonic/QA framework
4. **Air navigation** — gnomonic charts are standard for plotting great-circle routes

## Cross-references

- [135] QA_PYTHAGOREAN_TREE_CERT.v1 — Berggren generators = discrete geodesic moves
- [138] QA_PLIMPTON322_CERT.v1 — Babylonian tablet = Berggren tree nodes = nav table
- [156] QA_WGS84_ELLIPSE_CERT.v1 — Earth = QA quantum ellipse
- [161] QA_ECEF_RATIONAL_CERT.v1 — ECEF coordinates via spreads/crosses
- [163] QA_DEAD_RECKONING_CERT.v1 — T-operator exact DR on the same lattice
