# [517] QA Volk Apollonian Bipolar Mapping Cert

**Family ID**: 517
**Slug**: `qa_volk_apollonian_mapping_cert_v1`
**Status**: Active
**Registered**: 2026-07-07

## Claim (narrow, falsifiable)

For any QA tuple (b,e,d,a) with d=b+e, a=b+2e (QA's own A2 definitions), the identity

```
d^2 - e^2 = (d-e)(d+e) = b*(b+2e) = b*a = F
```

holds exactly — verified over an exhaustive grid of 2,500 (b,e) pairs (b,e in 1..50), zero exceptions — and this is precisely Greg Volk's own Apollonian orthogonality relation `R^2 = a_volk^2 + r_volk^2` (from "Toroids, Vortices, Knots, Topology and Quanta, Part 2", NPA-18, 2010, section 2: "there may exist many M-circles or toroids centered at B=(±R,0)... which we'll see satisfies [this relation], with r the radius of the M-circle and R the distance from the origin to the circle center") under the mapping:

| Volk quantity | QA equivalent |
|---|---|
| `a_volk` (bipolar pole half-distance) | `e` |
| `R_volk` (M-circle center) | `d` |
| `r_volk` (M-circle radius) | `sqrt(F) = sqrt(a*b)` |
| `eta` (bipolar hyperbolic coordinate) | `arccoth(d/e) = 0.5*ln(a/b)` |

## Provenance

**Primary source recovery**: the on-disk OCR text extraction of Volk's paper had lost every equation body (visible only as bare `EMBED Equation.DSMT4` placeholders). The original `.doc` was located and recovered, and its 233 embedded MathType "Equation Native" OLE objects were decoded with a from-scratch MTEF v5 binary parser (`qa_lab/qa_mtef_parser.py`), built directly from the documented format spec. Two decoded equation objects give direct, byte-level confirmation (not inference from a secondary AI summary):

- Volk's own toroidal coordinate vector: numerator `(sinh(η)cos(θ), sinh(η)sin(θ), sin(φ))` over denominator `(cosh(η)−cos(φ))` — his equation (4.3).
- The hyperbolic↔circular identity chain `tanh(η)=sin(ψ)`, `cosh(η)=sec(ψ)`, `coth(η)=csc(ψ)`, etc. — self-consistent.

**Independent corroboration**: a GeoGebra construction the user built over a year before this cert (`geogebra.org/calculator/nwkeyb7j`, titled "grant,volk-toroid1235"), built directly from BEDA=(1,2,3,5) against Volk's own Figure 2, places point `A=(e,0)=(2,0)`, point `R=(d,0)=(3,0)`, and a circle centered at `R` with radius `sqrt(5)=sqrt(F)` — an exact numeric match to this derivation, found independently and only afterward connected to it.

## Implementation

`qa_lab/qa_volk_coordinates.py`, function `beda_to_volk(b, e, d, a, rho=0.0)`.

## Limitation (not resolved by this cert)

A single static BEDA tuple only fixes the M-circle/E-circle **family** (`a_volk`, `eta`) — i.e. which torus shape — not a specific point on it (the bipolar angle `rho`). Which `rho` a QA *orbit* should trace as it iterates (echoing Volk's own m:n winding-number framework) is an open question.

## Checks

| ID | Description |
|---|---|
| VAM_IDENTITY | `d^2-e^2 == a*b` exactly, for the fixture's (b,e) |
| VAM_A_VOLK | declared `expected_a_volk == e` |
| VAM_R_VOLK | declared `expected_R_volk == d == b+e` |
| VAM_R2_VOLK | declared `expected_r_volk_squared == a*b` |
| VAM_APOLLONIAN | `R_volk^2 == a_volk^2 + r_volk^2` (Volk's own relation) |
| VAM_ETA | `eta=0.5*ln(a/b)` matches `arccoth(d/e)` AND independently reconstructs `R_volk` via `a_volk*coth(eta)` through `cosh`/`sinh` directly (self-test gate, not a per-fixture field — `eta` is an observer-projection float per Theorem NT, not raw QA state) |

**Fixtures**: 3 PASS + 3 FAIL
**Self-test**: exhaustive identity check over 2,500 (b,e) pairs, Apollonian relation over the same grid, eta cross-checked three independent ways over a 400-pair grid, canonical-tuple witness cross-checked against the GeoGebra construction, mapping-protocol gate, fixture gate.

## Primary Sources

- Volk, G. (2010). "Toroids, Vortices, Knots, Topology and Quanta, Part 2." *Proceedings of the NPA* (NPA-18), College Park, MD.
- Ginzburg, V. (2006). *Prime Elements of Ordinary Matter, Dark Matter & Dark Energy*. Helicola Press. (Source of the R/r "helicola" parameter naming Volk builds on.)

## Mechanism Chain

- [291] QA Fibonacci Matrix Orbit Periods — shares the mod-3/root-derivation machinery
- [292]/[293] QA Koenig Spread Optimality / Shell Structure — related quadrance identities
