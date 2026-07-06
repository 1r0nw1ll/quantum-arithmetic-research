# Family [166] QA_LOXODROME_CERT.v1

## One-line summary

Loxodromes (rhumb lines) arise naturally from QA T-operator iteration as constant-bearing paths on a mod-m lattice, with the Mercator identity s_phi = tanh^2(psi).

## Mathematical content

### Discrete loxodrome

A QA T-operator path with fixed generator IS a constant-bearing path — the discrete analogue of a loxodrome. Period = Pisano period pi(m). Unlike continuous loxodromes (which never close except on cardinal bearings), QA loxodromes always close after pi(m) steps.

### Bearing spread

For direction (d, e): bearing_spread = e^2 / G where G = d^2 + e^2.

### Mercator identity

s_phi = tanh^2(psi) where psi = arctanh(sin(phi)) = isometric latitude.

Spreads ARE Mercator-native. The isometric latitude's squared hyperbolic tangent IS the spread of latitude.

### Orbit partition

- Cosmos (period 24): full circumnavigation loxodromes
- Satellite (period 8): reduced loxodromes (8 principal winds)
- Singularity (period 1): degenerate (no motion)

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_loxodrome_cert_v1
python qa_loxodrome_cert_validate.py --self-test
```

## Cross-references

- [163] QA_DEAD_RECKONING — T-operator path mechanics
- [164] QA_GNOMONIC_RT — gnomonic (great circle) vs Mercator (rhumb) duality
- [128] QA_SPREAD_PERIOD — Pisano period = loxodrome cycle length

## Verification Note (2026-07-06)

Self-contained cartography/geodesy math (Mercator's 1569 projection is
real, foundational, well-established history of cartography — not
independently re-searched given very high baseline confidence, same
treatment as other canonical references this cycle). Independently
recomputed both central identities from scratch: the Mercator identity
ψ=arctanh(sin φ) ⟹ tanh²(ψ)=sin²(φ)=s_φ confirmed exact at 5 test
latitudes (10°-80°), and independently cross-checked that this arctanh
form equals the textbook ln(tan(π/4+φ/2)) isometric-latitude formula
(exact match, <10⁻¹⁰ diff) — this is genuine, standard cartographic
math, not a QA-invented identity. Bearing-spread formula e²/G=sin²(θ)
for direction (d,e) confirmed exact against atan2 for 4 test directions
including two Pythagorean triples.

Validator confirmed genuinely computing, not fixture-trusting:
`t_operator`/`loxodrome_period`/`classify_orbit` simulate the actual QA
path and Pisano period at runtime, and the Mercator/bearing checks use
live `math.tanh`/`atan2`, not declared values. `--self-test` passes on
both fixtures. No bugs found.
