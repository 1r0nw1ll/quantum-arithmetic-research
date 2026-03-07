# Family [97] — QA Orbit Curvature Cert

**Cert root:** `qa_orbit_curvature_cert_v1/`  
**Validator:** `qa_orbit_curvature_cert_v1/validator.py --self-test`  
**Schema:** `QA_ORBIT_CURVATURE_CERT.v1.schema.json`

## What it certifies

Given a starting state `(b₀, e₀)` and a modulus, the QA update rule
`(b,e) → (d,a)` where `d=(b+e) mod* m`, `a=(b+2e) mod* m`
generates a finite deterministic orbit. This cert enumerates that orbit
completely and pins the **minimum κ across all orbit states** as the
multi-step stability margin.

## Key insight

The QA step is a **2-step Fibonacci recurrence**: each application advances
two terms of the Fibonacci sequence modulo `m`. Orbit lengths therefore equal
half the Pisano period `π(m)/2`. For `m=9`, `π(9)=24`, giving orbits of
length 12.

## κ_min interpretation

For each orbit state `t`:
```
η_eff(t) = lr · gain · H_QA(t)
κ(t)     = 1 − |1 − η_eff(t)|
```
`κ_min = min_t κ(t)` is the tightest single-step stability bottleneck
across the entire orbit. `κ_min > 0` certifies no state hits the degenerate
case `η_eff = 1` (which would make the curvature-scaled update collapse).

## Three gates

| Gate | Check |
|------|-------|
| A | Enumerate orbit from `(b₀,e₀)`; verify `claimed.orbit_length` |
| B | Recompute `H_QA` at every state via the substrate formula |
| C | Compute `κ(t)` at every state; verify `claimed.kappa_min` |

## Canonical fixture

- `orbit_start`: `(b=1, e=2)`, `modulus=9`
- `optimizer`: `lr=0.5`, `gain=1.0`
- `orbit_length`: 12 (half Pisano period of 9)
- `kappa_min`: 0.10766 (bottleneck at state `(8,1,9,1)`, `H_QA≈0.2153`)

## Orbit structure (mod 9)

| Type | Count | Length | Pairs |
|------|-------|--------|-------|
| Fixed point | 1 | 1 | (9,9) |
| 4-cycles | 2 | 4 | 8 pairs |
| 12-cycles | 6 | 12 | 72 pairs |

The 12-cycle group (72 pairs) corresponds to the documented "Cosmos" orbit;
the 4-cycle group (8 pairs) to the "Satellite" orbit. The documentation's
"24-cycle / 8-cycle" labels refer to the full Pisano period, not the 2-step
QA orbit length.
