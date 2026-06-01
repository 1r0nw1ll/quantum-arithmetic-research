# [290] QA Classical Subfamily Ford Cusps Cert

**Family ID**: 290
**Slug**: `qa_classical_subfamily_ford_cusps_cert_v1`
**Status**: Active
**Registered**: 2026-06-01

## Claim (narrow, falsifiable)

The three classical subfamilies of primitive Pythagorean triples form three distinct Farey-adjacent Ford circle chains from the common seed **(1,1)**, each converging to a distinct cusp:

| Subfamily | Condition | Chain | Curvature | Cusp |
|---|---|---|---|---|
| Pythagoras | b = 1 | (1,1),(1,2),(1,3),... | 2·1², 2·2², 2·3²,... (increasing) | 0 (rational) |
| Plato | e = 1 | (1,1),(2,1),(3,1),... | 2·1², 2·1², 2·1²,... (uniform = 2) | ∞ (rational, periodic) |
| Fermat | I = 1 | (1,1),(3,2),(7,5),(17,12),... | 2·1², 2·2², 2·5²,... (geometric) | √2 (quadratic irrational) |

## Subfamily Conditions

From Wildberger (2005) *Divine Proportions*, in BEDA coordinates (b,e,d=b+e,a=b+2e):

- **Pythagoras**: (d−e)² = 1 ↔ b² = 1 ↔ **b = 1** (one step off the diagonal)
- **Plato**: |G−F| = 2 ↔ 2e² = 2 ↔ **e = 1** (unit denominator)
- **Fermat**: |C−F| = 1 ↔ |b²−2e²| = 1 (Pell equation — see cert [289])

## Farey Adjacency Proofs

**Pythagoras**: For consecutive (1,e), (1,e+1): |1·(e+1) − 1·e| = 1 ✓ — always adjacent, for any e.

**Plato**: For consecutive (b,1), (b+1,1): |b·1 − (b+1)·1| = 1 ✓ — always adjacent, for any b.

**Fermat**: |b_n·e_{n+1} − b_{n+1}·e_n| = |b_n²−2e_n²| = 1 ✓ — proven in cert [289].

## Ford Circle Curvature

The Ford circle C(b/e) has curvature 2e². Therefore:

- **Pythagoras** (e = 1,2,3,...): curvatures 2, 8, 18, 32,... — strictly increasing, Ford circles shrink toward x-axis at 0
- **Plato** (e = 1 fixed): curvature = 2 for all b — uniform row of equal circles at integer positions
- **Fermat** (Pell denominators e = 1,2,5,12,29,...): curvatures 2, 8, 50, 288,... — geometrically growing, Ford circles shrink toward √2

## The Three Cusps

The three chains diverge from (1,1) in three directions on the Stern-Brocot tree:

```
           √2 (Fermat)
           ↗
(1,1) ——→ 0 (Pythagoras, leftward)
           ↘
           ∞ (Plato, rightward/upward)
```

- **0** is a rational cusp — Pythagoras circles fill the left edge
- **∞** is a rational cusp — Plato circles are the periodic integer row
- **√2** is a quadratic irrational cusp — Fermat circles are the Pell chain (cert [289])

The three cusps are distinct elements of ℝ ∪ {∞}: 0 < √2 < ∞.

## Checks

| ID | Description |
|---|---|
| PYTH_COND | b = 1 for every element in Pythagoras chain |
| PYTH_FAREY | \|b_n·e_{n+1}−b_{n+1}·e_n\| = 1 for consecutive pairs |
| PYTH_CURV | Ford curvatures 2e² strictly increasing |
| PLATO_COND | e = 1 for every element in Plato chain |
| PLATO_FAREY | \|b_n·1−b_{n+1}·1\| = 1 for consecutive pairs |
| PLATO_CURV | Ford curvatures 2e² = 2 (uniform) |
| FERMAT_PELL | \|b²−2e²\| = 1 for every element |
| FERMAT_FAREY | \|b_n·e_{n+1}−b_{n+1}·e_n\| = 1 for consecutive pairs |
| FERMAT_MAP | (b+2e, b+e) generates next element |

**Fixtures**: 3 PASS + 3 FAIL
**Self-test**: 12-term chains for all three types, cusp direction checks, fail-case verification

## Primary Sources

- Wildberger, N. J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8. Pythagoras/Plato/Fermat subfamily conditions in BEDA coordinates.
- Hardy, G. H., & Wright, E. M. (2008). *An Introduction to the Theory of Numbers* (6th ed.). Oxford University Press. ISBN 978-0-19-921986-5. Ch. III: Ford circles, Farey adjacency.

## Mechanism Chain

- [289] QA Koenig Pell Ford Circle — Fermat chain is the Pell/Ford [289] result; this cert extends to Pythagoras and Plato
- [125] QA Chromogeometry — C²+F²=G² underlies the subfamily definitions
