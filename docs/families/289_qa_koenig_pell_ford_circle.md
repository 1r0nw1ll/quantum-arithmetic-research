# [289] QA Koenig Pell Ford Circle Cert

**Family ID**: 289
**Slug**: `qa_koenig_pell_ford_circle_cert_v1`
**Status**: Active
**Registered**: 2026-06-01

## Claim (narrow, falsifiable)

The Koenig I=1 BEDA sequence S = {(b,e) : Koenig I(b,e) = 1, b ≥ 1, e ≥ 1, ordered by e} equals the Pell equation solution set {(b,e) : |b² − 2e²| = 1}. Consecutive elements are **Farey neighbors** — satisfying the Ford circle tangency condition |b_n·e_{n+1} − b_{n+1}·e_n| = 1. The sequence converges to √2 via alternating over/under approximation.

## Key Identity

The Koenig I invariant is algebraically identical to the Pell discriminant:

```
I(b,e) = |C − F|
       = |2(b+e)e − (b+2e)b|
       = |2e² − b²|
       = |b² − 2e²|
```

So I = 1 iff (b,e) solves x² − 2y² = ±1.

## QA-Generation Theorem

The QA state map **(b,e) → (a,d) = (b+2e, b+e)** generates each Pell solution from the previous. This is multiplication by (1+√2) in ℤ[√2]:

| n | (b,e) | b²−2e² | Koenig (I,G,H) |
|---|---|---|---|
| 1 | (1,1) | −1 | (1, 5, 7) |
| 2 | (3,2) | +1 | (1, 29, 41) |
| 3 | (7,5) | −1 | (1, 169, 239) |
| 4 | (17,12) | +1 | (1, 985, 1393) |
| 5 | (41,29) | −1 | (1, 5741, 8119) |
| 6 | (99,70) | +1 | (1, 33461, 47321) |

## Farey Adjacency Proof

Let (b', e') = (b+2e, b+e). Then:

```
|b·e' − b'·e| = |b(b+e) − (b+2e)e| = |b² − 2e²| = 1
```

So consecutive Pell solutions are always Farey neighbors — and their Ford circles C(b/e) and C(b'/e') are always tangent.

## Connection to the Stern-Brocot Tree

The two SL(2,Z) generators are exactly the two QA transitions:

- **L-move**: b → b+e (denominator fixed) = Ford left-step
- **R-move**: e → b+e (numerator fixed) = Ford right-step

The Pell sequence walks the Stern-Brocot tree along the geodesic from 1/1 toward √2. Each step is the QA (a,d) map applied from the √2 side.

## Connection to the Five Families

Each generalized Fibonacci family in the five-families paper (cert [282]) defines a sequence of (b,e) pairs whose ratio b/e converges to a different quadratic irrational cusp in the Stern-Brocot tree:

| Family | Limit ratio | Cusp |
|---|---|---|
| Fibonacci | φ = (1+√5)/2 | golden ratio |
| Tribonacci | T ≈ 1.839 | tribonacci constant |
| Ninbonacci | 1/1 | Singularity fixed point |

The Pell/Fermat family (this cert) corresponds to the √2 cusp.

## Checks

| ID | Description |
|---|---|
| PELL_1 | \|b²−2e²\| = 1 for every (b,e) in sequence |
| KOENIG_1 | koenig_I(b,e) = \|2(b+e)e−(b+2e)b\| = 1 (QA/BEDA path) |
| FAREY_1 | \|b_n·e_{n+1}−b_{n+1}·e_n\| = 1 for all consecutive pairs |
| ALT_1 | b²−2e² alternates sign (convergence from both sides of √2) |

**Fixtures**: 2 PASS + 2 FAIL
**Self-test**: 12-term Pell sequence, all 4 checks, both fail-case patterns

## Primary Sources

- Hardy, G. H., & Wright, E. M. (2008). *An Introduction to the Theory of Numbers* (6th ed.). Oxford University Press. ISBN 978-0-19-921986-5. Chapter III: Farey sequences, Ford circles, tangency condition |pq'−p'q|=1.
- Wildberger, N. J. (2005). *Divine Proportions: Rational Trigonometry to Universal Geometry*. Wild Egg Books. ISBN 978-0-9757492-0-8. BEDA tuples, Koenig I=|C−F| invariant.

## Mechanism Chain

- [288] QA Anchor Geodesic Separation — QA distance coordinate framework
- [282] QA Fibonacci Orbit Index — five families / cusp structure
- Koenig HSI benchmark (OB 2026-05-31): AUROC=0.9566, confirmed Koenig features carry structural QA signal
