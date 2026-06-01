# [292] QA Koenig Spread Optimality Cert

**Family ID**: 292
**Slug**: `qa_koenig_spread_optimality_cert_v1`
**Status**: Active
**Registered**: 2026-06-01

## Claim (narrow, falsifiable)

The Koenig I=1 (Pell chain) result from cert [289] has a **purely rational trig reformulation** — no √2 appears in any statement or proof.

## Definitions

For any (b,e) ∈ ℤ>0 × ℤ>0:

```
G̃(b,e)  =  b² + e²             (blue quadrance of direction vector (b,e))
s(b,e)   =  e² / G̃              (Wildberger spread from horizontal, ∈ ℚ)
I(b,e)   =  |b² − 2e²|          (Koenig I invariant, cert [289])
```

## Key Identity

```
I(b,e)  =  3 · G̃ · |s − 1/3|
```

*Proof*: |2e²−b²| = |3e²−(b²+e²)| = G̃·|3e²/G̃ − 1| = G̃·|3s−1| = 3G̃·|s−1/3|. □

This is an algebraic identity. It says: **the Koenig I invariant equals 3 × quadrance × spread-deviation from 1/3**.

## Claims

**(1) SPREAD_ID** — The identity I = 3G̃|s−1/3| holds for all b,e ≥ 1.

**(2) NO_EXACT** — No (b,e) ∈ ℤ>0 × ℤ>0 satisfies s = 1/3 exactly. Proof: s=1/3 ↔ b²=2e² ↔ b/e=√2, but √2 is irrational. The spread-1/3 direction has **no rational representative**.

**(3) PELL_OPT** — I(b,e)=1 iff |s−1/3| = 1/(3G̃). Since I ≥ 1 for all integer (b,e) (claim 2), the Pell solutions achieve the minimum nonzero spread-deviation at each scale G̃.

**(4) INTER_SPREAD** — For consecutive Pell pairs (b_n,e_n), (b_{n+1},e_{n+1}), the Wildberger spread between the two direction lines is:

```
spread(n, n+1)  =  (b_n e_{n+1} − b_{n+1} e_n)² / (G̃_n · G̃_{n+1})
                =  1 / (G̃_n · G̃_{n+1})          (det=±1 by Farey, cert [289])
```

This decreases monotonically since G̃_n grows geometrically (ratio → (1+√2)² ≈ 5.83).

**(5) ALT_SIDE** — Consecutive Pell spreads s_n lie on alternating sides of 1/3:

```
s_1 = 1/2 > 1/3,   s_2 = 4/13 < 1/3,   s_3 = 25/74 > 1/3,   ...
```

## The √2 Cusp as a Spread

The "√2 cusp" of the Ford circle chain is, in rational trig language: **the direction with spread exactly 1/3 from horizontal**. Spread 1/3 is rational. No rational direction achieves it. The Pell chain approaches it optimally.

| Pair (b,e) | G̃ | s = e²/G̃ | \|s − 1/3\| | = 1/(3G̃)? |
|---|---|---|---|---|
| (1,1) | 2 | 1/2 | 1/6 | ✓ |
| (3,2) | 13 | 4/13 | 1/39 | ✓ |
| (7,5) | 74 | 25/74 | 1/222 | ✓ |
| (17,12) | 433 | 144/433 | 1/1299 | ✓ |
| (41,29) | 2522 | 841/2522 | 1/7566 | ✓ |

## Checks

| ID | Description |
|---|---|
| SPREAD_ID | I = 3·G̃·\|s−1/3\| (algebraic identity; integer/Fraction arithmetic) |
| I_MATCH | Declared expected_I matches computed \|b²−2e²\| |
| PELL_OPT | If I=1: \|s−1/3\|=1/(3G̃); if I>1: \|s−1/3\|>1/(3G̃) |

**Self-test also verifies**: SPREAD_ID for all b,e ∈ [1,29]; NO_EXACT for b,e ∈ [1,50]; INTER_SPREAD for 12-pair Pell chain; ALT_SIDE.

**Fixtures**: 4 PASS + 2 FAIL

## Why This Is Distinct from [289] and [141]

- **[289]** (Koenig Pell Ford Circle): certifies I=1 ↔ Pell ↔ Ford tangency, references "√2" as the limit
- **[141]** (QA Pell Norm): certifies I = −(Pell norm), algebraic identity, no spread interpretation
- **[292]** (this cert): reformulates in Wildberger spread language, eliminates √2 from statement, adds the inter-direction spread formula and ALT_SIDE convergence

## Primary Sources

- Wildberger, N. J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8. Spread s = sin²θ between lines, blue quadrance.
- Hardy, G. H., & Wright, E. M. (2008). *An Introduction to the Theory of Numbers* (6th ed.). Oxford University Press. ISBN 978-0-19-921986-5. Pell equation, Diophantine approximation.

## Mechanism Chain

- [289] QA Koenig Pell Ford Circle — Farey det=±1 underlies INTER_SPREAD
- [141] QA Pell Norm — I = −Pell norm; same object, different language
- [125] QA Chromogeometry — Wildberger quadrance/spread framework
