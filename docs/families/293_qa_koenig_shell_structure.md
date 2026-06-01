# [293] QA Koenig Shell Structure Cert

**Family ID**: 293
**Slug**: `qa_koenig_shell_structure_cert_v1`
**Status**: Active
**Registered**: 2026-06-01

## Claim (narrow, falsifiable)

The Ford circle packing is stratified by the Koenig I invariant. The **I=k shell** is:

```
S_k  =  { (b,e) ∈ ℤ>0 × ℤ>0 :  |b² − 2e²| = k }
```

For any (b,e) ∈ S_k with QA successor (b',e') = (b+2e, b+e):

| Check | Claim | Algebraic proof |
|---|---|---|
| SHELL_PRESERVE | I(b',e') = k | (b+2e)²−2(b+e)² = −(b²−2e²), so \|I\|=k |
| SIGN_FLIP | b'²−2e'² = −(b²−2e²) | same |
| FAREY_K | \|be'−b'e\| = k | \|b(b+e)−(b+2e)e\| = \|b²−2e²\| = k |
| SPREAD_K | spread(d_n, d_{n+1}) = k²/(G̃·G̃') | det²/(G̃G̃') = k²/(G̃G̃') |
| SPREAD_DEV_K | \|s(b,e)−1/3\| = k/(3G̃) | cert [292] identity |
| SHELL_UNIQUE | k=1 is the only shell with Farey det=1 | trivial from FAREY_K |

## Shell Examples

| Shell k | Seed | Chain | Farey det | Notes |
|---|---|---|---|---|
| k=1 | (1,1) | (1,1)→(3,2)→(7,5)→(17,12)→... | 1 | Pell chain (cert [289]) |
| k=2 | (2,1) | (2,1)→(4,3)→(10,7)→(24,17)→... | 2 | Ford circles not tangent |
| k=7 | (3,1) | (3,1)→(5,4)→(13,9)→(31,22)→... | 7 | 7≡−1(mod 8), splits in ℤ[√2] |
| k=8 | (4,2) | (4,2)→(8,6)→(20,14)→... | 8 | = 2²·(k=2 chain scaled ×2) |

## Empty Shells

Not every k has solutions. Prime p is **inert in ℤ[√2]** iff p≡±3(mod 8), meaning b²−2e²≠±p for any integer (b,e). This propagates: any k whose prime factorization contains an inert prime to an odd power has no solutions.

Empty shells verified for b,e ∈ [1,100]:

| k | p mod 8 | Status |
|---|---|---|
| 3 | 3≡3 | empty (3 inert) |
| 5 | 5≡5≡−3 | empty (5 inert) |
| 6 | 2·3 | empty (3 inert) |
| 10 | 2·5 | empty (5 inert) |
| 11 | 11≡3 | empty (11 inert) |

Non-empty k values (up to 14): 1, 2, 4, 7, 8, 9, 14, ...

## The Stratification Picture

The I=k shells stratify the Ford circle packing by angular distance from the √2 cusp. From cert [292]: |s−1/3| = k/(3G̃). So:

- Shell k=1 (Pell): closest to spread-1/3 direction, Ford circles tangent
- Shell k=2: twice the spread-deviation, Farey gap=2
- Shell k=7: seven times the gap

The inter-direction spread = k²/(G̃G̃') — **k² times larger** than the I=1 value at comparable scale. Deeper shells sit more "inward" from the √2 geodesic.

## Sign Alternation

Within each shell, the signed value b²−2e² alternates sign along the QA chain:

```
k=1: −1, +1, −1, +1, ...   (Pell solutions alternating above/below √2)
k=2: +2, −2, +2, −2, ...
k=7: +7, −7, +7, −7, ...
```

This means consecutive elements in S_k lie on alternating sides of the √2 direction in the Stern-Brocot tree.

## Checks

| ID | Description |
|---|---|
| SHELL_I | I(b,e) = \|b²−2e²\| = shell_k |
| SHELL_PRES | I(b+2e, b+e) = shell_k (QA successor stays in shell) |
| SIGN_FLIP | (b+2e)²−2(b+e)² = −(b²−2e²) |
| FAREY_K | \|b(b+e)−(b+2e)e\| = shell_k |
| SPREAD_K | Wildberger spread between directions = k²/(G̃·G̃') |
| SPREAD_DEV_K | \|s−1/3\| = k/(3G̃) |

**Fixtures**: 4 PASS + 2 FAIL
**Self-test**: shells k=1,2,7 chains (8 steps each); SHELL_UNIQUE for k=2..19; empty shells k=3,5,6,10,11 for b,e≤100; sign alternation for k=2

## Primary Sources

- Hardy, G. H., & Wright, E. M. (2008). *An Introduction to the Theory of Numbers* (6th ed.). Oxford University Press. ISBN 978-0-19-921986-5. Ch. XIII: Pell equation, norm form b²−2e², representability in ℤ[√2].
- Wildberger, N. J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8. Koenig I=|C−F|, BEDA tuples.

## Mechanism Chain

- [289] QA Koenig Pell Ford Circle — k=1 is the I=1 shell (tangent chain to √2)
- [292] QA Koenig Spread Optimality — SPREAD_DEV_K = k/(3G̃) corollary
- [141] QA Pell Norm — I = −Pell norm; same invariant
