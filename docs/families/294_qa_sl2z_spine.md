# [294] QA SL(2,Z) Spine Cert

**Family ID**: 294
**Slug**: `qa_sl2z_spine_cert_v1`
**Status**: Active
**Registered**: 2026-06-01

## Claim (narrow, falsifiable)

The two QA increment moves **L** and **R** are the generators of SL(2,Z). Every primitive QA state is a unique word in {L,R}*. The QA Fibonacci matrix M has M²=L·R ∈ SL(2,Z). This is the group-theoretic backbone of the Ford circle arc [289–293].

## The Generators

```
L = [[1,0],[1,1]]:  (b,e) → (b, b+e)     det(L) = 1  ∈ SL(2,Z)
R = [[1,1],[0,1]]:  (b,e) → (b+e, e)     det(R) = 1  ∈ SL(2,Z)
M = [[0,1],[1,1]]:  (b,e) → (e, b+e)     det(M) = −1  ∈ GL(2,Z) \ SL(2,Z)

M² = [[1,1],[1,2]] = L·R                  ∈ SL(2,Z)
```

L and R are the two elementary moves of the **subtractive Euclidean algorithm** — and simultaneously the two generators of SL(2,Z). The Fibonacci/QA T-step M is a single-step "shortcut" that crosses the det=1 boundary; two M-steps give L·R, landing back in SL(2,Z).

## The Stern-Brocot Theorem

For every (b,e) with b,e ≥ 1 and gcd(b,e)=1, there is a **unique word** W ∈ {L,R}* such that W·(1,1) = (b,e). W is computed by:

```python
while (b,e) != (1,1):
    if b > e: prepend 'R'; b -= e
    else:     prepend 'L'; e -= b
```

`len(W)` equals the number of subtractive Euclidean steps.

## Pell Chain Words

The Pell chain from cert [289] has words of length 2n with a striking period-4 structure:

| n | (b,e) | Word | Pattern |
|---|---|---|---|
| 0 | (1,1) | `` | root |
| 1 | (3,2) | `LR` | — |
| 2 | (7,5) | `RLLR` | `RL` + `LR` |
| 3 | (17,12) | `LRRLLR` | `LR` + `RLLR` |
| 4 | (41,29) | `RLLRRLLR` | `RL` + `LRRLLR` |
| 5 | (99,70) | `LRRLLRRLLR` | `LR` + `RLLRRLLR` |

**Recurrence**: w_{n+1} = (`LR` if n odd else `RL`) + w_n

**Descending Euclidean steps** follow the period-4 pattern `(R,L,L,R)^∞` — the Pell chain walks the Stern-Brocot tree along the √2 geodesic in a perfect (RLLR) rhythm.

## Why This Is the Backbone

The entire Ford circle arc rests on SL(2,Z) acting on (ℤ/mℤ)²:

| Cert | SL(2,Z) role |
|---|---|
| [289] Koenig Pell → Ford tangency | M-steps on the √2 geodesic; M²=L·R |
| [290] Classical subfamily cusps | Three geodesics from (1,1): L-chain, R-chain, M-chain |
| [291] Fibonacci matrix order 24 | M in GL(2,ℤ/9ℤ) has order 24 = π(9) |
| [292] Spread-1/3 optimality | I = det distance from √2 direction |
| [293] Shell structure | Every shell S_k preserved by L and R |
| [294] THIS CERT | L, R generate SL(2,Z); M²=L·R; Stern-Brocot theorem |

The orbit periods 24/8/1 (cert [291]) arise because SL(2,ℤ/9ℤ) has a specific orbit structure on (ℤ/9ℤ)² — not magic, but group theory on the free product ⟨L,R⟩.

## Checks

| ID | Description |
|---|---|
| DET_LR | det(L)=det(R)=1; det(M)=−1; M²=L·R |
| STERN_PRIM | gcd(b,e)=1 (SB representation requires primitive pair) |
| WORD_APPLY | Applying declared word W to (1,1) gives (b,e) |
| WORD_UNIQ | Euclidean algorithm on (b,e) produces the same word |
| EUCLID_LEN | len(W) = number of Euclidean subtraction steps |

**Fixtures**: 5 PASS + 2 FAIL
**Self-test**: Pell words [0..7]; period-4 descending pattern; round-trip for all gcd=1 pairs in [1,19]²; M²=L·R algebraic check; fail-case detection

## Primary Sources

- Hardy, G. H., & Wright, E. M. (2008). *An Introduction to the Theory of Numbers* (6th ed.). Oxford University Press. ISBN 978-0-19-921986-5. Ch. X: Continued fractions, SL(2,Z), Euclidean algorithm.
- Stern, M. A. (1858). Ueber eine zahlentheoretische Funktion. *Journal für die reine und angewandte Mathematik*, 55, 193–220.
- Brocot, A. (1861). Calcul des rouages par approximation. *Revue Chronométrique*, 3, 186–194.

## Mechanism Chain

- [289] QA Koenig Pell Ford Circle — M²=L·R generates the Pell chain
- [291] QA Fibonacci Matrix Orbit Periods — M in GL(2,ℤ/9ℤ) has order π(9)
- [293] QA Koenig Shell Structure — shells preserved by L,R words
