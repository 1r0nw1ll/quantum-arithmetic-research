# [295] QA Pell Sturmian Bridge Cert

**Family ID**: 295
**Slug**: `qa_pell_sturmian_bridge_cert_v1`
**Status**: Active
**Registered**: 2026-06-01

## What This Cert Corrects

Cert [294] established that the Pell Stern-Brocot path towards √2 converges to the **periodic** word (RLLR)∞. I initially claimed this was "Sturmian." It is not. The correct picture:

| Object | Encodes | CF | Word type | Density |
|---|---|---|---|---|
| SB path (cert [294]) | √2 = [1;2̄] | period 2, value > 1 | **(RLLR)∞ — PERIODIC** | R:½ L:½ |
| Sturmian s_α (this cert) | √2−1 = [0;2̄] | period 2, value < 1 | **aperiodic, p(n)=n+1** | 1: α≈0.414 |

The Pell **e-values** bridge both objects.

## The Sturmian Word

The characteristic Sturmian word of slope α = √2−1 is:

```
s(n) = ⌊(n+1)α⌋ − ⌊nα⌋  ∈ {0, 1}
```

First 30 terms (as R/L with 0→R, 1→L): `RRLRLRRLRLRRLRLRLRRLRLRRLRLRLR`

Properties:
- **Aperiodic**: no period T ≤ 50 exists (verified up to 100 terms)
- **Complexity**: p(n) = n+1 for all n (the Sturmian condition — exactly one new subword of each new length)
- **Gap sizes**: consecutive 1-positions are separated by 2 or 3 only (since 1/α = √2+1 ≈ 2.414)
- **No 'LL'**: maximal run of L is 1 (run of R is at most 2)

## The Pell Bridge

The Pell e-values {1, 2, 5, 12, 29, 70, 169, 408, ...} are **identical** to the denominators of the CF convergents to α = √2−1 = [0;2,2,2,...]:

| n | convergent p_n/q_n | q_n | Pell e_n |
|---|---|---|---|
| 1 | 1/2 | 2 | 2 ✓ |
| 2 | 2/5 | 5 | 5 ✓ |
| 3 | 5/12 | 12 | 12 ✓ |
| 4 | 12/29 | 29 | 29 ✓ |
| 5 | 29/70 | 70 | 70 ✓ |

**Why**: Both sequences satisfy the same recurrence with the same initial conditions:

```
q_{n+1} = 2q_n + q_{n-1},   q_0=1, q_1=2
e_{n+1} = 2e_n + e_{n-1},   e_0=1, e_1=2
```

This shared recurrence arises because √2=[1;2̄] and √2−1=[0;2̄] have the same **period-2 CF structure** — they differ only by the shift z → z−1 (Möbius transformation).

## Why (RLLR)∞ Is NOT Sturmian

A Sturmian word requires p(n) = n+1 for ALL n. The word (RLLR)∞ has:

| n | p(n) for (RLLR)∞ | p(n) for s_α | Sturmian? |
|---|---|---|---|
| 1 | 2 | 2 | — |
| 2 | **4** | 3 | (RLLR)∞ fails at n=2 |
| 3 | 4 | 4 | — |
| 4 | 4 | 5 | — |

The word (RLLR)∞ has p(2)=4 because all four 2-grams {RL,LL,LR,RR} appear. A Sturmian word must have p(2)=3 — exactly one 2-gram missing. So (RLLR)∞ **cannot** be Sturmian.

## What Is Known vs Novel

**Known** (classical): Sturmian words (Morse-Hedlund 1938; Lothaire 2002); CF convergents for quadratic irrationals; the recurrence q_{n+1}=2q_n+q_{n-1} for [0;2̄].

**Novel** (QA framing): The explicit identification of Pell e-values as the **bridge** between the two encodings — periodic SB word for √2 (cert [294]) and aperiodic Sturmian word for √2−1. The original claim that (RLLR)∞ "is" Sturmian was wrong; the correct statement is that both objects arise from the same arithmetic but encode different aspects of it.

## Checks

| ID | Description |
|---|---|
| COMPLEXITY_OK | p(n) = n+1 for s_α (Sturmian condition) |
| RECURRENCE_OK | Pell e-values satisfy q_{n+1}=2q_n+q_{n-1} |
| INIT_OK | Initial conditions q_0=1, q_1=2 match |
| CF_EQUAL_OK | Pell e-values = CF denominators for α=[0;2̄] |
| GAP_SUBSET_OK | Gap sizes in s_α ⊆ {2,3} |
| HAS_ONES | s_α contains 1s (non-trivial) |
| APERIODIC_OK | No period ≤ 50 in 100-term window |

**Fixtures**: 4 PASS + 2 FAIL
**Self-test**: complexity n+1 for n=1..11; aperiodic; gap sizes {2,3}; Pell e recurrence; (RLLR) periodicity and p(2)=4; density check

## Primary Sources

- Lothaire, M. (2002). *Algebraic Combinatorics on Words*. Cambridge University Press. ISBN 978-0-521-81220-7. Ch. 2: Sturmian words, complexity p(n)=n+1, characteristic word, CF connection.
- Hardy, G. H., & Wright, E. M. (2008). *An Introduction to the Theory of Numbers* (6th ed.). Oxford University Press. ISBN 978-0-19-921986-5. Ch. X: Continued fractions, convergents, best approximations.

## Mechanism Chain

- [294] QA SL(2,Z) Spine — (RLLR)∞ is the periodic SB encoding of √2; this cert shows it is NOT Sturmian
- [289] QA Koenig Pell Ford Circle — Pell e-values as denominators
