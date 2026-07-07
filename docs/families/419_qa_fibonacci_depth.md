<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Lucas (1878) American Journal of Mathematics 1(2), Wall (1960) doi:10.2307/2309169, Lehmer (1930) doi:10.2307/1968235 -->
# [419] QA Fibonacci Depth Decomposition

**Cert family**: `qa_fibonacci_depth_cert_v1`
**Claim**: The Fibonacci depth invariant δ(p) = F_{α(p)}/p mod p — the normalized first-zero
depth of the QA T-orbit mod p² — decomposes according to the parity of α(p) via classical
Fibonacci identities, placing all Fibonacci primes and the non-Fibonacci class {41,193,1621,...}
into a unified framework.

## The decomposition

**Even α(p)**:
```
delta(p) = F_{alpha/2} * (L_{alpha/2} / p)  mod p
```
This follows from the Lucas doubling identity F_{2n} = F_n × L_n (Lucas 1878), with the key
corollary: **p always divides L_{alpha/2}** (proved algebraically: p|F_{alpha}, p∤F_{alpha/2},
so F_{alpha} = F_{alpha/2}×L_{alpha/2} forces p|L_{alpha/2}).

**Odd α(p)**:
```
delta(p) = (F_k^2 + F_{k-1}^2) / p  mod p,   k = (alpha+1)/2
```
This follows from F_{2k−1} = F_k² + F_{k−1}² (Lucas 1878), with the observation that for odd
α both components F_k and F_{k-1} are non-zero mod p (since k, k-1 < α = α(p)).

## The Fibonacci prime theorem

For every Fibonacci prime p = F_m (rank α(p) = m by minimality of F_m):
```
F_{alpha(p)} = F_m = p   (exact equality, not just mod p^2)
```
Therefore F_{alpha}/p = 1 exactly, so **δ(p) = 1 for all Fibonacci primes**.

Fibonacci primes up to 2000: {2, 3, 5, 13, 89, 233, 1597} = {F₃, F₄, F₅, F₇, F₁₁, F₁₃, F₁₇}.

## Census of δ(p) = 1 primes up to 2000

```
delta=1 primes: {2, 3, 5, 13, 41, 89, 193, 233, 1597, 1621}
  Fibonacci primes: {2, 3, 5, 13, 89, 233, 1597}          (7 primes)
  Non-Fibonacci:    {41, 193, 1621}                        (3 primes)
```

Every non-Fibonacci δ=1 prime satisfies the **maximum rank condition**:
```
2 * alpha(p) = p - (5/p)
```

| p    | class | α(p) | 2α | p−(5/p) | δ(p) |
|------|-------|------|-----|---------|------|
| 41   | split | 20   | 40  | 40      | 1    |
| 193  | inert | 97   | 194 | 194     | 1    |
| 1621 | split | 810  | 1620| 1620    | 1    |

The maximum rank condition means φ/φ̄ (the ratio of the two Fibonacci roots in 𝔽_p or 𝔽_{p²})
has order exactly (p−(5/p))/2 — a "half-generator" in the relevant unit group.

## Explicit decomposition examples

**p=3** (Fibonacci, even α=4):
F₂=1, L₂=3. L₂/p=1. δ = 1×1 = 1 mod 3. ✓ (F₄=3=p exactly)

**p=41** (non-Fibonacci, even α=20):
F₁₀=55≡14 mod 41, L₁₀=123=3×41. c=3.
δ = 14×3 = 42 ≡ **1** mod 41. ✓

**p=11** (split, even α=10):
F₅=5, L₅=11=p. c=1.
δ = 5×1 = **5** mod 11. (Not δ=1; L_{α/2}=p but F_{α/2}≠1 mod p)

**p=193** (non-Fibonacci, odd α=97, k=49):
F₄₉²+F₄₈² = F₉₇ ≡ 193 mod 193². δ = **1** mod 193. ✓

**p=37** (inert, odd α=19, k=10):
F₁₀²+F₉² = F₁₉. F₁₉/37 ≡ **2** mod 37. (Maximum rank but δ≠1)

## Checks

- **C1**: Even-α decomposition for 62/62 primes in [2,500] — **PASS**
- **C2**: Odd-α decomposition for 33/33 primes in [2,500] — **PASS**
- **C3**: δ=1 for all 7 Fibonacci primes up to 2000 — **PASS**
- **C4**: Full δ census up to 2000: exactly {2,3,5,13,41,89,193,233,1597,1621} — **PASS**

## Open question

The deeper algebraic characterization of non-Fibonacci δ=1 primes beyond the maximum-rank
condition remains open. Why does 2α = p−(5/p) yield δ=1 for p=41 and p=1621 but not for
p=29, p=101, p=181 (all split with the same maximum-rank structure)?

## Chain

- **Inherits from**: cert [417] (δ(p) defined as F_{α}/p mod p, Wall-Sun-Sun context),
  cert [418] (α(p) parity via Cassini Gate)
- **Lucas doubling identity** F_{2n}=F_n×L_n (Lucas 1878) is the engine of C1
- **Fibonacci odd identity** F_{2k−1}=F_k²+F_{k−1}² (Lucas 1878) is the engine of C2
- **Open**: characterize non-Fibonacci δ=1 primes beyond maximum-rank condition

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-derived the full δ(p)=1
census up to 2000 in a fresh script (own rank-of-apparition and direct
Fibonacci/Lucas implementations) — reproduced {2,3,5,13,41,89,193,233,
1597,1621} exactly, and confirmed the maximum-rank condition
2α(p)=p−(5/p) for the 3 non-Fibonacci members {41,193,1621}. Genuine
falsifiable number theory, no fixture-trusting gap.
