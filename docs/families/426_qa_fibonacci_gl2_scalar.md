<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Lucas (1878) doi:10.2307/2369308, Wall (1960) doi:10.2307/2309169, Lagrange (1771) -->
# [426] QA Fibonacci GL₂ Scalar Identity & Pisano Period

**Cert family**: `qa_fibonacci_gl2_scalar_cert_v1`
**Claim**: The Fibonacci recurrence matrix M = [[1,1],[1,0]] ∈ GL₂(ℤ) becomes a scalar
multiple of the identity at step α(p): M^{α(p)} ≡ ε(p)·I₂ (mod p). The Pisano period
is T(p) = α(p) · ord_{𝔽_p×}(ε(p)).

## GL₂ Structure of the Fibonacci Recurrence

The Fibonacci sequence satisfies the matrix recurrence:

```
[F_{n+1}]   [1 1]^n   [F_1]         n
[F_n    ] = [1 0]   · [F_0]  =  M  · [1]
                                       [0]
```

where M = [[1,1],[1,0]] ∈ GL₂(ℤ), det(M) = −1, tr(M) = 1.

The n-th power:
```
M^n = [[F_{n+1}, F_n ], [F_n, F_{n-1}]]
```

**GL₂ invariants** (C1, C2):
- tr(M^n) = F_{n+1} + F_{n−1} = L_n  (Lucas numbers)
- det(M^n) = (−1)^n                   (Cassini identity)

## Scalar Collapse at the Rank of Apparition

For any prime p ≠ 2, 5, at step n = α(p):

```
F_{α(p)} ≡ 0    (mod p)               [definition of α(p)]
F_{α(p)+1} = F_{α(p)} + F_{α(p)−1}
            ≡ 0 + F_{α(p)−1}
            = F_{α(p)−1}   (mod p)
```

So:
```
M^{α(p)} ≡ [[F_{α−1}, 0  ],  =  ε(p) · I₂   (mod p)
             [0,     F_{α−1}]]
```

where **ε(p) = F_{α(p)−1} mod p**.

This is the GL₂ upgrade of cert [423]:
- **[423]** (GL₁): (φ̃/ψ̃)^{α(p)} = 1 — the eigenvalue *ratio* is 1
- **[426]** (GL₂): M^{α(p)} = ε·I₂ — the *matrix* is scalar (both eigenvalues equal ε)

These are logically equivalent but [426] reveals the 2×2 matrix structure directly.

## Epsilon Type: 4th Root of Unity

From the Cassini identity with F_{α}=0:
```
F_{α−1} · F_{α+1} − F_{α}² = (−1)^α
→  ε(p)² ≡ (−1)^{α(p)}   (mod p)
→  ε(p)⁴ ≡ 1             (mod p)
```

So ε(p) is always a **4th root of unity** in 𝔽_p×, and ord(ε(p)) ∈ {1, 2, 4}:

| ε(p) | ord(ε) | Geometric meaning |
|------|--------|-------------------|
| +1   | 1      | M^α = I (period closes immediately) |
| −1   | 2      | M^α = −I (period closes after doubling) |
| ±√−1 | 4     | M^α = ±iI (period closes after quadrupling; only possible when p ≡ 1 mod 4) |

Empirical distribution over 92 primes ≤ 500: {ord=1: 32, ord=2: 29, ord=4: 31} — roughly equal thirds.

## Pisano Period Formula

```
T(p) = α(p) · ord_{𝔽_p×}(ε(p))
```

**Proof**: M^{αk} = (M^α)^k = (ε·I)^k = ε^k·I. The smallest k with ε^k = 1 is k = ord(ε).
So M^T = I iff ord(ε) | T/α, i.e., minimum T = α · ord(ε).

This decomposes the Pisano period into two pure-integer factors:
- α(p) — the GL₁ rank of apparition (cert [416])
- ord(ε(p)) — the GL₂ scalar order ∈ {1, 2, 4}

**Verified** for 59/59 primes ≤ 300 by independent direct walk.

## Checks

| Check | Content | Status |
|-------|---------|--------|
| C1 | tr(M^n) = L_n for n=1..50 (exact integer arithmetic) | **PASS** (50/50) |
| C2 | det(M^n) = (−1)^n for n=1..50 (Cassini; exact integer) | **PASS** (50/50) |
| C3 | M^{α(p)} = ε(p)·I₂ mod p for all primes 7≤p≤500 | **PASS** (92/92) |
| C4 | T(p) = α(p)·ord(ε(p)) for all primes 7≤p≤300 | **PASS** (59/59) |

All checks are pure integer — no observer float layer (Theorem NT: no boundary crossing needed).

## Theorem NT Factorisation

```
QA layer (pure integer):
  fib_fast(n, m)          — F_n mod m via fast doubling
  rank_of_apparition(p)   — α(p) by integer walk
  pisano_period(p)        — T(p) by integer walk
  mult_order(eps, p)      — ord(ε) by Lagrange reduction
  lucas(n)                — L_n by recurrence

Observer layer: NONE (all 4 checks are integer equality comparisons)
```

## Langlands Ladder

| Cert | Rung |
|------|------|
| [416] | α(p) \| p−(5/p) |
| [423] | α(p) = ord_{GL₁(𝔽_p)}(φ̃/ψ̃) for split p |
| [424] | Frob_p: φ̃↔ψ̃ in 𝔽_{p²}; α(p)\|p+1 for inert p |
| [425] | density(split) = density(inert) = ½ (Chebotarev) |
| **[426]** | **M^{α(p)} = ε·I₂ in GL₂(𝔽_p); T(p) = α·ord(ε)** |

GL₁/ℚ(√5) picture [416]–[425] lifts to GL₂ at [426]: the 2-dimensional Fibonacci matrix representation becomes scalar at the rank of apparition, and the Pisano period formula T = α·ord(ε) follows.

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified the scalar
collapse Mᵅ=ε·I₂ and the Pisano formula T(p)=α(p)·ord(ε(p)) in a fresh
script (own matrix-power and rank-of-apparition implementations) —
zero failures. The ε-order distribution {1:32, 2:29, 4:31} for p≤500
reproduces exactly once the same `p>5` convention as [423]-[425] is
applied. Genuine falsifiable algebra, no fixture-trusting gap.
