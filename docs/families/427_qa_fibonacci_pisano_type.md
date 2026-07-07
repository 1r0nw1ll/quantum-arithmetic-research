<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Lucas (1878) doi:10.2307/2369308, Wall (1960) doi:10.2307/2309169, Euler (1750), Pearson (1900) -->
# [427] QA Fibonacci Pisano Type Distribution

**Cert family**: `qa_fibonacci_pisano_type_cert_v1`
**Claim**: The three Pisano types T(p) ∈ {α(p), 2α(p), 4α(p)} (from cert [426]) are
determined by whether ε(p) ≡ +1, −1, or ±√(−1) mod p. They occur with equal density
1/3 each over all primes — a three-way equidistribution arising from a Chebotarev-type
cancellation between the Kronecker symbol (−4/p) and the parity of α(p).

## The Three Pisano Types

From cert [426]: M^{α(p)} = ε(p)·I₂ mod p, where ε is a 4th root of unity.

| Type | ε(p) | ord(ε) | Pisano period | Condition |
|------|-------|--------|---------------|-----------|
| 1 | +1 | 1 | T = α(p) | α even, F_{α−1} ≡ +1 |
| 2 | −1 | 2 | T = 2α(p) | α even, F_{α−1} ≡ −1 |
| 4 | ±√(−1) | 4 | T = 4α(p) | α odd, p ≡ 1 mod 4 |

## C1: Type Gate (Euler's Criterion)

**Theorem**: Type 4 occurs iff p ≡ 1 mod 4 AND α(p) is odd.

**Proof**: ε(p)² ≡ (−1)^{α(p)} mod p (cert [426] C4).
- If α(p) odd: ε² ≡ −1 mod p. This requires −1 to be a QR mod p.
  By Euler's criterion: (−1/p) = (−1)^{(p−1)/2} = +1 iff p ≡ 1 mod 4.
  → Type 4 requires p ≡ 1 mod 4.
- If p ≡ 3 mod 4: −1 is not a QR, so ε² = −1 has no solution → α must be even.
  → Types 1 and 2 only for p ≡ 3 mod 4.

Empirically verified: 92 primes ≤ 500. All 406 Type 4 primes in [7, 10000] satisfy p ≡ 1 mod 4.

## C2: Lucas Bridge Identity

**Theorem**: L_{α(p)} ≡ 2ε(p) (mod p).

**Proof** (pure integer):
```
F_{α(p)} ≡ 0 (mod p)                    [definition of α(p)]
F_{α+1} = F_{α} + F_{α−1} ≡ F_{α−1} = ε  [since F_α = 0]
L_{α} = F_{α+1} + F_{α−1} = ε + ε = 2ε   [Lucas number definition]
```

This connects the **GL₂ trace** (Lucas number L_{α}) to the **GL₂ scalar** (ε from cert [426]):
- Type 1: L_{α(p)} ≡ +2 mod p
- Type 2: L_{α(p)} ≡ −2 mod p
- Type 4: L_{α(p)}² ≡ −4 mod p (L_{α} is a square root of −4 in 𝔽_p)

Verified: 92/92 primes ≤ 500.

## C3+C4: Equal Thirds Equidistribution

### Data (primes 7..10000, N = 1226)

| Type | Count | Fraction | Expected (1/3) |
|------|-------|----------|----------------|
| 1 | 412 | 33.6% | 33.3% |
| 2 | 408 | 33.3% | 33.3% |
| 4 | 406 | 33.1% | 33.3% |

Chi² = **0.046** vs critical 5.991 (df=2, α=0.05) — extremely flat.

### Stratification by p mod 4

**p ≡ 3 mod 4** (618 primes): Types 1 and 2 only.
| Type | Count | Fraction |
|------|-------|----------|
| 1 | 307 | 49.7% |
| 2 | 311 | 50.3% |
Chi² = 0.026 vs 3.841 (df=1, α=0.05). **Equal halves.**

**p ≡ 1 mod 4** (608 primes): All three types.
| Type | Count | Fraction |
|------|-------|----------|
| 1 | 105 | 17.3% |
| 2 | 97 | 16.0% |
| 4 | 406 | 66.8% |
Type 4 dominates: **2/3** of primes with p ≡ 1 mod 4 have Type 4.

### The Equal-Thirds Cancellation

The overall 1/3 density for each type is an exact Chebotarev cancellation:

```
p ≡ 3 mod 4 (density 1/2): Types 1+2 only, equal halves
  → Type 1 gets 1/4, Type 2 gets 1/4, Type 4 gets 0.

p ≡ 1 mod 4 (density 1/2): Type 4 at 2/3, Types 1+2 share 1/3
  → Type 1 gets (1/2)(1/6) = 1/12
  → Type 2 gets (1/2)(1/6) = 1/12
  → Type 4 gets (1/2)(2/3) = 1/3

Combined:
  Type 1: 1/4 + 1/12 = 3/12 + 1/12 = 4/12 = 1/3 ✓
  Type 2: 1/4 + 1/12 = 1/3 ✓
  Type 4: 0   + 1/3  = 1/3 ✓
```

The fraction 2/3 of Type 4 within p ≡ 1 mod 4 is itself a Chebotarev-type statement:
among primes with p ≡ 1 mod 4, exactly 2/3 have odd α(p).

## Checks

| Check | Content | Status |
|-------|---------|--------|
| C1 | Type gate: no Type 4 for p≡3 mod 4; all Type 4 have p≡1 mod 4 AND α odd; 92 primes ≤ 500 | **PASS** |
| C2 | L_{α(p)} ≡ 2ε(p) mod p for 92/92 primes ≤ 500 | **PASS** |
| C3 | p≡3 mod 4: equal halves chi²=0.026 < 3.841; n=618 | **PASS** |
| C4 | All primes: equal thirds chi²=0.046 < 5.991; n=1226 | **PASS** |

## Theorem NT Factorisation

```
QA layer (pure integer):
  rank_of_apparition, mult_order, fib_fast — all integer
  L_{alpha} = F_{alpha+1} + F_{alpha-1} = 2*eps (computed as integer mod p)
  Type classification: ord(eps) in {1,2,4}

Observer layer (float, lawful):
  chi-squared goodness-of-fit tests (Pearson 1900)
  fractional counts (output only, not fed back)
```

## Langlands Ladder

| Cert | Rung |
|------|------|
| [426] | M^{α(p)} = ε·I₂; T(p) = α·ord(ε) |
| **[427]** | **L_{α(p)} = 2ε(p) mod p; equal-thirds for types {1,2,4}** |

The Lucas bridge C2 connects the GL₂ trace (L_n as tr(M^n), cert [426] C1) to the
GL₂ scalar (ε(p), cert [426] C3). The equal-thirds equidistribution C4 is the GL₂
density theorem: the three Pisano types are Chebotarev-equidistributed.

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently reproduced every reported count
exactly in a fresh script (own matrix-power/rank-of-apparition
implementation): overall type counts {1:412, 2:408, 4:406} (n=1226),
p≡3 mod 4 stratum {1:307, 2:311} (n=618), and p≡1 mod 4 stratum
{1:105, 2:97, 4:406} — all exact matches, including the striking 2/3
Type-4 dominance within the p≡1 mod 4 stratum. Genuine, honestly
reported empirical result, no fixture-trusting gap.
