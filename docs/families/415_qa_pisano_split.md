<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.2307/2309169, Ribenboim (1996) ISBN 978-0-387-94457-9 -->
# [415] QA Pisano Period = Prime Splitting Criterion

**Cert family**: `qa_pisano_split_cert_v1`
**Claim**: The Pisano period π(p) — the period of the Fibonacci sequence mod p, which is exactly
the QA T-orbit period mod p — encodes prime splitting in ℚ(√5)/ℚ.

## The Three-Way Criterion (C1–C3)

| Class | p mod 5 | π(p) divides p−1? | π(p) divides 2(p+1)? | Result |
|---|---|---|---|---|
| Split | 1 or 4 | **YES** | (yes, trivially) | identified |
| Inert | 2 or 3 | **NO** | **YES** | identified |
| Ramified | 0 (p=5) | NO | NO | **π(5)=20=(p−1)·p** |

The two divisibility conditions together distinguish all three classes by integer arithmetic alone.

## Why the periods come out this way (algebraic reason)

**Split p** (p≡1,4 mod 5): the ring ℤ[φ]/p ≅ 𝔽_p × 𝔽_p (two independent copies of 𝔽_p).
The Fibonacci recurrence mod p reduces to two separate recurrences over 𝔽_p, each with period
dividing |𝔽_p*| = p−1 by Fermat's little theorem.

**Inert p** (p≡2,3 mod 5): ℤ[φ]/p ≅ 𝔽_{p²} (one copy of the degree-2 extension).
The Frobenius automorphism (φ ↦ φ^p) has order 2 in Gal(𝔽_{p²}/𝔽_p).
Combined period divides 2·|𝔽_p*| = 2(p+1). Non-divisibility of p−1 follows because
the Frobenius is non-trivial — it can't collapse to a 𝔽_p recurrence.

**Ramified p=5**: ℤ[φ]/5 ≅ 𝔽_5[x]/(x−3)² (repeated root at φ≡3 mod 5, since 3²=9≡4=3+1 mod 5).
Nilpotent structure gives period 20 = (p−1)·p = 4·5, matching the known formula for
the ramified case in quadratic fields.

## Selected witnesses

**Split primes** (π(p) | p−1):

| p | π(p) | p−1 | π(p)/(p−1) |
|---|---|---|---|
| 11 | 10 | 10 | 1 |
| 101 | 50 | 100 | 1/2 |
| 151 | 50 | 150 | 1/3 |
| 211 | 42 | 210 | 1/5 |
| 421 | 84 | 420 | 1/5 |

**Inert primes** (π(p) | 2(p+1), π(p) ∤ p−1):

| p | π(p) | 2(p+1) | p−1 |
|---|---|---|---|
| 3 | 8 | 8 | 2 |
| 7 | 16 | 16 | 6 |
| 47 | 32 | 96 | 46 |
| 127 | 256 | 256 | 126 |

**Ramified**: π(5) = 20, p−1 = 4, 2(p+1) = 12. Neither 4 nor 12 is divisible by 20.

## QA orbit connection (C4)

The QA T-step T(b,e) = (e, b+e) applied modulo p to the pair (0,1) generates the
Fibonacci sequence mod p. Its period is exactly π(p). The Cosmos orbit seed (1,1)
has the same period. Verified for all three prime classes:

| p | class | π(p) | period of (0,1) | period of (1,1) |
|---|---|---|---|---|
| 11 | split | 10 | 10 | 10 |
| 5 | ramified | 20 | 20 | 20 |
| 3 | inert | 8 | 8 | 8 |

## The chain (three levels of the same structure)

```
cert [133]: T-step sign-flip  f(e,b+e) = -f(b,e)
     |
     v  (same algebraic object)
cert [414]: f(a,b) = a²+ab-b² = N_{Q(sqrt5)/Q}(a+b*phi)
     |
     v  (same dynamics, period mod p)
cert [415]: pi(p) = T-orbit period mod p = prime splitting class
```

The QA T-step encodes prime splitting at three levels simultaneously:
1. Algebraically: the sign-flip formula equals the norm form
2. Representationally: which primes have integer solutions (split/inert boundary)
3. Dynamically: the orbit period mod p (Pisano period) distinguishes all three classes

All three levels use only integer arithmetic. All three are independently falsifiable.

## Checks

- **C1**: π(p) | p−1 for 22/22 split primes ≤ 491 — PASS
- **C2**: π(p) | 2(p+1) and π(p) ∤ p−1 for 22/22 inert primes ≤ 193 — PASS
- **C3**: π(5) = 20 = (p−1)·p; divides neither p−1=4 nor 2(p+1)=12 — PASS
- **C4**: QA T-orbit period = π(p) for 7 sample primes — PASS

## Chain

- **Inherits from**: cert [133] (QA T-step sign-flip), cert [414] (norm form bridge)
- **Bridges to**: cert [410] (Dedekind zeta factorization uses same trichotomy)
- **Parallel to**: cert [413] (BSD central value trichotomy — same three classes, different context)
