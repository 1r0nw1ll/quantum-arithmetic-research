<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.2307/2309169, Sun & Sun (1992) doi:10.4064/aa-60-4-371-388, Lengyel (1995) -->
# [429] QA Fibonacci Wall-Sun-Sun Lifting to p²

**Cert family**: `qa_fibonacci_wss_lifting_cert_v1`
**Claim**: For all odd primes p ≥ 5 in the tested range, F_{α(p)} is divisible
by p but not p² (non-Wall-Sun-Sun criterion). Consequently the p-adic valuation
lifts cleanly: α(p²) = p·α(p) and T(p²) = p·T(p). No Wall-Sun-Sun prime exists
in [5, 500]; the Pisano tower scales exactly with p.

## Background: Wall-Sun-Sun Primes

From cert [428] C3: F_{p−(5/p)} ≡ 0 (mod p) — the Wall zero. The finer question
is whether p² also divides it. A **Wall-Sun-Sun prime** (WSS prime) is a prime p
where p² ∣ F_{α(p)}, i.e., v_p(F_{α(p)}) ≥ 2.

Wall (1960) first raised this question. Sun & Sun (1992) showed that the existence
of a WSS prime would imply the first case of Fermat's Last Theorem for exponent p
(before Wiles' proof made this moot), and conjectured no WSS prime exists.
**No WSS prime is known as of 2024.** Computational searches have confirmed none
exists below p > 9.7 × 10¹⁴.

## C1: Non-Wall-Sun-Sun Criterion

**Claim**: v_p(F_{α(p)}) = 1 for all primes p in [5, 500].

This means F_{α(p)} ≡ 0 (mod p) (from [428]) but F_{α(p)} ≢ 0 (mod p²).

Verified: 93/93 primes. No WSS candidate found.

## C2: LTE Lifting Identity

**Claim**: F_{p·α(p)} ≡ 0 (mod p²).

**Proof** (Lifting-the-Exponent for Lucas sequences, Lengyel 1995):
For odd prime p and p ∣ F_α:
```
v_p(F_{k·α(p)}) = v_p(F_{α(p)}) + v_p(k)   for all k ≥ 1.
```
Setting k = p: v_p(F_{p·α}) = 1 + 1 = 2. So p² ∣ F_{p·α(p)}.

This is the key lifting step: multiplying the zero index by p raises the p-adic
valuation by exactly 1.

Verified: 93/93 primes.

## C3: Alpha Lifting

**Claim**: α(p²) = p·α(p) for all primes p in [5, 500].

**Proof**:
1. **Range** (Wall 1960): α(p) ∣ α(p²) and α(p²) ∣ p·α(p). So α(p²)/α(p) ∈ {1, 2, ..., p}.
2. **p is prime**: The divisors of p·α(p) that are multiples of α(p) are α(p) and p·α(p). So α(p²)/α(p) ∈ {1, p}.
3. **C1 excludes 1**: α(p²) = α(p) would require F_{α(p)} ≡ 0 (mod p²) — a WSS prime. C1 says no.
4. **C2 confirms p**: p·α(p) is a zero of F_k mod p² (from C2), so α(p²) ∣ p·α(p) and α(p²) ≥ p·α(p)/... wait — α(p²) | p·α(p) and α(p²)/α(p) = p. **QED**.

Verified directly: F_{α(p)} ≢ 0 (mod p²) AND F_{p·α(p)} ≡ 0 (mod p²) for all 93 primes.

## C4: Pisano Period Lifting

**Claim**: T(p²) = p·T(p) for all primes p in [5, 300].

**Sub-checks**:
- **(a)** (F_{p·T(p)}, F_{p·T(p)+1}) ≡ (0,1) (mod p²): p·T(p) is a period of F mod p².
  → T(p²) ∣ p·T(p).
- **(b)** (F_{T(p)}, F_{T(p)+1}) ≢ (0,1) (mod p²): T(p) is NOT a period of F mod p².
  → T(p²) > T(p).

Since T(p) ∣ T(p²) (reduction) and T(p²)/T(p) ∈ {1, p} (Wall structure theorem for
prime squares), (b) rules out ratio 1, giving T(p²) = p·T(p).

**Analytic proof of (a)**: T(p²) = α(p²)·ord(ε(p²)) = p·α(p)·ord(ε(p)) = p·T(p),
using C3 and the Hensel preservation ord(ε(p²)) = ord(ε(p)) (since ε(p) ∈ {±1,
primitive 4th root} all lift to the same order in (ℤ/p²ℤ)×).

Verified: 60/60 primes in [5, 300].

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 | v_p(F_{α(p)}) = 1; 93 primes in [5, 500] | **PASS** |
| C2 | F_{p·α(p)} ≡ 0 (mod p²); 93 primes | **PASS** |
| C3 | α(p²) = p·α(p); 93 primes | **PASS** |
| C4 | T(p²) = p·T(p); 60 primes in [5, 300] | **PASS** |

## p-Adic Tower

The cert establishes the base case of the full p-adic tower (Wall 1960):

```
alpha(p^n) = p^{n-1} * alpha(p)   for all n >= 1, non-WSS prime p.
T(p^n)     = p^{n-1} * T(p)       for all n >= 1, non-WSS prime p.
```

[429] certifies n=2. The general n follows by induction (LTE: each multiplication by p
raises the valuation by 1, so the zero index multiplies by p at each level).

## Theorem NT Factorisation

```
QA layer (pure integer):
  fib_fast(n, p*p) — fast doubling mod p^2; all intermediate values bounded by p^4
  rank_of_apparition(p) — linear walk mod p
  pisano_period(p) — linear walk mod p
  _fib_pair(n, p*p) — pair (F_n, F_{n+1}) mod p^2

Observer layer: none (no chi-squared, no floats)
```

No float arithmetic anywhere — all modular computations stay integer even mod p².

## Langlands Ladder

| Cert | Rung |
|------|------|
| [416] | α(p) exists for all p ≥ 5 |
| [428] | α(p) ∣ p−(5/p); F_p=(5/p); L_p=1 |
| **[429]** | **α(p²)=p·α(p); T(p²)=p·T(p); no WSS prime ≤ 500** |

The next rung ([430]?) would be the connection between Wall-Sun-Sun primes and
Wieferich primes: if p is WSS, then 2^{p-1} ≡ 1 (mod p²) (Wieferich criterion) under
certain conditions — making WSS primes doubly exotic.
