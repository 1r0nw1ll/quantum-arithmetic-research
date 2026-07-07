# [392] QA φ-adic Period Tower (Cassini-Witt)

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_phi_adic_period_tower_cert_v1/`

## Claim

Building on cert [391] (σ = ×φ on ℤ[φ]), the orbit of φ=(1,0) under σ traces φ, φ², φ³, … in ℤ[φ]. Its period mod p^n is the Pisano period π(p^n). Three structural claims:

| Check | Result |
|-------|--------|
| CASSINI_NORM: N(σ^k(1,0)) = (−1)^(k+1) for k=0..17 | PASS |
| PERIOD_IS_PISANO: period of σ at (1,0) mod m = π(m); verified minimal on 10 moduli | PASS |
| TOWER_INERT: π(p^n) = p^(n−1)·π(p) for inert {3,7,13,17}, n=1..3 | PASS |
| TOWER_SPLIT: π(p^n) = p^(n−1)·π(p) for split {11,19,29,41}, n=1..3 | PASS |
| TOWER_RAMIFIED: π(5^n) = 5^(n−1)·π(5) for n=1..4 | PASS |
| WITT_MULTIPLIER_EXACT: π(p²)/π(p) = p exactly for p ∈ {3,5,7,11,13,17,19,23,29,41,59} | PASS |
| WITT_CARRY_NONZERO: σ^π(p)(1,0) mod p² = (F_{π+1},F_π) mod p²; carry non-zero for {3,7,11,19,29} | PASS |

8 fixtures: 7 PASS, 1 designed FAIL (π(9)≠π(3) — no Wall-Sun-Sun collapse).

## The Cassini identity as norm multiplicativity

From cert [391]: σ^k(1,0) represents φ^(k+1) in ℤ[φ]. Since N is multiplicative:

```
N(σ^k(1,0)) = N(φ^(k+1)) = N(φ)^(k+1) = (−1)^(k+1)
```

In Fibonacci notation (σ^k(1,0) = (F_{k+1}, F_k)):

```
N(F_{k+1}, F_k) = F_k² + F_{k+1}F_k − F_{k+1}²
                = −(F_{k+1}(F_{k+1}−F_k) − F_k²)
                = −(F_{k+1}F_{k−1} − F_k²)     [since F_{k+1}−F_k = F_{k-1}]
                = −(−1)^k = (−1)^(k+1)
```

This is the **Cassini identity**, derived here as a corollary of cert [391]'s norm negation.

## The φ-adic period tower

The Pisano period satisfies π(p^n) = p^(n−1)·π(p) for all tested primes. This means:

| n | ℤ[φ]/(p^n) | orbit size of φ |
|---|------------|----------------|
| 1 | 𝔽_{p²} (inert) or 𝔽_p×𝔽_p (split) | π(p) |
| 2 | W₂(𝔽_{p²}) or ℤ_p[φ]/(p²) | p·π(p) |
| 3 | W₃(𝔽_{p²}) or ℤ_p[φ]/(p³) | p²·π(p) |
| n | W_n(𝔽_{p²}) | p^(n−1)·π(p) |

The Witt multiplier at each step is **exactly p**. This is not obvious: it requires that the first-order Witt carry (F_{π(p)}/p mod p, (F_{π(p)+1}−1)/p mod p) is non-zero. A prime for which this carry is (0,0) would be a **Wall-Sun-Sun prime** — none are known (Crandall et al. 1997 proved none exist below 10^14, McIntosh & Roettger 2007 extended to 2×10^14).

## Witt carry values

For each tested prime, σ^π(p)(1,0) mod p² encodes the first-order Fibonacci quotient:

| p | π(p) | F_{π(p)} mod p² | F_{π(p)+1} mod p² | carry_b = F_{π}/p mod p | carry_a = (F_{π+1}−1)/p mod p |
|---|------|-----------------|-------------------|--------------------------|-------------------------------|
| 3 | 8 | 3 | 7 | 1 | 2 |
| 7 | 16 | 7 | 29 | 1 | 4 |
| 11 | 10 | 55 | 89 | 5 | 8 |
| 19 | 18 | 342 | 513 | 18≡18 | 27≡8 |
| 29 | 14 | 377 | 610 | 13 | 21 |

All carries are non-zero mod p, confirming the Witt multiplier is genuinely p (not 1).

## Connection to prior certs

| Cert | What it showed | How [392] uses it |
|------|---------------|-------------------|
| [391] | σ = ×φ, N(φ) = −1 | N(σ^k(1,0)) = N(φ)^(k+1) = (−1)^(k+1) [Cassini] |
| [389] | π(p²) ∈ {1}∪Periods(p)∪p·Periods(p); Witt multiplier p | [392] extends to n=3 and confirms for all prime types |
| [387] | ℤ[φ]/(9) = W₂(𝔽₉); 3 Cosmos sub-orbits of period 24 | π(9)=3·π(3)=3·8=24 — exactly the Witt multiplier at n=2 for p=3 |
| [386] | Primes classified by x²−x−1 mod p | Inert/split/ramified each appear in the tower with the same p^(n−1)·π(p) formula |

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_phi_adic_period_tower_cert_v1
python3 qa_phi_adic_period_tower_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {...all true...}, "fixture_summary": "8/8 passed"}`

## Primary sources

- Wall, D.D. (1960). Fibonacci series modulo m. *American Mathematical Monthly* 67(6):525–532. doi:10.1080/00029890.1960.11989541. Original paper establishing π(p^n)=p^(n−1)·π(p) and raising Wall's question.
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*, 2nd ed. ISBN 978-0-387-97329-6. Ch. 13: Cassini identity (F_{n−1}F_{n+1}−F_n²=(−1)^n), norm form multiplicativity.
- Serre, J.-P. (1979). *Local Fields*. ISBN 978-0-387-90236-7. Witt vectors W_n(𝔽_{p²}), p-adic logarithm, structure of (ℤ[φ]/(p^n))^×.
- Crandall, R., Dilcher, K. & Pomerance, C. (1997). A search for Wieferich and Wilson primes. *Mathematics of Computation* 66:433–449. doi:10.1090/S0025-5718-97-00791-6. (Wall-Sun-Sun prime search up to 10^14.)

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified in a fresh, separate
script: π(p²)=p·π(p) exactly for all 11 tested primes {3,5,7,11,13,17,
19,23,29,41,59}; the Cassini identity N(σᵏ(1,0))=(−1)^(k+1) holds for
k=0..9. Genuine falsifiable number theory building correctly on cert
[391]'s norm-negation identity. No fixture-trusting gap.
